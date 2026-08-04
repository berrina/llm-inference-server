# Mini LLM Inference Server — Batching, Continuous Batching & Live Dashboard

A from-scratch LLM inference server built to understand (and measure) how production systems like vLLM handle concurrent requests efficiently. Built on GPT-2, running entirely on CPU.

## Why this project

Naive LLM serving — one request at a time — doesn't scale. This project builds up, in stages, the techniques real inference systems use to serve many requests efficiently: request batching, KV cache reuse, and continuous batching. Every claim below is backed by a benchmark I actually ran, and it not a theoretical estimate.

## Architecture

```
[Frontend Dashboard]
        |
        v
[FastAPI Server]
    |-- /generate              (static batching)
    |-- /generate_continuous   (continuous batching)
    |-- /generate_stream       (token-by-token streaming demo)
    |-- /metrics, /status      (live stats)
        |
        v
[GPT-2, CPU inference]
```

Two different scheduling strategies were built and benchmarked head-to-head:
- **Static batching**: requests are collected for up to 50ms (or until 8 arrive), then processed together in one batch, start to finish.
- **Continuous batching**: requests are processed token-by-token; the moment one finishes, its slot immediately opens up for a new arrival — mid-stream, without waiting for the rest of the batch.

## Results

### 1. Baseline — no batching
Sequential requests, one at a time:
- **35.71 tokens/sec**, 0.56s average latency

### 2. Static batching (concurrency = 8)
- **63.05 tokens/sec** (~1.8x over baseline), 2.70s average latency

The tradeoff here is real and expected: individual requests wait slightly longer (queued alongside others), but the system processes more total work per second. This throughput/latency tradeoff is one of the central design questions in real inference serving.

### 3. KV cache reuse
Isolated comparison, 100-token generation, cache on vs. off:
- **Without cache: 7.98s → With cache: 2.00s (≈4x speedup)**

Without a cache, generating each token re-processes the entire sequence so far from scratch — cost grows roughly quadratically with sequence length. With a cache, each new token only costs a small constant amount of extra work. (Note: Hugging Face's `.generate()` uses KV caching by default — this experiment isolates and measures a mechanism that was already quietly helping every other benchmark in this project.)

### 4. Continuous batching — a tradeoff, but not a full strict win

| Concurrency | Static: throughput | Continuous: throughput | Static: p50 latency | Continuous: p50 latency |
|---|---|---|---|---|
| 1  | 9.88 tok/s   | 46.74 tok/s  | 2.02s | 0.43s |
| 8  | 206.75 tok/s | 142.11 tok/s | 0.77s | 1.12s |
| 16 | 213.08 tok/s | 150.35 tok/s | 1.50s | 2.06s |

**Continuous batching wins decisively on time-to-first-token** at low concurrency (0.43s vs. 2.02s) — it returns results as soon as they're ready rather than waiting for a whole batch. **Static batching wins on raw throughput at high concurrency.** This is an honest, explainable result: my continuous batching implementation manages padding and KV-cache splicing in pure Python at every single generation step, while static batching leans on Hugging Face's internally optimized `.generate()` loop. Production systems (vLLM, TensorRT-LLM) get continuous batching's full throughput benefit because they implement this scheduling logic in optimized CUDA kernels, not Python — this project reproduces the *mechanism* correctly (verified against reference generation, including mid-stream slot replacement), but not the kernel-level performance of a production system.

Charts: `benchmark/chart_throughput.png`, `benchmark/chart_latency.png`, `benchmark/chart_summary.png`

### 5. A model-behavior finding: GPT-2 needs a nudge to answer directly

Asked "The capital of France is" (zero-shot, greedy decoding), GPT-2 base produced vague, looping continuations ("...the capital of the French Republic...") rather than stating "Paris." Providing one worked example first (few-shot prompting: *"Q: What is the capital of Japan? A: Tokyo / Q: What is the capital of France? A:"*) was enough to reliably elicit the correct, direct answer — with no changes to the model or decoding strategy. This is baked into the frontend dashboard automatically.

## What's in this repo

- `server/` — FastAPI app, model runner, both batching schedulers, metrics
- `benchmark/` — baseline, concurrent, continuous-batching, and full load-test scripts + result charts
- `frontend/` — live dashboard: streaming generation, live queue/throughput stats, concurrent load-test trigger

## Running it

```bash
pip install -r requirements.txt
uvicorn server.main:app --reload --port 8000
```
Then open `frontend/index.html` directly in a browser.

## Known limitations / what I'd do next

- Runs on CPU only, no GPU comparison numbers (would show a much larger batching benefit)
- Continuous batching implementation is pure Python; a production version would use paged attention to avoid padding overhead entirely
Greedy decoding only, sso no sampling, no repetition penalty (GPT-2 visibly loops on some prompts as a result)

- Next goals : add request preemption/priority, multi-GPU support, and a proper KV-cache memory accounting layer
