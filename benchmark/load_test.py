import asyncio
import aiohttp
import time
import random

random.seed(42)

BASE_URL = "http://localhost:8000"
PROMPT = "The capital of France is"
CONCURRENCY_LEVELS = [1, 2, 4, 8, 16]
MAX_NEW_TOKENS = 20

async def send_request(session, url):
    start = time.time()
    async with session.post(url, json={"prompt": PROMPT, "max_new_tokens": MAX_NEW_TOKENS}) as resp:
        await resp.json()
        return time.time() - start

async def run_sweep(url, label):
    print(f"\n=== {label} ===")
    results = []
    for concurrency in CONCURRENCY_LEVELS:
        async with aiohttp.ClientSession() as session:
            start = time.time()
            tasks = [send_request(session, url) for _ in range(concurrency)]
            latencies = await asyncio.gather(*tasks)
            elapsed = time.time() - start

        total_tokens = concurrency * MAX_NEW_TOKENS
        throughput = total_tokens / elapsed
        avg_latency = sum(latencies) / len(latencies)
        sorted_lat = sorted(latencies)
        p50 = sorted_lat[len(sorted_lat) // 2]
        p99 = sorted_lat[int(len(sorted_lat) * 0.99)] if len(sorted_lat) > 1 else sorted_lat[-1]

        print(f"Concurrency {concurrency:2d}: throughput={throughput:6.2f} tok/s, "
              f"avg_latency={avg_latency:.2f}s, p50={p50:.2f}s, p99={p99:.2f}s")

        results.append({
            "concurrency": concurrency,
            "throughput": throughput,
            "avg_latency": avg_latency,
            "p50": p50,
            "p99": p99,
        })

        await asyncio.sleep(0.5)  # brief pause between sweeps

    return results

async def main():
    static_results = await run_sweep(f"{BASE_URL}/generate", "STATIC BATCHING")
    continuous_results = await run_sweep(f"{BASE_URL}/generate_continuous", "CONTINUOUS BATCHING")

    import json
    with open("benchmark/results.json", "w") as f:
        json.dump({
            "static_batching": static_results,
            "continuous_batching": continuous_results,
        }, f, indent=2)
    print("\nSaved results to benchmark/results.json")

asyncio.run(main())