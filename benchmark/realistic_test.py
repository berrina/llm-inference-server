import asyncio
import aiohttp
import time
import random

# Mixed request "profile": (max_new_tokens, arrival_delay_seconds)
# Simulates realistic traffic: some short requests, some long, arriving at
# different times rather than all at once.
random.seed(42)
REQUEST_PROFILE = [
    (random.choice([10, 15, 40, 60]), round(random.uniform(0, 0.3), 2))
    for _ in range(10)
]

PROMPT = "The capital of France is"

async def send_request(session, url, i, max_new_tokens, delay):
    await asyncio.sleep(delay)  # simulate staggered arrival
    start = time.time()
    async with session.post(url, json={"prompt": PROMPT, "max_new_tokens": max_new_tokens}) as resp:
        await resp.json()
        elapsed = time.time() - start
        print(f"  Request {i+1} (max_tokens={max_new_tokens}, delay={delay}s): {elapsed:.2f}s")
        return elapsed, max_new_tokens

async def run_test(url, label):
    print(f"\n=== {label} ===")
    async with aiohttp.ClientSession() as session:
        overall_start = time.time()
        tasks = [
            send_request(session, url, i, tokens, delay)
            for i, (tokens, delay) in enumerate(REQUEST_PROFILE)
        ]
        results = await asyncio.gather(*tasks)
        overall_elapsed = time.time() - overall_start

    total_tokens = sum(tokens for _, tokens in results)
    avg_latency = sum(elapsed for elapsed, _ in results) / len(results)
    throughput = total_tokens / overall_elapsed

    print(f"Total time: {overall_elapsed:.2f}s")
    print(f"Average latency: {avg_latency:.2f}s")
    print(f"Throughput: {throughput:.2f} tokens/sec")
    return overall_elapsed, throughput

async def main():
    static_time, static_tp = await run_test("http://localhost:8000/generate", "STATIC BATCHING")
    continuous_time, continuous_tp = await run_test("http://localhost:8000/generate_continuous", "CONTINUOUS BATCHING")

    print(f"\n=== Comparison ===")
    print(f"Static:     {static_tp:.2f} tokens/sec, {static_time:.2f}s total")
    print(f"Continuous: {continuous_tp:.2f} tokens/sec, {continuous_time:.2f}s total")
    print(f"Speedup: {continuous_tp / static_tp:.2f}x")

asyncio.run(main())