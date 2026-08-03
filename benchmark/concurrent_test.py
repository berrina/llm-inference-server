import asyncio
import aiohttp
import time

URL = "http://localhost:8000/generate"
NUM_REQUESTS = 10
PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 20

async def send_request(session, i):
    start = time.time()
    async with session.post(URL, json={"prompt": PROMPT, "max_new_tokens": MAX_NEW_TOKENS}) as resp:
        result = await resp.json()
        elapsed = time.time() - start
        print(f"Request {i+1}: {elapsed:.2f}s")
        return elapsed

async def main():
    async with aiohttp.ClientSession() as session:
        overall_start = time.time()
        tasks = [send_request(session, i) for i in range(NUM_REQUESTS)]
        latencies = await asyncio.gather(*tasks)
        overall_elapsed = time.time() - overall_start

    avg_latency = sum(latencies) / len(latencies)
    total_tokens = NUM_REQUESTS * MAX_NEW_TOKENS
    throughput = total_tokens / overall_elapsed

    print(f"\n--- Concurrent Results ---")
    print(f"Total time: {overall_elapsed:.2f}s")
    print(f"Average latency per request: {avg_latency:.2f}s")
    print(f"Throughput: {throughput:.2f} tokens/sec")

asyncio.run(main())