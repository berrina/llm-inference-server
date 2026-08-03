import requests
import time

URL = "http://localhost:8000/generate"
NUM_REQUESTS = 10
PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 20

latencies = []

print(f"Sending {NUM_REQUESTS} sequential requests...\n")
overall_start = time.time()

for i in range(NUM_REQUESTS):
    start = time.time()
    response = requests.post(URL, json={"prompt": PROMPT, "max_new_tokens": MAX_NEW_TOKENS})
    elapsed = time.time() - start
    latencies.append(elapsed)
    print(f"Request {i+1}: {elapsed:.2f}s")

overall_elapsed = time.time() - overall_start
avg_latency = sum(latencies) / len(latencies)
total_tokens = NUM_REQUESTS * MAX_NEW_TOKENS
throughput = total_tokens / overall_elapsed

print(f"\n--- Results ---")
print(f"Total time: {overall_elapsed:.2f}s")
print(f"Average latency per request: {avg_latency:.2f}s")
print(f"Throughput: {throughput:.2f} tokens/sec")