import asyncio
import time
from server.model_runner import generate_batch
from server.metrics import record_request

MAX_BATCH_SIZE = 8
BATCH_WINDOW_MS = 50

class BatchingScheduler:
    def __init__(self):
        self.queue = asyncio.Queue()

    async def submit(self, prompt: str, max_new_tokens: int) -> str:
        future = asyncio.get_event_loop().create_future()
        submit_time = time.time()
        await self.queue.put((prompt, max_new_tokens, future, submit_time))
        return await future

    async def run_forever(self):
        while True:
            batch = []
            deadline = time.time() + (BATCH_WINDOW_MS / 1000)

            while len(batch) < MAX_BATCH_SIZE and time.time() < deadline:
                timeout = deadline - time.time()
                if timeout <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                    batch.append(item)
                except asyncio.TimeoutError:
                    break

            if not batch:
                continue

            prompts = [item[0] for item in batch]
            max_new_tokens = batch[0][1]

            print(f"Processing batch of {len(batch)} request(s)")
            results = generate_batch(prompts, max_new_tokens)
            completion_time = time.time()

            for (prompt, _, future, submit_time), result in zip(batch, results):
                if not future.done():
                    future.set_result(result)
                tokens_generated = len(result.split())  # rough estimate; good enough here
                record_request(
                    "static_batching", submit_time, completion_time,
                    ttft=completion_time - submit_time,  # no partial results, so ttft == total latency
                    tokens_generated=tokens_generated,
                )

scheduler = BatchingScheduler() 