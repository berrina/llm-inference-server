import asyncio
import time
import torch
from transformers import DynamicCache
from server.model_runner import model, tokenizer
from server.metrics import record_request

MAX_CONCURRENT_SLOTS = 8


class Slot:
    """One in-flight request: tracks its own KV cache and generation state."""
    def __init__(self, request_id, prompt_ids, max_new_tokens, future, submit_time):
        self.request_id = request_id
        self.prompt_ids = prompt_ids
        self.max_new_tokens = max_new_tokens
        self.future = future
        self.submit_time = submit_time
        self.first_token_time = None
        self.generated_ids = []
        self.past_key_values = None   # list of (key, value) tensors per layer
        self.cache_len = 0            # number of REAL tokens cached (no padding)
        self.pending_input_id = None
        self.done = False

    def prefill(self):  # this fills the cache with the prompt
        with torch.no_grad():
            out = model(input_ids=self.prompt_ids, use_cache=True)
        self.past_key_values = [(l.keys, l.values) for l in out.past_key_values.layers]
        self.cache_len = self.prompt_ids.shape[1]
        next_id = out.logits[:, -1, :].argmax(dim=-1)
        self.pending_input_id = next_id.unsqueeze(0)
        self.generated_ids.append(next_id.item())
        self.first_token_time = time.time()
        if next_id.item() == tokenizer.eos_token_id or len(self.generated_ids) >= self.max_new_tokens:
            self.done = True

    def text(self):
        return tokenizer.decode(self.generated_ids, skip_special_tokens=True)


def _pad_and_batch_past(slots, target_len):
    num_layers = len(slots[0].past_key_values)
    batched = []
    for layer_idx in range(num_layers):
        keys, values = [], []
        for s in slots:
            k, v = s.past_key_values[layer_idx]
            pad_len = target_len - k.shape[2]
            if pad_len > 0:
                pad_k = list(k.shape); pad_k[2] = pad_len
                pad_v = list(v.shape); pad_v[2] = pad_len
                k = torch.cat([torch.zeros(pad_k, dtype=k.dtype), k], dim=2)
                v = torch.cat([torch.zeros(pad_v, dtype=v.dtype), v], dim=2)
            keys.append(k)
            values.append(v)
        batched.append((torch.cat(keys, dim=0), torch.cat(values, dim=0)))
    return batched


def decode_step(active_slots):  # this is the core of continuous batching: one forward pass across all active slots
    """One batched forward pass across ALL currently active slots -
    the core of continuous batching: different requests, different ages,
    processed together in a single model call."""
    target_len = max(s.cache_len for s in active_slots)
    pad_lens = [target_len - s.cache_len for s in active_slots]

    batched_past = _pad_and_batch_past(active_slots, target_len)

    attn_rows, pos_rows, input_rows = [], [], []
    for s, pad_len in zip(active_slots, pad_lens):
        attn_rows.append([0] * pad_len + [1] * s.cache_len + [1])
        pos_rows.append([s.cache_len])
        input_rows.append(s.pending_input_id)

    attention_mask = torch.tensor(attn_rows, dtype=torch.long)
    position_ids = torch.tensor(pos_rows, dtype=torch.long)
    input_ids = torch.cat(input_rows, dim=0)

    cache_obj = DynamicCache()
    for layer_idx, (k, v) in enumerate(batched_past):
        cache_obj.update(k, v, layer_idx)

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            past_key_values=cache_obj,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )

    next_tokens = out.logits[:, -1, :].argmax(dim=-1)
    out_past_raw = [(l.keys, l.values) for l in out.past_key_values.layers]

    for i, (s, pad_len) in enumerate(zip(active_slots, pad_lens)):
        # Strip each slot's own padding back off - keep only its real tokens
        s.past_key_values = [
            (k[i:i+1, :, pad_len:, :], v[i:i+1, :, pad_len:, :])
            for (k, v) in out_past_raw
        ]
        s.cache_len += 1
        s.pending_input_id = next_tokens[i].unsqueeze(0).unsqueeze(0)
        tok = next_tokens[i].item()
        s.generated_ids.append(tok)
        if tok == tokenizer.eos_token_id or len(s.generated_ids) >= s.max_new_tokens:
            s.done = True


class ContinuousBatchingScheduler:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.active_slots = []
        self._next_id = 0

    async def submit(self, prompt: str, max_new_tokens: int) -> str:
        future = asyncio.get_event_loop().create_future()
        submit_time = time.time()
        await self.queue.put((prompt, max_new_tokens, future, submit_time))
        return await future

    def _make_slot(self, prompt, max_new_tokens, future, submit_time):
        ids = tokenizer(prompt, return_tensors="pt").input_ids
        slot = Slot(self._next_id, ids, max_new_tokens, future, submit_time)
        self._next_id += 1
        return slot

    def _finalize(self, s):
        completion_time = time.time()
        record_request(
            "continuous_batching", s.submit_time, completion_time,
            ttft=s.first_token_time - s.submit_time,
            tokens_generated=len(s.generated_ids),
        )
        s.future.set_result(s.text())

    async def run_forever(self):
        while True:
            # Admit new requests into any free slots, right now, mid-stream
            while len(self.active_slots) < MAX_CONCURRENT_SLOTS and not self.queue.empty():
                prompt, max_new_tokens, future, submit_time = self.queue.get_nowait()
                slot = self._make_slot(prompt, max_new_tokens, future, submit_time)
                slot.prefill()
                if slot.done:
                    self._finalize(slot)
                else:
                    self.active_slots.append(slot)

            if not self.active_slots:
                # nothing to do - wait for the next request to arrive
                prompt, max_new_tokens, future, submit_time = await self.queue.get()
                slot = self._make_slot(prompt, max_new_tokens, future, submit_time)
                slot.prefill()
                if slot.done:
                    self._finalize(slot)
                else:
                    self.active_slots.append(slot)
                continue

            decode_step(self.active_slots)

            still_active = []
            for s in self.active_slots:
                if s.done:
                    self._finalize(s)
                else:
                    still_active.append(s)
            self.active_slots = still_active

            await asyncio.sleep(0)  # yield to event loop between steps


continuous_scheduler = ContinuousBatchingScheduler()