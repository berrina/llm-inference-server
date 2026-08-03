from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import DynamicCache

MODEL_NAME = "gpt2"

print("Loading model... this may take a minute the first time (downloads weights).")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32, attn_implementation="eager")
model.eval()

tokenizer.pad_token = tokenizer.eos_token

def generate(prompt: str, max_new_tokens: int = 50) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def generate_batch(prompts: list[str], max_new_tokens: int = 50) -> list[str]:
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]

def generate_with_cache_setting(prompt: str, max_new_tokens: int, use_cache: bool) -> tuple[str, float]:
    """Runs generation with KV cache explicitly on or off, and times it."""
    import time
    inputs = tokenizer(prompt, return_tensors="pt")
    start = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=use_cache,
        )
    elapsed = time.time() - start
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return text, elapsed 