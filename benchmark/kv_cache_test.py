import sys
sys.path.append(".")
from server.model_runner import generate_with_cache_setting

PROMPT = "The capital of France is a beautiful city that"
MAX_NEW_TOKENS = 100  # longer generation makes the cache effect more visible

print("Running WITHOUT KV cache...")
text_no_cache, time_no_cache = generate_with_cache_setting(PROMPT, MAX_NEW_TOKENS, use_cache=False)
print(f"Time: {time_no_cache:.2f}s\n")

print("Running WITH KV cache...")
text_with_cache, time_with_cache = generate_with_cache_setting(PROMPT, MAX_NEW_TOKENS, use_cache=True)
print(f"Time: {time_with_cache:.2f}s\n")

print("--- Results ---")
print(f"Without cache: {time_no_cache:.2f}s")
print(f"With cache:    {time_with_cache:.2f}s")
print(f"Speedup:       {time_no_cache / time_with_cache:.2f}x")