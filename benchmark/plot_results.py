import json
import matplotlib.pyplot as plt

with open("benchmark/results.json") as f:
    data = json.load(f)

static = data["static_batching"]
continuous = data["continuous_batching"]

concurrency = [r["concurrency"] for r in static]

# --- Chart 1: Throughput vs Concurrency ---
plt.figure(figsize=(8, 5))
plt.plot(concurrency, [r["throughput"] for r in static], marker="o", label="Static Batching")
plt.plot(concurrency, [r["throughput"] for r in continuous], marker="o", label="Continuous Batching")
plt.xlabel("Concurrency Level")
plt.ylabel("Throughput (tokens/sec)")
plt.title("Throughput vs. Concurrency")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("benchmark/chart_throughput.png", dpi=150, bbox_inches="tight")
plt.close()

# --- Chart 2: Latency (p50 / p99) vs Concurrency ---
plt.figure(figsize=(8, 5))
plt.plot(concurrency, [r["p50"] for r in static], marker="o", label="Static - p50")
plt.plot(concurrency, [r["p99"] for r in static], marker="o", linestyle="--", label="Static - p99")
plt.plot(concurrency, [r["p50"] for r in continuous], marker="s", label="Continuous - p50")
plt.plot(concurrency, [r["p99"] for r in continuous], marker="s", linestyle="--", label="Continuous - p99")
plt.xlabel("Concurrency Level")
plt.ylabel("Latency (seconds)")
plt.title("Latency (p50 / p99) vs. Concurrency")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("benchmark/chart_latency.png", dpi=150, bbox_inches="tight")
plt.close()

# --- Chart 3: Before/After summary bar chart ---
plt.figure(figsize=(8, 5))
labels = ["Sequential\n(no batching)", "Static Batching\n(concurrency=8)", "Continuous Batching\n(concurrency=8)"]
sequential_throughput = 35.71  # your Phase 1 baseline number
static_8 = next(r["throughput"] for r in static if r["concurrency"] == 8)
continuous_8 = next(r["throughput"] for r in continuous if r["concurrency"] == 8)
values = [sequential_throughput, static_8, continuous_8]

bars = plt.bar(labels, values, color=["#888888", "#4C72B0", "#55A868"])
plt.ylabel("Throughput (tokens/sec)")
plt.title("Throughput: Before vs. After Batching")
for bar, val in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, val + 2, f"{val:.1f}", ha="center")
plt.savefig("benchmark/chart_summary.png", dpi=150, bbox_inches="tight")
plt.close()

print("Saved 3 charts to benchmark/:")
print("  chart_throughput.png")
print("  chart_latency.png")
print("  chart_summary.png")