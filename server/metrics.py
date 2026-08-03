import time
import statistics

# Simple in-memory store. Good enough for a benchmarking project;
# a production system would use Prometheus or similar instead.
_records = []

def record_request(endpoint: str, submit_time: float, complete_time: float,
                    ttft: float, tokens_generated: int):
    _records.append({
        "endpoint": endpoint,
        "total_latency": complete_time - submit_time,
        "ttft": ttft,
        "tokens_generated": tokens_generated,
        "timestamp": complete_time,
    })

def get_stats(endpoint: str = None, window_seconds: float = None):
    records = _records
    if endpoint:
        records = [r for r in records if r["endpoint"] == endpoint]
    if window_seconds:
        cutoff = time.time() - window_seconds
        records = [r for r in records if r["timestamp"] >= cutoff]

    if not records:
        return {"count": 0}

    latencies = sorted(r["total_latency"] for r in records)
    ttfts = sorted(r["ttft"] for r in records)
    total_tokens = sum(r["tokens_generated"] for r in records)
    span = max(r["timestamp"] for r in records) - min(r["timestamp"] for r in records)

    if span < 0.01:
        # Not enough of a real time window for "throughput" to mean anything -
        # fall back to averaging each request's own tokens/latency rate
        # instead of dividing by a near-zero span.
        throughput = round(statistics.mean(
            r["tokens_generated"] / r["total_latency"] for r in records
        ), 2)
    else:
        throughput = round(total_tokens / span, 2)

    def percentile(sorted_list, p):
        idx = int(len(sorted_list) * p)
        idx = min(idx, len(sorted_list) - 1)
        return sorted_list[idx]

    return {
        "count": len(records),
        "avg_latency": round(statistics.mean(latencies), 3),
        "p50_latency": round(percentile(latencies, 0.50), 3),
        "p99_latency": round(percentile(latencies, 0.99), 3),
        "avg_ttft": round(statistics.mean(ttfts), 3),
        "throughput_tokens_per_sec": throughput,
    }

def clear():
    _records.clear()