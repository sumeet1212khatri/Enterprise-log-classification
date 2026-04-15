"""
classify.py — 3-Tier Hybrid Pipeline (V5 — Multiprocessing & Max RAM Utilization)

Architecture:
  LegacyCRM → LLM directly
  Others    → Regex → BERT (batch) → LLM fallback

Changes in V5:
  - Added ProcessPoolExecutor for classify_csv to distribute workload across all CPU cores.
  - Increased LRU cache size to 500,000 to maximize RAM usage and minimize LLM calls during 2M+ log stress tests.
  - High-resolution telemetry and True Batch Latency tracking retained from V4.
"""
from __future__ import annotations
import os
import time
import hashlib
import statistics
import pandas as pd
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from processor_regex import classify_with_regex
from processor_bert  import classify_batch as bert_batch
from processor_llm   import classify_with_llm

LEGACY_SOURCE = "LegacyCRM"


# ── Result type ─────────────────────────────────────────────────────────────
def _make_result(label: str, tier: str, confidence, latency_ms: float) -> dict:
    return {
        "label":      label,
        "tier":       tier,
        "confidence": confidence,
        "latency_ms": round(latency_ms, 4), 
    }


# ── Caching Layer (V5 UPDATE: Max RAM Eater) ────────────────────────────────
@lru_cache(maxsize=500000) # Increased to 500k to absorb duplicate logs in 2M+ datasets
def cached_llm_call(log_hash: str, log_msg: str) -> str:
    """Only executes the expensive LLM call if the MD5 hash misses the cache."""
    return classify_with_llm(log_msg)


# ── Single log (backward-compatible) ────────────────────────────────────────
def classify_log(source: str, log_msg: str) -> dict:
    """Classify a single log. Returns label, tier, confidence, and latency_ms."""
    results = classify_logs([(source, log_msg)])
    return results[0]


# ── Batch pipeline (main entry point) ───────────────────────────────────────
def classify_logs(logs: list[tuple[str, str]]) -> list[dict]:
    """
    Batch classify with 3-tier routing + per-result latency.
    """
    n       = len(logs)
    results = [None] * n

    # ── Step 1: Route to groups ─────────────────────────────────────────────
    llm_indices   = []
    bert_indices  = []

    for i, (source, log_msg) in enumerate(logs):
        if source == LEGACY_SOURCE:
            llm_indices.append(i)
        else:
            t_start = time.perf_counter()
            label = classify_with_regex(log_msg)
            
            if label:
                latency_ms = (time.perf_counter() - t_start) * 1000
                results[i] = _make_result(label, "Regex", 1.0, latency_ms)
            else:
                bert_indices.append(i)

    # ── Step 2: BERT batch ──────────────────────────────────────────────────
    if bert_indices:
        bert_msgs = [logs[i][1] for i in bert_indices]

        t_bert_start = time.perf_counter()
        bert_results = bert_batch(bert_msgs)
        t_bert_end   = time.perf_counter()

        bert_ms_per_log = (t_bert_end - t_bert_start) * 1000 / len(bert_msgs)

        for idx, (label, conf) in zip(bert_indices, bert_results):
            if label != "Unclassified":
                results[idx] = _make_result(label, "BERT", conf, bert_ms_per_log)
            else:
                llm_indices.append(idx)

    # ── Step 3: LLM (Parallel Concurrency & Caching) ────────────────────────
    if llm_indices:
        def parallel_llm(idx):
            src, msg = logs[idx]
            
            log_hash = hashlib.md5(msg.encode('utf-8')).hexdigest()
            
            t_llm_0 = time.perf_counter()
            label = cached_llm_call(log_hash, msg)
            t_llm_ms = (time.perf_counter() - t_llm_0) * 1000
            
            base_tier = "LLM" if src == LEGACY_SOURCE else "LLM (fallback)"
            tier = f"{base_tier} (Cache Hit)" if t_llm_ms < 5 else f"{base_tier} (API Call)"
            
            return idx, _make_result(label, tier, None, t_llm_ms)

        with ThreadPoolExecutor(max_workers=4) as executor:
            llm_results = list(executor.map(parallel_llm, llm_indices))

        for idx, res in llm_results:
            results[idx] = res

    return results


# ── Pipeline summary ─────────────────────────────────────────────────────────
def pipeline_summary(results: list[dict]) -> dict:
    """
    Aggregate stats from classify_logs output.
    Useful for dashboard and benchmark reporting.
    """
    tier_groups: dict[str, list[float]] = {}
    label_counts: dict[str, int] = {}

    for r in results:
        tier = r["tier"]
        tier_groups.setdefault(tier, []).append(r["latency_ms"])
        label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1

    total = len(results)
    tier_stats = {}
    for tier, latencies in tier_groups.items():
        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)
        tier_stats[tier] = {
            "count":    n,
            "pct":      round(n / total * 100, 1),
            "p50_ms":   round(statistics.median(latencies_sorted), 4),
            "p95_ms":   round(latencies_sorted[min(int(n * 0.95), n - 1)], 4),
            "p99_ms":   round(latencies_sorted[min(int(n * 0.99), n - 1)], 4),
            "mean_ms":  round(statistics.mean(latencies_sorted), 4),
            "total_ms": round(sum(latencies_sorted), 4), 
        }

    return {
        "total":        total,
        "tier_stats":   tier_stats,
        "label_counts": label_counts,
    }


# ── Multiprocessing Helper ───────────────────────────────────────────────────
def _process_chunk(chunk: list[tuple[str, str]]) -> list[dict]:
    """Top-level helper function required for ProcessPoolExecutor mapping."""
    return classify_logs(chunk)


# ── CSV batch classify (V5 UPDATE: Multi-Core Processor) ─────────────────────
def classify_csv(input_path: str, output_path: str = "output.csv") -> tuple[str, pd.DataFrame]:
    """
    Ultra-Optimized Batch Processing for 2M+ Logs.
    Uses ProcessPoolExecutor to max out CPU cores and gorge on RAM.
    """
    df = pd.read_csv(input_path)
    required = {"source", "log_message"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns in CSV. Expected: {required}. Found: {set(df.columns)}")

    log_pairs = list(zip(df["source"], df["log_message"]))
    total_logs = len(log_pairs)
    
    # OS ke liye 1 core chhod kar baaki sab use karo
    max_cores = max(1, os.cpu_count() - 1)
    
    # 50,000 logs ke chunks banao (Memory distribution)
    chunk_size = 50000 
    chunks = [log_pairs[i:i + chunk_size] for i in range(0, total_logs, chunk_size)]
    
    results = []
    
    print(f"🔥 Firing up {max_cores} CPU cores to process {len(chunks)} chunks...")
    with ProcessPoolExecutor(max_workers=max_cores) as executor:
        for chunk_result in executor.map(_process_chunk, chunks):
            results.extend(chunk_result)

    df["predicted_label"] = [r["label"]       for r in results]
    df["tier_used"]       = [r["tier"]        for r in results]
    df["latency_ms"]      = [r["latency_ms"]  for r in results]
    df["confidence"]      = [
        f"{r['confidence']:.1%}" if r["confidence"] is not None else "N/A"
        for r in results
    ]

    df.to_csv(output_path, index=False)
    return output_path, df


# Aliases
classify = classify_logs


# ── Self-test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Required for safe multiprocessing on Windows/macOS
    # Ensures child processes don't recursively run the __main__ block
    sample = [
        ("ModernCRM",       "IP 192.168.133.114 blocked due to potential attack"),
        ("BillingSystem",   "User User12345 logged in."),
        ("AnalyticsEngine", "File data_6957.csv uploaded successfully by user User265."),
        ("ModernHR",        "GET /v2/servers/detail HTTP/1.1 status: 200 len: 1583 time: 0.19"),
        ("ModernHR",        "Admin access escalation detected for user 9429"),
        ("LegacyCRM",       "Case escalation for ticket ID 7324 failed because the assigned support agent is no longer active."),
        ("LegacyCRM",       "Case escalation for ticket ID 7324 failed because the assigned support agent is no longer active."), 
    ]

    print(f'{"Source":<20} {"Tier":<22} {"Conf":>6} {"Lat(ms)":>8}  {"Label":<25} Log')
    print("─" * 115)
    results = classify_logs(sample)
    for (source, log), r in zip(sample, results):
        conf = f"{r['confidence']:.0%}" if r["confidence"] else "  N/A"
        print(f'{source:<20} {r["tier"]:<22} {conf:>6} {r["latency_ms"]:>8.4f}  {r["label"]:<25} {log[:40]}')

    summary = pipeline_summary(results)
    print("\n📊 Pipeline Summary:")
    
    for tier, stats in summary["tier_stats"].items():
        if tier == "BERT":
             print(f"  BERT Batch Latency: {stats['total_ms']} ms (Amortized over {stats['count']} logs)")
        elif "Regex" in tier:
             print(f"  Regex Latency: < 0.1 ms (Recorded p50: {stats['p50_ms']} ms) | count={stats['count']}")
        else:
            print(f"  {tier}: {stats['count']} logs ({stats['pct']}%) | "
                  f"p50={stats['p50_ms']}ms p95={stats['p95_ms']}ms p99={stats['p99_ms']}ms")

    print("\n🏷️  Label distribution:")
    for label, count in sorted(summary["label_counts"].items(), key=lambda x: -x[1]):
        print(f"  • {label}: {count}")
