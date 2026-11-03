#!/usr/bin/env python3
"""log_summary.py [~/dsv41-logs/incr-tok.jsonl]: what the overlay did on live traffic (ON spark-one, stdlib only).

Counts per kind (hit, verified, miss_*, MISMATCH), median ms for the incremental path on hits and for the full tokenize on
verified/miss requests, and an estimate of the time saved: for every hit, (full-tokenize us/token measured on this
process's verified and miss requests) x tokens - ms_incr. That estimate is the live counterpart of the CPU bench.
"""
import json
import os
import statistics
import sys

p = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/dsv41-logs/incr-tok.jsonl")
recs = [json.loads(line) for line in open(p) if line.strip()]
kinds = {}
for r in recs:
    kinds[r.get("kind")] = kinds.get(r.get("kind"), 0) + 1
print(f"{p}: {len(recs)} records, {kinds}")
full = [r for r in recs if r.get("ms_full") and r.get("tokens")]
hits = [r for r in recs if r.get("kind") == "hit"]
if full:
    us_tok = statistics.median(r["ms_full"] * 1e3 / r["tokens"] for r in full)
    print(f"full tokenize: median {statistics.median(r['ms_full'] for r in full):.1f} ms, {us_tok:.2f} us/token "
          f"(n={len(full)}, {min(r['tokens'] for r in full)}-{max(r['tokens'] for r in full)} tokens)")
    if hits:
        saved = sum(us_tok * r["tokens"] / 1e3 - r["ms_incr"] for r in hits)
        span_h = max((recs[-1]["t"] - recs[0]["t"]) / 3600, 1e-9)
        print(f"hits: n={len(hits)}, median {statistics.median(r['ms_incr'] for r in hits):.1f} ms, median "
              f"{statistics.median(r['tokens'] for r in hits):.0f} tokens, reused {sum(r['reused'] for r in hits) / max(sum(r['tokens'] for r in hits), 1):.1%}; "
              f"estimated saved {saved / 1e3:.1f} s over {span_h:.2f} h ({saved / 1e3 / span_h * 24:.0f} s/day at this rate)")
bad = [r for r in recs if r.get("kind") in ("MISMATCH", "verify_failed")]
print("MISMATCH records:", len(bad), bad[:1])
