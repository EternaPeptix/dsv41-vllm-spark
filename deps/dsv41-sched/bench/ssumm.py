#!/usr/bin/env python3
"""ssumm.py <jsonl...> : per-label summary of sbench rows (pooled shorts)."""
import json, statistics, sys
from collections import defaultdict


def pct(v, q):
    v = sorted(v)
    if not v:
        return None
    k = (len(v) - 1) * q
    lo = int(k); hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


rows = []
for f in sys.argv[1:]:
    for l in open(f):
        l = l.strip()
        if l.startswith("{"):
            d = json.loads(l); d["_f"] = f; rows.append(d)
g = defaultdict(list)
for d in rows:
    big = (d.get("long_prompt_tokens") or d.get("prompt_tokens") or 0) > 400000
    g[(d["mode"], d["label"], "1M" if big else "128K")].append(d)
for (m, lab, sz), v in sorted(g.items()):
    if m == "starve":
        sh = [x for d in v for x in d["short_ttft_s"]]
        lt = [d["long_ttft_s"] for d in v]
        dg = [d.get("dec_gap_inside_max_ms") for d in v if d.get("dec_gap_inside_max_ms")]
        print(json.dumps({"mode": m, "label": lab, "size": sz, "runs": len(v),
                          "short_p50": round(statistics.median(sh), 2), "short_p95": round(pct(sh, .95), 2),
                          "short_max": round(max(sh), 2), "long_ttft": [round(x, 2) for x in lt],
                          "long_mean": round(statistics.mean(lt), 2),
                          "long_tok_s": round(statistics.mean([d["long_prompt_tokens"] / d["long_ttft_s"] for d in v]), 1),
                          "dec_gap_max_ms": dg, "run_p50s": [d["short_ttft_p50_s"] for d in v],
                          "long_tok": [d["long_prompt_tokens"] for d in v]}))
    elif m == "solo":
        t = [d["ttft_s"] for d in v]
        print(json.dumps({"mode": m, "label": lab, "size": sz, "reps": len(v), "ttft": t,
                          "mean": round(statistics.mean(t), 3),
                          "tok_s": [d["tok_s"] for d in v], "tok": [d["prompt_tokens"] for d in v]}))
