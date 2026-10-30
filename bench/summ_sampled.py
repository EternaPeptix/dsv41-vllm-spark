#!/usr/bin/env python3
"""summ_sampled.py <file.jsonl> [...] — per label: single decode tok/s by type + mean, accept len; conc agg tok/s."""
import collections, json, sys
rows = [json.loads(l) for f in sys.argv[1:] for l in open(f) if l.strip().startswith("{")]
by = collections.defaultdict(list)
for r in rows:
    by[r.get("label", "")].append(r)
for lab, rs in by.items():
    if rs[0].get("mode") == "single":
        t = collections.defaultdict(list); al = []
        for r in rs:
            t[r["type"]].append(r.get("decode_tok_s") or 0)
            if r.get("spec") and r["spec"].get("mean_accept_len"):
                al.append(r["spec"]["mean_accept_len"])
        m = {k: round(sum(v) / len(v), 1) for k, v in t.items()}
        print(f"{lab:28s} single " + " ".join(f"{k}={v}" for k, v in m.items())
              + f" mean={round(sum(m.values()) / len(m), 1)} accept_len={round(sum(al) / len(al), 2) if al else None}")
    else:
        for r in rs:
            print(f"{lab:28s} conc{r['conc']:<3d} agg={r['agg_tok_s']} per_req={r['per_req_decode_tok_s_mean']} "
                  f"accept_len={(r.get('spec') or {}).get('mean_accept_len')}")
