#!/usr/bin/env python3
"""dsumm.py <dsuite dir>... : decode + correctness summary rows (one per dir)."""
import json, os, re, sys
def jl(p):
    try: return [json.loads(l) for l in open(p) if l.strip()]
    except Exception: return []
print("dir  code_ms  prose_ms  single(c/p/m)  c8  c32  exact  gsm50  longctx")
for d in sys.argv[1:]:
    out = [os.path.basename(d.rstrip("/"))]
    m = open(f"{d}/matched.log").read() if os.path.exists(f"{d}/matched.log") else ""
    cm = re.search(r"^\s*coding\s+1\s+\S+\s+\S+\s+\S+\s+\S+\s+(\S+)", m, re.M); pm = re.search(r"^\s*prose\s+1\s+\S+\s+\S+\s+\S+\s+\S+\s+(\S+)", m, re.M)
    out += [cm.group(1) if cm else "-", pm.group(1) if pm else "-"]
    sp = {}
    for r in jl(f"{d}/single.jsonl"):
        if "type" in r: sp.setdefault(r["type"], []).append(r["decode_tok_s"])
    out.append("/".join(f"{sum(v)/len(v):.1f}" for k in ("code", "prose", "math") for v in [sp.get(k, [0])]))
    for c in (8, 32):
        r = [x for x in jl(f"{d}/conc{c}.jsonl") if "agg_tok_s" in x]; out.append(f"{r[-1]['agg_tok_s']:.1f}" if r else "-")
    for f in ("exact.log", "gsm8k50.log", "longctx.log"):
        p = f"{d}/{f}"
        out.append(open(p).read().strip().splitlines()[-1][:120] if os.path.exists(p) and os.path.getsize(p) else "-")
    print(" | ".join(out))
