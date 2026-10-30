#!/usr/bin/env python3
"""pool_e5.py <dir> — per instance and pooled per arm (label minus trailing a/b/c letter) e5 summary."""
import collections, glob, json, os, sys
d = sys.argv[1]
inst = {}
for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
    lab = os.path.basename(f)[:-6]
    if "." in lab:
        continue
    rows = [json.loads(l) for l in open(f) if l.strip().startswith("{")]
    res = {}
    for kind in ("greedy", "t1", "t06"):
        rs = [r for r in rows if r.get("label") == f"{lab}.{kind}"]
        if not rs:
            continue
        t = collections.defaultdict(list); al = []
        for r in rs:
            t[r["type"]].append(r.get("decode_tok_s") or 0)
            if (r.get("spec") or {}).get("mean_accept_len"):
                al.append(r["spec"]["mean_accept_len"])
        m = {k: sum(v) / len(v) for k, v in t.items()}
        res[kind] = sum(m.values()) / len(m)
        res[kind + "_al"] = sum(al) / len(al) if al else float("nan")
    for r in rows:
        if r.get("mode") == "conc":
            k = r["label"][len(lab) + 1:]
            res[k] = r["agg_tok_s"]
            res[k + "_al"] = (r.get("spec") or {}).get("mean_accept_len") or float("nan")
    try:
        res["g06"] = json.load(open(os.path.join(d, f"{lab}.g06.summary.json")))["mean_correct"]
    except Exception:
        res["g06"] = float("nan")
    inst[lab] = res
cols = ["greedy", "greedy_al", "greedy.c8", "t1", "t1_al", "t1.c4", "t1.c8", "t1.c16", "t1.c16_al", "t06", "t06_al", "g06"]
print("instance   " + " ".join(f"{c:>10s}" for c in cols))
pool = collections.defaultdict(list)
for lab, res in inst.items():
    print(f"{lab:10s} " + " ".join(f"{res.get(c, float('nan')):10.2f}" for c in cols))
    pool[lab.rstrip("abcdefgh")].append(res)
print("pooled")
for arm, rl in pool.items():
    print(f"{arm + '(' + str(len(rl)) + ')':10s} " + " ".join(
        f"{sum(r.get(c, 0) for r in rl) / len(rl):10.2f}" for c in cols))
