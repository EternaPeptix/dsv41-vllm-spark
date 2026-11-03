#!/usr/bin/env python3
"""verdict.py <tag>.allranks.json: gates from the brief. Means over 8 ranks of each rank's ABBA median; never edits data.
  split  GO only if at AG M=6 and AR M=6, in b2b AND l2pf, both in-kernel net (pub->flags) and us/op are >= 1.0 us lower.
  v1v2   per publish = saving / publishes (AG 1; two-shot AR 2): GO >= 0.6 us, KILL < 0.35, else HOLD. Saving = min(us/op, in-kernel total).
  others no threshold in the brief: printed, with the same deltas; decide by hand.
Needs RDMA_AR_TIMING=1 data for net/total; with timing off only us/op is printed."""
import json
import statistics as st
import sys

rows = json.load(open(sys.argv[1]))
print("failures per rank:", [r["fails"] for r in rows])
cells = {}
for r in rows:
    for c in r["L"]:
        cells.setdefault((c["mode"], c["kind"], c["M"]), []).append(c)
names = ("stage", "net", "tail", "total", "handoff", "post")


def mean(cs, arm, f):
    return st.mean(f(c, arm) for c in cs)


split_ok, p1, seen = True, [], False
for (mode, kind, M), cs in sorted(cells.items()):
    arms = [a for a in cs[0]["us_op"]]
    base = {"us": mean(cs, "ctl", lambda c, a: c["us_op"][a])}
    print(f"\n== {mode} {kind} M={M}   ctl {base['us']:.2f} us/op")
    for arm in arms:
        us = mean(cs, arm, lambda c, a: c["us_op"][a])
        pt = [mean(cs, arm, lambda c, a, i=i: c["parts"][a][i]) for i in range(6)]
        pc = [mean(cs, "ctl", lambda c, a, i=i: c["parts"][a][i]) for i in range(6)]
        tr = [st.median(c["trials"][arm][i] - c["trials"]["ctl"][i] for c in cs) for i in range(len(cs[0]["trials"][arm]))]
        wins = sum(1 for d in tr if d < 0)
        print(f"  {arm:6s} us/op {us:7.2f} ({us - base['us']:+6.2f}; lower in {wins}/{len(tr)} trials)  " +
              " ".join(f"{n} {v - w:+6.2f}" for n, v, w in zip(names, pt, pc)))
        if arm == "split" and kind + str(M) in ("ag6", "ar6"):
            seen = True
            split_ok &= (base["us"] - us >= 1.0) and (pc[1] - pt[1] >= 1.0)
        if arm == "v1v2":
            pubs = 1 if kind == "ag" else (2 if M * 5120 * 2 >= 32768 else 1)
            p1.append((mode, kind, M, min(base["us"] - us, pc[3] - pt[3]) / pubs))
print()
if seen:
    print("split post:", "GO (>= 1 us lower in net and us/op, every cell)" if split_ok else "KILL (misses 1 us in at least one cell)")
if p1:
    worst = min(x[3] for x in p1)
    print("V1+V2 per publish:", " ".join(f"{m}/{k}{M} {v:+.2f}" for m, k, M, v in p1))
    print("V1+V2:", "GO (worst >= 0.6)" if worst >= 0.6 else "KILL (worst < 0.35)" if worst < 0.35 else "HOLD (0.35..0.6: below the bar, above the kill)")
