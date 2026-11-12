#!/usr/bin/env python3
"""g130 fc1 verdict from probe JSONL.  python3 -I eval_probe.py results/probe.jsonl [--hist hist-prose.json hist-code.json] [--prior results/g_prior.json]
Weights: per-(tokens, G) histogram of serving FC1 calls. With --hist (recorded G1 routes, one file for each type) those are used.
Otherwise results/g_prior.json, the trace inversion of ~/dsv41-g130prof4 (G inferred from FC1 duration, so it is a prior).

Saving per call on each mix is the paired median saving, weighted over G, for each deal arm.
  GO    prose >= 3.5 us AND code >= 2.5 us per call  (about 0.10-0.14 ms per step, about +0.45% G1, bundle or pair test next)
  KILL  prose < 1.5 us AND code < 1.5 us per call     (below what a +0.5% bar could ever see)
  HOLD  otherwise: bundle rider only, no solo pinned pair
Extra KILL: any arm slower than 'all' by > 2 us per call on the G that dominates a mix (> 15% of its calls), which means C fell.
A0 checks: 'all' must reproduce the staircase jump (G 16 -> 17 at tokens >= 4, >= +15 us). If it does not, the model is wrong:
the probe has answered the question and the number below still decides, but say so.
STAIRCASE line: the model's only step below G 14 is G 8 -> 9. It holds if the median measured step on 'all' is at least half the
model's, and it counts as not tested if any recorded G from 1 to min(nmb, 14) is missing. It informs, it never changes the verdict."""
import collections
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # python3 -I drops the script directory
import fc1fluid  # noqa: E402

P = sys.argv[1]
rows = [json.loads(x) for x in open(P) if x.strip().startswith("{")]
pts = [r for r in rows if r.get("event") == "p"]
A5 = {(r["tokens"], r["G"], r["arm"]): r for r in pts if r["section"] == "A5"}
A0 = {(r["tokens"], r["G"], r["arm"]): r for r in pts if r["section"] in ("A0", "A5")}
args = sys.argv[2:]
hist = {}
if "--hist" in args:
    i = args.index("--hist")
    hist["prose"] = json.load(open(args[i + 1]))["by_tokens"]
    hist["code"] = json.load(open(args[i + 2]))["by_tokens"]
    src = "recorded routes"
else:
    i = args.index("--prior") if "--prior" in args else -1
    pr = json.load(open(args[i + 1])) if i >= 0 else json.load(open(P.rsplit("/", 1)[0] + "/g_prior.json"))
    hist = pr
    src = "trace-inverted prior (prof4)"

arms = sorted({a for _, _, a in A5 if a != "all"})
print("weights:", src)


def nearest_tokens(t):
    ts = sorted({k[0] for k in A5})
    return min(ts, key=lambda x: abs(x - t))


verdict = {}
slow = []
for arm in arms:
    res = {}
    for mix in ("prose", "code"):
        tot = cov = sav = 0.0
        byg = collections.Counter()
        for t, c in hist[mix].items():
            t = int(t)
            tt = nearest_tokens(t)
            for G, n in c.items():
                G = int(G)
                tot += n
                r = A5.get((tt, G, arm))
                if r is None:
                    continue
                cov += n
                sav += n * r["saving_us"]
                byg[(tt, G)] += n
        res[mix] = sav / cov if cov else float("nan")
        print("  %-6s %-5s saving %6.2f us/call, coverage %.0f%% of calls" % (arm, mix, res[mix], 100 * cov / max(tot, 1)))
        for (tt, G), n in byg.items():
            if n / max(cov, 1) > 0.15:
                r = A5[(tt, G, arm)]
                if r["saving_us"] < -2.0:
                    slow.append((arm, mix, tt, G, r["saving_us"]))
    if res["prose"] >= 3.5 and res["code"] >= 2.5:
        v = "GO"
    elif res["prose"] < 1.5 and res["code"] < 1.5:
        v = "KILL"
    else:
        v = "HOLD"
    if any(s[0] == arm for s in slow):
        v = "KILL"
    verdict[arm] = (v, res)
    print("  %-6s -> %s" % (arm, v))
for s in slow:
    print("  slow on a dominant G: arm %s mix %s tokens %d G %d saving %.2f us" % s)
print()
print("A0 staircase, arm all (us per call):")
stair_gap, stair_89 = [], []
for t in sorted({k[0] for k in A0}):
    line = [(g, A0[(t, g, "all")]["us"]) for g in range(1, 41) if (t, g, "all") in A0]
    print("  tokens %d: %s" % (t, " ".join("%d:%.1f" % x for x in line)))
    d = dict(line)
    nmb = A0[(t, line[0][0], "all")].get("nmb", 6 * t) if line else 6 * t
    gap = [g for g in range(1, min(nmb, 14) + 1) if g not in d]
    if gap:
        stair_gap.append("tokens %d G %s" % (t, ",".join(map(str, gap))))
    if 16 in d and 17 in d:
        print("    jump 16 -> 17: %+.1f us (fluid model %+.1f, GO if it is there: >= +15)" % (d[17] - d[16], fc1fluid.T(17, nmb) - fc1fluid.T(16, nmb)))
    if 8 in d and 9 in d:
        step = fc1fluid.T(9, nmb) - fc1fluid.T(8, nmb)
        print("    jump 8 -> 9:   %+.1f us (fluid model %+.1f)" % (d[9] - d[8], step))
        stair_89.append((d[9] - d[8], step))
if stair_gap or not stair_89:
    print("STAIRCASE not tested below G 14: missing %s" % ("; ".join(stair_gap) or "G 8 and 9 on arm all"))
else:
    med = statistics.median(x for x, _ in stair_89)
    want = statistics.median(m for _, m in stair_89)
    print("STAIRCASE %s below G 14: median G 8 -> 9 step %+.1f us, model %+.1f, %d tokens values" % (
        "holds" if med >= 0.5 * want else "model wrong", med, want, len(stair_89)))
print()
for sec, lab in (("A1", "t0 (G=0)"), ("A2", "L2 warm"), ("A34", "ablation")):
    s = [r for r in pts if r["section"] == sec]
    if s:
        print(lab, ":", " ".join("%s/t%d/G%d=%.1f" % (r["arm"], r["tokens"], r["G"], r["us"]) for r in s[:80]))
a6 = [r for r in pts if r["section"] == "A6"]
for r in a6:
    print("A6 tokens %d arm %s us %.2f saving %s meanG %.2f steps %d" % (r["tokens"], r["arm"], r["us"], r.get("saving_us"), r["meanG"], r["steps"]))
print()
best = max(verdict.items(), key=lambda kv: min(kv[1][1]["prose"], kv[1][1]["code"]) if verdict else 0) if verdict else None
if best:
    print("VERDICT", best[0], best[1][0], "prose %.2f code %.2f us/call" % (best[1][1]["prose"], best[1][1]["code"]))
sys.exit(0)
