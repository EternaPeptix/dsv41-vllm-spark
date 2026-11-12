# r5-glue: re-base RQPAD / RPSPLIT on today's t2 production traces (launcher 166d7aac).
# Run on 512S1 from ~/climb-traces:  python3 r5-glue/t2_glue.py
# Per trace: per-grid duration stats of _r2_reduce_quant and _dsv41_rp_kernel, calls/step, how many are on the
# walkB critical path (huntA walker, l2pf never a predecessor), and the gap from each kernel's end to the next
# kernel on the path (if the successor starts > 1 us after our end, our end is not what gates it).
# Steps: 60 per window; gaps > 50 ms (profiler/capture start) are excluded from the span.
import os, sys, statistics as S, collections
sys.path.insert(0, 'huntA')
from cp import load
from seg import walkB

IT = 60
NAMES = ("_r2_reduce_quant", "_dsv41_rp_kernel")
WL = {"1": "code", "2": "prose", "3": "maths"}


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))] if v else float("nan")


for w in ("1", "2", "3"):
    for q in ("q0", "q4"):
        p = os.path.expanduser(f"~/climb-traces/t2/prod/{q}.{w}.sqlite")
        k = load(p)
        # span without big idle gaps (> 50 ms)
        ks = sorted(k, key=lambda x: x[0])
        span = ks[-1][1] - ks[0][0]
        mx = ks[0][1]
        big = 0
        for a, b, *_ in ks[1:]:
            if a - mx > 50e6:
                big += a - mx
            mx = max(mx, b)
        step_ms = (span - big) / 1e6 / IT
        path = walkB(k)
        pos = {i: j for j, i in enumerate(path)}
        print(f"\n#### {WL[w]} {q}  step {step_ms:.3f} ms (span minus {big/1e6:.1f} ms idle)")
        for nm in NAMES:
            idx = [i for i, x in enumerate(k) if x[2] == nm]
            by = collections.defaultdict(list)
            for i in idx:
                by[k[i][4]].append(i)
            for g in sorted(by, key=lambda g: (g[0], g[1])):
                L = by[g]
                d = [(k[i][1] - k[i][0]) / 1e3 for i in L]
                onp = [i for i in L if i in pos]
                # successor gap on the path (us); onpath time = non-overlapped part
                sg = []
                ot = []
                for i in onp:
                    j = pos[i]
                    prev_end = k[path[j - 1]][1] if j > 0 else k[i][0]
                    ot.append(max(0, k[i][1] - max(k[i][0], prev_end)) / 1e3)
                    if j + 1 < len(path):
                        sg.append((k[path[j + 1]][0] - k[i][1]) / 1e3)
                print(f"   {nm:18s} grid {g[0]}x{g[1]}: n/st {len(L)/IT:5.2f} p10 {pct(d,.1):5.2f} p50 {pct(d,.5):5.2f} "
                      f"p90 {pct(d,.9):5.2f} mean {S.mean(d):5.2f} | on-path n/st {len(onp)/IT:5.2f} "
                      f"onpath-us/st {sum(ot)/IT:6.1f} succ-gap p50 {pct(sg,.5):6.2f} "
                      f"(succ starts >1us after: {sum(1 for x in sg if x > 1.0)}/{len(sg)})")
