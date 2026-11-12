# r5-glue: per-step critical-path saving and G1 estimate from t2_glue.txt (p50 per grid, both ranks averaged).
# RQPAD: new kernel time T_new (floor fit) replaces every call's time; RPSPLIT: fixed saving per call.
import re, collections
G = {"code": 131.63, "prose": 95.11, "maths": 163.95}; T = {"code": 22.2, "prose": 19.4, "maths": 22.0}  # brief (prod)
G1 = 130.23
rows = collections.defaultdict(list)
wl = None
for ln in open("t2_glue.txt"):
    m = re.match(r"#### (\w+) (q\d)", ln)
    if m: wl = m.group(1); continue
    m = re.search(r"(_r2_reduce_quant|_dsv41_rp_kernel)\s+grid (\d+)x(\d+): n/st\s+([\d.]+) .*p50\s+([\d.]+).*on-path n/st\s+([\d.]+).*succ-gap p50\s+(-?[\d.]+)", ln)
    if m:
        k, gx, gy, n, p50, onp, sg = m.groups()
        rows[(wl, k, int(gx), int(gy))].append((float(onp), float(p50), float(p50) + float(sg)))
def g1(dus):  # dus: workload -> us/step saved
    return sum(G[w] * dus[w] / (T[w] * 1000) for w in G) / 3 / G1 * 100
print("RQPAD (rq grid 4xM, M = grid y); saving per call = p50 - T_new")
for tnew in (3.2, 2.5, 2.1):
    d = {w: 0.0 for w in G}
    for (w, k, gx, gy), v in rows.items():
        if k != "_r2_reduce_quant": continue
        n = sum(x[0] for x in v) / len(v); p = sum(x[1] for x in v) / len(v)
        d[w] += n * max(0.0, p - tnew)
    print(f"  T_new {tnew}: us/step code {d['code']:.0f} prose {d['prose']:.0f} maths {d['maths']:.0f} -> G1 {g1(d):+.2f}%")
# review N7: wo_b (PDL secondary) launches `off` = p50 + succ-gap after rq starts and runs its pre-wait prologue P
# before griddepcontrol.wait; the path from rq's start is max(T_rq, off + P), so the saving per call is
# p50 - max(T_new, off + P), not p50 - T_new.
print("RQPAD with the wo_b launch-offset floor: saving per call = p50 - max(T_new, off + P), off = p50 + succ-gap")
offs = collections.defaultdict(list)
for (w, k, gx, gy), v in rows.items():
    if k == "_r2_reduce_quant":
        offs[gy] += [x[2] for x in v]
print("  off (us) by M: " + ", ".join(f"M={M} {sum(o) / len(o):.2f}" for M, o in sorted(offs.items())))
for tnew in (3.2, 2.5, 2.1):
    for P in (0.0, 0.5, 1.0, 1.5):
        d = {w: 0.0 for w in G}
        for (w, k, gx, gy), v in rows.items():
            if k != "_r2_reduce_quant": continue
            n = sum(x[0] for x in v) / len(v); p = sum(x[1] for x in v) / len(v); o = sum(x[2] for x in v) / len(v)
            d[w] += n * max(0.0, p - max(tnew, o + P))
        print(f"  T_new {tnew} P {P:.1f}: us/step code {d['code']:.0f} prose {d['prose']:.0f} maths {d['maths']:.0f}"
              f" -> G1 {g1(d):+.2f}%")
print("RPSPLIT (rp on path n/st x saving per call)")
for s in (1.0, 2.0, 3.0, 4.0):
    d = {w: 0.0 for w in G}
    for (w, k, gx, gy), v in rows.items():
        if k != "_dsv41_rp_kernel": continue
        d[w] += sum(x[0] for x in v) / len(v) * s
    print(f"  {s:.1f} us/call: us/step code {d['code']:.0f} prose {d['prose']:.0f} maths {d['maths']:.0f} -> G1 {g1(d):+.2f}%")
