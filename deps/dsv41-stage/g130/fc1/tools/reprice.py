#!/usr/bin/env python3
"""g130 fc1: re-price the G-aware deal on a trace (fluid model of followup-4 + the walk this kit actually builds). No GPU.
   python3 -I reprice.py <fc1fluid.py> <trace.tsv.gz,...> <routes.jsonl ...>
Trace rows: start|end|0|gx|gy|gz|0|0|name (anat-style export). Prints per-trace saving in ms/step, with n-loss sensitivity."""
import collections
import gzip
import importlib.util
import json
import math
import re
import sys

spec = importlib.util.spec_from_file_location("m", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
NN, KB, B, C, F0, T0 = m.NN, m.KB, m.B, m.C, m.F0, m.T0
TAIL = 24


def active(K, tail=TAIL):
    if K <= 48:
        return K
    q = (K + 47) // 48
    n = min(48, (K - tail) // (q - 1))
    return max(n, (K + q - 1) // q)


def per_deal(G):
    K = G * NN
    n = active(K)
    per = [0] * 48
    for c in range(n):
        per[c] = len(range(c, K, n))
    return per


def sim(per, floor=F0, cfn=lambda n: C, t0=T0):
    """Fluid: stage time = max(floor, B * active / C(active))."""
    rem = [p * KB for p in per]
    t = fl = 0.0
    while any(rem):
        act = [r for r in rem if r > 0]
        n = len(act)
        sd = B * n / cfn(n)
        s = max(floor, sd)
        k = min(act)
        t += s * k
        if sd < floor:
            fl += s * k
        rem = [r - k if r > 0 else 0 for r in rem]
    return t0 + t / 1000.0, fl / 1000.0


def Tlive(G, nmb, **kw):
    return sim(m.lists(nmb, G), **kw)[0]


def Tdeal(G, **kw):
    return sim(per_deal(G), **kw)[0]


# Measured sm32 +4.7%, sm40 -0.5..+0.9% (Athena, G >= 36). The loss arm applies C(n) to full FC1 running at n CTAs.
def closs(n):
    return C * (0.955 if n <= 32 else 0.955 + (min(n, 40) - 32) / 8 * 0.045)


rec = collections.defaultdict(collections.Counter)
for p in sys.argv[3:]:
    for line in open(p):
        d = json.loads(line)
        if d.get("event") != "routes" or d.get("has_prefill") in (True, "True") or int(d.get("num_requests", 0)) != 1:
            continue
        pad = d.get("is_padding") or [False] * d["physical_rows"]
        for la in d["ids_by_layer"]:
            es = set(e for r, row in enumerate(la) if not (r < len(pad) and pad[r]) for e in row if isinstance(e, int) and 0 <= e < 384)
            rec[int(d["physical_rows"])][len(es)] += 1


def load(p):
    M = None
    out = []
    for line in gzip.open(p, "rt"):
        f = line.rstrip("\n").split("|", 8)
        if len(f) < 9:
            continue
        mm = re.search(r"skinny_mx_kernel<\(int\)(\d+)", f[8])
        if mm:
            M = int(mm.group(1))
        if "1d1d_impl<(unsigned int)0, (unsigned int)576" in f[8]:
            out.append((M, (int(f[1]) - int(f[0])) / 1e3))
    return out


def post(M, d, flat=False, sigma=5.0):
    nmb = M * 6
    Gs = range(6, min(6 * M, 60) + 1)
    c = rec.get(M, {})
    n = sum(c.values()) if c else 0
    w = {}
    for G in Gs:
        prior = 1.0 if flat or not c else (c.get(G, 0) + 0.5) / (n + 0.5 * len(Gs))
        w[G] = prior * math.exp(-((d - Tlive(G, nmb)) ** 2) / (2 * sigma ** 2))
    s = sum(w.values())
    if s <= 0:
        G = min(Gs, key=lambda g: abs(Tlive(g, nmb) - d))
        return {G: 1.0}
    return {G: v / s for G, v in w.items()}


PRIOR = {}
print("G  live_M2 live_M4 live_M6 deal n lastround")
for G in list(range(6, 37)):
    K = G * NN
    print(G, " ".join("%6.1f" % (Tlive(G, nm) if G <= nm else float("nan")) for nm in (12, 24, 36)),
          "%6.1f" % Tdeal(G), active(K), len(range(0, 48)) and sum(1 for p in per_deal(G) if p == max(per_deal(G))))
print()
for tr in sys.argv[2].split(","):
    calls = load(tr)
    nst = len(calls) / 40
    Ms = collections.Counter(M for M, _ in calls)
    for flat in (False, True):
        s = collections.Counter()
        gh = collections.Counter()
        gm = collections.defaultdict(collections.Counter)
        n_used = 0
        for M, d in calls:
            if M is None:
                continue
            n_used += 1
            for G, w in post(M, d, flat).items():
                t = Tlive(G, M * 6)
                gh[G] += w
                gm[M][G] += w
                s["deal"] += w * (t - Tdeal(G))
                s["deal+0.7us"] += w * (t - Tdeal(G) - 0.7)
                s["deal+0.7us,C(n)"] += w * (t - Tdeal(G, cfn=closs) - 0.7)
                s["deal,F370"] += w * (t - Tdeal(G, floor=370.0))
                s["rider F370 on live"] += w * (t - Tlive(G, M * 6, floor=370.0))
        tot = sum(gh.values())
        print("==", tr.split("/")[-1], "recorded-prior" if not flat else "flat-prior", "steps %.0f FC1 %.3f ms/step Ms %s meanG %.1f"
              % (nst, sum(d for _, d in calls) / nst / 1e3, dict(Ms), sum(k * v for k, v in gh.items()) / tot))
        print("   saving ms/step:", " ".join("%s %.3f" % (a, v / nst / 1e3) for a, v in s.items()),
              "| us/call deal+0.7 %.2f" % (s["deal+0.7us"] / n_used))
        if not flat:
            PRIOR[tr.split("/")[-1]] = {str(M): {str(G): round(w, 3) for G, w in c.items() if w >= 0.5} for M, c in gm.items()}
if len(sys.argv) > 3 and "PRIOR_OUT" in __import__("os").environ:
    p = PRIOR
    names = sorted(p)
    json.dump(dict(code=p[names[0]], prose=p[names[1]], g4=p[names[2]], source="fluid inversion of prof4 q0.1/q0.2/q0.3, recorded-route prior"),
              open(__import__("os").environ["PRIOR_OUT"], "w"), indent=0, sort_keys=True)
