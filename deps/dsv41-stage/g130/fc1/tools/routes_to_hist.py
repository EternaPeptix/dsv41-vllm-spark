#!/usr/bin/env python3
"""g130 fc1: turn a route-recorder dump into per-(tokens, G) weights.  CPU only, no numpy needed.
  python3 -I routes_to_hist.py fc1-routes.<host>.<pid>.bin [--from N] [--to M] [--out hist.json]
Output: {"records": n, "by_tokens": {tokens: {G: count}}, "prefix_violations": v, "over_rec": k}
tools/eval_probe.py takes it as --hist. A prefix violation (a live m-block after a dead one) is a stop: the deal's search rests on
that contract and the recorder's own record would then be wrong too."""
import array
import json
import sys


def m_sum(t):
    routes = 6 * t
    rows = routes + min(routes, 384) * 31
    return -(-rows // 32) * 32


NMB_TO_T = {}
for t in range(1, 97):
    NMB_TO_T.setdefault(m_sum(t) // 32, t)


def read(path):
    with open(path, "rb") as f:
        h = array.array("i")
        h.frombytes(f.read(16))
        REC, CAP, total = h[0], h[1], h[2]
        v = array.array("i")
        v.frombytes(f.read())
    n = min(total, CAP)
    return REC, n, total, v


def hist(path, lo=0, hi=None):
    REC, n, total, v = read(path)
    by = {}
    viol = over = 0
    for i in range(lo, n if hi is None else min(n, hi)):
        base = i * REC
        nmb, G = v[base], v[base + 1]
        cap = (REC - 2) // 2
        if nmb > cap and G >= cap:
            over += 1
        dead = False
        g2 = 0
        for m in range(min(nmb, cap)):
            live = v[base + 2 + 2 * m] >= 0
            if live and dead:
                viol += 1
                break
            if not live:
                dead = True
            else:
                g2 += 1
        t = NMB_TO_T.get(nmb, -nmb)
        d = by.setdefault(t, {})
        d[G] = d.get(G, 0) + 1
    return dict(records=n, total=total, wrapped=total > n, by_tokens=by, prefix_violations=viol, over_rec=over)


if __name__ == "__main__":
    lo = int(sys.argv[sys.argv.index("--from") + 1]) if "--from" in sys.argv else 0
    hi = int(sys.argv[sys.argv.index("--to") + 1]) if "--to" in sys.argv else None
    r = hist(sys.argv[1], lo, hi)
    if "--out" in sys.argv:
        with open(sys.argv[sys.argv.index("--out") + 1], "w") as f:
            json.dump(r, f, indent=1, sort_keys=True)
    print(json.dumps({k: v for k, v in r.items() if k != "by_tokens"}))
    for t in sorted(r["by_tokens"]):
        c = r["by_tokens"][t]
        tot = sum(c.values())
        print("tokens %s calls %d meanG %.1f :" % (t, tot, sum(g * n for g, n in c.items()) / tot),
              " ".join("%d:%.0f" % (g, 100 * n / tot) for g, n in sorted(c.items()) if n / tot >= 0.01))
    if r["prefix_violations"]:
        print("STOP: prefix contract violated", r["prefix_violations"], "records")
        sys.exit(3)
