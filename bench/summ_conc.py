import json, sys, os
def rows(d, f):
    p = os.path.join(d, f)
    if not os.path.exists(p): return []
    out = []
    for l in open(p):
        try: out.append(json.loads(l))
        except Exception: pass
    return out
def agg(d, f):
    for r in rows(d, f):
        for k in ("agg_tok_s", "aggregate_tok_s", "agg_decode_tok_s", "aggregate"):
            if k in r: return r[k]
for d in sys.argv[1:]:
    s = {}
    for r in rows(d, "single.jsonl"):
        k = r.get("type") or r.get("kind") or r.get("ptype")
        s[k or len(s)] = r.get("decode_tok_s")
    sl = [v for v in s.values() if v]
    ex = None
    try: ex = json.load(open(os.path.join(d, "exact.json")))["combined"][:12]
    except Exception: pass
    gr = rows(d, "gsm8k.jsonl"); g = "%d/%d" % (sum(1 for r in gr if r.get("ok")), len(gr)) if gr else None
    mem = None
    try:
        v = [int(l.split()[2]) for l in open(os.path.join(d, "memavail.txt")) if len(l.split()) > 2 and l.split()[2].isdigit()]
        mem = round(min(v) / 1048576, 1)
    except Exception: pass
    print(os.path.basename(d), "single", s, "mean", round(sum(sl)/len(sl), 1) if sl else None,
          *[x for c in (4, 8, 12, 16) for x in ("c%d" % c, agg(d, "conc%d.jsonl" % c))],
          "exact", ex, "gsm", g, "minMemAvailGiB", mem)
