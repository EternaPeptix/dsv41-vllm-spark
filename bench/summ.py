import json, sys, os, glob
for d in sys.argv[1:]:
    def rows(f):
        p = os.path.join(d, f)
        if not os.path.exists(p): return []
        out = []
        for l in open(p):
            try: out.append(json.loads(l))
            except Exception: pass
        return out
    s = {}
    for r in rows("single.jsonl"):
        k = r.get("type") or r.get("kind") or r.get("ptype")
        s[k or len(s)] = r.get("decode_tok_s")
    sl = [v for v in s.values() if v]
    def agg(f):
        rr = rows(f)
        for r in rr:
            for k in ("agg_tok_s", "aggregate_tok_s", "agg_decode_tok_s", "aggregate"):
                if k in r: return r[k]
        return None
    al = None
    try:
        for l in open(os.path.join(d, "single.log")):
            if "summary_spec_delta" in l:
                j = json.loads(l)["summary_spec_delta"]; al = (j["mean_accept_len"], j["per_pos"])
    except Exception: pass
    ex = None
    try: ex = json.load(open(os.path.join(d, "exact.json")))["combined"][:12]
    except Exception: pass
    g = None
    try:
        gr = rows("gsm8k.jsonl"); g = "%d/%d" % (sum(1 for r in gr if r.get("ok")), len(gr))
    except Exception: pass
    print(os.path.basename(d), "single", s, "mean", round(sum(sl)/len(sl), 1) if sl else None,
          "conc4", agg("conc4.jsonl"), "conc8", agg("conc8.jsonl"), "AL", al, "exact", ex, "gsm", g)
