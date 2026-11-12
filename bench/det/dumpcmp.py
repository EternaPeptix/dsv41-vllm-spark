#!/usr/bin/env python3
"""dumpcmp.py <runA dir> [<runB dir> ...]   (ON spark-one, venv python; CPU only)

Each dir holds one request's g5det dump files (r0_<seq>_<kind>.pt, one per indexer call, in order).
Per run: tie statistics (rows whose k-th logit value is shared by more candidates than fit = boundary
ties; their k-th values). Run pairs (A vs each other run): calls aligned by order; per call, how many
rows differ in the attention input (hfp), index query (qfp), head weights (wfp), gathered index keys
(kfp/ksfp), logits (lfp), the stock top-k SET, the stock ORDER and the final (fixed) set -> where the
first difference appears and what kind it is.
"""
import collections
import glob
import os
import sys

import torch


def load(d):
    fs = sorted(glob.glob(os.path.join(d, "r0_*.pt")), key=lambda p: int(os.path.basename(p).split("_")[1]))
    return [torch.load(f, weights_only=False) for f in fs]


def sets(t):
    t = t.clone()
    t[t < 0] = 2**31 - 1
    return torch.sort(t, dim=1).values


def rowdiff(a, b):
    if a is None or b is None:
        return None
    if isinstance(a, int) or isinstance(b, int):
        return int(a != b)
    if a.shape != b.shape:
        return "shape %s/%s" % (tuple(a.shape), tuple(b.shape))
    if a.dim() == 1:
        return int((a != b).sum())
    return int((a != b).any(dim=1).sum())


def tie_stats(run, label):
    tot = collections.Counter()
    vks = collections.Counter()
    per_kind_layer = collections.Counter()
    for r in run:
        if "fixed" not in r:
            continue
        act = r["ngt"] >= 0
        fx = r["fixed"] == 1
        tot[(r["kind"], "calls")] += 1
        tot[(r["kind"], "rows")] += r["rows"]
        tot[(r["kind"], "active")] += int(act.sum())
        tot[(r["kind"], "tied")] += int(fx.sum())
        if fx.any():
            per_kind_layer[(r["kind"], r["call"], r["rows"], r["width"])] += int(fx.sum())
            for v, g, q in zip(r["vk"][fx].tolist(), r["ngt"][fx].tolist(), r["neq"][fx].tolist()):
                key = "0.0" if v == 0.0 else ("-inf" if v == float("-inf") else "other")
                vks[key] += 1
    print("[%s] calls/rows/active/tied: prefill %d/%d/%d/%d  decode %d/%d/%d/%d | tied k-th values: %s"
          % (label, tot[("prefill", "calls")], tot[("prefill", "rows")], tot[("prefill", "active")],
             tot[("prefill", "tied")], tot[("decode", "calls")], tot[("decode", "rows")],
             tot[("decode", "active")], tot[("decode", "tied")], dict(vks)))
    if per_kind_layer:
        print("   tied rows by (kind, call#, rows, width): %s" % dict(list(per_kind_layer.items())[:24]))
    return tot, vks


def example_ties(run, n=3):
    shown = 0
    for r in run:
        if "fixed" not in r or not (r["fixed"] == 1).any():
            continue
        idx = torch.nonzero(r["fixed"] == 1).flatten()[:2]
        for i in idx.tolist():
            print("   e.g. %s call %d row %d/%d: k-th value %r, #greater %d, #equal %d, row length %d"
                  % (r["kind"], r["call"], i, r["rows"], r["vk"][i].item(), r["ngt"][i].item(), r["neq"][i].item(),
                     int(r["ends"][i] - (r["starts"][i] if r["starts"] is not None else 0))))
            shown += 1
            if shown >= n:
                return


def compare(A, B, la, lb):
    m = min(len(A), len(B))
    print("== %s vs %s: %d / %d calls" % (la, lb, len(A), len(B)))
    first = None
    lines = 0
    for i in range(m):
        a, b = A[i], B[i]
        if (a["kind"], a["rows"], a["width"]) != (b["kind"], b["rows"], b["width"]):
            print("   call %d: structure differs (%s %d x %d vs %s %d x %d) -> stop"
                  % (i, a["kind"], a["rows"], a["width"], b["kind"], b["rows"], b["width"]))
            break
        d = collections.OrderedDict()
        for k in ("hfp", "qfp", "wfp", "kfp", "ksfp", "lfp"):
            if k in a and k in b:
                d[k] = rowdiff(a[k], b[k])
        d["set"] = rowdiff(sets(a["raw"]), sets(b["raw"]))
        d["order"] = rowdiff(a["raw"], b["raw"])
        d["out_set"] = rowdiff(sets(a["out"]), sets(b["out"]))
        d["tiedA"] = int((a.get("fixed", torch.zeros(1)) == 1).sum())
        d["tiedB"] = int((b.get("fixed", torch.zeros(1)) == 1).sum())
        anyd = any(v not in (0, None) for k, v in d.items() if k not in ("order", "tiedA", "tiedB"))
        if anyd and first is None:
            first = (i, a["kind"], a["fwd"], a["call"], dict(d))
        if (anyd or d["order"]) and lines < 40:
            lines += 1
            print("   call %3d %-7s fwd %d #%d rows %5d width %6d cand %s: %s"
                  % (i, a["kind"], a["fwd"], a["call"], a["rows"], a["width"], a.get("cand"),
                     " ".join("%s=%s" % kv for kv in d.items())))
    if first is None:
        print("   no difference in %d aligned calls (order-only differences shown above, if any)" % m)
    else:
        i, kind, fwd, call, d = first
        upstream = [k for k in ("hfp", "qfp", "wfp", "kfp", "ksfp") if d.get(k) not in (0, None)]
        if upstream:
            verdict = "inputs already differ (%s): cause is upstream of this indexer call" % ",".join(upstream)
        elif d.get("lfp") not in (0, None):
            verdict = "logits differ with identical inputs: logits kernel"
        elif d.get("set"):
            verdict = ("identical logits, different stock top-k SET in %d rows (%s): top-k tie-breaking"
                       % (d["set"], "fix canonicalizes it" if not d["out_set"] else "NOT fixed"))
        else:
            verdict = "?"
        print("   FIRST DIFFERENCE: call %d (%s fwd %d #%d): %s -> %s" % (i, kind, fwd, call, d, verdict))
    return first


runs = [(os.path.basename(d.rstrip("/")), load(d)) for d in sys.argv[1:]]
for lab, r in runs:
    tie_stats(r, lab)
    example_ties(r)
for lab, r in runs[1:]:
    compare(runs[0][1], r, runs[0][0], lab)
