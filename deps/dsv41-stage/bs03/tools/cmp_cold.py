#!/usr/bin/env python3
"""cmp_cold.py <control.jsonl> <arm.jsonl>: compare two dbgrun.py files bit for bit, stdlib only.

Each row of dbgrun.py holds tok_sha and lp_sha for one (case, mode, run). Exact.py and G1 cannot see the indexer's selection
at long context, so this per-case compare across arms is the end-to-end check: every hash must match, and each file must also
show distinct_tok=1 distinct_lp=1 per case. Exit 0 = all equal, 1 = a mismatch, 2 = nothing to compare or missing rows.
"""
import json
import sys


def load(path):
    rows, summ = {}, {}
    with open(path) as f:
        for line in f:
            if not line.strip().startswith("{"):
                continue
            r = json.loads(line)
            if r.get("summary"):
                summ[(r["case"], r["mode"])] = r
            elif "tok_sha" in r:
                rows[(r["case"], r["mode"], r["run"])] = r
    return rows, summ


def main():
    a, sa = load(sys.argv[1])
    b, sb = load(sys.argv[2])
    keys = sorted(set(a) | set(b))
    bad = missing = 0
    for k in keys:
        if k not in a or k not in b:
            missing += 1
            print("MISSING  %-34s control %s arm %s" % (k, k in a, k in b))
            continue
        ra, rb = a[k], b[k]
        same = ra["tok_sha"] == rb["tok_sha"] and ra["lp_sha"] == rb["lp_sha"] and ra.get("pt") == rb.get("pt")
        print("%-8s %-34s pt %s tok %s %s lp %s %s" % ("ok" if same else "MISMATCH", k, rb.get("pt"), ra["tok_sha"], rb["tok_sha"], ra["lp_sha"], rb["lp_sha"]))
        bad += 0 if same else 1
    for name, summ in (("control", sa), ("arm", sb)):
        for k, r in sorted(summ.items()):
            if r.get("distinct_tok") != 1 or r.get("distinct_lp") != 1:
                bad += 1
                print("NOT REPEATABLE in %s: %s distinct_tok=%s distinct_lp=%s" % (name, k, r.get("distinct_tok"), r.get("distinct_lp")))
    print("COLD COMPARE %s: %d rows, %d mismatches, %d missing" % ("OK" if keys and not bad and not missing else "FAIL", len(keys), bad, missing))
    if not keys or missing:
        sys.exit(2)
    sys.exit(1 if bad else 0)


main()
