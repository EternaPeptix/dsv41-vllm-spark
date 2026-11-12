#!/usr/bin/env python3
"""g130 fc1 SASS gate (CPU): deal arm vs the live 'all' FC1.  python3 -I sass_gate.py cc/live.fc1.sass cc/deal.fc1.sass
Stops at the first failure. PASS means only that the per-stage math loop and the TMA/store instruction mix did not change:
  - QMMA count 8, and the QMMA lines (opcode + operands) identical up to register renaming, in the same order
  - opcode sequence from the stage barrier before the first QMMA to the first global load after the last QMMA is identical
  - UTMALDG, UTMAPF, LDGSTS, STG, QMMA and LDS.U8 counts unchanged
  - REG / LDL/STL not larger than live (checked by build-deal.sh output)
It says nothing about values; only the GPU test can."""
import collections
import re
import sys


def parse(p):
    out = []
    for raw in open(p):
        s = re.sub(r"/\*.*?\*/", "", raw).strip().strip("{};").strip()
        s = re.sub(r"^@!?U?P\w+\s+", "", s)
        m = re.match(r"([A-Z][A-Za-z0-9_.]*)", s)
        if m:
            out.append((m.group(1), s))
    return out


def rename(line):
    table = {}

    def f(mm):
        return table.setdefault(mm.group(0), "r%d" % len(table))
    return re.sub(r"\bU?R\d+\b", f, line)


def counts(o):
    c = collections.Counter(a.split(".")[0] for a, _ in o)
    c["LDS.U8"] = sum(1 for a, _ in o if a.startswith("LDS.U8"))
    return c


def window(o):
    q = [i for i, (a, _) in enumerate(o) if a.startswith("QMMA")]
    assert len(q) == 8, q
    i0 = q[0]
    while i0 > 0 and not o[i0][0].startswith("SYNCS"):
        i0 -= 1
    i1 = q[-1]
    while i1 < len(o) - 1 and not o[i1][0].startswith(("LDG", "STG")):  # the epilogue starts here; its loop-back holds the walker
        i1 += 1
    return o[i0:i1 + 1], q


live, cand = parse(sys.argv[1]), parse(sys.argv[2])
cl, cc = counts(live), counts(cand)
fail = 0
for k in ("QMMA", "UTMALDG", "UTMAPF", "LDGSTS", "STG", "LDS.U8", "ARRIVES", "UTMASTG"):
    ok = cl[k] == cc[k]
    print("count %-8s live %4d cand %4d %s" % (k, cl[k], cc[k], "ok" if ok else "FAIL"))
    fail += not ok
wl, ql = window(live)
wc, qc = window(cand)
ops_same = [a for a, _ in wl] == [a for a, _ in wc]
print("math window opcodes: live %d cand %d identical=%s" % (len(wl), len(wc), ops_same))
fail += not ops_same
ql_text = [rename(live[i][1]) for i in ql]
qc_text = [rename(cand[i][1]) for i in qc]
print("QMMA lines identical up to register names:", ql_text == qc_text)
fail += ql_text != qc_text
print("total instructions: live %d cand %d" % (len(live), len(cand)))
print("SASS GATE", "PASS" if not fail else "FAIL")
sys.exit(1 if fail else 0)
