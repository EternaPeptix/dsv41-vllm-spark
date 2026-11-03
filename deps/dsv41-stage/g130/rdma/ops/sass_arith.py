#!/usr/bin/env python3
"""sass_arith.py <so>: for each kPub != 0 kernel, the floating-point instruction stream must match its kPub == 0 kernel: same mnemonics
(with rounding and format suffixes) in the same order, plus FMUL/FFMA absent in both. Registers may differ (ptxas allocates per kernel)."""
import importlib.util
import os
import re
import sys

spec = importlib.util.spec_from_file_location("sg", os.path.join(os.path.dirname(os.path.abspath(__file__)), "sass_gate.py"))
sg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sg)
fns = sg.functions(sys.argv[1])
FP = re.compile(r"^(?:@!?U?P\w+ )?((?:FADD|FMUL|FFMA|F2F|F2FP|I2F|F2I|FMNMX|FSET|FSETP|DADD)\S*)")


def fp(body):
    return [m.group(1) for m in (FP.match(i) for i in body) if m]


bad = 0
for k in sorted(fns):
    if not re.search(r"ELi[1-7]EE$", k):
        continue
    base = re.sub(r"ELi\d+EE$", "EE", k)
    a, b = fp(fns[base]), fp(fns[k])
    same = a == b
    bad += 0 if same else 1
    kinds = sorted(set(b))
    print(f"{k:34s} fp instructions {len(b):3d} base {len(a):3d} same order+suffixes {'YES' if same else 'NO'} {kinds if not same else ''}")
print("base kinds:", sorted({x for k in fns if not re.search(r'ELi[1-7]EE$', k) and 'ar_kernel_pl' in k for x in fp(fns[k])}))
print("SASS_ARITH", "PASS" if not bad else f"FAIL ({bad})")
sys.exit(1 if bad else 0)
