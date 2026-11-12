"""CPU check of device_test.py's input builders: Q1/Q2 rows cover what they claim, and the CPU emulation of the
live vs fused quant agrees on them (so a device mismatch would be a lowering difference, not an input artefact).
Run: python3 tests/test_device_inputs.py"""
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from emul import live_quant, new_quant  # noqa: E402

spec = importlib.util.spec_from_file_location("dt", os.path.join(os.path.dirname(HERE), "device_test.py"))
dt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dt)
np.seterr(all="ignore")
fails = 0
rows, nsc = dt.q1_rows()
g = rows.reshape(-1, 128)
bl, sl = live_quant(g)
bn, sn = new_quant(g)
ok = np.array_equal(bl, bn) and np.array_equal(sl, sn)
cover = {}
for grp, s in zip(g, sl):
    cover.setdefault(int(s), set()).update(int(v) for v in grp[1:])
fin = dt.all_finite_bf16_bits()
full = sum(1 for s in cover if len(cover[s]) >= 1)
print(f"{'PASS' if ok else 'FAIL'} Q1: {rows.shape[0]} rows, {nsc} scales, {g.shape[0]} groups, emulation live == fused")
fails += not ok
ok = len(cover) == nsc
print(f"{'PASS' if ok else 'FAIL'} Q1 covers every reachable scale ({len(cover)} of {nsc})")
fails += not ok
fv = (fin.astype(np.uint32) << 16).view(np.float32)
top = max(cover)  # the largest scale: every finite bf16 x must be in its sweep
ok = len(cover[top]) == fin.size
print(f"{'PASS' if ok else 'FAIL'} Q1 largest scale sweeps all {fin.size} finite bf16 x ({len(cover[top])})")
fails += not ok
rows2 = dt.q2_rows()
g2 = rows2.reshape(-1, 128)
bl, sl = live_quant(g2)
bn, sn = new_quant(g2)
ok = np.array_equal(sl, sn) and np.array_equal(bl, bn) and set(int(v) for v in g2[:, 5]) >= set(int(v) for v in fin)
print(f"{'PASS' if ok else 'FAIL'} Q2: {rows2.shape[0]} rows, every finite bf16 absmax, emulation live == fused")
fails += not ok
for M in (1, 6, 96):
    x = dt.hostile_x(M)
    assert x.shape == (M, 5120)
    for kind in ("same", "distinct", "alldead", "onelive", "hot"):
        ids = dt.synth_ids(M, kind)
        assert ids.shape == (M, 6)
        assert all(len(set(r[r >= 0])) == (r >= 0).sum() for r in ids), kind  # distinct experts per token
print("PASS synthetic routings / hostile rows well-formed" if not fails else "")
print("ALL PASS" if not fails else f"FAILED {fails}")
sys.exit(1 if fails else 0)
