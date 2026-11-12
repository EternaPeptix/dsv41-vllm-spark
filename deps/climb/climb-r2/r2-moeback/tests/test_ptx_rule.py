#!/usr/bin/env python3
"""Unit test of the G25 PTX acceptance rule (tools/ptx_rule_g25.py) on synthetic PTX (stdlib only).

The rule must accept both proven epilogue lowerings and reject the known-wrong one the review flagged
(shared folded into the fp32 accumulator: still 8 cvt.rn.bf16x2.f32, plus fp32 adds), its init-variant, ftz,
and stray fp ops. Run: python3 test_ptx_rule.py <kit_dir>
"""
import os
import sys

KIT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(KIT, "tools"))
from ptx_rule_g25 import census, verdict  # noqa: E402

FAILS = []


def ptx(**ops):
    lines = []
    for op, n in ops.items():
        lines += [f"\t{op.replace('_', '.')} %r1, %r2, %r3;"] * n
    return "\n".join(lines) + "\n"


# live _gated_gather per-thread census (BLOCK_D=1024, 64 threads): 6 x 16 inputs converted, 48 fma, 8 roundings
LIVE = dict(cvt_f32_bf16=96, fma_rn_f32x2=48, cvt_rn_bf16x2_f32=8)
CASES = [
    ("A: f32 add + 2nd RN (add.f32 x16)", dict(LIVE, cvt_f32_bf16=96 + 32, add_f32=16, cvt_rn_bf16x2_f32=16), True),
    ("A: f32x2 add (add.rn.f32x2 x8)", dict(LIVE, cvt_f32_bf16=96 + 32, add_rn_f32x2=8, cvt_rn_bf16x2_f32=16), True),
    ("A: scalar 2nd RN (cvt.rn.bf16.f32 x16)", dict(LIVE, add_rn_f32=16, cvt_rn_bf16_f32=16), True),
    ("B: native add.rn.bf16x2 x8", dict(LIVE, add_rn_bf16x2=8), True),
    ("B: native add.bf16 x16", dict(LIVE, add_bf16=16), True),
    ("WRONG: shared into fp32 acc (8 cvt + 16 add.f32)", dict(LIVE, cvt_f32_bf16=96 + 16, add_f32=16), False),
    ("WRONG: shared as acc init (no add)", dict(LIVE, cvt_f32_bf16=96 + 16), False),
    ("WRONG: f32 add but only half the 2nd roundings", dict(LIVE, add_f32=16, cvt_rn_bf16x2_f32=12), False),
    ("WRONG: both epilogues (f32 add + bf16 add)", dict(LIVE, add_f32=16, cvt_rn_bf16x2_f32=16, add_bf16x2=8), False),
    ("WRONG: extra fma (shared via fma)", dict(LIVE, fma_rn_f32x2=56), False),
    ("WRONG: stray mul.f32", dict(LIVE, add_f32=16, cvt_rn_bf16x2_f32=16, mul_rn_f32=16), False),
    ("WRONG: scalar fma for the add", dict(LIVE, fma_rn_f32=16, cvt_rn_bf16x2_f32=16), False),
]


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'} {name}{(' ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


p_live = ptx(**LIVE)
c_live = census(p_live)
check("live census parsed", c_live == {"cvt.f32.bf16": 96, "fma.rn.f32x2": 48, "cvt.rn.bf16x2.f32": 8}, str(c_live))
f, info = verdict(c_live, c_live, p_live, p_live)
check("identity (no epilogue) rejected", bool(f), info["path"])
for name, ops, ok in CASES:
    p_add = ptx(**ops)
    f, info = verdict(c_live, census(p_add), p_live, p_add)
    check(f"{'accept' if ok else 'reject'} {name}", (not f) == ok, f"path={info['path']} fails={f}")
p_add = ptx(**dict(LIVE, add_f32=16, cvt_rn_bf16x2_f32=16)).replace("add.f32", "add.ftz.f32")
f, info = verdict(c_live, census(p_add), p_live, p_add)
check("reject A with .ftz", bool(f) and any("ftz" in x for x in f), str(f))
p_bad_live = ptx(**dict(LIVE, fma_rn_f32x2=40))
f, _ = verdict(census(p_bad_live), census(p_bad_live), p_bad_live, p_bad_live)
check("reject a live census that is not production's", any("production cache census" in x for x in f))
print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
