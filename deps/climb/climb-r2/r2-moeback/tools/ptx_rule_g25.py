#!/usr/bin/env python3
"""G25 PTX acceptance rule (pure Python, no Triton): used by ptx_check_g25.py and tests/test_ptx_rule.py.

census(ptx) counts the fp instructions; verdict(c_live, c_add, ptx_live, ptx_add) returns a list of failures
(empty = PASS). Per thread (BLOCK_D=1024, 2 warps) the live _gated_gather holds E = 16 output elements:
48 fma.rn.f32x2 (6 experts x 16 / 2) and one bf16 rounding per element (8 cvt.rn.bf16x2.f32).
_gated_gather_add must keep that accumulation and rounding unchanged and add EXACTLY one of the two epilogues
proven bit-identical to torch's CUDAFunctor_add<bf16> by tests/emul_add_exhaustive.py:
  A  f32(bf16(acc)) + f32(shared) in fp32, then one more RN to bf16:
     extra bf16 roundings == E elements (total 2E: 16 cvt.rn.bf16x2.f32), extra fp32 add elements == E,
     no bf16 adds;
  B  native bf16 add of bf16(acc) and shared:
     extra bf16 roundings == 0 (total E), bf16 add elements == E, no extra fp32 adds.
Anything else fails, in particular the known-wrong lowering that folds shared into the fp32 accumulator
before the single rounding (E roundings + E fp32 adds, or E roundings + an initialised accumulator).
"""
import re

OPS = r"(fma\.[a-z0-9.]+|add\.[a-z0-9.]*(?:f32x2|f32|bf16x2|bf16)\b|mul\.[a-z0-9.]*f32|cvt\.[a-z0-9.]+|div\.[a-z0-9.]+)"
ALLOWED_EXTRA = {"cvt.rn.bf16x2.f32", "cvt.rn.bf16.f32", "cvt.f32.bf16", "add.rn.f32", "add.f32", "add.rn.f32x2",
                 "add.f32x2", "add.rn.bf16x2", "add.bf16x2", "add.rn.bf16", "add.bf16"}


def census(ptx):
    c = {}
    for m in re.findall(OPS, ptx):
        c[m] = c.get(m, 0) + 1
    return c


def _n(c, k):
    return c.get(k, 0)


def rounding_elems(c):
    return 2 * _n(c, "cvt.rn.bf16x2.f32") + _n(c, "cvt.rn.bf16.f32")


def f32_add_elems(c):
    return _n(c, "add.f32") + _n(c, "add.rn.f32") + 2 * (_n(c, "add.f32x2") + _n(c, "add.rn.f32x2"))


def bf16_add_elems(c):
    return _n(c, "add.bf16") + _n(c, "add.rn.bf16") + 2 * (_n(c, "add.bf16x2") + _n(c, "add.rn.bf16x2"))


def verdict(c_live, c_add, ptx_live="", ptx_add=""):
    fails = []
    for name, ptx in (("live", ptx_live), ("add", ptx_add)):
        if ".ftz" in ptx:
            fails.append(f"{name}: .ftz present")
    if _n(c_live, "fma.rn.f32x2") != 48 or _n(c_live, "cvt.rn.bf16x2.f32") != 8:
        fails.append("live lowering differs from the production cache census (48 fma.rn.f32x2, 8 cvt.rn.bf16x2.f32)")
    if _n(c_add, "fma.rn.f32x2") != _n(c_live, "fma.rn.f32x2"):
        fails.append("accumulation fma count differs")
    for k in set(c_add) | set(c_live):
        if k.startswith("fma.") and _n(c_add, k) != _n(c_live, k):
            fails.append(f"fma census differs: {k} live {_n(c_live, k)} add {_n(c_add, k)}")
    extra = {k: _n(c_add, k) - _n(c_live, k) for k in set(c_add) | set(c_live) if _n(c_add, k) != _n(c_live, k)}
    bad = [k for k in extra if k not in ALLOWED_EXTRA and not k.startswith(("cvt.u", "cvt.s"))]
    if bad:
        fails.append(f"unexpected fp ops in the fold kernel: {sorted(bad)}")
    if any(v < 0 for k, v in extra.items() if k in ALLOWED_EXTRA):
        fails.append(f"fold kernel has fewer fp ops than live: { {k: v for k, v in extra.items() if v < 0} }")
    e = rounding_elems(c_live)                       # 16 elements per thread
    r_extra = rounding_elems(c_add) - e
    f_extra = f32_add_elems(c_add) - f32_add_elems(c_live)
    b_extra = bf16_add_elems(c_add) - bf16_add_elems(c_live)
    path_a = r_extra == e and f_extra == e and b_extra == 0
    path_b = r_extra == 0 and b_extra == e and f_extra == 0
    if not (path_a or path_b):
        fails.append(f"epilogue is neither A (f32 add + 2nd RN) nor B (native bf16 add): per thread "
                     f"{e} elements, extra roundings {r_extra}, extra f32-add elements {f_extra}, "
                     f"bf16-add elements {b_extra}")
    return fails, {"elements": e, "extra_roundings": r_extra, "extra_f32_add": f_extra, "bf16_add": b_extra,
                   "path": "A" if path_a else "B" if path_b else "none", "extra": extra}
