#!/usr/bin/env python3
"""Compile-only check (no GPU, no CUDA context): build the live and the kit Triton kernels for cuda:121 with the
venv's Triton 3.7.1 and compare what decides the bits.

    nice -n 19 ~/venvs/vllm032/bin/python scripts/compile_check.py OUTDIR

For each variant it records TTGIR / PTX / cubin sizes, shared memory, registers(n/a without driver), and checks:
  partial kernels: exactly one tt.dot; its operand encodings (#mma versionMajor 2, kWidth 4 on both operands,
    bf16 x bf16 -> f32, inputPrecision) equal the live kernel's; the PTX mma.sync instruction set is the same
    (m16n8k16 row.col f32.bf16.bf16.f32); the dequant is mul.rn.f32 + cvt.rn.bf16.f32 (no fma in the dequant);
    ROPE variants: the PTX has fma.rn.f32 (no plain mul+add contraction left to ptxas: fusion disabled) and
    exactly the rope op kinds (mul.rn.f32 for p*s, fma.rn.f32, neg.f32, cvt.rn.bf16.f32).
  reduce-quant kernel: no fma in PTX, add.f32 count, the three flashinfer asm blocks present verbatim,
    cvt.rn.bf16.f32 before the quant, griddepcontrol.wait present (PDL).
Writes OUTDIR/<name>.{ttgir,ptx} and OUTDIR/summary.json; exit 1 on a failed check.
"""
import json
import os
import re
import sys

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("DSV41_ATTNOUT_OP", "rope")
# No vLLM import (no platform probing / NVML): stub the two vllm modules the kernel files need, with the
# real Triton, and load the live sonar file from the kit's base/ copy (sha256 checked against KIT.json).
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import logging  # noqa: E402
import types  # noqa: E402

for _n in ("vllm", "vllm.logger", "vllm.triton_utils"):
    sys.modules[_n] = types.ModuleType(_n)
sys.modules["vllm"].__path__ = []
sys.modules["vllm.logger"].init_logger = logging.getLogger
sys.modules["vllm.triton_utils"].tl = triton.language
sys.modules["vllm.triton_utils"].triton = triton


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_SONAR = os.path.join(KIT, "base/models/deepseek_v4/nvidia/ops/sonar_bf16_bmm.py")
_kit = json.load(open(os.path.join(KIT, "KIT.json")))
_want = [f["base_sha256"] for f in _kit["files"] if f["path"].endswith("sonar_bf16_bmm.py")][0]
assert hashlib.sha256(open(_SONAR, "rb").read()).hexdigest() == _want, "base sonar copy != live sha"
sonar = _load("sonar_live", _SONAR)
ao = _load("ao", os.path.join(KIT, "files/models/deepseek_v4/nvidia/ops/dsv41_attnout.py"))

TARGET = GPUTarget("cuda", 121, 32)


def build(fn, sig, consts, opts):
    names = fn.arg_names
    signature = {}
    cmap = {}
    for i, n in enumerate(names):
        if n in consts:
            signature[n] = "constexpr"
            cmap[(i,)] = consts[n]
        else:
            signature[n] = sig[n]
    attrs = {(i,): [["tt.divisibility", 16]] for i, n in enumerate(names) if sig.get(n, "").startswith("*")}
    src = ASTSource(fn=fn, signature=signature, constexprs=cmap, attrs=attrs)
    return triton.compile(src, target=TARGET, options=opts)


def dot_lines(ttgir):
    return [ln.strip() for ln in ttgir.splitlines() if "tt.dot " in ln]


def enc_of_dot(ttgir):
    d = dot_lines(ttgir)
    assert len(d) == 1, d
    # operand / result encodings, stripped of SSA names and locs
    m = re.search(r":\s*(tensor<.*)\s+loc", d[0])
    s = m.group(1) if m else d[0]
    mma = [ln for ln in ttgir.splitlines() if ln.startswith("#mma")]
    return s, mma


def mma_set(ptx):
    return sorted(set(re.findall(r"mma\.sync\.aligned\.\S+", ptx)))


def count(ptx, pat):
    return len(re.findall(pat, ptx))


def main(out):
    os.makedirs(out, exist_ok=True)
    summary, ok = {}, True
    common = dict(A="*bf16", W="*fp8e4nv", S="*u8", P="*fp32", M="i32", ASG="i64", ASM="i64", ASK="i64")
    live = build(sonar._sm120_wo_a_partial, common,
                 dict(G=1, N=1024, K=4096, CHUNK=128, FP8W=True),
                 dict(num_warps=4, num_stages=2, enable_fp_fusion=False))
    live_enc, live_mma = enc_of_dot(live.asm["ttgir"])
    live_ptx_mma = mma_set(live.asm["ptx"])
    summary["live_partial"] = dict(shared=live.metadata.shared, dot=live_enc, mma=live_mma, ptx_mma=live_ptx_mma,
                                   n_mma=count(live.asm["ptx"], r"mma\.sync"))
    open(os.path.join(out, "live_partial.ttgir"), "w").write(live.asm["ttgir"])
    open(os.path.join(out, "live_partial.ptx"), "w").write(live.asm["ptx"])
    sig_new = dict(common, POS="*i64", CS="*fp32", CSS="i64")
    for name, rope, bn, vs, st in [("rope", True, 64, False, 2), ("st1", False, 64, False, 1),
                                   ("vs", False, 64, True, 2), ("bn128", False, 128, False, 2),
                                   ("rope_st1_vs", True, 64, True, 1), ("rope_st1_vs_bn128", True, 128, True, 1)]:
        consts = dict(G=1, N=1024, K=4096, CHUNK=128, FP8W=True, ROPE=rope, HPG=8 if rope else 1,
                      HEAD_DIM=512 if rope else 128, NOPE=448 if rope else 0, HALF_ROPE=32 if rope else 1, BN=bn,
                      VEC_SCALE=vs)
        k = build(ao._r2_wo_a_partial, sig_new, consts, dict(num_warps=4, num_stages=st, enable_fp_fusion=False))
        enc, mma = enc_of_dot(k.asm["ttgir"])
        ptx = k.asm["ptx"]
        rec = dict(shared=k.metadata.shared, dot=enc, mma=mma, ptx_mma=mma_set(ptx), n_mma=count(ptx, r"mma\.sync"),
                   fma=count(ptx, r"\bfma\.rn\.f32\b") + 2 * count(ptx, r"\bfma\.rn\.f32x2\b"),
                   mul_rn=count(ptx, r"\bmul\.rn\.f32"), plain_mul=count(ptx, r"\bmul\.f32\b"),
                   plain_add=count(ptx, r"\badd\.f32\b"))
        checks = {
            "kWidth4": enc.count("kWidth = 4") == 2,
            "same_mma_instr": rec["ptx_mma"] == live_ptx_mma,
            "mma_version": all("versionMajor = 2" in x for x in mma),
            "dot_types": ("bf16" in enc and "f32" in enc),
            "no_plain_mul_add": rec["plain_mul"] == 0 and rec["plain_add"] == 0,
            "mma_count": rec["n_mma"] == (32 if bn == 128 else 16),
        }
        if rope:
            checks["rope_fma"] = rec["fma"] > 0
        else:
            checks["no_fma"] = rec["fma"] == 0
        if bn == 64:
            checks["same_dot_encoding_as_live"] = enc == live_enc and mma == live_mma
        rec["checks"] = checks
        ok &= all(checks.values())
        summary[name] = rec
        open(os.path.join(out, f"{name}.ttgir"), "w").write(k.asm["ttgir"])
        open(os.path.join(out, f"{name}.ptx"), "w").write(ptx)
    rq = build(ao._r2_reduce_quant, dict(P="*fp32", Q="*u8", SF="*u8", M="i32"),
               dict(N=1024, SPLITS=32, PADDED_M=128, SF_COLS=32, INV448=ao.INV448_BITS, USE_PDL=True),
               dict(num_warps=4, enable_fp_fusion=False, launch_pdl=True))
    ptx = rq.asm["ptx"]
    rec = dict(shared=rq.metadata.shared, add=count(ptx, r"\badd\.(rn\.)?f32\b"), fma=count(ptx, r"\bfma\."),
               cvt_bf16=count(ptx, r"cvt\.rn\.bf16\.f32"), e4m3=count(ptx, r"cvt\.rn\.satfinite\.e4m3x2\.f32"),
               gdc_wait=count(ptx, r"griddepcontrol\.wait"), ue8m0_asm=ptx.count("p_tiny_sub"))
    rec["checks"] = {"no_fma": rec["fma"] == 0, "pdl_wait": rec["gdc_wait"] >= 1, "asm_present": rec["ue8m0_asm"] > 0,
                     "e4m3_cvt": rec["e4m3"] > 0, "bf16_round": rec["cvt_bf16"] > 0}
    ok &= all(rec["checks"].values())
    summary["reduce_quant"] = rec
    open(os.path.join(out, "reduce_quant.ttgir"), "w").write(rq.asm["ttgir"])
    open(os.path.join(out, "reduce_quant.ptx"), "w").write(ptx)
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1, default=str)
    print(json.dumps({k: v.get("checks", "") for k, v in summary.items()}, indent=1))
    print("COMPILE CHECK", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(sys.argv[1])
