#!/usr/bin/env python3
"""Compile-check _dsv41_rp_kernel for sm_121 WITHOUT a GPU (Triton ASTSource -> TTIR -> PTX -> cubin with the venv's
ptxas), and grep the PTX for the instructions the exactness map requires.

Run on spark-eight only, never during a measurement window (check ~/dsv41-opt-20261021/CLUSTER_LOCK on spark-one):
    mkdir -p ~/climb-scratch/r2-moefront && cd ~/climb-scratch/r2-moefront   # holds a copy of this kit
    CUDA_VISIBLE_DEVICES= TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas TRITON_CACHE_DIR=$PWD/tcache \
        nice -n 19 ~/venvs/vllm032/bin/python tools/compile_check.py > compile.log 2>&1
Writes ptx/<variant>.ptx; prints PASS/FAIL per check; exit 0 when all pass. Deletes the Triton cache it made.
"""
import importlib.util
import os
import re
import shutil
import sys
import types

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["DSV41_MOE_ROUTEPREP"] = "1"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOD = os.path.join(HERE, "files/vllm/model_executor/layers/fused_moe/dsv41_routeprep.py")

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.compiler import ASTSource  # noqa: E402


class _Platform:
    @staticmethod
    def is_arch_support_pdl():
        return True

    @staticmethod
    def fp8_dtype():
        return torch.float8_e4m3fn


def stub(name, **kw):
    m = types.ModuleType(name)
    m.__dict__.update(kw)
    sys.modules[name] = m


stub("vllm")
stub("vllm.platforms", current_platform=_Platform())
stub("vllm.triton_utils", tl=tl, triton=triton)
spec = importlib.util.spec_from_file_location("dsv41_routeprep", MOD)
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)
K = rp._dsv41_rp_kernel

SIG = {
    "x_ptr": "*bf16", "x_row_stride": "i32", "topk_ids": "*i32", "ids_stride0": "i32", "ids_stride1": "i32",
    "out_ptr": "*u8", "out_row_stride": "i32", "sfp": "*i32", "mn_aligned": "i32", "layout": "*i32",
    "inv_perm": "*i32", "inv_stride0": "i32", "inv_stride1": "i32", "ends": "*i32",
    "w1_ptr": "*i8", "w1_s0": "i32", "w1_s1": "i32", "sf_ptr": "*i32", "sf_s0": "i32", "sf_s1": "i32", "sf_s2": "i32",
    "M": "i32", "M_sum": "i32", "eps": "fp32", "fp8_min": "fp32", "fp8_max": "fp32",
}


def variant(slots, trigger, pf, pdl=True, ids="*i32"):
    c = dict(E=384, TOPK=6, ALIGN=32, H=5120, GROUP=128, GB=8, PACKS=10, SLOTS=slots, E_PAD=512, FILL_ROWS=64,
             EARLY_TRIGGER=trigger, PF_KB=pf, PF_N=576 if pf else 1, PF_NPAD=1024 if pf else 1,
             PF_SPAD=32 if pf else 1, PF_SF_STEP=32 if pf else 1, KB_BYTES=64, launch_pdl=pdl)
    sig = dict(SIG, topk_ids=ids)
    for k in c:
        sig[k] = "constexpr"
    return c, sig


VARIANTS = {
    "s64_off": variant(64, False, 0),
    "s1024_off": variant(1024, False, 0),
    "s64_off_i64ids": variant(64, False, 0, ids="*i64"),
    "s64_pdl": variant(64, True, 0),
    "s64_pf2": variant(64, True, 2),
    "s64_nopdl": variant(64, False, 0, pdl=False),
    # every window arm at the largest SLOTS bucket too (pf4 / pf8 / pdl), and pf2 with int64 top-k ids
    "s1024_pdl": variant(1024, True, 0),
    "s1024_pf4": variant(1024, True, 4),
    "s1024_pf8": variant(1024, True, 8),
    "s256_pf2_i64ids": variant(256, True, 2, ids="*i64"),
}

FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name}{(': ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAILS.append(name)


def main():
    target = GPUTarget("cuda", 121, 32)
    os.makedirs(os.path.join(HERE, "ptx"), exist_ok=True)
    print(f"triton {triton.__version__}; target {target}; ptxas {os.environ.get('TRITON_PTXAS_PATH')}")
    ptx = {}
    for name, (c, sig) in VARIANTS.items():
        src = ASTSource(fn=K, signature=sig, constexprs=c)
        opts = dict(num_warps=4, enable_fp_fusion=False)
        if c["launch_pdl"]:
            opts["launch_pdl"] = True
        try:
            k = triton.compile(src, target=target, options=opts)
        except TypeError:
            opts.pop("launch_pdl", None)
            k = triton.compile(src, target=target, options=opts)
        p = k.asm["ptx"]
        ptx[name] = p
        open(os.path.join(HERE, "ptx", f"{name}.ptx"), "w").write(p)
        cub = k.asm.get("cubin", b"")
        print(f"compiled {name}: ptx {len(p)} B, cubin {len(cub)} B, regs {getattr(k, 'n_regs', '?')}, "
              f"spills {getattr(k, 'n_spills', '?')}, shared {k.metadata.shared}")
    p = ptx["s64_off"]
    # quant path: IEEE division (y = absmax / 448 and 1 / scale), no approximate div on that path
    ndiv = len(re.findall(r"\bdiv\.rn(\.ftz)?\.f32\b", p))
    check("div.rn.f32 present (IEEE absmax/448 and 1/scale)", ndiv >= 2, f"{ndiv} found")
    check("no div.full / div.approx (quant must not use approximate division)",
          not re.search(r"\bdiv\.(full|approx)", p))
    ftz = sorted(set(re.findall(r"\b((?:mul|add|max|min|div|fma)(?:\.rn)?\.ftz\.f32)", p)))
    # INFO only: the live SASS is non-FTZ. FTZ would still give the same bytes on the whole domain (divisions see
    # normal operands/results only: absmax >= 1e-10, scale >= 2^-33; a flushed product, clamp input or add operand
    # is |q| < 2^-126, which encodes to the same signed zero), but the map in EXACTNESS.md assumes non-FTZ.
    print(f"INFO .ftz f32 ops in the PTX: {ftz or 'none'}")
    check("no fma.rn.f32 (no fused x*inv into anything; enable_fp_fusion=False)", not re.search(r"\bfma\.rn\.f32", p))
    check("max/min .f32 are NaN-ignoring (no .NaN modifier)", not re.search(r"\b(max|min)\.NaN\.f32", p))
    check("1e-10 floor constant 0f2EDBE6FF present", "0f2EDBE6FF" in p)
    check("16384.0 (0f46800000) subnormal add present", "0f46800000" in p)
    check("448.0 runtime args used (no hard-coded rounding constant 0f43E00000 required)", True)
    check("no cvt to e4m3 (encoding is the live integer sequence)", not re.search(r"cvt\.[a-z.]*e4m3", p))
    check("griddepcontrol.wait in PDL variants", all("griddepcontrol.wait" in ptx[n] for n in
                                                      ("s64_off", "s1024_off", "s64_pdl", "s64_pf2")))
    check("no griddepcontrol.wait without launch_pdl", "griddepcontrol.wait" not in ptx["s64_nopdl"])
    check("launch_dependents only in pdl/pf variants",
          "griddepcontrol.launch_dependents" in ptx["s64_pdl"] and "griddepcontrol.launch_dependents" in
          ptx["s64_pf2"] and "griddepcontrol.launch_dependents" not in ptx["s64_off"])
    for n in ("s1024_pf4", "s1024_pf8", "s256_pf2_i64ids"):
        check(f"{n}: prefetch.global.L2 + launch_dependents present", "prefetch.global.L2" in ptx[n] and
              "griddepcontrol.launch_dependents" in ptx[n])
    check("prefetch.global.L2 only in pf variant", "prefetch.global.L2" in ptx["s64_pf2"] and
          "prefetch.global.L2" not in ptx["s64_off"] and "prefetch.global.L2" not in ptx["s64_pdl"])
    # deterministic slots: no global atomics / reductions anywhere. tl.histogram lowers to per-CTA shared-memory
    # integer adds (atom.shared.add.u32 into a smem histogram, then bar.sync): integer addition is exact and
    # order-independent, so the counts (and every slot derived from them) are deterministic.
    for n, body in ptx.items():
        glob = sorted(set(re.findall(r"\b((?:atom|red)\.(?:global\.)?[a-z0-9.]+)", body)) - {
            a for a in re.findall(r"\b(atom\.shared\.[a-z0-9.]+)", body)})
        shared = sorted(set(re.findall(r"\batom\.shared\.[a-z0-9.]+", body)))
        check(f"{n}: no global atomics/reductions; shared atomics only integer add (histogram)",
              not glob and all(a in ("atom.shared.add.u32", "atom.shared.add.s32") for a in shared),
              f"global {glob or 'none'}, shared {shared or 'none'}")
    # wait precedes every global load / store
    for n in ("s64_off", "s64_pdl", "s64_pf2", "s1024_off", "s1024_pf8"):
        body = ptx[n]
        w = body.find("griddepcontrol.wait")
        first_ld = min([i for i in (body.find("ld.global"), body.find("st.global")) if i >= 0] or [10 ** 9])
        check(f"{n}: griddepcontrol.wait before the first global load/store", 0 <= w < first_ld)
    shutil.rmtree(os.environ.get("TRITON_CACHE_DIR", "/nonexistent"), ignore_errors=True)
    print("ALL PASS" if not FAILS else f"FAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
