#!/usr/bin/env python3
"""r5-glue compile check WITHOUT a GPU: Triton ASTSource -> PTX -> cubin for sm_121 with the venv's ptxas, for the live
r2 kernels and the kit kernels side by side, IN PRODUCTION'S SPECIALIZATION, then PTX/SASS checks.

Run on spark-eight only, never while ~/dsv41-opt-20261021/CLUSTER_LOCK exists on spark-one (tools/run_spark8_checks.sh
does the lock polling and sources the mandatory environment):
    cd ~/climb-scratch/r5-glue && CUDA_VISIBLE_DEVICES= TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
        TRITON_CACHE_DIR=$PWD/tcache nice -n 19 ~/venvs/vllm032/bin/python tools/compile_check.py

Specialization (review B1). Triton's JIT launch specializes every non-constexpr argument: pointers whose address and
ints whose value are 16-divisible get `tt.divisibility 16` (vector loads depend on it), ints equal to 1 become
constexpr 1 (e.g. M == 1 for the RP kernel, ids_stride1 / inv_stride1 always). A bare ASTSource(signature, constexprs)
has none of this and lowers the x loads to scalar ld.global.b16, which is NOT what production runs. So this script:
  1. calls the REAL launch functions (dsv41_routeprep._launch, dsv41_attnout._launch_rq, and the kit's _launch_new /
     _launch_new_into) with CPU tensors shaped and strided exactly as fused_permute / o_proj pass them (x [M, 5120]
     bf16, topk_ids [M, 6], slot rows [M_sum, 5120], sfp (M_sum, 10) strides (1, mn_aligned), M_sum the 32-aligned
     value of the fused path; P [32, 1, M, 1024] fp32), with the kernel object replaced by a recorder, so the
     arguments, constexprs and options are the ones production builds;
  2. derives (signature, constexprs, attrs) from those arguments with Triton's own launch binder
     (runtime.jit.create_function_from_signature + the CUDA backend's parse_attr, as JITFunction._pack_args does);
     if that internal API is unavailable it falls back to a model of the same rules and says so;
  3. asserts the production specialization explicitly: tt.divisibility 16 on every pointer and on x_row_stride,
     out_row_stride, mn_aligned and M_sum; M constexpr 1 in the M == 1 variants, plain i32 otherwise; for RQ, M is
     never specialized (do_not_specialize); r2 and kit kernels get the identical specialization for every shared
     argument.
Variants: RP r2 / hoist / split x {prod M=1, prod M=4 (= every M 2..10: same SLOTS bucket, same spec), pdl M=4,
pf2 M=4, int64 ids M=1 and M=4, nopdl M=4}; RQ r2 / kit at M=1 and M=4.

Checks (exit 0 when all pass):
  RP  every kit variant: compiles; NO local-memory spills (no STL/LDL in SASS, cuobjdump -res-usage LOCAL/STACK 0 when
      reported: n_spills is '?' in this Triton build, review N1); f32 opcode SET == the r2 kernel's at the same variant
      (no fma, no approx/full div, no .NaN min/max, no e4m3 cvt); griddepcontrol.wait before the first global ld/st;
      the x-row loads come BEFORE the first shared-memory atomic (the histogram): global-load BYTES per thread issued
      before the first atom.shared (PTX) / ATOMS (SASS) exceed the r2 kernel's at the same variant by >= 80 B (hoist:
      5 chunks x 8 bf16) or >= 16 B (split: 1 chunk); counted over every ld.global / LDG width, so a vector-width change
      cannot turn the check into a false FAIL; launch_dependents only with pdl/pf; prefetch.global.L2 only with pf.
      INFO: the load widths (v4.b32 / LDG.E.128 expected in the production spec).
  RQ  _r5g_reduce_quant vs _r2_reduce_quant (PADDED_M 128, N 1024, SPLITS 32): compiles, no spills, f32 opcode multiset
      IDENTICAL, the three inline-asm bodies present verbatim, cvt.rn.bf16.f32 present.
"""
import collections
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
import types

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["DSV41_MOE_ROUTEPREP"] = "1"
os.environ.pop("DSV41_MOE_FC1_FRONT", None)  # production: unset (variants set the module globals explicitly)
os.environ["DSV41_GLUE_RQPAD"] = "1"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.compiler import ASTSource  # noqa: E402

TARGET = GPUTarget("cuda", 121, 32)
FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name}{(': ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAILS.append(name)


class _Platform:
    @staticmethod
    def is_arch_support_pdl():
        return True

    @staticmethod
    def fp8_dtype():
        return torch.float8_e4m3fn

    @staticmethod
    def is_device_capability_family(x):
        return True


def stub(name, **kw):
    m = types.ModuleType(name)
    m.__dict__.update(kw)
    sys.modules[name] = m
    return m


class _Log:
    def __getattr__(self, k):
        return lambda *a, **kw: None


for n in ("vllm", "vllm.models", "vllm.models.deepseek_v4", "vllm.models.deepseek_v4.nvidia",
          "vllm.models.deepseek_v4.nvidia.ops", "vllm.model_executor", "vllm.model_executor.layers",
          "vllm.model_executor.layers.fused_moe"):
    stub(n)
stub("vllm.platforms", current_platform=_Platform())
stub("vllm.triton_utils", tl=tl, triton=triton)
stub("vllm.logger", init_logger=lambda n: _Log())


def load(name, path, parent=None):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    if parent:
        setattr(sys.modules[parent], name.rsplit(".", 1)[1], m)
    spec.loader.exec_module(m)
    return m


AO = load("vllm.models.deepseek_v4.nvidia.ops.dsv41_attnout", os.path.join(HERE, "base/dsv41_attnout.py"),
          "vllm.models.deepseek_v4.nvidia.ops")
RPL = load("vllm.model_executor.layers.fused_moe.dsv41_routeprep", os.path.join(HERE, "base/dsv41_routeprep.py"),
           "vllm.model_executor.layers.fused_moe")
RQK = load("vllm.models.deepseek_v4.nvidia.ops.dsv41_r5g_rq",
           os.path.join(HERE, "files/vllm/models/deepseek_v4/nvidia/ops/dsv41_r5g_rq.py"),
           "vllm.models.deepseek_v4.nvidia.ops")
os.environ["DSV41_GLUE_RPSPLIT"] = "split"
RPK = load("vllm.model_executor.layers.fused_moe.dsv41_r5g_rp",
           os.path.join(HERE, "files/vllm/model_executor/layers/fused_moe/dsv41_r5g_rp.py"),
           "vllm.model_executor.layers.fused_moe")
assert RPL.PF_KB == 0 and RPL.EARLY_TRIGGER is False and RPL._PDL is True, "production FC1 front must be off, PDL on"

# ------------------------------------------------------------------ production arguments via the real launch code


class _Rec:
    """Stands in for a JITFunction: records kernel[grid](*args, **kw) / kernel.warmup(*args, grid=, **kw)."""

    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        return lambda *a, **kw: self.calls.append((grid, a, kw))

    def warmup(self, *a, grid, **kw):
        self.calls.append((grid, a, kw))


def record(mod, attr, call):
    rec, old = _Rec(), getattr(mod, attr)
    setattr(mod, attr, rec)
    try:
        call()
    finally:
        setattr(mod, attr, old)
    assert len(rec.calls) == 1, rec.calls
    return rec.calls[0]


try:  # Triton's MockTensor (what JITFunction.warmup passes for dtype-only args): data_ptr() == 0
    from triton.runtime.jit import MockTensor as _MockBase
except Exception:  # noqa: BLE001
    _MockBase = object


class FakeW(_MockBase):
    """FC1 weight / scale geometry for the (non-production) pf2 variant without allocating 0.5 GB on the host."""

    def __init__(self, dtype, shape, strides):
        self.dtype, self.shape, self._st = dtype, list(shape), tuple(strides)

    def dim(self):
        return len(self.shape)

    def size(self, i=None):
        return tuple(self.shape) if i is None else self.shape[i]

    def stride(self, i=None):
        return self._st if i is None else self._st[i]

    def element_size(self):
        return torch.empty(0, dtype=self.dtype).element_size()

    @staticmethod
    def data_ptr():
        return 0


def m_sum_fused(M):
    """M_sum on the fused path (fused_permute requires align_used == 32): round_up(M*6 + min(M*6, 384)*31, 32)."""
    s = M * RPL._TOPK + min(M * RPL._TOPK, RPL._E) * (RPL._ALIGN - 1)
    return (s + RPL._ALIGN - 1) // RPL._ALIGN * RPL._ALIGN


def rp_prod_args(M, ids_dtype, pf):
    """fused_permute's _launch(...) argument list, built from CPU tensors with production shapes and strides."""
    H, TOPK = RPL._H, RPL._TOPK
    Ms = m_sum_fused(M)
    mn_aligned = (Ms + 3) // 4 * 4
    packs = H // RPL._GROUP // 4
    aq = torch.empty((M, H), dtype=torch.bfloat16)
    topk_ids = torch.empty((M, TOPK), dtype=ids_dtype)
    aq_out = torch.empty((Ms, H), dtype=torch.uint8)  # the fp8 workspace's uint8 view: same address and strides
    sfp = torch.empty_strided((Ms, packs), (1, mn_aligned), dtype=torch.int32)
    layout = torch.empty((Ms,), dtype=torch.int32)
    inv_perm = torch.empty(topk_ids.shape, dtype=torch.int32)
    ends = torch.empty((RPL._E,), dtype=torch.int32)
    if pf:  # modelled FC1 geometry (w1 [E, N, K/2] int8, UE8M0 words MN-major): only for the non-production pf2 arm
        w1 = FakeW(torch.int8, (RPL._E, 576, 2560), (576 * 2560, 2560, 1))
        sf = FakeW(torch.int32, (RPL._E, 576, 10), (5760, 1, 576))
    else:
        w1 = sf = None  # PF_KB == 0: _pf_args fills the pointer slots with `layout` and the ints with 0
    return (aq, aq.stride(0), topk_ids, topk_ids.stride(0), topk_ids.stride(1), aq_out, aq_out.stride(0), sfp,
            mn_aligned, layout, inv_perm, inv_perm.stride(0), inv_perm.stride(1), ends, w1, sf, M, Ms)


def rp_record(kit_split, M, ids_dtype=torch.int32, trigger=False, pf=0, pdl=True):
    old = (RPL.EARLY_TRIGGER, RPL.PF_KB, RPL._PDL)
    RPL.EARLY_TRIGGER, RPL.PF_KB, RPL._PDL = trigger, pf, pdl
    try:
        a = rp_prod_args(M, ids_dtype, pf)
        if kit_split is None:
            return RPL._dsv41_rp_kernel, record(RPL, "_dsv41_rp_kernel", lambda: RPL._launch(*a))
        return RPK._r5g_rp_kernel, record(RPK, "_r5g_rp_kernel", lambda: RPK._launch_new(*a, split=kit_split))
    finally:
        RPL.EARLY_TRIGGER, RPL.PF_KB, RPL._PDL = old


def rq_record(kit, M):
    P = torch.empty((32, 1, M, 1024), dtype=torch.float32)
    if not kit:  # the live dsv41_attnout._launch_rq itself (its own two allocations)
        return AO._r2_reduce_quant, record(AO, "_r2_reduce_quant", lambda: AO._launch_rq(P, M, 1024, 32))
    q, sf = RQK._alloc(P, M, 1024)
    return RQK._r5g_reduce_quant, record(RQK, "_r5g_reduce_quant",
                                         lambda: RQK._launch_new_into(P, q, sf, M, 1024, 32))


# ------------------------------------------------------------------ specialization (Triton's own launch rules)

OPTION_KEYS = ("num_warps", "num_stages", "num_ctas", "enable_fp_fusion", "launch_pdl", "maxnreg")
try:
    from triton.compiler.compiler import make_backend
    from triton.runtime.jit import create_function_from_signature

    _BACKEND = make_backend(TARGET)
    _BINDER_ERR = None
except Exception as _e:  # noqa: BLE001
    _BACKEND, _BINDER_ERR = None, repr(_e)
_SPEC_SRC = collections.Counter()

_DT = {torch.bfloat16: "bf16", torch.float32: "fp32", torch.int32: "i32", torch.int64: "i64", torch.uint8: "u8",
       torch.int8: "i8", torch.float16: "fp16"}


def _bound(fn, args, kw):
    names = [p.name for p in fn.params]
    b = dict(zip(names, args))
    b.update({k: v for k, v in kw.items() if k in names})
    assert len(b) == len(names), (fn.__name__, sorted(set(names) - set(b)))
    return names, [b[n] for n in names]


def spec_triton(fn, args, kw):
    binder = create_function_from_signature(fn.signature, fn.params, _BACKEND)
    _, specialization, _ = binder(*args, **kw)
    names, vals = _bound(fn, args, kw)
    sigvals = [s[0] for s in specialization]
    signature = dict(zip(names, sigvals))
    constexprs = {(i,): vals[i] for i, v in enumerate(sigvals) if v == "constexpr"}
    # verbatim JITFunction._pack_args: constexpr slots carry '' (-> no attribute), others their specialization key
    attrvals = ["" if s[0] == "constexpr" else s[1] for s in specialization]
    attrs = {(i,): _BACKEND.parse_attr(a) for i, a in enumerate(attrvals) if isinstance(a, str)}
    return signature, constexprs, attrs


def spec_model(fn, args, kw):
    """Fallback: Triton 3.x specialize_impl rules (int == 1 -> constexpr 1; int % 16 == 0 / data_ptr % 16 == 0 ->
    tt.divisibility 16; bool -> u1, float -> fp32, None -> constexpr; do_not_specialize honoured)."""
    names, vals = _bound(fn, args, kw)
    signature, constexprs, attrs = {}, {}, {}
    for i, (p, v) in enumerate(zip(fn.params, vals)):
        spec = not p.do_not_specialize
        align = not p.do_not_specialize_on_alignment
        if p.is_constexpr or v is None:
            signature[p.name], constexprs[(i,)], attrs[(i,)] = "constexpr", v, []
        elif isinstance(v, bool):
            signature[p.name] = "u1"
        elif isinstance(v, int):
            if spec and v == 1:
                signature[p.name], constexprs[(i,)], attrs[(i,)] = "constexpr", 1, []
                continue
            signature[p.name] = "i32" if -2 ** 31 <= v < 2 ** 31 else "i64"
            if spec:
                attrs[(i,)] = [["tt.divisibility", 16]] if (align and v % 16 == 0) else []
        elif isinstance(v, float):
            signature[p.name] = "fp32"
        else:
            signature[p.name] = "*" + _DT[v.dtype]
            attrs[(i,)] = [["tt.divisibility", 16]] if (align and v.data_ptr() % 16 == 0) else []
    return signature, constexprs, attrs


def spec_of(fn, args, kw):
    if _BACKEND is not None:
        try:
            s = spec_triton(fn, args, kw)
            _SPEC_SRC["triton binder"] += 1
            m = spec_model(fn, args, kw)
            if m != s:
                print(f"INFO {fn.__name__}: fallback rule model differs from Triton's binder (binder used): "
                      f"{_spec_diff(fn, m, s)}"[:1200])
            return s
        except Exception as e:  # noqa: BLE001
            print(f"INFO Triton binder unavailable for {fn.__name__} ({e!r}); using the rule model")
    _SPEC_SRC["rule model"] += 1
    return spec_model(fn, args, kw)


def _spec_diff(fn, a, b):
    ba, bb = by_name(fn, *a), by_name(fn, *b)
    return {n: (ba[n], bb[n]) for n in ba if ba[n] != bb[n]}


def div16(fn, attrs, name):
    i = [p.name for p in fn.params].index(name)
    return ["tt.divisibility", 16] in attrs.get((i,), [])


def by_name(fn, sig, const, attrs):
    """Specialization as {arg name: (type, constexpr value or None, attrs)} for comparing r2 vs kit kernels."""
    out = {}
    for i, p in enumerate(fn.params):
        c = const.get((i,))
        out[p.name] = (sig[p.name], c if sig[p.name] == "constexpr" else None, attrs.get((i,)))
    return out


# ------------------------------------------------------------------ compile + inspect

def compile_spec(fn, sig, const, attrs, kw):
    opts = {k: kw[k] for k in OPTION_KEYS if k in kw}
    src = ASTSource(fn=fn, signature=sig, constexprs=const, attrs=attrs)
    try:
        return triton.compile(src, target=TARGET, options=opts)
    except TypeError:
        opts.pop("launch_pdl", None)
        return triton.compile(ASTSource(fn=fn, signature=sig, constexprs=const, attrs=attrs), target=TARGET,
                              options=opts)


def cuobjdump(cubin, flag):
    d = tempfile.mkdtemp()
    p = os.path.join(d, "k.cubin")
    open(p, "wb").write(cubin)
    try:
        r = subprocess.run(["/usr/local/cuda/bin/cuobjdump", flag, p], capture_output=True, text=True, timeout=120)
        return r.stdout + r.stderr if flag == "-res-usage" else r.stdout
    finally:
        shutil.rmtree(d, ignore_errors=True)


F32 = re.compile(r"\b((?:add|sub|mul|div|fma|max|min|neg|abs|rcp|sqrt)(?:\.[a-z]+)*\.f32)\b")
PTX_LD = re.compile(r"\bld\.global((?:\.[A-Za-z0-9_:]+)*)")
SASS_LDG = re.compile(r"\bLDG((?:\.[A-Z0-9_]+)*)")
SPILL = re.compile(r"\b(?:STL|LDL)(?:\.[A-Z0-9_]+)*\s")


def f32ops(ptx):
    return collections.Counter(F32.findall(ptx))


def ptx_ld_bytes(mods):
    v = re.search(r"\.v(\d)(?=\.|$)", mods)
    bits = re.findall(r"\.[bsuf](\d+)(?=\.|$)", mods)
    return (int(v.group(1)) if v else 1) * (int(bits[-1]) // 8 if bits else 4)


def sass_ldg_bytes(mods):
    toks = mods.split(".")[1:]
    for t, b in (("128", 16), ("64", 8), ("U16", 2), ("S16", 2), ("U8", 1), ("S8", 1)):
        if t in toks:
            return b
    return 4


def loads_before(text, marker, rx, width):
    """(bytes, {width-tag: count}) of global loads that appear before the first `marker`; None if no marker."""
    a = text.find(marker)
    if a < 0:
        return None, {}
    n = collections.Counter()
    tot = 0
    for m in rx.finditer(text[:a]):
        n[m.group(0)] += 1
        tot += width(m.group(1))
    return tot, dict(n)


def res_usage(cubin_text):
    m = re.search(r"REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)", cubin_text)
    return dict(zip(("REG", "STACK", "SHARED", "LOCAL"), map(int, m.groups()))) if m else None


def no_spills(name, sass, ru):
    """Spills go to local memory through STL/LDL; LOCAL > 0 in -res-usage means a local frame (STACK is INFO)."""
    nsp = len(SPILL.findall(sass))
    ok = bool(sass) and nsp == 0 and (ru is None or ru["LOCAL"] == 0)
    check(f"{name}: no local-memory spills (STL/LDL in SASS: {nsp}; res-usage {ru or 'not parsed'})", ok)


def main():
    out = os.path.join(HERE, "results/ptx")
    os.makedirs(out, exist_ok=True)
    print(f"triton {triton.__version__}; ptxas {os.environ.get('TRITON_PTXAS_PATH')}; target {TARGET}")
    if _BINDER_ERR:
        print(f"INFO Triton launch binder not importable ({_BINDER_ERR}); the rule model derives the specialization")

    # variant: (fn-kind, M, ids dtype, trigger, pf, pdl)
    rp_cfg = {
        "m1": (1, torch.int32, False, 0, True),  # production, M == 1 (M specialized to constexpr 1)
        "m4": (4, torch.int32, False, 0, True),  # production, M = 2..10 (identical spec; SLOTS 64)
        "i64_m1": (1, torch.int64, False, 0, True),
        "i64_m4": (4, torch.int64, False, 0, True),
        "pdl_m4": (4, torch.int32, True, 0, True),  # FC1_FRONT=pdl arm
        "pf2_m4": (4, torch.int32, True, 2, True),  # FC1_FRONT=pf2 arm (modelled FC1 geometry)
        "nopdl_m4": (4, torch.int32, False, 0, False),
    }
    jobs = {}
    for v, (M, idt, trig, pf, pdl) in rp_cfg.items():
        for kind, split in (("r2", None), ("hoist", False), ("split", True)):
            fn, (grid, args, kw) = rp_record(split, M, idt, trig, pf, pdl)
            jobs[f"{kind}_{v}"] = (fn, grid, args, kw)
    for M in (1, 4):
        for kind, kit in (("rq_r2", False), ("rq_kit", True)):
            fn, (grid, args, kw) = rq_record(kit, M)
            jobs[f"{kind}_m{M}"] = (fn, grid, args, kw)

    ptx, sas, ru, spec = {}, {}, {}, {}
    for name, (fn, grid, args, kw) in jobs.items():
        sig, const, attrs = spec_of(fn, args, kw)
        spec[name] = (fn, sig, const, attrs)
        try:
            k = compile_spec(fn, sig, const, attrs, kw)
        except Exception as e:  # noqa: BLE001
            check(f"compile {name}", False, repr(e)[:800])
            continue
        ptx[name] = k.asm["ptx"]
        open(os.path.join(out, f"{name}.ptx"), "w").write(ptx[name])
        cub = k.asm.get("cubin", b"")
        sas[name] = cuobjdump(cub, "-sass") if cub else ""
        open(os.path.join(out, f"{name}.sass"), "w").write(sas[name])
        ru[name] = res_usage(cuobjdump(cub, "-res-usage")) if cub else None
        print(f"compiled {name}: grid {grid}, ptx {len(ptx[name])} B, cubin {len(cub)} B, res-usage {ru[name]}, "
              f"shared {k.metadata.shared}")
    print(f"INFO specialization source: {dict(_SPEC_SRC)}")
    if FAILS:
        print(f"FAILED: {FAILS}")
        sys.exit(1)

    # ---- the specialization is production's (B1)
    for name, (fn, sig, const, attrs) in spec.items():
        nm = [p.name for p in fn.params]
        if name.startswith("rq"):
            need = ["P", "Q", "SF"]
            mi = nm.index("M")
            check(f"{name}: tt.divisibility 16 on {need}; M plain i32 (do_not_specialize)",
                  all(div16(fn, attrs, a) for a in need) and sig["M"] == "i32" and not attrs.get((mi,)),
                  f"sig M {sig['M']}, attrs {attrs}")
            continue
        need = ["x_ptr", "topk_ids", "out_ptr", "sfp", "layout", "inv_perm", "ends", "x_row_stride",
                "out_row_stride", "mn_aligned", "M_sum"]
        missing = [a for a in need if not div16(fn, attrs, a)]
        check(f"{name}: tt.divisibility 16 on every pointer and on x_row_stride / out_row_stride / mn_aligned / M_sum",
              not missing, f"missing {missing}")
        mi = nm.index("M")
        if name.endswith("m1"):
            check(f"{name}: M specialized to constexpr 1", sig["M"] == "constexpr" and const.get((mi,)) == 1,
                  f"{sig['M']} {const.get((mi,))}")
        else:
            check(f"{name}: M = 4 plain i32 (no constexpr, no divisibility)",
                  sig["M"] == "i32" and not attrs.get((mi,)), f"{sig['M']} {attrs.get((mi,))}")
        check(f"{name}: ids_stride1 / inv_stride1 specialized to constexpr 1",
              sig["ids_stride1"] == "constexpr" and sig["inv_stride1"] == "constexpr")
    for v in rp_cfg:
        r2 = by_name(*spec[f"r2_{v}"])
        for kind in ("hoist", "split"):
            k = by_name(*spec[f"{kind}_{v}"])
            diff = {a: (r2[a], k.get(a)) for a in r2 if r2[a] != k.get(a)}
            extra = sorted(set(k) - set(r2))
            check(f"{kind}_{v}: specialization identical to the r2 kernel's on every shared argument (+ SPLIT only)",
                  not diff and extra == ["SPLIT"], f"diff {diff} extra {extra}")
    for M in (1, 4):
        r2, kt = by_name(*spec[f"rq_r2_m{M}"]), by_name(*spec[f"rq_kit_m{M}"])
        diff = {a: (r2[a], kt.get(a)) for a in r2 if r2[a] != kt.get(a)}
        check(f"rq m{M}: kit specialization identical to the r2 kernel's (+ PAD_J only)",
              not diff and sorted(set(kt) - set(r2)) == ["PAD_J"], f"diff {diff}")

    # ---- RP
    for name in [n for n in ptx if n.startswith(("hoist", "split"))]:
        kind, v = name.split("_", 1)
        r2n = f"r2_{v}"
        p, s = ptx[name], sas[name]
        no_spills(name, s, ru[name])
        ref_set = set(f32ops(ptx[r2n]))
        check(f"{name}: f32 opcode set == {r2n}'s {sorted(ref_set)}", set(f32ops(p)) == ref_set,
              str(sorted(f32ops(p).items())))
        check(f"{name}: no fma.rn.f32 / div.full / div.approx / .NaN min-max / e4m3 cvt",
              not re.search(r"\bfma\.rn(\.ftz)?\.f32|\bdiv\.(full|approx)|\b(max|min)\.NaN\.f32|cvt\.[a-z.]*e4m3", p))
        if "nopdl" not in name:
            w = p.find("griddepcontrol.wait")
            first = min([i for i in (p.find("ld.global"), p.find("st.global")) if i >= 0] or [10 ** 9])
            check(f"{name}: griddepcontrol.wait before the first global ld/st", 0 <= w < first)
        else:
            check(f"{name}: no griddepcontrol.wait without launch_pdl", "griddepcontrol.wait" not in p)
        want = 16 if kind == "split" else 80
        kb, kn = loads_before(p, "atom.shared", PTX_LD, ptx_ld_bytes)
        rb, rn = loads_before(ptx[r2n], "atom.shared", PTX_LD, ptx_ld_bytes)
        check(f"{name}: PTX global-load bytes/thread before the first atom.shared exceed {r2n}'s by >= {want}",
              kb is not None and rb is not None and kb - rb >= want, f"kit {kb} B {kn}; r2 {rb} B {rn}")
        kb, kn = loads_before(s, "ATOMS", SASS_LDG, sass_ldg_bytes)
        rb, rn = loads_before(sas[r2n], "ATOMS", SASS_LDG, sass_ldg_bytes)
        check(f"{name}: SASS LDG bytes/thread before the first ATOMS exceed {r2n}'s by >= {want}",
              kb is not None and rb is not None and kb - rb >= want, f"kit {kb} B {kn}; r2 {rb} B {rn}")
        trig = "pdl" in v and "nopdl" not in v or "pf2" in v
        check(f"{name}: launch_dependents {'present' if trig else 'absent'}",
              ("griddepcontrol.launch_dependents" in p) == trig)
        check(f"{name}: prefetch.global.L2 {'present' if 'pf2' in v else 'absent'}",
              ("prefetch.global.L2" in p) == ("pf2" in v))
    for v in rp_cfg:
        n = f"r2_{v}"
        print(f"INFO {n}: loads before the histogram PTX {loads_before(ptx[n], 'atom.shared', PTX_LD, ptx_ld_bytes)} "
              f"SASS {loads_before(sas[n], 'ATOMS', SASS_LDG, sass_ldg_bytes)}; all PTX load widths "
              f"{dict(collections.Counter(m.group(0) for m in PTX_LD.finditer(ptx[n])))}; res-usage {ru[n]}")
    for n in ("hoist_m4", "split_m4", "hoist_m1", "split_m1"):
        print(f"INFO {n}: PTX load widths {dict(collections.Counter(m.group(0) for m in PTX_LD.finditer(ptx[n])))}; "
              f"SASS LDG widths {dict(collections.Counter(m.group(0) for m in SASS_LDG.finditer(sas[n])))} "
              f"(production spec: the x chunks are expected as ld.global.v4.b32 / LDG.E.128)")
    print(f"INFO f32 op counts r2_m4 {dict(f32ops(ptx['r2_m4']))} hoist_m4 {dict(f32ops(ptx['hoist_m4']))} "
          f"split_m4 {dict(f32ops(ptx['split_m4']))}")
    # ---- RQ
    for M in (1, 4):
        r2, kt = ptx[f"rq_r2_m{M}"], ptx[f"rq_kit_m{M}"]
        check(f"rq m{M}: f32 opcode multiset identical to _r2_reduce_quant", f32ops(r2) == f32ops(kt),
              f"{dict(f32ops(r2))} vs {dict(f32ops(kt))}")
        for nm in ("_ASM_UE8M0", "_ASM_INV_SCALE", "_ASM_FP8"):
            body = [ln.strip() for ln in str(getattr(AO, nm).value if hasattr(getattr(AO, nm), "value")
                                             else getattr(AO, nm)).splitlines() if ln.strip() and "$" not in ln]
            check(f"rq m{M}: inline asm {nm} instructions present in the kit PTX as in r2",
                  all(b in kt for b in body) and all(b in r2 for b in body))
        check(f"rq m{M}: cvt.rn.bf16.f32 in both", "cvt.rn.bf16.f32" in r2 and "cvt.rn.bf16.f32" in kt)
        no_spills(f"rq_kit_m{M}", sas[f"rq_kit_m{M}"], ru[f"rq_kit_m{M}"])
        print(f"INFO rq m{M} SASS STG.E.U8 count r2 {sas[f'rq_r2_m{M}'].count('STG.E.U8')} kit "
              f"{sas[f'rq_kit_m{M}'].count('STG.E.U8')}; BRA r2 {sas[f'rq_r2_m{M}'].count('BRA')} kit "
              f"{sas[f'rq_kit_m{M}'].count('BRA')}; res-usage r2 {ru[f'rq_r2_m{M}']} kit {ru[f'rq_kit_m{M}']}")
    shutil.rmtree(os.environ.get("TRITON_CACHE_DIR", "/nonexistent"), ignore_errors=True)
    print("ALL PASS" if not FAILS else f"FAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
