"""CPU unit tests of dsv41_routeprep's Python glue (torch CPU; vLLM / Triton replaced by stubs, no GPU).

  * env parsing of DSV41_MOE_ROUTEPREP / DSV41_MOE_FC1_FRONT (valid values, ValueError on invalid ones)
  * expects_unquantized_inputs: True only for the target experts object (E384, top-6, H5120, target_m32 set)
  * front(): fused path exactly when the live code would take the M32 align-32 path (M 1..96) with mfront on and the
    static quant constants match; otherwise the very prepare() call `_quantize_input(x, quant_config)`
  * fused_permute(): launch grid / constexprs / buffer shapes and strides (same contract as dsv41_fused_permute),
    lever-2 arguments only with valid w1 / w1_scale geometry, and refusal (RuntimeError) for anything that is not the
    fused geometry or where x and the slot rows overlap in memory
  * one-time precompile (_ready): at the first eligible call every SLOTS bucket (64..1024) is compiled through
    JITFunction.warmup with exactly the constexprs / options / int args the real launch passes for that M; it runs once
    per process; a warmup exception (or the static check failing) turns the fused path off for the whole process and
    every call then takes the prepare() quant
Run: python3 tests/test_module.py
"""
import ast
import importlib
import os
import sys
import types

import torch

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOD = os.path.join(KIT, "files/vllm/model_executor/layers/fused_moe/dsv41_routeprep.py")
BASE_UTILS = os.path.join(KIT, "base/model_executor/layers/fused_moe/deep_gemm_utils.py")
FAIL = []
CALLS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name}{(': ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAIL.append(name)


class _Any:
    def __getattr__(self, k):
        return _Any()

    def __call__(self, *a, **k):
        return _Any()


def install_stubs(fp_mode="1", e8m0=True):
    for name in list(sys.modules):
        if name == "vllm" or name.startswith("vllm."):
            del sys.modules[name]

    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        sys.modules[name] = m
        return m

    class Platform:
        @staticmethod
        def is_arch_support_pdl():
            return True

        @staticmethod
        def fp8_dtype():
            return torch.float8_e4m3fn

    triton = types.SimpleNamespace(jit=lambda f: f, next_power_of_2=lambda n: 1 << (int(n) - 1).bit_length(),
                                   cdiv=lambda a, b: -(-a // b))
    mod("vllm")
    mod("vllm.platforms", current_platform=Platform())
    mod("vllm.triton_utils", tl=_Any(), triton=triton)
    mod("vllm.logger", init_logger=lambda n: types.SimpleNamespace(info=lambda *a, **k: None))
    mod("vllm.utils")
    mod("vllm.utils.deep_gemm", is_deep_gemm_e8m0_used=lambda: e8m0)
    mod("vllm.model_executor")
    mod("vllm.model_executor.layers")
    mod("vllm.model_executor.layers.quantization")
    mod("vllm.model_executor.layers.quantization.utils")
    mod("vllm.model_executor.layers.quantization.utils.quant_utils", get_fp8_min_max=lambda: (-448.0, 448.0))
    mod("vllm.model_executor.layers.fused_moe")
    # deep_gemm_utils: the live compute_aligned_M_and_alignment source, plus the mfront mode
    src = open(BASE_UTILS).read()
    t = ast.parse(src)
    fn = next(n for n in t.body if isinstance(n, ast.FunctionDef) and n.name == "compute_aligned_M_and_alignment")
    ns = {"round_up": lambda x, a: (x + a - 1) // a * a, "mk": types.SimpleNamespace(ExpertTokensMetadata=object)}
    exec(compile(ast.Module([fn], []), "dgu", "exec"), ns)
    # its SM120 helper import fails here -> alignment stays as passed (the live behaviour when it is <= 32)
    mod("vllm.model_executor.layers.fused_moe.deep_gemm_utils",
        compute_aligned_M_and_alignment=ns["compute_aligned_M_and_alignment"], _DSV41_FP_MODE=fp_mode)

    def _quantize_input(a1, qc, defer_input_quant=False):
        CALLS.append(("quant", a1, qc, defer_input_quant))
        return "A1Q", "A1Q_SCALE"
    mod("vllm.model_executor.layers.fused_moe.prepare_finalize")
    mod("vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep", _quantize_input=_quantize_input)


def load(env):
    for k in ("DSV41_MOE_ROUTEPREP", "DSV41_MOE_FC1_FRONT"):
        os.environ.pop(k, None)
    os.environ.update(env)
    spec = importlib.util.spec_from_file_location("dsv41_routeprep_t", MOD)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_env():
    install_stubs()
    for front, want in (("", (False, 0)), ("0", (False, 0)), ("pdl", (True, 0)), ("pf1", (True, 1)),
                        ("PF4", (True, 4)), ("pf8", (True, 8))):
        m = load({"DSV41_MOE_ROUTEPREP": "1", "DSV41_MOE_FC1_FRONT": front})
        check(f"env FC1_FRONT={front!r} -> trigger/prefetch {want}", (m.EARLY_TRIGGER, m.PF_KB) == want)
    for env in ({"DSV41_MOE_ROUTEPREP": "2"}, {"DSV41_MOE_ROUTEPREP": "verify"},
                {"DSV41_MOE_ROUTEPREP": "1", "DSV41_MOE_FC1_FRONT": "pf3"},
                {"DSV41_MOE_ROUTEPREP": "1", "DSV41_MOE_FC1_FRONT": "early"}):
        try:
            load(env)
            check(f"env {env} raises ValueError", False)
        except ValueError:
            check(f"env {env} raises ValueError", True)


class Cfg:
    def __init__(self, E=384, k=6, H=5120):
        self.num_experts, self.experts_per_token, self.hidden_dim = E, k, H


class QC:
    def __init__(self, dtype=torch.float8_e4m3fn, block=(128, 128), a1_scale=None, per_tok=False):
        self.quant_dtype, self.block_shape, self.a1_scale, self.per_act_token_quant = dtype, block, a1_scale, per_tok


class Experts:
    def __init__(self, cfg=None, target=True, qc=None, force=False):
        self.moe_config = cfg or Cfg()
        self._dsv41_target_m32 = object() if target else None
        self.quant_config = qc or QC()
        self._dsv41_force_m32 = force
        self.w1_scale = "W1S"

    @staticmethod
    def _workspace_m_alignment(M, N, K, topk, experts):  # live: 32 for target geometry, 1 <= M <= 96
        return 32 if (1 <= M <= 96 and (N, K, topk, experts) in ((576, 5120, 6, 384), (768, 5120, 6, 384))) else 128


def test_expects():
    install_stubs()
    m = load({"DSV41_MOE_ROUTEPREP": "1"})
    cases = [(Experts(), True), (Experts(target=False), False), (Experts(Cfg(E=128, k=3)), False),
             (Experts(Cfg(k=8)), False), (Experts(Cfg(H=4096)), False)]
    check("expects_unquantized_inputs truth table", all(m.expects_unquantized_inputs(e) == w for e, w in cases))


def test_front():
    w1 = torch.empty((384, 576, 2560), dtype=torch.uint8, device="meta")
    for fp_mode, e8m0, want_on in (("1", True, True), ("0", True, False), ("verify", True, False),
                                   ("1", False, False)):
        install_stubs(fp_mode, e8m0)
        m = load({"DSV41_MOE_ROUTEPREP": "1"})
        m._dsv41_rp_kernel = Recorder()
        ok = True
        for M in (1, 2, 4, 6, 16, 48, 96, 97, 128, 512):
            x = torch.zeros((M, 5120), dtype=torch.bfloat16)
            ids = torch.zeros((M, 6), dtype=torch.int32)
            CALLS.clear()
            exp = Experts()
            out = m.front(exp, x, w1, ids, None, None)
            fused = want_on and M <= 96
            if fused:
                ok &= out[0] is x and isinstance(out[1], m.Deferred) and out[1].w1 is w1 and not CALLS
            else:
                ok &= out == ("A1Q", "A1Q_SCALE") and len(CALLS) == 1 and CALLS[0][1] is x and \
                    CALLS[0][2] is exp.quant_config and CALLS[0][3] is False
        check(f"front(): fused iff M<=96 with mfront={fp_mode} e8m0={e8m0} (else the prepare() call)", ok)
    install_stubs()
    m = load({"DSV41_MOE_ROUTEPREP": "1"})
    m._dsv41_rp_kernel = Recorder()
    x = torch.zeros((4, 5120), dtype=torch.bfloat16)
    ids = torch.zeros((4, 6), dtype=torch.int32)
    bad = [("expert_map", dict(expert_map=torch.zeros(384))), ("force_m32", dict(exp=Experts(force=True))),
           ("a1_scale", dict(exp=Experts(qc=QC(a1_scale=torch.ones(1))))),
           ("group 32", dict(exp=Experts(qc=QC(block=(1, 32))))), ("fp16 x", dict(x=x.half())),
           ("int16 ids", dict(ids=ids.short())), ("non-contiguous x", dict(x=torch.zeros((5120, 4)).bfloat16().t()))]
    ok = True
    for name, kw in bad:
        CALLS.clear()
        out = m.front(kw.get("exp", Experts()), kw.get("x", x), w1, kw.get("ids", ids), kw.get("expert_map"), None)
        ok &= out == ("A1Q", "A1Q_SCALE")
    check("front(): every ineligible variant takes the prepare() quant", ok)


class Recorder:
    def __init__(self, fail_warmup=False):
        self.calls = []
        self.warm = []
        self.fail_warmup = fail_warmup

    def __getitem__(self, grid):
        def launch(*args, **kw):
            self.calls.append((grid, args, kw))
        return launch

    def warmup(self, *args, grid, **kw):
        self.warm.append((grid, args, kw))
        if self.fail_warmup:
            raise RuntimeError("PassManager::run failed (simulated ptxas error)")


def test_fused_permute():
    install_stubs()
    for front, pf in (("0", 0), ("pdl", 0), ("pf2", 2)):
        m = load({"DSV41_MOE_ROUTEPREP": "1", "DSV41_MOE_FC1_FRONT": front})
        rec = Recorder()
        m._dsv41_rp_kernel = rec
        w1 = torch.empty((384, 576, 2560), dtype=torch.int8, device="meta")
        sf = torch.empty_strided((384, 576, 40), (23040, 1, 576), dtype=torch.int32, device="meta")
        ok = True
        for M in (1, 4, 6, 11, 96, 170):
            Ms = M * 6 + min(M * 6, 384) * 31
            Ms = (Ms + 31) // 32 * 32
            x = torch.zeros((M, 5120), dtype=torch.bfloat16)
            ids = torch.zeros((M, 6), dtype=torch.int64)
            aq_out = torch.empty((Ms, 5120), dtype=torch.float8_e4m3fn)
            r = m.fused_permute(x, m.Deferred(w1, sf), ids, 384, None, None, aq_out, Ms, 32, 128, False, True)
            grid, args, kw = rec.calls[-1]
            sfp, layout, inv, ends = r[1], r[2], r[3], r[5]
            ok &= grid == (M * 6,) and r[0] is aq_out and r[4] == 32 and len(r) == 6
            ok &= tuple(sfp.shape) == (Ms, 10) and sfp.stride() == (1, (Ms + 3) // 4 * 4) and sfp.dtype == torch.int32
            ok &= tuple(layout.shape) == (Ms,) and tuple(inv.shape) == (M, 6) and tuple(ends.shape) == (384,)
            ok &= kw["SLOTS"] >= M * 6 and kw["SLOTS"] >= 64 and kw["E_PAD"] == 512 and kw["PACKS"] == 10
            ok &= kw["EARLY_TRIGGER"] == (front != "0") and kw["PF_KB"] == pf and kw["launch_pdl"] is True
            ok &= kw["enable_fp_fusion"] is False and kw["num_warps"] == 4
            ok &= args[-5:] == (M, Ms, 1e-10, -448.0, 448.0)
            if pf:
                ok &= kw["PF_N"] == 576 and kw["PF_SF_STEP"] == 32 and kw["PF_SPAD"] == 32 and kw["PF_NPAD"] == 1024
                ok &= args[14] is w1 and args[15:17] == (576 * 2560, 2560) and args[17] is sf
                ok &= args[18:21] == (23040, 1, 576)
            else:
                ok &= kw["PF_N"] == 1 and args[15:17] == (0, 0) and args[18:21] == (0, 0, 0)
            r5 = m.fused_permute(x, m.Deferred(w1, sf), ids, 384, None, None, aq_out, Ms, 32, 128, False, False)
            ok &= len(r5) == 5
        check(f"fused_permute launch contract (FC1_FRONT={front})", ok)
        # lever-2 geometry that does not fit -> prefetch silently off, everything else unchanged
        if pf:
            m.fused_permute(x, m.Deferred(w1, sf[:, :, :1]), ids, 384, None, None, aq_out, Ms, 32, 128, False, True)
            check("fused_permute: w1_scale too short for pf2 -> PF_KB 0", rec.calls[-1][2]["PF_KB"] == 0)
    m = load({"DSV41_MOE_ROUTEPREP": "1"})
    m._dsv41_rp_kernel = Recorder()
    x = torch.zeros((4, 5120), dtype=torch.bfloat16)
    ids = torch.zeros((4, 6), dtype=torch.int32)
    Ms = 4 * 6 + 24 * 31
    Ms = (Ms + 31) // 32 * 32
    aq_out = torch.empty((Ms, 5120), dtype=torch.float8_e4m3fn)
    d = m.Deferred(None, None)
    refusals = {
        "plain tensor scale": (x, torch.ones(4, 40), ids, 384, None, aq_out, Ms, 32, 128, False),
        "align 64": (x, d, ids, 384, None, aq_out, Ms, 64, 128, False),
        "psum layout": (x, d, ids, 384, None, aq_out, Ms, 32, 128, True),
        "expert_map": (x, d, ids, 384, torch.zeros(384), aq_out, Ms, 32, 128, False),
        "E 128": (x, d, ids, 128, None, aq_out, Ms, 32, 128, False),
        "aq_out None": (x, d, ids, 384, None, None, Ms, 32, 128, False),
        "fp8 x": (x.to(torch.float8_e4m3fn), d, ids, 384, None, aq_out, Ms, 32, 128, False),
        "x overlaps slot rows": (aq_out.view(torch.uint8)[:8].view(torch.bfloat16)[:4, :5120].contiguous()
                                 if False else None, d, ids, 384, None, aq_out, Ms, 32, 128, False),
    }
    # overlap: x is a bf16 view INTO aq_out's storage
    big = torch.empty((Ms, 5120), dtype=torch.float8_e4m3fn)
    xv = big.view(torch.uint8)[: 4 * 2].view(torch.bfloat16)  # rows 0..7 bytes -> [8, 2560] bf16
    xv = big.view(torch.uint8).reshape(-1)[: 4 * 5120 * 2].view(torch.bfloat16).view(4, 5120)
    refusals["x overlaps slot rows"] = (xv, d, ids, 384, None, big, Ms, 32, 128, False)
    ok = True
    for name, (a, s, i, E, emap, out, ms, al, bk, ps) in refusals.items():
        try:
            m.fused_permute(a, s, i, E, emap, None, out, ms, al, bk, ps, False)
            ok = False
            print("   no refusal:", name)
        except RuntimeError:
            pass
    check("fused_permute refuses non-fused geometry and overlapping x / slot rows", ok)


def test_precompile():
    w1 = torch.empty((384, 576, 2560), dtype=torch.int8, device="meta")
    sf = torch.empty_strided((384, 576, 40), (23040, 1, 576), dtype=torch.int32, device="meta")
    for front, pf in (("0", 0), ("pdl", 0), ("pf2", 2)):
        install_stubs()
        m = load({"DSV41_MOE_ROUTEPREP": "1", "DSV41_MOE_FC1_FRONT": front})
        rec = Recorder()
        m._dsv41_rp_kernel = rec
        exp = Experts()
        exp.w1_scale = sf
        x = torch.zeros((4, 5120), dtype=torch.bfloat16)
        ids = torch.zeros((4, 6), dtype=torch.int64)
        out = m.front(exp, x, w1, ids, None, None)
        ok = isinstance(out[1], m.Deferred) and m._READY is True
        ok &= [g for g, _, _ in rec.warm] == [(6,), (96,), (192,), (384,), (576,)] and not rec.calls
        ok &= sorted(kw["SLOTS"] for _, _, kw in rec.warm) == [64, 128, 256, 512, 1024]
        ok &= all(a[2].dtype == torch.int64 for _, a, _ in rec.warm)  # ids pointer dtype of the real call
        # every warmup = the real launch for that M: same constexprs/options and the same int args
        for grid, wa, wkw in rec.warm:
            M = grid[0] // 6
            Ms, _ = m._m_sum(M)
            xr = torch.zeros((M, 5120), dtype=torch.bfloat16)
            idr = torch.zeros((M, 6), dtype=torch.int64)
            m.fused_permute(xr, m.Deferred(w1, sf), idr, 384, None, None,
                            torch.empty((Ms, 5120), dtype=torch.float8_e4m3fn), Ms, 32, 128, False, True)
            g2, la, lkw = rec.calls[-1]
            ints = lambda a: [v for v in a if not isinstance(v, torch.Tensor)]  # noqa: E731
            dts = lambda a: [v.dtype for v in a if isinstance(v, torch.Tensor)]  # noqa: E731
            ok &= g2 == grid and lkw == wkw and ints(la) == ints(wa) and dts(la) == dts(wa)
        ok &= rec.warm[0][2]["PF_KB"] == pf and rec.warm[0][2]["EARLY_TRIGGER"] == (front != "0")
        n_warm = len(rec.warm)
        for M in (1, 6, 96):  # decided once: no further warmups
            m.front(exp, torch.zeros((M, 5120), dtype=torch.bfloat16), w1, torch.zeros((M, 6), dtype=torch.int64),
                    None, None)
        ok &= len(rec.warm) == n_warm
        check(f"precompile once, 5 SLOTS buckets, same args as the real launch (FC1_FRONT={front})", ok)
    # a compile failure turns the fused path off for the process: every call takes the prepare() quant
    install_stubs()
    m = load({"DSV41_MOE_ROUTEPREP": "1", "DSV41_MOE_FC1_FRONT": "pf2"})
    rec = Recorder(fail_warmup=True)
    m._dsv41_rp_kernel = rec
    exp = Experts()
    exp.w1_scale = sf
    ok = True
    for M in (4, 1, 96, 4):
        CALLS.clear()
        out = m.front(exp, torch.zeros((M, 5120), dtype=torch.bfloat16), w1, torch.zeros((M, 6), dtype=torch.int32),
                      None, None)
        ok &= out == ("A1Q", "A1Q_SCALE") and len(CALLS) == 1
    ok &= m._READY is False and len(rec.warm) == 1 and not rec.calls
    check("precompile failure -> fused path OFF for the process, every call takes the prepare() quant", ok)
    # static check failing: no precompile at all
    install_stubs("0")
    m = load({"DSV41_MOE_ROUTEPREP": "1"})
    rec = Recorder()
    m._dsv41_rp_kernel = rec
    out = m.front(Experts(), torch.zeros((4, 5120), dtype=torch.bfloat16), w1, torch.zeros((4, 6), dtype=torch.int32),
                  None, None)
    check("static check failing -> no precompile, prepare() quant", out == ("A1Q", "A1Q_SCALE") and not rec.warm)


if __name__ == "__main__":
    test_env()
    test_expects()
    test_front()
    test_fused_permute()
    test_precompile()
    print("ALL PASS" if not FAIL else f"FAILED: {FAIL}")
    sys.exit(1 if FAIL else 0)
