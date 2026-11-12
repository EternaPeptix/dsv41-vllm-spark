#!/usr/bin/env python3
"""r2-draftmoe plan tests (torch CPU): the patched Python control flow, run for real with stubbed GPU calls.

The patched functions are extracted (AST) from the installed copy of the live files and executed against stubs
that record every GPU call (kernel, shapes, alignment, kwargs). The same scenarios run through the BASE functions.
  * flags off: the call log of every scenario is identical to the base file's.
  * flags on : target MoE and padded-drafter scenarios are identical to base; the compact-drafter scenario issues
               FC1 at N=576, the compact SiLU writer (padded_output_size=384, logical width 288) and an unchanged FC2.
  * permute  : with perm the drafter geometry on the M32 path calls dsv41_fused_permute(E=128), everything else
               stays stock; permverify serves stock and calls the verifier; the target path is untouched.
Run: python3 tests/test_plan.py   (needs torch; numpy not needed)
"""
import ast
import contextlib
import enum
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import types

import torch

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(KIT, "base")
DGMOE = "model_executor/layers/fused_moe/experts/deep_gemm_moe.py"
DGUTL = "model_executor/layers/fused_moe/deep_gemm_utils.py"
DRM32 = "model_executor/layers/fused_moe/experts/dsv41_draft_m32.py"
MXFP4 = "model_executor/layers/quantization/mxfp4.py"
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""), flush=True)


# installed copy of the four files
INST = tempfile.mkdtemp(prefix="r2dm-plan-")
for rel in (DGMOE, DGUTL, DRM32, MXFP4):
    os.makedirs(os.path.dirname(os.path.join(INST, rel)), exist_ok=True)
    shutil.copy2(os.path.join(BASE, rel), os.path.join(INST, rel))
r = subprocess.run([sys.executable, os.path.join(KIT, "install.py"), "install"],
                   env=dict(os.environ, R2DM_VLLM=INST, R2DM_HOME=INST), capture_output=True, text=True)
assert r.returncode == 0, r.stderr

spec = importlib.util.spec_from_file_location(
    "dsv41_r2_draftmoe", os.path.join(KIT, "new/model_executor/layers/fused_moe/experts/dsv41_r2_draftmoe.py"))
R2DM = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R2DM)
R2DM._log = lambda *a, **k: None


def extract(path, names, cls=None):
    """{name: source} of top-level functions or methods of class `cls` (with decorators)."""
    src = open(path).read()
    tree = ast.parse(src)
    body = tree.body
    if cls:
        body = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body
    out = {}
    for n in body:
        if isinstance(n, ast.FunctionDef) and n.name in names:
            out[n.name] = ast.unparse(n)
    missing = set(names) - set(out)
    assert not missing, missing
    return out


# ---------------------------------------------------------------- stubs
class MoEActivation(enum.Enum):
    SILU = 1
    SWIGLUSTEP = 2
    SITU = 3


class Fmt(enum.Enum):
    UE8M0 = 1
    FLOAT32_CEIL_UE8M0 = 2

    @staticmethod
    def from_oracle():
        return Fmt.UE8M0


class Log(list):
    pass


MK = types.SimpleNamespace(ExpertTokensMetadata=object)


def make_globals(log, flags):
    def resize(x, v):
        n = 1
        for d in v:
            n *= d
        return x.flatten()[:n].view(*v)

    def permute(aq, aq_scale, topk_ids, local_num_experts, expert_map, expert_tokens_meta, aq_out=None,
                alignment_m=None, return_expert_ends=False, **kw):
        log.append(("permute", tuple(aq.shape), local_num_experts, alignment_m, return_expert_ends))
        M_sum = aq_out.shape[0]
        ids = torch.zeros(M_sum, dtype=torch.int32)
        res = (aq_out, torch.zeros(M_sum, 40), ids, torch.zeros(topk_ids.shape, dtype=torch.int32), alignment_m)
        return res + ((torch.zeros(local_num_experts, dtype=torch.int32),) if return_expert_ends else ())

    class Kern:
        def gemm(self, a, b, d, ids, **kw):
            log.append(("gemm32", tuple(a[0].shape), tuple(b[0].shape), tuple(d.shape), tuple(sorted(kw.items()))))

        @contextlib.contextmanager
        def alignment_scope(self):
            log.append(("scope32",))
            yield

    def dg_gemm(a, b, d, ids, **kw):
        log.append(("gemm128", tuple(a[0].shape), tuple(b[0].shape), tuple(d.shape), tuple(sorted(kw.items()))))

    @contextlib.contextmanager
    def scope(a):
        log.append(("scope", a))
        yield

    def silu(input, output_q, **kw):
        log.append(("silu", tuple(input.shape), tuple(output_q.shape),
                    tuple(sorted((k, v if not torch.is_tensor(v) else tuple(v.shape)) for k, v in kw.items()))))
        return output_q, torch.zeros(1)

    def gather(a, topk_ids, topk_weights, inv_perm, expert_map, output):
        log.append(("gather", tuple(a.shape), tuple(output.shape)))

    plat = types.SimpleNamespace(is_device_capability=lambda c: c == 121,
                                 is_device_capability_family=lambda c: c == 120)
    g = dict(torch=torch, os=os, MoEActivation=MoEActivation, DeepGemmQuantScaleFMT=Fmt, current_platform=plat,
             get_mk_alignment_for_contiguous_layout=lambda: (128, 128), deepgemm_moe_permute=permute,
             _resize_cache=resize, mk_alignment_scope=scope, m_grouped_fp8_fp4_gemm_nt_contiguous=dg_gemm,
             fused_silu_mul_fp8_quant_packed=silu, deepgemm_unpermute_and_reduce=gather,
             silu_mul_per_token_group_quant_fp8_colmajor=None, mk=MK, _KERN=Kern())
    g.update(flags)
    # real compute_aligned_M_and_alignment (+ round_up) from the live deep_gemm_utils
    exec("def round_up(x, y):\n    return ((x + y - 1) // y) * y\n", g)
    for src in extract(os.path.join(BASE, DGUTL), {"compute_aligned_M_and_alignment"}).values():
        exec(src, g)
    return g


METHODS = {"apply", "_act_mul_quant", "_workspace_m_alignment", "_dsv41_compact_fc1", "_dsv41_target_geometry",
           "_dsv41_live_act"}


def build_expert(path, log, flags):
    g = make_globals(log, flags)
    ns = {}
    for name, src in extract(path, METHODS, cls="DeepGemmFP4Experts").items():
        exec(src, g, ns)
    cls = type("DeepGemmFP4Experts", (), dict(ns))
    cls._ACT_BLOCK_K, cls._WEIGHT_BLOCK_K = 128, 32
    cls.adjust_N_for_activation = staticmethod(lambda N, act: N // 2)
    g["DeepGemmFP4Experts"] = cls
    e = cls()
    e.gemm1_clamp_limit = 10.0
    e._dsv41_draft_verify = None
    e._dsv41_force_m32 = False
    e._dsv41_target_m32 = g["_KERN"]
    e._dsv41_live_act_min_rows, e._dsv41_live_act_max_rows = 192, 0
    e.w1_scale = torch.zeros(1)
    e.w2_scale = torch.zeros(1)
    return e


ENV = dict(DSV41_SM121_TARGET_M32="1", DSV41_TARGET_M32_MAX_M="96", DSV41_DRAFT_MOE_M32="1",
           DSV41_DRAFT_M32_MAX_M="160", DSV41_MOE_COMPACT="1", DSV41_MOE_LIVE_ACT="1")
os.environ.update(ENV)
os.environ.pop("DSV41_SM121_TARGET_M64", None)

# (name, E, topk, N) -- the three geometries that can exist in production
SCEN = [("target-compact", 384, 6, 576), ("drafter-padded", 128, 3, 768), ("drafter-compact", 128, 3, 576)]
MS = [1, 2, 5, 10, 32, 64, 96, 97, 160, 161, 300]


def run(path, flags, E, topk, N, M):
    log = Log()
    e = build_expert(path, log, flags)
    K = 5120
    hidden = torch.zeros(M, K, dtype=torch.float8_e4m3fn)
    w1 = torch.zeros(E, N, K // 2, dtype=torch.uint8)
    w2 = torch.zeros(E, K, 192, dtype=torch.uint8)
    ids = torch.zeros(M, topk, dtype=torch.int32)
    tw = torch.zeros(M, topk)
    rows = topk * M + E * 128 + 128
    ws13 = torch.empty(rows * K // 2, dtype=torch.bfloat16)
    ws2 = torch.empty(rows * K, dtype=torch.bfloat16)
    out = torch.zeros(M, K, dtype=torch.bfloat16)
    try:
        e.apply(output=out, hidden_states=hidden, w1=w1, w2=w2, topk_weights=tw, topk_ids=ids,
                activation=MoEActivation.SILU, global_num_experts=-1, expert_map=None, a1q_scale=torch.zeros(M, 40),
                a2_scale=None, workspace13=ws13, workspace2=ws2, expert_tokens_meta=None,
                apply_router_weight_on_input=False)
    except AssertionError as exc:
        log.append(("AssertionError", str(exc)))
    return list(log)


OFF = dict(_R2DM_FC1=False, _R2DM_PERM="")
ON = dict(_R2DM_FC1=True, _R2DM_PERM="1", _r2dm=R2DM)
patched_apply = os.path.join(INST, DGMOE)
base_apply = os.path.join(BASE, DGMOE)

for name, E, topk, N in SCEN:
    same_off = all(run(patched_apply, OFF, E, topk, N, M) == run(base_apply, {}, E, topk, N, M) for M in MS)
    check(f"flags off: apply() call log == base for {name} at M={MS}", same_off)
for name, E, topk, N in SCEN[:2]:
    same_on = all(run(patched_apply, ON, E, topk, N, M) == run(base_apply, {}, E, topk, N, M) for M in MS)
    check(f"flags on: apply() call log == base for {name} (lever only touches the compact drafter)", same_on)

# compact drafter, flags on: explicit expectations
ok, detail = True, ""
for M in MS:
    log = run(patched_apply, ON, 128, 3, 576, M)
    padded = run(base_apply, {}, 128, 3, 768, M)
    kinds = [x[0] for x in log]
    m32 = M <= 160
    M_sum = -(-(3 * M + min(3 * M, 128) * ((32 if m32 else 128) - 1)) // (32 if m32 else 128)) * (32 if m32 else 128)
    gk = "gemm32" if m32 else "gemm128"
    fc1 = [x for x in log if x[0] == gk][0]
    fc2 = [x for x in log if x[0] == gk][1]
    silu = [x for x in log if x[0] == "silu"][0]
    p_fc2 = [x for x in padded if x[0] == gk][1]
    p_silu = [x for x in padded if x[0] == "silu"][0]
    want = (kinds[0] == "permute" and log[0][3] == (32 if m32 else 128) and "AssertionError" not in kinds
            and fc1[1] == (M_sum, 5120) and fc1[2] == (128, 576, 2560) and fc1[3] == (M_sum, 576)
            and silu[1] == (M_sum, 576) and silu[2] == (M_sum, 384) and ("padded_output_size", 384) in silu[3]
            and fc2 == p_fc2                                   # FC2 call identical to the padded drafter's
            and silu[2] == p_silu[2]                           # same FC2-input buffer shape
            and [x for x in log if x[0] == "gather"] == [x for x in padded if x[0] == "gather"]
            and (("expert_alignment", (32 if m32 else 128) if M_sum >= 192 else 0) in silu[3]))
    if not want:
        ok, detail = False, f"M={M}: {log}"
        break
check("flags on, compact drafter: FC1 N=576, compact SiLU writer (576 -> 384, padded_output_size=384), FC2 and "
      "gather calls identical to the padded drafter's, M32 for M<=160 / 128-row above, LIVE_ACT at M_sum>=192",
      ok, detail)
# compact weights can only exist with the flag on (the mxfp4 gate is flag-gated): with flags off the compact
# geometry would not be recognised -- shown here to document why the two gates must agree
log = run(patched_apply, OFF, 128, 3, 576, 5)
check("documented: compact drafter weights without the flag would take the non-compact branch (gates agree: both "
      "need DSV41_DRAFT_MOE_G17 fc1)", [x for x in log if x[0] == "silu"][0][2][1] == 288)

# ---------------------------------------------------------------- _workspace_m_alignment table
g_on, g_off = make_globals(Log(), ON), make_globals(Log(), OFF)
for g, path in ((g_on, patched_apply), (g_off, patched_apply)):
    ns = {}
    for src in extract(path, {"_workspace_m_alignment", "_dsv41_compact_fc1", "_dsv41_target_geometry"},
                       cls="DeepGemmFP4Experts").values():
        exec(src, g, ns)
    g["DeepGemmFP4Experts"] = type("D", (), ns)
gb = make_globals(Log(), {})
ns = {}
for src in extract(base_apply, {"_workspace_m_alignment", "_dsv41_compact_fc1", "_dsv41_target_geometry"},
                   cls="DeepGemmFP4Experts").values():
    exec(src, gb, ns)
gb["DeepGemmFP4Experts"] = type("D", (), ns)
tab_ok = True
for M in range(1, 401):
    for (N, K, t, E) in [(576, 5120, 6, 384), (768, 5120, 6, 384), (768, 5120, 3, 128), (576, 5120, 3, 128)]:
        b = gb["DeepGemmFP4Experts"]._workspace_m_alignment(M, N, K, t, E)
        on = g_on["DeepGemmFP4Experts"]._workspace_m_alignment(M, N, K, t, E)
        off = g_off["DeepGemmFP4Experts"]._workspace_m_alignment(M, N, K, t, E)
        exp_on = (32 if M <= 160 else 128) if (N, E) == (576, 128) else b
        if off != b or on != exp_on:
            tab_ok = False
check("_workspace_m_alignment M=1..400 x 4 geometries: off == base; on == base except compact drafter "
      "(32 for M<=160, 128 above)", tab_ok)
check("_dsv41_compact_fc1 unchanged (target-only) with the flag on",
      g_on["DeepGemmFP4Experts"]._dsv41_compact_fc1(576, 5120, 3, 128) is False
      and g_on["DeepGemmFP4Experts"]._dsv41_compact_fc1(576, 5120, 6, 384) is True)

# ---------------------------------------------------------------- deepgemm_moe_permute dispatch
def permute_runner(path, flags, fp_mode):
    log = Log()
    g = dict(torch=torch, os=os, mk=MK, count_expert_num_tokens=lambda ids, E, m: log.append(("count", E)) or
             torch.zeros(E, dtype=torch.int32),
             get_mk_alignment_for_contiguous_layout=lambda: (128, 128))
    exec("def round_up(x, y):\n    return ((x + y - 1) // y) * y\n", g)
    for src in extract(os.path.join(BASE, DGUTL), {"compute_aligned_M_and_alignment", "dsv41_fp_eligible"}).values():
        exec(src, g)
    g.update(_DSV41_FP_MODE=fp_mode, _DSV41_FP_SLOTS=1024, _DSV41_FP_FILL_ROWS=1024)
    g["ep_scatter"] = lambda **kw: log.append(("ep_scatter", kw["align_m"], kw["recv_topk"].shape[1]))
    g["dsv41_fused_permute"] = lambda aq, s, ids, E, out, M_sum, al: log.append(("fused", E, ids.shape[1], M_sum, al)) \
        or (out, None, None, None, None)
    g["_dsv41_fp_verify"] = lambda *a: log.append(("verify", a[3]))
    g.update(flags)
    exec(extract(path, {"deepgemm_moe_permute"})["deepgemm_moe_permute"], g)
    return g["deepgemm_moe_permute"], log


def call_permute(fn, E, topk, M, align, expert_map=None, ids_dtype=torch.int32):
    H = 5120
    from_m = -(-(M * topk + min(M * topk, E) * (align - 1)) // align) * align
    aq = torch.zeros(M, H, dtype=torch.float8_e4m3fn)
    return fn(aq=aq, aq_scale=torch.zeros(M, H // 128), topk_ids=torch.zeros(M, topk, dtype=ids_dtype),
              local_num_experts=E, expert_map=expert_map, expert_tokens_meta=None,
              aq_out=torch.zeros(from_m, H, dtype=torch.float8_e4m3fn), alignment_m=align,
              return_expert_ends=True)


def r2_view(mode):
    def perm_eligible(*a, **k):
        saved, R2DM.PERM = R2DM.PERM, mode
        try:
            return R2DM.perm_eligible(*a, **k)
        finally:
            R2DM.PERM = saved
    return types.SimpleNamespace(perm_eligible=perm_eligible, DRAFT_COMPACT_GEOMETRY=R2DM.DRAFT_COMPACT_GEOMETRY)


R_PERM, R_VER = r2_view("1"), r2_view("verify")
patched_utl, base_utl = os.path.join(INST, DGUTL), os.path.join(BASE, DGUTL)
cases = [(128, 3, M, 32) for M in (1, 2, 5, 16, 96, 160)] + [(128, 3, 200, 128), (128, 3, 400, 128)] + \
        [(384, 6, M, 32) for M in (1, 5, 96)] + [(384, 6, 200, 128)]
for fp_mode in ("0", "1"):
    bfn, blog = permute_runner(base_utl, {}, fp_mode)
    for label, flags in (("off", dict(_R2DM_PERM="", _R2DM_FC1=False)),
                         ("perm", dict(_R2DM_PERM="1", _R2DM_FC1=False, _r2dm=R_PERM)),
                         ("permverify", dict(_R2DM_PERM="verify", _R2DM_FC1=False, _r2dm=R_VER))):
        pfn, plog = permute_runner(patched_utl, flags, fp_mode)
        good = True
        for c in cases:
            blog.clear(), plog.clear()
            call_permute(bfn, *c)
            call_permute(pfn, *c)
            E, topk, M, al = c
            drafter_m32 = (E, al) == (128, 32)
            if label == "off" or not drafter_m32:
                want = list(blog)
            elif label == "perm":
                want = [("fused", 128, 3, plog[0][3] if plog else None, 32)]
            else:
                want = list(blog) + [("verify", 128)]
            if list(plog) != want:
                good = False
                print("   ", label, fp_mode, c, list(plog), want)
        check(f"permute dispatch DSV41_MOE_FUSED_PERMUTE={fp_mode} G17={label}: drafter M32 -> "
              f"{'stock' if label == 'off' else 'fused' if label == 'perm' else 'stock + verify'}; "
              f"128-row drafter and target unchanged", good)
pfn, plog = permute_runner(patched_utl, dict(_R2DM_PERM="1", _R2DM_FC1=False, _r2dm=R_PERM), "1")
call_permute(pfn, 128, 3, 5, 32, expert_map=torch.zeros(128, dtype=torch.int32))
check("perm: expert_map present -> stock (gate)", [x[0] for x in plog] == ["count", "ep_scatter"])
plog.clear()
call_permute(pfn, 128, 3, 5, 32, ids_dtype=torch.int64)
check("perm: int64 topk ids accepted (as the target gate)", [x[0] for x in plog] == ["fused"])

# ---------------------------------------------------------------- drafter_compact_eligible + geometry


def method(E, topk, unpadded, tp=8, ep=False, w13=384, act=MoEActivation.SILU):
    par = types.SimpleNamespace(tp_size=tp, use_ep=ep, enable_eplb=False)
    moe = types.SimpleNamespace(num_experts=E, experts_per_token=topk, intermediate_size_per_partition_unpadded=unpadded,
                                moe_parallel_config=par, tp_shard_with_padding=False, has_bias=False,
                                is_lora_enabled=False, activation=act)
    return types.SimpleNamespace(moe=moe, w13_intermediate_size=w13)


R2DM.FC1 = True
f = R2DM.drafter_compact_eligible
S = MoEActivation.SILU
check("drafter_compact_eligible: drafter geometry -> True", f(method(128, 3, 288), 128, 5120, 384, True, True, S))
check("drafter_compact_eligible: target / EP / unpadded!=288 / non-SM121 / non-DeepGEMM / target-gate-fired / "
      "non-SiLU -> False",
      not any([f(method(384, 6, 288), 384, 5120, 384, True, True, S),
               f(method(128, 3, 288, ep=True), 128, 5120, 384, True, True, S),
               f(method(128, 3, 256), 128, 5120, 384, True, True, S),
               f(method(128, 3, 288), 128, 5120, 384, False, True, S),
               f(method(128, 3, 288), 128, 5120, 384, True, False, S),
               f(method(128, 3, 288, w13=288), 128, 5120, 384, True, True, S),
               f(method(128, 3, 288, tp=4), 128, 5120, 384, True, True, S),
               f(method(128, 3, 288, act=MoEActivation.SITU), 128, 5120, 384, True, True, S)]))
_saved_compact = os.environ.pop("DSV41_MOE_COMPACT", None)
no_compact = f(method(128, 3, 288), 128, 5120, 384, True, True, S)
os.environ["DSV41_MOE_COMPACT"] = "0"
no_compact = no_compact or f(method(128, 3, 288), 128, 5120, 384, True, True, S)
os.environ["DSV41_MOE_COMPACT"] = _saved_compact
check("drafter_compact_eligible: DSV41_MOE_COMPACT unset or 0 -> False (the compact SiLU writer asserts it == '1')",
      not no_compact)
# with DSV41_MOE_COMPACT != 1 the compact drafter would hit _act_mul_quant's assert: shown on the patched apply
os.environ["DSV41_MOE_COMPACT"] = "0"
_log = run(patched_apply, ON, 128, 3, 576, 5)
os.environ["DSV41_MOE_COMPACT"] = _saved_compact
check("documented: compact drafter weights with DSV41_MOE_COMPACT=0 would assert in _act_mul_quant (why the gate "
      "requires it)", [x[0] for x in _log][-1] == "AssertionError", str(_log[-1:]))
R2DM.FC1 = False
check("drafter_compact_eligible: flag off -> False", not f(method(128, 3, 288), 128, 5120, 384, True, True, S))
g = {"os": os}
src = extract(os.path.join(INST, DRM32), {"geometry"})["geometry"]
g.update(DRAFT_GEOMETRY=(768, 5120, 3, 128), _R2DM_FC1=True, _r2dm=R2DM)
exec(src, g)
g2 = dict(g, _R2DM_FC1=False)
exec(src, g2)
check("dsv41_draft_m32.geometry: on accepts 768 and 576; off accepts 768 only",
      g["geometry"](576, 5120, 3, 128) and g["geometry"](768, 5120, 3, 128)
      and not g2["geometry"](576, 5120, 3, 128) and g2["geometry"](768, 5120, 3, 128)
      and not g["geometry"](576, 5120, 6, 384))

# ---------------------------------------------------------------- sizes / slots
ok = True
for M in range(1, 161):
    M_sum = -(-(3 * M + min(3 * M, 128) * 31) // 32) * 32
    ok &= 3 * M <= 1024 and M_sum % 4 == 0 and M_sum * 384 <= M_sum * max(288, 5120) and M_sum * 576 <= M_sum * 5120
check("sizes M=1..160: slots M*3 <= 1024 (_DSV41_FP_SLOTS), mn_aligned == M_sum, compact FC1 out and FC2 input "
      "fit the existing workspaces", ok)
shutil.rmtree(INST)
print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
