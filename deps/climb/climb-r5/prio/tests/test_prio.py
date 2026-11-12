"""r5-prio CPU tests. Run on 512S1:  cd <kit> && ~/exo-venv313/bin/python3 -m pytest -q tests/test_prio.py

What is covered without a GPU:
  * anchors apply to the live base files with the exact counts; patched sha == KIT.json;
  * flag-off identity: every patched file, with the r5-prio flags constant-folded to False, is AST-identical to
    the live file (so with the flags unset the executed code is the live code);
  * dynamic flag-off identity: the patched functions with the flags off issue the same kernels / streams /
    events in the same order as the live functions (fake CUDA, tests/fakecuda.py);
  * flag-on correctness: a vector-clock race checker (stream order + event record/wait + wait_stream) finds
    no read-before-write and no un-joined side stream, for CMPPRIO prio/serial (QFUSE and plain shard paths),
    SHARED and ENGRAM_HOIST; the priority / stream placement is the intended one; fall-backs fire where the
    lever must not apply;
  * must-fail mutants (a missing join in each lever) are caught by the checker;
  * installer: install / idempotent install / status / uninstall on a temp tree; refuses a wrong base without
    writing anything.
"""
from __future__ import annotations

import __future__
import ast
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import textwrap
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
KITDIR = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import fakecuda  # noqa: E402

BASE = os.path.join(KITDIR, "base", "vllm")
MODSRC = os.path.join(KITDIR, "files", "vllm", "models", "deepseek_v41", "nvidia", "dsv41_prio.py")
REF_MSU = os.path.join(KITDIR, "base", "ref_multi_stream_utils.py")
FLAGS = {"_DSV41_PRIO_CMP", "_DSV41_PRIO_SHARED", "_DSV41_PRIO_EH"}
PRIO_ENV = ("DSV41_PRIO_CMP", "DSV41_PRIO_SHARED", "DSV41_PRIO_ENGRAM_HOIST", "DSV41_PRIO_ENGRAM_HOIST_AT",
            "DSV41_PRIO_HI", "DSV41_PRIO_ENGRAM_MAX_TOKENS")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


INST = _load("r5prio_install", os.path.join(KITDIR, "install.py"))
KIT = json.load(open(os.path.join(KITDIR, "KIT.json")))


def base_text(rel):
    return open(os.path.join(BASE, rel)).read()


def patched_text(rel):
    return INST.patched(base_text(rel), rel)


# ------------------------------------------------------------------------------------------- static checks
def test_anchor_counts_and_shas():
    for rel in INST.FILES:
        src = base_text(rel)
        f = INST.kit_file(rel)
        assert hashlib.sha256(src.encode()).hexdigest() == f["base_sha256"], rel
        new = INST.patched(src, rel)
        ast.parse(new)
        assert hashlib.sha256(new.encode()).hexdigest() == f["patched_sha256"], rel
        # every changed line carries the tag; nothing untagged was added or removed
        b, p = src.splitlines(), new.splitlines()
        untagged = [ln for ln in p if INST.TAG not in ln]
        removed = [ln for ln in b if ln not in set(p)]
        if rel.endswith("engram.py"):
            assert removed == ["            kv = self.wkv(self.embed(hash_ids).flatten(-2))"], removed
        else:
            assert removed == [], (rel, removed)
        # untagged lines of the patched file are the base lines in order (plus blank lines between helper defs)
        assert [ln for ln in untagged if ln.strip()] == [ln for ln in b if ln.strip() and ln not in removed], rel
    assert hashlib.sha256(open(MODSRC, "rb").read()).hexdigest() == INST.kit_file(INST.REL_MOD)["kit_sha256"]


class _Fold(ast.NodeTransformer):
    """Constant-fold the r5-prio flags to False and drop the kit's flag definitions and helper defs."""

    def visit_Name(self, node):
        if node.id in FLAGS and isinstance(node.ctx, ast.Load):
            return ast.copy_location(ast.Constant(False), node)
        return node

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Not) and isinstance(node.operand, ast.Constant):
            return ast.copy_location(ast.Constant(not node.operand.value), node)
        return node

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        vals = node.values
        if isinstance(node.op, ast.And):
            if any(isinstance(v, ast.Constant) and not v.value for v in vals):
                return ast.copy_location(ast.Constant(False), node)
            vals = [v for v in vals if not (isinstance(v, ast.Constant) and v.value)]
        else:
            if any(isinstance(v, ast.Constant) and v.value for v in vals):
                return ast.copy_location(ast.Constant(True), node)
            vals = [v for v in vals if not (isinstance(v, ast.Constant) and not v.value)]
        if not vals:
            return ast.copy_location(ast.Constant(isinstance(node.op, ast.And)), node)
        if len(vals) == 1:
            return vals[0]
        node.values = vals
        return node

    def visit_IfExp(self, node):
        self.generic_visit(node)
        if isinstance(node.test, ast.Constant):
            return node.body if node.test.value else node.orelse
        return node

    def visit_If(self, node):
        self.generic_visit(node)
        if isinstance(node.test, ast.Constant):
            keep = node.body if node.test.value else node.orelse
            return keep or None
        return node

    def visit_Assign(self, node):
        self.generic_visit(node)
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            n = node.targets[0].id
            if n in FLAGS:
                return None
            if n.startswith("_dsv41_") and isinstance(node.value, ast.Constant) and node.value.value is None:
                return None
        return node

    def visit_FunctionDef(self, node):
        if node.name.startswith("_dsv41_prio"):
            return None
        self.generic_visit(node)
        return node


def _folded_dump(src):
    return ast.dump(_Fold().visit(ast.parse(src)))


def test_flag_off_is_ast_identical():
    for rel in INST.FILES:
        assert _folded_dump(patched_text(rel)) == _folded_dump(base_text(rel)), rel


def test_flag_lines_default_off(monkeypatch):
    for k in PRIO_ENV:
        monkeypatch.delenv(k, raising=False)
    for rel in INST.FILES:
        for ln in patched_text(rel).splitlines():
            if ln.startswith("_DSV41_PRIO") and "=" in ln:
                name, expr = ln.split(INST.TAG)[0].split("=", 1)
                env_name = expr.split('get("')[1].split('"')[0]
                g = {"os": os}
                for val, want in ((None, False), ("", False), ("0", False), ("off", False), ("false", False),
                                  ("1", True), ("prio", True), ("serial", True)):
                    if val is None:
                        monkeypatch.delenv(env_name, raising=False)
                    else:
                        monkeypatch.setenv(env_name, val)
                    assert eval(expr.strip(), g) is want, (rel, name, val)


# ---------------------------------------------------------------------------------------- fake vllm world
@pytest.fixture
def world(monkeypatch):
    saved = []
    w = fakecuda.install(saved)
    for k in PRIO_ENV:
        monkeypatch.delenv(k, raising=False)
    mods = {}

    def mk(name, **attrs):
        m = types.ModuleType(name)
        m.__path__ = []
        for k, v in attrs.items():
            setattr(m, k, v)
        mods[name] = m
        return m

    for n in ("vllm", "vllm.models", "vllm.models.deepseek_v41", "vllm.models.deepseek_v41.nvidia", "vllm.utils",
              "vllm.compilation"):
        mk(n)
    msu = _load("vllm.utils.multi_stream_utils", REF_MSU)
    mods["vllm.utils.multi_stream_utils"] = msu

    def all_gather(x, dim=-1):
        out = fakecuda.T("ag_out", shape=(x.shape[0], x.shape[-1] * 8) if dim in (-1, 1) else x.shape)
        fakecuda.WORLD[0].kernel("ag", reads=[x], writes=[out])
        return out

    mk("vllm.distributed", tensor_model_parallel_all_gather=all_gather)

    class _BC:
        active = False

        @classmethod
        def is_active(cls):
            return cls.active

    mk("vllm.compilation.breakable_cudagraph", BreakableCUDAGraphCapture=_BC)
    r3 = mk("vllm.models.deepseek_v41.nvidia.dsv41_r3_dense", QKV_MAX_M=8,
            _num_tokens=lambda x: x.shape[0])

    def qkv_a_sharded(attn, x):  # == the live function past the build step
        st = getattr(attn, "_dsv41_qkv_shard", None)
        if not st:
            return None
        M = x.shape[0]
        if M == 0 or M > r3.QKV_MAX_M:
            return None
        return st(x)

    r3.qkv_a_sharded = qkv_a_sharded
    for n, m in mods.items():
        monkeypatch.setitem(sys.modules, n, m)
    for n in [k for k in sys.modules if k.endswith("dsv41_prio")]:
        monkeypatch.delitem(sys.modules, n, raising=False)
    w.mods = mods
    w.monkeypatch = monkeypatch
    yield w
    fakecuda.uninstall(saved)


def second_world(world):
    """A fresh fake-CUDA world sharing the fixture's monkeypatch / module stubs (for live-vs-patched runs)."""
    w2 = fakecuda.World()
    fakecuda.WORLD[0] = w2
    torch = sys.modules["torch"]
    torch.cuda.stream, torch.cuda.current_stream = w2.stream, w2.current_stream
    torch.cuda.is_current_stream_capturing = lambda: w2.capturing
    w2.monkeypatch, w2.mods = world.monkeypatch, world.mods
    return w2


def load_prio(w, **env):
    for k, v in env.items():
        w.monkeypatch.setenv(k, v)
    name = "vllm.models.deepseek_v41.nvidia.dsv41_prio"
    m = _load(name, MODSRC)
    w.monkeypatch.setitem(sys.modules, name, m)
    setattr(w.mods["vllm.models.deepseek_v41.nvidia"], "dsv41_prio", m)
    return m


def _func(src, qual, ns):
    """exec one function (Class.method or name) from a source file into ns, annotations unevaluated."""
    tree = ast.parse(src)
    cls, _, fn = qual.rpartition(".")
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and cls and node.name == cls:
            for f in node.body:
                if isinstance(f, ast.FunctionDef) and f.name == fn:
                    break
            else:
                continue
            break
    else:
        f = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fn)
    code = textwrap.dedent(ast.get_source_segment(src, f))
    flags = __future__.annotations.compiler_flag
    exec(compile(code, qual, "exec", flags=flags, dont_inherit=True), ns)
    return ns[fn]


# --------------------------------------------------------------------------------------------- CMPPRIO
class TorchShim:
    float32 = "f32"

    def __init__(self, w):
        self.w = w
        self.cuda = sys.modules["torch"].cuda

    def mm(self, a, b, out_dtype=None):
        out = fakecuda.T("kv_score")
        self.w.kernel("cmp_mm", reads=[a, b], writes=[out])
        return out

    def empty_like(self, x):
        return fakecuda.T("empty", shape=x.shape)


def build_attn(w, rel_variant, flag_on, qfuse=None, shard=True):
    """Fake DeepseekV4Attention with the live (or patched) _run_parallel_input_projections and
    _fused_wqa_wkv_gemm and a QkvAShard whose __call__ is the live (or patched) one."""
    src_attn = base_text("models/deepseek_v41/attention.py") if rel_variant == "base" else \
        patched_text("models/deepseek_v41/attention.py")
    src_r3 = base_text("models/deepseek_v41/nvidia/dsv41_r3_dense.py") if rel_variant == "base" else \
        patched_text("models/deepseek_v41/nvidia/dsv41_r3_dense.py")
    msu = sys.modules["vllm.utils.multi_stream_utils"]
    envs = types.SimpleNamespace(VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD=1024)
    ns = {"torch": TorchShim(w), "envs": envs, "execute_in_parallel": msu.execute_in_parallel,
          "_DSV41_PRIO_CMP": flag_on, "_DSV41_IDX_WPROJ": False, "_DSV41_QKV_A_SHARD": True}
    run = _func(src_attn, "DeepseekV4Attention._run_parallel_input_projections", ns)
    fused = _func(src_attn, "DeepseekV4Attention._fused_wqa_wkv_gemm", ns)

    def quantize(x, K):
        xq, xs = fakecuda.T("xq"), fakecuda.T("xs")
        w.kernel("quant", reads=[x], writes=[xq, xs])
        return xq, xs, x.shape

    r3ns = {"R2_QFUSE": qfuse is not None, "_DSV41_PRIO_CMP": flag_on, "_quantize_act": quantize, "torch": None}
    call = _func(src_r3, "QkvAShard.__call__", r3ns)
    if rel_variant != "base":
        _func(src_r3, "_dsv41_prio_enter", r3ns)
        _func(src_r3, "_dsv41_prio_leave", r3ns)

    class Shard:
        K, N, n = 64, 1792, 224

        def __init__(self):
            self._r2 = None
            if qfuse is not None:
                class R2:
                    def fused(_s, x):
                        if qfuse == "declines":
                            return None
                        y = fakecuda.T("y_fused")
                        w.kernel("qfuse", reads=[x], writes=[y])
                        return y
                self._r2 = R2()

        def gemm(self, xq, xs):
            y = fakecuda.T("y")
            w.kernel("gemm", reads=[xq, xs], writes=[y])
            return y

    Shard.__call__ = call

    class Lin:
        def __init__(self, tag):
            self.tag = tag
            self.weight = types.SimpleNamespace(T=fakecuda.T("W_" + tag))

        def __call__(self, x):
            out = fakecuda.T(self.tag)
            w.kernel(self.tag, reads=[x], writes=[out])
            return out, None

    class Attn:
        prefix = "layers.2.attn"

        def __init__(self):
            self.aux_stream_list = [fakecuda.Stream(priority=0, name=f"aux{i}") for i in range(3)]
            self.ln_events = [fakecuda.Event() for _ in range(4)]
            self.compressor = types.SimpleNamespace(fused_wkv_wgate=Lin("cmpw"))
            self.indexer = types.SimpleNamespace(weights_proj=Lin("idx_wproj"))
            self.fused_wqa_wkv = Lin("stock_qkv")
            self._dsv41_qkv_shard = Shard() if shard else False

    Attn._run_parallel_input_projections = run
    Attn._fused_wqa_wkv_gemm = fused
    return Attn()


def run_cmp(w, attn, M=4):
    h = fakecuda.T("hidden", shape=(M, 64))
    w.kernel("producer", writes=[h])
    out = attn._run_parallel_input_projections(h)
    assert len(out) == 3
    w.kernel("consume", reads=[o for o in out if o is not None])
    w.assert_no_races()
    w.join_check()
    return out


def norm_log(w):
    """Kernel/event sequence with stream ids renumbered by first appearance (names of events dropped)."""
    ids = {}
    out = []
    for e in w.log:
        sid = ids.setdefault(e[2], len(ids))
        out.append((e[0], e[1] if e[0] == "k" else "", sid, e[3]))
    return out


@pytest.mark.parametrize("qfuse", [None, "fused", "declines"])
@pytest.mark.parametrize("M", [1, 4, 8, 9])
def test_cmp_flag_off_matches_live(world, qfuse, M):
    a = build_attn(world, "base", False, qfuse)
    run_cmp(world, a, M)
    live = norm_log(world)
    w2 = second_world(world)
    b = build_attn(w2, "patched", False, qfuse)
    run_cmp(w2, b, M)
    assert norm_log(w2) == live
    assert not any(k.endswith("dsv41_prio") for k in sys.modules)


@pytest.mark.parametrize("qfuse", [None, "fused", "declines"])
@pytest.mark.parametrize("M", [1, 2, 4, 6, 8])
def test_cmp_prio(world, qfuse, M):
    prio = load_prio(world, DSV41_PRIO_CMP="prio")
    assert prio.CMP_MODE == "prio"
    a = build_attn(world, "patched", True, qfuse)
    run_cmp(world, a, M)
    gemm = "qfuse" if qfuse == "fused" else "gemm"
    sid, pr, _, _ = world.kentry(gemm)
    assert pr == fakecuda.Stream._greatest and sid != world.default.id
    if qfuse != "fused":
        assert world.kentry("quant")[0] == sid
    assert world.kentry("ag")[0] == world.default.id
    assert world.kentry("cmp_mm")[0] == a.aux_stream_list[0].id
    assert world.before(gemm, "ag")
    assert not world.before(gemm, "cmp_mm") and not world.before("cmp_mm", gemm)  # still concurrent
    assert prio.COUNTS["cmp_prio"] == 1 and prio.COUNTS["cmp_fallback"] == 0


@pytest.mark.parametrize("qfuse", [None, "fused"])
@pytest.mark.parametrize("M", [1, 4, 8])
def test_cmp_serial(world, qfuse, M):
    prio = load_prio(world, DSV41_PRIO_CMP="serial")
    a = build_attn(world, "patched", True, qfuse)
    run_cmp(world, a, M)
    gemm = "qfuse" if qfuse == "fused" else "gemm"
    assert world.kentry(gemm)[0] == world.default.id  # no stream change
    assert world.kentry("cmp_mm")[0] == a.aux_stream_list[0].id
    assert world.before(gemm, "cmp_mm")  # the compressor waits for the shard GEMM
    assert not world.before("ag", "cmp_mm")  # ... but not for its all-gather
    assert world.kentry("idx_wproj")[0] == a.aux_stream_list[1].id
    assert not world.before(gemm, "idx_wproj")
    assert prio.COUNTS["cmp_serial"] == 1


@pytest.mark.parametrize("mode", ["prio", "serial"])
def test_cmp_fallbacks(world, mode):
    prio = load_prio(world, DSV41_PRIO_CMP=mode)
    a = build_attn(world, "patched", True)
    run_cmp(world, a, M=9)  # > QKV_MAX_M: stock GEMM, live execute_in_parallel
    assert prio.COUNTS["cmp_fallback"] == 1 and "gemm" not in [k for k, _ in world.kernels()]
    b = build_attn(world, "patched", True, shard=False)
    run_cmp(world, b, M=4)
    assert prio.COUNTS["cmp_fallback"] == 2
    c = build_attn(world, "patched", True)
    world.capturing = True  # first use inside a capture: no event can be created -> live path
    run_cmp(world, c, M=4)
    world.capturing = False
    assert prio.COUNTS["cmp_fallback"] == 3


def test_cmp_unknown_mode_is_off(world):
    prio = load_prio(world, DSV41_PRIO_CMP="bogus")
    assert prio.CMP_MODE == ""
    a = build_attn(world, "patched", True)
    run_cmp(world, a, M=4)
    assert prio.COUNTS["cmp_fallback"] == 1


def test_cmp_prio_mutant_missing_join_is_caught(world):
    prio = load_prio(world, DSV41_PRIO_CMP="prio")

    def bad_leave(tok, y):  # mutant: no main.wait_event(done)
        tok.done.record(tok.hi)
        tok.ctx.__exit__(None, None, None)
        tok.ctx = None
        tok.used = True
        return y

    world.monkeypatch.setattr(prio, "cmp_leave", bad_leave)
    a = build_attn(world, "patched", True)
    with pytest.raises(fakecuda.Race):
        run_cmp(world, a, M=4)


def test_cmp_exception_restores_stream(world):
    prio = load_prio(world, DSV41_PRIO_CMP="prio")
    a = build_attn(world, "patched", True)

    def boom(xq, xs):
        raise ValueError("gemm failed")

    a._dsv41_qkv_shard.gemm = boom
    h = fakecuda.T("hidden", shape=(4, 64))
    with pytest.raises(ValueError):
        a._run_parallel_input_projections(h)
    assert world.current_stream() is world.default
    assert prio.ARM.st is None


# ---------------------------------------------------------------------------------------------- SHARED
def load_shared_experts(w, flag_on, variant="patched"):
    rel = "model_executor/layers/fused_moe/runner/shared_experts.py"
    src = patched_text(rel) if variant == "patched" else base_text(rel)
    aux = fakecuda.Stream(priority=0, name="global_aux")
    stubs = {
        "vllm.envs": types.SimpleNamespace(VLLM_DISABLE_SHARED_EXPERTS_STREAM=False,
                                           VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=256),
        "vllm.logger": types.SimpleNamespace(init_logger=lambda n: types.SimpleNamespace(
            debug_once=lambda *a, **k: None, info=lambda *a, **k: None, warning=lambda *a, **k: None)),
        "vllm.model_executor.layers.fused_moe.runner.shared_fold_glue": types.SimpleNamespace(
            maybe_fold_forward=lambda layer, x: None),
        "vllm.model_executor.layers.fused_moe.config": types.SimpleNamespace(FusedMoEConfig=object),
        "vllm.platforms": types.SimpleNamespace(current_platform=types.SimpleNamespace(is_cuda_alike=lambda: True)),
        "vllm.utils.torch_utils": types.SimpleNamespace(aux_stream=lambda: aux,
                                                        current_stream=lambda: w.current_stream()),
        "vllm.v1.worker.ubatching": types.SimpleNamespace(dbo_current_ubatch_id=lambda: 0),
    }
    for n in ("vllm.model_executor", "vllm.model_executor.layers", "vllm.model_executor.layers.fused_moe",
              "vllm.model_executor.layers.fused_moe.runner", "vllm.v1", "vllm.v1.worker"):
        if n not in sys.modules:
            pk = types.ModuleType(n)
            pk.__path__ = []
            w.monkeypatch.setitem(sys.modules, n, pk)
    for n, m in stubs.items():
        w.monkeypatch.setitem(sys.modules, n, m)
    w.monkeypatch.setenv("DSV41_PRIO_SHARED", "1" if flag_on else "")
    mod = types.ModuleType("se_" + variant)
    exec(compile(src, rel, "exec"), mod.__dict__)
    return mod, aux


class _Down:
    def __init__(self, reduce):
        self.reduce_results = reduce
        self.tp_size = 8


def make_layer(w, reduce=False):
    import torch

    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.down_proj = _Down(reduce)

        def forward(self, x):
            g = fakecuda.T("gate_up")
            w.kernel("n640", reads=[x], writes=[g])
            a = fakecuda.T("act")
            w.kernel("act_and_mul", reads=[g], writes=[a])
            o = fakecuda.T("shared_out")
            w.kernel("down", reads=[a], writes=[o])
            return o

    return MLP()


def moe_cfg():
    return types.SimpleNamespace(moe_parallel_config=types.SimpleNamespace(
        enable_eplb=False, all2all_backend="x", use_fi_nvl_two_sided_kernels=False))


def run_shared(w, se, mutant=False):
    x = fakecuda.T("ffn_in", shape=(4, 64))
    w.kernel("mhc_pre", writes=[x])
    assert se.maybe_forward_async(x)
    r = fakecuda.T("routed")
    w.kernel("fc1_fc2", reads=[x], writes=[r])
    if not mutant:
        se.wait()
    out = se.output
    w.kernel("gather_add", reads=[r, out])
    w.assert_no_races()
    w.join_check()


@pytest.mark.parametrize("flag_on", [False, True])
def test_shared_stream(world, flag_on):
    if flag_on:
        prio = load_prio(world, DSV41_PRIO_SHARED="1")
    mod, aux = load_shared_experts(world, flag_on)
    se = mod.SharedExperts(make_layer(world), moe_cfg(), False, lambda: False)
    if flag_on:
        assert se._stream is not aux and se._stream.priority == fakecuda.Stream._greatest
        se2 = mod.SharedExperts(make_layer(world), moe_cfg(), False, lambda: False)
        assert se2._stream is se._stream and prio.COUNTS["shared_streams"] == 2
    else:
        assert se._stream is aux
        assert not any(k.endswith("dsv41_prio") for k in sys.modules)
    run_shared(world, se)
    for k in ("n640", "act_and_mul", "down"):
        assert world.kentry(k)[0] == se._stream.id
    assert world.kentry("gather_add")[0] == world.default.id


def test_shared_flag_off_matches_live(world):
    mod, _ = load_shared_experts(world, False, "base")
    run_shared(world, mod.SharedExperts(make_layer(world), moe_cfg(), False, lambda: False))
    live = norm_log(world)
    w2 = second_world(world)
    mod2, _ = load_shared_experts(w2, False, "patched")
    run_shared(w2, mod2.SharedExperts(make_layer(w2), moe_cfg(), False, lambda: False))
    assert norm_log(w2) == live


def test_shared_refuses_reducing_mlp(world):
    load_prio(world, DSV41_PRIO_SHARED="1")
    mod, aux = load_shared_experts(world, True)
    se = mod.SharedExperts(make_layer(world, reduce=True), moe_cfg(), False, lambda: False)
    assert se._stream is aux


def test_shared_mutant_missing_wait_is_caught(world):
    load_prio(world, DSV41_PRIO_SHARED="1")
    mod, _ = load_shared_experts(world, True)
    se = mod.SharedExperts(make_layer(world), moe_cfg(), False, lambda: False)
    with pytest.raises(fakecuda.Race):
        run_shared(world, se, mutant=True)


# ---------------------------------------------------------------------------------------- ENGRAM HOIST
def build_engram_model(w, variant, flag_on, n_layers=16, eng=(1, 14)):
    import torch

    src = patched_text("models/deepseek_v41/common/engram.py") if variant == "patched" else \
        base_text("models/deepseek_v41/common/engram.py")

    class _Post:
        def __getitem__(self, grid):
            def launch(hidden, kv, qw, kw, mask, out, *a, **k):
                w.kernel("post", reads=[hidden, kv], writes=[out])
            return launch

    ns = {"_DSV41_PRIO_EH": flag_on, "torch": TorchShim(w), "_fused_engram_post_wkv_kernel": _Post(),
          "triton": types.SimpleNamespace(next_power_of_2=lambda d: 16)}
    fwd = _func(src, "Engram.forward", ns)

    class QM:
        def apply(self, layer, x, bias=None):
            xq = fakecuda.T("wkv_xq")
            w.kernel(f"wkv_quant{layer.idx}", reads=[x], writes=[xq])
            o = fakecuda.T("wkv_local", shape=(x.shape[0], 10))
            w.kernel(f"wkv_gemm{layer.idx}", reads=[xq, layer.weight], writes=[o])
            return o

    class ColumnParallelLinear:  # the live forward, verbatim in structure (vllm linear.py:603)
        def __init__(self, idx):
            self.idx, self.quant_method, self.bias, self.skip_bias_add = idx, QM(), None, False
            self.gather_output, self.tp_size, self.return_bias = True, 8, False
            self.weight = fakecuda.T(f"wkv_w{idx}")

        def forward(self, input_):
            bias = self.bias if not self.skip_bias_add else None
            output_parallel = self.quant_method.apply(self, input_, bias)
            if self.gather_output and self.tp_size > 1:
                from vllm.distributed import tensor_model_parallel_all_gather

                output = tensor_model_parallel_all_gather(output_parallel)
            else:
                output = output_parallel
            return output

        __call__ = forward

    class Engram:
        _batch = None
        use_sequence_parallel = False
        hc_mult, dim, eps, clamp_value = 4, 16, 1e-6, 1e-6

        def __init__(self, li, hidx):
            self.li, self.layer_hash_index = li, hidx
            self.embed_tokens = types.SimpleNamespace(tp_size=8)
            self.wkv = ColumnParallelLinear(li)
            self.q_weight, self.k_weight = fakecuda.T("qw"), fakecuda.T("kw")
            self.staged = fakecuda.T(f"staged{li}")
            w.kernel(f"lookup{li}", writes=[self.staged])

        def embed(self, hash_ids):
            rows = fakecuda.T(f"rows{self.li}", shape=(hash_ids.shape[0], 3, 8))
            w.kernel(f"rows_ag{self.li}", reads=[self.staged], writes=[rows])
            return rows

    Engram.forward = fwd

    class Layer(torch.nn.Module):
        def __init__(self, i):
            super().__init__()
            self.i = i
            self.engram = Engram(i, eng.index(i)) if i in eng else None

        def forward(self, x, positions, input_ids, pre_mix=None, post_mix=None, res_mix=None, residual=None,
                    engram_hashes=None, engram_mask=None, *, capture_previous_aux=False, mega_gate_metadata=None,
                    dsv41_sp=False):
            if self.engram is not None and engram_hashes is not None:
                hs = fakecuda.T(f"hc{self.i}", shape=(engram_hashes.shape[0], 4, 16))
                w.kernel(f"mhc_post{self.i}", reads=[x], writes=[hs])
                x = self.engram.forward(hs, engram_hashes[:, self.engram.layer_hash_index], engram_mask)
            y = fakecuda.T(f"x{self.i}", shape=x.shape)
            w.kernel(f"L{self.i}", reads=[x], writes=[y])
            return y

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer(i) for i in range(n_layers)])
            self.start_layer, self.end_layer = 0, n_layers

        def forward(self, x, hashes, sp=False, skip=()):
            for i, layer in enumerate(self.layers):
                if i in skip:
                    continue
                x = layer(x, None, None, None, None, None, None, hashes, None, dsv41_sp=sp)
            return x

    return Model()


def run_model(w, model, T=4, **kw):
    h = fakecuda.T("emb", shape=(T, 16))
    w.kernel("embed_tokens", writes=[h])
    hashes = fakecuda.T("hashes", shape=(T, 2))
    w.kernel("hash", writes=[hashes])
    out = model(h, hashes, **kw)
    w.kernel("head", reads=[out])
    w.assert_no_races()
    w.join_check()
    return out


def test_engram_flag_off_matches_live(world):
    run_model(world, build_engram_model(world, "base", False))
    live = norm_log(world)
    w2 = second_world(world)
    run_model(w2, build_engram_model(w2, "patched", False))
    assert norm_log(w2) == live


def test_engram_hoist(world):
    prio = load_prio(world, DSV41_PRIO_ENGRAM_HOIST="1")
    model = build_engram_model(world, "patched", True)
    prio.engram_install(model)
    assert prio.EH.on and sorted(prio.EH.plan) == [0, 13]
    run_model(world, model)
    assert prio.COUNTS["eh_hoist"] == 2 and prio.COUNTS["eh_take"] == 2 and prio.COUNTS["eh_stale"] == 0
    side = prio.EH.side
    assert side.priority == fakecuda.Stream._least
    for li, host in ((1, 0), (14, 13)):
        assert world.kentry(f"wkv_gemm{li}")[0] == side.id and world.kentry(f"wkv_quant{li}")[0] == side.id
        assert world.kentry(f"rows_ag{li}")[0] == world.default.id
        assert world.before(f"rows_ag{li}", f"L{host}")  # issued at the host layer start, on main
        assert world.kentry("ag")[0] == world.default.id
        # overlap: the main-stream layers between the hoist and the take do not wait for the GEMM
        for mid in range(host, li):
            assert not world.before(f"wkv_gemm{li}", f"L{mid}"), (li, mid)
        assert world.before(f"wkv_gemm{li}", f"post")  if li == 1 else True
    # the output all-gathers run on main, in layer order, after their GEMMs
    ags = [e for e in world.log if e[0] == "k" and e[1] == "ag"]
    assert len(ags) == 2 and all(e[2] == world.default.id for e in ags)
    assert world.before("wkv_gemm1", "ag", 0, 0) and world.before("wkv_gemm14", "ag", 0, 1)
    assert not world.before("wkv_gemm14", "ag", 0, 0)


def test_engram_hoist_kernel_set_matches_live(world):
    """Same kernels on the same operands (multiset of names), only issue point / stream differ."""
    run_model(world, build_engram_model(world, "base", False))
    live = sorted(k for k, _ in world.kernels() if not k.startswith("lookup"))
    w2 = second_world(world)
    prio = load_prio(w2, DSV41_PRIO_ENGRAM_HOIST="1")
    m = build_engram_model(w2, "patched", True)
    prio.engram_install(m)
    run_model(w2, m)
    assert sorted(k for k, _ in w2.kernels() if not k.startswith("lookup")) == live


@pytest.mark.parametrize("case", ["too_many_tokens", "sp"])
def test_engram_hoist_skips(world, case):
    prio = load_prio(world, DSV41_PRIO_ENGRAM_HOIST="1", DSV41_PRIO_ENGRAM_MAX_TOKENS="8")
    model = build_engram_model(world, "patched", True)
    prio.engram_install(model)
    if case == "too_many_tokens":
        run_model(world, model, T=9)
    else:
        run_model(world, model, sp=True)
    assert prio.COUNTS["eh_hoist"] == 0 and prio.COUNTS["eh_take"] == 0
    assert all(world.kentry(f"wkv_gemm{li}")[0] == world.default.id for li in (1, 14))


def test_engram_unconsumed_hoist_is_flushed_and_lever_off(world):
    prio = load_prio(world, DSV41_PRIO_ENGRAM_HOIST="1")
    model = build_engram_model(world, "patched", True)
    prio.engram_install(model)
    # skip engram layer 14 entirely so its hoist (issued at layer 13) is never taken
    run_model(world, model, skip=(14,))
    assert prio.COUNTS["eh_stale"] == 1 and not prio.EH.on
    run_model(world, model)  # lever off: live path, no hoist
    assert prio.COUNTS["eh_hoist"] == 2


def test_engram_install_refusals(world):
    prio = load_prio(world, DSV41_PRIO_ENGRAM_HOIST="1", DSV41_PRIO_ENGRAM_HOIST_AT="1:0,14:20,3:2")
    model = build_engram_model(world, "patched", True)
    model.layers[14].engram._batch = object()  # EngramBatch path -> not hoisted
    prio.engram_install(model)
    assert sorted(prio.EH.plan) == [0]  # 14:20 host > engram, 3:2 no engram, 14 batch path
    prio2 = load_prio(world, DSV41_PRIO_ENGRAM_HOIST="1")
    m2 = build_engram_model(world, "patched", True)
    import torch

    class Odd(torch.nn.Module):
        def forward(self, x, engram_hashes=None):
            return x

    m2.layers[0] = Odd()
    prio2.engram_install(m2)
    assert not prio2.EH.on


def test_engram_mutant_missing_join_is_caught(world):
    prio = load_prio(world, DSV41_PRIO_ENGRAM_HOIST="1")
    model = build_engram_model(world, "patched", True)
    prio.engram_install(model)
    orig = prio.engram_take

    def bad_take(engram, hash_ids=None):
        rec = prio.EH.pending.pop(id(engram), None)
        from vllm.distributed import tensor_model_parallel_all_gather
        return tensor_model_parallel_all_gather(rec.out)  # mutant: no main.wait_event(rec.done)

    for host, es in prio.EH.plan.items():
        for e in es:
            object.__setattr__(e, "_dsv41_prio_take", lambda h, _e=e: bad_take(_e, h))
    with pytest.raises(fakecuda.Race):
        run_model(world, model)
    assert orig is not None


def test_parse_at(world):
    prio = load_prio(world)
    assert prio.parse_at("1:0, 14:13") == [(1, 0), (14, 13)]
    assert prio.ENGRAM is False and prio.CMP_MODE == "" and prio.SHARED is False


# -------------------------------------------------------------------------------------------- installer
def _tree(tmp_path):
    root = tmp_path / "vllm"
    for rel in INST.FILES:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(os.path.join(BASE, rel), p)
    return root


def _run(root, *args):
    env = dict(os.environ, PRIO_VLLM=str(root))
    return subprocess.run([sys.executable, os.path.join(KITDIR, "install.py"), *args], env=env,
                          capture_output=True, text=True)


def test_installer_roundtrip(tmp_path):
    root = _tree(tmp_path)
    r = _run(root, "status")
    assert r.returncode == 0 and "CLEAN" in r.stdout, r.stdout + r.stderr
    r = _run(root, "install")
    assert r.returncode == 0, r.stderr
    for rel in INST.FILES:
        assert (root / rel).read_text() == patched_text(rel)
    assert (root / INST.REL_MOD).read_bytes() == open(MODSRC, "rb").read()
    r = _run(root, "install")
    assert r.returncode == 0 and r.stdout.count("already patched") == len(INST.FILES)
    r = _run(root, "status")
    assert r.returncode == 0 and "INSTALLED" in r.stdout, r.stdout
    r = _run(root, "dry")
    assert r.returncode == 0 and r.stdout.count(INST.TAG) == sum(patched_text(x).count(INST.TAG) for x in INST.FILES)
    r = _run(root, "uninstall")
    assert r.returncode == 0, r.stderr
    for rel in INST.FILES:
        assert (root / rel).read_text() == base_text(rel)
        assert not (root / (rel + INST.BAKSUF)).exists()
    assert not (root / INST.REL_MOD).exists()
    assert "CLEAN" in _run(root, "status").stdout


def test_installer_refuses_wrong_base_atomically(tmp_path):
    root = _tree(tmp_path)
    p = root / INST.FILES[-1]
    p.write_text(p.read_text() + "# drift\n")
    before = {rel: (root / rel).read_bytes() for rel in INST.FILES}
    r = _run(root, "install")
    assert r.returncode != 0 and "refusing" in (r.stdout + r.stderr)
    assert {rel: (root / rel).read_bytes() for rel in INST.FILES} == before
    assert not (root / INST.REL_MOD).exists()
    assert not any((root / (rel + INST.BAKSUF)).exists() for rel in INST.FILES)


def test_installer_uninstall_refuses_edited(tmp_path):
    root = _tree(tmp_path)
    assert _run(root, "install").returncode == 0
    p = root / INST.FILES[0]
    p.write_text(p.read_text() + "# later edit\n")
    r = _run(root, "uninstall")
    assert r.returncode != 0 and "refusing" in (r.stdout + r.stderr)
    assert "# later edit" in p.read_text()
    assert _run(root, "status").returncode == 2


def test_compose_check(tmp_path):
    root = _tree(tmp_path)
    p = root / INST.FILES[2]
    p.write_text("# another kit's header line  # r5-ffnfront\n" + p.read_text())
    r = _run(root, "compose-check")
    assert r.returncode == 0 and "NOT in KIT.json" in r.stdout and "anchors OK" in r.stdout


# --------------------------------------------------------------------------------------------- device test
def test_device_test_cpu_selftest():
    r = subprocess.run([sys.executable, os.path.join(KITDIR, "device_test.py"), "--cpu-selftest"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "cpu-selftest: OK" in r.stdout


def _load_device_test():
    spec = importlib.util.spec_from_file_location("prio_device_test", os.path.join(KITDIR, "device_test.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.mark.parametrize("case", ["build_none", "build_raised", "partial_m", "v2_off", "not_skinny"])
def test_device_test_b1_serving_qkv_gemm_must_be_v2(case):
    """Review B1: with DSV41_SKINNY_V2 on, a failed/None v2 build is a FAIL (exit 1), never a note."""
    dt = _load_device_test()
    v2 = types.SimpleNamespace(ok_a8=set(range(1, 9)))
    args = {"build_none": (True, "skinny", True, None, 8, ""),
            "build_raised": (True, "skinny", True, None, 8, "OSError('no .so')"),
            "partial_m": (True, "skinny", True, types.SimpleNamespace(ok_a8={1, 2, 3}), 8, ""),
            "v2_off": (False, "skinny", True, None, 8, ""),
            "not_skinny": (True, "cutlass", False, v2, 8, "")}[case]
    res = dt.skinny_serving_checks(*args)
    bad = [n for n, ok, _ in res if not ok]
    assert bad, res
    dt.FAILS.clear()
    for n, ok, d in res:
        dt.check(n, ok, d)
    assert dt.exit_code("PASS", dt.FAILS) == 1 and dt.exit_code("FAIL", dt.FAILS) == 1


def test_device_test_b1_good_v2_passes():
    dt = _load_device_test()
    res = dt.skinny_serving_checks(True, "skinny", True, types.SimpleNamespace(ok_a8=set(range(1, 9))), 8)
    assert all(ok for _, ok, _ in res) and len(res) == 3, res


def test_device_test_b1_no_v1_fallback_note():
    src = open(os.path.join(KITDIR, "device_test.py")).read()
    assert "the v1 skinny GEMM is under test" not in src
    assert "skinny_serving_checks(r3.R2_V2" in src
