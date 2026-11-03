#!/usr/bin/env python3
"""g130 boundary kit, CPU test. No GPU, no service contact: CUDA_VISIBLE_DEVICES is cleared before torch loads, Triton runs in
its interpreter. Passing proves my kernels follow torch CPU semantics on these rows. It does NOT prove CUDA bit-exactness.

  cd ~/dsv41-stage/g130/boundary && CUDA_VISIBLE_DEVICES= nice -n 19 timeout 600 ~/venvs/vllm032/bin/python tests/test_boundary_cpu.py

Sections:
  P  patcher on the base copies: anchors, flags-off AST difference, round trip
  I  installer on a scratch site: state reporting, exit codes, service check fails closed
  R  Lin.__init__ and kv_norm_of from test_boundary_gpu.py on the real checkpoint, against the real production loader, swizzle and RMSNorm
  S  swizzled scale slice == swizzle of the tile-aligned row range (real vllm function if importable)
  L  mk_local_pair (tiles 1 and n) vs the stock ops, injected ties, num_reqs 1..32
  W  mk_winner vs the stock ops, TP=8, injected cross-rank ties
  F  the real _sample_sequential_sharded source, flags off and fold on, 5 positions, TP=8 emulated, num_reqs 1..32
  C  ctxkv_precompute control flow with a fake exact runner: layer order, slices, fallbacks
"""
import ast
import contextlib
import importlib.util
import io
import json
import os
import shutil
import sys
import textwrap
import time
import types
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TRITON_INTERPRET"] = "1"
os.environ.update(DSV41_MK_FOLD="1", DSV41_CTXKV_FOLD="1")

import torch  # noqa: E402

KIT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KIT / "tools"))
import boundary_install as bi  # noqa: E402

torch.set_num_threads(2)
FAIL = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        FAIL.append(name)


def load_side(tiles="16"):
    os.environ["DSV41_MK_FOLD_TILES"] = tiles
    p = KIT / "site/vllm/models/deepseek_v41/nvidia/dsv41_boundary.py"
    spec = importlib.util.spec_from_file_location("dsv41_boundary_cpu", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


B = load_side()


# ---------------------------------------------------------------- P
def strip_new(tree, flagnames):
    """Remove module-level flag assigns and `if <flag...>` statements; reduce `range(0 if _mk_done else n_spec)` to the stock
    call; drop `_mk_done = False`. What is left must equal the base AST."""
    class T(ast.NodeTransformer):
        def visit_Assign(self, n):
            if isinstance(n.targets[0], ast.Name) and n.targets[0].id in flagnames | {"_mk_done"}:
                return None
            return n

        def visit_If(self, n):
            self.generic_visit(n)
            names = {x.id for x in ast.walk(n.test) if isinstance(x, ast.Name)}
            return None if names & flagnames else n

        def visit_Call(self, n):
            self.generic_visit(n)
            if isinstance(n.func, ast.Name) and n.func.id == "range" and n.args and isinstance(n.args[0], ast.IfExp):
                n.args = [n.args[0].orelse]
            return n

    return T().visit(tree)


flags = {"_DSV41_MK_FOLD", "_DSV41_CTXKV_FOLD"}
for rel, fn in ((bi.DSPARK, "dspark.py"), (bi.SPEC, "speculator.py")):
    base = (KIT / "base" / fn).read_text()
    check(f"P base sha {fn}", bi.sha(base.encode()).startswith(bi.BASE[rel]))
    new = bi.patched(rel, base)
    compile(new, fn, "exec")
    a = ast.dump(strip_new(ast.parse(new), flags))
    b = ast.dump(ast.parse(base))
    check(f"P flags-off AST equals base {fn}", a == b)


# ---------------------------------------------------------------- I
def tool(fn, *args):
    """(exit code, stdout) of a boundary_install function; ABORT arrives as SystemExit."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            rc = fn(*args)
    except SystemExit as e:
        return 1, buf.getvalue() + str(e.code)
    return rc, buf.getvalue()


scratch = KIT / "results" / "site-scratch"
shutil.rmtree(scratch, ignore_errors=True)
for rel, fn in ((bi.DSPARK, "dspark.py"), (bi.SPEC, "speculator.py")):
    (scratch / rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(KIT / "base" / fn, scratch / rel)
real_site, real_active, real_run = bi.SITE, bi.service_active, bi.subprocess.run
bi.SITE, bi.service_active = scratch, lambda: False
rc, out = tool(bi.status)
check("I base reads as base", rc == 0 and out.endswith("STATE base\n"), out[-40:])
check("I want patched on a base site fails", tool(bi.status, "patched")[0] == 1)
rc, out = tool(bi.install, False)
check("I install ends patched", rc == 0 and out.endswith("STATE patched\n"), out[-40:])
side = scratch / bi.SIDE
good = side.read_bytes()
side.write_bytes(good + b"\n")
check("I stale side file is UNKNOWN", tool(bi.status)[0] == 1 and "STATE UNKNOWN" in tool(bi.status)[1])
side.unlink()
check("I missing side file is UNKNOWN", tool(bi.status)[0] == 1)
side.write_bytes(good)
bak = scratch / (bi.DSPARK + ".bak-boundary")
good_bak = bak.read_bytes()
bak.write_bytes(b"x")
check("I bad backup is UNKNOWN", tool(bi.status)[0] == 1)
bak.write_bytes(good_bak)
rc, out = tool(bi.uninstall)
check("I uninstall ends base", rc == 0 and out.endswith("STATE base\n"), out[-40:])
bi.service_active = real_active
for answer, want_active in (("active", True), ("activating", True), ("deactivating", True), ("", True), ("inactive", False), ("failed", False)):
    bi.subprocess.run = lambda *a, _s=answer, **k: types.SimpleNamespace(stdout=_s + "\n")
    check(f"I service state {answer!r}", bi.service_active() is want_active)
bi.subprocess.run = real_run
bi.SITE = real_site
shutil.rmtree(scratch, ignore_errors=True)


# ---------------------------------------------------------------- S
def swizzle_copy(sf, M, K):
    factor = 128
    num_m_tiles = (M + 127) // 128
    num_k_tiles = (K + factor - 1) // factor
    m_padded, k_scale_padded = num_m_tiles * 128, num_k_tiles * 4
    scale_cols = K // 32
    sf_padded = torch.zeros((m_padded, k_scale_padded), dtype=sf.dtype)
    sf_padded[:M, :scale_cols] = sf
    return sf_padded.view(num_m_tiles, 4, 32, num_k_tiles, 4).transpose(1, 3).contiguous().view(-1)


try:
    sys.path.insert(0, str(Path.home() / "venvs/vllm032/lib/python3.12/site-packages"))
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import swizzle_mxfp8_scale as swz

    src = "real"
except Exception as e:  # noqa: BLE001
    swz, src = swizzle_copy, f"copy ({e!r})"[:60]
check("S uses the real swizzle, not the copy", src == "real", src)
sf = torch.randint(0, 255, (1792, 160), dtype=torch.uint8)
full = swz(sf, 1792, 5120)
per_tile = full.numel() // 14
check(f"S scale slice, {src}", torch.equal(full[10 * per_tile:], swz(sf[1280:], 512, 5120)))
merged = torch.cat([full[10 * per_tile:]] * 3)
check("S merged scale", torch.equal(merged, swz(torch.cat([sf[1280:]] * 3), 1536, 5120)))


# ---------------------------------------------------------------- R
def gpu_test_def(name):
    """The class or function exactly as test_boundary_gpu.py has it. That file needs a GPU to import."""
    text = (KIT / "tests/test_boundary_gpu.py").read_text()
    for c in ast.walk(ast.parse(text)):
        if isinstance(c, (ast.ClassDef, ast.FunctionDef)) and c.name == name:
            return textwrap.dedent(" " * c.col_offset + ast.get_source_segment(text, c))
    raise KeyError(name)


def real_scales():
    from safetensors import safe_open

    model = Path.home() / "models/dsv41/uncensored-fp8"
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]

    def tensor(name):
        with safe_open(str(model / index[name]), framework="pt") as f:
            return f.get_tensor(name)

    ns = {"torch": torch, "tensor": tensor, "swizzle_mxfp8_scale": swz, "vfi": None, "DEV": "cpu", "ws": None, "runner": None}
    exec(compile(gpu_test_def("Lin"), "Lin", "exec"), ns)  # noqa: S102
    lins = [ns["Lin"](i) for i in range(3)]
    check("R Lin builds on real scales", all(lin.weight_scale.numel() == 286720 and lin.weight_scale.dtype == torch.uint8 for lin in lins))
    check("R weight shape and dtype", all(tuple(lin.weight.shape) == (1792, 5120) and lin.weight.dtype == torch.float8_e4m3fn for lin in lins))

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.quantization.modelopt import KMxfp8Static

    ns.update(RMSNorm=RMSNorm, VllmConfig=VllmConfig, set_current_vllm_config=set_current_vllm_config)
    exec(compile(gpu_test_def("kv_norm_of"), "kv_norm_of", "exec"), ns)  # noqa: S102
    cfg = json.loads((model / "config.json").read_text())
    norms = [ns["kv_norm_of"](i) for i in range(3)]
    x = torch.randn(2, 512).to(torch.bfloat16)
    check("R kv_norm builds, runs, real eps", all(n(x).dtype == torch.bfloat16 and n.variance_epsilon == cfg["text_config"]["rms_norm_eps"] for n in norms))
    check("R kv_norm has the real weights", all(torch.equal(n.weight.data, tensor(f"mtp.{i}.attn.kv_norm.weight")) for i, n in enumerate(norms)))

    scaled = KMxfp8Static.get_scale_weight_loader(lambda p, w, *a, **k: w, types.SimpleNamespace(scale_block_size=(32, 32)))
    two_d = []
    for i, lin in enumerate(lins):
        q = scaled(None, tensor(f"mtp.{i}.attn.wq_a.scale"))
        kv = scaled(None, tensor(f"mtp.{i}.attn.wkv.scale"))
        s2 = torch.cat([q, kv], dim=0)
        two_d.append(s2)
        check(f"R layer {i} equals modelopt loader + swizzle", s2.shape == (1792, 160) and torch.equal(lin.weight_scale, swz(s2, 1792, 5120)))
    layers = [types.SimpleNamespace(attn=types.SimpleNamespace(fused_wqa_wkv=lin, q_lora_rank=1280)) for lin in lins]
    merge = B.CtxKvFold(layers, "merge", runner=object(), workspace=None)
    want = swz(torch.cat([s[1280:] for s in two_d], dim=0), 1536, 5120)
    check("R merged fold scale on real scales", merge.parts[0][1].numel() == 245760 and torch.equal(merge.parts[0][1], want))
    check("R merged fold weight shape", tuple(merge.parts[0][0].shape) == (5120, 1536))
    split = B.CtxKvFold(layers, "split", runner=object(), workspace=None)
    check("R split fold scale on real scales", all(torch.equal(split.parts[i][1], swz(two_d[i][1280:], 512, 5120)) for i in range(3)))


try:
    real_scales()
except Exception as e:  # noqa: BLE001 - a missing model or import is a failure here, not a skip
    check("R ran", False, repr(e)[:200])


# ---------------------------------------------------------------- L / W helpers
def rows(M, V, g, kind):
    x = (torch.randint(-40, 41, (M, V), generator=g).float() / 2.0)
    top = x.max(dim=1).values + 1.0
    for r in range(M):
        a, b = torch.randint(0, V, (2,), generator=g).tolist()
        x[r, a] = top[r]
        if kind == "tie":
            x[r, b] = top[r]
        elif kind == "tie_far":
            x[r, 0] = top[r]
            x[r, V - 1] = top[r]
        elif kind == "zero":
            x[r] = -3.0
            x[r, min(a, b)] = -0.0
            x[r, max(a, b)] = 0.0
        elif kind == "zero2":
            x[r] = -3.0
            x[r, min(a, b)] = 0.0
            x[r, max(a, b)] = -0.0
        elif kind == "neginf":
            x[r] = float("-inf")
        elif kind == "inf":
            x[r, a] = float("inf")
        elif kind == "nan":
            x[r, a] = float("inf")
            x[r, b] = float("nan")
        elif kind == "subnormal":
            x[r] = -3.0
            x[r, min(a, b)] = 2.0 ** -130
            x[r, max(a, b)] = 2.0 ** -129
    return x.to(torch.bfloat16)


def stock_pair(x, vstart):
    vals, idx = x.max(dim=-1)
    vals = vals.float()
    zero = torch.zeros_like(vals)
    return torch.stack([vals, (idx + vstart).float(), zero, zero], dim=-1)


KINDS = ("plain", "tie", "tie_far", "zero", "zero2", "neginf", "inf", "nan", "subnormal")
t0 = time.time()
g = torch.Generator().manual_seed(1)
n = bad = 0
for V in (2048, 16160):
    Ms = (1, 2, 3, 5, 8, 16, 32) if V == 2048 else (1, 3)
    for M in Ms:
        for kind in KINDS if V == 2048 else ("plain", "tie_far", "zero"):
            x = rows(M, V, g, kind)
            ref = stock_pair(x, 16160 * 3)
            for tiles in (1, 16):
                got = B.mk_local_pair(x, 16160 * 3, tiles)
                same = torch.equal(got.view(torch.int32), ref.view(torch.int32))
                n += 1
                if not same:
                    bad += 1
                    print("  local diff", V, M, kind, tiles, got[:2].view(torch.int32).tolist(), ref[:2].view(torch.int32).tolist())
check("L local pair", bad == 0, f"{n} cases, {time.time() - t0:.0f} s")

t0 = time.time()
TP = 8
w1 = torch.randint(-32000, 32000, (16384, 256), dtype=torch.int16, generator=g).view(torch.bfloat16)
n = bad = 0
for M in list(range(1, 33)):
    for kind in ("plain", "tie", "zero", "neginf", "nan"):
        pairs = []
        shared = rows(M, 64, g, kind)
        for r in range(TP):
            x = rows(M, 64, g, kind)
            if r in (2, 5):
                x = shared  # identical maxima on two ranks: the lower rank must win
            pairs.append(stock_pair(x, r * 2048))
        G = torch.cat(pairs, dim=0)
        gv = G.view(TP, M, 4)
        best = gv[:, :, 0].argmax(dim=0, keepdim=True)
        tok = gv[:, :, 1].gather(0, best).squeeze(0).to(torch.int64)
        dt = torch.zeros((M, 5), dtype=torch.int64)
        e = B.mk_winner(G, TP, M, dt, 3, w1, True)
        n += 1
        if not (torch.equal(dt[:, 3], tok) and torch.equal(e.view(torch.int16), w1[tok].view(torch.int16))):
            bad += 1
            print("  winner diff", M, kind, dt[:, 3].tolist(), tok.tolist())
check("W winner", bad == 0, f"{n} cases, {time.time() - t0:.0f} s")


# ---------------------------------------------------------------- F: the real function source, both paths
def extract(text, name):
    tree = ast.parse(text)
    for c in ast.walk(tree):
        if isinstance(c, ast.FunctionDef) and c.name == name:
            return ast.get_source_segment(text, c)
    raise KeyError(name)


pkg = types.ModuleType("vllm.models.deepseek_v41.nvidia.dsv41_boundary")
pkg.markov_fold_loop = B.markov_fold_loop
pkg.prebuild = B.prebuild
pkg.audit_tick = B.audit_tick
sys.modules["vllm.models.deepseek_v41.nvidia.dsv41_boundary"] = pkg

src = bi.patched(bi.SPEC, (KIT / "base/speculator.py").read_text())


def make_fn(fold):
    mod = {"torch": torch, "_DSV41_MK_FOLD": fold, "tensor_model_parallel_all_gather": None}
    import textwrap

    code = textwrap.dedent(extract(src, "_sample_sequential_sharded"))
    exec(compile(code, "sharded", "exec"), mod)  # noqa: S102
    return mod["_sample_sequential_sharded"], mod


class Emu:
    """One rank under test (rank R); the other ranks' pairs come from the stock formula in the fake all-gather."""

    def __init__(self, num_reqs, R, seed, kind):
        self.g = torch.Generator().manual_seed(seed)
        self.n_spec, self.tp, self.R, self.kind = 5, 8, R, kind
        self.Vl, self.Vt = 1024, 8192
        self.num_reqs = num_reqs
        self.w1 = torch.randn(self.Vt, 256, generator=self.g).to(torch.bfloat16)
        self.w2 = torch.randn(self.Vt, 256, generator=self.g).to(torch.bfloat16)
        if kind == "tie":
            self.w2 = torch.zeros_like(self.w2)  # bias 0: base is on a coarse grid, so ties inside and across ranks are common
        self.base_all = (torch.randint(-30, 31, (self.tp, num_reqs * 5, self.Vl), generator=self.g).float() / 4).to(torch.bfloat16)
        self.anchor = torch.randint(0, self.Vt, (num_reqs,), generator=self.g)


def run(emu, fold):
    fn, mod = make_fn(fold)
    e = emu
    num_reqs = e.num_reqs
    state = {"embed": None, "pos": 0}
    w1, w2 = e.w1, e.w2
    R, tp, Vl = e.R, e.tp, e.Vl

    def bias_for(rank, embed):
        return (embed.float() @ w2[rank * Vl : (rank + 1) * Vl].float().t()).to(torch.bfloat16)

    def lp(head, x, skip_gather=False):
        if head == "base":
            return e.base_all[R].clone()
        state["embed"] = x
        return bias_for(R, x)

    def all_gather(pair, dim=0):
        i = state["pos"]
        state["pos"] += 1
        parts = []
        for r in range(tp):
            if r == R:
                parts.append(pair)
            else:
                b = e.base_all[r].view(num_reqs, 5, Vl)[:, i] + bias_for(r, state["embed"])
                parts.append(stock_pair(b, r * Vl))
        return torch.cat(parts, dim=0)

    mod["tensor_model_parallel_all_gather"] = all_gather

    class M2:
        logits_processor = None

        def markov_embed(self, ids):
            return w1[ids]

        def compute_confidence(self, h, emb):
            return emb.float().sum(-1) * 0 + (emb.view(torch.int16).long().sum(-1) % 1000) / 1000.0

    sp = types.SimpleNamespace()
    sp._sharded_markov = {"shim": types.SimpleNamespace(weight=w2[R * Vl : (R + 1) * Vl]), "vocab_start": R * Vl,
                          "num_pad": 0, "tp": tp}
    sp.num_speculative_steps = 5
    sp.sample_indices = torch.arange(num_reqs * 5)
    inner = types.SimpleNamespace(norm=lambda h: h)
    model = M2()
    model.logits_processor = lambda head, h, skip_gather=False: lp("base" if head == "lm" else head, h, skip_gather)
    model.lm_head = "lm"
    model.model = inner
    inner.markov_head = types.SimpleNamespace(markov_w1=types.SimpleNamespace(weight=w1))
    sp.model = model
    sp.use_confidence_head = True
    sp.input_buffers = types.SimpleNamespace(input_ids=e.anchor)
    sp._anchor_idx = torch.arange(num_reqs)
    sp.draft_tokens = torch.zeros((num_reqs + 3, 5), dtype=torch.int64)
    sp.draft_token_confidence_probs = torch.zeros((num_reqs + 3, 5))
    # The self-check is slow in the interpreter: run it for real once, then reuse its verdict with this emulation's weights.
    sp._dsv41_mk = dict(w1=w1, tp=tp, vstart=R * Vl) if MK_CHECKED else None
    fn(sp, num_reqs, torch.zeros((num_reqs * 5, 4)))
    return sp.draft_tokens[:num_reqs].clone(), sp.draft_token_confidence_probs[:num_reqs].clone(), getattr(sp, "_dsv41_mk", None)


t0 = time.time()
n = bad = 0
taken = 0
e = Emu(2, 3, 7, "plain")
MK_CHECKED = False
_a, _c, st0 = run(e, True)
check("F self-check passes on real calls", bool(st0), f"{time.time() - t0:.0f} s")
MK_CHECKED = True
for num_reqs in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32):
    for seed in (1, 2):
        for R in (0, 5):
            e = Emu(num_reqs, R, 100 * num_reqs + seed, "plain")
            a, ca, _ = run(e, False)
            b, cb, st = run(e, True)
            n += 1
            taken += int(bool(st))
            if not (torch.equal(a, b) and torch.equal(ca, cb)):
                bad += 1
                print("  fold diff", num_reqs, seed, R, a.tolist(), b.tolist())
check("F full loop stock vs fold", bad == 0 and taken == n, f"{n} cases, fold taken {taken}, {time.time() - t0:.0f} s")

t0 = time.time()
n = bad = 0
for num_reqs in range(1, 33):
    for R in (0, 3, 7):
        e = Emu(num_reqs, R, 7000 + num_reqs, "tie")
        a, ca, _ = run(e, False)
        b, cb, _ = run(e, True)
        n += 1
        if not (torch.equal(a, b) and torch.equal(ca, cb)):
            bad += 1
            print("  tie fold diff", num_reqs, R, a.tolist(), b.tolist())
check("F injected ties, num_reqs 1..32", bad == 0, f"{n} cases, {time.time() - t0:.0f} s")

# audit mode: stock serves, counters stay 0; a forced difference is counted
os.environ.update(DSV41_MK_FOLD="audit", DSV41_BOUNDARY_AUDIT_DIR=str(KIT / "results"))
(KIT / "results").mkdir(exist_ok=True)
for stale in (KIT / "results").glob("boundary-audit.*.json"):  # a file left by an earlier run must not pass the report check
    stale.unlink()
A = load_side()
pkg.markov_fold_loop = A.markov_fold_loop
e = Emu(4, 2, 99, "tie")
a, ca, _ = run(e, False)
b, cb, _ = run(e, True)
c = A._AUDIT["ctr"].tolist()
check("A audit serves stock, counts 0", torch.equal(a, b) and torch.equal(ca, cb) and c[3] == 5 and c[4] == 20 and c[5] == c[6] == c[7] == 0, str(c))
real_pair = A.mk_local_pair
A.mk_local_pair = lambda x, v, tiles=None: real_pair(x, v, tiles) + 1.0
run(e, True)
check("A audit counts a forced difference", A._AUDIT["ctr"].tolist()[5] > 0, str(A._AUDIT["ctr"].tolist()))
A.mk_local_pair = real_pair
A.audit_tick(256)
check("A audit report written", any((KIT / "results").glob("boundary-audit.*.json")))
pkg.markov_fold_loop = B.markov_fold_loop


# ---------------------------------------------------------------- C
class FakeRunner:
    """Exact integer GEMM stand-in. Enough to test wiring; says nothing about CUTLASS."""

    def __call__(self, inputs, tactic):
        xq, wt, xs, s, dt, out, ws = inputs
        self.calls.append((tuple(wt.shape), int(tactic)))
        r = xq.float() @ wt.float()
        out.copy_(r.to(torch.bfloat16))

    def __init__(self):
        self.calls = []

    def get_valid_tactics(self, a, b):
        return [0, 1, 2, 3, 4, 5]


class Lin:
    def __init__(self, seed, N=256 + 128, K=128):
        g = torch.Generator().manual_seed(seed)
        self.weight = torch.randint(-3, 4, (N, K), generator=g).float()
        self.weight_scale = torch.arange(N // 128 * 40, dtype=torch.float32)
        self.bias = None

    def __call__(self, x):
        return (x.float() @ self.weight.t()).to(torch.bfloat16), None


class Attn:
    def __init__(self, seed):
        self.fused_wqa_wkv = Lin(seed)
        self.q_lora_rank = 256
        self.kv_norm = lambda v: v * 2


class Layer:
    def __init__(self, seed):
        self.attn = Attn(seed)


def quant(x):
    return x.float(), torch.zeros(1)


layers = [Layer(1), Layer(2), Layer(3)]
model = types.SimpleNamespace(layers=layers)
runner = FakeRunner()
f = B.CtxKvFold(layers, "merge", runner=runner, workspace=None)
tm = B.parse_map(B.CTX_MAP_RAW)
check("C tactic map", tm[6] == 1 and tm[64] == 1 and tm[65] == 2 and tm[128] == 2 and tm[192] == 4 and 193 not in tm)
f.self_check(layers, quant, tm, torch.device("cpu"))
check("C self-check accepts equal bits", not f.rejected and set(f.ok) == {m for m in B.CTX_CHECK_MS if m in tm}, str(f.rejected)[:200])
model._dsv41_ctxkv = f
ins = []
x = (torch.randint(-3, 4, (6, 128)).float()).to(torch.bfloat16)
ok = B.ctxkv_precompute(model, x, "pos", ["s0", None, "s2"], lambda a, kv, p, s: ins.append((a, kv.clone(), p, s)), quantize=quant)
refs = [ly.attn.fused_wqa_wkv(x)[0][..., 256:] * 2 for ly in layers]
check("C fold ran, slot None skipped", ok and [i[3] for i in ins] == ["s0", "s2"])
check("C layer order and bits", torch.equal(ins[0][1], refs[0]) and torch.equal(ins[1][1], refs[2]))
check("C one GEMM, N=3*H, pinned tactic", runner.calls[-1] == ((128, 384), 1))
check("C stock on dummy run", B.ctxkv_precompute(model, x, "pos", None, None, quantize=quant) is False)
x7 = (torch.randint(-3, 4, (7, 128)).float()).to(torch.bfloat16)
check("C stock on an unchecked row count", B.ctxkv_precompute(model, x7, "pos", ["a", "b", "c"], None, quantize=quant) is False)
bad_f = B.CtxKvFold(layers, "merge", runner=type("W", (FakeRunner,), {"__call__": lambda self, inputs, tactic: inputs[5].fill_(1.0)})(), workspace=None)
bad_f.self_check(layers, quant, {6: 1}, torch.device("cpu"))
check("C self-check rejects wrong bits", 6 in bad_f.rejected and not bad_f.ok)
s = B.CtxKvFold(layers, "split", runner=FakeRunner(), workspace=None)
check("C split parts", [tuple(p[0].shape) for p in s.parts] == [(128, 128)] * 3)

# A CTX-only audit arm: the counters must exist before capture, or ctxkv_precompute returns False in every graph and measures nothing.
os.environ.update(DSV41_CTXKV_FOLD="audit", DSV41_MK_FOLD="0")
CA = load_side()
CA.CtxKvFold = lambda ls, mode: B.CtxKvFold(ls, mode, runner=FakeRunner(), workspace=None)
CA._default_quantize = quant
ca_model = types.SimpleNamespace(layers=layers)
check("C ctx-only audit builds its state", CA.ctxkv_state(ca_model) is not None)
check("C ctx-only audit has counters before capture", CA._AUDIT["ctr"] is not None)

print("VERDICT", "FAIL " + ",".join(FAIL) if FAIL else "PASS")
sys.exit(1 if FAIL else 0)
