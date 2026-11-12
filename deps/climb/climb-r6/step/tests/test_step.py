"""CPU tests for r6-step (pytest). Run: cd <kit> && python3 -m pytest -q tests/test_step.py
Needs torch (CPU) + numpy. No vllm import: dsv41_step is loaded from the kit's files/ tree with its vllm-facing
helpers replaced by fakes; the installer is exercised on copies of the live (and r5-composed) files in base/."""
import ast
import contextlib
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KIT)
import install as I  # noqa: E402

R5 = os.environ.get("STEP_R5") or os.path.join(os.path.dirname(os.path.dirname(KIT)), "climb-r5")
REL_E = "models/deepseek_v41/common/engram.py"
REL_R4 = "models/deepseek_v41/nvidia/dsv41_r4_small.py"
BASES = {REL_E: ["base/live/engram.py", "base/r5/engram.py"], REL_R4: ["base/live/dsv41_r4_small.py"]}


def load_step(env=None):
    env = env or {}
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        spec = importlib.util.spec_from_file_location("dsv41_step_t", I.MOD_SRC)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return m


def sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


# ------------------------------------------------------------------------------------------------ installer
def test_base_shas_match_kit():
    k = json.load(open(os.path.join(KIT, "KIT.json")))
    for f in k["files"]:
        if f["action"] != "patch":
            continue
        shas = {sha(open(os.path.join(KIT, p)).read()) for p in BASES[f["path"]]}
        assert shas == {f["base_sha256"], *f.get("compose_sha256", [])}
        for p in BASES[f["path"]]:
            new = I.patch_text(f["path"], open(os.path.join(KIT, p)).read())
            assert sha(new) in f["patched_sha256"]


@pytest.mark.parametrize("rel,src", [(r, s) for r, ss in BASES.items() for s in ss])
def test_anchors_apply_roundtrip_compile(rel, src):
    t = open(os.path.join(KIT, src)).read()
    n = I.patch_text(rel, t)
    assert I.unpatch_text(rel, n) == t
    ast.parse(n)
    added = [l for l in n.splitlines() if l not in set(t.splitlines())]
    assert added and all(l.rstrip().endswith(I.TAG) or l.strip() == "" for l in added), added
    with pytest.raises(ValueError):
        I.patch_text(rel, n)  # anchors are consumed: a second patch cannot apply


class _Fold(ast.NodeTransformer):
    FLAGS = ("_DSV41_STEP_C3", "_DSV41_STEP_C4")

    def _has(self, node):
        return any(isinstance(x, ast.Name) and x.id in self.FLAGS for x in ast.walk(node))

    def visit_If(self, node):
        self.generic_visit(node)
        t = node.test
        first = t.values[0] if isinstance(t, ast.BoolOp) and isinstance(t.op, ast.And) else t
        if isinstance(first, ast.Name) and first.id in self.FLAGS:
            return node.orelse or None  # flag False: the body never runs
        return node

    def visit_Assign(self, node):
        if any(isinstance(t, ast.Name) and t.id in self.FLAGS for t in node.targets):
            return None
        return node


@pytest.mark.parametrize("rel,src", [(r, s) for r, ss in BASES.items() for s in ss])
def test_flag_off_is_ast_identical(rel, src):
    t = open(os.path.join(KIT, src)).read()
    folded = _Fold().visit(ast.parse(I.patch_text(rel, t)))
    assert ast.dump(folded) == ast.dump(ast.parse(t))


def _tree(tmp, srcs):
    v = os.path.join(tmp, "vllm")
    for rel, src in srcs.items():
        os.makedirs(os.path.dirname(os.path.join(v, rel)), exist_ok=True)
        shutil.copy2(src, os.path.join(v, rel))
    os.makedirs(os.path.join(v, os.path.dirname(I.REL_MOD)), exist_ok=True)
    return v


def _run(v, *args):
    env = dict(os.environ, STEP_VLLM=v)
    r = subprocess.run([sys.executable, os.path.join(KIT, "install.py"), *args], env=env, capture_output=True,
                       text=True)
    return r.returncode, r.stdout + r.stderr


@pytest.mark.parametrize("variant", ["live", "r5"])
def test_install_cycle(tmp_path, variant):
    srcs = {REL_E: os.path.join(KIT, f"base/{variant}/engram.py"), REL_R4: os.path.join(KIT, BASES[REL_R4][0])}
    v = _tree(str(tmp_path), srcs)
    before = {r: open(os.path.join(v, r), "rb").read() for r in srcs}
    rc, out = _run(v, "dry")
    assert rc == 0 and "would patch" in out
    assert all(open(os.path.join(v, r), "rb").read() == before[r] for r in srcs)
    rc, out = _run(v, "install")
    assert rc == 0 and out.count("INSTALLED") == 3, out
    rc2, out2 = _run(v, "install")
    assert rc2 == 0 and "patch " not in out2  # idempotent
    rc, out = _run(v, "uninstall")
    assert rc == 0, out
    assert all(open(os.path.join(v, r), "rb").read() == before[r] for r in srcs)
    assert not os.path.exists(os.path.join(v, I.REL_MOD))
    assert not any(f.endswith(I.BAKSUF) for _, _, fs in os.walk(v) for f in fs)


def test_refuses_foreign(tmp_path):
    srcs = {REL_E: os.path.join(KIT, "base/live/engram.py"), REL_R4: os.path.join(KIT, BASES[REL_R4][0])}
    v = _tree(str(tmp_path), srcs)
    with open(os.path.join(v, REL_R4), "a") as f:
        f.write("# drift\n")
    rc, out = _run(v, "install")
    assert rc == 2 and "REFUSE" in out
    assert open(os.path.join(v, REL_E)).read() == open(srcs[REL_E]).read()  # all-or-nothing


@pytest.mark.skipif(not os.path.isdir(os.path.join(R5, "integrate", "live", "vllm")), reason="climb-r5 tree absent")
def test_compose_with_r5_prio_both_orders(tmp_path):
    """r5-prio is the only r5 kit that touches a file of ours (engram.py). r5-prio then r6-step: both install,
    r6-step uninstalls back to the r5 bytes, r5-prio uninstalls back to live. r6-step then r5-prio: r5-prio
    refuses engram.py (it pins its bases) without writing anything; r6-step uninstall restores live."""
    v = os.path.join(str(tmp_path), "vllm")
    shutil.copytree(os.path.join(R5, "integrate", "live", "vllm"), v)
    shutil.copy2(os.path.join(KIT, BASES[REL_R4][0]), os.path.join(v, REL_R4))
    live = {r: open(os.path.join(v, r), "rb").read() for r in (REL_E, REL_R4)}
    prio = [sys.executable, os.path.join(R5, "prio", "install.py")]
    penv = dict(os.environ, PRIO_VLLM=v)
    r = subprocess.run(prio + ["install"], env=penv, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    r5_e = open(os.path.join(v, REL_E), "rb").read()
    rc, out = _run(v, "install")
    assert rc == 0, out
    rc, out = _run(v, "uninstall")
    assert rc == 0 and open(os.path.join(v, REL_E), "rb").read() == r5_e, out
    r = subprocess.run(prio + ["uninstall"], env=penv, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert open(os.path.join(v, REL_E), "rb").read() == live[REL_E]
    rc, out = _run(v, "install")
    assert rc == 0, out
    mine = open(os.path.join(v, REL_E), "rb").read()
    r = subprocess.run(prio + ["install"], env=penv, capture_output=True, text=True)
    assert r.returncode != 0
    assert open(os.path.join(v, REL_E), "rb").read() == mine
    rc, out = _run(v, "uninstall")
    assert rc == 0 and all(open(os.path.join(v, x), "rb").read() == live[x] for x in live), out


# ------------------------------------------------------------------------------------------------ AR map
def _gpu_fadd(a: np.float32, b: np.float32) -> np.float32:
    """FADD (sm_80+, no .FTZ): IEEE RN, denormals kept, any NaN operand -> canonical 0x7FFFFFFF."""
    if np.isnan(a) or np.isnan(b):
        return np.array([0x7FFFFFFF], dtype=np.uint32).view(np.float32)[0]
    return np.float32(a) + np.float32(b)


def _f2fp_bf16(x: np.float32) -> int:
    u = int(np.array([x], dtype=np.float32).view(np.uint32)[0])
    if (u & 0x7F800000) == 0x7F800000 and (u & 0x7FFFFF):
        return 0x7FFF  # canonical NaN
    lsb = (u >> 16) & 1
    return ((u + 0x7FFF + lsb) >> 16) & 0xFFFF


def _ar_emulate(bits16: int, owner: int, other_bits: int, tp: int = 8) -> int:
    acc = np.float32(0.0)
    for r in range(tp):
        b = bits16 if r == owner else other_bits
        x = np.array([b << 16], dtype=np.uint32).view(np.float32)[0]
        acc = _gpu_fadd(acc, x)
    return _f2fp_bf16(acc)


def test_canon_bits_exhaustive_vs_ar_emulation():
    m = load_step()
    allb = torch.arange(0, 65536, dtype=torch.int32)
    w = ((allb ^ 0x8000) - 0x8000).to(torch.int16).view(torch.bfloat16)
    got = (m.canon_bits(w).to(torch.int32) & 0xFFFF).numpy()
    rng = np.random.default_rng(0)
    for b in list(range(0, 65536, 1)) if os.environ.get("STEP_SLOW") else \
            list(range(0, 256)) + list(range(0x7F00, 0x8100)) + list(range(0xFF00, 0x10000)) + \
            [int(x) for x in rng.integers(0, 65536, 4000)]:
        for owner in (0, 3, 7):
            for other in (0x0000, 0x8000):  # masked partials written as +0 or -0
                assert _ar_emulate(b, owner, other) == got[b], (hex(b), owner, hex(other))
    # the map in words
    assert got[0x8000] == 0 and got[0x0000] == 0 and got[0x7FC1] == 0x7FFF and got[0xFFC0] == 0x7FFF
    assert got[0x7F80] == 0x7F80 and got[0xFF80] == 0xFF80 and got[0x0001] == 0x0001 and got[0x8001] == 0x8001


def test_identity_map_mutant_is_caught():
    m = load_step()
    w = torch.tensor([0x8000, 0x3F80], dtype=torch.int32)
    w = ((w ^ 0x8000) - 0x8000).to(torch.int16).view(torch.bfloat16)
    assert not torch.equal(w.view(torch.int16), m.canon_bits(w))  # a table copied without the map differs


# ------------------------------------------------------------------------------------------------ COLEMB
class _SI:
    def __init__(self, vs, ve):
        self.org_vocab_start_index, self.org_vocab_end_index, self.num_added_elements = vs, ve, 0


class _Emb:
    def __init__(self, Wfull, rank, tp):
        V, D = Wfull.shape
        per = (V + tp - 1) // tp
        self.vs, self.ve = rank * per, min(V, (rank + 1) * per)
        self.weight = torch.nn.Parameter(Wfull[self.vs:self.ve].clone(), requires_grad=False)
        self.org_vocab_size, self.embedding_dim, self.shard_indices = V, D, _SI(self.vs, self.ve)


def _stock_forward_factory(Wfull, tp, map_neg0=True):
    """Emulated live path: every rank's fused kernel (owner row or +0) + the rdma AR map."""
    m = load_step()

    def fwd(emb, ids):
        V = Wfull.shape[0]
        ok = (ids >= 0) & (ids < V)
        rows = Wfull[ids.clamp(0, V - 1).long()]
        bits = m.canon_bits(rows) if map_neg0 else rows.view(torch.int16).clone()
        bits[~ok] = 0
        return bits.view(torch.bfloat16)

    return fwd


def _table_with_specials(V=300, D=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(V, D, generator=g).to(torch.bfloat16)
    bits = W.view(torch.int16)
    bits[5, :] = -32768  # a planted all -0 row (0x8000)
    bits[17, 3] = 0x7FC1  # NaN, non-canonical payload
    bits[42, 0] = 0x0003  # denormal
    bits[99, 9] = -32768
    return W


def test_build_coltable_full_table_and_lookup():
    m = load_step()
    tp, V, D = 8, 300, 64
    W = _table_with_specials(V, D)
    fwd = _stock_forward_factory(W, tp)
    tables = []
    for r in range(tp):
        emb = _Emb(W, r, tp)
        ct, st = m.build_coltable(emb, fwd, r, tp, max_rows=16, chunk=16)
        assert st["mismatch"] == 0, st
        tables.append(ct)
    assert tables[0].table.shape == (V, D // tp)
    # lookup (torch reference of the Triton kernel's semantics) + emulated column all-gather == stock AR output
    ids = torch.tensor([0, 5, 17, 42, 99, 299, -1, 300, 1000, 5], dtype=torch.int64)
    ref = fwd(None, ids).view(torch.int16)

    def lookup(ct):
        ok = (ids >= 0) & (ids < ct.V)
        o = ct.table[ids.clamp(0, ct.V - 1)].view(torch.int16).clone()
        o[~ok] = 0
        return o

    got = torch.cat([lookup(ct) for ct in tables], dim=-1)
    assert torch.equal(got, ref)
    assert int(got[1].abs().sum()) == 0  # the -0 row came out +0


def test_build_detects_unmapped_ar():
    m = load_step()
    tp, V, D = 8, 300, 64
    W = _table_with_specials(V, D)
    bad = _stock_forward_factory(W, tp, map_neg0=False)  # an "AR" that keeps -0: the map check must fail
    mism = sum(m.build_coltable(_Emb(W, r, tp), bad, r, tp, 16, 16)[1]["mismatch"] for r in range(tp))
    assert mism > 0


# ------------------------------------------------------------------------------------------------ Engram pack
def _gather_factory(m, per_rank_operand):
    """_all_gather stub: SPMD emulation; per_rank_operand(r) gives rank r's operand (same shape)."""

    def ag(x, dim):
        return torch.cat([per_rank_operand(r) for r in range(8)], dim=dim)

    return ag


def test_packed_rows_gather_layout():
    m = load_step()
    T, H, hd, tp = 5, 3, 16, 8
    ra = [torch.randn(T, H, hd).to(torch.bfloat16) for _ in range(tp)]
    rb = [torch.randn(T, H, hd).to(torch.bfloat16) for _ in range(tp)]
    calls = []

    class E:
        def __init__(self, rows):
            self.rows = rows

        def _ready_rows(self, n):
            calls.append(n)
            return self.rows[:n]

    me = 2
    m._all_gather = _gather_factory(m, lambda r: torch.cat([ra[r], rb[r]], dim=1))
    o1, o2 = m.packed_rows_gather(E(ra[me]), E(rb[me]), T)
    assert torch.equal(o1.view(torch.int16), torch.cat(ra, dim=1).view(torch.int16))  # stock AG(rows, dim=1)
    assert torch.equal(o2.view(torch.int16), torch.cat(rb, dim=1).view(torch.int16))
    assert calls == [T, T] and o1.is_contiguous() and o2.is_contiguous()


def test_packed_cols_gather_layout():
    m = load_step()
    T, n, tp = 4, 24, 8
    y1 = [torch.randn(T, n).to(torch.bfloat16) for _ in range(tp)]
    y2 = [torch.randn(T, n).to(torch.bfloat16) for _ in range(tp)]
    m._all_gather = _gather_factory(m, lambda r: torch.cat([y1[r], y2[r]], dim=-1))
    k1, k2 = m.packed_cols_gather(y1[0], y2[0])
    assert torch.equal(k1.view(torch.int16), torch.cat(y1, dim=-1).view(torch.int16))
    assert torch.equal(k2.view(torch.int16), torch.cat(y2, dim=-1).view(torch.int16))


class _FakeTable:
    def __init__(self):
        self.tp_size, self.dp_size, self.part_n_hash_cols, self.n_hash_cols, self.dim = 8, 1, 3, 24, 16


class _Lin:
    def __init__(self, tag, log):
        self.gather_output, self.skip_bias_add, self.tag, self.log, self.bias = True, False, tag, log, None
        outer = self

        class QM:
            def apply(self, layer, x, bias):
                outer.log.append(("gemm", outer.tag, tuple(x.shape)))
                return x[:, :8] * (2 if outer.tag == "a" else 3)

        self.quant_method = QM()

    def forward(self, x):
        self.log.append(("stock_wkv", self.tag))
        y = self.quant_method.apply(self, x, None)
        return torch.cat([y] * 8, dim=-1)

    def __call__(self, x):
        return self.forward(x)


def _engram_world(m, mode_env):
    log = []
    m.PK.__init__()

    class FakeEngram:
        def __init__(self, idx):
            self.layer_hash_index, self._batch, self.use_sequence_parallel = idx, None, False
            self.embed_tokens = _FakeTable()
            self.staged_rows = torch.zeros(64, 3, 16)
            self.wkv = _Lin("ab"[idx], log)

        def _ready_rows(self, n):
            log.append(("wait", self.layer_hash_index))
            return self.staged_rows[:n] + self.layer_hash_index

        def embed(self, hash_ids):
            log.append(("stock_embed", self.layer_hash_index))
            return torch.zeros(hash_ids.shape[0], 24, 16)

        def forward(self, hash_ids):
            return self.wkv(self.embed(hash_ids).flatten(-2))

    _Lin.forward.__qualname__ = "ColumnParallelLinear.forward"
    m._install_engram(FakeEngram)
    a, b = FakeEngram(0), FakeEngram(1)

    def ag(x, dim):
        log.append(("AG", tuple(x.shape), dim))
        return torch.cat([x] * 8, dim=dim)

    m._all_gather = ag
    m._breakable_capture = lambda: False
    return a, b, log


@pytest.mark.parametrize("mode", ["look", "all"])
def test_engram_pack_sequence(mode):
    m = load_step({"DSV41_STEP_ENGRAM_PACK": mode})
    a, b, log = _engram_world(m, mode)
    h = torch.zeros(5, 2, dtype=torch.int64)
    for _ in range(2):  # two forwards
        log.clear()
        a.forward(h)
        b.forward(h)
        ags = [x for x in log if x[0] == "AG"]
        assert len(ags) == (1 if mode == "look" else 2), log  # stock: 4 (2 rows + 2 wkv)
        assert ("stock_embed", 0) not in log and ("stock_embed", 1) not in log
        if mode == "all":
            assert [x for x in log if x[0] == "gemm"] == [("gemm", "a", (5, 384)), ("gemm", "b", (5, 384))]
            assert ("stock_wkv", "b") not in log
        else:
            assert ("stock_wkv", "a") in log and ("stock_wkv", "b") in log


def test_engram_pack_skips_large_T_and_hoist_degrades_all():
    m = load_step({"DSV41_STEP_ENGRAM_PACK": "all", "DSV41_PRIO_ENGRAM_HOIST": "1"})
    a, b, log = _engram_world(m, "all")
    h = torch.zeros(5, 2, dtype=torch.int64)
    a.forward(h)
    b.forward(h)
    assert m.PK.mode == "look"
    assert ("stock_wkv", "a") in log and ("stock_wkv", "b") in log
    log.clear()
    big = torch.zeros(m.PACK_MAX_T + 1, 2, dtype=torch.int64)
    a.forward(big)
    b.forward(big)
    assert ("stock_embed", 0) in log and ("stock_embed", 1) in log


def test_engram_pack_under_r5_hoist_order():
    """r5-prio hoist: embed(a) at layer 0, embed(b) at layer 13, then the takes. Rows stay exact."""
    m = load_step({"DSV41_STEP_ENGRAM_PACK": "look", "DSV41_PRIO_ENGRAM_HOIST": "1"})
    a, b, log = _engram_world(m, "look")
    h = torch.zeros(3, 2, dtype=torch.int64)
    ra = a.embed(h)
    rb = b.embed(h)
    assert [x[0] for x in log].count("AG") == 1 and ra.shape == rb.shape == (3, 24, 16)
    assert torch.equal(rb, (b.staged_rows[:3] + 1).repeat(1, 8, 1))


# ------------------------------------------------------------------------------------------------ C4 races
class _VC:
    """Vector-clock CUDA facade: streams, events, accesses; race = conflicting accesses without happens-before."""

    def __init__(self):
        self.n = 0
        self.cur = None
        self.acc = []
        self.main = self.Stream()
        self.cur = self.main

    def Stream(self):
        vc = self

        class S:
            def __init__(s):
                s.id = vc.n
                vc.n += 1
                s.clock = {s.id: 0}

            def wait_event(s, ev):
                for k, v in ev.clock.items():
                    s.clock[k] = max(s.clock.get(k, 0), v)

            def tick(s):
                s.clock[s.id] += 1
                return dict(s.clock)

        return S()

    def Event(self):
        class E:
            clock = {}

            def record(e, s):
                e.clock = s.tick()

        return E()

    def current_stream(self):
        return self.cur

    @contextlib.contextmanager
    def stream(self, s):
        old, self.cur = self.cur, s
        try:
            yield
        finally:
            self.cur = old

    def is_available(self):
        return True

    def is_current_stream_capturing(self):
        return False

    def access(self, buf, kind):
        self.acc.append((buf, kind, self.cur.id, self.cur.tick()))

    def races(self):
        out = []
        for i, (b1, k1, s1, c1) in enumerate(self.acc):
            for b2, k2, s2, c2 in self.acc[i + 1:]:
                if b1 != b2 or "w" not in (k1, k2) or s1 == s2:
                    continue
                if c2.get(s1, -1) < c1[s1]:  # access 2 does not happen after access 1
                    out.append((b1, k1, s1, k2, s2))
        return out


class _Split:
    def __init__(self, vc):
        self.vc, self.checked = vc, {4: True}

    def local(self, x):
        self.vc.access("aux", "r")
        self.vc.access("y", "w")
        return x[:, :2].clone(), tuple(x.shape)

    def gather(self, y, shape):
        self.vc.access("y", "r")
        return y


def _c4_step(m, vc, st, aux, mutant=None):
    vc.access("aux", "w")  # the target graph writes the taps
    vc.access("screen", "w")
    launch = m._Launch(st, 4, aux)
    if mutant == "no_screen_wait" and m.C4S.stream is None:
        m.C4S.stream, m.C4S.ev_screen, m.C4S.ev_done = vc.Stream(), vc.Event(), vc.Event()
        m.C4S.stream.wait_event = lambda ev: None  # the side stream never waits for the screen / taps
    launch.after_screen()
    vc.access("refine", "w")
    if mutant == "aux_write":
        vc.access("aux", "w")  # something on main writes a tap between the fork and propose
    x = torch.cat(aux, dim=-1)
    if mutant == "no_join":
        p, m.C4S.pending = m.C4S.pending, None
        out = st.gather(p.y, p.shape)  # consumer without the event wait
    else:
        out = m.c4_take(st, x, 4)
    assert out is not None
    return out


@pytest.mark.parametrize("mutant", [None, "no_join", "aux_write", "no_screen_wait"])
def test_c4_happens_before(mutant):
    m = load_step({"DSV41_STEP_C4": "1"})
    vc = _VC()
    m._cu = vc
    m.C4S.__init__()
    st = _Split(vc)
    aux = [torch.randn(4, 3) for _ in range(3)]
    for _ in range(3):  # three steps: the next target graph write must be after the previous step's read
        _c4_step(m, vc, st, aux, mutant)
    r = vc.races()
    if mutant is None:
        assert r == [], r
    else:
        assert r, f"mutant {mutant} not caught"


def test_c4_drop_when_not_consumed_joins():
    m = load_step({"DSV41_STEP_C4": "1"})
    vc = _VC()
    m._cu = vc
    m.C4S.__init__()
    st = _Split(vc)
    aux = [torch.randn(4, 3) for _ in range(3)]
    vc.access("aux", "w")
    m._Launch(st, 4, aux).after_screen()
    m.c4_drop("test")
    vc.access("aux", "w")  # next target replay
    assert vc.races() == [] and m.COUNTS["c4_drop"] == 1


def test_c4_take_rejects_wrong_operand():
    m = load_step({"DSV41_STEP_C4": "1"})
    vc = _VC()
    m._cu = vc
    m.C4S.__init__()
    st = _Split(vc)
    aux = [torch.randn(4, 3) for _ in range(3)]
    m._Launch(st, 4, aux).after_screen()
    assert m.c4_take(st, torch.randn(4, 8), 4) is None and m.C4S.pending is None
    m._Launch(st, 4, aux).after_screen()
    assert m.c4_take(st, torch.cat(aux, -1), 3) is None


@pytest.mark.parametrize("corrupt", [False, True])
def test_c4_audit_serves_stock_y(corrupt):
    """r6-review: during the DSV41_STEP_C4_AUDIT window the stock local GEMM's y is served (exact even if the hoisted
    y were wrong); a wrong hoisted y is reported and disables C4 at step N; after the window the hoisted y is served."""
    m = load_step({"DSV41_STEP_C4": "1", "DSV41_STEP_C4_AUDIT": "2"})
    vc = _VC()
    m._cu = vc
    m.C4S.__init__()
    st = _Split(vc)
    for _ in range(2):
        aux = [torch.randn(4, 3) for _ in range(3)]
        m._Launch(st, 4, aux).after_screen()
        if corrupt:
            m.C4S.pending.y.add_(1.0)  # a hoisted result that is not the stock one
        x = torch.cat(aux, -1)
        out = m.c4_take(st, x, 4)
        assert torch.equal(out, x[:, :2])  # served == stock local GEMM
    assert m.C4S.audit_n == 2 and m.C4S.disabled == corrupt
    assert (int(m.C4S.audit_ctr[1]) > 0) == corrupt and int(m.C4S.audit_ctr[0]) == 2
    if not corrupt:  # after the window: the hoisted y itself is served
        aux = [torch.randn(4, 3) for _ in range(3)]
        m._Launch(st, 4, aux).after_screen()
        m.C4S.pending.y.add_(1.0)
        out = m.c4_take(st, torch.cat(aux, -1), 4)
        assert torch.equal(out, torch.cat(aux, -1)[:, :2] + 1.0)


def test_flags_parse():
    with pytest.raises(ValueError):
        load_step({"DSV41_STEP_ENGRAM_PACK": "bogus"})
    m = load_step({})
    assert not m.COLEMB and m.PACK == "0" and not m.C4
