"""r1-attn: CPU plan tests for DSV41_IDXPAR.

Runs the REAL `_prepare_and_attn` source of the base and of the patched attention.py (extracted with ast and
exec'd), the REAL multi_stream_utils.maybe_execute_in_parallel and the REAL helper dsv41_idxpar.py against a
fake CUDA layer that logs every kernel (with the stream it is issued on, the buffers it reads / writes),
every event record / wait and every module call (hook order). A vector-clock happens-before analysis then
checks, per layer and across a 9-layer sequence:
  * flag off (or flag on but eager / unprimed / breakable / profile): the log equals the stock log exactly;
  * flag on, captured: the same kernels with the same inputs (dataflow producers identical to stock),
    the same module-call (L2 recorder / wq_b ordinal) order, every read after its producer, no unordered
    write/write or read/write pair, every branch joined into main before return, the indexer q chain on
    the priority -2 stream, the compressor chain on priority -1, the main q chain on aux_stream_list[1],
    and the WOA fork (last wq_b post-hook) issued from the indexer-q stream right after the indexer wq_b.

Run: python3 tests/test_plan.py   (needs torch, CPU build is fine)
"""
import ast
import collections
import contextlib
import importlib.util
import os
import sys
import textwrap
import types
import unittest

import torch

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(KIT, "tests"))
import test_install  # noqa: E402

BASE_ATTN = open(os.path.join(KIT, "base/models/deepseek_v41/attention.py")).read()
PATCHED_ATTN = test_install.INST.patched(BASE_ATTN)


# ----------------------------------------------------------------------------- fake CUDA ------
class Tracer:
    def __init__(self):
        self.log = []
        self.capturing = True
        self.stack = []
        self.hooks = []  # callables(name, stream)


TR = Tracer()


class FakeStream:
    _n = collections.Counter()

    def __init__(self, device=None, priority=0, name=None):
        if name is None:
            FakeStream._n[priority] += 1
            name = f"pool[{priority}]#{FakeStream._n[priority]}"
        self.name = name
        self.priority = priority

    @staticmethod
    def priority_range():
        return (0, -3)

    def wait_stream(self, other):
        ev = FakeEvent()
        ev.record(other)
        ev.wait(self)

    def __repr__(self):
        return self.name


MAIN = FakeStream(name="main")


class FakeEvent:
    _n = 0

    def __init__(self, *a, **k):
        FakeEvent._n += 1
        self.name = f"ev{FakeEvent._n}"

    def record(self, stream=None):
        TR.log.append(("rec", self, stream or cur()))

    def wait(self, stream=None):
        TR.log.append(("wait", self, stream or cur()))


def cur():
    return TR.stack[-1] if TR.stack else MAIN


@contextlib.contextmanager
def stream_ctx(s):
    TR.stack.append(s)
    try:
        yield
    finally:
        TR.stack.pop()


fake_cuda = types.SimpleNamespace(
    Stream=FakeStream, Event=FakeEvent, current_stream=lambda device=None: cur(), stream=stream_ctx,
    is_current_stream_capturing=lambda: TR.capturing, current_device=lambda: 0,
)


def op(name, reads, writes):
    TR.log.append(("op", name, cur(), tuple(reads), tuple(writes)))


def hook(module):
    """Module __call__: forward pre-hooks fire here (the L2 recorder's weight-read events)."""
    TR.log.append(("call", module, cur()))


def post(module):
    """Forward post-hooks (dsv41_l2_prefetch registers one on every wq_b: the WOA fork point)."""
    for h in TR.hooks:
        h(module, cur())


class T:
    """Fake tensor: just a buffer name."""

    def __init__(self, name):
        self.name = name

    def view(self, *a):
        return self


# ----------------------------------------------------------------------------- fake model -----
class FakeCompressor:
    def __init__(self, lid):
        self.lid = lid

    def __call__(self, kv_score, positions):
        hook(f"L{self.lid}.compressor")
        op("compress", [kv_score.name, positions.name, f"state@{self.lid}"], [f"latent@{self.lid}", f"state@{self.lid}"])
        return T(f"latent@{self.lid}")

    def insert_cache(self, latent, positions, rope):
        if latent is None:
            return
        op("comp_insert", [latent.name, positions.name], [f"comp_cache@{self.lid}"])


class FakeIndexer:
    """Mirrors DeepseekV4Indexer.forward's capture path (the short-context branch is never taken under capture)."""

    def __init__(self, lid, owns_k, ksrc):
        self.lid, self.owns_k, self.ksrc = lid, owns_k, ksrc

    def __call__(self, qr, latent, weights, positions, rope, qr_scale=None):
        hook(f"L{self.lid}.indexer")
        if self.owns_k:
            self._produce_k(latent, positions, rope)
        return self.forward_q(qr, qr_scale, weights, positions, rope)

    def _produce_k(self, latent, positions, rope):
        if latent is None:
            return
        hook(f"L{self.lid}.indexer.wk")
        op("wk", [latent.name], [f"k_pre@{self.lid}"])
        op("kstore", [f"k_pre@{self.lid}", positions.name], [f"idx_kcache@{self.lid}"])

    def forward_q(self, qr, qr_scale, weights, positions, rope):
        hook(f"L{self.lid}.indexer.wq_b")
        op("idx_wq_b", [qr.name], [f"iq_raw@{self.lid}"])
        post(f"L{self.lid}.indexer.wq_b")
        op("idx_q_rope_quant", [f"iq_raw@{self.lid}", positions.name, weights.name],
           [f"index_q@{self.lid}", f"index_q_scale@{self.lid}", f"index_w@{self.lid}"])
        return T(f"index_q@{self.lid}"), T(f"index_q_scale@{self.lid}"), T(f"index_w@{self.lid}")


class FakeAttn:
    def __init__(self, lid, aux, kv_src, idx_src):
        self.layer_id = lid
        self.aux_stream_list = aux
        self.n_local_heads, self.head_dim = 8, 512
        self.indexer_rotary_emb = self.rotary_emb = object()
        self.ln_events = [FakeEvent() for _ in range(4)]
        self.kv_source_layer_id = kv_src
        is_kv = lid == kv_src
        self.compressor = FakeCompressor(lid) if is_kv else None
        self.indexer = FakeIndexer(lid, is_kv, kv_src) if lid == idx_src else None
        self._dsv41_idxpar = None

    def _wq_b_proj(self, qr, qr_scale=None):
        hook(f"L{self.layer_id}.wq_b")
        op("wq_b", [qr.name], [f"q_raw@{self.layer_id}"])
        post(f"L{self.layer_id}.wq_b")
        return T(f"q_raw@{self.layer_id}")

    def _fused_qnorm_rope_kv_insert(self, q, kv, positions, md):
        op("swa_insert", [q.name, kv.name, positions.name], [f"q@{self.layer_id}", f"swa_cache@{self.layer_id}"])
        return T(f"q@{self.layer_id}")

    def _sparse_indexer_and_attn(self, hidden, index_q, index_q_scale, index_w, q, kv, positions, out):
        lid = self.layer_id
        if self.indexer is not None and index_q is not None:
            op("indexer_op", [index_q.name, index_q_scale.name, index_w.name, f"idx_kcache@{self.indexer.ksrc}",
                              hidden.name], ["topk"])
        op("mqa", [q.name, f"swa_cache@{lid}", f"comp_cache@{self.kv_source_layer_id}", "topk", positions.name],
           [out.name])


class FakeFC:
    def __init__(self, md):
        self.attn_metadata = md


FC = FakeFC({"x": 1})


# ----------------------------------------------------------------------------- code under test
def _method_src(src, cls, name):
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for f in node.body:
                if isinstance(f, ast.FunctionDef) and f.name == name:
                    seg = ast.get_source_segment(src, f)
                    return textwrap.dedent(" " * f.col_offset + seg)
    raise KeyError(name)


def _load(path, name, extra_modules=None):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# real maybe_execute_in_parallel (base copy) with the breakable stub
_bk = types.ModuleType("vllm.compilation.breakable_cudagraph")


class _BCG:
    active = False

    @classmethod
    def is_active(cls):
        return cls.active


_bk.BreakableCUDAGraphCapture = _BCG
for n in ("vllm", "vllm.compilation"):
    sys.modules.setdefault(n, types.ModuleType(n))
sys.modules["vllm.compilation.breakable_cudagraph"] = _bk
_lg = types.ModuleType("vllm.logger")
_lg.init_logger = lambda name: __import__("logging").getLogger(name)
sys.modules["vllm.logger"] = _lg
MSU = _load(os.path.join(KIT, "base/utils/multi_stream_utils.py"), "msu_base")

os.environ["DSV41_IDXPAR"] = "1"
HELPER = _load(os.path.join(KIT, "files/models/deepseek_v41/nvidia/dsv41_idxpar.py"), "dsv41_idxpar_t")
HELPER._cuda = fake_cuda
HELPER._get_forward_context = lambda: FC
HELPER._breakable_active = lambda: _BCG.active
HELPER._warm_blas = lambda s, dev: op("warm_blas", [], [f"warm@{s.name}"])


def _make_fn(src, flag):
    fn_src = _method_src(src, "DeepseekV4Attention", "_prepare_and_attn")
    ns = {"get_forward_context": lambda: FC, "maybe_execute_in_parallel": MSU.maybe_execute_in_parallel,
          "torch": torch, "QuantizedActivation": type("QA", (), {}), "_DSV41_IDXPAR": flag}
    exec(fn_src, ns)
    return ns["_prepare_and_attn"]


STOCK = _make_fn(BASE_ATTN, None)
PATCHED_OFF = _make_fn(PATCHED_ATTN, False)
PATCHED_ON = _make_fn(PATCHED_ATTN, True)


@contextlib.contextmanager
def patched_torch_cuda():
    """maybe_execute_in_parallel uses torch.cuda.stream(...) and Event.record()/wait() directly."""
    saved = {k: getattr(torch.cuda, k) for k in ("stream", "current_stream")}
    torch.cuda.stream = stream_ctx
    torch.cuda.current_stream = lambda device=None: cur()
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(torch.cuda, k, v)


# config: kv_source [2,8,14,20], index_source [2,8,14,20,24,28,32,36]
KV = [2, 8, 14, 20]
IDX = KV + [24, 28, 32, 36]


def src_of(lid, ids):
    c = [s for s in ids if s <= lid]
    return max(c) if c else None


def make_layer(lid, aux, flag_on):
    a = FakeAttn(lid, aux, src_of(lid, KV), lid if lid in IDX else None)
    if flag_on:
        a._dsv41_idxpar = HELPER.LayerPlan.create(a)
    return a


def call(fn, a):
    lid = a.layer_id
    # inputs are produced on main before _prepare_and_attn (input projections joined, q_kv_norm on main)
    op("prologue", [], [f"hidden@{lid}", f"qr@{lid}", f"kv@{lid}", f"kv_score@{lid}", f"iw@{lid}", f"pos@{lid}"])
    fn(a, T(f"hidden@{lid}"), T(f"qr@{lid}"), T(f"kv@{lid}"), None, T(f"kv_score@{lid}"), T(f"iw@{lid}"),
       T(f"pos@{lid}"), T(f"out@{lid}"))
    op("epilogue", [f"out@{lid}"], [f"h_next@{lid}"])  # o_proj etc. on main


def run(fn, layers, capturing=True, md=None, breakable=False):
    TR.log.clear()
    TR.capturing = capturing
    FC.attn_metadata = {"x": 1} if md is None else md
    _BCG.active = breakable
    with patched_torch_cuda():
        for a in layers:
            call(fn, a)
    return list(TR.log)


def norm(log):
    """Comparable form: stream/event objects -> names (events renamed by first appearance)."""
    names, out = {}, []
    for e in log:
        if e[0] == "op":
            out.append(("op", e[1], e[2].name, e[3], e[4]))
        elif e[0] == "call":
            out.append(("call", e[1], e[2].name))
        else:
            names.setdefault(id(e[1]), f"E{len(names)}")
            out.append((e[0], names[id(e[1])], e[2].name))
    return out


# ----------------------------------------------------------------------------- HB analysis ----
def analyse(log):
    """Vector clocks per stream. Returns ops [(idx, name, stream, reads, writes, vc)] and final main vc."""
    vc = collections.defaultdict(lambda: collections.Counter())
    evs = {}
    ops = []
    for e in log:
        if e[0] == "op":
            s = e[2].name
            vc[s][s] += 1
            ops.append((len(ops), e[1], s, e[3], e[4], collections.Counter(vc[s])))
        elif e[0] == "rec":
            evs[id(e[1])] = collections.Counter(vc[e[2].name])
        elif e[0] == "wait":
            s = e[2].name
            for k, v in evs.get(id(e[1]), {}).items():
                vc[s][k] = max(vc[s][k], v)
    return ops, vc


def hb(a, b):
    """op a happens-before op b."""
    return a[0] != b[0] and a[5][a[2]] <= b[5][a[2]] and (a[2] != b[2] or a[0] < b[0])


def check_hb(test, log):
    ops, vc = analyse(log)
    last_writer = {}
    for o in ops:
        for r in o[3]:
            w = last_writer.get(r)
            if w is not None:
                test.assertTrue(hb(w, o), f"{o[1]}@{o[2]} reads {r} without waiting for {w[1]}@{w[2]}")
        for w in o[4]:
            last_writer[w] = o
    for i, a in enumerate(ops):
        for b in ops[i + 1:]:
            if hb(a, b) or hb(b, a):
                continue
            ra, wa, rb, wb = set(a[3]), set(a[4]), set(b[3]), set(b[4])
            test.assertFalse(wa & (rb | wb) or wb & ra,
                             f"race: {a[1]}@{a[2]} vs {b[1]}@{b[2]} on {(wa & (rb | wb)) | (wb & ra)}")
    # every op is joined into main by the end
    end = vc["main"]
    for o in ops:
        test.assertLessEqual(o[5][o[2]], end[o[2]], f"{o[1]}@{o[2]} not joined into main")
    return ops


def producers(log):
    """{(op name, buffer read): producing op name} -- the dataflow graph."""
    out, lw = set(), {}
    for e in log:
        if e[0] != "op":
            continue
        for r in e[3]:
            out.add((e[1], r, lw.get(r)))
        for w in e[4]:
            lw[w] = e[1]
    return out


def kernels(log):
    return collections.Counter((e[1], e[3], e[4]) for e in log if e[0] == "op")


def calls(log, drop=None):
    return [e[1] for e in log if e[0] == "call" and not (drop and e[1].endswith(drop))]


# ----------------------------------------------------------------------------- tests ----------
SEQ = [1, 2, 3, 8, 20, 21, 24, 28, 36]  # sliding/consumer, kv-source (ratio 1 / 2), index-only


class TestIdxpar(unittest.TestCase):
    def setUp(self):
        HELPER._SHARED.clear()
        HELPER._LOGGED.clear()
        self.aux = [FakeStream(name="aux0"), FakeStream(name="aux1"), FakeStream(name="aux2")]
        TR.hooks = []

    def layers(self, flag_on, prime=True):
        ls = [make_layer(l, self.aux, flag_on) for l in SEQ]
        if flag_on and prime:
            for a in ls:
                if a._dsv41_idxpar is not None:
                    a._dsv41_idxpar.ready = True
                    a._dsv41_idxpar.shared.warmed = True
        return ls

    def test_layerplan_only_on_index_sources(self):
        ls = self.layers(True)
        have = [a.layer_id for a in ls if a._dsv41_idxpar is not None]
        self.assertEqual(have, [2, 8, 20, 24, 28, 36])
        p = ls[1]._dsv41_idxpar
        self.assertIs(p.low, self.aux[1])
        self.assertEqual((p.shared.hi.priority, p.shared.med.priority), (-2, -1))
        # one shared stream pair per device
        self.assertIs(ls[1]._dsv41_idxpar.shared, ls[-1]._dsv41_idxpar.shared)
        # no aux streams (ROCm) / inconsistent compressor-owns_k: stock
        a = make_layer(24, None, False)
        self.assertIsNone(HELPER.LayerPlan.create(a))
        a = make_layer(2, self.aux, False)
        a.indexer.owns_k = False
        self.assertIsNone(HELPER.LayerPlan.create(a))

    def test_flag_off_identical(self):
        stock = norm(run(STOCK, self.layers(False)))
        off = norm(run(PATCHED_OFF, self.layers(False)))
        self.assertEqual(stock, off)
        # flag on but the layer has no plan object (non-indexer layer) -> also identical
        on_noplan = norm(run(PATCHED_ON, self.layers(False)))
        self.assertEqual(stock, on_noplan)

    def test_stock_itself_is_race_free(self):
        check_hb(self, run(STOCK, self.layers(False)))

    def test_flag_on_eager_and_fallbacks_are_stock(self):
        stock = norm(run(STOCK, self.layers(False), capturing=False))
        ls = self.layers(True, prime=False)
        first = norm(run(PATCHED_ON, ls, capturing=False))
        # first eager forward primes: warm cuBLAS on the two new streams (once), record each layer's events
        prim_ops = [e for e in first if e[0] == "op" and e[1] == "warm_blas"]
        self.assertEqual(len(prim_ops), 2)  # once per new stream, not per layer
        self.assertTrue(all(a._dsv41_idxpar.ready for a in ls if a._dsv41_idxpar is not None))
        # apart from the priming, the first eager forward runs exactly the stock kernels / calls
        self.assertEqual([e for e in first if e[0] in ("op", "call") and e not in prim_ops],
                         [e for e in stock if e[0] in ("op", "call")])
        second = norm(run(PATCHED_ON, ls, capturing=False))
        self.assertEqual(second, stock)
        # breakable (piecewise) capture and profile runs: stock
        self.assertEqual(norm(run(PATCHED_ON, ls, breakable=True)),
                         norm(run(STOCK, self.layers(False), breakable=True)))
        self.assertEqual(norm(run(PATCHED_ON, ls, md=[{"x": 1}])),
                         norm(run(STOCK, self.layers(False), md=[{"x": 1}])))

    def test_unprimed_capture_falls_back(self):
        ls = self.layers(True, prime=False)
        self.assertEqual(norm(run(PATCHED_ON, ls)), norm(run(STOCK, self.layers(False))))

    def test_captured_schedule(self):
        stock_log = run(STOCK, self.layers(False))
        ls = self.layers(True)
        on_log = run(PATCHED_ON, ls)
        # same kernels with the same operands, same dataflow producers
        self.assertEqual(kernels(on_log), kernels(stock_log))
        self.assertEqual(producers(on_log), producers(stock_log))
        # same module-call order (L2 recorder events, wq_b post-hook ordinals)
        # (IDXPAR calls indexer._produce_k / forward_q directly, not the DeepseekV4Indexer module itself;
        # that module holds no direct parameter, so the L2 recorder registers no hook on it)
        self.assertEqual(calls(on_log, ".indexer"), calls(stock_log, ".indexer"))
        # ordering / races / joins
        check_hb(self, on_log)
        # placement
        sh = ls[1]._dsv41_idxpar.shared
        where = collections.defaultdict(set)
        for e in on_log:
            if e[0] == "op":
                where[e[1]].add((e[2].name, e[2].priority))
        self.assertEqual(where["idx_wq_b"], {(sh.hi.name, -2)})
        self.assertEqual(where["idx_q_rope_quant"], {(sh.hi.name, -2)})
        for k in ("compress", "wk", "kstore", "comp_insert"):
            self.assertEqual(where[k], {(sh.med.name, -1)}, k)
        # main q chain: aux1 in indexer layers, main elsewhere (stock)
        self.assertEqual(where["wq_b"], {("aux1", 0), ("main", 0)})
        self.assertEqual(where["indexer_op"], {("main", 0)})
        self.assertEqual(where["mqa"], {("main", 0)})
        # per indexer layer the host order is aux1 branch, then med, then hi
        for a in ls:
            if a._dsv41_idxpar is None:
                continue
            seq = [e[2].name for e in on_log if e[0] == "op" and e[1] in ("wq_b", "compress", "idx_wq_b")
                   and any(w.endswith(f"@{a.layer_id}") for w in e[4])]
            want = ["aux1"] + ([sh.med.name] if a.compressor else []) + [sh.hi.name]
            self.assertEqual(seq, want, a.layer_id)

    def test_woa_fork_point(self):
        """dsv41_l2_prefetch forks the WOA window at the LAST wq_b post-hook of the attention block with
        side2.wait_stream(torch.cuda.current_stream()). With IDXPAR that is the indexer wq_b on the high
        stream, right after the indexer wq_b kernel -- never after the main wq_b."""
        side2 = FakeStream(name="l2pf_side2")

        def woa(module, stream):
            if module.endswith(".indexer.wq_b"):
                side2.wait_stream(stream)
                op("woa_prefetch", [], [f"woa@{module}"])  # no tensor read / write of the model

        def woa_side(module, stream):
            if module.endswith(".indexer.wq_b"):
                with stream_ctx(side2):
                    woa(module, stream)

        TR.hooks = [woa_side]
        ls = self.layers(True)
        log = run(PATCHED_ON, ls)
        ops, _ = analyse(log)
        sh = ls[1]._dsv41_idxpar.shared
        for o in ops:
            if o[1] == "woa_prefetch":
                lid = o[4][0].split("L")[1].split(".")[0]
                idx = [p for p in ops if p[1] == "idx_wq_b" and p[4][0].endswith(f"@{lid}")][0]
                main_q = [p for p in ops if p[1] == "wq_b" and p[4][0].endswith(f"@{lid}")][0]
                self.assertTrue(hb(idx, o))
                self.assertEqual(idx[2], sh.hi.name)
                self.assertFalse(hb(main_q, o), "WOA must not wait for the main wq_b")
        self.assertEqual(sum(1 for o in ops if o[1] == "woa_prefetch"), 6)

    def test_flat_mode_priorities(self):
        os.environ["DSV41_IDXPAR"] = "flat"
        try:
            h = _load(os.path.join(KIT, "files/models/deepseek_v41/nvidia/dsv41_idxpar.py"), "idxpar_flat")
        finally:
            os.environ["DSV41_IDXPAR"] = "1"
        self.assertEqual((h.HI_PRIORITY, h.MED_PRIORITY), (0, 0))
        os.environ["DSV41_IDXPAR"] = "bogus"
        try:
            with self.assertRaises(ValueError):
                _load(os.path.join(KIT, "files/models/deepseek_v41/nvidia/dsv41_idxpar.py"), "idxpar_bad")
        finally:
            os.environ["DSV41_IDXPAR"] = "1"


if __name__ == "__main__":
    unittest.main(verbosity=2)
