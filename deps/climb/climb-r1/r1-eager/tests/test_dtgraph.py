"""Unit tests for dsv41_dtgraph (CPU only). "meta" tensors stand in for device tensors; a fake graph
backend records begin/close/fence so segmentation, staging and ordering can be checked without a GPU.
"""
import os
import sys
import types
from collections import Counter

import pytest
import torch

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KIT)
import dsv41_dtgraph as D  # noqa: E402

aten = torch.ops.aten
META = lambda t: getattr(t, "device", None) is not None and t.device.type == "meta"  # noqa: E731


# ---------------------------------------------------------------------------- flag parsing
@pytest.mark.parametrize("raw,regions,update,warm,refused", [
    ("", set(), False, 2, []),
    ("0", set(), False, 2, []),
    ("1", {"dt"}, False, 2, []),
    ("dt", {"dt"}, False, 2, []),
    ("update", {"dt"}, True, 2, []),
    ("1,warm=5,seg=2,log=0", {"dt"}, False, 5, []),
    ("td", set(), False, 2, ["td"]),
    ("dt,td", {"dt"}, False, 2, ["td"]),
])
def test_parse_flag(raw, regions, update, warm, refused):
    c = D.parse_flag(raw)
    assert c["regions"] == regions and c["update"] == update and c["warm"] == warm
    assert c["refused"] == refused


def test_parse_flag_rejects_unknown():
    with pytest.raises(ValueError):
        D.parse_flag("dtt")


def test_module_off_by_default():
    assert os.environ.get("DSV41_DTGRAPH") in (None, "") and D.ON is False


# ---------------------------------------------------------------------------- classification
def m(*shape, dtype=torch.int32):
    return torch.empty(*shape, dtype=dtype, device="meta")


def c(*shape, dtype=torch.int32):
    return torch.zeros(*shape, dtype=dtype)


def cls(op, *args, **kw):
    return D.classify(op, args, kw, META)


def test_classify_h2d():
    assert cls(aten.copy_.default, m(4), c(4), True) == (D.STAGE, "h2d")
    assert cls(aten.copy_.default, m(4), c(4), False) == (D.HAZ, "h2d_blocking")
    assert cls(aten.copy_.default, m(4), c(4, dtype=torch.int64), True) == (D.HAZ, "h2d_cast")
    assert cls(aten.copy_.default, m(4, 4), c(4, 4).t(), True) == (D.HAZ, "h2d_noncontig")
    assert cls(aten._to_copy.default, c(4), device=torch.device("meta"), non_blocking=True) == (D.STAGE, "h2d")
    assert cls(aten._to_copy.default, c(4), device=torch.device("meta"))[0] == D.HAZ
    assert cls(aten._to_copy.default, c(4), device=torch.device("meta"), dtype=torch.int64,
               non_blocking=True) == (D.HAZ, "h2d_cast")


def test_classify_arena_owned_source_passes():
    a = D.Arena(pin=False)
    v = a.take(c(4))
    assert D.classify(aten.copy_.default, (m(4), v, True), {}, META, a.owns) == (D.PASS, "")
    assert D.classify(aten.copy_.default, (m(4), c(4), True), {}, META, a.owns) == (D.STAGE, "h2d")


def test_classify_d2h_sync_rng_index_coll():
    assert cls(aten.copy_.default, c(4), m(4), True) == (D.HAZ, "d2h")
    assert cls(aten._to_copy.default, m(4), device=torch.device("cpu"))[1] == "d2h"
    assert cls(aten._local_scalar_dense.default, m(1)) == (D.HAZ, "sync:_local_scalar_dense")
    assert cls(aten.nonzero.default, m(4))[0] == D.HAZ
    assert cls(aten.where.default, m(4, dtype=torch.bool)) == (D.HAZ, "sync:where")
    assert cls(aten.where.self, m(4, dtype=torch.bool), m(4), m(4)) == (D.PASS, "")
    assert cls(aten.uniform_.default, m(4, dtype=torch.float32))[0] == D.HAZ
    assert cls(aten.uniform_.default, c(4, dtype=torch.float32)) == (D.PASS_CPU, "")
    assert cls(aten.index.Tensor, m(4), [m(4, dtype=torch.bool)]) == (D.HAZ, "mask_index")
    assert cls(aten.index.Tensor, m(4), [m(4, dtype=torch.int64)]) == (D.PASS, "")
    assert cls(aten.record_stream.default, m(4), None)[0] == D.HAZ
    assert cls(aten.add.Tensor, m(4), c(4))[1] == "cpu_arg"
    assert cls(aten.add.Tensor, m(4), c(()))[0] == D.PASS  # 0-dim CPU scalar is a launch argument


def test_classify_pass_and_cpu():
    assert cls(aten.add.Tensor, m(4), m(4)) == (D.PASS, "")
    assert cls(aten.add.Tensor, c(4), c(4)) == (D.PASS_CPU, "")
    assert cls(aten._local_scalar_dense.default, c(1)) == (D.PASS_CPU, "")  # CPU .item(): no sync
    assert cls(aten.slice.Tensor, m(8), 0, 0, 4) == (D.PASS, "")
    assert cls(aten.empty.memory_format, [4], dtype=torch.int32, device=torch.device("meta")) == (D.PASS, "")
    assert cls(aten.empty.memory_format, [4], dtype=torch.int32, device=torch.device("cpu"),
               pin_memory=True) == (D.PIN, "pin_alloc")
    assert cls(aten._pin_memory.default, c(4)) == (D.PIN, "pin_copy")


def test_classify_collectives():
    class Op:
        def __init__(self, n):
            self._schema = types.SimpleNamespace(name=n)
    assert D.classify(Op("_c10d_functional::all_reduce"), (m(4),), {}, META)[0] == D.HAZ
    assert D.classify(Op("vllm::all_gather"), (m(4),), {}, META)[0] == D.HAZ
    assert D.classify(Op("vllm::rms_norm"), (m(4),), {}, META) == (D.PASS, "")


# ---------------------------------------------------------------------------- arena
def test_arena_take_alloc_align_full_and_slots():
    a = D.Arena(slot_bytes=1024, nslot=2, pin=False)
    a.start(1)
    x = torch.arange(10, dtype=torch.int64)
    v = a.take(x)
    assert torch.equal(v, x) and v.dtype == x.dtype and v.shape == x.shape
    assert a.owns(v) and not a.owns(x)
    assert v.data_ptr() - a.buf.data_ptr() == 1024  # slot 1
    w = a.alloc([3], torch.int32)
    assert (w.data_ptr() - a.buf.data_ptr()) % D.ALIGN == 0 and w.shape == (3,)
    assert a.take(torch.zeros(1024, dtype=torch.uint8)) is None  # does not fit any more
    a.start(0)
    v0 = a.take(torch.tensor([7], dtype=torch.int32))
    assert v0.data_ptr() == a.buf.data_ptr() and int(v0) == 7
    assert torch.equal(v, x)  # slot 1 untouched by slot 0 use


def test_pin_result():
    a = D.Arena(pin=False)
    t = D.pin_result(aten.empty.memory_format, ([2, 3],), {"dtype": torch.int64, "pin_memory": True}, a)
    assert t.shape == (2, 3) and t.dtype == torch.int64 and t.is_contiguous() and a.owns(t)
    t = D.pin_result(aten.empty_strided.default, ([2, 3], [3, 1]), {"dtype": torch.float32}, a)
    assert t is not None and t.stride() == (3, 1)
    assert D.pin_result(aten.empty_strided.default, ([2, 3], [1, 2]), {"dtype": torch.float32}, a) is None
    src = torch.arange(5, dtype=torch.int32)
    t = D.pin_result(aten._pin_memory.default, (src,), {}, a)
    assert torch.equal(t, src) and a.owns(t)


# ---------------------------------------------------------------------------- region / segments
class FakeStream:
    def __init__(self, h):
        self.cuda_stream = h


class FakeBackend:
    def __init__(self, cur):
        self.log = []
        self.cur = cur  # list holding the "current stream"
        self.n_begin = 0
        self.waited = []

    def stream(self):
        return self.cur[0]

    def wait_slot(self, slot):
        self.waited.append(slot)
        return False

    def begin(self):
        self.n_begin += 1
        g = f"g{self.n_begin}"
        self.log.append(("begin", g))
        return g

    def end_and_launch(self, g, seg, slot, stream):
        self.log.append(("launch", g, seg, slot))

    def fence(self, slot, stream):
        self.log.append(("fence", slot))


class Op:
    """Callable with an aten-like schema name; records whether it ran inside a capture."""

    def __init__(self, name, region_ref, trace):
        self._schema = types.SimpleNamespace(name=name)
        self.name, self.r, self.trace = name, region_ref, trace

    def __call__(self, *a, **k):
        r = self.r[0]
        self.trace.append((self.name, "cap" if r.capturing else "eager"))
        return None


def make_region(max_seg=4):
    cur = [FakeStream(16)]
    b = FakeBackend(cur)
    r = D.Region(b, max_seg=max_seg, arena=D.Arena(pin=False), is_dev=META)
    return r, b, cur


def test_region_single_segment_and_fence():
    r, b, cur = make_region()
    trace, ref = [], [r]
    add = Op("aten::add.Tensor", ref, trace)
    r.begin(cur[0])
    for _ in range(3):
        r.dispatch(add, (m(4), m(4)), {})
    r.end()
    assert trace == [("aten::add.Tensor", "cap")] * 3
    assert b.log == [("begin", "g1"), ("launch", "g1", 0, 0), ("fence", 0)]
    assert r.stats["segments"] == 1 and not r.active and D.current_region() is None


def test_region_hazard_runs_eager_then_resumes_in_order():
    r, b, cur = make_region()
    trace, ref = [], [r]
    add = Op("aten::add.Tensor", ref, trace)
    item = Op("aten::_local_scalar_dense", ref, trace)
    r.begin(cur[0])
    r.dispatch(add, (m(4), m(4)), {})
    r.dispatch(item, (m(1),), {})  # hazard: segment 0 launched first, then the sync runs eagerly
    r.dispatch(add, (m(4), m(4)), {})  # resumes capture (segment 1)
    r.end()
    assert trace == [("aten::add.Tensor", "cap"), ("aten::_local_scalar_dense", "eager"),
                     ("aten::add.Tensor", "cap")]
    assert b.log == [("begin", "g1"), ("launch", "g1", 0, 0), ("begin", "g2"), ("launch", "g2", 1, 0),
                     ("fence", 0)]
    assert r.stats["haz:sync:_local_scalar_dense"] == 1


def test_region_latch_on_stream_event_hazard():
    r, b, cur = make_region()
    trace, ref = [], [r]
    add = Op("aten::add.Tensor", ref, trace)
    r.begin(cur[0])
    r.dispatch(add, (m(4), m(4)), {})
    r.hazard("event:record", latch=True)
    r.dispatch(add, (m(4), m(4)), {})
    r.end()
    assert trace == [("aten::add.Tensor", "cap"), ("aten::add.Tensor", "eager")]
    assert [x[0] for x in b.log] == ["begin", "launch", "fence"]


def test_region_no_resume_on_other_stream_and_max_seg():
    r, b, cur = make_region(max_seg=2)
    trace, ref = [], [r]
    add = Op("aten::add.Tensor", ref, trace)
    item = Op("aten::_local_scalar_dense", ref, trace)
    r.begin(cur[0])
    r.dispatch(item, (m(1),), {})  # seg0 closed (empty)
    cur[0] = FakeStream(99)
    r.dispatch(add, (m(4), m(4)), {})  # other stream: stays eager
    cur[0] = r.stream
    r.dispatch(add, (m(4), m(4)), {})  # back on the region stream: seg1
    r.dispatch(item, (m(1),), {})
    r.dispatch(add, (m(4), m(4)), {})  # seg cap (2) reached: eager
    r.end()
    assert [t[1] for t in trace] == ["eager", "eager", "cap", "eager", "eager"]
    assert r.stats["segments"] == 2


def test_region_slots_cycle_and_wait():
    r, b, cur = make_region()
    for i in range(6):
        r.begin(cur[0])
        r.end()
    assert b.waited == [0, 1, 2, 3, 0, 1]
    assert [x[1] for x in b.log if x[0] == "fence"] == [0, 1, 2, 3, 0, 1]


def test_region_refuses_reentry_and_abort():
    r, b, cur = make_region()
    r.begin(cur[0])
    with pytest.raises(RuntimeError):
        r.begin(cur[0])
    r.abort()
    assert not r.active and not r.capturing and D.current_region() is None


# ---------------------------------------------------------------------------- real torch ops via the mode
def test_dispatch_mode_stages_h2d_and_serves_pinned():
    r, b, cur = make_region()
    r.begin(cur[0])
    mode = D._Mode.make(r)
    src = torch.arange(6, dtype=torch.int32)
    dst = torch.empty(6, dtype=torch.int32, device="meta")
    with mode:
        p = torch.empty(4, dtype=torch.int64, pin_memory=True)  # would need CUDA if not intercepted
        dst.copy_(src, non_blocking=True)
        y = dst + 1
        z = src * 2  # host-only
    r.end()
    assert r.arena.owns(p) and p.shape == (4,)
    assert r.stats["pinned"] == 1 and r.stats["staged"] == 1
    off = (p.numel() * 8 + D.ALIGN - 1) // D.ALIGN * D.ALIGN
    staged = r.arena.buf[off:off + 24].view(torch.int32)
    assert torch.equal(staged, src)
    assert y.device.type == "meta" and torch.equal(z, src * 2)
    assert [x[0] for x in b.log] == ["begin", "launch", "fence"]


def test_dispatch_mode_async_tensor_h2d_pattern():
    """vllm.utils.torch_utils.async_tensor_h2d: pinned temp + CPU copy + non_blocking copy into `out`."""
    r, b, cur = make_region()
    r.begin(cur[0])
    data = torch.tensor([3, 1, 4, 1, 5], dtype=torch.int64)
    out = torch.empty(5, dtype=torch.int64, device="meta")
    with D._Mode.make(r):
        t = torch.empty(data.shape, dtype=data.dtype, device="cpu", pin_memory=True).copy_(data)
        out.copy_(t, non_blocking=True)
    r.end()
    assert r.arena.owns(t) and torch.equal(t, data)
    assert r.stats["pinned"] == 1 and r.stats.get("staged", 0) == 0  # source already in the arena
    assert sum(1 for k in r.stats if k.startswith("haz:")) == 0


# ---------------------------------------------------------------------------- eligibility
BASEK = dict(cg_full=True, num_ubatches=1, has_prefill=False, pcp=False, dcp_size=1, lora=False,
             encoder_inputs=False, encoder_decoder=False, fused_av_ok=True, pp_single=True)


def test_eligible_decode():
    assert D.dt_ineligible(**BASEK) is None


@pytest.mark.parametrize("k,v,why", [
    ("cg_full", False, "not_full_graph"), ("num_ubatches", 2, "ubatch"), ("has_prefill", True, "prefill"),
    ("pcp", True, "pcp"), ("dcp_size", 2, "dcp"), ("lora", True, "lora"), ("encoder_inputs", True, "encoder"),
    ("encoder_decoder", True, "encoder"), ("fused_av_ok", False, "av_not_fused"), ("pp_single", False, "pp"),
])
def test_ineligible(k, v, why):
    assert D.dt_ineligible(**dict(BASEK, **{k: v})) == why


# ---------------------------------------------------------------------------- runner glue with fakes
@pytest.fixture
def glue(monkeypatch):
    calls = []
    fake_logger = types.SimpleNamespace(info=lambda *a, **k: calls.append(("info", a[0])),
                                        warning=lambda *a, **k: calls.append(("warning", a[0])),
                                        error=lambda *a, **k: calls.append(("error", a[0])))
    monkeypatch.setattr(D, "_logger", lambda: fake_logger)
    monkeypatch.setattr(D, "CFG", D.parse_flag("1,warm=1,log=2"))
    monkeypatch.setattr(D, "_install_hooks", lambda: calls.append(("hooks",)))
    monkeypatch.setattr(D, "_hook_rdma", lambda: None)
    monkeypatch.setattr(D, "PIN_ARENA", False)
    cur = [FakeStream(16)]
    backends = []

    def make_backend(update):
        fb = FakeBackend(cur)
        fb.is_default = lambda s: False
        backends.append(fb)
        return fb
    monkeypatch.setattr(D, "TorchBackend", make_backend)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a: cur[0])

    swa_log = []

    class FakeSwa:
        def __enter__(self):
            swa_log.append(("swa_enter", D.current_region() is not None))
            return self

        def __exit__(self, *e):
            r = D.current_region()
            swa_log.append(("swa_exit", r is not None and r.capturing))
            return False
    mod = types.ModuleType("vllm.v1.attention.backends.mla.sparse_swa")
    mod.dsv41_swa_batch = FakeSwa
    mod._SWAM = {"pending": None}
    monkeypatch.setitem(sys.modules, "vllm.v1.attention.backends.mla.sparse_swa", mod)

    class MS:
        def _dsv41_prepare_attn(self):
            pass

    runner = types.SimpleNamespace(
        adaptive_verification=types.SimpleNamespace(dsv41_fused_av_on=True), pcp_manager=None,
        dcp_size=1, lora_config=None, is_encoder_decoder=False, is_first_pp_rank=True,
        is_last_pp_rank=True, model_state=MS())
    so = types.SimpleNamespace(scheduled_encoder_inputs={})
    brs = types.SimpleNamespace(has_prefill=False, req_ids=["a"], num_tokens=4)
    bd = types.SimpleNamespace(cg_mode=types.SimpleNamespace(name="FULL"), num_ubatches=1, num_tokens=4)
    return types.SimpleNamespace(runner=runner, so=so, brs=brs, bd=bd, calls=calls, backends=backends,
                                 swa_log=swa_log, mod=mod)


def test_dt_begin_end_flow(glue):
    g = glue
    D.dt_begin(g.runner, g.so, g.brs, g.bd)  # warm (1 eager run)
    assert g.runner._dsv41_dtg.region is None and g.runner._dsv41_dtg.skips["warm"] == 1
    D.dt_end(g.runner)  # no-op
    D.dt_begin(g.runner, g.so, g.brs, g.bd)
    st = g.runner._dsv41_dtg
    assert st.region.active and st.region.capturing and D.current_region() is st.region
    # swa batch armed BEFORE the capture began
    assert g.swa_log == [("swa_enter", False)]
    x = torch.empty(4, device="meta") + 1  # captured through the mode
    D.dt_end(g.runner)
    # the fused SWA flush ran while still capturing, i.e. inside the region's last graph
    assert g.swa_log[-1] == ("swa_exit", True)
    fb = g.backends[0]
    assert [e[0] for e in fb.log] == ["begin", "launch", "fence"]
    assert D.current_region() is None and st.mode is None
    assert x.device.type == "meta"
    # second region also logs stats (log=2)
    D.dt_begin(g.runner, g.so, g.brs, g.bd)
    D.dt_end(g.runner)
    assert any(c[0] == "info" and "stats" in c[1] for c in g.calls)


def test_dt_begin_skips_ineligible(glue):
    g = glue
    g.brs.has_prefill = True
    for _ in range(3):
        D.dt_begin(g.runner, g.so, g.brs, g.bd)
        D.dt_end(g.runner)
    st = g.runner._dsv41_dtg
    assert st.region is None and st.skips["prefill"] == 3


def test_wrap_execute_model_aborts_and_reraises(glue):
    g = glue

    class R:
        def execute_model(self, so, brs, bd):
            D.dt_begin(self, so, brs, bd)
            torch.empty(2, device="meta") * 2
            raise ValueError("boom")

    D.wrap_execute_model(R)
    D.wrap_execute_model(R)  # idempotent
    r = R()
    for k, v in vars(g.runner).items():
        setattr(r, k, v)
    r._dsv41_dtg = D._State(r)
    r._dsv41_dtg.seen[(1, 4, 4)] = 5  # already warm
    with pytest.raises(ValueError):
        r.execute_model(g.so, g.brs, g.bd)
    st = r._dsv41_dtg
    assert st.disabled and not st.region.active and D.current_region() is None and st.mode is None
    assert g.mod._SWAM["pending"] is None
    from torch.utils._python_dispatch import _get_current_dispatch_mode
    assert _get_current_dispatch_mode() is None
    # nothing was launched for the aborted capture
    assert [e[0] for e in g.backends[0].log] == ["begin"]


def test_hz_wrapper_is_inert_outside_region():
    seen = []

    def orig(*a, **k):
        seen.append(a)
        return 7
    w = D._hz(orig, "event:record", True)
    assert D.current_region() is None
    assert w(1, 2) == 7 and seen == [(1, 2)]
    r, b, cur = make_region()
    r.begin(cur[0])
    assert w(3) == 7
    assert r.latch and not r.capturing and r.stats["haz:event:record"] == 1
    r.end()


def test_install_hooks_patch_and_restore():
    E, S, G = torch.cuda.Event, torch.cuda.Stream, torch.cuda.CUDAGraph
    saved = {n: getattr(E, n) for n in ("record", "wait", "synchronize", "query")}
    saved_s = {n: getattr(S, n) for n in ("wait_stream", "wait_event", "record_event", "synchronize", "query")}
    saved_sync, saved_set, saved_replay = torch.cuda.synchronize, torch.cuda.set_stream, G.replay
    try:
        D._ORIG.clear()
        D._install_hooks()
        assert getattr(E.record, "_dsv41_dtg", False) and getattr(S.wait_stream, "_dsv41_dtg", False)
        assert getattr(torch.cuda.set_stream, "_dsv41_dtg", False)
        assert getattr(G.replay, "_dsv41_dtg", False) and D._ORIG["CUDAGraph.replay"] is saved_replay
        D._install_hooks()  # idempotent: no double wrapping
        assert E.record.__wrapped__ is saved["record"]
    finally:
        for n, f in saved.items():
            setattr(E, n, f)
        for n, f in saved_s.items():
            setattr(S, n, f)
        torch.cuda.synchronize, torch.cuda.set_stream, G.replay = saved_sync, saved_set, saved_replay
        D._ORIG.clear()
