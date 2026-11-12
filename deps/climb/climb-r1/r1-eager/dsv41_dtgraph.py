# SPDX-License-Identifier: Apache-2.0
# DSv4.1 climb r1 (r1-eager, lever G08 "dtgraph"), 2026-11-08. Default OFF.
#
# DSV41_DTGRAPH captures the drafter->target eager glue of each decode step (prepare_inputs,
# prepare_attn, model_state.prepare_attn: fused AV, expand_idx, positions/seq_lens, combine, live cap,
# block tables, slot mappings, token->request map, indexer / paged-MQA / ring / SWA metadata) into a
# CUDA graph and launches it once, instead of issuing ~24 kernels one by one on the main stream.
#
# The graph is captured EVERY step from the very Python code that would otherwise launch eagerly, and
# launched immediately on the same stream ("capture-then-launch-once"). So the graph always contains
# exactly this step's kernels with this step's arguments, in program order; nothing is replayed from an
# older step, and no host-side value can go stale. What changes is only the launch mechanism (graph
# nodes instead of individual launches) and, for host->device copies, where the bytes are staged.
#
# Safety rails (all host-side):
#   * Hazard exit: anything a single-launch graph cannot represent exactly - a host sync (item/tolist,
#     nonzero, D2H copy, blocking H2D), a stream switch, an event record/wait/sync, a graph replay, a
#     collective (GroupCoordinator, rdma_ar, c10d), a torch RNG op, a CPU tensor argument of a device
#     op, a boolean-mask index, a Triton compile or module load - is caught BEFORE it runs. The capture
#     so far is ended and launched, the hazard runs eagerly, and capture resumes at the next aten op
#     (never after a stream/event/graph hazard). Program order on the stream is preserved either way.
#   * H2D copies whose source is CPU memory, and pinned host allocations made inside a segment (the
#     async_tensor_h2d staging temporaries), are served from a persistent pinned arena (4 slots x 64 KiB;
#     a slot is reused only after an event recorded behind the region's last launch has completed), so a
#     captured memcpy node never reads a temporary pinned block (torch 2.13 never recycles pinned blocks
#     used inside a capture: CachingHostAllocator.h:383-405) and no cudaHostAlloc runs during a capture.
#   * Only FULL-cudagraph decode steps without prefill / ubatching / PCP / DCP / LoRA / encoder inputs and
#     with the fused AV path (or no AV) are captured; a step shape must have run eagerly `warm` times.
#   * The SWA metadata batch (DSV41_EAGER_GLUE=...swam) is disabled during any capture
#     (sparse_swa.py:1484), so the region opens its own dsv41_swa_batch() BEFORE capture starts; the
#     builders queue into it and the single fused launch is issued at the region end, at exactly the
#     place the inner context would have issued it.
#
# Values (comma separated): "1"|"dt" = capture the DT region, instantiating one exec per step;
#   "update" = keep one exec per (segment, slot) and cudaGraphExecUpdate it (cuda.bindings; falls back
#   to instantiate on any failure); "warm=N" (default 2); "seg=N" max segments per region (default 4);
#   "log=N" stats line every N regions (default 5000, 0 = off). "td" is recognised and refused: the
#   target->drafter region runs code whose path changes under capture (dsv41_r4_small.py:835, :957,
#   :976, dsv41_boundary.py:269/:498, dsv41_oblit.py:83), so capturing it would not be exact.
import os
import threading
import warnings
from collections import Counter, deque

import torch

TAG = "dsv41-dtgraph"


def parse_flag(raw: str | None) -> dict:
    cfg = {"regions": set(), "update": False, "warm": 2, "seg": 4, "log": 5000, "refused": []}
    for tok in (raw or "").replace("+", ",").split(","):
        tok = tok.strip().lower()
        if tok in ("", "0", "off", "false", "no"):
            continue
        if tok in ("1", "on", "true", "dt"):
            cfg["regions"].add("dt")
        elif tok == "update":
            cfg["update"] = True
        elif tok == "td":
            cfg["refused"].append("td")
        elif "=" in tok and tok.split("=", 1)[0] in ("warm", "seg", "log"):
            k, v = tok.split("=", 1)
            cfg[k] = max(0 if k != "seg" else 1, int(v))
        else:
            raise ValueError(f"DSV41_DTGRAPH: unknown token {tok!r} (use dt, update, warm=N, seg=N, log=N)")
    if cfg["update"] and not cfg["regions"]:
        cfg["regions"].add("dt")
    return cfg


CFG = parse_flag(os.environ.get("DSV41_DTGRAPH", ""))
ON = bool(CFG["regions"])

PIN_ARENA = True  # tests on CPU-only torch set this False
SLOT_BYTES = 64 * 1024
NSLOT = 4
ALIGN = 256

# ----------------------------------------------------------------------------------------------
# Op classification (pure; unit-tested on CPU with "meta" standing in for the device)
# ----------------------------------------------------------------------------------------------
PASS, PASS_CPU, STAGE, HAZ, PIN = "pass", "pass_cpu", "stage", "haz", "pin"

_SYNC_OPS = frozenset({
    "_local_scalar_dense", "item", "nonzero", "nonzero_numpy", "argwhere", "masked_select",
    "_unique", "_unique2", "unique_dim", "unique_consecutive", "unique_dim_consecutive",
    "repeat_interleave", "bincount", "histc", "equal", "is_nonzero", "allclose", "isin_sync",
    "_assert_scalar", "_print", "segment_reduce", "_nested_tensor_from_mask",
})
_RNG_OPS = frozenset({
    "uniform", "uniform_", "normal", "normal_", "exponential", "exponential_", "bernoulli",
    "bernoulli_", "random", "random_", "multinomial", "rand", "rand_like", "randn", "randn_like",
    "randint", "randint_like", "randperm", "geometric", "geometric_", "cauchy", "cauchy_",
    "log_normal", "log_normal_", "rrelu", "rrelu_with_noise", "dropout", "native_dropout",
    "poisson", "_standard_gamma", "_sample_dirichlet", "binomial", "feature_dropout",
    "alpha_dropout", "_fused_dropout",
})
_COLL_WORDS = ("all_reduce", "allreduce", "all_gather", "allgather", "reduce_scatter", "broadcast",
               "all_to_all", "alltoall", "send", "recv", "barrier", "gather")
_COLL_NS = frozenset({"c10d", "_c10d_functional", "_c10d_functional_autograd", "_dtensor"})
_INDEX_OPS = frozenset({"index", "index_put", "index_put_", "_index_put_impl_", "_unsafe_index",
                        "_unsafe_index_put"})


def _default_is_dev(x) -> bool:
    return getattr(x, "is_cuda", False)


def _tensors(obj, out):
    if isinstance(obj, torch.Tensor):
        out.append(obj)
    elif isinstance(obj, (list, tuple)):
        for o in obj:
            _tensors(o, out)
    elif isinstance(obj, dict):
        for o in obj.values():
            _tensors(o, out)
    return out


def _dev_kw(kwargs, is_dev) -> bool:
    d = kwargs.get("device")
    if d is None:
        return False
    if isinstance(d, str):
        d = torch.device(d)
    if isinstance(d, torch.device):
        return is_dev(torch.empty(0, device="meta")) if d.type == "meta" else d.type == "cuda"
    return False


def _is_cpu(t) -> bool:
    return isinstance(t, torch.Tensor) and t.device.type == "cpu"


def op_name(func) -> tuple[str, str]:
    """('aten', 'copy_') from an OpOverload; ('', str(func)) when there is no schema."""
    try:
        full = func._schema.name
    except AttributeError:
        return "", str(func)
    ns, _, name = full.partition("::")
    return ns, name


_VIEW_OPS = frozenset({
    "view", "_unsafe_view", "slice", "select", "as_strided", "alias", "t", "transpose", "permute",
    "unsqueeze", "squeeze", "expand", "narrow", "detach", "reshape", "_reshape_alias", "flatten",
    "unflatten", "view_as", "split", "split_with_sizes", "chunk", "unbind", "diagonal", "numpy_T",
    "lift_fresh", "dim", "size", "stride", "sym_size", "sym_stride", "sym_numel", "is_pinned",
})


def classify(func, args, kwargs, is_dev=_default_is_dev, owned=lambda t: False) -> tuple[str, str]:
    """Decide what the capture does with one aten call: PASS (capture it), PASS_CPU (host-only work),
    STAGE (CPU->device copy: stage the source in the pinned arena, then capture), PIN (a pinned host
    allocation: serve it from the arena), HAZ (end the capture and run it eagerly; second value =
    reason). `owned(t)` is True for tensors that already live in the arena."""
    kwargs = kwargs or {}
    ns, name = op_name(func)
    if ns == "aten" and name in _VIEW_OPS:
        return PASS, ""  # metadata only: no launch, no sync
    if name in ("empty", "empty_strided") and kwargs.get("pin_memory"):
        return PIN, "pin_alloc"
    if name == "_pin_memory":
        return PIN, "pin_copy"
    ts = _tensors(args, [])
    _tensors(kwargs, ts)
    dev = [t for t in ts if is_dev(t)]
    devkw = _dev_kw(kwargs, is_dev)
    if ns in _COLL_NS or (ns == "vllm" and any(w in name for w in _COLL_WORDS)):
        return HAZ, f"collective:{ns}::{name}"
    if name in _RNG_OPS:
        return (HAZ, f"rng:{name}") if (dev or devkw) else (PASS_CPU, "")
    if name == "record_stream":
        return HAZ, "stream:record_stream"
    if name == "copy_" and len(args) >= 2:
        dst, src = args[0], args[1]
        nb = bool(args[2]) if len(args) > 2 else bool(kwargs.get("non_blocking", False))
        if is_dev(dst) and _is_cpu(src):
            if not nb:
                return HAZ, "h2d_blocking"
            if src.dtype != dst.dtype:
                return HAZ, "h2d_cast"
            if owned(src):
                return PASS, ""
            if not src.is_contiguous():
                return HAZ, "h2d_noncontig"
            return STAGE, "h2d"
        if _is_cpu(dst) and is_dev(src):
            return HAZ, "d2h"
        return (PASS, "") if dev else (PASS_CPU, "")
    if name == "_to_copy" and args:
        src = args[0]
        to_dev = devkw
        to_cpu = kwargs.get("device") is not None and torch.device(kwargs["device"]).type == "cpu"
        nb = bool(kwargs.get("non_blocking", False))
        if _is_cpu(src) and to_dev:
            if not nb:
                return HAZ, "h2d_blocking"
            dt = kwargs.get("dtype")
            if dt is not None and dt != src.dtype:
                return HAZ, "h2d_cast"
            if owned(src):
                return PASS, ""
            if not src.is_contiguous():
                return HAZ, "h2d_noncontig"
            return STAGE, "h2d"
        if is_dev(src) and to_cpu:
            return HAZ, "d2h"
        return (PASS, "") if (dev or to_dev) else (PASS_CPU, "")
    if not dev and not devkw:
        return PASS_CPU, ""
    if name in _SYNC_OPS or (name == "where" and len(args) == 1):
        return HAZ, f"sync:{name}"
    if name in _INDEX_OPS:
        idx = args[1] if len(args) > 1 else kwargs.get("indices")
        for t in _tensors(idx, []):
            if t.dtype in (torch.bool, torch.uint8):
                return HAZ, "mask_index"
    for t in ts:
        if _is_cpu(t) and t.dim() > 0:
            return HAZ, "cpu_arg"
    return PASS, ""


# ----------------------------------------------------------------------------------------------
# Pinned staging arena
# ----------------------------------------------------------------------------------------------
class Arena:
    def __init__(self, slot_bytes: int = SLOT_BYTES, nslot: int = NSLOT, pin: bool | None = None):
        self.slot_bytes, self.nslot = slot_bytes, nslot
        pin = PIN_ARENA if pin is None else pin
        self.buf = torch.zeros(slot_bytes * nslot, dtype=torch.uint8, pin_memory=pin)
        self.slot, self.off = 0, 0

    def start(self, slot: int) -> None:
        self.slot, self.off = slot, 0

    def owns(self, t) -> bool:
        if not isinstance(t, torch.Tensor) or t.device.type != "cpu":
            return False
        lo = self.buf.data_ptr()
        return lo <= t.data_ptr() < lo + self.buf.numel()

    def alloc(self, shape, dtype) -> torch.Tensor | None:
        """A contiguous [shape] dtype tensor in this slot (contents undefined), or None when full."""
        n = 1
        for d in shape:
            n *= int(d)
        nb = n * torch.empty(0, dtype=dtype).element_size()
        a = (self.off + ALIGN - 1) // ALIGN * ALIGN
        if a + nb > self.slot_bytes:
            return None
        base = self.slot * self.slot_bytes + a
        self.off = a + nb
        return self.buf[base:base + nb].view(dtype).view(tuple(int(d) for d in shape))

    def take(self, src: torch.Tensor) -> torch.Tensor | None:
        """Copy src (CPU, any layout) into this slot; return a contiguous same-dtype/shape view of the
        pinned bytes, or None when the slot is full."""
        n = src.numel() * src.element_size()
        a = (self.off + ALIGN - 1) // ALIGN * ALIGN
        if a + n > self.slot_bytes:
            return None
        base = self.slot * self.slot_bytes + a
        view = self.buf[base:base + n].view(src.dtype).view(src.shape)
        view.copy_(src)
        self.off = a + n
        return view


def _contig_strides(shape) -> list[int]:
    st, acc = [], 1
    for d in reversed([int(x) for x in shape]):
        st.append(acc)
        acc *= max(d, 1)
    return st[::-1]


def pin_result(func, args, kwargs, arena: Arena):
    """Arena-backed result for a pinned allocation / pin copy; None = cannot serve (caller hazards)."""
    name = op_name(func)[1]
    if name == "_pin_memory":
        src = args[0]
        if not src.is_contiguous():
            return None
        return arena.take(src)
    dtype = kwargs.get("dtype") or torch.get_default_dtype()
    if kwargs.get("layout") not in (None, torch.strided):
        return None
    shape = args[0]
    if name == "empty_strided" and list(args[1]) != _contig_strides(shape):
        return None
    return arena.alloc(shape, dtype)


def stage_args(func, args, kwargs, arena: Arena):
    """Rewrite a STAGE call so its CPU source is the arena view. None when the arena is full."""
    name = op_name(func)[1]
    if name == "copy_":
        v = arena.take(args[1])
        return None if v is None else ((args[0], v, *args[2:]), kwargs)
    v = arena.take(args[0])
    return None if v is None else ((v, *args[1:]), kwargs)


# ----------------------------------------------------------------------------------------------
# Eligibility (pure)
# ----------------------------------------------------------------------------------------------
def dt_ineligible(*, cg_full: bool, num_ubatches: int, has_prefill: bool, pcp: bool, dcp_size: int,
                  lora: bool, encoder_inputs: bool, encoder_decoder: bool, fused_av_ok: bool,
                  pp_single: bool) -> str | None:
    if not cg_full:
        return "not_full_graph"
    if num_ubatches > 1:
        return "ubatch"
    if has_prefill:
        return "prefill"
    if pcp:
        return "pcp"
    if dcp_size > 1:
        return "dcp"
    if lora:
        return "lora"
    if encoder_inputs or encoder_decoder:
        return "encoder"
    if not fused_av_ok:
        return "av_not_fused"
    if not pp_single:
        return "pp"
    return None


# ----------------------------------------------------------------------------------------------
# Graph backends
# ----------------------------------------------------------------------------------------------
class TorchBackend:
    """torch.cuda.CUDAGraph per segment, private pool shared by all segments, thread-local capture."""

    def __init__(self, update: bool):
        self.update = update
        self.pool = torch.cuda.graph_pool_handle()
        self.keep: deque = deque(maxlen=8)  # recent graphs: keep the pool referenced, execs alive
        self.execs: dict = {}  # update mode: (seg, slot) -> owner CUDAGraph (holds the exec)
        self.events = [torch.cuda.Event() for _ in range(NSLOT)]
        self.recorded = [False] * NSLOT
        self.cudart = None
        self.update_fail = 0
        if update:
            try:
                from cuda.bindings import runtime as cudart  # noqa: PLC0415
                self.cudart = cudart
            except Exception as e:  # noqa: BLE001
                warnings.warn(f"DSV41_DTGRAPH update: cuda.bindings unavailable ({e}); instantiate mode")
                self.update = False

    def stream(self):
        return torch.cuda.current_stream()

    def is_default(self, s) -> bool:
        return s.cuda_stream == torch.cuda.default_stream(s.device).cuda_stream or s.cuda_stream in (0, 1, 2)

    def wait_slot(self, slot: int) -> bool:
        ev = self.events[slot]
        if self.recorded[slot] and not _ORIG["Event.query"](ev):
            _ORIG["Event.synchronize"](ev)
            return True
        return False

    def begin(self):
        g = torch.cuda.CUDAGraph(keep_graph=self.update)
        g.capture_begin(pool=self.pool, capture_error_mode="thread_local")
        return g

    def fence(self, slot: int, stream) -> None:
        # Eager record after the region's last graph launch / eager op: completes when they do.
        _ORIG["Event.record"](self.events[slot], stream)
        self.recorded[slot] = True

    def end_and_launch(self, g, seg: int, slot: int, stream) -> None:
        g.capture_end()
        self.keep.append(g)
        replay = _ORIG.get("CUDAGraph.replay", torch.cuda.CUDAGraph.replay)  # unhooked
        if not self.update:
            replay(g)  # instantiated by capture_end (keep_graph=False)
            return
        if self._update_launch(g, seg, slot, stream):
            return
        g.instantiate()
        replay(g)
        self.execs[(seg, slot)] = g

    def _update_launch(self, g, seg: int, slot: int, stream) -> bool:
        owner = self.execs.get((seg, slot))
        if owner is None or self.cudart is None:
            return False
        rt = self.cudart
        try:
            ex = rt.cudaGraphExec_t(init_value=owner.raw_cuda_graph_exec())
            gr = rt.cudaGraph_t(init_value=g.raw_cuda_graph())
            res = rt.cudaGraphExecUpdate(ex, gr)
            err = res[0] if isinstance(res, tuple) else res
            if err != rt.cudaError_t.cudaSuccess:
                self.update_fail += 1
                return False
            res = rt.cudaGraphLaunch(ex, rt.cudaStream_t(init_value=stream.cuda_stream))
            err = res[0] if isinstance(res, tuple) else res
            if err != rt.cudaError_t.cudaSuccess:
                # Not launched: the caller instantiates g and replays it instead.
                self.update_fail += 1
                return False
            return True
        except Exception as e:  # noqa: BLE001 - binding mismatch: never launched, fall back for good
            warnings.warn(f"DSV41_DTGRAPH update path disabled ({type(e).__name__}: {e})")
            self.cudart = None
            return False


# ----------------------------------------------------------------------------------------------
# Region state machine
# ----------------------------------------------------------------------------------------------
_TLS = threading.local()
_ORIG: dict = {}


class Region:
    """One capture region per step: begin() ... end(). Segments end at hazards; capture resumes at the
    next aten op unless a stream/event/graph hazard latched it off."""

    def __init__(self, backend, *, max_seg: int = 4, arena: Arena | None = None, is_dev=_default_is_dev):
        self.b = backend
        self.max_seg = max_seg
        self.arena = arena if arena is not None else Arena()
        self.is_dev = is_dev
        self.active = False
        self.capturing = False
        self.g = None
        self.seg = 0
        self.latch = False
        self.want = False
        self.slot = 0
        self.stream = None
        self.n = 0
        self.stats: Counter = Counter()

    # -- lifecycle ------------------------------------------------------------------------------
    def begin(self, stream) -> bool:
        if self.active:
            raise RuntimeError("DSV41_DTGRAPH: a capture region is still open (an exception escaped the "
                               "previous step's glue); refusing to continue - restart with DSV41_DTGRAPH=0")
        self.slot = self.n % NSLOT
        if self.b.wait_slot(self.slot):
            self.stats["slot_wait"] += 1
        self.arena.start(self.slot)
        self.stream = stream
        self.seg, self.latch, self.want = 0, False, False
        self.g = self.b.begin()
        self.capturing = True
        self.active = True
        _TLS.region = self
        self.stats["regions"] += 1
        return True

    def end(self) -> None:
        if not self.active:
            return
        try:
            if self.capturing:
                self._close()
            self.b.fence(self.slot, self.stream)
        finally:
            self.active = False
            self.capturing = False
            _TLS.region = None
            self.n += 1

    def abort(self) -> None:
        """Exception path: end a capture without launching (its work is lost; caller must raise)."""
        if self.capturing:
            try:
                self.g.capture_end()
            except Exception:  # noqa: BLE001
                pass
        self.active = False
        self.capturing = False
        _TLS.region = None

    # -- segments -------------------------------------------------------------------------------
    def _close(self) -> None:
        g, self.g = self.g, None
        self.capturing = False
        self.b.end_and_launch(g, self.seg, self.slot, self.stream)
        self.stats["segments"] += 1
        self.seg += 1

    def hazard(self, why: str, latch: bool = False) -> None:
        self.stats["haz:" + why] += 1
        if latch:
            self.latch = True
        if self.capturing:
            self._close()
        self.want = not self.latch

    def maybe_resume(self) -> None:
        if (self.want and not self.latch and not self.capturing and self.seg < self.max_seg
                and self.b.stream().cuda_stream == self.stream.cuda_stream):
            self.want = False
            self.g = self.b.begin()
            self.capturing = True

    # -- aten interception ----------------------------------------------------------------------
    def dispatch(self, func, args, kwargs):
        if not self.capturing:
            if self.want:
                self.maybe_resume()
            if not self.capturing:
                return func(*args, **kwargs)
        act, why = classify(func, args, kwargs, self.is_dev, self.arena.owns)
        if act in (PASS, PASS_CPU):
            return func(*args, **kwargs)
        if act == PIN:
            out = pin_result(func, args, kwargs, self.arena)
            if out is not None:
                self.stats["pinned"] += 1
                return out
            why = "arena_full"
        if act == STAGE:
            st = stage_args(func, args, kwargs, self.arena)
            if st is not None:
                self.stats["staged"] += 1
                return func(*st[0], **st[1])
            why = "arena_full"
        self.hazard(why)
        return func(*args, **kwargs)


def current_region():
    return getattr(_TLS, "region", None)


# ----------------------------------------------------------------------------------------------
# Python-level hazard hooks (installed once, only when the flag is on; inert in other threads and
# outside a region: one thread-local lookup)
# ----------------------------------------------------------------------------------------------
def _hz(orig, why, latch):
    def w(*a, **k):
        r = getattr(_TLS, "region", None)
        if r is not None:
            r.hazard(why, latch=latch)
        return orig(*a, **k)
    w.__wrapped__ = orig
    w._dsv41_dtg = True
    return w


def _install_hooks() -> None:
    if _ORIG.get("_installed"):
        return
    E, S = torch.cuda.Event, torch.cuda.Stream
    for cls, cname, names in ((E, "Event", ("record", "wait", "synchronize", "query")),
                              (S, "Stream", ("wait_stream", "wait_event", "record_event",
                                             "synchronize", "query"))):
        for n in names:
            orig = getattr(cls, n)
            _ORIG[f"{cname}.{n}"] = orig
            setattr(cls, n, _hz(orig, f"{cname.lower()}:{n}", True))
    _ORIG["synchronize"] = torch.cuda.synchronize
    torch.cuda.synchronize = _hz(torch.cuda.synchronize, "sync:device", True)
    orig_set = torch.cuda.set_stream
    _ORIG["set_stream"] = orig_set

    def set_stream(stream):
        r = getattr(_TLS, "region", None)
        if r is not None and stream is not None and stream.cuda_stream != r.stream.cuda_stream:
            r.hazard("stream:set", latch=True)
        return orig_set(stream)
    set_stream._dsv41_dtg = True
    torch.cuda.set_stream = set_stream
    _ORIG["CUDAGraph.replay"] = torch.cuda.CUDAGraph.replay
    torch.cuda.CUDAGraph.replay = _hz(torch.cuda.CUDAGraph.replay, "graph_replay", True)
    # Collectives: every TP/PP/DP collective goes through GroupCoordinator.
    try:
        from vllm.distributed.parallel_state import GroupCoordinator  # noqa: PLC0415
        for n in ("all_reduce", "all_gather", "all_gatherv", "reduce_scatter", "reduce_scatterv",
                  "reduce_scatterv_into_output", "gather", "broadcast", "broadcast_tensor_dict",
                  "send", "recv", "send_tensor_dict", "recv_tensor_dict", "barrier", "dispatch",
                  "combine", "combine_into_output", "dispatch_router_logits"):
            if hasattr(GroupCoordinator, n):
                setattr(GroupCoordinator, n, _hz(getattr(GroupCoordinator, n), "collective:" + n, False))
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"DSV41_DTGRAPH: GroupCoordinator hooks not installed ({e})")
    # Triton: a compile or module load inside a capture ends the segment first.
    try:
        from triton import knobs  # noqa: PLC0415
        prev = knobs.runtime.jit_cache_hook

        def cache_hook(*a, **k):
            r = getattr(_TLS, "region", None)
            if r is not None:
                r.hazard("triton_compile", latch=False)
            return prev(*a, **k) if prev is not None else None
        knobs.runtime.jit_cache_hook = cache_hook

        def load_hook(*a, **k):
            r = getattr(_TLS, "region", None)
            if r is not None:
                r.hazard("triton_load", latch=False)
        hc = knobs.runtime.kernel_load_start_hook
        if hasattr(hc, "add"):
            hc.add(load_hook)
        elif hc is None:
            knobs.runtime.kernel_load_start_hook = load_hook
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"DSV41_DTGRAPH: Triton hooks not installed ({e})")
    warnings.filterwarnings("ignore", message=".*The CUDA Graph is empty.*")
    _ORIG["_installed"] = True


def _hook_rdma() -> None:
    """rdma_ar is imported only when DSV41_RDMA_AR is on; hook it when it is already loaded."""
    if _ORIG.get("_rdma"):
        return
    import sys  # noqa: PLC0415
    m = sys.modules.get("vllm.distributed.device_communicators.rdma_all_reduce")
    if m is None:
        return
    cls = getattr(m, "RdmaAllReduce", None) or next(
        (v for v in vars(m).values() if isinstance(v, type) and hasattr(v, "all_gather_stacked")), None)
    if cls is not None:
        for n in ("all_reduce", "all_gather_stacked", "all_gather_cols"):
            if hasattr(cls, n):
                setattr(cls, n, _hz(getattr(cls, n), "collective:rdma_" + n, False))
    _ORIG["_rdma"] = True


# ----------------------------------------------------------------------------------------------
# Runner glue (called from model_runner.py behind `if _DSV41_DTGRAPH:`)
# ----------------------------------------------------------------------------------------------
class _Mode:
    """Lazily-built TorchDispatchMode forwarding to the active Region."""

    cls = None

    @classmethod
    def make(cls, region):
        if cls.cls is None:
            from torch.utils._python_dispatch import TorchDispatchMode  # noqa: PLC0415

            class _DTGMode(TorchDispatchMode):
                def __init__(self, r):
                    super().__init__()
                    self.r = r

                def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                    return self.r.dispatch(func, args, kwargs or {})
            cls.cls = _DTGMode
        return cls.cls(region)


class _State:
    def __init__(self, runner):
        self.region = None
        self.mode = None
        self.swa = None
        self.seen: Counter = Counter()
        self.skips: Counter = Counter()
        self.disabled = False
        self.logged = False
        self.rank0 = True
        try:
            from vllm.distributed import get_tensor_model_parallel_rank  # noqa: PLC0415
            self.rank0 = get_tensor_model_parallel_rank() == 0
        except Exception:  # noqa: BLE001
            pass


def _logger():
    from vllm.logger import init_logger  # noqa: PLC0415
    return init_logger(__name__)


def dt_begin(runner, scheduler_output, batch_req_state, batch_desc) -> None:
    st = getattr(runner, "_dsv41_dtg", None)
    if st is None:
        st = runner._dsv41_dtg = _State(runner)
        if CFG["refused"]:
            _logger().warning("DSV41_DTGRAPH: %s refused (target->drafter glue takes different code "
                              "paths under capture; not exact)", CFG["refused"])
    if st.disabled or "dt" not in CFG["regions"]:
        return
    if st.region is not None and st.region.active:
        st.region.abort()
        raise RuntimeError("DSV41_DTGRAPH: previous glue region never closed (exception mid-region)")
    av = getattr(runner, "adaptive_verification", None)
    why = dt_ineligible(
        cg_full=getattr(batch_desc.cg_mode, "name", str(batch_desc.cg_mode)) == "FULL",
        num_ubatches=int(getattr(batch_desc, "num_ubatches", 1) or 1),
        has_prefill=bool(batch_req_state.has_prefill),
        pcp=runner.pcp_manager is not None,
        dcp_size=int(getattr(runner, "dcp_size", 1) or 1),
        lora=bool(runner.lora_config),
        encoder_inputs=bool(getattr(scheduler_output, "scheduled_encoder_inputs", None)),
        encoder_decoder=bool(getattr(runner, "is_encoder_decoder", False)),
        fused_av_ok=av is None or bool(getattr(av, "dsv41_fused_av_on", False)),
        pp_single=bool(getattr(runner, "is_first_pp_rank", True) and getattr(runner, "is_last_pp_rank", True)),
    )
    if why is not None:
        st.skips[why] += 1
        return
    key = (len(batch_req_state.req_ids), int(batch_req_state.num_tokens), int(batch_desc.num_tokens))
    st.seen[key] += 1
    if st.seen[key] <= CFG["warm"]:
        st.skips["warm"] += 1
        return
    try:
        if st.region is None:
            _install_hooks()
            st.region = Region(TorchBackend(CFG["update"]), max_seg=CFG["seg"])
            if st.rank0:
                _logger().info("DSV41_DTGRAPH armed: region dt, mode %s, warm %d, max segments %d",
                               "update" if CFG["update"] else "instantiate", CFG["warm"], CFG["seg"])
        _hook_rdma()
        stream = torch.cuda.current_stream()
        if st.region.b.is_default(stream):
            raise RuntimeError("current stream is the default stream")
    except Exception as e:  # noqa: BLE001 - nothing captured yet: stay eager for good
        st.disabled = True
        _logger().warning("DSV41_DTGRAPH disabled before any capture: %s", e)
        return
    swa = None
    if hasattr(runner.model_state, "_dsv41_prepare_attn"):
        from vllm.v1.attention.backends.mla.sparse_swa import dsv41_swa_batch  # noqa: PLC0415
        swa = dsv41_swa_batch()
        swa.__enter__()  # not capturing yet: the batch is armed exactly as the inner one would be
    try:
        st.region.begin(stream)
    except Exception as e:  # noqa: BLE001 - capture_begin failed: nothing captured
        if swa is not None:
            swa.__exit__(None, None, None)  # nothing queued yet
        st.region.active = False
        _TLS.region = None
        st.disabled = True
        _logger().warning("DSV41_DTGRAPH disabled (capture_begin failed: %s)", e)
        return
    st.swa = swa
    st.mode = _Mode.make(st.region)
    st.mode.__enter__()


def dt_end(runner) -> None:
    st = getattr(runner, "_dsv41_dtg", None)
    if st is None or st.region is None or not st.region.active:
        return
    r = st.region
    try:
        if st.swa is not None:
            swa, st.swa = st.swa, None
            swa.__exit__(None, None, None)  # the single fused SWA launch, at the inner block's place
    finally:
        mode, st.mode = st.mode, None
        if mode is not None:
            mode.__exit__(None, None, None)
    r.end()
    every = CFG["log"]
    if every and r.n % every == 0 and st.rank0:
        s = dict(r.stats)
        _logger().info("DSV41_DTGRAPH stats after %d regions: segments/region %.3f, staged %d, "
                       "hazards %s, skips %s, update failures %d", r.n,
                       s.get("segments", 0) / max(r.n, 1), s.get("staged", 0),
                       {k: v for k, v in s.items() if k.startswith("haz:")}, dict(st.skips),
                       getattr(r.b, "update_fail", 0))


def dt_abort(runner) -> None:
    """execute_model raised while a region was open: drop the open capture (its work never ran), pop the
    dispatch mode, reset a pending SWA batch, disable the lever. The caller re-raises: the step failed."""
    st = getattr(runner, "_dsv41_dtg", None)
    if st is None or st.region is None or not st.region.active:
        return
    mode, st.mode = st.mode, None
    if mode is not None:
        try:
            mode.__exit__(None, None, None)
        except Exception:  # noqa: BLE001
            pass
    if st.swa is not None:
        st.swa = None
        try:
            from vllm.v1.attention.backends.mla import sparse_swa  # noqa: PLC0415
            sparse_swa._SWAM["pending"] = None
        except Exception:  # noqa: BLE001
            pass
    st.region.abort()
    st.disabled = True
    _logger().error("DSV41_DTGRAPH: exception inside the glue region; capture dropped, lever disabled")


def wrap_execute_model(cls) -> None:
    """Flag-on only: route execute_model exceptions through dt_abort (never leave a capture open)."""
    orig = cls.execute_model
    if getattr(orig, "_dsv41_dtg", False):
        return

    def execute_model(self, *args, **kwargs):
        try:
            return orig(self, *args, **kwargs)
        except BaseException:
            dt_abort(self)
            raise
    execute_model.__wrapped__ = orig
    execute_model.__doc__ = orig.__doc__
    execute_model._dsv41_dtg = True
    cls.execute_model = execute_model

