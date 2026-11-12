# SPDX-License-Identifier: Apache-2.0
"""climb-r1 r1-drafter: three default-off scheduling levers for the DSpark drafter.

Install path: vllm/models/deepseek_v41/nvidia/dsv41_drafter_sched.py (copied by r1-drafter/install.py).
The patched site files import this module ONLY inside a branch guarded by their own module-level flag, so with
every flag unset this file is never imported. None of the levers changes an arithmetic op: the same kernels run on
the same inputs at the same shapes with the same tactics; only streams, events and launch points move.

DSV41_MAINPROJ_HOIST=0|gemm|chain   (G12, audit spec:C3 / dense:D4 / host:H4; skeptic hybrid = chain; 1 == chain)
    DSpark main_proj (cat of the target aux taps -> MXFP8 quant -> column-split CUTLASS GEMM -> rdma all-gather ->
    main_norm) runs today in DFlashSpeculator.propose, serially after the CERT screen, refine, argmax all-gather and
    the rejection glue. Its inputs are target-graph outputs, so it can start once the screen has finished.
      gemm  : cat + quant + GEMM on a side stream that waits on an event recorded right after the _screen_kernel
              launch (D4). propose waits on the side event, then runs the all-gather, main_norm and the copy into
              speculator.hidden_states on the main stream at today's position.
      chain : as gemm, and the all-gather + main_norm also go to the side stream, enqueued right after the
              sync-free argmax all-gather (the side stream waits on an event recorded after it), so the eager
              collective order stays argmax AG -> main_proj AG -> drafter graph on every rank (hybrid).
    Engages only on steps whose target logits take the certified head + sync-free argmax path, whose main_proj
    column-split verdict for this M is already cached as bit-identical (mainproj_split checked[M] is True) and
    whose drafter is the DSpark speculator. Every other step runs the stock inline path.

DSV41_DRAFT_PROLOGUE_FORK=0|1|shared   (G21, audit spec:C1)
    DFlashSpeculator._prepare_graph_forward captures the ctx-KV precompute (quant, folded GEMM, 3x kv_norm+insert)
    and the 3-group draft decode-metadata update serially ahead of the drafter forward. Under capture, 1 forks the
    ctx-KV precompute to side stream A and the metadata update to side stream B (shared: both, in today's order, on
    one side stream); the main chain (embedding, AR, mHC pre, q_a, AG, q_kv_norm) starts at once, and draft layer
    0's attention joins both branches before its first op (_prepare_and_attn_fn: wq_b, query KV insert, indexer,
    sparse MLA), so every KV-cache write keeps its order. A fallback join after the drafter forward keeps capture
    legal if that hook is bypassed. Applies to both DSpark graph sets (stock and greedy-sharded): both are captured
    through the same _prepare_graph_forward. The ctx branch is forked only when the CTXKV fold serves that row
    count (fold GEMM with a pinned tactic); the fold then uses a private CUTLASS workspace so it can never share
    FlashInfer's process-wide mm_mxfp8 scratch with a main-chain GEMM.

DSV41_MHC_OVERLAP_DRAFT=0|1   (G14, audit glue:C / spec:C5)
    The three DSpark DeepseekV4DecoderLayers get their own mHC side stream from _DSV41_MHC_OVL.make_stream with
    fuse_mhc_all_reduce=False, so their five shifted boundaries take the target's bitwise moet/moet overlap path
    (post_norm on the critical path, mhc_proj + stats on the side stream, join after each sublayer). Requires
    DSV41_MHC_OVERLAP to be active (the target's flag); otherwise it logs and does nothing.
"""
from __future__ import annotations

import collections
import functools
import logging
import os

import torch

logger = logging.getLogger(__name__)


def _norm(name: str, allowed: dict[str, str]) -> str:
    raw = os.environ.get(name, "0").strip().lower()
    if raw in ("", "0", "off", "false", "no"):
        return "0"
    if raw not in allowed:
        raise ValueError(f"{name} must be one of 0|{'|'.join(sorted(set(allowed)))}, got {raw!r}")
    return allowed[raw]


HOIST_MODE = _norm("DSV41_MAINPROJ_HOIST", {"1": "chain", "on": "chain", "chain": "chain", "gemm": "gemm"})
FORK_MODE = _norm("DSV41_DRAFT_PROLOGUE_FORK", {"1": "two", "on": "two", "two": "two", "shared": "shared"})
MHC_DRAFT = _norm("DSV41_MHC_OVERLAP_DRAFT", {"1": "1", "on": "1"}) != "0"
LOG_EVERY = int(os.environ.get("DSV41_DRAFTER_SCHED_LOG_EVERY", "20000") or 0)

# Injectable CUDA facade (tests replace it with a recording fake; production = torch.cuda).
_cu = torch.cuda


def _record_stream(t: torch.Tensor, s) -> None:
    """Allocator bookkeeping for a side-stream allocation consumed on another stream (eager only)."""
    t.record_stream(s)


# =====================================================================================================================
# G12  DSV41_MAINPROJ_HOIST
# =====================================================================================================================
class _HoistState:
    def __init__(self) -> None:
        self.stream = None
        self.ev_screen = None
        self.ev_ag = None
        self.ev_done = None
        self.ctx = None  # (aux_hidden_states list, speculator) for the sample_tokens call in flight
        self.pending = None  # _Pending launched by the target-logits hook, consumed by DSpark propose
        self.installed = False
        self.disabled = False
        self.counts = collections.Counter()


_H = _HoistState()


class _Pending:
    __slots__ = ("aux", "st", "inner", "x", "y", "shape", "out", "mode", "done", "handoff")

    def __init__(self, aux, st, inner, mode):
        self.aux, self.st, self.inner, self.mode = aux, st, inner, mode
        self.x = self.y = self.shape = self.out = self.done = None
        self.handoff = None  # chain: the rdma communicator whose eager stream hand-off we took over


def _hoist_objects():
    """Side stream + events, created once, eagerly (never inside a capture)."""
    if _H.stream is None:
        _H.stream = _cu.Stream()
        _H.ev_screen = _cu.Event()
        _H.ev_ag = _cu.Event()
        _H.ev_done = _cu.Event()
    return _H.stream  # (eager only: _plan refuses while capturing, so the lazy first record is eager too)


def install_mainproj_hoist() -> None:
    """Called once from DSparkSpeculator.__init__ (flag on). Wraps GPUModelRunner.sample_tokens so the target-logits
    hook can see this step's aux taps and speculator; the wrapper also joins any hoisted work nobody consumed."""
    if _H.installed or HOIST_MODE == "0":
        return
    _H.installed = True
    try:
        from vllm.v1.worker.gpu import model_runner as mr

        cls = mr.GPUModelRunner
        if getattr(cls, "_dsv41_mainproj_hoist", False):
            return
        orig = cls.sample_tokens

        @functools.wraps(orig)
        def sample_tokens(self, *args, **kwargs):
            _H.ctx = _make_ctx(self)
            try:
                return orig(self, *args, **kwargs)
            finally:
                _H.ctx = None
                if _H.pending is not None:
                    _drop_pending("not consumed by propose")

        cls.sample_tokens = sample_tokens
        cls._dsv41_mainproj_hoist = True
        logger.info("DSV41_MAINPROJ_HOIST=%s: GPUModelRunner.sample_tokens hook installed", HOIST_MODE)
    except Exception as e:  # noqa: BLE001 - fail closed: no ctx -> never hoists
        logger.warning("DSV41_MAINPROJ_HOIST: could not hook sample_tokens (%r): stock main_proj", e)


def _make_ctx(runner):
    try:
        ems = runner.execute_model_state
        if ems is None or not getattr(runner, "is_last_pp_rank", True):
            return None
        if getattr(runner, "pcp_manager", None) is not None or getattr(runner, "batch_sharder", None) is not None:
            return None
        aux = ems.aux_hidden_states
        spec = getattr(runner, "speculator", None)
        if not aux or not isinstance(aux, (list, tuple)) or spec is None:
            return None
        if type(spec).__name__ != "DSparkSpeculator":
            return None
        return (aux, spec)
    except Exception:  # noqa: BLE001
        return None


def _rdma_comm():
    """The TP group's RdmaAllReduce, or None (then no eager stream guard exists to hand off)."""
    try:
        from vllm.distributed import get_tp_group

        dc = getattr(get_tp_group(), "device_communicator", None)
        return getattr(dc, "rdma_comm", None)
    except Exception:  # noqa: BLE001
        return None


def _xs_mode() -> str:
    try:
        from vllm.distributed.device_communicators import rdma_all_reduce as ra

        return str(getattr(ra, "_XS_MODE", "off"))
    except Exception:  # noqa: BLE001
        return "off"


def _plan(ctx):
    """(column-split state, inner DSpark model, mode) when this step can hoist, else None. Every input is
    TP-replicated (scheduler state, the TP-agreed checked[M] verdict, the launch config), so all ranks agree."""
    aux, spec = ctx
    if _H.disabled or _H.pending is not None or _cu.is_current_stream_capturing():
        return None
    try:
        from vllm.models.deepseek_v41.nvidia import dsv41_r4_small as r4

        inner = spec.model.model
        if not r4.MAINPROJ_SPLIT or not getattr(type(inner), "_dsv41_r4_mainproj", False):
            return None
        st = getattr(inner.main_proj, "_dsv41_r4_split", None)
        if not st:
            return None
        rows = {int(t.shape[0]) for t in aux}
        if len(rows) != 1 or any(t.dim() != 2 for t in aux):
            return None
        M = rows.pop()
        if M == 0 or M > r4.MAINPROJ_MAX_M or st.checked.get(M) is not True:
            return None
    except Exception:  # noqa: BLE001
        return None
    mode = HOIST_MODE
    if mode == "chain" and _xs_mode() != "off":
        rc = _rdma_comm()
        if rc is not None and not hasattr(rc, "_xs_last"):
            mode = "gemm"  # unknown guard layout: keep the collective on main
    return st, inner, mode


class _Launch:
    """Returned to dsv41_r4_small.target_logits; its two callbacks run at the two launch points."""

    def __init__(self, aux, st, inner, mode):
        self.p = _Pending(aux, st, inner, mode)

    def after_screen(self) -> None:
        # No collective in here, so a failure can fail closed without desynchronising the ranks.
        p = self.p
        if _H.disabled:
            return
        try:
            S = _hoist_objects()
            main = _cu.current_stream()
            _H.ev_screen.record(main)
            S.wait_event(_H.ev_screen)
            with _cu.stream(S):
                p.x = torch.cat(p.aux, dim=-1)
                p.y, p.shape = p.st.local(p.x)
        except Exception as e:  # noqa: BLE001
            _H.disabled = True
            logger.warning("DSV41_MAINPROJ_HOIST disabled after a launch failure (%r); stock main_proj", e)
            if _H.stream is not None:
                _H.ev_done.record(_H.stream)
                _cu.current_stream().wait_event(_H.ev_done)
            return
        if p.mode == "gemm":
            _H.ev_done.record(S)
            p.done = _H.ev_done
        _H.pending = p
        _H.counts["launched"] += 1

    def after_ag(self) -> None:
        p = self.p
        if p.mode != "chain" or _H.pending is not p or p.y is None:
            return
        S = _H.stream
        main = _cu.current_stream()
        _H.ev_ag.record(main)
        S.wait_event(_H.ev_ag)
        rc = _rdma_comm() if _xs_mode() != "off" else None
        with _cu.stream(S):
            if rc is not None:
                # Explicit hand-off: S is ordered after the argmax all-gather by ev_ag, and main joins S (ev_done)
                # before its next collective, so the eager guard has nothing left to order. Without this every
                # step would log an xstream switch with busy=True.
                rc._xs_last = _cu.current_stream()
                p.handoff = rc
            g = p.st.gather(p.y, p.shape)
            p.out = p.inner.main_norm(g)
        _H.ev_done.record(S)
        p.done = _H.ev_done


def mainproj_target_hook():
    """dsv41_r4_small.target_logits, armed sync-free argmax step: a _Launch, or None (stock path)."""
    ctx = _H.ctx
    if ctx is None:
        return None
    plan = _plan(ctx)
    if plan is None:
        _H.counts["skipped"] += 1
        return None
    st, inner, mode = plan
    return _Launch(ctx[0], st, inner, mode)


def _join_main(p) -> None:
    main = _cu.current_stream()
    if p.done is None:  # chain whose second half never launched: join everything S has so far
        _H.ev_done.record(_H.stream)
        p.done = _H.ev_done
    main.wait_event(p.done)
    if p.handoff is not None:
        p.handoff._xs_last = main  # main is now ordered after S's all-gather


def _drop_pending(why: str) -> None:
    p, _H.pending = _H.pending, None
    if p is None:
        return
    _join_main(p)
    _H.counts["dropped"] += 1
    if _H.counts["dropped"] <= 4:
        logger.warning("DSV41_MAINPROJ_HOIST: hoisted main_proj dropped (%s); propose runs the stock path", why)


def mainproj_consume(spec, args: tuple, kwargs: dict):
    """DSparkSpeculator.propose: swap the hoisted result in as last_hidden_states with aux_hidden_states=None, so
    DFlashSpeculator.propose skips combine_hidden_states and only copies (the same copy as today)."""
    p = _H.pending
    if p is None:
        return args, kwargs
    if "aux_hidden_states" in kwargs:
        aux = kwargs["aux_hidden_states"]
    else:
        aux = args[3] if len(args) > 3 else None
    if aux is not p.aux or kwargs.get("dummy_run", False):
        _drop_pending("aux mismatch")
        return args, kwargs
    _H.pending = None
    main = _cu.current_stream()
    _join_main(p)
    if p.mode == "gemm":
        _record_stream(p.y, main)
        out = p.inner.main_norm(p.st.gather(p.y, p.shape))
    else:
        _record_stream(p.out, main)
        out = p.out
    _H.counts["consumed"] += 1
    n = _H.counts["consumed"]
    if n == 1 or (LOG_EVERY and n % LOG_EVERY == 0):
        logger.info("DSV41_MAINPROJ_HOIST=%s: consumed %d, launched %d, skipped %d, dropped %d (M=%d)",
                    p.mode, n, _H.counts["launched"], _H.counts["skipped"], _H.counts["dropped"],
                    int(p.aux[0].shape[0]))
    if "aux_hidden_states" in kwargs:
        kwargs = dict(kwargs)
        kwargs["aux_hidden_states"] = None
    else:
        args = args[:3] + (None,) + args[4:]
    if "last_hidden_states" in kwargs:
        kwargs = dict(kwargs)
        kwargs["last_hidden_states"] = out
    else:
        args = args[:2] + (out,) + args[3:]
    return args, kwargs


# =====================================================================================================================
# G21  DSV41_DRAFT_PROLOGUE_FORK
# =====================================================================================================================
class _ForkState:
    def __init__(self) -> None:
        self.prepared = False
        self.ctx_ok = False
        self.sA = self.sB = None
        self.ev_fork = self.evA = self.evB = None
        self.pending: list = []
        self.forks = 0
        self.joins_attn = 0
        self.joins_fallback = 0


_F = _ForkState()


def prepare_fork(spec) -> None:
    """DSparkSpeculator.capture, eagerly, before any draft graph records: streams, events, the layer-0 join hook and
    the fold's private workspace. Any failure leaves the stock serial prologue."""
    if FORK_MODE == "0" or _F.prepared:
        return
    try:
        inner = spec.model.model
        attn0 = inner.layers[0].attn
        if not callable(getattr(attn0, "_prepare_and_attn_fn", None)):
            raise RuntimeError("draft layer 0 attention has no _prepare_and_attn_fn")
        if int(getattr(spec.block_tables, "cp_size", 1)) != 1:
            raise RuntimeError("context parallel drafter")
        _F.sA = _cu.Stream()
        _F.sB = _cu.Stream() if FORK_MODE == "two" else _F.sA
        _F.ev_fork, _F.evA, _F.evB = _cu.Event(), _cu.Event(), _cu.Event()
        cur = _cu.current_stream()
        for ev in (_F.ev_fork, _F.evA, _F.evB):
            ev.record(cur)  # torch creates the CUDA event lazily at its first record: do that here, not in capture
        if not getattr(attn0, "_dsv41_fork_join", False):
            orig = attn0._prepare_and_attn_fn

            def _prepare_and_attn_joined(*a, **k):
                if _F.pending:
                    _F.joins_attn += 1
                    _join_fork()
                return orig(*a, **k)

            attn0._prepare_and_attn_fn = _prepare_and_attn_joined
            attn0._dsv41_fork_join = True
        _F.ctx_ok = _prepare_ctx_fold(inner)
        _F.prepared = True
        logger.info("DSV41_DRAFT_PROLOGUE_FORK=%s: ctx-KV branch %s, metadata branch on, join before draft layer 0 "
                    "attention", FORK_MODE, "on (private fold workspace)" if _F.ctx_ok else "off (stock ctx path)")
    except Exception as e:  # noqa: BLE001
        logger.warning("DSV41_DRAFT_PROLOGUE_FORK disabled, serial prologue: %r", e)


def _prepare_ctx_fold(inner) -> bool:
    from vllm.models.deepseek_v41.nvidia import dsv41_boundary as bd

    if bd.CTX_MODE not in ("1", "split"):
        return False
    f = bd.ctxkv_state(inner)
    if f is None:
        return False
    if not getattr(f, "_dsv41_private_ws", False):
        f.ws = torch.empty_like(f.ws)  # same size/dtype as flashinfer's mm_mxfp8 scratch (also torch.empty there)
        f._dsv41_private_ws = True
    return True


def _ctx_forkable(spec, n: int) -> bool:
    """True when _precompute_context_kv(0, n) under capture takes the fold path (ctxkv_precompute returns True)."""
    if not _F.ctx_ok:
        return False
    from vllm.models.deepseek_v41.nvidia import dsv41_boundary as bd

    f = getattr(spec.model.model, "_dsv41_ctxkv", None)
    if not f or f.ok.get(int(n)) is None or bd._tuning():
        return False
    hs = spec.hidden_states
    return hs.dtype == torch.bfloat16 and hs.dim() == 2


def _join_fork() -> None:
    p, _F.pending = _F.pending, []
    cur = _cu.current_stream()
    for ev in p:
        cur.wait_event(ev)


def fork_prologue(spec, num_reqs: int, attn_metadata) -> bool:
    """Replaces _prepare_graph_forward while capturing. False = nothing ran; the caller runs the stock serial body."""
    if not _F.prepared or not _cu.is_current_stream_capturing() or _F.pending:
        return False
    n = spec._num_graph_context_tokens(num_reqs)
    do_ctx = _ctx_forkable(spec, n)
    do_meta = attn_metadata is not None and bool(spec.supports_decode_metadata_update)
    if not (do_ctx or do_meta):
        return False
    main = _cu.current_stream()
    if not do_ctx:
        spec._precompute_context_kv(0, n)  # stock ctx path stays on the capture stream, first, as today
    _F.ev_fork.record(main)
    pend = []
    if do_ctx:
        _F.sA.wait_event(_F.ev_fork)
        with _cu.stream(_F.sA):
            spec._precompute_context_kv(0, n)
        if FORK_MODE == "two" or not do_meta:
            _F.evA.record(_F.sA)
            pend.append(_F.evA)
    if do_meta:
        if not (FORK_MODE == "shared" and do_ctx):
            _F.sB.wait_event(_F.ev_fork)
        with _cu.stream(_F.sB):
            spec._update_draft_decode_metadata(attn_metadata, num_reqs)
        _F.evB.record(_F.sB)
        pend.append(_F.evB)
    _F.pending = pend
    _F.forks += 1
    if _F.forks == 1:
        logger.info("DSV41_DRAFT_PROLOGUE_FORK=%s: first forked capture (num_reqs %d, ctx rows %d, ctx %s, meta %s)",
                    FORK_MODE, num_reqs, n, do_ctx, do_meta)
    return True


def fork_join_fallback() -> None:
    """DSparkSpeculator._generate_draft after the forward: joins a fork the layer-0 hook did not (normally none)."""
    if _F.pending:
        _F.joins_fallback += 1
        if _F.joins_fallback <= 2:
            logger.warning("DSV41_DRAFT_PROLOGUE_FORK: fallback join after the draft forward (layer-0 hook bypassed)")
        _join_fork()


# =====================================================================================================================
# G14  DSV41_MHC_OVERLAP_DRAFT
# =====================================================================================================================
def draft_mhc_kwargs(vllm_config) -> dict:
    """Extra DeepseekV4DecoderLayer kwargs for the DSpark layers: the target's mHC side-stream path, own stream."""
    if not MHC_DRAFT:
        return {}
    try:
        from vllm.models.deepseek_v41.nvidia import model as m

        ovl = getattr(m, "_DSV41_MHC_OVL", None)
        if ovl is None:
            logger.warning("DSV41_MHC_OVERLAP_DRAFT=1 ignored: DSV41_MHC_OVERLAP is off for the target")
            return {}
        sup = getattr(m, "supports_mhc_overlap", None)
        if sup is not None and sup(vllm_config):
            logger.warning("DSV41_MHC_OVERLAP_DRAFT=1 ignored: the stock mHC overlap path applies (not the proven one)")
            return {}
        if not ovl.supported(vllm_config):
            return {}
        s = ovl.make_stream(vllm_config)
        if s is None:
            return {}
        logger.info("DSV41_MHC_OVERLAP_DRAFT=1: DSpark layers on their own mHC side stream (mode %s, proj %s, "
                    "pre %d, join %s, fuse_mhc_all_reduce False)", ovl.MODE, ovl.PROJ, int(ovl.OVERLAP_PRE), ovl.JOIN)
        return {"mhc_stream": s, "fuse_mhc_all_reduce": False}
    except Exception as e:  # noqa: BLE001
        logger.warning("DSV41_MHC_OVERLAP_DRAFT=1 ignored: %r", e)
        return {}
