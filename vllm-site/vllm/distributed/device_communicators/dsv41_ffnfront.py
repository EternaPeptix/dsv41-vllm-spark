# SPDX-License-Identifier: Apache-2.0
"""DSv4.1 climb round 5 (r5-ffnfront): launch-order levers for the FFN front (post-attention AR -> router top-k).

Imported only when one of its levers is switched on (lazily, from inside the gated branches of moe_runner.py,
dsv41_mhc_overlap.py, model.py and dsv41_l2pf_r1.py). With every flag below unset this file is never imported
and none of the patched files changes a kernel, stream, event, allocation or graph node.

Every lever here moves WHEN a kernel is enqueued (its CUDA-graph node creation order) and never what it computes:
the same kernels with the same inputs and the same dependency edges (each side branch still waits on an event
recorded at exactly today's fork point), so every output bit is unchanged by construction.

  DSV41_MOE_GATE_FIRST=a   the router GEMM (+ splitK reduce) is enqueued BEFORE the shared-expert branch; the shared
                           branch's input-ready event is still recorded at today's point (before the router), so its
                           dependency edges are today's. Only the launch order of the two ready siblings changes.
  DSV41_MOE_GATE_FIRST=b   the router GEMM first AND the shared branch forked after it (the event is recorded after
                           the router GEMM: one more edge, router -> shared). Delays the shared branch ~3-6 us.
  DSV41_MHC_OVERLAP_FORK_LATE=1  the mHC coefficient half (Moet mhc_proj + pre_norm "stats") is enqueued after the
                           main successor of the boundary (the oblit projection, model.py hook) instead of right after
                           mhc_post_norm; it waits on an event recorded right after mhc_post_norm (today's fork point).
  DSV41_L2_PREFETCH_R1_POST_EVENT=mpn|succ  (dsv41_l2pf_r1.py) the post-AR SHW2 prefetch is launched from an event
                           recorded right after the all-reduce kernel, enqueued after the mHC boundary (mpn) or after
                           the boundary's main successor (succ, model.py hook).
"""
from __future__ import annotations

import collections
import os
import sys

import torch

TAG = "r5-ffnfront"
R1_MODULE = "vllm.distributed.device_communicators.dsv41_l2pf_r1"


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip().lower()


def _gate_first_mode() -> str:
    raw = _env("DSV41_MOE_GATE_FIRST", "")
    if raw in ("", "0", "off"):
        return ""
    if raw not in ("a", "b"):
        raise ValueError(f"DSV41_MOE_GATE_FIRST={raw!r}: use a or b (unset = off)")
    return raw


GATE_FIRST = _gate_first_mode()
FORK_LATE = _env("DSV41_MHC_OVERLAP_FORK_LATE", "0") in ("1", "on", "true")
POST_EVENT_SUCC = _env("DSV41_L2_PREFETCH_R1_POST_EVENT", "") == "succ"

STATS: collections.Counter = collections.Counter()
_LOGGED: set = set()
_logger = None


def _log():
    global _logger
    if _logger is None:
        from vllm.logger import init_logger

        _logger = init_logger(__name__)
    return _logger


def _once(key, level: str, msg: str, *args) -> None:
    if key in _LOGGED:
        return
    _LOGGED.add(key)
    try:
        getattr(_log(), level)(msg, *args)
    except Exception:  # noqa: BLE001 - logging must never break a step (CPU tests run without vllm)
        pass


# ------------------------------------------------------------------------------ successor hook (model.py) -----
def after_successor(mhc_stream) -> None:
    """model.py: right after the main successor of an mHC boundary was enqueued (the oblit projection of the
    sublayer input; the boundary itself when the layer has no projection or the r2-oblit fold took it)."""
    if mhc_stream is not None:
        flush_mhc(mhc_stream, "succ")
    if POST_EVENT_SUCC:
        r1 = sys.modules.get(R1_MODULE)
        if r1 is not None and getattr(r1.R, "deferred", None):
            r1.flush_deferred("succ")


# ------------------------------------------------------------------------- mHC fork-late (dsv41_mhc_overlap) -----
def defer_mhc(stream, side_fn) -> None:
    """Record today's fork point (right after the critical boundary kernels) on the caller stream and keep the
    side-stream work for flush_mhc. The event is the one torch's stream.wait_stream(main) would record here."""
    if getattr(stream, "_dsv41_r5_deferred", None) is not None:  # never two open boundaries on one stream
        flush_mhc(stream, "stale")  # raises (fail stop, see _FATAL)
    ev = torch.cuda.current_stream().record_event()
    stream._dsv41_r5_deferred = (ev, side_fn)


# Flush points that are SAFE: the side stream is joined into the caller stream right after the flush, before
# anything reads post / comb / next_pre (succ: the decoder layer's join follows; join / ar: the join itself).
# Every other point means a boundary's coefficient half is still deferred when the NEXT boundary (or mhc_pre)
# starts, i.e. the decoder layer's join did not run: there the flushed side work would race the reader of its
# outputs (RAW on post / comb / next_pre) and the caller drops _dsv41_mhc_pending right after (early release).
# Stock has no such path, so it is a bug, not a fallback: fail stop instead of flushing.
_FATAL = {"stale": "a second boundary was deferred on the same side stream before the first one was flushed",
          "next": "the next mHC boundary started with the previous coefficient half still deferred",
          "pre": "mhc_pre (overlap) started with the previous coefficient half still deferred",
          "empty": "an empty fork (served mHC path) found the previous coefficient half still deferred"}


def flush_mhc(stream, why: str) -> bool:
    """Enqueue the deferred coefficient half: the side stream waits on the recorded fork-point event (so its
    dependency edges are today's), then the unchanged kernels run on it. Returns True when something ran.
    At an unsafe point (_FATAL) with work still deferred it raises instead (no join would cover the work)."""
    d = getattr(stream, "_dsv41_r5_deferred", None)
    if d is None:
        return False
    if why in _FATAL:
        STATS[f"mhc_fatal_{why}"] += 1
        raise RuntimeError(f"DSV41_MHC_OVERLAP_FORK_LATE ({TAG}): {_FATAL[why]} (flush point {why!r}); the decoder "
                           "layer's join did not run for that boundary. Refusing to flush there (it would race the "
                           "next boundary's read of post/comb/next_pre). Unset DSV41_MHC_OVERLAP_FORK_LATE.")
    stream._dsv41_r5_deferred = None
    ev, side_fn = d
    stream.wait_event(ev)
    side_fn()
    STATS[f"mhc_{why}"] += 1
    if why == "succ":
        _once(("mhc", why), "info", "DSV41_MHC_OVERLAP_FORK_LATE (%s): mHC coefficient half enqueued after the "
              "boundary's main successor (first flush: succ)", TAG)
    else:
        _once(("mhc", why), "warning", "DSV41_MHC_OVERLAP_FORK_LATE (%s): fallback flush at %s (the successor hook "
              "did not run for this boundary; the join that follows covers it, order as stock there)", TAG, why)
    return True


# --------------------------------------------------------------------------- router first (moe_runner) ---------
def _gate(runner, hidden_states):
    """The runner's own router step, verbatim (moe_runner._forward_impl, 'if self.gate is not None:' block)."""
    if runner._fse_fuse_gate:
        runner._maybe_fuse_gate_weights()
        from vllm.model_executor.layers.utils import dispatch_unquantized_gemm  # the object moe_runner.py imports

        return dispatch_unquantized_gemm()(runner, hidden_states, runner._combined_gate_weight, None)
    router_logits, _ = runner.gate(hidden_states)
    return router_logits


def gate_first(runner, hidden_states, shared_experts_input):
    """(shared_experts_overlapping, router_logits) for MoERunner._forward_impl with the router GEMM enqueued
    before the shared-expert branch. Called only when the runner holds the gate."""
    se = runner._shared_experts
    if se is None:
        return False, _gate(runner, hidden_states)
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExpertsOrder

    overlapped = (se._determine_shared_experts_order(shared_experts_input)
                  == SharedExpertsOrder.MULTI_STREAM_OVERLAPPED)
    if GATE_FIRST == "b" or not overlapped:
        # b: stock maybe_forward_async AFTER the router, i.e. its input-ready event is recorded after the router
        # GEMM. Not overlapped: maybe_forward_async enqueues nothing and returns False (shared runs later in
        # _apply_quant_method, exactly as stock), so the order of the two calls is immaterial.
        router_logits = _gate(runner, hidden_states)
        ok = se.maybe_forward_async(shared_experts_input)
        STATS[f"gate_first_{GATE_FIRST}_{int(ok)}"] += 1
        _once(("gf", GATE_FIRST, ok), "info", "DSV41_MOE_GATE_FIRST=%s (%s): router GEMM enqueued before the "
              "shared-expert branch (overlapped=%s)", GATE_FIRST, TAG, ok)
        return ok, router_logits
    # a: SharedExperts.maybe_forward_async split around the router GEMM. The input-ready event is recorded where
    # stock records it (before the router), so the shared branch keeps exactly its stock dependencies.
    from vllm.utils.torch_utils import current_stream  # the object shared_experts.py imports

    idx = se._output_idx
    assert se._output[idx] is None
    se._input_ready_event[idx].record(current_stream())
    router_logits = _gate(runner, hidden_states)
    with torch.cuda.stream(se._stream):
        se._input_ready_event[idx].wait(se._stream)
        se._output[idx] = se._forward_layer(shared_experts_input)
        se._output_ready_event[idx].record(se._stream)
    STATS["gate_first_a_1"] += 1
    _once(("gf", "a", True), "info", "DSV41_MOE_GATE_FIRST=a (%s): router GEMM enqueued before the shared-expert "
          "branch, shared input event at the stock point (overlapped=True)", TAG)
    return True, router_logits


def forward_impl_gate_first(runner, hidden_states, router_logits, shared_experts_input, input_ids):
    """MoERunner._forward_impl after its first statement (_ensure_moe_quant_config_init), with the shared-fork +
    router block replaced by gate_first(). The tail is the stock tail verbatim (tests/test_ffnfront.py compares
    the AST of this function's `with` block against the live moe_runner.py one)."""
    shared_experts_overlapping, router_logits = gate_first(runner, hidden_states, shared_experts_input)
    self = runner  # keeps the tail below token-identical to the stock source
    with self._sequence_parallel_context():
        # TODO(bnell): parts of the dispatch/combine steps will go away once
        # #32567 lands and the remaining kernels are made MKs.  The PCP
        # code will probably remain
        hidden_states, router_logits = self._maybe_dispatch(
            hidden_states,
            router_logits,
        )

        shared_output, hidden_states = self._apply_quant_method(
            hidden_states=hidden_states,
            router_logits=router_logits,
            shared_experts_input=shared_experts_input,
            input_ids=input_ids,
            shared_experts_overlapping=shared_experts_overlapping,
        )

        return self._maybe_combine(
            shared_output,
            hidden_states,
        )
