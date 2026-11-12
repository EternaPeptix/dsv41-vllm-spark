# SPDX-License-Identifier: Apache-2.0
"""DSv4.1 r1-attn IDXPAR (audit lever G20 / attn:L-IP): overlap the three attention-prologue chains
of the 8 index-source layers inside captured FULL decode graphs.

Gate: DSV41_IDXPAR (default unset/0 = this module is never imported; attention.py runs stock).
  DSV41_IDXPAR=1     three branches, stream priorities (indexer q chain high, compressor chain medium,
                     main q chain at the default priority 0)
  DSV41_IDXPAR=flat  same three branches, every branch at priority 0 (diagnostic arm)

Stock (attention.py _prepare_and_attn), after q_kv_norm, all on the capture (main) stream unless noted:
  kv-source layers 2/8/14/20:   main wq_b -> fused qnorm/rope/SWA insert  || aux0: compressor (latent)
                                join; wk -> indexer-K store -> indexer wq_b -> fused q rope/quant
                                      || aux0: compressed-cache insert; join
  index-only layers 24..36:     main wq_b -> SWA insert -> indexer wq_b -> fused q rope/quant (serial)
  then logits / top-k / sparse MLA (_sparse_indexer_and_attn) on main.

IDXPAR (captured FULL graphs only; eager, profile and breakable/piecewise forwards run the stock path):
  main:  record E0
  low  = aux_stream_list[1] (stock stream, priority 0):  wait E0; main wq_b -> SWA insert; record El
  med  = own stream (priority -1), kv-source only:       wait E0; compressor -> wk -> indexer-K store
                                                          -> compressed-cache insert; record Em
  high = own stream (priority -2):                        wait E0; indexer wq_b -> q rope/quant; record Eh
  main:  wait El, (Em), Eh -> _sparse_indexer_and_attn (unchanged)
Host order low -> med -> high keeps the module call order of the stock path (main wq_b, compressor, wk,
indexer wq_b), so the L2-prefetch recorder learns the same weight-read sequence and the same wq_b
ordinals, and the WOA fork at the LAST wq_b post-hook (the indexer wq_b) waits on the high stream right
after the indexer wq_b -- the same point in the indexer chain as today, never after the main wq_b.

Priorities inside the graph: torch (2.13.0+cu130, git cf30153c) instantiates every CUDAGraph with
cudaGraphInstantiateFlagAutoFreeOnLaunch | cudaGraphInstantiateFlagUseNodePriority (flags = 9; verified in
the live libtorch_cuda.so: `mov x2, #0x9` before `bl cudaGraphInstantiateWithFlags`), and CUDA copies the
capturing stream's priority into each captured kernel node. So the node priorities set here are honoured
at replay. See README.md.

Exactness: the same kernels with the same launch configurations read the same inputs; only the stream a
kernel is issued on, its priority and the cross-stream event edges change. Every buffer a branch writes is
private to that branch (SWA cache rows / q, compressor state + latent + indexer-K cache + compressed cache,
indexer q / weights) and every consumer waits on an event recorded after its producer.
"""
from __future__ import annotations

import os
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

MODE = os.environ.get("DSV41_IDXPAR", "0").strip().lower()
_MODES = {"1": True, "on": True, "prio": True, "flat": False}
if MODE not in _MODES:
    raise ValueError(f"DSV41_IDXPAR={MODE!r}: expected one of {sorted(_MODES)} (or unset/0 = off)")
PRIO = _MODES[MODE]
# CUDA: lower number = higher priority; 0 is the default and the LEAST priority. torch clamps to the
# device range (torch keeps at most 4 levels: 0, -1, -2, -3).
HI_PRIORITY = -2 if PRIO else 0
MED_PRIORITY = -1 if PRIO else 0

# Test seams (CPU plan tests replace these with fakes).
_cuda: Any = torch.cuda


def _get_forward_context():
    from vllm.forward_context import get_forward_context

    return get_forward_context()


def _breakable_active() -> bool:
    try:
        from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
    except Exception:  # noqa: BLE001
        return False
    return BreakableCUDAGraphCapture.is_active()


def _warm_blas(stream, device) -> None:
    """Allocate cuBLAS's per-(handle, stream) workspace for this stream eagerly, so the first cuBLAS call
    on it (wk on the medium stream) never allocates inside a graph capture."""
    with _cuda.stream(stream):
        x = torch.zeros((2, 512), dtype=torch.bfloat16, device=device)
        w = torch.zeros((128, 512), dtype=torch.bfloat16, device=device)
        torch.nn.functional.linear(x, w)
        torch.mm(x, w.t())


class _Shared:
    """The two IDXPAR streams of one device (created once, at model construction, outside any capture)."""

    def __init__(self, device: int) -> None:
        self.device = device
        self.hi = _cuda.Stream(device=device, priority=HI_PRIORITY)
        self.med = _cuda.Stream(device=device, priority=MED_PRIORITY)
        self.warmed = False
        rng = None
        try:
            rng = _cuda.Stream.priority_range()
        except Exception:  # noqa: BLE001
            pass
        hp, mp = _stream_priority(self.hi), _stream_priority(self.med)
        logger.info(
            "DSV41_IDXPAR=%s armed: indexer-q stream priority %s, compressor stream priority %s, "
            "main-q stream = aux_stream_list[1] (priority 0); torch priority range %s",
            MODE, hp, mp, rng,
        )
        if PRIO and hp is not None and mp is not None and hp == mp:
            logger.warning("DSV41_IDXPAR: the device offers only one priority level above the default; "
                           "the indexer-q and compressor streams share priority %s", hp)

    def warm(self) -> None:
        if self.warmed:
            return
        main = _cuda.current_stream()
        for s in (self.hi, self.med):
            s.wait_stream(main)
            _warm_blas(s, self.device)
            main.wait_stream(s)
        self.warmed = True


def _stream_priority(s) -> int | None:
    try:
        return int(s.priority)
    except Exception:  # noqa: BLE001
        return None


_SHARED: dict[int, _Shared] = {}
_LOGGED: set = set()


def _shared(device: int) -> _Shared:
    sh = _SHARED.get(device)
    if sh is None:
        sh = _SHARED[device] = _Shared(device)
    return sh


class LayerPlan:
    """Per-attention-layer IDXPAR state: its events and the shared streams."""

    @classmethod
    def create(cls, attn) -> "LayerPlan | None":
        indexer = getattr(attn, "indexer", None)
        if indexer is None:
            return None  # sliding-window / consumer layers: stock
        aux = getattr(attn, "aux_stream_list", None)
        if aux is None or len(aux) < 2:
            if "noaux" not in _LOGGED:
                _LOGGED.add("noaux")
                logger.warning("DSV41_IDXPAR: no aux streams (ROCm / no multi-stream); stock path")
            return None
        compressor = getattr(attn, "compressor", None)
        if (compressor is not None) != bool(getattr(indexer, "owns_k", False)):
            logger.warning("DSV41_IDXPAR: layer %s has compressor=%s owns_k=%s; stock path",
                           getattr(attn, "layer_id", "?"), compressor is not None, indexer.owns_k)
            return None
        return cls(attn, aux[1], _shared(_cuda.current_device()))

    def __init__(self, attn, low, shared: _Shared) -> None:
        self.attn = attn
        self.low = low
        self.shared = shared
        self.kv_source = attn.compressor is not None
        # E0 fork, El / Em / Eh joins. Created now (outside capture); materialised by the first eager forward.
        self.events = [_cuda.Event() for _ in range(4)]
        self.ready = False

    def _prime(self) -> None:
        """First eager forward of this layer: warm the shared streams' cuBLAS workspaces and create the
        CUDA events (torch creates them lazily at the first record), so nothing is created in a capture."""
        self.shared.warm()
        main = _cuda.current_stream()
        for e in self.events:
            e.record(main)
        self.ready = True

    def run(self, hidden_states, qr, kv, qr_scale, kv_score, indexer_weights, positions, attn_out) -> bool:
        """IDXPAR prologue + the stock _sparse_indexer_and_attn. Returns False (having enqueued nothing on
        the GPU apart from the one-time priming in an eager forward) when the stock path must run."""
        if not _cuda.is_current_stream_capturing():
            if not self.ready:
                self._prime()
            return False
        if not self.ready:
            key = ("unprimed", self.attn.layer_id)
            if key not in _LOGGED:
                _LOGGED.add(key)
                logger.warning("DSV41_IDXPAR: layer %s captured before any eager forward; stock path in "
                               "this graph", self.attn.layer_id)
            return False
        if _breakable_active():
            return False  # breakable (piecewise) capture serialises multi-stream work: stock path
        attn_metadata = _get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return False
        self._schedule(hidden_states, qr, kv, qr_scale, kv_score, indexer_weights, positions, attn_out,
                       attn_metadata)
        return True

    def _schedule(self, hidden_states, qr, kv, qr_scale, kv_score, indexer_weights, positions, attn_out,
                  attn_metadata) -> None:
        a = self.attn
        indexer = a.indexer
        compressor = a.compressor
        low, hi, med = self.low, self.shared.hi, self.shared.med
        ev0, ev_low, ev_med, ev_hi = self.events
        main = _cuda.current_stream()

        # Every input (qr, kv, positions, kv_score, indexer_weights) was produced on, or joined into, main
        # before this point (_run_parallel_input_projections joins its aux streams; _split_qkv_and_norm
        # runs on main).
        ev0.record(main)

        # 1) main q chain on aux_stream_list[1], priority 0. Launched FIRST on the host (see module doc).
        with _cuda.stream(low):
            ev0.wait(low)
            q = a._wq_b_proj(qr, qr_scale).view(-1, a.n_local_heads, a.head_dim)
            q = a._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)
            ev_low.record(low)

        # 2) kv-source layers: compressor -> wk -> indexer-K store -> compressed-cache insert.
        if compressor is not None:
            with _cuda.stream(med):
                ev0.wait(med)
                latent = compressor(kv_score, positions)
                indexer._produce_k(latent, positions, a.indexer_rotary_emb)
                compressor.insert_cache(latent, positions, a.rotary_emb)
                ev_med.record(med)

        # 3) indexer q chain, highest priority. Its wq_b is the layer's last wq_b call on the host, so the
        # L2-prefetch WOA fork (dsv41_l2_prefetch._wq_b_post_hook) forks from this stream.
        with _cuda.stream(hi):
            ev0.wait(hi)
            index_q, index_q_scale, index_weights_out = indexer.forward_q(
                qr, qr_scale, indexer_weights, positions, a.indexer_rotary_emb
            )
            ev_hi.record(hi)

        # Join every branch before the first reader (logits read the indexer-K cache and the indexer q;
        # sparse MLA reads q, the SWA cache and the compressed cache). No branch outlives this function.
        ev_low.wait(main)
        if compressor is not None:
            ev_med.wait(main)
        ev_hi.wait(main)

        a._sparse_indexer_and_attn(
            hidden_states,
            index_q,
            index_q_scale,
            index_weights_out,
            q,
            kv,
            positions,
            attn_out,
        )
