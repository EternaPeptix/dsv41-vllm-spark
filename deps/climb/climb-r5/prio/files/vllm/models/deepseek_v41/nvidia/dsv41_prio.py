# SPDX-License-Identifier: Apache-2.0
"""climb r5 (r5-prio): stream-priority / ordering levers for the DSv4.1 decode graph. Scheduling only.

Imported lazily, and only from branches guarded by one of the master flags below; with every flag unset no
patched file imports this module and the executed path (kernels, streams, events, graph contents) is the
live one. Every lever runs the same kernels on the same inputs; only the stream a kernel is issued on, the
priority of that stream, or one extra event edge changes. No reduction order, tile, tactic or rounding
point is touched.

Flags (read once, at import of this module):

  DSV41_PRIO_CMP = prio | serial   (default off)
      KV-source layers (attention has a compressor: layers 2/8/14/20). Today the qkv_a shard GEMM
      (fi_mxfp8_quant + skinny_v2, 48 CTAs) is issued on the main stream right after the replicated BF16
      compressor GEMM (torch.mm fused_wkv_wgate, cuBLAS Kernel2 grid 8x8x16 = 1024 CTAs) on aux stream 0, and
      waits ~32 us for the compressor's CTA backlog (t2: 127-134 us/step of waits, 83-94 us/step on the CP).
        prio:   the qkv_a quant + shard GEMM run on a high-priority stream (fork from main right before the
                quant, join back into main right after the GEMM; the all-gather stays on main). The block
                scheduler then serves the 48 skinny CTAs ahead of the pending compressor CTAs. Needs the
                device gate (device_test.py T1) to show that a captured graph honours stream priority.
        serial: no priority: the compressor GEMM on aux stream 0 waits for an event recorded on main right
                after the shard GEMM (before its all-gather). Exact by construction, no device gate.
  DSV41_PRIO_SHARED = 1            (default off)
      MoE shared experts run on a dedicated high-priority stream instead of vLLM's global aux_stream(). Same
      kernels, same events (input_ready / output_ready), same join (the FC2 gather-add take() wait).
  DSV41_PRIO_ENGRAM_HOIST = 1      (default off)
      Engram layers (1, 14): the rows all-gather (same op, main stream) and the wkv local GEMM (quant + MXFP8
      GEMM, the exact quant_method.apply the ColumnParallelLinear runs) are issued at the start of an earlier
      decoder layer (DSV41_PRIO_ENGRAM_HOIST_AT, default "1:0,14:13" = engram layer:host layer); the GEMM
      runs on a lowest-priority side stream. The engram layer joins on the GEMM's done event and issues the
      wkv output all-gather exactly where ColumnParallelLinear.forward issued it.

  DSV41_PRIO_HI                    priority of the high streams: "max" (greatest the device allows) or an int
  DSV41_PRIO_ENGRAM_MAX_TOKENS     largest forward (tokens) that hoists (128)
  DSV41_PRIO_ENGRAM_HOIST_AT       engram_layer:host_layer pairs ("1:0,14:13")
"""
from __future__ import annotations

import functools
import os

import torch

try:  # vLLM logger when available (it is, in serving); plain logging in the CPU tests
    from vllm.logger import init_logger

    logger = init_logger(__name__)
except Exception:  # noqa: BLE001
    import logging

    logger = logging.getLogger(__name__)

TAG = "r5-prio"


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _on(v: str) -> bool:
    return v.lower() not in ("", "0", "off", "false", "no")


_cmp = _env("DSV41_PRIO_CMP").lower()
if _cmp in ("1", "prio", "priority", "hi"):
    CMP_MODE = "prio"
elif _cmp in ("serial", "order"):
    CMP_MODE = "serial"
else:
    if _on(_cmp):
        logger.warning("DSV41_PRIO_CMP=%r not understood (prio|serial); lever off", _cmp)
    CMP_MODE = ""
SHARED = _on(_env("DSV41_PRIO_SHARED"))
ENGRAM = _on(_env("DSV41_PRIO_ENGRAM_HOIST"))
HI = _env("DSV41_PRIO_HI", "max") or "max"
ENGRAM_MAX_TOKENS = int(_env("DSV41_PRIO_ENGRAM_MAX_TOKENS", "128") or "128")
ENGRAM_AT = _env("DSV41_PRIO_ENGRAM_HOIST_AT", "1:0,14:13") or "1:0,14:13"

COUNTS = {"cmp_prio": 0, "cmp_serial": 0, "cmp_fallback": 0, "shared_streams": 0, "eh_hoist": 0, "eh_take": 0,
          "eh_skip": 0, "eh_stale": 0}
_warned: set[str] = set()


def _warn_once(key: str, msg: str, *args) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning(msg, *args)


def _capturing() -> bool:
    try:
        return torch.cuda.is_current_stream_capturing()
    except Exception:  # noqa: BLE001
        return False


def priority_range() -> tuple[int, int]:
    """(least, greatest) stream priority of the device; CUDA: a lower number is a higher priority."""
    fn = getattr(torch.cuda.Stream, "priority_range", None)
    if fn is not None:
        least, greatest = fn()
    else:  # pragma: no cover - older torch
        least, greatest = torch._C._cuda_getPriorityRange()
    return int(least), int(greatest)


def hi_priority() -> int:
    """Greatest priority the device allows, or DSV41_PRIO_HI."""
    if HI.lower() in ("max", "hi", "greatest"):
        return priority_range()[1]
    return int(HI)


def lo_priority() -> int:
    return priority_range()[0]


_HI_STREAMS: dict = {}


def _hi_stream(role: str, device=None) -> torch.cuda.Stream | None:
    """One high-priority stream per (device, role); created outside capture only (None during capture).
    Roles get separate streams so the levers never add a stream-order edge between each other."""
    dev = torch.cuda.current_device() if device is None else device
    if isinstance(dev, torch.device):
        dev = dev.index if dev.index is not None else torch.cuda.current_device()
    s = _HI_STREAMS.get((dev, role))
    if s is None:
        if _capturing():
            return None
        s = torch.cuda.Stream(device=dev, priority=hi_priority())
        _HI_STREAMS[(dev, role)] = s
        logger.info("DSV41_PRIO: %s high-priority stream on device %s, priority %d (range %s)", role, dev,
                    s.priority, priority_range())
    return s


def active_line() -> str:
    return (f"DSV41_PRIO {TAG}: cmp={CMP_MODE or 'off'} shared={int(SHARED)} engram_hoist={int(ENGRAM)}"
            f" at={ENGRAM_AT if ENGRAM else '-'} hi={HI}")


logger.info(active_line())


# --------------------------------------------------------------------------------------------- CMPPRIO ---
class _Arm:
    """Host-side hand-off between cmp_parallel_input_projections() and the two shard-GEMM hooks in
    dsv41_r3_dense.QkvAShard.__call__ (cmp_enter / cmp_leave). Single-threaded: armed only for the duration
    of the default_fn call it wraps."""

    __slots__ = ("attn", "st", "mode", "main", "hi", "ctx", "done", "fired", "used")

    def __init__(self):
        self.clear()

    def clear(self):
        self.attn = self.st = self.mode = self.main = self.hi = self.ctx = self.done = None
        self.fired = False
        self.used = False


ARM = _Arm()


def _cmp_ready(attn, hidden_states) -> bool:
    """True when the r3-dense qkv_a shard path is the one default_fn will take (the hooks live there)."""
    st = getattr(attn, "_dsv41_qkv_shard", None)
    if not st or not hasattr(st, "gemm"):
        return False
    try:
        from vllm.models.deepseek_v41.nvidia import dsv41_r3_dense as r3

        m = r3._num_tokens(hidden_states)
        return 0 < m <= r3.QKV_MAX_M
    except Exception:  # noqa: BLE001
        return False


def _cmp_events(attn):
    ev = getattr(attn, "_dsv41_prio_ev", None)
    if ev is None:
        if _capturing():
            return None
        ev = torch.cuda.Event()
        attn._dsv41_prio_ev = ev
    return ev


def cmp_parallel_input_projections(attn, hidden_states, aux_fns, aux_streams, enable):
    """Drop-in for the execute_in_parallel call in DeepseekV4Attention._run_parallel_input_projections.

    Returns (qr_kv, kv_score, indexer_weights) exactly like the live code. Falls back to the live
    execute_in_parallel call (identical behaviour) whenever the lever does not apply."""
    from vllm.utils.multi_stream_utils import execute_in_parallel

    default_fn = lambda: attn._fused_wqa_wkv_gemm(hidden_states)  # noqa: E731
    start_event = attn.ln_events[0]
    done_events = attn.ln_events[1:3]
    hi = ev = None
    ok = (CMP_MODE in ("prio", "serial") and aux_streams is not None and enable and aux_fns[0] is not None
          and _cmp_ready(attn, hidden_states))
    if ok:
        ev = _cmp_events(attn)
        if CMP_MODE == "prio":
            hi = _hi_stream("cmp", hidden_states.device)
        ok = ev is not None and (CMP_MODE == "serial" or hi is not None)
        if not ok:
            _warn_once("cmp_cap", "DSV41_PRIO_CMP: first use during capture (%s); live path kept",
                       getattr(attn, "prefix", "?"))
    if not ok:
        COUNTS["cmp_fallback"] += 1
        qr_kv, (kv_score, indexer_weights) = execute_in_parallel(
            default_fn, aux_fns, start_event, done_events, aux_streams, enable=enable)
        return qr_kv, kv_score, indexer_weights

    assert len(aux_fns) == len(aux_streams) == len(done_events)
    aux_results: list = [None] * len(aux_fns)
    pending: list = []
    start_event.record()
    for i, fn in enumerate(aux_fns):  # same enqueue order as execute_in_parallel(default_first=False)
        if fn is None or (CMP_MODE == "serial" and i == 0):
            continue
        with torch.cuda.stream(aux_streams[i]):
            start_event.wait()
            aux_results[i] = fn()
            done_events[i].record()
        pending.append(done_events[i])
    ARM.clear()
    ARM.attn, ARM.st, ARM.mode = attn, attn._dsv41_qkv_shard, CMP_MODE
    ARM.main, ARM.hi, ARM.done = torch.cuda.current_stream(), hi, ev
    try:
        default_result = default_fn()
    finally:
        fired = ARM.used
        leaked = ARM.ctx is not None
        if leaked:  # cmp_leave never ran (exception inside the GEMM): restore the main stream
            ARM.ctx.__exit__(None, None, None)
        ARM.clear()
    if leaked:
        raise RuntimeError("DSV41_PRIO_CMP: shard GEMM returned without cmp_leave (high-priority stream left open)")
    if CMP_MODE == "serial":
        with torch.cuda.stream(aux_streams[0]):
            start_event.wait()
            if fired:
                ev.wait()  # recorded on main right after the shard GEMM, before its all-gather
            aux_results[0] = aux_fns[0]()
            done_events[0].record()
        pending.insert(0, done_events[0])
    if fired:
        COUNTS["cmp_" + CMP_MODE] += 1
    else:
        COUNTS["cmp_fallback"] += 1
        _warn_once("cmp_nofire", "DSV41_PRIO_CMP: shard hook did not fire for %s (stock qkv_a path?)",
                   getattr(attn, "prefix", "?"))
    for e in pending:
        e.wait()
    return default_result, aux_results[0], aux_results[1]


def cmp_enter(st, x):
    """Called by QkvAShard.__call__ right before its quant (or the r2 fused qkv path). Returns a token or None."""
    if ARM.st is None or ARM.st is not st or ARM.fired:
        return None
    ARM.fired = True
    if ARM.mode == "prio":
        ARM.hi.wait_stream(ARM.main)
        ARM.ctx = torch.cuda.stream(ARM.hi)
        ARM.ctx.__enter__()
    return ARM


def cmp_leave(tok, y):
    """Called exactly once per cmp_enter, right after the shard GEMM produced y ([M, n], before the all-gather).

    prio:   record done on the high stream, return to the main stream, main waits on done (join).
    serial: record done on main (the compressor on aux stream 0 waits on it).
    No record_stream is needed: y (and the quant buffers) live in the high stream's pool and the next use of
    that stream (the next cmp_enter) starts with hi.wait_stream(main), i.e. after every main-stream reader of y."""
    if tok is None:
        return y
    if tok.mode == "prio":
        tok.done.record(tok.hi)
        tok.ctx.__exit__(None, None, None)
        tok.ctx = None
        tok.main.wait_event(tok.done)
    else:
        tok.done.record(tok.main)
    tok.used = True
    return y


# ------------------------------------------------------------------------------------------ SHARED ---
def _reduces(layer) -> bool:
    """True if the shared MLP would issue a collective itself (down_proj reduce_results). rdma_ar refuses
    collectives only on vLLM's global aux_stream, so such a layer must stay there."""
    for name in ("down_proj", "w2"):
        m = getattr(layer, name, None)
        if m is not None and bool(getattr(m, "reduce_results", False)) and int(getattr(m, "tp_size", 1)) > 1:
            return True
    return False


def shared_stream(orig, layer=None):
    """Stream for SharedExperts (called once per SharedExperts.__init__, at model construction)."""
    if not SHARED or orig is None:
        return orig
    if layer is not None and _reduces(layer):
        _warn_once("sh_reduce", "DSV41_PRIO_SHARED: shared MLP reduces its own output; aux_stream kept")
        return orig
    try:
        s = _hi_stream("shared")
    except Exception as e:  # noqa: BLE001
        _warn_once("sh_fail", "DSV41_PRIO_SHARED: high-priority stream unavailable (%r); aux_stream kept", e)
        return orig
    if s is None:
        return orig
    COUNTS["shared_streams"] += 1
    if COUNTS["shared_streams"] == 1:
        logger.info("DSV41_PRIO_SHARED: shared experts on stream priority %d (was aux_stream priority %d)",
                    s.priority, getattr(orig, "priority", 0))
    return s


# ---------------------------------------------------------------------------------- ENGRAM HOIST ---
class _Hoisted:
    __slots__ = ("out", "x", "done", "n", "host")

    def __init__(self, out, x, done, n, host):
        self.out, self.x, self.done, self.n, self.host = out, x, done, n, host


class _EH:
    def __init__(self):
        self.on = False
        self.side = None
        self.plan: dict[int, list] = {}  # host layer idx -> [engram, ...]
        self.events: dict[int, torch.cuda.Event] = {}
        self.pending: dict[int, _Hoisted] = {}


EH = _EH()


def parse_at(spec: str) -> list[tuple[int, int]]:
    out = []
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        a, b = part.split(":")
        out.append((int(a), int(b)))
    return out


def _engram_ok(engram) -> str | None:
    """None when the live Engram.forward would take the plain (non-batch, TP-gathered) wkv path."""
    if getattr(engram, "_batch", None) is not None:
        return "EngramBatch path"
    if getattr(engram, "use_sequence_parallel", False):
        return "sequence parallel"
    wkv = getattr(engram, "wkv", None)
    if wkv is None or not hasattr(wkv, "quant_method"):
        return "no wkv"
    if getattr(wkv, "skip_bias_add", False) or getattr(wkv, "return_bias", True):
        return "wkv returns bias"
    qn = type(wkv).forward.__qualname__
    if qn != "ColumnParallelLinear.forward" and not qn.endswith(".ColumnParallelLinear.forward"):
        return f"wkv forward is {qn}"  # a subclass with its own forward: the split below would not mirror it
    if getattr(engram.embed_tokens, "tp_size", 1) == 1:
        return "rows not gathered (tp_size 1)"
    return None


def engram_install(model) -> None:
    """Called once from DeepseekV4Model.__init__ (after the layers and the EngramBatch exist)."""
    if not ENGRAM:
        return
    try:
        pairs = parse_at(ENGRAM_AT)
    except Exception as e:  # noqa: BLE001
        logger.warning("DSV41_PRIO_ENGRAM_HOIST_AT=%r unparsable (%r); lever off", ENGRAM_AT, e)
        return
    start, end = model.start_layer, model.end_layer
    plan: dict[int, list] = {}
    if not _hook_signature_ok(model, start, end):
        return
    for eid, host in pairs:
        if not (start <= host < eid < end):
            logger.warning("DSV41_PRIO_ENGRAM_HOIST: pair %d:%d outside local layers [%d,%d) or host>=engram; skipped",
                           eid, host, start, end)
            continue
        engram = getattr(model.layers[eid], "engram", None)
        if engram is None:
            logger.warning("DSV41_PRIO_ENGRAM_HOIST: layer %d has no engram; skipped", eid)
            continue
        why = _engram_ok(engram)
        if why is not None:
            logger.warning("DSV41_PRIO_ENGRAM_HOIST: layer %d not hoisted: %s", eid, why)
            continue
        plan.setdefault(host, []).append(engram)
        EH.events[id(engram)] = torch.cuda.Event()
        object.__setattr__(engram, "_dsv41_prio_take", functools.partial(engram_take, engram))  # hash_ids -> kv
    if not plan:
        logger.warning("DSV41_PRIO_ENGRAM_HOIST: nothing to hoist; lever off")
        return
    EH.side = torch.cuda.Stream(priority=lo_priority())
    EH.plan = plan
    for host in plan:
        model.layers[host].register_forward_pre_hook(functools.partial(_host_pre_hook, host), with_kwargs=True)
    model.register_forward_hook(_model_post_hook)
    EH.on = True
    logger.info("DSV41_PRIO_ENGRAM_HOIST: %s (side stream priority %d)",
                ", ".join(f"engram {e.layer_hash_index}@layer {h}" for h, es in plan.items() for e in es),
                EH.side.priority)


ENGRAM_HASHES_POS = 7  # DeepseekV4Model._run_layers: layer(hidden, positions, input_ids, pre_mix, post_mix,
                       #                                   res_mix, residual, engram_hashes, engram_mask, **kw)


def _hook_signature_ok(model, start, end) -> bool:
    """The host pre-hook reads engram_hashes positionally; refuse (lever off) if the layer signature moved."""
    import inspect

    try:
        layer = model.layers[start]
        params = [p for p in inspect.signature(layer.forward).parameters]
        pos = params.index("engram_hashes")
    except Exception as e:  # noqa: BLE001
        logger.warning("DSV41_PRIO_ENGRAM_HOIST: cannot read the decoder-layer signature (%r); lever off", e)
        return False
    if pos != ENGRAM_HASHES_POS:
        logger.warning("DSV41_PRIO_ENGRAM_HOIST: engram_hashes is parameter %d, expected %d; lever off", pos,
                       ENGRAM_HASHES_POS)
        return False
    return True


def _breakable_capture() -> bool:
    try:
        from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

        return bool(BreakableCUDAGraphCapture.is_active())
    except Exception:  # noqa: BLE001
        return False


def _host_pre_hook(host, _module, args, kwargs):
    if not EH.on:
        return None
    hashes = kwargs.get("engram_hashes", args[ENGRAM_HASHES_POS] if len(args) > ENGRAM_HASHES_POS else None)
    if hashes is None or kwargs.get("dsv41_sp", False):
        return None
    n = int(hashes.shape[0])
    if n == 0 or n > ENGRAM_MAX_TOKENS or _breakable_capture():
        COUNTS["eh_skip"] += 1
        return None
    for engram in EH.plan[host]:
        hoist(engram, hashes[:, engram.layer_hash_index], host)
    return None


def hoist(engram, hash_ids, host=-1) -> None:
    """Main stream: the engram's rows all-gather (Engram.embed, unchanged). Side stream: wkv local GEMM."""
    key = id(engram)
    if key in EH.pending:
        _flush("re-hoist before take")
    main = torch.cuda.current_stream()
    x = engram.embed(hash_ids).flatten(-2)  # == the operand of self.wkv(...) in Engram.forward
    # x is read on the side stream; rec keeps the Python reference (no main-stream reuse of its block)
    # until engram_take has joined the side stream back into main.
    wkv = engram.wkv
    bias = wkv.bias if not wkv.skip_bias_add else None
    done = EH.events[key]
    EH.side.wait_stream(main)
    with torch.cuda.stream(EH.side):
        out = wkv.quant_method.apply(wkv, x, bias)  # == ColumnParallelLinear.forward's matrix multiply
        done.record(EH.side)
    EH.pending[key] = _Hoisted(out, x, done, int(hash_ids.shape[0]), host)
    COUNTS["eh_hoist"] += 1


def engram_take(engram, hash_ids):
    """Called from Engram.forward in place of `self.wkv(self.embed(hash_ids).flatten(-2))` whenever the hoist is
    installed for this engram. Returns the gathered wkv output: the hoisted GEMM joined + its all-gather, or, when
    nothing was hoisted for this forward (skipped size, SP, lever switched off), exactly the live statement."""
    rec = EH.pending.pop(id(engram), None)
    if rec is None:
        return engram.wkv(engram.embed(hash_ids).flatten(-2))  # the live statement, unchanged
    if int(hash_ids.shape[0]) != rec.n:
        raise RuntimeError(f"DSV41_PRIO_ENGRAM_HOIST: hoisted {rec.n} rows, layer has {int(hash_ids.shape[0])}")
    main = torch.cuda.current_stream()
    main.wait_event(rec.done)  # join: everything below reads rec.out on main
    # No record_stream: rec.out lives in the side stream's pool and the side stream's next use (the next
    # hoist) starts with side.wait_stream(main), i.e. after this all-gather; rec.x (main pool) was kept
    # alive by rec until this join.
    wkv = engram.wkv
    if wkv.gather_output and wkv.tp_size > 1:  # == ColumnParallelLinear.forward
        from vllm.distributed import tensor_model_parallel_all_gather

        out = tensor_model_parallel_all_gather(rec.out)
    else:
        out = rec.out
    COUNTS["eh_take"] += 1
    return out


def _flush(why: str) -> None:
    """A hoisted GEMM nobody consumed: join it (keeps a capture legal) and switch the lever off for good."""
    if not EH.pending:
        return
    main = torch.cuda.current_stream()
    for rec in EH.pending.values():
        main.wait_event(rec.done)
    COUNTS["eh_stale"] += len(EH.pending)
    EH.pending.clear()
    EH.on = False
    logger.error("DSV41_PRIO_ENGRAM_HOIST: unconsumed hoist (%s); joined and lever switched OFF", why)


def _model_post_hook(_module, _args, _out):
    if EH.pending:
        _flush("forward ended")
    return None
