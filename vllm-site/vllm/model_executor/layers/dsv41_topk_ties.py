# SPDX-License-Identifier: Apache-2.0
"""g5det (goal window 5, Determinism phase, 2026-10-27): deterministic tie-breaking for the
sparse-indexer top-k, plus a flag-gated debug dump. Default OFF: nothing here runs unless
DSV41_TOPK_CANON_TIES or DSV41_DET_DUMP is set (sparse_attn_indexer.py checks ENABLED).

Why: vLLM's top-k kernels (top_k_per_row_prefill, persistent_topk / top_k_per_row_decode)
select the right index SET when the k-th value is unique, but when several logits equal the
row's k-th value ("boundary ties") they keep an arbitrary subset of them, chosen by an
atomic race (g5det tk_tie_test: the set changes run to run on ~every tied row). DSV41_TOPK_CANON
only fixes the ORDER of the selected indices, not the SET, so a boundary tie changes which
KV entries a query attends to -> run-to-run drift on long prompts.

DSV41_TOPK_CANON_TIES:
  0 / unset : stock (default)
  prefill   : prefill rows only (top_k_per_row_prefill path)
  all       : prefill and decode rows
Rule: every logit strictly above the row's k-th value is kept (the stock kernels already
keep them: they return a valid top-k), and among the logits EQUAL to the k-th value the
lowest column indices are kept. Rows without a boundary tie are not touched (bitwise the
stock output); rewritten rows come out ascending (= the DSV41_TOPK_CANON order). Rows no
longer than k, rows with a NaN k-th value and unsupported layouts are left to the stock kernel.

DSV41_DET_DUMP=<dir> (debug instances only): while <dir>/ON exists on a node, TP rank 0 (all
ranks with DSV41_DET_DUMP_ALL=1) saves, per indexer call, the raw stock top-k rows, the final
rows, per-row boundary statistics (k-th value, #greater, #equal, rewritten) and per-row
fingerprints of the logits, the index query, the head weights, the attention input and the
gathered index keys to <dir>/r<rank>_<seq>_<kind>.pt (synchronous; slow). With DUMP set, a
file <dir>/TIES (0 / prefill / all) overrides DSV41_TOPK_CANON_TIES at run time; it must be
written on all 8 nodes between requests (each rank ranks its own copy of the rows).
"""

from __future__ import annotations

import os

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

_MODES = ("0", "prefill", "all")


def _norm_mode(m: str) -> str:
    m = (m or "").strip().lower()
    if m in ("", "0", "off", "false", "no"):
        return "0"
    if m == "1":
        return "prefill"
    return m


_MODE = _norm_mode(os.environ.get("DSV41_TOPK_CANON_TIES", "0"))
if _MODE not in _MODES:
    raise ValueError(f"DSV41_TOPK_CANON_TIES={_MODE!r}: expected 0, prefill or all")
TIES_PREFILL = _MODE in ("prefill", "all")
TIES_DECODE = _MODE == "all"
DUMP_DIR = os.environ.get("DSV41_DET_DUMP", "").strip()
DUMP = bool(DUMP_DIR)
DUMP_ALL_RANKS = os.environ.get("DSV41_DET_DUMP_ALL", "0") == "1"
ENABLED = TIES_PREFILL or TIES_DECODE or DUMP
_BLOCK = 1024
K_SUPPORTED = 512

if TIES_PREFILL or TIES_DECODE:
    logger.warning(
        "DSV41_TOPK_CANON_TIES=%s: boundary ties of the indexer top-k keep the lowest "
        "indices (prefill=%s decode=%s)", _MODE, TIES_PREFILL, TIES_DECODE,
    )
if DUMP:
    logger.warning(
        "DSV41_DET_DUMP=%s armed (dumps while %s/ON exists; %s/TIES overrides the tie mode)",
        DUMP_DIR, DUMP_DIR, DUMP_DIR,
    )


@triton.jit(do_not_specialize=["l_stride", "t_stride"])
def _dsv41_topk_ties_kernel(
    Logits,
    l_stride,
    Starts,
    Ends,
    Topk,
    t_stride,
    Vk,
    Ngt,
    Neq,
    Fixed,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    LOCAL: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    DO_FIX: tl.constexpr,
    STATS: tl.constexpr,
):
    # One program per row. Branch-free scalar control: loops run 0 times when inactive.
    row = tl.program_id(0).to(tl.int64)
    if HAS_STARTS:
        s = tl.load(Starts + row).to(tl.int64)
    else:
        s = row * 0
    e = tl.load(Ends + row).to(tl.int64)
    n = e - s
    base = Logits + row * l_stride
    trow = Topk + row * t_stride
    sel = tl.load(trow + tl.arange(0, K))
    if LOCAL:
        col = sel.to(tl.int64) + s
    else:
        col = sel.to(tl.int64)
    vals = tl.load(base + col, mask=(sel >= 0) & (n > K), other=float("inf"))
    vk = tl.min(vals, axis=0)
    # -inf boundaries count too (masked rows); NaN and +inf (nothing valid selected) do not.
    active = (n > K) & (vk == vk) & (vk < float("inf"))
    n_scan = tl.where(active, n, 0)
    zero = tl.sum(tl.zeros([BLOCK], dtype=tl.int32), axis=0)
    ngt = zero
    neq = zero
    for off in range(0, n_scan, BLOCK):
        c = s + off + tl.arange(0, BLOCK)
        x = tl.load(base + c, mask=c < e, other=float("nan"))
        ngt += tl.sum((x > vk).to(tl.int32), axis=0)
        neq += tl.sum((x == vk).to(tl.int32), axis=0)
    tie = active & ((ngt + neq) > K)
    if DO_FIX:
        # Keep every x > vk and the lowest-index x == vk until K are kept, ascending.
        n_fix = tl.where(tie, n, 0)
        need = K - ngt
        pos = zero
        seen = zero
        for off in range(0, n_fix, BLOCK):
            c = s + off + tl.arange(0, BLOCK)
            x = tl.load(base + c, mask=c < e, other=float("nan"))
            eq = (x == vk).to(tl.int32)
            take = (x > vk) | ((eq != 0) & ((seen + tl.cumsum(eq, axis=0)) <= need))
            ti = take.to(tl.int32)
            dst = pos + tl.cumsum(ti, axis=0) - 1
            if LOCAL:
                val = (c - s).to(tl.int32)
            else:
                val = c.to(tl.int32)
            tl.store(trow + dst, val, mask=take)
            pos += tl.sum(ti, axis=0)
            seen += tl.sum(eq, axis=0)
    if STATS:
        tl.store(Vk + row, vk)
        tl.store(Ngt + row, tl.where(active, ngt, -1))
        tl.store(Neq + row, tl.where(active, neq, -1))
        tl.store(Fixed + row, tie.to(tl.int32))


# Decode rows are few and long (the logits width is max_model_len / ratio): split each row's
# count over SPLITS programs (partials, no atomics), then one program per row decides and, for a
# tied row only, rewrites it exactly as the single-pass kernel does.
@triton.jit(do_not_specialize=["l_stride", "t_stride", "chunk"])
def _dsv41_ties_partial_kernel(
    Logits,
    l_stride,
    Ends,
    Topk,
    t_stride,
    Part,
    chunk,
    SPLITS: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    sp = tl.program_id(1).to(tl.int64)
    n = tl.load(Ends + row).to(tl.int64)
    lo = sp * chunk
    hi = tl.minimum(lo + chunk, n)
    base = Logits + row * l_stride
    sel = tl.load(Topk + row * t_stride + tl.arange(0, K))
    vals = tl.load(base + sel.to(tl.int64), mask=(sel >= 0) & (n > K) & (lo < n), other=float("inf"))
    vk = tl.min(vals, axis=0)
    active = (n > K) & (lo < n) & (vk == vk) & (vk < float("inf"))
    cnt = tl.where(active, hi - lo, 0)
    zero = tl.sum(tl.zeros([BLOCK], dtype=tl.int32), axis=0)
    ngt = zero
    neq = zero
    for off in range(0, cnt, BLOCK):
        c = lo + off + tl.arange(0, BLOCK)
        x = tl.load(base + c, mask=c < hi, other=float("nan"))
        ngt += tl.sum((x > vk).to(tl.int32), axis=0)
        neq += tl.sum((x == vk).to(tl.int32), axis=0)
    tl.store(Part + (row * SPLITS + sp) * 2, ngt)
    tl.store(Part + (row * SPLITS + sp) * 2 + 1, neq)


@triton.jit(do_not_specialize=["l_stride", "t_stride"])
def _dsv41_ties_finish_kernel(
    Logits,
    l_stride,
    Ends,
    Topk,
    t_stride,
    Part,
    Vk,
    Ngt,
    Neq,
    Fixed,
    SPLITS: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    DO_FIX: tl.constexpr,
    STATS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    n = tl.load(Ends + row).to(tl.int64)
    base = Logits + row * l_stride
    trow = Topk + row * t_stride
    sel = tl.load(trow + tl.arange(0, K))
    vals = tl.load(base + sel.to(tl.int64), mask=(sel >= 0) & (n > K), other=float("inf"))
    vk = tl.min(vals, axis=0)
    active = (n > K) & (vk == vk) & (vk < float("inf"))
    sp = tl.arange(0, SPLITS).to(tl.int64)
    ngt = tl.sum(tl.load(Part + (row * SPLITS + sp) * 2), axis=0)
    neq = tl.sum(tl.load(Part + (row * SPLITS + sp) * 2 + 1), axis=0)
    tie = active & ((ngt + neq) > K)
    if DO_FIX:
        zero = tl.sum(tl.zeros([BLOCK], dtype=tl.int32), axis=0)
        n_fix = tl.where(tie, n, 0)
        need = K - ngt
        pos = zero
        seen = zero
        for off in range(0, n_fix, BLOCK):
            c = off + tl.arange(0, BLOCK)
            x = tl.load(base + c, mask=c < n, other=float("nan"))
            eq = (x == vk).to(tl.int32)
            take = (x > vk) | ((eq != 0) & ((seen + tl.cumsum(eq, axis=0)) <= need))
            ti = take.to(tl.int32)
            dst = pos + tl.cumsum(ti, axis=0) - 1
            tl.store(trow + dst, c.to(tl.int32), mask=take)
            pos += tl.sum(ti, axis=0)
            seen += tl.sum(eq, axis=0)
    if STATS:
        tl.store(Vk + row, vk)
        tl.store(Ngt + row, tl.where(active, ngt, -1))
        tl.store(Neq + row, tl.where(active, neq, -1))
        tl.store(Fixed + row, tie.to(tl.int32))


SPLIT_MIN_WIDTH = 16384  # decode rows wider than this use the split count
SPLIT_CHUNK = 8192
SPLIT_MAX = 64


def _splits_for(width: int) -> int:
    s = 1
    while s < SPLIT_MAX and s * SPLIT_CHUNK < width:
        s *= 2
    return s


_DUMMY: dict = {}
_WARNED: set = set()


def _dummy(device, dtype):
    key = (device, dtype)
    t = _DUMMY.get(key)
    if t is None:
        t = _DUMMY[key] = torch.zeros(1, dtype=dtype, device=device)
    return t


def _supported(logits, starts, ends, topk) -> str | None:
    rows = topk.shape[0]
    if logits.dtype != torch.float32 or logits.dim() != 2 or logits.stride(1) != 1:
        return "logits %s %s" % (logits.dtype, tuple(logits.stride()))
    if topk.dtype != torch.int32 or topk.dim() != 2 or topk.stride(1) != 1 or topk.shape[1] != K_SUPPORTED:
        return "topk %s %s %s" % (topk.dtype, tuple(topk.shape), tuple(topk.stride()))
    if ends.dtype != torch.int32 or ends.shape[0] < rows or logits.shape[0] < rows:
        return "ends %s %s rows %d logits %s" % (ends.dtype, tuple(ends.shape), rows, tuple(logits.shape))
    if starts is not None and (starts.dtype != torch.int32 or starts.shape[0] < rows):
        return "starts %s %s" % (starts.dtype, tuple(starts.shape))
    if not (logits.is_cuda and topk.is_cuda and ends.is_cuda):
        return "device"
    return None


def run_ties(
    logits: torch.Tensor,
    starts: torch.Tensor | None,
    ends: torch.Tensor,
    topk: torch.Tensor,
    local: bool,
    do_fix: bool,
    stats: bool,
):
    """In place on `topk` (rows of `logits`). Returns (vk, ngt, neq, fixed) when stats.
    `starts` None = every row starts at column 0."""
    rows = topk.shape[0]
    if rows == 0:
        return None
    why = _supported(logits, starts, ends, topk)
    if why is not None:
        if why not in _WARNED:
            _WARNED.add(why)
            logger.warning("DSV41_TOPK_CANON_TIES: unsupported layout (%s); stock rows kept", why)
        return None
    dev = logits.device
    if stats:
        vk = torch.empty(rows, dtype=torch.float32, device=dev)
        ngt = torch.empty(rows, dtype=torch.int32, device=dev)
        neq = torch.empty(rows, dtype=torch.int32, device=dev)
        fixed = torch.empty(rows, dtype=torch.int32, device=dev)
    else:
        vk = _dummy(dev, torch.float32)
        ngt = neq = fixed = _dummy(dev, torch.int32)
    width = logits.shape[1]
    if starts is None and not local and width > SPLIT_MIN_WIDTH:
        splits = _splits_for(width)
        chunk = -(-width // splits)
        chunk = -(-chunk // _BLOCK) * _BLOCK
        part = torch.empty(rows * splits * 2, dtype=torch.int32, device=dev)
        _dsv41_ties_partial_kernel[(rows, splits)](
            logits, logits.stride(0), ends, topk, topk.stride(0), part, chunk,
            SPLITS=splits, K=K_SUPPORTED, BLOCK=_BLOCK, num_warps=4,
        )
        _dsv41_ties_finish_kernel[(rows,)](
            logits, logits.stride(0), ends, topk, topk.stride(0), part, vk, ngt, neq, fixed,
            SPLITS=splits, K=K_SUPPORTED, BLOCK=_BLOCK, DO_FIX=do_fix, STATS=stats, num_warps=4,
        )
        return (vk, ngt, neq, fixed) if stats else None
    _dsv41_topk_ties_kernel[(rows,)](
        logits,
        logits.stride(0),
        starts if starts is not None else ends,
        ends,
        topk,
        topk.stride(0),
        vk,
        ngt,
        neq,
        fixed,
        K=K_SUPPORTED,
        BLOCK=_BLOCK,
        LOCAL=local,
        HAS_STARTS=starts is not None,
        DO_FIX=do_fix,
        STATS=stats,
        num_warps=4,
    )
    return (vk, ngt, neq, fixed) if stats else None


def decode_row_ends(seq_lens: torch.Tensor, next_n: int, num_rows: int) -> torch.Tensor:
    if seq_lens.numel() == num_rows:
        e = seq_lens.reshape(-1)
        return e if e.dtype == torch.int32 else e.to(torch.int32)
    off = torch.arange(next_n, dtype=torch.int32, device=seq_lens.device)
    return (seq_lens.reshape(-1, 1).to(torch.int32) - next_n + 1 + off).clamp_(min=0).reshape(-1)[:num_rows]


# ----------------------------------------------------------------------------------------
# Debug dump (DSV41_DET_DUMP)
# ----------------------------------------------------------------------------------------
_STATE: dict = {"seq": 0, "fwd": 0, "last_fwd_key": None, "call": 0}
_CW: dict = {}


def _rank() -> int:
    try:
        from vllm.distributed import get_tensor_model_parallel_rank

        return get_tensor_model_parallel_rank()
    except Exception:  # noqa: BLE001
        return 0


def _dump_on() -> bool:
    if not DUMP or torch.cuda.is_current_stream_capturing():
        return False
    if not DUMP_ALL_RANKS and _rank() != 0:
        return False
    return os.path.exists(os.path.join(DUMP_DIR, "ON"))


def _fix_now(kind: str) -> bool:
    if DUMP:
        try:
            with open(os.path.join(DUMP_DIR, "TIES")) as f:
                m = _norm_mode(f.read())
            if m in _MODES:
                return m in ("prefill", "all") if kind == "prefill" else m == "all"
        except OSError:
            pass
    return TIES_PREFILL if kind == "prefill" else TIES_DECODE


def _col_weights(n: int, device, salt: int) -> torch.Tensor:
    key = (device, salt)
    w = _CW.get(key)
    if w is None or w.numel() < n:
        g = torch.Generator(device="cpu").manual_seed(1234 + salt)
        w = (torch.rand(max(n, 1 << 16), generator=g, dtype=torch.float64) + 0.5).to(device)
        _CW[key] = w
    return w[:n]


def _row_fp(x, salt: int):
    """Per-row float64 fingerprint of the raw bytes of x (rows = dim 0)."""
    if not isinstance(x, torch.Tensor):
        return x
    r = x.shape[0]
    if r == 0:
        return torch.zeros(0, dtype=torch.float64)
    out = []
    for a in range(0, r, 2048):
        b = x[a : a + 2048].contiguous()
        b = b.view(b.shape[0], -1).view(torch.uint8).to(torch.float64)
        out.append((b * _col_weights(b.shape[1], x.device, salt)).sum(1).cpu())
    return torch.cat(out)


def _logits_fp(logits, starts, ends, rows) -> torch.Tensor:
    out = []
    n = min(logits.shape[1], int(ends[:rows].max().item()) if rows else 0)
    w = _col_weights(max(n, 1), logits.device, 7)
    cols = torch.arange(max(n, 1), device=logits.device)
    for a in range(0, rows, 256):
        b = min(rows, a + 256)
        blk = logits[a:b, : max(n, 1)]
        lo = starts[a:b, None].long() if starts is not None else 0
        valid = (cols[None, :] >= lo) & (cols[None, :] < ends[a:b, None].long())
        bits = blk.view(torch.int32).to(torch.float64)
        out.append((torch.where(valid, bits, 0.0) * w).sum(1).cpu())
    return torch.cat(out) if out else torch.zeros(0, dtype=torch.float64)


def _hook(kind, logits, starts, ends, topk, local, q, w, meta, fwd_key):
    do_fix = _fix_now(kind) if DUMP else (TIES_PREFILL if kind == "prefill" else TIES_DECODE)
    if not _dump_on():
        if do_fix:
            run_ties(logits, starts, ends, topk, local, True, False)
        return
    rows = topk.shape[0]
    if _STATE["last_fwd_key"] is not fwd_key:
        _STATE["last_fwd_key"] = fwd_key
        _STATE["fwd"] += 1
        _STATE["call"] = 0
    _STATE["call"] += 1
    raw = topk.clone()
    st = run_ties(logits, starts, ends, topk, local, do_fix, True)
    rec = {
        "kind": kind,
        "fwd": _STATE["fwd"],
        "call": _STATE["call"],
        "rows": rows,
        "width": logits.shape[1],
        "starts": starts[:rows].cpu() if starts is not None else None,
        "ends": ends[:rows].cpu(),
        "raw": raw.cpu(),
        "out": topk.clone().cpu(),
        "do_fix": do_fix,
        "lfp": _logits_fp(logits, starts, ends, rows),
        "qfp": _row_fp(q[:rows] if q is not None else None, 1),
        "wfp": _row_fp(w[:rows] if w is not None else None, 2),
    }
    if st is not None:
        rec.update(zip(("vk", "ngt", "neq", "fixed"), (t.cpu() for t in st)))
    for k, v in meta.items():
        rec[k] = _row_fp(v, 3 + len(k))
    _STATE["seq"] += 1
    path = os.path.join(DUMP_DIR, "r%d_%06d_%s.pt" % (_rank(), _STATE["seq"], kind))
    torch.save(rec, path)


def prefill_hook(logits, cu_seqlen_ks, cu_seqlen_ke, topk, q, w, meta, fwd_key) -> None:
    """After ops.top_k_per_row_prefill (indices relative to cu_seqlen_ks), before CANON."""
    if topk.shape[0] == 0 or logits.shape[1] == 0:
        return
    if not (TIES_PREFILL or DUMP):
        return
    _hook("prefill", logits, cu_seqlen_ks, cu_seqlen_ke, topk, True, q, w, meta, fwd_key)


def decode_hook(logits, seq_lens, next_n, topk, q, w, meta, fwd_key) -> None:
    """After the decode top-k (absolute column indices), before CANON."""
    if topk.shape[0] == 0 or logits.shape[1] == 0:
        return
    if not (TIES_DECODE or DUMP):
        return
    rows = topk.shape[0]
    ends = decode_row_ends(seq_lens, next_n, rows)
    if ends.shape[0] < rows:
        return
    _hook("decode", logits, None, ends, topk, False, q, w, meta, fwd_key)
