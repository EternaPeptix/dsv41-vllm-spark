# SPDX-License-Identifier: Apache-2.0
"""g130 boundary folds, both default off. dspark.py and dspark/speculator.py import this file only when a flag is set.

DSV41_CTXKV_FOLD=0|1|split|audit    (queue rank 11, BS06 part a)
    DSparkDeepseekV4Model.precompute_and_store_context_kv runs fused_wqa_wkv three times and keeps the last 512 of 1792
    output columns each time. 1 = one quant and one N=1536 GEMM over a concatenated wkv-only weight copy (8.1 MB per rank),
    with a pinned CUTLASS tactic, because N=1536 has no autotune entry and a miss means tactic 0.
    split = one quant and three N=512 GEMMs on views of the live weights (no copy). audit = serve the stock GEMMs, also run
    the fold, count differing bf16 words on the device.
    Rows outside DSV41_CTXKV_TACTIC_MAP, dummy/profile runs (no slot mappings) and autotune passes stay stock.
    The output columns start on a 128-column tile boundary in both shapes, which is why the same bits are expected. That is
    reasoning only: the build-time self-check and tests/test_boundary_gpu.py decide.

DSV41_MK_FOLD=0|1|audit             (queue rank 12, Markov half)
    Greedy sharded Markov loop, dspark/speculator.py _sample_sequential_sharded. Kept as they are: the first embed, Kernel2
    (the bias), the torch bf16 add (so rounding and flush behaviour cannot change) and the 16-byte all-gather.
    Replaced: max, 2 casts, zeros, id add, cast, stack -> one or two integer-only kernels that write the same [M, 4] pair.
    argmax, gather, cast, draft_tokens store, embed select -> one integer-only kernel.
    audit = serve the stock chain, also run the fold on the same inputs, count differing words on the device.
    The fold issues the same all-gather with the same shape on every position, so a rank that falls back to stock does not
    desynchronise the collectives.

DSV41_MK_FOLD_TILES=16|1  16 = 16 CTAs per row plus a combine kernel, 1 = one CTA per row. The GPU test times both.
DSV41_BOUNDARY_AUDIT_EVERY=256   greedy steps between counter reads in audit mode (a synchronize; audit arms only).
DSV41_BOUNDARY_AUDIT_DIR         default ~/dsv41-logs; audit counters go to boundary-audit.<host>.<pid>.json
"""
from __future__ import annotations

import json
import logging
import os
import socket
import time
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

CTX_MODE = os.environ.get("DSV41_CTXKV_FOLD", "0").strip().lower()
if CTX_MODE in ("", "off"):
    CTX_MODE = "0"
if CTX_MODE not in ("0", "1", "split", "audit"):
    raise ValueError(f"DSV41_CTXKV_FOLD must be 0|1|split|audit, got {CTX_MODE!r}")
MK_MODE = os.environ.get("DSV41_MK_FOLD", "0").strip().lower()
if MK_MODE in ("", "off"):
    MK_MODE = "0"
if MK_MODE not in ("0", "1", "audit"):
    raise ValueError(f"DSV41_MK_FOLD must be 0|1|audit, got {MK_MODE!r}")
MK_TILES = int(os.environ.get("DSV41_MK_FOLD_TILES", "16"))
if MK_TILES not in (1, 16):
    raise ValueError("DSV41_MK_FOLD_TILES must be 1 or 16")
CTX_MAP_RAW = os.environ.get("DSV41_CTXKV_TACTIC_MAP", "1-64:1,65-128:2,129-192:4")
AUDIT_EVERY = int(os.environ.get("DSV41_BOUNDARY_AUDIT_EVERY", "256"))
AUDIT_DIR = os.environ.get("DSV41_BOUNDARY_AUDIT_DIR", "")
MK_TILE_BLOCK = 1024
MK_LOCAL_BLOCK = 4096
TILE_ROWS = 128
# Rows the build-time self-check compares. Rows in the map but not here stay stock.
CTX_CHECK_MS = sorted({1, 2, 4, 8, 16, 32, 64, 128} | {6 * r for r in (1, 2, 3, 4, 5, 7, 8, 10, 12, 13, 15, 16, 18, 20, 21, 23, 24,
                                                                      26, 28, 29, 31, 32)})
MK_CHECK_MS = (1, 2, 3, 4, 8, 16, 32)


def _capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _tuning() -> bool:
    """True inside flashinfer.autotune(True): the stock calls must reach the tuner, so everything stays stock."""
    try:
        from flashinfer.autotuner import AutoTuner

        return bool(AutoTuner.get().is_tuning_mode)
    except Exception:  # noqa: BLE001 - no flashinfer in the CPU test
        return False


def parse_map(raw: str) -> dict[int, int]:
    out: dict[int, int] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        rng, tac = part.split(":")
        lo, _, hi = rng.partition("-")
        for m in range(int(lo), int(hi or lo) + 1):
            if m in out:
                raise ValueError(f"DSV41_CTXKV_TACTIC_MAP: M={m} listed twice")
            out[m] = int(tac)
    return out


# =====================================================================================================
# Audit counters: device tensors bumped inside the captured graphs, read here between steps
# =====================================================================================================
_AUDIT: dict = {"ctr": None, "last": 0, "warned": False}
# indices: 0 ctx calls, 1 ctx words, 2 ctx mismatches, 3 mk positions, 4 mk rows, 5 mk pair mismatches,
#          6 mk token mismatches, 7 mk embed mismatches
_AUDIT_NAMES = ("ctx_calls", "ctx_words", "ctx_mismatching_words", "mk_positions", "mk_rows", "mk_pair_mismatching_words",
                "mk_token_mismatches", "mk_embed_mismatching_words")


def _audit_ctr(device) -> torch.Tensor | None:
    c = _AUDIT["ctr"]
    if c is None and not _capturing():
        c = _AUDIT["ctr"] = torch.zeros((len(_AUDIT_NAMES),), dtype=torch.int64, device=device)
    return c


def _bump(c: torch.Tensor, first: int, *vals) -> None:
    """In-place on the device, also inside a captured graph."""
    for i, v in enumerate(vals):
        if isinstance(v, torch.Tensor):
            c[first + i : first + i + 1].add_(v.to(torch.int64).reshape(1))
        else:
            c[first + i : first + i + 1].add_(int(v))


def audit_tick(step: int) -> None:
    """Called from DSparkSpeculator.propose in audit arms only. Never raises."""
    if CTX_MODE != "audit" and MK_MODE != "audit":
        return
    if AUDIT_EVERY <= 0 or step % AUDIT_EVERY != 0:
        return
    c = _AUDIT["ctr"]
    if c is None:
        return
    try:
        vals = [int(v) for v in c.tolist()]
        rep = dict(zip(_AUDIT_NAMES, vals))
        bad = vals[2] + vals[5] + vals[6] + vals[7]
        rep.update(host=socket.gethostname(), pid=os.getpid(), time=time.time(), step=step,
                   ctx_mode=CTX_MODE, mk_mode=MK_MODE, status="MISMATCH" if bad else "OK")
        root = Path(AUDIT_DIR) if AUDIT_DIR else Path.home() / "dsv41-logs"
        out = root / f"boundary-audit.{socket.gethostname()}.{os.getpid()}.json"
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(rep, indent=1))
        tmp.replace(out)
        if bad:
            logger.warning("DSV41 boundary audit MISMATCH: %s", rep)
    except Exception as e:  # noqa: BLE001 - diagnostics must never kill serving
        if not _AUDIT["warned"]:
            _AUDIT["warned"] = True
            logger.warning("DSV41 boundary audit report failed: %r", e)


# =====================================================================================================
# Part a: context KV
# =====================================================================================================
class CtxKvFold:
    """Weights for the folded projection.

    The swizzled F8_128x4 scale keeps each 128-row tile contiguous and outermost (swizzle_mxfp8_scale: (m_tiles, 4, 32,
    k_tiles, 4) -> transpose(1, 3)), so a tile-aligned row range is a plain 1-D slice. tests/test_boundary_cpu.py checks this
    against the real function.
    """

    def __init__(self, layers, mode: str, runner=None, workspace=None):
        lins = [ly.attn.fused_wqa_wkv for ly in layers]
        q = {int(ly.attn.q_lora_rank) for ly in layers}
        if len(q) != 1:
            raise ValueError(f"q_lora_rank differs across draft layers: {q}")
        q = q.pop()
        shapes = {tuple(lin.weight.shape) for lin in lins}
        if len(shapes) != 1:
            raise ValueError(f"fused_wqa_wkv shapes differ: {shapes}")
        N, K = shapes.pop()
        H = N - q
        if q % TILE_ROWS or H % TILE_ROWS or H <= 0:
            raise ValueError(f"wkv is not tile aligned: q {q} H {H}")
        self.K, self.H, self.q, self.N = K, H, q, N
        self.n_layers = len(lins)
        m_tiles = N // TILE_ROWS
        self.parts = []
        wkv, scales = [], []
        for lin in lins:
            s = lin.weight_scale.data
            w = lin.weight.data
            if s.dim() != 1 or s.numel() % m_tiles or not w.is_contiguous():
                raise ValueError(f"unexpected layout: scale {tuple(s.shape)} weight contiguous {w.is_contiguous()}")
            if getattr(lin, "bias", None) is not None or getattr(lin, "gather_output", False) and lin.tp_size > 1:
                raise ValueError("fused_wqa_wkv has a bias or a gathered output")
            per_tile = s.numel() // m_tiles
            wkv.append(w[q:])
            scales.append(s[(q // TILE_ROWS) * per_tile:])
        if mode == "split":
            for w, s in zip(wkv, scales):
                self.parts.append((w.t(), s, H))
        else:
            w = torch.cat(wkv, dim=0).contiguous()
            s = torch.cat(scales, dim=0).contiguous()
            self.parts.append((w.t(), s, H * self.n_layers))
        self.mode = mode
        dev = lins[0].weight.device
        if runner is None:
            from flashinfer.gemm.gemm_base import DEFAULT_WORKSPACE_SIZE, get_cutlass_mxfp8_gemm_module
            from flashinfer.utils import _get_cache_buf

            major, _ = torch.cuda.get_device_capability(dev)
            runner = get_cutlass_mxfp8_gemm_module(major).cutlass_mxfp8_gemm_runner()
            workspace = _get_cache_buf("mm_mxfp8_workspace", DEFAULT_WORKSPACE_SIZE, dev)
        self.runner, self.ws = runner, workspace
        self.ok: dict[int, int] = {}
        self.rejected: dict[int, str] = {}

    def project(self, xq: torch.Tensor, xs: torch.Tensor, tactic: int) -> list[torch.Tensor]:
        """Pre-norm KV per layer, same M x H bf16 words as qr_kv[..., q:] of the stock GEMM."""
        M = xq.shape[0]
        outs = []
        for w_t, s, n in self.parts:
            out = torch.empty((M, n), dtype=torch.bfloat16, device=xq.device)
            self.runner(inputs=[xq, w_t, xs, s, torch.bfloat16, out, self.ws], tactic=tactic)
            outs.append(out)
        if self.mode == "split":
            return outs
        o = outs[0]
        return [o[:, i * self.H : (i + 1) * self.H] for i in range(self.n_layers)]

    def self_check(self, layers, quantize, tactic_map: dict[int, int], dev) -> None:
        lins = [ly.attn.fused_wqa_wkv for ly in layers]
        for M in CTX_CHECK_MS:
            t = tactic_map.get(M)
            if t is None:
                continue
            why = ""
            for seed in (M, 7919 + M):
                x = _check_inputs(M, self.K, seed, dev)
                xq, xs = quantize(x)
                fold = self.project(xq, xs, t)
                for i, lin in enumerate(lins):
                    ref = lin(x)
                    ref = (ref[0] if isinstance(ref, tuple) else ref)[..., self.q :]
                    if not torch.equal(fold[i].view(torch.int16), ref.view(torch.int16)):
                        why = f"layer {i} differs (seed {seed}, tactic {t})"
                        break
                if why:
                    break
            if why:
                self.rejected[M] = why
                logger.warning("DSV41_CTXKV_FOLD: M=%d kept on the stock path: %s", M, why)
            else:
                self.ok[M] = t
        if torch.cuda.is_available():
            torch.cuda.synchronize(dev)


def _check_inputs(M: int, K: int, seed: int, dev) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(M, K, generator=g)
    blk = torch.exp2(torch.randint(-6, 7, (M, K // 32), generator=g).float())
    x = x * blk.repeat_interleave(32, dim=1)
    x[:, ::97] = 0.0
    x[torch.arange(M), torch.randint(0, K, (M,), generator=g)] = 300.0
    return x.to(torch.bfloat16).to(dev)


def _default_quantize(x: torch.Tensor):
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize

    return mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)


def ctxkv_state(model) -> CtxKvFold | None:
    """Built once, eagerly, never while capturing or tuning. Returns None when the stock path must run."""
    st = getattr(model, "_dsv41_ctxkv", None)
    if st is not None:
        return st or None
    if CTX_MODE == "0" or _capturing() or _tuning():
        return None
    try:
        f = CtxKvFold(model.layers, "split" if CTX_MODE == "split" else "merge")
        dev = model.layers[0].attn.fused_wqa_wkv.weight.device
        tm = parse_map(CTX_MAP_RAW)
        bad = sorted({t for t in tm.values() if t not in set(f.runner.get_valid_tactics(None, None))})
        if bad:
            raise ValueError(f"tactics {bad} not valid")
        f.self_check(model.layers, _default_quantize, tm, dev)
        if not f.ok:
            raise ValueError(f"no row count passed the self-check: {f.rejected}")
        if CTX_MODE == "audit":
            _audit_ctr(dev)  # eager, or a CTX-only audit arm would find no counters in capture and measure nothing
    except Exception as e:  # noqa: BLE001 - fail closed
        logger.warning("DSV41_CTXKV_FOLD disabled, stock context KV: %r", e)
        model._dsv41_ctxkv = False
        return None
    model._dsv41_ctxkv = f
    logger.warning("DSV41_CTXKV_FOLD=%s: rows pinned %s, kept stock %s, weight copy %.2f MB",
                   CTX_MODE, sorted(f.ok), sorted(f.rejected),
                   0.0 if f.mode == "split" else (f.parts[0][0].numel() + f.parts[0][1].numel()) / 1e6)
    return f


def ctxkv_precompute(model, main_x, context_positions, context_slot_mappings, insert, quantize=None) -> bool:
    """Replaces the loop in precompute_and_store_context_kv. False = nothing ran, the caller runs the stock loop."""
    if context_slot_mappings is None or not isinstance(main_x, torch.Tensor) or main_x.dim() != 2:
        return False
    if main_x.dtype != torch.bfloat16:
        return False
    f = ctxkv_state(model)
    if f is None:
        return False
    M = int(main_x.shape[0])
    tactic = f.ok.get(M)
    if tactic is None or _tuning():
        return False
    q = quantize or _default_quantize
    xq, xs = q(main_x)
    kvs = f.project(xq, xs, tactic)
    if CTX_MODE == "audit":
        ctr = _audit_ctr(main_x.device)
        if ctr is None:
            return False
        for i, layer in enumerate(model.layers):
            attn = layer.attn
            qr_kv, _ = attn.fused_wqa_wkv(main_x)
            ref = qr_kv[..., attn.q_lora_rank :]
            _bump(ctr, 1, ref.numel(), (ref.view(torch.int16) != kvs[i].view(torch.int16)).sum())
            kv = attn.kv_norm(ref)
            slot = context_slot_mappings[i]
            if slot is not None:
                insert(attn, kv, context_positions, slot)
        _bump(ctr, 0, 1)
        return True
    for i, layer in enumerate(model.layers):
        attn = layer.attn
        kv = attn.kv_norm(kvs[i])
        slot = context_slot_mappings[i]
        if slot is None:
            continue
        insert(attn, kv, context_positions, slot)
    return True


# =====================================================================================================
# Part b: Markov glue. Integer-only kernels: every comparison is on bits, no FP op, so FTZ and rounding modes do not enter.
# Order keys: NaN above everything (first NaN wins, as torch max/argmax do), then numeric order with -0 == +0.
# =====================================================================================================
_MKK: dict = {}


def _mk_kernels() -> dict:
    if _MKK:
        return _MKK
    import triton
    import triton.language as tl

    @triton.jit(do_not_specialize=["LG", "sl", "OUT", "V", "vstart"])
    def _mk_local_kernel(LG, sl, OUT, V, vstart, BLOCK: tl.constexpr):
        """One CTA per row. OUT[row] = (fp32 bits of the leftmost max, fp32 bits of vstart + its index, 0, 0)."""
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        u = tl.load(LG + row * sl + cols, mask=cols < V, other=0).to(tl.int32) & 0xFFFF
        mag = u & 0x7FFF
        key = tl.where(mag > 0x7F80, 0x7FFF, tl.where((u >> 15) != 0, -mag, mag))
        key = tl.where(cols < V, key, -0x8000)
        best_k, best_i = tl.max(key, axis=0, return_indices=True, return_indices_tie_break_left=True)
        for v0 in range(BLOCK, V, BLOCK):
            c = v0 + cols
            u = tl.load(LG + row * sl + c, mask=c < V, other=0).to(tl.int32) & 0xFFFF
            mag = u & 0x7FFF
            key = tl.where(mag > 0x7F80, 0x7FFF, tl.where((u >> 15) != 0, -mag, mag))
            key = tl.where(c < V, key, -0x8000)
            k, i = tl.max(key, axis=0, return_indices=True, return_indices_tie_break_left=True)
            upd = k > best_k
            best_i = tl.where(upd, v0 + i, best_i)
            best_k = tl.where(upd, k, best_k)
        bits = tl.load(LG + row * sl + best_i).to(tl.int32) & 0xFFFF
        base = OUT + row * 4
        tl.store(base, bits << 16)
        tl.store(base + 1, (vstart + best_i).to(tl.float32).to(tl.int32, bitcast=True))
        tl.store(base + 2, 0)
        tl.store(base + 3, 0)

    @triton.jit(do_not_specialize=["LG", "sl", "SC", "T", "V"])
    def _mk_tile_kernel(LG, sl, SC, T, V, BLOCK: tl.constexpr):
        """Grid (M, n_tiles). SC[row, tile] = (key, column of the leftmost max in this tile)."""
        row = tl.program_id(0)
        tile = tl.program_id(1)
        c = tile * BLOCK + tl.arange(0, BLOCK)
        u = tl.load(LG + row * sl + c, mask=c < V, other=0).to(tl.int32) & 0xFFFF
        mag = u & 0x7FFF
        key = tl.where(mag > 0x7F80, 0x7FFF, tl.where((u >> 15) != 0, -mag, mag))
        key = tl.where(c < V, key, -0x8000)
        k, i = tl.max(key, axis=0, return_indices=True, return_indices_tie_break_left=True)
        s = SC + (row * T + tile) * 2
        tl.store(s, k)
        tl.store(s + 1, tile * BLOCK + i)

    @triton.jit(do_not_specialize=["SC", "LG", "sl", "OUT", "NT", "vstart"])
    def _mk_combine_kernel(SC, LG, sl, OUT, NT, vstart, T: tl.constexpr):
        """Grid (M,). Earliest tile wins a tie, which with the tile kernel is the leftmost column of the row."""
        row = tl.program_id(0)
        t = tl.arange(0, T)
        keys = tl.load(SC + (row * T + t) * 2, mask=t < NT, other=-0x8000)
        _k, bt = tl.max(keys, axis=0, return_indices=True, return_indices_tie_break_left=True)
        col = tl.load(SC + (row * T + bt) * 2 + 1)
        bits = tl.load(LG + row * sl + col).to(tl.int32) & 0xFFFF
        base = OUT + row * 4
        tl.store(base, bits << 16)
        tl.store(base + 1, (vstart + col).to(tl.float32).to(tl.int32, bitcast=True))
        tl.store(base + 2, 0)
        tl.store(base + 3, 0)

    @triton.jit(do_not_specialize=["G", "M", "DT", "dt_row", "col", "W1", "ws", "E", "es"])
    def _mk_winner_kernel(G, M, DT, dt_row, col, W1, ws, E, es, TP: tl.constexpr, R: tl.constexpr, WITH_EMB: tl.constexpr):
        """Grid (M,). G = gathered [TP * M, 4] rows as int32, rank-major. Winner = first rank with the largest key
        (stock: gathered[:, :, 0].argmax(dim=0)). DT[row, col] = its id. With WITH_EMB, E[row, :R] = W1[id, :R]."""
        row = tl.program_id(0)
        u = tl.load(G + row * 4)
        mag = u & 0x7FFFFFFF
        best_k = tl.where(mag > 0x7F800000, 0x7FFFFFFF, tl.where(u < 0, -mag, mag))
        best_t = tl.load(G + row * 4 + 1)
        for r in tl.static_range(1, TP):
            u = tl.load(G + (r * M + row) * 4)
            mag = u & 0x7FFFFFFF
            k = tl.where(mag > 0x7F800000, 0x7FFFFFFF, tl.where(u < 0, -mag, mag))
            t = tl.load(G + (r * M + row) * 4 + 1)
            upd = k > best_k
            best_t = tl.where(upd, t, best_t)
            best_k = tl.where(upd, k, best_k)
        tok = best_t.to(tl.float32, bitcast=True).to(tl.int64)
        tl.store(DT + row * dt_row + col, tok)
        if WITH_EMB:
            cols = tl.arange(0, R)
            tl.store(E + row * es + cols, tl.load(W1 + tok * ws + cols))

    _MKK.update(triton=triton, local=_mk_local_kernel, tile=_mk_tile_kernel, combine=_mk_combine_kernel,
                winner=_mk_winner_kernel)
    return _MKK


def mk_local_pair(logits_i: torch.Tensor, vstart: int, tiles: int | None = None) -> torch.Tensor:
    """[M, V] bf16 -> [M, 4] fp32, same 16 bytes per row as
    vals, idx = logits_i.max(-1); stack([vals.float(), (idx + vstart).float(), 0, 0], -1)."""
    assert logits_i.dim() == 2 and logits_i.stride(1) == 1 and logits_i.dtype == torch.bfloat16
    k = _mk_kernels()
    M, V = logits_i.shape
    LG = logits_i.view(torch.int16)
    pair = torch.empty((M, 4), dtype=torch.float32, device=logits_i.device)
    P = pair.view(torch.int32)
    tiles = MK_TILES if tiles is None else tiles
    if tiles == 1:
        k["local"][(M,)](LG, LG.stride(0), P, V, vstart, BLOCK=MK_LOCAL_BLOCK, num_warps=8)
        return pair
    nt = k["triton"].cdiv(V, MK_TILE_BLOCK)
    T = 1
    while T < nt:
        T *= 2
    sc = torch.empty((M, T, 2), dtype=torch.int32, device=logits_i.device)
    k["tile"][(M, nt)](LG, LG.stride(0), sc, T, V, BLOCK=MK_TILE_BLOCK, num_warps=4)
    k["combine"][(M,)](sc, LG, LG.stride(0), P, nt, vstart, T=T, num_warps=1)
    return pair


def mk_winner(gathered: torch.Tensor, tp: int, M: int, draft_tokens: torch.Tensor, col: int, w1: torch.Tensor,
              with_embed: bool) -> torch.Tensor | None:
    """[tp * M, 4] gathered rows -> draft_tokens[:M, col] and, optionally, the next Markov embed [M, R] bf16."""
    assert gathered.shape == (tp * M, 4) and gathered.is_contiguous() and gathered.dtype == torch.float32
    assert draft_tokens.dtype == torch.int64 and draft_tokens.stride(1) == 1
    k = _mk_kernels()
    R = w1.shape[1]
    E = torch.empty((M, R), dtype=w1.dtype, device=w1.device)
    k["winner"][(M,)](gathered.view(torch.int32), M, draft_tokens, draft_tokens.stride(0), col,
                      w1.view(torch.int16), w1.stride(0), E.view(torch.int16), E.stride(0),
                      TP=tp, R=R, WITH_EMB=with_embed, num_warps=1)
    return E if with_embed else None


def _stock_pair(logits_i: torch.Tensor, vstart: int) -> torch.Tensor:
    vals, idx = logits_i.max(dim=-1)
    vals = vals.float()
    zero = torch.zeros_like(vals)
    return torch.stack([vals, (idx + vstart).float(), zero, zero], dim=-1)


def _mk_check_rows(M: int, V: int, seed: int, dev) -> torch.Tensor:
    """Tie-heavy rows on a coarse grid; a planted cross-tile tie, a +0/-0 tie and an all -inf row."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.randint(-40, 41, (M, V), generator=g).float() / 2.0).to(torch.bfloat16)
    for r in range(M):
        c = torch.randint(0, V, (3,), generator=g)
        x[r, c] = float(x[r].float().max()) + 1.0
        if r % 3 == 1:
            x[r] = -1.0
            x[r, int(c[0])] = -0.0
            x[r, min(int(c[1]) + V // 2, V - 1)] = 0.0
        elif r % 3 == 2 and r % 2 == 0:
            x[r] = float("-inf")
    return x.to(dev)


def mk_state(spec, st: dict):
    """Eligibility + bitwise self-check + kernel JIT. Eager only. No collective in here, so ranks need not agree."""
    cur = getattr(spec, "_dsv41_mk", None)
    if cur is not None:
        return cur or None
    if MK_MODE == "0" or _capturing():
        return None
    try:
        w1 = spec.model.model.markov_head.markov_w1.weight
        tp = int(st["tp"])
        if w1.dtype != torch.bfloat16 or not w1.is_contiguous() or w1.shape[1] & (w1.shape[1] - 1):
            raise ValueError(f"markov_w1 {w1.dtype} {tuple(w1.shape)}")
        if spec.draft_tokens.dtype != torch.int64 or spec.draft_tokens.stride(1) != 1:
            raise ValueError("draft_tokens layout")
        V = int(st["shim"].weight.shape[0])
        vstart = int(st["vocab_start"])
        dev = w1.device
        for M in MK_CHECK_MS:
            rows = [_mk_check_rows(M, V, 1000 * M + r, dev) for r in range(tp)]
            pairs = [_stock_pair(x, r * V) for r, x in enumerate(rows)]
            for r, x in enumerate(rows):
                for tiles in (MK_TILES,):
                    if not torch.equal(mk_local_pair(x, r * V, tiles).view(torch.int32), pairs[r].view(torch.int32)):
                        raise ValueError(f"local pair differs at M={M} rank {r} tiles {tiles}")
            g = torch.cat(pairs, dim=0)
            gv = g.view(tp, M, 4)
            best = gv[:, :, 0].argmax(dim=0, keepdim=True)
            tok = gv[:, :, 1].gather(0, best).squeeze(0).to(torch.int64)
            dt = torch.zeros((M, 5), dtype=torch.int64, device=dev)
            e = mk_winner(g, tp, M, dt, 2, w1.data, True)
            mk_winner(g, tp, M, dt, 3, w1.data, False)
            if not torch.equal(dt[:, 2], tok) or not torch.equal(dt[:, 3], tok):
                raise ValueError(f"winner differs at M={M}")
            if not torch.equal(e.view(torch.int16), w1.data[tok].view(torch.int16)):
                raise ValueError(f"embed differs at M={M}")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if MK_MODE == "audit":
            _audit_ctr(dev)
        s = dict(w1=w1.data, tp=tp, vstart=vstart)
    except Exception as e:  # noqa: BLE001 - fail closed; same collectives either way
        logger.warning("DSV41_MK_FOLD disabled, stock Markov loop: %r", e)
        spec._dsv41_mk = False
        return None
    spec._dsv41_mk = s
    logger.warning("DSV41_MK_FOLD=%s tiles %d: self-check passed at M %s, V %d, tp %d", MK_MODE, MK_TILES, MK_CHECK_MS, V, tp)
    return s


def markov_fold_loop(spec, st, lp, base, prev, num_reqs, n_spec, confidence_markov_embeds, all_gather) -> bool:
    """Replaces the n_spec loop of _sample_sequential_sharded. False = nothing ran, the caller runs the stock loop."""
    s = mk_state(spec, st)
    if s is None or base.dtype != torch.bfloat16:
        return False
    tp, vstart, w1 = s["tp"], s["vstart"], s["w1"]
    use_conf = spec.use_confidence_head
    if MK_MODE == "audit":
        ctr = _audit_ctr(base.device)
        if ctr is None:
            return False
        shadow = torch.empty((num_reqs, n_spec), dtype=torch.int64, device=base.device)
        pending = None
        for i in range(n_spec):
            markov_embed = spec.model.markov_embed(prev)
            if pending is not None:
                _bump(ctr, 7, (pending.view(torch.int16) != markov_embed.view(torch.int16)).sum())
            if use_conf:
                confidence_markov_embeds.append(markov_embed)
            bias = lp(st["shim"], markov_embed, skip_gather=True)
            logits_i = base[:, i] + bias
            pair = _stock_pair(logits_i, vstart)
            gathered = all_gather(pair, dim=0)
            g4 = gathered.view(tp, num_reqs, 4)
            best = g4[:, :, 0].argmax(dim=0, keepdim=True)
            tok = g4[:, :, 1].gather(0, best).squeeze(0).to(torch.int64)
            spec.draft_tokens[:num_reqs, i] = tok
            prev = tok
            # shadow, same inputs, stock already served
            p2 = mk_local_pair(logits_i, vstart)
            pending = mk_winner(gathered.contiguous(), tp, num_reqs, shadow, i, w1, i + 1 < n_spec)
            _bump(ctr, 3, 1, num_reqs, (p2.view(torch.int32) != pair.view(torch.int32)).sum(),
                  (shadow[:, i] != tok).sum())
        return True
    markov_embed = spec.model.markov_embed(prev)
    for i in range(n_spec):
        if use_conf:
            confidence_markov_embeds.append(markov_embed)
        bias = lp(st["shim"], markov_embed, skip_gather=True)
        logits_i = base[:, i] + bias
        pair = mk_local_pair(logits_i, vstart)
        gathered = all_gather(pair, dim=0)
        markov_embed = mk_winner(gathered.contiguous(), tp, num_reqs, spec.draft_tokens, i, w1, i + 1 < n_spec)
    return True


def prebuild(spec) -> None:
    """DSparkSpeculator.capture: build both states eagerly, before any graph records. No collective."""
    try:
        if CTX_MODE != "0":
            ctxkv_state(spec.model.model)
        if MK_MODE != "0":
            st = getattr(spec, "_sharded_markov_greedy", None) or getattr(spec, "_sharded_markov", None)
            if st is not None:
                mk_state(spec, st)
    except Exception as e:  # noqa: BLE001 - both states fail closed on their own; belt and braces
        logger.warning("DSV41 boundary prebuild failed, stock paths: %r", e)
