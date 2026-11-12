"""DSV41_DENSE_TACTIC (g130 push-A, 2026-11-06): pinned CUTLASS SM120 MXFP8 tactics for the dense GEMMs.

Every dense MXFP8 GEMM of the DSv4.1 stack (wq_b, indexer wq_b, wo_b, shared gate_up/down, draft main_proj,
ctxkv fold, draft-side, FP8 draft head) goes through vllm.utils.flashinfer.mm_mxfp8 -> flashinfer mm_mxfp8
(backend="cutlass") -> AutoTuner.choose_one -> runner(inputs=[A, B, A_scale, B_scale, out_dtype, out, ws],
tactic=t). The serving autotune cache picks tactic 1 (128x32 tile, swap_ab) for nearly every decode shape, which
at M<=6 runs as few as 5 CTAs on the 48-SM GB10. This module makes the SAME runner call with a pinned tactic for
the (K, N, M) listed in DSV41_DENSE_TACTIC_MAP.

Exactness: the six SM120 tactics differ only in CTA tile (128x{32,64,128}) and operand swap; the MMA atom, the
fp32 accumulator and the increasing-k visit order are the same (no split-K / stream-K on SM120). That is the
argument; it is not trusted on its own. A (K, N, M) is pinned only after a self-check, run once per (K, N) at its
first EAGER call on the layer's real weight: for each mapped M and 2 synthetic activation sets (block-scale spread
2^-6..2^6 with a 300.0 outlier; RMSNorm-like rows with large channels), all six tactics AND the stock mm_mxfp8 call
(the serving autotuner choice) must give bitwise-identical bf16 outputs. Anything else keeps the stock path. A
(K, N) whose first call happens during CUDA-graph capture keeps the stock path (no host sync is allowed there).

  DSV41_DENSE_TACTIC=0|1          0 (default): never imported (vllm/utils/flashinfer.py checks the env once)
  DSV41_DENSE_TACTIC_MAP="K,N:lo-hi:t[,lo-hi:t];K,N:..."   e.g. "5120,576:1-8:0;1024,5120:1-6:0"
Log: one WARNING per (K, N) with the pinned / rejected row counts.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger("vllm.dsv41_dense_tactic")

MXFP8_BLOCK = 32
_runner = None
_ws = None
_tactics: list[int] = []
_checked: set[tuple[int, int]] = set()
_ok: dict[tuple[int, int], dict[int, int]] = {}
_warned: set = set()


def parse_map(raw: str) -> dict[tuple[int, int], dict[int, int]]:
    out: dict[tuple[int, int], dict[int, int]] = {}
    for shape_part in (p.strip() for p in raw.split(";") if p.strip()):
        shape, _, spec = shape_part.partition(":")
        K, N = (int(v) for v in shape.split(","))
        dst = out.setdefault((K, N), {})
        for item in (s.strip() for s in spec.split(",") if s.strip()):
            rng, _, t = item.rpartition(":")
            lo, _, hi = rng.partition("-")
            lo_i, hi_i, t_i = int(lo), int(hi or lo), int(t)
            if not (1 <= lo_i <= hi_i <= 4096) or not (0 <= t_i <= 63):
                raise ValueError(f"DSV41_DENSE_TACTIC_MAP entry {item!r} for {K},{N} out of range")
            for m in range(lo_i, hi_i + 1):
                if m in dst:
                    raise ValueError(f"DSV41_DENSE_TACTIC_MAP: {K},{N} M={m} listed twice")
                dst[m] = t_i
    return out


MAP = parse_map(os.environ.get("DSV41_DENSE_TACTIC_MAP", ""))


def _capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _init(dev) -> None:
    global _runner, _ws, _tactics
    if _runner is None:
        from flashinfer.gemm.gemm_base import DEFAULT_WORKSPACE_SIZE, get_cutlass_mxfp8_gemm_module
        from flashinfer.utils import _get_cache_buf

        major, _ = torch.cuda.get_device_capability(dev)
        _runner = get_cutlass_mxfp8_gemm_module(major).cutlass_mxfp8_gemm_runner()
        _tactics = list(_runner.get_valid_tactics(None, None))
        _ws = _get_cache_buf("mm_mxfp8_workspace", DEFAULT_WORKSPACE_SIZE, dev)
        bad = sorted({t for d in MAP.values() for t in d.values() if t not in _tactics})
        if bad:
            raise ValueError(f"DSV41_DENSE_TACTIC_MAP tactics {bad} not in valid tactics {_tactics}")


def _run(A, B, A_scale, B_scale, out_dtype, t: int) -> torch.Tensor:
    """Exactly flashinfer mm_mxfp8(backend='cutlass') minus the autotuner lookup."""
    out = torch.empty((A.shape[0], B.shape[1]), dtype=out_dtype, device=A.device)
    _runner(inputs=[A, B, A_scale, B_scale, out_dtype, out, _ws], tactic=t)
    return out


def _inputs(M: int, K: int, s: int, dev) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(1009 * M + K + 7 * s)
    x = torch.randn(M, K, generator=g)
    if s == 0:
        blk = torch.exp2(torch.randint(-6, 7, (M, K // MXFP8_BLOCK), generator=g).float())
        x = x * blk.repeat_interleave(MXFP8_BLOCK, dim=1)
        x[:, ::97] = 0.0
        x[torch.arange(M), torch.randint(0, K, (M,), generator=g)] = 300.0
    else:
        x = x * torch.exp(torch.randn(K, generator=g) * 0.5)
        x[:, :8] *= 40
    return x.to(torch.bfloat16).to(dev)


def _self_check(K: int, N: int, B, B_scale, out_dtype) -> None:
    from flashinfer import mm_mxfp8 as stock
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize

    dev = B.device
    ok, rejected = {}, {}
    for M, t in sorted(MAP[(K, N)].items()):
        why = ""
        for s in (0, 1):
            xq, xs = mxfp8_e4m3_quantize(_inputs(M, K, s, dev), is_sf_swizzled_layout=True)
            ref = stock(xq, B, xs, B_scale, out=None, out_dtype=out_dtype, backend="cutlass").view(torch.int16)
            for tt in _tactics:
                try:
                    same = torch.equal(_run(xq, B, xs, B_scale, out_dtype, tt).view(torch.int16), ref)
                except Exception as e:  # noqa: BLE001 - an unusable tactic disqualifies this M
                    same, why = False, f"tactic {tt} raised {e!r}"[:160]
                if not same:
                    why = why or f"tactic {tt} != stock (input {s})"
                    break
            if why:
                break
        if why:
            rejected[M] = why
        else:
            ok[M] = t
    torch.cuda.synchronize(dev)
    _ok[(K, N)] = ok
    logger.warning("DSV41_DENSE_TACTIC: K=%d N=%d pinned %s%s (all %d tactics + stock bitwise on 2 inputs, real weight)",
                   K, N, _fmt(ok), f"; kept stock {sorted(rejected)}: {next(iter(rejected.values()))}" if rejected else "",
                   len(_tactics))


def pinned(A, B, A_scale, B_scale, out_dtype, backend: str, use_8x4_sf_layout: bool):
    """Return the pinned-tactic output, or None to keep the stock path."""
    if backend != "cutlass" or use_8x4_sf_layout or out_dtype != torch.bfloat16 or A.dim() != 2 or B.dim() != 2:
        return None
    key = (A.shape[1], B.shape[1])
    if key not in MAP:
        return None
    # self-check at the shape's first EAGER call of any M (e.g. the startup profile run), so that the decode
    # row counts captured later find their verdict ready
    if key not in _checked:
        if _capturing():
            if key not in _warned:
                _warned.add(key)
                logger.warning("DSV41_DENSE_TACTIC: K=%d N=%d first seen during capture; stock path kept", *key)
            return None
        _init(A.device)
        _checked.add(key)
        _self_check(key[0], key[1], B, B_scale, out_dtype)
    t = _ok.get(key, {}).get(A.shape[0])
    if t is None:
        return None
    return _run(A, B, A_scale, B_scale, out_dtype, t)


def _fmt(d: dict[int, int]) -> str:
    out, run = [], None
    for m in sorted(d):
        if run and m == run[1] + 1 and d[m] == run[2]:
            run[1] = m
        else:
            if run:
                out.append(run)
            run = [m, m, d[m]]
    if run:
        out.append(run)
    return ",".join(f"{a}-{b}:{t}" if a != b else f"{a}:{t}" for a, b, t in out) or "-"
