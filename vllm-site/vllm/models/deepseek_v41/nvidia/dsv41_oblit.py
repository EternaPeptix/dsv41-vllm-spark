"""DSV41_OBLIT: exact runtime port of the OBLITERATUS refusal-direction ablation onto the official native weights.

OBLITERATUS/DeepSeek-V4.1-Flash-1017-obliterated (BF16, 1.5 TB) is deepseek-ai/DeepSeek-V4.1-Flash-1017 with, for
every target layer l in 2..39 and one unit direction d_l (verified by diffing tensors against the official
checkpoint: alpha ~0.999, all tensors of a layer share d_l at |cos| 1.0000):
    readers  W <- W (I - d d^T)   wq_a, wkv, indexer.weights_proj, compressor.wkv/wgate  (attention input)
                                  router gate, routed w1/w3, shared w1/w3                (FFN input)
    writers  W <- (I - d d^T) W   wo_b (attention output), routed w2, shared w2           (FFN output)
plus the output head as a reader with its own direction. Projection is linear, so the same model is obtained
exactly (up to one bf16 rounding of each projected activation, instead of their bf16 rounding of every edited
weight) by projecting d_l out of: the normalised attention input, the attention output, the normalised FFN input,
the FFN output (per expert row when the routed top-k reduction is deferred), and the final hidden state before
the head. The FP4 experts never have to hold the edit.

Env: DSV41_OBLIT=/path/obliteratus_dirs.npz with float32 [hidden] arrays named "L<layer>" and "head".
Unset or empty = OFF: the hooks are not taken and the model is bitwise the unpatched stack.
Only decoder layers named "...model.layers.<N>" with a direction are bound (layers 0-1 and the MTP/DSpark layers
40+ have none). Directions are created on the device at the first forward, which vLLM runs eagerly
(memory profiling / warmup) before any CUDA graph capture; creating one during capture raises.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import re

import numpy as np
import torch

logger = logging.getLogger("vllm.dsv41_oblit")
_DSV41_GLUE_PDL = __import__('os').environ.get('DSV41_GLUE_PDL', '0').strip().lower() not in ('', '0', 'off')  # r1-pdl
if _DSV41_GLUE_PDL:  # r1-pdl
    from vllm.triton_utils import dsv41_glue_pdl as _dsv41_glue_pdl  # r1-pdl
    _DSV41_GLUE_PDL = _dsv41_glue_pdl.enabled('proj')  # r1-pdl

PATH = os.path.expanduser(os.environ.get("DSV41_OBLIT", "").strip())
ON = bool(PATH)
_R7 = None  # r7-comm2
_R7_ON = os.environ.get("DSV41_RDMA_PROJ", "").strip().lower() in ("1", "on", "true", "yes")  # r7-comm2
if _R7_ON and _DSV41_GLUE_PDL:  # r7-comm2
    raise RuntimeError("DSV41_RDMA_PROJ=1 refused: DSV41_GLUE_PDL enables its 'proj' family, so the served projection "  # r7-comm2
                       "would be dsv41_glue_pdl.proj_rows (PDL), whose SASS record is unproven against f29c9ff9 "  # r7-comm2
                       "(r7-comm2 gate/projgate.py); unset one of them")  # r7-comm2
if _R7_ON and os.environ.get("DSV41_OBLIT_FOLD", "0").strip().lower() not in ("", "0", "off", "false", "no"):  # r7-comm2
    raise RuntimeError("DSV41_RDMA_PROJ=1 refused with DSV41_OBLIT_FOLD (r2-oblit): the folded layers never arm "  # r7-comm2
                       "their own direction, so no fold would be taken; unset one of them")  # r7-comm2
if _R7_ON:  # r7-comm2
    from vllm.distributed.device_communicators import rdma_all_reduce as _R7  # r7-comm2
_LAYER_RE = re.compile(r"(?:^|\.)model\.layers\.(\d+)$")

_host: dict[str, np.ndarray] | None = None   # name -> float32 unit vector (CPU)
_dev: dict[tuple[str, str], torch.Tensor] = {}  # (name, device) -> float32 unit vector on device
_bound: set[int] = set()


def _load() -> dict[str, np.ndarray]:
    global _host
    if _host is None:
        z = np.load(PATH)
        out = {}
        for k in z.files:
            v = np.asarray(z[k], dtype=np.float64).reshape(-1)
            n = float(np.linalg.norm(v))
            if not np.isfinite(n) or n < 1e-6:
                raise ValueError(f"DSV41_OBLIT {PATH}: direction {k!r} has norm {n}")
            out[k] = (v / n).astype(np.float32)
        hidden = {v.shape[0] for v in out.values()}
        if len(hidden) != 1:
            raise ValueError(f"DSV41_OBLIT {PATH}: mixed sizes {sorted(hidden)}")
        if "head" not in out:
            raise ValueError(f"DSV41_OBLIT {PATH}: no 'head' direction")
        _host = out
        logger.info("DSV41_OBLIT: %d directions from %s (layers %s + head)", len(out), PATH,
                    sorted(int(k[1:]) for k in out if k[1:].isdigit()))
    return _host


def layer_for(prefix: str) -> int:
    """Layer index to project for a decoder layer with this module prefix, or -1 (OFF / no direction)."""
    if not ON:
        return -1
    m = _LAYER_RE.search(str(prefix))
    if m is None:
        return -1
    idx = int(m.group(1))
    if f"L{idx}" not in _load():
        return -1
    _bound.add(idx)
    return idx


def _vec(name: str, like: torch.Tensor) -> torch.Tensor:
    key = (name, str(like.device))
    v = _dev.get(key)
    if v is None:
        if like.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(f"DSV41_OBLIT: direction {name} first used inside CUDA graph capture")
        v = torch.from_numpy(_load()[name]).to(device=like.device, dtype=torch.float32)
        _dev[key] = v
    if v.shape[0] != like.shape[-1]:
        raise RuntimeError(f"DSV41_OBLIT: direction {name} has {v.shape[0]} dims, activations {like.shape[-1]}")
    return v


try:  # Triton row kernel: y = x - (x . d) d, fp32 accumulation, one read + one write per row
    import triton
    import triton.language as tl

    @triton.jit
    def _proj_rows(X, Y, D, stride_x, stride_y, H, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, BLOCK)
        m = offs < H
        x = tl.load(X + row * stride_x + offs, mask=m, other=0.0).to(tl.float32)
        d = tl.load(D + offs, mask=m, other=0.0)
        s = tl.sum(x * d, axis=0)
        tl.store(Y + row * stride_y + offs, (x - s * d).to(Y.dtype.element_ty), mask=m)

    _TRITON = True
except Exception:  # noqa: BLE001  (CPU-only test environments)
    _TRITON = False


def _project(x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Out-of-place projection of d (unit, fp32) out of the last dim of x."""
    H = x.shape[-1]
    if x.numel() == 0:
        return x
    if _R7 is not None:  # r7-comm2
        y7 = _R7.r7_take(x, d)  # r7-comm2
        _R7.r7_arm(d, x.shape[0] if x.dim() == 2 else -1)  # r7-comm2
        if y7 is not None:  # r7-comm2
            return y7.view(x.shape)  # r7-comm2
    if _TRITON and x.is_cuda:
        x2 = x if (x.dim() == 2 and x.stride(-1) == 1) else x.reshape(-1, H)
        if x2.stride(-1) != 1:
            x2 = x2.contiguous()
        y = torch.empty((x2.shape[0], H), dtype=x.dtype, device=x.device)
        block = 1 << (H - 1).bit_length()
        if _DSV41_GLUE_PDL:  # r1-pdl
            _dsv41_glue_pdl.proj_rows(x2=x2, y=y, d=d, H=H, block=block)  # r1-pdl
            return y.view(x.shape)  # r1-pdl
        _proj_rows[(x2.shape[0],)](x2, y, d, x2.stride(0), y.stride(0), H, BLOCK=block,
                                   num_warps=8 if block >= 4096 else 4)
        return y.view(x.shape)
    xf = x.float()
    return (xf - (xf @ d).unsqueeze(-1) * d).to(x.dtype)


def project(x: torch.Tensor, layer: int) -> torch.Tensor:
    """Attention / FFN input or attention output of a bound layer (a [tokens, hidden] tensor)."""
    return _project(x, _vec(f"L{layer}", x))


def project_moe(x, layer: int):
    """FFN output: a tensor, or a MoEOutput whose routed part may be an UnfinalizedMoEOutput (deferred top-k)."""
    if isinstance(x, torch.Tensor):
        return project(x, layer)
    routed = x.routed
    if isinstance(routed, torch.Tensor):
        routed = project(routed, layer)
    elif routed is not None:
        g = routed.gemm2_permuted  # per (token, expert) GEMM2 rows; the finalize is a weighted sum of them
        routed = dataclasses.replace(routed, gemm2_permuted=project(g, layer))
    shared = x.shared_output
    if shared is not None:
        shared = project(shared, layer)
    return dataclasses.replace(x, routed=routed, shared_output=shared)


def project_head(h: torch.Tensor) -> torch.Tensor:
    """Final normalised hidden state before the LM head."""
    if _R7 is not None:  # r7-comm2
        y = _project(h, _vec("head", h))  # r7-comm2
        _R7.r7_disarm()  # r7-comm2
        return y  # r7-comm2
    return _project(h, _vec("head", h))


def bound_layers() -> list[int]:
    return sorted(_bound)
