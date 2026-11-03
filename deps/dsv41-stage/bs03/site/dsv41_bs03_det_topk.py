# SPDX-License-Identifier: Apache-2.0
"""BS03 Design B (2026-11-02): one canonical top-512 kernel per decode index layer. Default OFF.

sparse_attn_indexer.py imports this module only when DSV41_BS03_DET_TOPK=1. With the flag on, try_decode() runs vLLM-Moet's
det_topk (7f60bee, Apache-2.0; sm_121 build in ~/dsv41-stage/bs03/build) as ONE launch in place of the live decode chain
(persistent_topk or top_k_per_row_decode, then TIES, then CANON). Prefill, layer 20's torch.topk and the candidate mask do
not change. Only int32 indices change hands.

Function computed, per row of n columns: n <= 512: [0..n-1, -1...]; else every x > v_k, then the lowest-column x == v_k
(float compares, -0 == +0), ascending. It equals the live chain only with DSV41_TOPK_CANON=1, DSV41_TOPK_CANON_TIES=all and
at least 512 finite values per row longer than 512 (the candidate mask guarantees that). try_decode returns False, and the
caller runs the live chain, on anything outside the domain test_det_topk.py covers.

Environment (same value on all 8 ranks):
  DSV41_BS03_SO           the extension, default ~/dsv41-stage/bs03/build/vllm_bs03_det_topk_sm121.so (sha checked)
  DSV41_BS03_SMEM_CAP     dynamic shared memory per CTA in bytes, default 65536
  DSV41_BS03_SMALL        rows of at most this many columns use one CTA, default 12288
  DSV41_BS03_CLUSTER_MAP  "<max rows>:<CTAs per row>,...", rows above the last entry use 1. Default "5:8,11:4,22:2"
                          (the largest C in 8/4/2 with C * rows <= 44 of the 48 SMs, so a 2-CTA l2pf can sit beside it)
  DSV41_BS03_MAX_ROWS     above this many rows the live chain runs, default 128. 8 keeps persistent_topk above 8 rows, where
                          the 10-way merge this kernel exists to remove does not run.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.machinery
import importlib.util
import os

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers import dsv41_topk_ties as _g5det

logger = init_logger(__name__)

K = 512
EXT_NAME = "vllm_bs03_det_topk_sm121"
SO_PATH = os.path.expanduser(os.environ.get("DSV41_BS03_SO", f"~/dsv41-stage/bs03/build/{EXT_NAME}.so"))
SO_SHA256 = "e28cd324f1ad563ddc9a5dbbc1f55bef7f63234f21d832e29fe23c546d4c1513"
SMEM_CAP = int(os.environ.get("DSV41_BS03_SMEM_CAP", "65536"))
SMALL = int(os.environ.get("DSV41_BS03_SMALL", "12288"))
MAX_ROWS = min(int(os.environ.get("DSV41_BS03_MAX_ROWS", "128")), 128)  # 128: the largest row count test_det_topk.py covers
WIDTHS = (524288, 1048576)  # the logits widths test_det_topk.py covers; other widths keep the live chain


def parse_cluster_map(spec: str) -> tuple[tuple[int, int], ...]:
    out = []
    for part in spec.split(","):
        rows, cluster = part.split(":")
        rows, cluster = int(rows), int(cluster)
        if cluster not in (1, 2, 4, 8):
            raise ValueError(f"DSV41_BS03_CLUSTER_MAP: cluster {cluster} is not 1, 2, 4 or 8")
        out.append((rows, cluster))
    return tuple(out)


CLUSTER_MAP = parse_cluster_map(os.environ.get("DSV41_BS03_CLUSTER_MAP", "5:8,11:4,22:2"))


def cluster_for(rows: int) -> int:
    """By row count only, so one captured graph serves every context length."""
    for max_rows, cluster in CLUSTER_MAP:
        if rows <= max_rows:
            return cluster
    return 1


# The live chain's function needs both flags; a dump (DSV41_DET_DUMP) wants the stock rows, so it keeps the chain.
ACTIVE = (
    os.environ.get("DSV41_TOPK_CANON", "0") == "1" and _g5det.TIES_DECODE and not _g5det.DUMP
)
if not ACTIVE:
    logger.warning(
        "DSV41_BS03_DET_TOPK=1 has no effect: it needs DSV41_TOPK_CANON=1 and DSV41_TOPK_CANON_TIES=all, without DSV41_DET_DUMP"
    )

_WARNED: set = set()
_first_call = True


@functools.lru_cache(maxsize=None)
def ext():
    """Load the prebuilt extension once. Raises when it is missing or not the reviewed build: a rank that asked for the
    flag fails at startup rather than serving a different stack from the one it was told to run."""
    with open(SO_PATH, "rb") as f:
        got = hashlib.sha256(f.read()).hexdigest()
    if got != SO_SHA256:
        raise RuntimeError(f"DSV41_BS03_DET_TOPK: {SO_PATH} is {got[:12]}, expected {SO_SHA256[:12]}")
    loader = importlib.machinery.ExtensionFileLoader(EXT_NAME, SO_PATH)
    spec = importlib.util.spec_from_loader(EXT_NAME, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    mod.det_topk_configure(SMEM_CAP)
    return mod


def _why_not(logits, seq_lens, topk, topk_tokens, backend, global_seq_lens) -> str | None:
    if not ACTIVE:
        return "flags"
    if backend != "auto":
        return f"backend {backend}"
    if topk_tokens != K:
        return f"topk_tokens {topk_tokens}"
    if global_seq_lens is not None:
        return "DCP"
    rows = logits.shape[0] if logits.dim() == 2 else -1
    if logits.dtype != torch.float32 or logits.dim() != 2 or logits.stride(1) != 1:
        return f"logits {logits.dtype} {tuple(logits.stride())}"
    if topk.dtype != torch.int32 or topk.dim() != 2 or topk.stride(1) != 1 or tuple(topk.shape) != (rows, K):
        return f"topk {topk.dtype} {tuple(topk.shape)} {tuple(topk.stride())}"
    if seq_lens.dtype != torch.int32 or not seq_lens.is_contiguous() or seq_lens.numel() != rows:
        return f"seq_lens {seq_lens.dtype} {tuple(seq_lens.shape)} rows {rows}"
    if not (0 < rows <= MAX_ROWS):
        return f"rows {rows}"
    if not (logits.is_cuda and topk.is_cuda and seq_lens.is_cuda):
        return "device"
    if logits.shape[1] not in WIDTHS:
        return f"width {logits.shape[1]}"
    if torch.cuda.get_device_capability(logits.device) != (12, 1):
        return f"device capability {torch.cuda.get_device_capability(logits.device)}"
    # Keeps the launch in the float4 read path the test covers.
    if logits.stride(0) % 4 != 0 or logits.data_ptr() % 16 != 0:
        return f"row layout stride {logits.stride(0)} ptr {logits.data_ptr() % 16}"
    return None


def decode(logits: torch.Tensor, seq_lens: torch.Tensor, next_n: int, topk: torch.Tensor, cluster: int | None = None) -> None:
    """The launch itself; `topk` holds the rows afterwards. seq_lens.numel() == rows, as persistent_topk requires."""
    rows = logits.shape[0]
    ext().det_topk_decode(logits, seq_lens, int(next_n), topk, rows, K, cluster_for(rows) if cluster is None else int(cluster), SMALL)


def try_decode(logits, seq_lens, next_n, topk, topk_tokens, backend, global_seq_lens) -> bool:
    """True: the kernel wrote `topk` (top-k + TIES + CANON in one launch). False: the caller runs the live chain."""
    global _first_call
    why = _why_not(logits, seq_lens, topk, topk_tokens, backend, global_seq_lens)
    if why is not None:
        if why not in _WARNED:
            _WARNED.add(why)
            logger.warning("DSV41_BS03_DET_TOPK: live chain kept for %s", why)
        return False
    decode(logits, seq_lens, next_n, topk)
    if _first_call:
        _first_call = False
        logger.info(
            "DSV41_BS03_DET_TOPK: det_topk in use (first call: rows=%d width=%d cluster=%d smem_cap=%d small=%d map=%s)",
            logits.shape[0], logits.shape[1], cluster_for(logits.shape[0]), SMEM_CAP, SMALL, CLUSTER_MAP,
        )
    return True
