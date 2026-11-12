"""r2-draftmoe (climb round 2, lever G17 = audit moe:L-D; default off): DSpark drafter routed-MoE compaction.

The DSpark drafter (mtp.0-2: 128 routed experts, top-3, hidden 5120; per TP8 rank 288 live intermediate columns)
stores FC1 padded to N=768 = [gate 288 + 96 zero rows | up 288 + 96 zero rows] and permutes its activations with the
stock 5-kernel glue (count, zeros, full(-1), ep_scatter_1, ep_scatter_2, plus transpose_and_pack inside FC1). The
target MoE already runs both improvements in production (DSV41_MOE_COMPACT, DSV41_MOE_FUSED_PERMUTE=1); this module
extends them to the drafter geometry only.

  DSV41_DRAFT_MOE_G17      read once at import by the four patched modules; with it unset or 0 this module is never
                           imported and no line below runs.
    0 / unset              original behaviour
    fc1                    drafter FC1 stored unpadded, w13 [128, 576, 2560] (rows [gate 288 | up 288]); FC1 runs at
                           N=576 on the same M32 template (and on the 128-row DeepGEMM path for M > DSV41_DRAFT_M32_MAX_M);
                           the existing compact SiLU-mul-quant writer (padded_output_size=384) writes the FC2 input with
                           its 96 zero columns, so FC2 (K=384) and everything after it are untouched.
    perm                   drafter fused permute on the M32 path (align 32): plan + scatter Triton kernels of g3 mfront
                           (deep_gemm_utils.dsv41_fused_permute, E/TOPK constexpr) instead of the stock glue; FC1
                           receives packed UE8M0 int32 scales and skips transpose_and_pack.
    1                      fc1 + perm (the treatment arm)
    permverify             serve the stock drafter permute and also run the fused one into shadow buffers through the
                           existing mfront verifier (report: touch ~/dsv41-logs/mfront-verify.trigger, see
                           deep_gemm_utils._Dsv41FpVerifier). Diagnostic only; combinable with fc1 ("fc1,permverify").
  Values may be comma-separated ("fc1,perm" == "1"). perm and permverify are exclusive. Anything else is a hard
  startup error.

FC2 K 384 -> 288 (the third part of audit moe:L-D) is NOT built: it drops trailing k32 MMA chunks, i.e. it changes the
per-element accumulation sequence (the -0/+0 class), needs a rebuilt FC2C/Athena native for E=128, and is worth ~0.05% G1
(skeptic verdict G17). See README.md.
"""
from __future__ import annotations

import os
import threading

# (FC1 N, hidden K, top-k, local experts) of the drafter routed MoE on one TP8 rank.
DRAFT_PADDED_GEOMETRY = (768, 5120, 3, 128)
DRAFT_COMPACT_GEOMETRY = (576, 5120, 3, 128)
_TOKENS = {"fc1", "perm", "permverify"}
_LOCK = threading.Lock()
_COMPACT_LAYERS = 0


def parse(raw: str | None) -> tuple[bool, str]:
    """-> (FC1 compaction on, permute mode '' | '1' | 'verify')."""
    raw = (raw or "0").strip().lower()
    if raw in ("", "0", "off", "false"):
        return False, ""
    parts = {p.strip() for p in raw.split(",") if p.strip()}
    if parts in ({"1"}, {"on"}, {"true"}):
        parts = {"fc1", "perm"}
    bad = parts - _TOKENS
    if bad or not parts:
        raise ValueError(f"DSV41_DRAFT_MOE_G17={raw!r}: use 0, 1, or a comma list of fc1, perm, permverify")
    if {"perm", "permverify"} <= parts:
        raise ValueError(f"DSV41_DRAFT_MOE_G17={raw!r}: perm and permverify are exclusive")
    return "fc1" in parts, ("1" if "perm" in parts else "verify" if "permverify" in parts else "")


FC1, PERM = parse(os.environ.get("DSV41_DRAFT_MOE_G17", "0"))


def _log(msg: str, *args) -> None:
    from vllm.logger import init_logger

    init_logger(__name__).info(msg, *args)


def drafter_compact_eligible(method, num_experts: int, hidden_size: int, intermediate_size_per_partition: int,
                             is_sm121: bool, is_deepgemm_mxfp4: bool, silu) -> bool:
    """Mxfp4MoEMethod.create_weights gate: the drafter twin of the target DSV41_MOE_COMPACT gate (mxfp4.py)."""
    global _COMPACT_LAYERS
    if not FC1:
        return False
    moe = method.moe
    parallel = moe.moe_parallel_config
    compact_writer = os.environ.get("DSV41_MOE_COMPACT") == "1"  # _act_mul_quant's compact branch asserts it
    ok = (
        compact_writer
        and is_sm121
        and is_deepgemm_mxfp4
        and method.w13_intermediate_size == intermediate_size_per_partition  # the target gate did not fire
        and (num_experts, hidden_size, intermediate_size_per_partition) == (128, 5120, 384)
        and moe.num_experts == 128
        and moe.experts_per_token == 3
        and moe.intermediate_size_per_partition_unpadded == 288
        and parallel.tp_size == 8
        and not parallel.use_ep
        and not parallel.enable_eplb
        and not moe.tp_shard_with_padding
        and not moe.has_bias
        and not moe.is_lora_enabled
        and moe.activation == silu
    )
    if ok:
        with _LOCK:
            _COMPACT_LAYERS += 1
            n = _COMPACT_LAYERS
        _log("DSV41_DRAFT_MOE_G17 fc1: drafter compact FC1 enabled for routed-MoE layer %d (E128 top-3, w13 rows 576 "
             "instead of 768, FC2 input width 384; supersedes the N768 wording of the DSV41_DRAFT_MOE_M32 line)", n)
    elif (num_experts, hidden_size) == (128, 5120):
        _log("DSV41_DRAFT_MOE_G17 fc1: drafter-sized MoE layer NOT compacted (gate mismatch: ispp=%s unpadded=%s "
             "topk=%s tp=%s sm121=%s deepgemm=%s DSV41_MOE_COMPACT=1:%s); it stays padded (N768)",
             intermediate_size_per_partition, moe.intermediate_size_per_partition_unpadded, moe.experts_per_token,
             parallel.tp_size, is_sm121, is_deepgemm_mxfp4, compact_writer)
    return ok


def compact_layers() -> int:
    return _COMPACT_LAYERS


def perm_eligible(aq, aq_scale, topk_ids, local_num_experts, expert_map, expert_tokens_meta, block_k,
                  use_psum_layout, align_used, slots) -> bool:
    """deep_gemm_utils.dsv41_fp_eligible with the drafter geometry (E 128, top-3) instead of the target's (384, 6)."""
    import torch

    if not PERM:
        return False
    if expert_map is not None or use_psum_layout or align_used != 32:
        return False
    if expert_tokens_meta is not None and getattr(expert_tokens_meta, "expert_num_tokens", None) is not None:
        return False
    if local_num_experts != 128 or topk_ids.dim() != 2 or topk_ids.size(1) != 3:
        return False
    M, H = aq.shape
    if topk_ids.size(0) != M or M * 3 > slots or block_k != 128 or H % 128 or H % 4:
        return False
    if aq.dtype != torch.float8_e4m3fn or not aq.is_contiguous():
        return False
    if aq_scale.dtype != torch.float32 or aq_scale.dim() != 2 or tuple(aq_scale.shape) != (M, H // 128):
        return False
    return topk_ids.dtype in (torch.int32, torch.int64)
