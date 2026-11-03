# SPDX-License-Identifier: Apache-2.0
# Derived from the copied vLLM deep_gemm_utils.py gather, itself based on
# ModelTC/LightLLM deepep_scatter_gather.py. Detached, default-off experiment.
"""Source-only target-TP8 gather that does not read failed FC2 output.

The caller must establish stream ordering from status writer to this kernel,
retain all buffers, prevent publishing failed steps, and coordinate all TP
ranks. A zero placeholder on failure is not a valid model result. No serving
integration, all-rank propagation or CUDA-graph lifecycle is provided here.
"""
from __future__ import annotations
import math
import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _gated_gather(
    total_token_num, input_tensor, input_stride0, input_stride1,
    recv_topk_ids, ids_stride0, ids_stride1,
    recv_topk_weight, weights_stride0, weights_stride1,
    input_index, index_stride0, index_stride1,
    output_tensor, output_stride0, output_stride1,
    step_status, topk_num: tl.constexpr, BLOCK_D: tl.constexpr,
):
    cur_block = tl.program_id(0)
    start_cur_token = tl.program_id(1)
    grid_num = tl.num_programs(1)
    # The status writer must finish on this stream, or be joined by a CUDA
    # event, before launch. Status must remain stable until this gather finishes.
    ok = (tl.load(step_status) & 11) == 0
    for cur_token in range(start_cur_token, total_token_num, grid_num):
        off_d = tl.arange(0, BLOCK_D)
        accumulator = tl.zeros([BLOCK_D], dtype=tl.float32)
        if ok:
            # Keep the normal source's loop and arithmetic, but do not even
            # read top-k indices or partial FC2 output on a failed status.
            for topk_index in range(0, topk_num):
                expert_id = tl.load(recv_topk_ids + cur_token * ids_stride0 + topk_index * ids_stride1)
                if expert_id >= 0:
                    source_index = tl.load(input_index + cur_token * index_stride0 + topk_index * index_stride1)
                    weight = tl.load(recv_topk_weight + cur_token * weights_stride0 + topk_index * weights_stride1)
                    value = tl.load(input_tensor + source_index.to(tl.int64) * input_stride0 +
                                    cur_block * BLOCK_D + off_d * input_stride1)
                    accumulator += value.to(tl.float32) * weight
        # Zeros on failure are placeholders only; the entire step and any
        # affected state must be discarded before anything is published.
        tl.store(output_tensor + tl.cast(cur_token, tl.int64) * output_stride0 +
                 cur_block * BLOCK_D + off_d * output_stride1,
                 accumulator.to(output_tensor.dtype.element_ty))


@torch.no_grad()
def gated_gather(input_tensor, topk_ids, topk_weights, inv_perm, expert_map,
                 output, status, *, experimental: bool = False):
    """Enqueue only; caller must implement status ownership and publication."""
    if not experimental:
        raise RuntimeError("explicit experimental status-gated gather opt-in required")
    if expert_map is not None:
        raise ValueError("detached target TP8 gather only supports EP-off expert_map=None")
    tensors = (input_tensor, topk_ids, topk_weights, inv_perm, output, status)
    if not all(t.is_cuda and t.device == input_tensor.device for t in tensors):
        raise ValueError("all tensors must reside on the same CUDA device")
    if status.dtype != torch.int32 or tuple(status.shape) != (1,) or not status.is_contiguous():
        raise ValueError("status must be contiguous CUDA int32[1]")
    if input_tensor.ndim != 2 or output.ndim != 2 or output.shape[1] != input_tensor.shape[1]:
        raise ValueError("invalid FC2 input/output metadata")
    n = output.shape[0]
    if (topk_ids.ndim != 2 or topk_ids.shape[0] != n or topk_weights.shape != topk_ids.shape
            or inv_perm.shape != topk_ids.shape):
        raise ValueError("inconsistent top-k metadata")
    if n == 0:
        return output
    block_d = math.gcd(input_tensor.shape[1], 1024)
    if block_d == 0:
        raise ValueError("hidden dimension must be positive")
    grid = (triton.cdiv(input_tensor.shape[1], block_d), min(n, 1024))
    _gated_gather[grid](n, input_tensor, input_tensor.stride(0), input_tensor.stride(1),
                        topk_ids, topk_ids.stride(0), topk_ids.stride(1),
                        topk_weights, topk_weights.stride(0), topk_weights.stride(1),
                        inv_perm, inv_perm.stride(0), inv_perm.stride(1),
                        output, output.stride(0), output.stride(1), status,
                        topk_num=topk_ids.shape[1], BLOCK_D=block_d, num_warps=2)
    return output
