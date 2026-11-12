# SPDX-License-Identifier: Apache-2.0
# [r2-moeback G25] DSV41_FC2_GATHER_ADD: fold the routed+shared MoE add into Athena's status-gated gather.
"""Hand-off between MoERunner._apply_quant_method and the Athena FC2 gather.

Imported only when DSV41_FC2_GATHER_ADD=1 (from the gated branches in moe_runner.py and
~/dsv41-fc2/fc2_status_gated_gather_trial.py). With the flag unset nothing imports this module.

Protocol (one MoE call):
  1. ``arm(runner, overlapping)`` right before the routed experts run. It records the shared-expert
     output tensor of this call (peeked from ``SharedExperts._output[idx]``; the consuming ``.output``
     property is NOT touched, so the runner still consumes it exactly once, as today) and, on the
     multi-stream path, the shared expert's ``_output_ready_event[idx]``. It returns a token only when
     every runner-side condition for an exact fold holds; otherwise ``None`` and nothing changes.
  2. ``take(output)`` from the Athena gather. Only if a token is armed and the shared tensor matches the
     gather output (shape, dtype, device, no overlap): it makes the current stream wait on the shared
     expert's ready event (the same event the runner waits on today right after the experts), marks the
     token used and returns the shared tensor; the gather then writes bf16(f32(bf16(acc)) + f32(shared)).
     Any other gather (drafter MoE, stock deep_gemm gather, non-Athena layers) never calls take().
  3. ``finish(token, fused_out)`` right after the routed experts return: disarms and returns True iff
     the shared output was folded, in which case MoERunner.forward skips its ``shared + fused`` add.

Everything here runs on the host at enqueue (or graph-capture) time; it adds one event wait (an edge
the add already had) and replaces one kernel by another; it allocates nothing on the device.
"""
from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton

TAG = "r2-moeback-g25"


@triton.jit
def _gated_gather_add(
    total_token_num, input_tensor, input_stride0, input_stride1,
    recv_topk_ids, ids_stride0, ids_stride1,
    recv_topk_weight, weights_stride0, weights_stride1,
    input_index, index_stride0, index_stride1,
    output_tensor, output_stride0, output_stride1,
    shared_tensor, shared_stride0, shared_stride1,
    step_status, topk_num: tl.constexpr, BLOCK_D: tl.constexpr,
):
    # Byte-for-byte the body of ~/dsv41-fc2/fc2_status_gated_gather_trial.py:_gated_gather (sha 733c3a87),
    # plus: one independent load of the shared-expert row block issued before the k-loop, and the
    # epilogue  out = bf16( f32(bf16(acc)) + f32(shared) )  in place of  out = bf16(acc).
    # bf16(acc) is the store rounding the gather does today (cvt.rn.bf16x2.f32); the f32 add with one
    # round-to-nearest-even to bf16 is what torch's CUDAFunctor_add<BFloat16> did on the next kernel.
    cur_block = tl.program_id(0)
    start_cur_token = tl.program_id(1)
    grid_num = tl.num_programs(1)
    ok = (tl.load(step_status) & 11) == 0
    for cur_token in range(start_cur_token, total_token_num, grid_num):
        off_d = tl.arange(0, BLOCK_D)
        shared = tl.load(shared_tensor + tl.cast(cur_token, tl.int64) * shared_stride0 +
                         cur_block * BLOCK_D + off_d * shared_stride1)
        accumulator = tl.zeros([BLOCK_D], dtype=tl.float32)
        if ok:
            for topk_index in range(0, topk_num):
                expert_id = tl.load(recv_topk_ids + cur_token * ids_stride0 + topk_index * ids_stride1)
                if expert_id >= 0:
                    source_index = tl.load(input_index + cur_token * index_stride0 + topk_index * index_stride1)
                    weight = tl.load(recv_topk_weight + cur_token * weights_stride0 + topk_index * weights_stride1)
                    value = tl.load(input_tensor + source_index.to(tl.int64) * input_stride0 +
                                    cur_block * BLOCK_D + off_d * input_stride1)
                    accumulator += value.to(tl.float32) * weight
        routed = accumulator.to(output_tensor.dtype.element_ty)
        summed = routed.to(tl.float32) + shared.to(tl.float32)
        tl.store(output_tensor + tl.cast(cur_token, tl.int64) * output_stride0 +
                 cur_block * BLOCK_D + off_d * output_stride1,
                 summed.to(output_tensor.dtype.element_ty))


def launch_gather_add(grid, n, input_tensor, topk_ids, topk_weights, inv_perm, output, status, shared,
                      block_d):
    """Same grid, BLOCK_D and num_warps as the Athena gather launch it replaces."""
    _gated_gather_add[grid](n, input_tensor, input_tensor.stride(0), input_tensor.stride(1),
                            topk_ids, topk_ids.stride(0), topk_ids.stride(1),
                            topk_weights, topk_weights.stride(0), topk_weights.stride(1),
                            inv_perm, inv_perm.stride(0), inv_perm.stride(1),
                            output, output.stride(0), output.stride(1),
                            shared, shared.stride(0), shared.stride(1), status,
                            topk_num=topk_ids.shape[1], BLOCK_D=block_d, num_warps=2)
    return output


class _Token:
    __slots__ = ("shared", "event", "used", "declined")

    def __init__(self, shared, event):
        self.shared = shared
        self.event = event
        self.used = False
        self.declined = None


_ARMED: _Token | None = None
_COUNTS = {"armed": 0, "folded": 0, "declined": 0, "not_armed": 0}


def _runner_ok(runner) -> str | None:
    """Return None when the runner's post-op path between the gather and the add is a no-op."""
    se = getattr(runner, "_shared_experts", None)
    if se is None:
        return "no shared experts"
    if getattr(runner, "enable_dbo", False) or getattr(se, "enable_dbo", False):
        return "dbo"
    if getattr(runner, "routed_scaling_factor", 1.0) != 1.0:
        return "routed_scaling_factor != 1"
    if getattr(runner, "routed_output_transform", None) is not None:
        return "routed_output_transform"
    if getattr(runner, "routed_input_transform", None) is not None:
        return "routed_input_transform"
    cfg = runner.moe_config
    if getattr(cfg, "is_sequence_parallel", False):
        return "sequence parallel"
    if getattr(cfg, "pcp_size", 1) != 1:
        return "pcp"
    if runner.do_naive_dispatch_combine:
        return "naive dispatch/combine"
    if runner._fused_output_is_reduced:
        return "fused output already reduced"
    return None


def arm(runner, overlapping: bool):
    """Arm the fold for this MoE call; returns a token or None (fold impossible -> unchanged path)."""
    global _ARMED
    _ARMED = None
    why = _runner_ok(runner)
    if why is not None:
        _COUNTS["not_armed"] += 1
        return None
    se = runner._shared_experts
    idx = se._output_idx
    shared = se._output[idx]
    if not isinstance(shared, torch.Tensor):
        # NO_OVERLAP computes it before the experts, MULTI_STREAM_OVERLAPPED assigns the tensor at
        # enqueue time. MK_INTERNAL_OVERLAPPED computes it later, inside finalize: no fold.
        _COUNTS["not_armed"] += 1
        return None
    event = None
    if overlapping:
        if getattr(se, "_stream", None) is None:
            _COUNTS["not_armed"] += 1
            return None
        event = se._output_ready_event[idx]
    tok = _Token(shared, event)
    _ARMED = tok
    _COUNTS["armed"] += 1
    return tok


def _overlaps(a: torch.Tensor, b: torch.Tensor) -> bool:
    a0 = a.data_ptr()
    b0 = b.data_ptr()
    a1 = a0 + a.numel() * a.element_size()
    b1 = b0 + b.numel() * b.element_size()
    return a0 < b1 and b0 < a1


def take(output: torch.Tensor):
    """Called by the Athena gather. Returns the shared tensor to fold, or None (plain gather)."""
    tok = _ARMED
    if tok is None:
        return None
    if tok.used:
        # A second Athena gather under one armed MoE call would mean chunking: the first one already
        # folded the shared output. Fail closed instead of producing a double or missing add.
        raise RuntimeError("DSV41_FC2_GATHER_ADD: second Athena gather under one armed MoE call")
    shared = tok.shared
    if (shared.dtype != torch.bfloat16 or output.dtype != torch.bfloat16
            or shared.dim() != 2 or output.dim() != 2
            or tuple(shared.shape) != tuple(output.shape)
            or shared.device != output.device
            or shared.numel() == 0
            or _overlaps(shared, output)):
        tok.declined = "shape/dtype/device/overlap"
        _COUNTS["declined"] += 1
        return None
    if tok.event is not None:
        # The same ready event MoERunner._apply_quant_method waits on right after the experts today;
        # waiting here as well moves the dependency from the add to the gather (the second wait is a
        # no-op edge).
        tok.event.wait(torch.cuda.current_stream(output.device))
    tok.used = True
    _COUNTS["folded"] += 1
    if _COUNTS["folded"] == 1:  # one host-side line per process, for the window log (rank0.log)
        print(f"DSV41_FC2_GATHER_ADD: first folded MoE call (M={output.shape[0]}, "
              f"event={'yes' if tok.event is not None else 'no'})", flush=True)
    return shared


def finish(tok, fused_out) -> bool:
    """Disarm; True iff the shared output was folded into ``fused_out`` (then the runner skips its add)."""
    global _ARMED
    if _ARMED is tok:
        _ARMED = None
    if tok is None or not tok.used:
        return False
    if not isinstance(fused_out, torch.Tensor) or tuple(fused_out.shape) != tuple(tok.shared.shape):
        raise RuntimeError("DSV41_FC2_GATHER_ADD: folded gather but the routed output is not the folded tensor")
    return True


def counts() -> dict:
    return dict(_COUNTS)
