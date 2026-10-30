# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.deep_gemm_utils import (
    compute_aligned_M_and_alignment,
    deepgemm_moe_permute,
    deepgemm_unpermute_and_reduce,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
    per_token_group_quant_fp8_packed_for_deepgemm,
    silu_mul_per_token_group_quant_fp8_colmajor,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    silu_mul_quant_fp8_packed_triton as fused_silu_mul_fp8_quant_packed,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
    kMxfp4Static,
    kMxfp8Dynamic,
    kMxfp8Static,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    DeepGemmQuantScaleFMT,
    get_mk_alignment_for_contiguous_layout,
    is_deep_gemm_supported,
    m_grouped_fp8_fp4_gemm_nt_contiguous,
    m_grouped_fp8_gemm_nt_contiguous,
    mk_alignment_scope,
)
from vllm.utils.import_utils import has_deep_gemm

logger = init_logger(__name__)


def _fp8_workspace_shape(
    num_rows: int, num_columns: int, workspace_dtype: torch.dtype
) -> tuple[int, int]:
    """Size an FP8 byte buffer stored in a workspace of another dtype."""
    bytes_per_workspace_element = workspace_dtype.itemsize
    fp8_columns_per_workspace_element = (
        bytes_per_workspace_element // torch.float8_e4m3fn.itemsize
    )
    assert bytes_per_workspace_element % torch.float8_e4m3fn.itemsize == 0
    return (
        num_rows,
        -(-num_columns // fp8_columns_per_workspace_element),
    )


def _valid_deep_gemm_shape(M: int, N: int, K: int) -> bool:
    align = get_mk_alignment_for_contiguous_layout()[0]
    return align <= M and N % align == 0 and K % align == 0


def _valid_deep_gemm(
    hidden_states: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor
) -> bool:
    """Check if the given problem size is supported by the DeepGemm grouped
    gemm kernel.  All of M, N, K and the quantization block_shape must be
    aligned by `dg.get_m_alignment_for_contiguous_layout()`.
    """
    if not has_deep_gemm():
        logger.debug_once("DeepGemm disabled: deep_gemm not available.")
        return False

    M = hidden_states.size(0)
    _, K, N = w2.size()

    align = get_mk_alignment_for_contiguous_layout()[0]

    if not _valid_deep_gemm_shape(M, N, K):
        logger.debug_once(
            "DeepGemm disabled due to unaligned problem size. "
            "M: %s, N: %s, K: %s. M should >= %s "
            "and N and K must be multiples of %s. "
            "This is not an error and we will fall back to triton.",
            M,
            N,
            K,
            align,
            align,
        )
        return False
    elif N <= 512:
        logger.debug_once(
            "DeepGemm disabled for N <= 512. M: %s, N: %s, K: %s. "
            "This means we will fallback to triton "
            "for this specific shape for further speed up.",
            M,
            N,
            K,
        )
        return False

    if w1.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn:
        logger.debug_once(
            "DeepGemm disabled: invalid weight dtype(s). w1.dtype: %s, w2.dtype: %s",
            w1.dtype,
            w2.dtype,
        )
        return False

    if (
        not hidden_states.is_contiguous()
        or not w1.is_contiguous()
        or not w2.is_contiguous()
    ):
        logger.debug_once(
            "DeepGemm disabled: weights or activations not contiguous. "
            "hidden_states.is_contiguous(): %s, w1.is_contiguous(): %s, "
            "w2.is_contiguous(): %s",
            hidden_states.is_contiguous(),
            w1.is_contiguous(),
            w2.is_contiguous(),
        )
        return False

    return True


class DeepGemmExperts(mk.FusedMoEExpertsModular):
    """DeepGemm-based fused MoE expert implementation."""

    def __init__(self, moe_config: FusedMoEConfig, quant_config: FusedMoEQuantConfig):
        super().__init__(moe_config=moe_config, quant_config=quant_config)
        # MXFP8: FP8 e4m3 values + UE8M0 1x32 block scales (Blackwell). Reuses
        # the same grouped GEMM (aliased to fp8_fp4) with recipe (1, 32).
        self.mxfp8 = quant_config.block_shape == [1, 32]
        if self.mxfp8:
            assert quant_config.quant_dtype == "mxfp8"
        else:
            assert quant_config.block_shape == get_mk_alignment_for_contiguous_layout()
            assert quant_config.quant_dtype == torch.float8_e4m3fn
        assert not quant_config.per_act_token_quant
        assert not quant_config.per_out_ch_quant

        self.gemm1_clamp_limit = quant_config.gemm1_clamp_limit
        # Gated-activation params: silu == swigluoai with alpha=1, beta=0.
        # FP8 (silu) configs leave these None, reproducing plain silu.
        self.gemm1_alpha = (
            quant_config.gemm1_alpha if quant_config.gemm1_alpha is not None else 1.0
        )
        self.gemm1_beta = (
            quant_config.gemm1_beta if quant_config.gemm1_beta is not None else 0.0
        )

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return is_deep_gemm_supported()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        if (weight_key, activation_key) == (kFp8Static128BlockSym, kFp8Dynamic128Sym):
            return True
        # MXFP8 1x32 uses the fp8_fp4 grouped GEMM with recipe (1, 32) — only
        # available on Blackwell (SM100).
        if (weight_key, activation_key) == (kMxfp8Static, kMxfp8Dynamic):
            return current_platform.is_device_capability_family(100)
        return False

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        # silu/swigluoai go through the fused alpha/beta kernel; swiglustep
        # uses the unfused activation path. The fused kernel reads packed w13
        # (gate = first half, up = second half), so it implements the
        # *uninterleaved* SwiGLU-OAI variant.
        return activation in [
            MoEActivation.SILU,
            MoEActivation.SWIGLUSTEP,
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        ]

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        # NOTE(rob): discovered an IMA with this combination. Needs investigation.
        return not (
            moe_parallel_config.use_fi_nvl_two_sided_kernels
            or moe_parallel_config.use_fi_nvl_one_sided_kernels
        )

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        assert self.block_shape is not None
        # Use the contiguous-layout M alignment (matches apply()); block_shape[0]
        # is the quant block (1 for MXFP8) and would under-size the workspace.
        block_m = get_mk_alignment_for_contiguous_layout()[0]
        M_sum, align_used = compute_aligned_M_and_alignment(
            M, topk, local_num_experts, block_m, expert_tokens_meta
        )
        assert M_sum % align_used == 0

        activation_out_dim = self.adjust_N_for_activation(N, activation)
        # workspace1 is allocated in the activation dtype by the workspace
        # manager, but is only ever viewed and used as FP8 in apply(). Size it
        # by bytes instead of reserving one BF16/FP16 element per FP8 element.
        workspace1 = _fp8_workspace_shape(
            M_sum,
            max(activation_out_dim, K),
            self.workspace_dtype(self.moe_config.in_dtype),
        )
        workspace2 = (M_sum, max(N, K))
        output = (M, K)
        return (workspace1, workspace2, output)

    def _act_mul_quant(
        self,
        input: torch.Tensor,
        output: torch.Tensor,
        activation: MoEActivation,
        expert_ends: torch.Tensor | None = None,
        expert_alignment: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.block_shape is not None
        block_k = self.block_shape[1]
        scale_fmt = DeepGemmQuantScaleFMT.from_oracle()

        M_sum, N = input.size()
        activation_out_dim = self.adjust_N_for_activation(N, activation)

        # silu and swigluoai are both expressible by the fused gated kernel via
        # (alpha, beta): silu uses alpha=1, beta=0; swigluoai uses config values.
        # The fused kernel reads packed w13, hence SWIGLUOAI_UNINTERLEAVE.
        fused_gated = activation in (
            MoEActivation.SILU,
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        )

        # 1. DeepGemm UE8M0: fused gate+mul+clamp+quant+pack
        if scale_fmt == DeepGemmQuantScaleFMT.UE8M0:
            if fused_gated:
                return fused_silu_mul_fp8_quant_packed(
                    input=input,
                    output_q=output,
                    group_size=block_k,
                    clamp_limit=self.gemm1_clamp_limit,
                    alpha=self.gemm1_alpha,
                    beta=self.gemm1_beta,
                    expert_ends=expert_ends,
                    expert_alignment=expert_alignment,
                )
            act_out = torch.empty(
                (M_sum, activation_out_dim), dtype=input.dtype, device=input.device
            )
            self.activation(activation, act_out, input)
            a2q, a2q_scale = per_token_group_quant_fp8_packed_for_deepgemm(
                act_out,
                block_k,
                out_q=output,
            )
            return a2q, a2q_scale

        # 2. Hopper / non‑E8M0: prefer the fused gate+mul+quant kernel
        if fused_gated:
            use_ue8m0 = scale_fmt == DeepGemmQuantScaleFMT.FLOAT32_CEIL_UE8M0
            return silu_mul_per_token_group_quant_fp8_colmajor(
                input=input,
                output=output,
                use_ue8m0=use_ue8m0,
                clamp_limit=self.gemm1_clamp_limit,
                group_size=block_k,
                alpha=self.gemm1_alpha,
                beta=self.gemm1_beta,
                expert_ends=expert_ends,
                expert_alignment=expert_alignment,
            )

        # 3. fallback path for non-SiLU activations in non‑UE8M0 cases.
        act_out = torch.empty(
            (M_sum, activation_out_dim), dtype=input.dtype, device=input.device
        )
        self.activation(activation, act_out, input)
        return per_token_group_quant_fp8(
            act_out, block_k, column_major_scales=True, out_q=output
        )

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        assert a1q_scale is not None
        assert a2_scale is None
        assert self.block_shape is not None
        assert self.w1_scale is not None
        assert self.w2_scale is not None

        a1q = hidden_states
        _, N, K = w1.size()

        local_num_experts = w1.size(0)
        if global_num_experts == -1:
            global_num_experts = local_num_experts

        assert w2.size(1) == K

        M_sum, _ = compute_aligned_M_and_alignment(
            M=topk_ids.size(0),
            num_topk=topk_ids.size(1),
            local_num_experts=local_num_experts,
            alignment=get_mk_alignment_for_contiguous_layout()[0],
            expert_tokens_meta=expert_tokens_meta,
        )

        a1q_perm = _resize_cache(
            workspace13.view(dtype=torch.float8_e4m3fn), (M_sum, K)
        )
        # Both GEMMs and quantization must agree on the live expert ranges.
        use_psum_layout = (
            not self.mxfp8
            and (
                current_platform.is_device_capability_family(90)
                or (
                    current_platform.is_device_capability_family(100)
                    and DeepGemmQuantScaleFMT.from_oracle()
                    == DeepGemmQuantScaleFMT.UE8M0
                )
            )
            and activation in (MoEActivation.SILU, MoEActivation.SWIGLUOAI_UNINTERLEAVE)
        )
        a1q, a1q_scale, grouped_layout, inv_perm, align_used = deepgemm_moe_permute(
            aq=a1q,
            aq_scale=a1q_scale,
            topk_ids=topk_ids,
            local_num_experts=local_num_experts,
            expert_map=expert_map,
            expert_tokens_meta=expert_tokens_meta,
            aq_out=a1q_perm,
            # MXFP8 uses a 32-element activation-scale group (block_shape[1]);
            # FP8-block keeps the default (128) alignment.
            block_size=self.block_shape[1] if self.mxfp8 else None,
            use_psum_layout=use_psum_layout,
        )
        assert a1q.size(0) == M_sum

        # MXFP8 (1x32) drives the fp8_fp4-aliased grouped GEMM with recipe
        # (1, 32); the FP8 block path keeps the default (128) recipe.
        gemm_kwargs: dict = (
            {"recipe_a": (1, self.block_shape[1]), "recipe_b": (1, self.block_shape[1])}
            if self.mxfp8
            else {}
        )

        if use_psum_layout:
            gemm_kwargs["use_psum_layout"] = True

        # Cap DG's BLOCK_M heuristic at the workspace's per-expert alignment;
        # otherwise the scheduler can pick the wrong expert id from m_indices
        # under cudagraph replay.
        with mk_alignment_scope(align_used):
            mm1_out = _resize_cache(workspace2, (M_sum, N))
            m_grouped_fp8_gemm_nt_contiguous(
                (a1q, a1q_scale),
                (w1, self.w1_scale),
                mm1_out,
                grouped_layout,
                **gemm_kwargs,
            )

            activation_out_dim = self.adjust_N_for_activation(N, activation)
            quant_out = _resize_cache(
                workspace13.view(dtype=torch.float8_e4m3fn), (M_sum, activation_out_dim)
            )
            a2q, a2q_scale = self._act_mul_quant(
                input=mm1_out.view(-1, N),
                output=quant_out,
                activation=activation,
                expert_ends=grouped_layout if use_psum_layout else None,
                expert_alignment=align_used if use_psum_layout else 0,
            )

            mm2_out = _resize_cache(workspace2, (M_sum, K))
            m_grouped_fp8_gemm_nt_contiguous(
                (a2q, a2q_scale),
                (w2, self.w2_scale),
                mm2_out,
                grouped_layout,
                **gemm_kwargs,
            )

        if apply_router_weight_on_input:
            topk_weights = torch.ones_like(topk_weights)

        deepgemm_unpermute_and_reduce(
            a=mm2_out,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            inv_perm=inv_perm,
            expert_map=expert_map,
            output=output,
        )


class DeepGemmFP4Experts(mk.FusedMoEExpertsModular):
    """DeepGemm-based fused MoE expert implementation for FP4 weights.

    Uses m_grouped_fp8_fp4_gemm_nt_contiguous with FP8 activations and
    MXFP4 (FP4 E2M1 packed as uint8) weights. Requires Blackwell-family
    GPUs (SM100 datacenter or SM120 consumer).
    """

    # FP8 activation block size (hardcoded since mxfp4_w4a8 quant config
    # does not set a block_shape on the activation descriptor).
    _ACT_BLOCK_K = 128
    # FP4 weight block size
    _WEIGHT_BLOCK_K = 32

    def __init__(self, moe_config: FusedMoEConfig, quant_config: FusedMoEQuantConfig):
        super().__init__(moe_config=moe_config, quant_config=quant_config)
        assert quant_config.weight_quant_dtype == "mxfp4"
        assert not quant_config.per_act_token_quant
        assert not quant_config.per_out_ch_quant

        self.gemm1_clamp_limit = quant_config.gemm1_clamp_limit

        self._dsv41_target_m32 = None
        if os.environ.get("DSV41_SM121_TARGET_M32") == "1" and moe_config.num_experts == 384:
            assert moe_config.hidden_dim == 5120 and moe_config.experts_per_token == 6
            parallel = moe_config.moe_parallel_config
            assert parallel.tp_size == 8 and parallel.dp_size == 1
            assert not parallel.use_ep and not parallel.enable_eplb
            from vllm.model_executor.layers.fused_moe.experts.sm121_target_m32 import get_target_m32_kernel
            self._dsv41_target_m32 = get_target_m32_kernel()

        # w6 drafter-m32 (DSV41_DRAFT_MOE_M32, default off; see experts/dsv41_draft_m32.py): the DSpark
        # drafter MoE (E=128, top-3, padded FC1 N=768) shares the target's M32 kernel object. Unset/0 runs
        # none of this. _dsv41_force_m32 is set only while the verify shadow re-runs apply().
        self._dsv41_force_m32 = False
        self._dsv41_draft_verify = None
        if os.environ.get("DSV41_DRAFT_MOE_M32", "0").strip().lower() not in ("", "0", "off", "false"):
            from vllm.model_executor.layers.fused_moe.experts import dsv41_draft_m32 as drm32

            drm32.max_m()  # validate the cap at startup
            if drm32.config_eligible(moe_config):
                from vllm.model_executor.layers.fused_moe.experts.sm121_target_m32 import get_target_m32_kernel
                self._dsv41_target_m32 = get_target_m32_kernel()
                if drm32.mode() == "verify":
                    self._dsv41_draft_verify = drm32.get_verifier()
                drm32.announce("shadow-verified (served: 128-row)" if drm32.mode() == "verify" else "served")

        # DSV41_MOE_LIVE_ACT (k1 exp 2, default off): run the fused SiLU-mul-
        # FP8-quant only over live rows (expert_ends from ep_scatter) when the
        # aligned workspace has DSV41_MOE_LIVE_ACT_MIN_ROWS <= M_sum <=
        # DSV41_MOE_LIVE_ACT_MAX_ROWS (0 = no upper bound) rows. Live rows are
        # bitwise unchanged (same BM/pack config, chosen from M_sum); padding
        # rows of the FC2 input are left unwritten (ep_gather never reads
        # their FC2 output rows).
        live = os.environ.get("DSV41_MOE_LIVE_ACT") == "1"
        self._dsv41_live_act_min_rows = (
            int(os.environ.get("DSV41_MOE_LIVE_ACT_MIN_ROWS", "1024")) if live else None
        )
        self._dsv41_live_act_max_rows = (
            int(os.environ.get("DSV41_MOE_LIVE_ACT_MAX_ROWS", "0")) if live else 0
        )

    def _dsv41_live_act(self, M_sum: int, activation: MoEActivation) -> bool:
        return (
            self._dsv41_live_act_min_rows is not None
            and M_sum >= self._dsv41_live_act_min_rows
            and (self._dsv41_live_act_max_rows <= 0 or M_sum <= self._dsv41_live_act_max_rows)
            and activation == MoEActivation.SILU
            and DeepGemmQuantScaleFMT.from_oracle() == DeepGemmQuantScaleFMT.UE8M0
        )

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        from vllm.platforms import current_platform

        return is_deep_gemm_supported() and (
            current_platform.is_device_capability_family(100)
            or current_platform.is_device_capability_family(120)
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (kMxfp4Static, kFp8Dynamic128Sym),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        # SILU has fused gate+mul+quant kernels; SWIGLUSTEP/SITU take the
        # general path (activation applied via self.activation, which forwards
        # the situ betas, then FP8 requant).
        return activation in [
            MoEActivation.SILU,
            MoEActivation.SWIGLUSTEP,
            MoEActivation.SITU,
        ]

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return not (
            moe_parallel_config.use_fi_nvl_two_sided_kernels
            or moe_parallel_config.use_fi_nvl_one_sided_kernels
        )

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    @staticmethod
    def _dsv41_compact_fc1(N: int, K: int, topk: int, experts: int) -> bool:
        # DSV41_MOE_COMPACT: target FC1 stored unpadded as [gate288 | up288]
        # (N=576) instead of [gate288+pad96 | up288+pad96] (N=768). FC2 stays
        # K=384 (packed-FP4 TMA needs K % 128 == 0); the activation kernel
        # writes the 96 zero columns that the padded FC1 used to produce.
        return (
            os.environ.get("DSV41_MOE_COMPACT") == "1"
            and current_platform.is_device_capability(121)
            and (N, K, topk, experts) == (576, 5120, 6, 384)
        )

    @staticmethod
    def _dsv41_target_geometry(N: int, K: int, topk: int, experts: int) -> bool:
        return (N, K, topk, experts) == (
            768, 5120, 6, 384
        ) or DeepGemmFP4Experts._dsv41_compact_fc1(N, K, topk, experts)

    @staticmethod
    def _workspace_m_alignment(M: int, N: int, K: int, topk: int, experts: int) -> int:
        alignment = get_mk_alignment_for_contiguous_layout()[0]
        if (
            os.environ.get("DSV41_SM121_TARGET_M32") == "1"
            and current_platform.is_device_capability(121)
            and 1 <= M <= int(os.environ.get("DSV41_TARGET_M32_MAX_M", "48"))
            and DeepGemmFP4Experts._dsv41_target_geometry(N, K, topk, experts)
        ):
            return min(alignment, 32)
        # w6 drafter-m32: DSpark drafter routed MoE (E=128 top-3, padded FC1 N=768) on the M32 native for
        # 1 <= M <= DSV41_DRAFT_M32_MAX_M (default 160). Mode "verify" stays 128 (served = original path).
        if (
            os.environ.get("DSV41_DRAFT_MOE_M32", "0").strip().lower() in ("1", "on", "true")
            and os.environ.get("DSV41_SM121_TARGET_M32") == "1"
            and current_platform.is_device_capability(121)
            and (N, K, topk, experts) == (768, 5120, 3, 128)
            and 1 <= M <= int(os.environ.get("DSV41_DRAFT_M32_MAX_M", "160"))
        ):
            return min(alignment, 32)
        # Qualified only for the TP8 V4.1 target; the 128-expert DSpark model
        # and prefill batches above 48 tokens keep their existing layout.
        if (
            os.environ.get("DSV41_SM121_TARGET_M64") == "1"
            and current_platform.is_device_capability(121)
            and 1 <= M <= int(os.environ.get("DSV41_TARGET_M64_MAX_M", "48"))
            and DeepGemmFP4Experts._dsv41_target_geometry(N, K, topk, experts)
        ):
            return min(alignment, 64)
        return alignment

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        block_m = self._workspace_m_alignment(M, N, K, topk, local_num_experts)
        M_sum, align_used = compute_aligned_M_and_alignment(
            M, topk, local_num_experts, block_m, expert_tokens_meta
        )
        assert M_sum % align_used == 0

        activation_out_dim = self.adjust_N_for_activation(N, activation)
        # workspace1 holds the permuted and requantized FP8 activations. The
        # workspace manager allocates it in the model activation dtype, so
        # account for the dtype sizes rather than overallocating BF16/FP16.
        workspace1 = _fp8_workspace_shape(
            M_sum,
            max(activation_out_dim, K),
            self.workspace_dtype(self.moe_config.in_dtype),
        )
        workspace2 = (M_sum, max(N, K))
        output = (M, K)
        return (workspace1, workspace2, output)

    def _act_mul_quant(
        self,
        input: torch.Tensor,
        output: torch.Tensor,
        activation: MoEActivation,
        expert_ends: torch.Tensor | None = None,
        expert_alignment: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        block_k = self._ACT_BLOCK_K
        scale_fmt = DeepGemmQuantScaleFMT.from_oracle()

        M_sum, N = input.size()
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        if expert_ends is not None:
            # DSV41_MOE_LIVE_ACT: only the two fused UE8M0 SILU launches below.
            assert activation == MoEActivation.SILU
            assert scale_fmt == DeepGemmQuantScaleFMT.UE8M0
            assert expert_alignment > 0

        if output.size(-1) != activation_out_dim:
            # DSV41_MOE_COMPACT only: logical width 288 from FC1 N=576, written
            # into the 384-wide FP8 buffer FC2 consumes (cols 288..383 = 0,
            # same 128-col UE8M0 scale groups as the padded layout).
            assert os.environ.get("DSV41_MOE_COMPACT") == "1"
            assert current_platform.is_device_capability(121)
            assert (N, output.size(-1), block_k) == (576, 384, 128)
            assert activation == MoEActivation.SILU
            assert scale_fmt == DeepGemmQuantScaleFMT.UE8M0
            return fused_silu_mul_fp8_quant_packed(
                input=input,
                output_q=output,
                group_size=block_k,
                clamp_limit=self.gemm1_clamp_limit,
                padded_output_size=384,
                expert_ends=expert_ends,
                expert_alignment=expert_alignment,
            )

        if activation == MoEActivation.SILU:
            # Fused gate+mul+quant kernels for the common SILU case.
            if scale_fmt == DeepGemmQuantScaleFMT.UE8M0:
                return fused_silu_mul_fp8_quant_packed(
                    input=input,
                    output_q=output,
                    group_size=block_k,
                    clamp_limit=self.gemm1_clamp_limit,
                    expert_ends=expert_ends,
                    expert_alignment=expert_alignment,
                )
            use_ue8m0 = scale_fmt == DeepGemmQuantScaleFMT.FLOAT32_CEIL_UE8M0
            return silu_mul_per_token_group_quant_fp8_colmajor(
                input=input,
                output=output,
                use_ue8m0=use_ue8m0,
                clamp_limit=self.gemm1_clamp_limit,
            )

        # General gated activations (SWIGLUSTEP, SITU): apply the activation
        # (self.activation forwards the situ betas from moe_config) then
        # FP8-requant into the layout DeepGEMM expects for this scale format.
        act_out = torch.empty(
            (M_sum, activation_out_dim), dtype=input.dtype, device=input.device
        )
        self.activation(activation, act_out, input)
        if scale_fmt == DeepGemmQuantScaleFMT.UE8M0:
            return per_token_group_quant_fp8_packed_for_deepgemm(
                act_out, block_k, use_ue8m0=True, out_q=output
            )
        use_ue8m0 = scale_fmt == DeepGemmQuantScaleFMT.FLOAT32_CEIL_UE8M0
        return per_token_group_quant_fp8(
            act_out,
            block_k,
            column_major_scales=True,
            out_q=output,
            use_ue8m0=use_ue8m0,
        )

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        assert a1q_scale is not None
        assert a2_scale is None
        assert self.w1_scale is not None
        assert self.w2_scale is not None

        dsv41_call = None
        if self._dsv41_draft_verify is not None and not self._dsv41_force_m32:
            # w6 drafter-m32 verify: the original arguments, for the 32-row shadow re-run below.
            dsv41_call = dict(hidden_states=hidden_states, w1=w1, w2=w2, topk_weights=topk_weights,
                              topk_ids=topk_ids, activation=activation, global_num_experts=global_num_experts,
                              expert_map=expert_map, a1q_scale=a1q_scale, a2_scale=a2_scale,
                              expert_tokens_meta=expert_tokens_meta,
                              apply_router_weight_on_input=apply_router_weight_on_input)

        a1q = hidden_states
        _, N, _ = w1.size()
        # K comes from activations (full hidden dim), not from w1 which is
        # packed FP4 (E, N, K//2).
        K = a1q.size(1)

        local_num_experts = w1.size(0)
        if global_num_experts == -1:
            global_num_experts = local_num_experts

        compact_fc1 = self._dsv41_compact_fc1(
            N, K, topk_ids.size(1), local_num_experts
        )
        if compact_fc1:
            assert tuple(w2.shape) == (384, 5120, 192)
            assert activation == MoEActivation.SILU

        block_m = 32 if self._dsv41_force_m32 else self._workspace_m_alignment(
            topk_ids.size(0), N, K, topk_ids.size(1), local_num_experts
        )
        M_sum, _ = compute_aligned_M_and_alignment(
            M=topk_ids.size(0),
            num_topk=topk_ids.size(1),
            local_num_experts=local_num_experts,
            alignment=block_m,
            expert_tokens_meta=expert_tokens_meta,
        )

        a1q_perm = _resize_cache(
            workspace13.view(dtype=torch.float8_e4m3fn), (M_sum, K)
        )
        live_act = self._dsv41_live_act(M_sum, activation)
        permuted = deepgemm_moe_permute(
            aq=a1q,
            aq_scale=a1q_scale,
            topk_ids=topk_ids,
            local_num_experts=local_num_experts,
            expert_map=expert_map,
            expert_tokens_meta=expert_tokens_meta,
            aq_out=a1q_perm,
            alignment_m=block_m,
            **({"return_expert_ends": True} if live_act else {}),
        )
        a1q, a1q_scale, expert_ids, inv_perm, align_used = permuted[:5]
        expert_ends = permuted[5] if live_act else None
        assert a1q.size(0) == M_sum

        # Cap DG's BLOCK_M heuristic at the workspace's per-expert alignment;
        # see DeepGemmExperts.apply for rationale.
        target_m32 = self._dsv41_target_m32 if align_used == 32 else None
        assert align_used != 32 or target_m32 is not None
        grouped_gemm = target_m32.gemm if target_m32 is not None else m_grouped_fp8_fp4_gemm_nt_contiguous
        alignment_scope = target_m32.alignment_scope() if target_m32 is not None else mk_alignment_scope(align_used)
        with alignment_scope:
            # FC1: FP8 activations x FP4 weights
            # DeepGEMM 2.4.2 requires FP4-packed weights as int8 (kPackedFP4).
            mm1_out = _resize_cache(workspace2, (M_sum, N))
            grouped_gemm(
                (a1q, a1q_scale),
                (w1.view(torch.int8), self.w1_scale),
                mm1_out,
                expert_ids,
                recipe_a=(1, self._ACT_BLOCK_K),
                recipe_b=(1, self._WEIGHT_BLOCK_K),
            )

            # SwiGLU activation + FP8 requant
            # (compact FC1: two logical widths of 288 -> FC2 input stays 384 wide)
            activation_out_dim = (
                384 if compact_fc1 else self.adjust_N_for_activation(N, activation)
            )
            quant_out = _resize_cache(
                workspace13.view(dtype=torch.float8_e4m3fn), (M_sum, activation_out_dim)
            )
            a2q, a2q_scale = self._act_mul_quant(
                input=mm1_out.view(-1, N),
                output=quant_out,
                activation=activation,
                expert_ends=expert_ends,
                expert_alignment=align_used if live_act else 0,
            )

            # FC2: FP8 activations x FP4 weights
            mm2_out = _resize_cache(workspace2, (M_sum, K))
            grouped_gemm(
                (a2q, a2q_scale),
                (w2.view(torch.int8), self.w2_scale),
                mm2_out,
                expert_ids,
                recipe_a=(1, self._ACT_BLOCK_K),
                recipe_b=(1, self._WEIGHT_BLOCK_K),
            )

        if apply_router_weight_on_input:
            topk_weights = torch.ones_like(topk_weights)

        deepgemm_unpermute_and_reduce(
            a=mm2_out,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            inv_perm=inv_perm,
            expert_map=expert_map,
            output=output,
        )
        if dsv41_call is not None:
            self._dsv41_draft_verify.shadow(self, output, dsv41_call)
