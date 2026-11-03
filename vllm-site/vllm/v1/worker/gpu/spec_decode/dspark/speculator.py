# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark speculator: semi-autoregressive parallel drafting.

DSpark drafts a block of ``num_speculative_tokens`` tokens in one parallel pass
(reusing the DFlash machinery: context-KV precompute + a query-block forward),
then injects intra-block dependency with a lightweight sequential Markov head.

Differences from DFlash:
  * Anchor-as-first-prediction: each request emits exactly ``N =
    num_speculative_tokens`` query tokens (anchor + N-1 noise), NOT ``1 + N``.
    Every query position is a prediction (the anchor predicts the first draft
    token), so we sample at all N positions and ``sample_pos = query_pos + 1``
    (standard next-token), whereas DFlash's masks sit AT the predicted position.
    This is the ``sample_from_anchor`` path in the shared prepare-inputs kernel.
    Speculators-format checkpoints instead use the DFlash ``1 + N`` fill-in
    layout (anchor is the bonus token).
  * Sequential Markov sampling: instead of DFlash's single parallel sample, we
    sample left-to-right, adding a prefix-dependent Markov bias derived from the
    previously sampled token at each step.

CUDA graphs (FULL, mirroring DFlash) cover the whole draft step: the parallel
backbone forward AND the sequential Markov sampling.
"""

import os
import types
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import tensor_model_parallel_all_gather
from vllm.logger import init_logger
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

# g4p E7X (default off; see vllm/models/deepseek_v41/decoder_replay_layers.py).
_DSV41_E7X = os.environ.get("DSV41_E7_EXACT", "0").strip().lower() in ("1", "post_and_draft")
from vllm.v1.worker.gpu.spec_decode.dspark.utils import load_dspark_model

logger = init_logger(__name__)

# DSV41 patch (r3 spec-sampling): the drafter's Gumbel noise is keyed by
# (seed, predicted position). Block verification inspects drafts beyond the
# accepted prefix, so re-drafting those positions in the next step with the
# same noise biases the proposal (lossy). DSV41_DRAFT_FRESH_NOISE=1 keys the
# draw by (anchor position, step) instead: key = 2**29 + 8 * anchor + step,
# unique per draft step, disjoint from target keys (< 2**30) and stock draft
# keys (2**30 + pos) after the in-kernel salt. Greedy (temp 0) draws are pure
# argmax and unaffected. Default 0 = stock keys.
_DSV41_DRAFT_FRESH_NOISE = os.environ.get("DSV41_DRAFT_FRESH_NOISE", "0") == "1"
_FRESH_NOISE_BASE = 1 << 29

# DSV41 patch (r4 greedy-sharded): with probabilistic drafting on
# (DSV41_SPEC_SAMPLED=1) the vocab-sharded greedy Markov path is disabled
# because sampled rows need full draft logits. DSV41_DSPARK_GREEDY_SHARDED=1
# builds the sharded state anyway, captures a SECOND set of FULL draft graphs
# with it, and replays that set (or runs it eagerly) for steps in which every
# request is greedy (temperature 0). Greedy rows never read draft_logits in the
# rejection sampler, and a temperature-0 Gumbel draw is a plain argmax, so the
# drafts equal the full-vocab path (same argmax up to GEMM rounding/ties, as
# for the r3-e1 sharded drafter). Mixed or sampled batches use the stock
# probabilistic graphs. Default 0 = unchanged.
_DSV41_GREEDY_SHARDED = os.environ.get("DSV41_DSPARK_GREEDY_SHARDED", "0") == "1"
# g130 boundary, default off: unset = dsv41_boundary.py is never imported.
_DSV41_MK_FOLD = os.environ.get("DSV41_MK_FOLD", "0").strip().lower() not in ("", "0", "off")
_DSV41_CTXKV_FOLD = os.environ.get("DSV41_CTXKV_FOLD", "0").strip().lower() not in ("", "0", "off")


class DSparkSpeculator(DFlashSpeculator):
    _speculator_name = "DSpark"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

        # Whether to sample from the anchor position. When True, uses anchor-as-first
        # (N slots, each position predicts the next token). When False, uses 1+N
        # fill-in block (anchor is a bonus token).
        self.sample_from_anchor = getattr(
            self.draft_model_config.hf_config, "sample_from_anchor", True
        )
        if self.sample_from_anchor:
            self.num_query_per_req = self.num_speculative_steps
        else:
            self.num_query_per_req = 1 + self.num_speculative_steps

        # DSpark consumes mean-pooled target aux hidden states at the target
        # layers, combined to hidden_size via main_proj. Store that combined
        # main_x (hidden_size wide). DSpark does not use the same pre-allocated buffer
        # that DeepSeek-V4's MTP uses.
        draft_hidden = self.draft_model_config.get_hidden_size()
        self.hidden_states = torch.zeros(
            self.max_num_tokens, draft_hidden, dtype=self.dtype, device=device
        )

        self._step_cols = torch.arange(
            self.num_speculative_steps, dtype=torch.int32, device=device
        )

        self._anchor_idx = (
            torch.arange(self.max_num_reqs, dtype=torch.int64, device=device)
            * self.num_query_per_req
        )

        # Reduced-vocab probabilistic drafting only; set in load_draft_model.
        self._d2t_scatter_index: torch.Tensor | None = None
        self._draft_scatter_buf: torch.Tensor | None = None
        self._draft_topk: int | None = getattr(
            self.draft_model_config.hf_config, "dspark_draft_topk", None
        )

        self.use_confidence_head: bool = False
        # DSV41_DSPARK_SHARDED_MARKOV state (None = stock path).
        self._sharded_markov: dict[str, Any] | None = None
        # DSV41_DSPARK_GREEDY_SHARDED state: sharded Markov dict used only for
        # all-greedy steps, its graph manager, and the CPU temperature array.
        self._sharded_markov_greedy: dict[str, Any] | None = None
        self.query_cudagraph_manager_greedy = None
        self._dsv41_temp_np = None
        self._dsv41_greedy_steps = 0
        self._dsv41_mixed_steps = 0

    def load_draft_model(
        self,
        target_model: torch.nn.Module,
        target_attn_layer_names: set[str],
    ) -> torch.nn.Module:
        model = load_dspark_model(target_model, self.vllm_config)
        # Reduced draft vocab: probabilistic rejection sampling indexes draft
        # logits by target id, so precompute the draft->target column map and a
        # scratch buffer to scatter logits into target vocab before sampling.
        if self.draft_logits is not None and model.draft_id_to_target_id is not None:
            d2t = model.draft_id_to_target_id
            self._d2t_scatter_index = (
                torch.arange(d2t.shape[0], device=d2t.device) + d2t
            )
            # -inf once; the per-step scatter overwrites the draft->target
            # columns. Kept separate from draft_logits to avoid aliasing.
            self._draft_scatter_buf = torch.full(
                (self.max_num_reqs, self.vocab_size),
                float("-inf"),
                dtype=self.draft_logits.dtype,
                device=self.device,
            )
        self.use_confidence_head = (
            self.enable_adaptive_verification
            and model.model.confidence_head is not None
        )
        if self.use_confidence_head:
            # The acceptance estimator is not needed when a trained confidence head
            # is available.
            self.use_acceptance_estimator = False
        self._sharded_markov = self._maybe_init_sharded_markov(model)
        return model

    def _maybe_init_sharded_markov(self, model: torch.nn.Module) -> dict | None:
        """DSV41 patch: vocab-parallel Markov bias + distributed argmax.

        Stock DSpark multiplies the replicated [V, r] Markov projection on every
        rank for every draft position (5 x 66 MB bf16 reads per step) and
        all-gathers full-vocab base logits. With greedy drafting only the
        argmax is needed, so each rank adds the bias to its own lm_head vocab
        shard and the ranks exchange (max, id) pairs per position.
        """
        if os.environ.get("DSV41_DSPARK_SHARDED_MARKOV", "0") != "1":
            return None
        why = []
        if type(model).__name__ != "DSparkDeepseekV4ForCausalLM":
            why.append(f"model {type(model).__name__}")
        # r4 greedy-sharded: probabilistic drafting alone does not block the
        # greedy-only graph set; any other reason does.
        greedy_only = _DSV41_GREEDY_SHARDED and self.draft_logits is not None
        if self.draft_logits is not None and not greedy_only:
            why.append("probabilistic drafting needs full logits")
        if self._draft_topk is not None:
            why.append("dspark_draft_topk set")
        if self.draft_watermarker is not None:
            why.append("draft watermarker")
        if self.use_acceptance_estimator:
            why.append("online acceptance estimator needs full logits")
        if getattr(model, "draft_id_to_target_id", None) is not None:
            why.append("reduced draft vocab")
        lm_head = getattr(model, "lm_head", None)
        inner = getattr(model, "model", None)
        markov = getattr(inner, "markov_head", None)
        si = getattr(lm_head, "shard_indices", None)
        if lm_head is None or markov is None or si is None:
            why.append("missing lm_head/markov_head/shard_indices")
        elif lm_head.tp_size <= 1:
            why.append("lm_head not vocab-parallel")
        elif si.num_added_elements != 0:
            why.append("added vocab")
        if why:
            logger.warning(
                "DSV41_DSPARK_SHARDED_MARKOV=1 ignored: %s", "; ".join(why)
            )
            if greedy_only:
                logger.warning(
                    "DSV41_DSPARK_GREEDY_SHARDED=1 ignored (sharded state unavailable)"
                )
            return None
        w2_layer = markov.markov_w2
        w2 = w2_layer.weight
        start, end = si.org_vocab_start_index, si.org_vocab_end_index
        n_local = lm_head.weight.shape[0]  # org rows + trailing padding
        num_pad = si.num_org_vocab_padding
        assert end - start + num_pad == n_local, (start, end, num_pad, n_local)
        assert w2.shape[0] >= end, (tuple(w2.shape), end)
        if num_pad == 0:
            w2_local = w2[start:end]  # contiguous row view, no copy
        else:
            w2_local = torch.zeros(
                n_local, w2.shape[1], dtype=w2.dtype, device=w2.device
            )
            w2_local[: end - start].copy_(w2[start:end])
        shim = types.SimpleNamespace(
            weight=w2_local, quant_method=w2_layer.quant_method, tp_size=1
        )
        logger.info(
            "DSV41_DSPARK_SHARDED_MARKOV: vocab shard [%d, %d) pad %d, "
            "markov rows/rank %d of %d, tp %d",
            start, end, num_pad, n_local, w2.shape[0], lm_head.tp_size,
        )
        state = {
            "shim": shim,
            "vocab_start": start,
            "num_pad": num_pad,
            "tp": lm_head.tp_size,
        }
        if greedy_only:
            # Probabilistic drafting stays the default path; the sharded state
            # is used only for all-greedy steps (see propose()).
            self._sharded_markov_greedy = state
            logger.warning(
                "DSV41_DSPARK_GREEDY_SHARDED active: probabilistic drafting for "
                "mixed/sampled steps, sharded greedy Markov graphs for all-greedy steps"
            )
            return None
        return state

    # ---- r4 greedy-sharded: second graph set + per-step dispatch ----------

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        super().init_cudagraph_manager(cudagraph_mode)
        if self._sharded_markov_greedy is None:
            self.query_cudagraph_manager_greedy = None
            return
        stock = self.query_cudagraph_manager
        super().init_cudagraph_manager(cudagraph_mode)
        self.query_cudagraph_manager_greedy = self.query_cudagraph_manager
        self.query_cudagraph_manager = stock

    def capture(self) -> None:
        if _DSV41_MK_FOLD or _DSV41_CTXKV_FOLD:
            from vllm.models.deepseek_v41.nvidia.dsv41_boundary import prebuild

            prebuild(self)
        super().capture()
        if (
            self._sharded_markov_greedy is None
            or self.query_cudagraph_manager_greedy is None
        ):
            return
        stock_mgr, stock_sm = self.query_cudagraph_manager, self._sharded_markov
        self.query_cudagraph_manager = self.query_cudagraph_manager_greedy
        self._sharded_markov = self._sharded_markov_greedy
        try:
            logger.info("DSV41_DSPARK_GREEDY_SHARDED: capturing greedy draft graphs")
            super().capture()
        finally:
            self.query_cudagraph_manager = stock_mgr
            self._sharded_markov = stock_sm

    def _dsv41_all_greedy(self, input_batch) -> bool:
        """CPU-only check, identical on every TP rank (replicated scheduler state)."""
        if self._dsv41_temp_np is None:
            # One-time lookup of the runner's CPU temperature mirror from the
            # caller frame (GPUModelRunner.propose_draft / sample_tokens).
            import sys

            f = sys._getframe(2)
            runner = None
            for _ in range(4):
                if f is None:
                    break
                cand = f.f_locals.get("self")
                if getattr(cand, "speculator", None) is self:
                    runner = cand
                    break
                f = f.f_back
            try:
                self._dsv41_temp_np = runner.sampler.sampling_states.temperature.np
            except Exception as e:  # noqa: BLE001 - fail closed
                logger.warning(
                    "DSV41_DSPARK_GREEDY_SHARDED: no CPU temperature mirror (%r); "
                    "disabled", e,
                )
                self._sharded_markov_greedy = None
                self.query_cudagraph_manager_greedy = None
                return False
        idx = getattr(input_batch, "idx_mapping_np", None)
        n = input_batch.num_reqs
        if idx is None or n <= 0:
            return False
        return bool((self._dsv41_temp_np[idx[:n]] == 0.0).all())

    def propose(self, input_batch, *args, **kwargs):
        # g4p E7X: a step whose every request is mid-prompt (neither F nor F-1); its
        # drafts are discarded and nothing reads its context KV or confidences.
        if _DSV41_E7X and not kwargs.get("dummy_run", False):
            from vllm.models.deepseek_v41.decoder_replay_layers import e7x_step_skipped

            if e7x_step_skipped(input_batch):
                return self.draft_tokens[: input_batch.num_reqs]
        if (
            self.query_cudagraph_manager_greedy is None
            or kwargs.get("dummy_run", False)
            or kwargs.get("is_profile", False)
            or not self._dsv41_all_greedy(input_batch)
        ):
            if self.query_cudagraph_manager_greedy is not None:
                self._dsv41_mixed_steps += 1
            return super().propose(input_batch, *args, **kwargs)
        self._dsv41_greedy_steps += 1
        if _DSV41_MK_FOLD or _DSV41_CTXKV_FOLD:
            from vllm.models.deepseek_v41.nvidia.dsv41_boundary import audit_tick

            audit_tick(self._dsv41_greedy_steps)
        if self._dsv41_greedy_steps in (1, 1000, 100000):
            logger.info(
                "DSV41_DSPARK_GREEDY_SHARDED: greedy steps %d, other steps %d",
                self._dsv41_greedy_steps, self._dsv41_mixed_steps,
            )
        stock_mgr, stock_sm = self.query_cudagraph_manager, self._sharded_markov
        self.query_cudagraph_manager = self.query_cudagraph_manager_greedy
        self._sharded_markov = self._sharded_markov_greedy
        try:
            return super().propose(input_batch, *args, **kwargs)
        finally:
            self.query_cudagraph_manager = stock_mgr
            self._sharded_markov = stock_sm

    def _sample_logits(
        self,
        logits: torch.Tensor,
        idx_map: torch.Tensor,
        sample_pos: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        self._maybe_predict_acceptance(logits, idx_map, self._step_cols[step])
        if self.draft_logits is None:
            draft_ids = logits.argmax(dim=-1)
            return self.model.map_draft_to_target(draft_ids)

        # Probabilistic sampling and rejection operate in target-vocabulary
        # space. A reduced draft vocabulary is scattered into its target rows.
        if self._d2t_scatter_index is not None:
            assert self._draft_scatter_buf is not None
            buf = self._draft_scatter_buf[: logits.shape[0]]
            buf.index_copy_(1, self._d2t_scatter_index, logits.to(buf.dtype))
            logits = buf

        # sample_pos is the predicted token's position P. Sampling keys a draw
        # by the position before the sampled token, P-1.
        sampler = (
            gumbel_sample
            if self.draft_watermarker is None
            else self.draft_watermarker.sample
        )
        return sampler(
            logits,
            idx_map,
            self.temperature,
            self.seeds,
            sample_pos - 1,
            apply_temperature=True,
            is_drafting=True,
            logits_cache=self.draft_logits,
            logits_cache_col=self._step_cols[step],
            use_fp64=self.use_fp64_gumbel,
        )

    def _sample_sequential(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        if self._draft_topk is not None:
            self._sample_sequential_topk(num_reqs, head_hidden)
            return
        if self._sharded_markov is not None:
            self._sample_sequential_sharded(num_reqs, head_hidden)
            return

        # Sequential Markov sampling over the backbone's output hidden states.
        n_spec = self.num_speculative_steps
        num_sample = num_reqs * n_spec
        # Per-(req, position) head hidden, ordered (req, step).
        sample_hidden = head_hidden[self.sample_indices[:num_sample]]
        # Draft-vocab logits; sampled ids are remapped to target vocab below.
        base_logits = self.model.compute_draft_logits(sample_hidden)
        vocab_size = base_logits.shape[-1]
        base_logits = base_logits.view(num_reqs, n_spec, vocab_size)

        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)
        if _DSV41_DRAFT_FRESH_NOISE and self.draft_logits is not None:
            assert n_spec <= 8
            # _sample_logits subtracts 1 from the key it is given.
            sample_pos = (
                (sample_pos[:, :1] - 1) * 8
                + self._step_cols.to(sample_pos.dtype).unsqueeze(0)
                + (_FRESH_NOISE_BASE + 1)
            )
        confidence_markov_embeds = []

        # Anchor (bonus) token per request = the input id at query offset 0,
        # read via the precomputed persistent index (fixed buffer for capture).
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]

        for i in range(n_spec):
            # Sequential stage: Markov bias from the previously sampled token.
            markov_embed = self.model.markov_embed(prev)
            if self.use_confidence_head:
                confidence_markov_embeds.append(markov_embed)
            bias = self.model.markov_bias(markov_embed)
            logits_i = base_logits[:, i] + bias
            draft_sampled_i = self._sample_logits(
                logits_i, idx_map[:, i], sample_pos[:, i], i
            )
            self.draft_tokens[:num_reqs, i] = draft_sampled_i
            prev = draft_sampled_i

        if self.use_confidence_head:
            confidence = self.model.compute_confidence(
                sample_hidden,
                torch.stack(confidence_markov_embeds, dim=1).flatten(0, 1),
            )
            self.draft_token_confidence_probs[:num_reqs] = confidence.view(
                num_reqs, n_spec
            )

    def _sample_sequential_sharded(
        self, num_reqs: int, head_hidden: torch.Tensor
    ) -> None:
        """Greedy sequential Markov drafting on this rank's vocab shard.

        Same tokens as the stock path (argmax of base + bias over the full
        vocab) up to GEMM rounding/ties; ties resolve to the lowest global id.
        """
        st = self._sharded_markov
        assert st is not None
        if num_reqs == 0:
            return
        n_spec = self.num_speculative_steps
        num_sample = num_reqs * n_spec
        sample_hidden = head_hidden[self.sample_indices[:num_sample]]
        lp = self.model.logits_processor
        # Local vocab shard of the base logits; no all-gather.
        base = lp(
            self.model.lm_head,
            self.model.model.norm(sample_hidden),
            skip_gather=True,
        )
        num_pad = st["num_pad"]
        if num_pad > 0:
            base[..., -num_pad:] = float("-inf")
        base = base.view(num_reqs, n_spec, -1)
        tp = st["tp"]
        vocab_start = st["vocab_start"]
        confidence_markov_embeds = []
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]
        _mk_done = False
        if _DSV41_MK_FOLD:
            from vllm.models.deepseek_v41.nvidia.dsv41_boundary import markov_fold_loop

            _mk_done = markov_fold_loop(
                self, st, lp, base, prev, num_reqs, n_spec,
                confidence_markov_embeds, tensor_model_parallel_all_gather,
            )
        for i in range(0 if _mk_done else n_spec):
            markov_embed = self.model.markov_embed(prev)
            if self.use_confidence_head:
                confidence_markov_embeds.append(markov_embed)
            bias = lp(st["shim"], markov_embed, skip_gather=True)
            logits_i = base[:, i] + bias
            vals, idx = logits_i.max(dim=-1)
            vals = vals.float()
            zero = torch.zeros_like(vals)
            # 16 B per row so the RDMA all-gather path accepts it.
            pair = torch.stack(
                [vals, (idx + vocab_start).float(), zero, zero], dim=-1
            )
            gathered = tensor_model_parallel_all_gather(pair, dim=0)
            gathered = gathered.view(tp, num_reqs, 4)
            best = gathered[:, :, 0].argmax(dim=0, keepdim=True)
            tok = gathered[:, :, 1].gather(0, best).squeeze(0).to(torch.int64)
            self.draft_tokens[:num_reqs, i] = tok
            prev = tok

        if self.use_confidence_head:
            confidence = self.model.compute_confidence(
                sample_hidden,
                torch.stack(confidence_markov_embeds, dim=1).flatten(0, 1),
            )
            self.draft_token_confidence_probs[:num_reqs] = confidence.view(
                num_reqs, n_spec
            )

    def _sample_sequential_topk(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        """Apply the sequential Markov head only to top-k base-logit candidates.

        Candidate selection is done once for all draft positions. At each
        sequential step, the selected logits are corrected in place and every
        other entry is set to ``-inf``. The normal dense sampling and rejection
        paths then consume that truncated distribution unchanged.
        """
        assert self._draft_topk is not None
        n_spec = self.num_speculative_steps
        num_sample = num_reqs * n_spec
        sample_hidden = head_hidden[self.sample_indices[:num_sample]]
        base_logits = self.model.compute_draft_logits(sample_hidden)
        base_logits = base_logits.view(num_reqs, n_spec, -1)
        base_values, draft_indices = base_logits.topk(self._draft_topk, dim=-1)
        # Reuse the dense backbone output as the normal sampler's input. Fill
        # once for all positions, then scatter only the corrected candidates
        # during the sequential loop.
        base_logits.fill_(float("-inf"))
        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)
        confidence_markov_embeds = []
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]

        for i in range(n_spec):
            markov_embed = self.model.markov_embed(prev)
            if self.use_confidence_head:
                confidence_markov_embeds.append(markov_embed)
            logits_i = self.model.apply_markov_bias_gathered(
                markov_embed,
                base_logits[:, i],
                base_values[:, i],
                draft_indices[:, i],
            )
            draft_sampled_i = self._sample_logits(
                logits_i, idx_map[:, i], sample_pos[:, i], i
            )
            self.draft_tokens[:num_reqs, i] = draft_sampled_i
            prev = draft_sampled_i

        if self.use_confidence_head:
            confidence = self.model.compute_confidence(
                sample_hidden,
                torch.stack(confidence_markov_embeds, dim=1).flatten(0, 1),
            )
            self.draft_token_confidence_probs[:num_reqs] = confidence.view(
                num_reqs, n_spec
            )

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        # Full draft step (captured under CUDA graph): parallel backbone forward
        # then sequential Markov sampling over its hidden state outputs.
        head_hidden = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        self._sample_sequential(num_reqs, head_hidden)
