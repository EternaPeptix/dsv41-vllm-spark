# Candidate-only scoring: independent native-path review

The proposed first arm changes which K rows are scored, while retaining the original-width logits and original top-k call. This is a useful qualification candidate, not a claim of GPU equivalence or a measured speedup.

## Native compatibility

Reviewed `work/deepgemm-source/csrc/apis/attention.hpp:92–223` and `csrc/jit_kernels/impls/sm120_mqa_logits.hpp:68–160`, plus the captured SM120 FP8 kernel. The local kernel header matches the captured header byte for byte (SHA256 `6f6dbf1158c2e5571bb90835a673f74cfc64f4c28e22a0f3f77ce73c554d702c`). The host sources are local build-source evidence; the GPU test must verify the installed binary accepts the compressed-output argument.

- FP8 inputs use FP32 K scales and FP32 weights, with no Q-scale tensor. The existing sparse FP8/FP4 interface is not a drop-in replacement: its weight/output precision differs.
- The dense native API accepts `max_seqlen_k=16384` with `clean_logits=False`, producing an `[M,16384]` result. The existing vLLM wrapper needs to forward that optional argument.
- The native query tile remains `BLOCK_Q=128/num_heads`. The historical profile uses 32 heads, so `BLOCK_Q=4`. Giving each query a disjoint 16,384-position interval causes each query tile to score the union of its four intervals. This computes four times the useful candidate scores; performance estimates must include it. The 64-head case would have a factor of two.
- Both paths retain the same four `mma.sync.m16n8k32` steps for head dimension 128; the same head-tile accumulation; multiplication by K scale; and XOR-shuffle additions at offsets 1 then 2. There is no reduction across K positions. The compressed specialization changes the output bounds and address.
- Split-KV partitions output positions without combining partial scores, so a different split count does not introduce a cross-partition floating-point reduction.

This source argument supports testing exactness. It does not establish compiled bitwise equivalence: register allocation and compiler transformations can differ between specializations.

## Required qualification observations

Compare original dense-plus-mask logits against packed-score-plus-scatter logits bitwise, then compare the original top-k and CANON results. Cover nonaligned original starts, odd query counts and tile boundaries, partial final candidate blocks, permuted candidate lists, repeated scores, negative/invalid candidate entries, and IEEE edge values. Compare CUDA graph replays after changing device input contents, not just repeating one captured input.

Retain the original dense allocation stride as well as logical width for top-k. Preserve the mask's out-of-range candidate clamp behavior, or explicitly reject such inputs. Bound packed workspace by replay-row count before allocating; 128 rows require about 264 MiB for packed FP8 K and FP32 scales alone. The retained dense logits, gather buffers, and captured graph pools are additional memory.

Any saved timing must include packing, scoring, initialization/scattering, and the unchanged top-k. Separately report component timings to identify whether pack bandwidth or dense output initialization absorbs the scoring saving. E7 and candidate-only need a four-arm comparison (neither, each alone, both); do not add their isolated savings.

No GPU or cluster execution was performed for this review.
