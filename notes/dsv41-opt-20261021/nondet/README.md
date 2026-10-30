# 1M non-determinism bisect (2026-10-23)

All arms: DSV41_VERIFY_CAP=0 DSV41_ROUTE_LIVE=0 via ops/conc/launch-conc.sh (Engram punch check on).
ex3.py = exact.py x3 in one instance (same prompts and hash scheme) plus first divergent token; lx3.py/lp2.py = long-prompt probes.

## Cause
At max_model_len >= ~398K the decode CUDA graphs are captured with max_seq_len = max_model_len.
persistent_topk (csrc/topk.cu) needs ceil(width/chunk) co-resident CTAs, which exceeds GB10's 48 SMs, so it
falls back to top_k_per_row_decode. Rows of 200K+ columns take its 10-way split+merge, and the merge writes
indices via smem atomicAdd, so the index ORDER changes from run to run. The SET stays exact and the valid
prefix holds. Sparse MLA sums in stored order -> bitwise drift -> greedy flips at near-ties.
Benign numeric non-determinism, not a garbage read.

## Arms
| arm | config | 3 combined hashes | verdict |
|---|---|---|---|
| E0 | kernel test, no vLLM: persistent_topk L=300 x30 | order varies iff (W>=409600,R<=4) or (W=1M,R<=8); set exact, prefix ok everywhere | mechanism confirmed |
| A1 | 1M | cf6676 / 757df1 / c9821d (first div tok5-86; p1 stable) | reproduced |
| A2 | 1M + TOPK_CANON=1 | a8c3394af9d2 x3 (= the 262k anchor) | FIXED |
| A3 | 409600 SPEC=0 | 3 distinct | non-det just above threshold, spec off -> target path |
| A4 | 393216 SPEC=0 | 1 distinct | deterministic just below threshold |
| A5 | 1M CANON noPC | a8c3 per-prompt x3; long 3.8K stable, 18.7K 3 distinct | residual = long context |
| A6 | 1M CANON noPC SPEC0 | short stable; 18.7K/30K logprobs differ at step 0 (prefill); 3.8K/15K bit-exact | residual = long prefill |
| A7 | A6 + --max-num-batched-tokens 32768 | 15K, 18.7K bit-exact; 30.4K still differs at step 0 | chunked/sub-chunked prefill path |
| A8 | 262k CANON noPC (spec on) | 9330f03c4f97 x3 (stable; p2/p4 != anchor); 15K/18.7K logprobs differ from step 2 (decode), 30K from step 0 | long-ctx residual is not 1M-specific |
| A9 | 262k CANON (PC on) | a8c3394af9d2 x3 | the fix keeps the 262k anchor |

Fix: topk-canon.diff (patch_canon.py), sparse_attn_indexer.py 75d14da5e2b8 -> 7112c01dee55 on all 8 nodes,
original at *.bak-canon. Inert unless DSV41_TOPK_CANON=1. The live config was relaunched WITHOUT the flag.

## Verification + promotion (2026-10-24)
A/B (nondet/ab/, absuite.sh) passed the perf gates; DSV41_TOPK_CANON=1 promoted as launcher default (80c749ca), indexer
7112c01d live on 8/8. Details: ../README.md "1M determinism fix". (Note: the original indexer is at both .bak-nondet and .bak-canon.)
