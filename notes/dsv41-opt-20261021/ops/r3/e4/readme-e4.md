
## Round 3 / e4 dense-fix + sonar o_proj (2026-10-22) — PROMOTED
Change (all behind env flags, now launcher defaults):
- DSV41_QKV_A_SHARD=1: replicated fused wq_a|wkv GEMM (1792x5120 MXFP8, every rank) column-sharded to 224 rows/rank
  (r2 skinny_mx_ext split-K GEMV) + rdma_ar all-gather, for M<=8 (DSV41_QKV_A_SHARD_MAX_M); larger M = stock.
- DSV41_IDX_WPROJ=bmm32: indexer weights_proj (32x5120 bf16) as 32-way split-K bmm + fp32 sum (stock cuBLAS 33 us -> 8.5 us at M 2-8).
- DSV41_FP8_LMHEAD=draft: MXFP8 copy of the lm_head shard used only for DSpark draft logits (target logits stay bf16 -> lossless).
- DSV41_SONAR_OPROJ=fp8 SPLITS=32 MAX_M=32: sonar PR #1788 split-K W8A16 wo_a kernel on an exact MXFP8 re-encode of the bf16
  weight (bf16 rebuilt in-kernel); M>32 = torch.bmm (fp8 kernel loses above 32 rows).
Site files (backups .bak-r3dense / .bak-r3sonar on all 8):
  deepseek_v41/attention.py c39e06c1 -> f2f48198, model_executor/layers/logits_processor.py 35cf8235 -> caf6d216,
  deepseek_v41/nvidia/dspark.py 480149ff -> fa56d07d, new deepseek_v41/nvidia/dsv41_r3_dense.py 993023f2,
  deepseek_v4/nvidia/ops/o_proj.py f2893ee6 -> 34b9e1a4, new deepseek_v4/nvidia/ops/sonar_bf16_bmm.py 9f6f3f84.
Launcher 776de10b -> d5c41fd0 on all 8 (backup ~/dsv41-tp8.sh.bak-r3dense).

Kernel gates (spark-two, service stopped):
  qkv_a shard: replicated 44.2-44.8 us; skinny 10.9/11.8/13.3/16.1/17.7 us at M 1/2/4/6/8 (G1 <=14 us: miss at M=6 by 2 us,
  still fastest option); cutlass shard 17.4-17.8 us bitwise == replicated (G2 PASS); skinny relerr 0, deterministic (G3 PASS).
  weights_proj: stock 33.4 us (M>=2); mm32 35 (worse); bmm16 8.5-10.3; bmm32 8.0-8.7 (winner, G4 PASS).
  FP8 draft head a8 382-386 us vs bf16 719-729 us (G5 PASS); a16 fails (765 us at M=6).
  sonar o_proj: re-encode exact, deterministic, relerr 1.6-1.7e-3 (= torch.bmm); fp8k 23-24 us (s32) vs bmm 40-55 us at M<=16
  (>=1.5x at M<=8, PASS); fp8k vs bf16-kernel NOT bitwise at M 1/6/8 (same splits, same rebuilt bf16 weights -> two Triton
  compilations of the dot; relerr identical) - accepted, quality gated end-to-end below. Above M=32 fp8k is 0.58-0.74x -> MAX_M=32.
  all-gather (M,224) bf16: not directly benchable; nearest in-graph rdma AG (6,3,256) 9.2 KB = 22.3 us (E2), so <=~22 us.

A/B (order C1 D3 D2 C2 D3b D1 C3 D3c; D1 = shard+bmm32, D2 = D1+FP8 draft head, D3 = D2+sonar):
inst | code prose math mean | c4 c8 c12 c16 | AL code/prose/math | gsm50 | draft_ms(1) verify_ms(1)
C1  | 71.29 59.46 112.89 81.21 | 146.2 223.5 279.5 325.8 | 2.48/1.73/3.75 | 49 | 3.69 18.32
C2  | 74.84 59.56 118.27 84.22 | 145.7 219.0 283.9 323.5 | 2.63/1.72/3.94 | 50 | 3.73 18.41
C3  | 78.10 61.18 112.37 83.88 | 151.7 214.9 276.7 324.4 | 2.75/1.77/3.73 | 50 | 3.70 18.41
D1  | 76.66 65.08 117.25 86.33 | 145.3 217.4 278.2 323.2 | 2.60/1.85/3.81 | 50 | 3.69 17.70
D2  | 76.29 63.53 119.10 86.31 | 151.4 218.9 276.7 327.0 | 2.61/1.79/3.88 | 49 | 3.41 20.15
D3  | 78.74 63.73 116.46 86.31 | 153.7 217.5 279.8 327.1 | 2.60/1.72/3.60 | 50 | 3.31 16.32
D3b | 76.64 63.35 119.70 86.56 | 148.3 216.9 279.6 329.3 | 2.50/1.71/3.70 | 49 | 3.34 16.31
D3c | 78.01 64.48 115.41 85.97 | 150.6 218.2 282.2 330.2 | 2.59/1.77/3.59 | 50 | 3.30 16.61
Pooled C (3) -> D3 (3): single 83.10 -> 86.28 (+3.8%); c4 147.9 -> 150.8 (+2.0%); c8 219.1 -> 217.5 (-0.7%);
c12 280.0 -> 280.6 (+0.2%); c16 324.6 -> 328.9 (+1.3%). verify_ms(1) 18.38 -> 16.41, draft_ms(1) 3.71 -> 3.32.
D1/D2 single ~= D3 (one instance each); D3-D2 <1.5% single but verify(1) is lowest with sonar and it is free at conc -> kept.
AL: single-prompt math AL lower on sonar instances (3.60/3.70/3.59 vs C 3.75/3.94/3.73; greedy trajectory change from
target numerics, not the FP8 head: D2 math 3.88). Wider check, single-user GSM8K-24 /metrics AL: D3c 3.396 vs C4 3.339 -> no drop.
D3c: GSM8K-200 @conc16 196/200 (misses 93,100,119,184 = reference); LONGCTX PASS needle 60/60 copy 1.000.
exact.py self-consistent in every instance; no "first call during capture"; sonar logs mode=fp8 weight=float8_e4m3fn.
Note: first D3 launch hung at dspark graph capture (rdma_ar timeouts; rank 5 / spark-six "unspecified launch failure");
identical relaunch and 3 later D3 launches + final relaunch clean -> treated as a transient; watch for it.
Final: dsv41-launch-all.sh, health 200, GSM8K-50 @conc8 50/50, 229.1 tok/s. Raw: ~/dsv41-opt-20261021/r3-e4/.
Rollback: per knob DSV41_QKV_A_SHARD=0 DSV41_IDX_WPROJ= DSV41_FP8_LMHEAD= DSV41_SONAR_OPROJ= (launch-conc.sh, all ranks
identical!); launcher: cp -p ~/dsv41-tp8.sh.bak-r3dense ~/dsv41-tp8.sh on all 8; full revert (service stopped):
bash $S/r3/dense/uninstall-r3dense.sh && bash $S/r3/sonar/oproj/uninstall-sonar-oproj.sh, then verify attention c39e06c1,
logits_processor 35cf8235, dspark 480149ff, deepseek_v4 o_proj f2893ee6 on all 8.
