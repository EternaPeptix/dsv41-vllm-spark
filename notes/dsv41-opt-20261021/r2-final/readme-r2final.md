
## Round 2 summary and final verification (2026-10-22 ~01:35-02:00 UTC)

**Verified live state (all 8 ranks):** unit dsv41-rank active since 01:25:47 UTC, unit env only `DSV41_EXTRA=` (empty), /health 200, 17*23 -> "391".
rank0.log shows `rdma_ar enabled ... max_reduce=1048576`, `target compact FC1 enabled ... FC1 rows 576`, and the adaptive-verification calibration line (scale=0.6 n0=48 n1=56 fixed_ms=2.0). No nsys or Codex processes.

sha256 (first 8 hex), identical on all 8 ranks:
- launcher ~/dsv41-tp8.sh e270215f (backups .bak-avcal .bak-rdmaext .bak-compact .bak-conc .bak-m32 .bak-rdma-ar present; .bak-spec-len exists only on some ranks, from round 1)
- site-packages (venvs/vllm032): deep_gemm_moe.py 45418259, mxfp4.py 67339029, fp8_utils.py 8dc7dab8, adaptive_verification.py ec240369, speculator.py 97bcfaa6 (stock), cuda_communicator.py 52fc8cf5, rdma_all_reduce.py 3a128082, deep_gemm_utils.py b7e8de4d, warmup.py 19009d42. 10 .bak-* files are present on every rank.

**Final measurement** (one instance, ./r2-final/, run.sh. Single is 1024 greedy tokens x 2 repeats per type; conc runs are 512 tokens each):

| metric | original baseline | round-1 final | **round-2 final** | vs R1 | vs baseline |
|---|---|---|---|---|---|
| single code | - | 71.4 | **72.5** | +1.5% | |
| single prose | - | 53.9 | **58.8** | +9.1% | |
| single math | - | 104.4 | **111.1** | +6.4% | |
| single mean | 57 | 76.6 | **80.8** | +5.5% | +41.8% |
| conc4 agg | 102 | 134.7 | **144.0** | +6.9% | +41.1% |
| conc8 agg | 147 | 192.6 | **210.1** | +9.1% | +43.0% |
| conc16 agg | - | 277.3 | **313.5** | +13.0% | |
| mean accept len single / c16 | | ~2.35 | 2.48 / 2.39 | | |
| GSM8K-50 (conc 8) | | 49-50/50 | **50/50** | | |
| GSM8K-200 (conc 16) | | (no reference) | **196/200 (98.0%)** | | |

- exact.py was run twice in this instance and matched itself (823ce1923c16).
- GSM8K-200 misses are items 93, 100, 119 (truncated, no answer) and 184. There is no earlier GSM8K-200 run to compare against, so this is the reference for future rounds.
- Single-stream varies between restarts by about ±4%. This instance's code number (72.5) is at the low end; across exp5's two A5 instances the single mean was 84.1. A fair estimate for round 2 is single ~81-84 tok/s.

**Round 2 outcome:**
- PROMOTED exp1 moe-compact (DSV41_MOE_COMPACT=1). Unpadded target FC1 (576 rows/expert/rank), bit-exact, -7.5 GiB/rank. Gains: single +5%, conc4-16 +7-10%.
- NOT promoted: exp2 skinny_mx dense MXFP8. It failed the CUDA-graph kernel gate, because autotuned CUTLASS already streams ~200 GB/s. Only WO-A A16 wins (1.4-2.3x), worth about 2% single-stream.
- PROMOTED exp3 rdma-thresholds (RDMA_AR_MAX_REDUCE_BYTES=1048576). It is env-only. Two-shot RDMA all-reduce now covers up to 102 rows, conc16 gains about 5%, and draft time at 16 requests drops 19.1 -> 12.5 ms.
- NOT promoted: exp4 sharded-Markov drafter. Drafts are identical and draft time drops 1.4 ms, but single was only +1.0%. The stock trimmer over-trims when drafting is cheaper.
- PROMOTED exp5 adaptive-verification cost calibration (scale 0.6, fixed 2 ms, flat to M=48, back to profile by 56). Single +6.1%, concurrency +1-2.5%.

**Full rollback to round-1 config** (run from the Mac; stop the service first for site files):
1. Launcher: `cp ~/dsv41-tp8.sh.bak-compact ~/dsv41-tp8.sh` on all 8. 3045aa83 is the round-1 launcher.
2. Site files: `bash $S/r2/moe-compact/uninstall.sh` (deep_gemm_moe/mxfp4/fp8_utils back to 84ecc6ef/a7fc1bcd/013df536) and `bash $D/install.sh av rollback` (adaptive_verification back to b0526416).
3. `bash $S/dsv41-launch-all.sh`.

Per experiment: see the exp1/exp3/exp5 sections above. Env-only partial rollbacks are DSV41_MOE_COMPACT=0, RDMA_AR_MAX_REDUCE_BYTES=524288, and DSV41_AV_VERIFY_SCALE=1 DSV41_AV_FIXED_MS=0.

**Scripts archived:** ./ops/r2/ holds a copy of the Mac scratchpad r2/ tree: moe-compact, dense/skinny_mx, rdma-ext, drafter, av-profile, exp1/exp4/exp5 helpers, and finalsumm.py.

**Next steps (round 3 candidates):**
1. Re-test exp4 sharded-Markov on top of the exp5 calibration. Cheaper drafting should no longer over-trim. With AV off it measured +6% single.
2. DSV41_TARGET_M32_MAX_M=96 together with max-num-seqs 24/32. The verify cliffs at 48/56 are gone (verify at 56 tokens took 119 ms with M32_MAX_M=96), and exp1 freed 7.5 GiB/rank. Also try a larger --kv-cache-memory-bytes (12-14G).
3. A separate RDMA all-gather size cap (~1 MiB), so 48-96-row logits gathers fall back to NCCL, which is faster there.
4. Compact FC2 (K=288). This needs a DeepGEMM mixed-FP4 kernel change, for about another 8% of routed bytes. Also try 64-wide N tiles for M>48, where the kernel gain is only 1.05x.
5. WO-A A16 skinny alone (DSV41_SKINNY_MX_WOA=1), for about +2% single. It needs at least 3 ABAB pairs, and should be compared against Codex's WO-A Marlin.
