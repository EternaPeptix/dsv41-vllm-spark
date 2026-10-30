
## Round 4 summary and final verification (2026-10-23 04:35-04:55 UTC)

**Verified live state.** All 8 `dsv41-rank` user units active since 04:30:57 UTC (r4-e6 relaunch from the Mac), /health 200, no foreign activity. Live rank-0 env has DSV41_L2_PREFETCH=1, R4_MAINPROJ_SPLIT=1, R4_CERT_HEAD=1 (M=2-96), EAGER_GLUE=all, DSPARK_GREEDY_SHARDED=1, ROUTE_LIVE=skip, VERIFY_CAP=auto, IDX_ROW_SHARD_MIN=1024, IDX_ROW_SHARD_MIN_CTX=32768, AV_FIXED_MS=3.4.
sha256 (first 8 hex), identical on all 8 ranks (checked with ops/r4/shacheck.sh):
- launcher ~/dsv41-tp8.sh **5ccc4b79** (backups .bak-r4e1=2971eb50 (r3 final), .bak-r4e2gs=709abe3d, .bak-r4e4=f9140d38, .bak-r4e5=22e83204)
- r4 site files: models/deepseek_v41/nvidia/model.py e7264781 (.bak-r4-model-merged) + dsv41_r4_small.py 02f94f29 (new); distributed/device_communicators/rdma_all_reduce.py 99102ff1 (.bak-r4-l2-prefetch) + dsv41_l2_prefetch.py 7d2acdc8 (new); v1/attention/backends/mla/sparse_swa.py 38b16c51 and v1/worker/gpu/spec_decode/dflash/speculator.py 17ddccf1 (.bak-r4-eager-glue); spec_decode/dspark/speculator.py bbcee9e3 (.bak-r4-greedy-sharded = 0cc6e2bf); fused_moe/router/dsv4_topk.py 45c5fc4f, router/fused_topk_bias_router.py ea6720bc, spec_decode/adaptive_verification.py 7e30eebc, v1/worker/gpu/model_runner.py 9fecbda4 (.bak-r4-verify-router); v1/attention/backends/mla/indexer.py 4753707e (.bak-r4-prefill-idxshard)
- restored to pre-r4 (not promoted): utils/flashinfer.py c0db917f (e3), v1/core/sched/scheduler.py d9a39870 (e5 sched). FC2-compact (e6) never installed.
- r3 files unchanged: rejection_sampler_utils 09522889, attention.py f2f48198, logits_processor.py caf6d216, dspark.py fa56d07d, o_proj.py 34b9e1a4, dsv41_r3_dense.py 993023f2, sonar_bf16_bmm.py 9f6f3f84.

**Final measurement** (one instance, results in ./r4-final/, script ops/r4-final-run.sh; single = 1024 tokens x 2 repeats per type, conc = 512 tokens each):

| metric | original baseline | round-3 final | **round-4 final** | vs R3 | vs baseline |
|---|---|---|---|---|---|
| single code | - | 79.9 | **86.3** | +8.0% | |
| single prose | - | 60.0 | **67.7** | +12.8% | |
| single math | - | 113.5 | **117.8** | +3.8% | |
| single mean | 57 | 84.5 | **90.6** | +7.2% | +59.0% |
| conc4 agg | 102 | 146 | **153.3** | +5.0% | +50.3% |
| conc8 agg | 147 | 212 | **221.9** | +4.7% | +51.0% |
| conc16 agg | - | 321 | **330.9** | +3.1% | |
| conc24 agg | - | 407 | **415.2** | +2.0% | |
| conc32 agg | - | 465 | **479.0** | +3.0% | |
| sampled single t1.0/p0.95 mean | - | 85.1 | **90.0** | +5.8% | |
| sampled conc16 t1.0 | - | 312 | **312.2** | 0.0% | |
| accept len greedy single code/prose/math | | 2.62/1.71/3.70 | 2.52/1.74/3.48 | | |
| GSM8K-50 conc8 | | 49/50 | 49/50 (item 12, known flip) | | |
| GSM8K-200 conc16 | | 196/200 (279 tok/s) | 196/200, misses 93/100/119/184 = reference set (286 tok/s) | | |
| long-context | | PASS to 250K | 60/60 needles + copy 1.000 (4K-200K), 250,916 tok 5/5 + copy exact: PASS | | |
| exact.py | | a8c3394af9d2 | a8c3394af9d2 | | |

- Lowest MemAvailable over the run 11.2 GB (spark-one, during longctx; r3: 11.4 GB).
- Sampled single code had one 110.6 tok/s rep (79.9 for the other), so the sampled +5.8% is noisier than the greedy number. Restart-to-restart spread is still about ±4% single.

**Matched bench** (rhys-ep matched_bench.py default plan, ./r4-final/matched.json, vs r4/matched/matched-post-r3.json):

| workload | C | post-r3 per-stream | r4 per-stream | post-r3 ms/step | r4 ms/step | AL post-r3 -> r4 |
|---|---|---|---|---|---|---|
| coding | 1 | 143.72 | 153.15 | 36.22 | 33.75 (-6.8%) | 5.21 -> 5.17 |
| coding | 4 | 100.22 | 101.54 | 51.52 | 50.14 | 5.16 -> 5.08 |
| coding | 8 | 75.12 (agg 506) | 77.28 (agg 513) | 67.90 | 66.24 | 5.10 -> 5.11 |
| prose | 1 | 80.23 | 88.12 | 32.41 | 28.18 (-13.1%) | 2.60 -> 2.48 |
| prose | 4 | 52.59 | 52.55 | 50.97 | 47.04 | 2.68 -> 2.47 |
| prose | 8 | 39.35 | 39.83 | 65.73 | 61.91 | 2.59 -> 2.47 |
| coding_div | 8 | 54.09 | 52.55 | 88.20 | 85.96 | 4.77 -> 4.52 |

ms/step fell in every row. Prose/coding_div AL drops by design (e4 verify cap stops verification early on low-confidence drafts).

**Round 4 outcome:**
- PROMOTED e1 bitstack: L2 prefetch + main_proj split + certified lm-head (M 2-96) + eager glue. Exact. About +5% single.
- PROMOTED e2 greedy-sharded drafter graphs for all-greedy steps. Exact. About +5% single, +2-3% conc.
- NOT promoted e3 b12x small-M MXFP8: bitwise exact but +0.0% (live CUTLASS is autotuned). Uninstalled.
- PROMOTED e4 vcauto (VERIFY_CAP=auto + ROUTE_LIVE=skip). Greedy-equivalent, not bitwise: one restart hashed b7e868c9d700 (AV-profile dependent). +1.55% single, prose C1 -3% ms/step.
- PROMOTED e5 prefill idxshard (env only): 128K +4.2%, 200K +7.3% prefill tok/s. Greedy-equivalent at >=32K context, exact on decode. prefill-sched failed its gate (uninstalled); prefill-sp stopped at the debug compare.
- NOT promoted e6 FC2-compact v2 kernel: root cause (16U4_ALIGN16B TMA needs inner coord %128) fixed and bitwise exact, but only 1.10x vs a 1.15x gate. Native build only on spark-four ~/dsv41-stage/fc2c-v2.

**Rollback** (run from the Mac, never from spark-one; S=~/dsv41-opt-20261021/ops on spark-one; stop the service before touching site files; then `ssh spark-one 'cat ~/dsv41-opt-20261021/ops/dsv41-launch-all.sh' | bash` and verify sha256 on 8/8):
- e5 idxshard: env `bash $S/conc/launch-conc.sh DSV41_IDX_ROW_SHARD_MIN=16384 DSV41_IDX_ROW_SHARD_MIN_CTX=0`, or restore ~/dsv41-tp8.sh.bak-r4e5; site: `bash $S/r4/prefill-idxshard/uninstall.sh`.
- e4 vcauto: restore ~/dsv41-tp8.sh.bak-r4e4 (or env DSV41_VERIFY_CAP=0 DSV41_ROUTE_LIVE=0 via launch-conc.sh); site: `bash $S/r4/verify-router/uninstall.sh`.
- e2 greedy-sharded: env DSV41_DSPARK_GREEDY_SHARDED=0, or restore .bak-r4e2gs; site: `bash $S/r4/rebased/greedy-sharded/uninstall.sh`.
- e1 bitstack: restore .bak-r4e1 (= r3 final launcher 2971eb50); site: `bash $S/r4/model-merged/uninstall.sh`, `bash $S/r4/rebased/l2-prefetch/uninstall.sh`, `bash $S/r4/eager-glue/uninstall.sh`.
- Full revert to round-3 final: launcher .bak-r4e1 on all 8, then each uninstall above (or copy every *.bak-r4-* back over its file); expected r3 shas: dspark/speculator.py 0cc6e2bf, others as in the Round 3 summary.
- Undo launcher steps in reverse order (e5, e4, e2, e1) before site files, so no flag is left on with its file removed.

**Ops notes:** dsv41-launch-all.sh must run from the Mac (spark-one has no key for spark-six/seven/eight). The rdma_ar timeout during first DSpark graph capture happened about 1 in 8 launches all round. It is a transient; stop all ranks and relaunch.

**Scripts archived:** ./ops/r4/ is a copy of the Mac scratchpad r4/ tree (queue.md, rhys-ep, rebased patch bundles with install/uninstall, e1-e6 drivers and READMEs, prefill/verify-router/small-kernels/b12x work, sha snapshots). Left out: b12x-moe/overlay (13 MB vendored b12x/flashinfer copy), the 7.9 MB e5 torch profiler trace, *.so builds, __pycache__.

**Next steps (round 5 candidates):**
1. FC2-compact: pack the 32-col tail to 16 B/row (4-row packing or ALIGN8B smem tail) to get the full -25% FC2 bytes; v2 is at 1.10x.
2. Post-/health warmup: one >=24K prompt + one short greedy request, or add the e1/e4 Triton kernels to engine warmup, to remove first-long-prompt TTFT noise (+0 to +1.3 s).
3. Log distinct experts per layer per verify step to test the AL-dependent step-cost hypothesis (0.6-0.7 ms per accepted token plus a per-workload baseline).
4. Drafter gather onto RDMA (e4 STEP 4, not run).
5. Find the rdma_ar first-capture timeout transient.
