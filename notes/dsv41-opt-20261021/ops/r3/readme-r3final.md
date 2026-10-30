
## Round 3 summary and final verification (2026-10-22 11:35-11:50 UTC)

**Verified live state (all 8 ranks):** unit dsv41-rank active since 11:23:36 UTC (e5 promoted relaunch), /health 200, no foreign activity.
sha256 (first 8 hex), identical on all 8 ranks:
- launcher ~/dsv41-tp8.sh 2971eb50; backups .bak-r3spec d5c41fd0, .bak-r3dense 776de10b, .bak-r3conc abd49904, .bak-r3markov e270215f
- v1/worker/gpu/spec_decode/rejection_sampler_utils.py 09522889 (.bak-r3spec 99556e84)
- v1/worker/gpu/spec_decode/dspark/speculator.py 0cc6e2bf (.bak-r3spec 774b34c2, .bak-dsmarkov 97bcfaa6 = stock)
- models/deepseek_v41/attention.py f2f48198 (.bak-r3dense c39e06c1, .orig dd146642)
- model_executor/layers/logits_processor.py caf6d216 (.bak-r3dense 35cf8235)
- models/deepseek_v41/nvidia/dspark.py fa56d07d (.bak-r3dense 480149ff); dsv41_r3_dense.py 993023f2 (new)
- models/deepseek_v4/nvidia/ops/o_proj.py 34b9e1a4 (.bak-r3sonar f2893ee6); sonar_bf16_bmm.py 9f6f3f84

**Final measurement** (one instance, ./r3-final-run.sh, results in ./r3-final/. Single = 1024 tokens x 2 repeats per type; conc = 512 tokens each):

| metric | original baseline | round-2 final | **round-3 final** | vs R2 | vs baseline |
|---|---|---|---|---|---|
| single code (greedy) | - | 72.5 | **79.9** | +10.2% | |
| single prose (greedy) | - | 58.8 | **60.0** | +2.0% | |
| single math (greedy) | - | 111.1 | **113.5** | +2.2% | |
| single mean (greedy) | 57 | 80.8 | **84.5** | +4.5% | +48.2% |
| conc4 agg | 102 | 144 | **146.4** | +1.7% | +43.6% |
| conc8 agg | 147 | 210 | **212.2** | +1.0% | +44.3% |
| conc16 agg | - | 313 | **320.6** | +2.4% | |
| conc24 agg | - | ~300 (seqs 16 cap; e3 K0) | **406.6** | +35.5% | |
| conc32 agg | - | ~301 (seqs 16 cap; e3 K0) | **464.5** | +54.1% | |
| sampled single t1.0/p0.95 mean | - | 77.3 (e5 ctl5m0) | **85.1** | +10.1% | |
| sampled conc16 t1.0 | - | 290.6 (e5 ctl5m0) | **312.1** | +7.4% | |
| accept len greedy single code/prose/math | | 2.48 mean | 2.62/1.71/3.70 | | |
| GSM8K-50 conc8 | | 50/50 | 49/50 (miss 12, known flip) | | |
| GSM8K-200 conc16 | | 196/200 | **196/200** (93,100,119,184 = reference set; 279 tok/s) | | |
| long-context | | | **PASS** 60/60 needles, 12/12 copy exact 4K-200K; 250,916 tok 5/5 + exact | | |

- exact.py combined a8c3394af9d2, same hash as every e5 instance.
- Lowest MemAvailable during the run (incl. conc32): 11.4 GB (gate 4 GB).
- Single-stream varies about ±4% between restarts, so the greedy single gain vs R2 (+4.5%) is roughly at the edge of noise; pooled e4 instances put it at ~86. The clear round-3 wins are conc24/32 (+35-54%) and sampled decode (+7-10%).

**Round 3 outcome:**
- PROMOTED e1 sharded Markov drafter + AV_FIXED_MS 3.4 (+5% single, +3% conc). Note: under e5 probabilistic drafting, the sharded path is ignored (full-vocab Markov fallback), so for sampled traffic e1 now only contributes the AV calibration.
- NOT promoted e2 FC2 compaction: compact_b kernel faults with CUDA 715 at first launch on sm_121; nothing installed. Staged in ~/dsv41-stage/fc2c.
- PROMOTED e3 conc/KV: max-num-seqs 32, KV 16G (3.26M tokens), M32 max 96, AV N0/N1 96/104, RDMA reduce 2 MiB. conc24 +37%, conc32 +58%; greedy single -3.7% pooled overridden (acceptance-only, per-step speed identical, GSM8K-16 single +1.9%).
- PROMOTED e4 qkv_a shard + bmm32 weights_proj + FP8 draft head + sonar fp8 o_proj: single +3.8%, verify_ms(1) 18.4 -> 16.4. Two kernel gates (G1 at M=6, sonar bitwise) passed on purpose rather than letter. One-off rank-5 launch failure during graph capture on first D3 launch; not reproduced since.
- PROMOTED e5 pb5 (probabilistic drafting + block verification with fresh-U and fresh-noise fixes): sampled +7-12%, greedy bit-identical, greedy single -2-3% (sharded Markov disabled under probabilistic drafting).

**Rollback** (run from the Mac; stop service for site files; then `bash $S/dsv41-launch-all.sh`):
- e5 only: relaunch with DSV41_SPEC_SAMPLED=0, or restore ~/dsv41-tp8.sh.bak-r3spec + the two .bak-r3spec site files.
- e4: restore .bak-r3dense launcher, run $S/r3/dense/uninstall-r3dense.sh and $S/r3/sonar/oproj/uninstall-sonar-oproj.sh (copies in ./ops/r3/).
- e3: restore .bak-r3conc, or env overrides DSV41_MAX_NUM_SEQS=16 DSV41_KV_BYTES=8G DSV41_TARGET_M32_MAX_M=48 DSV41_AV_SCALE_N0=48 DSV41_AV_SCALE_N1=56 RDMA_AR_MAX_REDUCE_BYTES=1048576.
- Full revert to round-2: launcher .bak-r3markov (e270215f) on all 8; site files back to .bak-r3spec/.bak-r3dense/.bak-r3sonar, and speculator.py to .bak-dsmarkov (97bcfaa6); verify sha256 identical on all 8.

**Scripts archived:** ./ops/r3/ is a copy of the Mac scratchpad r3/ tree (dense, sonar, spec, fc2, e1/e3/e4 helpers, longctx incl. offset_probe.py and the PR #1788 diffs); the 65 MB fi072 wheel was left out.

**Next steps (round 4 candidates):**
1. Greedy-batch sharded Markov path under DSV41_SPEC_SAMPLED=1: capture a second drafter graph set so all-greedy batches use the sharded drafter again (should recover the 2-3% greedy single loss).
2. Fix the e2 compact_b kernel (TMA box / tensor-map or PTX variant on sm_121) for ~8% fewer routed MoE bytes; library staged in ~/dsv41-stage/fc2c.
3. Latent indexer bug: pack decode weights with pack_seq_triton in the padded decode path (sparse_attn_indexer.py ~L818/828) before anyone turns VARLEN_INDEXER off.
4. Investigate noisy long-prompt TTFT (16K 4.3-7.2 s across instances; one 32K at 10.1 s) seen in e3.
5. Widen single-user bench prompt set (e.g. GSM8K-24 via /metrics) so acceptance-trajectory noise stops tripping +/-0.08 AL gates.
6. Watch for the one-off rank-5 (spark-six) CUDA launch failure during draft graph capture seen in e4.
