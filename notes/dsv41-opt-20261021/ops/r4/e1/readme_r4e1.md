
## Round 4 / e1 bitstack: L2 prefetch + main_proj split + certified lm-head + eager glue. PROMOTED (2026-10-22 17:05 UTC)

**What changed.** Four exact (class X) decode-step micro-wins, promoted together as one stack. All are env-gated, and every site file is inert when its flag is off.
- `DSV41_L2_PREFETCH=1`: prefetch the next GEMM's weights into L2 while the ~125 in-graph RDMA collectives run (the plan includes the r3 QKV_A_SHARD all-gathers).
- `DSV41_R4_MAINPROJ_SPLIT=1`: DSpark main_proj is split by columns across TP plus one all-gather. It is bitwise self-checked per M at runtime.
- `DSV41_R4_CERT_HEAD=1`, `DSV41_R4_CERT_HEAD_M=2-96`: the target lm-head screens with MXINT8 and computes exactly only the tiles that can hold the argmax. The argmax is identical to stock.
  - It arms only on steps where every request is greedy with no logprobs or logits processing. Sampled traffic gets nothing from it.
  - M=1 was disqualified because full mode is not bitwise there; that is also where all of the GPU test's computed-logit mismatches came from.
- `DSV41_EAGER_GLUE=all`: one fused SWA-metadata kernel replaces 15x3 eager launches, and the dflash pad loop runs 1024 wide. There is no log line; the flag is read from the env.

**Site files** (sha identical 8/8; backups `*.bak-r4-<idea>` on each rank, made by `$S/r4/common/r4install.sh`):

| file | live | stock / backup |
|---|---|---|
| models/deepseek_v41/nvidia/model.py (model-merged: CERT + MAINPROJ + inert prefill-sp) | e7264781 | 5fe8b4a7 |
| models/deepseek_v41/nvidia/dsv41_r4_small.py (new) | 02f94f29 | absent |
| distributed/device_communicators/rdma_all_reduce.py (rebased l2-prefetch bundle) | 99102ff1 | 3a128082 |
| distributed/device_communicators/dsv41_l2_prefetch.py (new) | 7d2acdc8 | absent |
| v1/attention/backends/mla/sparse_swa.py | 38b16c51 | 2afd3136 |
| v1/worker/gpu/spec_decode/dflash/speculator.py | 17ddccf1 | b731b09c |

The launcher `~/dsv41-tp8.sh` is **709abe3d** on all 8 ranks, with 5 new `export ...:-default` lines after DSV41_DRAFT_FRESH_NOISE. The backup is `~/dsv41-tp8.sh.bak-r4e1` (2971eb50).

**GPU unit tests** (service stopped). Raw results are in `r4/e1/unit/`.
- **CERT** (spark-one, then spark-two with `--ms 2-96 --trials 3`):
  - All counters are 0 on all 8 shards for M=2-96, and the accumulation margin is 0.0009 of c_acc.
  - The default run showed 7-27 computed-logit mismatches per rank. They all came from M=1, which is excluded.
  - Microbench on a peaked activation at M=6: 480 vs 730 us. Gaussian activations are slower than stock at M≥6. In serving, a call computes about 19.4 of 1010 tiles (6000 armed calls on rank 0).
- **main_proj:** every slice is bit-identical at every M. At M=6 a rank's slice takes 140 us vs 930 us for the full GEMM.
  - The test needs `source venv/bin/activate` (for ninja) and CUDA_HOME. Without them, `mm_mxfp8` is missing from vllm.utils.flashinfer.
- **L2 prefetch** (spark-five): exact at M=1/6/24/96, with graph == eager. In the microbench it saves 30-39 us per window at M=1-6 (2.4-3.1 ms per forward).
- **eager-glue** (spark-six): bit-exact. It saves 89-91 us per step (SWA) and 25-27 us per step (dpad).

**A/B** (one fresh launch per row, order c0p c1 [spin] s1 c2 s2 c3 s3). Single = greedy, 1024 tokens, code/prose/math x1. Matched = matched_bench C1 ms/step.

| inst | single mean | c4 | c8 | c12 | c16 | matched coding ms / AL | matched prose ms / AL | verify_ms(1) | draft_ms(1) | GSM50 | exact |
|---|---|---|---|---|---|---|---|---|---|---|---|
| c0p | 82.19 | 147.4 | 218.6 | 274.6 | 323.3 | 36.03 / 5.20 | 32.53 / 2.81 | 16.32 | 8.14 | 49 | a8c3394af9d2 |
| c1 | 81.21 | 150.7 | 216.3 | 271.5 | 317.3 | 36.32 / 5.23 | 32.39 / 2.86 | 16.50 | 8.08 | 50 | a8c3394af9d2 |
| c2 | 81.49 | 147.9 | 217.5 | 276.6 | 315.3 | 35.99 / 5.29 | 32.24 / 2.79 | 16.40 | 8.15 | 49 | a8c3394af9d2 |
| c3 | 82.15 | 146.8 | 208.7 | 273.5 | 317.4 | 35.86 / 5.08 | 32.03 / 2.71 | 16.35 | 8.09 | 50 | a8c3394af9d2 |
| s1 | 86.04 | 148.6 | 216.2 | 272.5 | 320.3 | 35.57 / 5.09 | 31.68 / 2.84 | 16.11 | 7.86 | 50 | a8c3394af9d2 |
| s2 | 86.14 | 149.4 | 218.4 | 270.6 | 321.3 | 35.40 / 5.18 | 31.43 / 2.79 | 16.19 | 7.76 | 50 | a8c3394af9d2 |
| s3 | 85.09 | 146.7 | 212.9 | 272.4 | 318.2 | 35.45 / 5.24 | 31.34 / 2.68 | 16.24 | 7.78 | 50 | a8c3394af9d2 |
| spin* | 86.92 | 148.8 | 217.4 | 267.4 | 313.1 | 35.80 / 5.21 | 31.84 / 2.76 | (pinned) | (pinned) | 49 | a8c3394af9d2 |

\* spin = S with `DSV41_AV_COST_JSON` pinned to c1's raw profile. It is an exactness probe and is not pooled.

Pooled C (4 instances) vs S (3 instances):
- Greedy single: 81.76 -> 85.76, **+4.9%**. Each pair: +5.9 / +5.7 / +3.6%.
- conc: c4 +0.0%, c8 +0.2%, c12 -0.8%, c16 318.3 -> 319.9 (+0.5%).
- Matched ms/step: coding 36.05 -> 35.47 (-0.58), prose 32.30 -> 31.48 (-0.82).
- verify_ms(1): 16.39 -> 16.18 (L2 + glue in-graph part). draft_ms(1): 8.11 -> 7.80 (main_proj).
- Matched AL: coding -0.03, prose -0.02. Both are inside +/-0.10.
- Sampled quick (c3 vs s3): t1 single 86.0 -> 87.0 (+1.2%), t1 c8 211.9 -> 210.7 (-0.6%), t0.6 single 89.2 -> 90.7, GSM8K-50 t0.6 x2 49.5 -> 50.0.
- The step saving (~0.6-0.8 ms) is smaller than the expected 0.9-1.5 ms. Part of the +4.9% single gain comes from adaptive verification: it re-profiles the cheaper draft step and verifies a little deeper (single-bench code AL 2.55 -> 2.67).

**Exactness: class X (exact).**
- Every C and S instance gives exact.py a8c3394af9d2. With the AV profile pinned (spin), the hash and all 6 per-prompt hashes are identical to control.
- The one different hash, 9330f03c4f97 on the E1-chk instance, was self-consistent over 3 repeats. That instance ran CERT=check, its own AV re-profile and a 512-token single. It reflects the known cross-restart AV-profile sensitivity (Caveats), not the kernels.

**Promote gates on s3** (identical config to s2, server up):
- GSM8K-200 greedy conc16: **195/200**, misses 12, 93, 100, 119, 184 (the reference set plus the known item-12 flip), 0 truncated, 313 tok/s.
- longctx 4K-200K: **LONGCTX PASS**, needles 25/25, copy 1.000.
- No 'first call during capture', MISMATCH or 'not bit-identical' lines. L2 plan tokens=6 collectives=125 (85 + 40 r3 QKV_A_SHARD/IDX gathers).
- CERT compiles 2 Triton kernels during the first request; there is no JIT after that.
- Lowest MemAvailable: 20.4 GB.

**Promoted relaunch** (17:02 UTC, no env overrides): 8/8 active on launcher 709abe3d, with CERT built, MAINPROJ active and the L2 plan relearned. GSM8K-50 conc8 50/50 at 217.6 tok/s; exact.py a8c3394af9d2.

**Incidents.**
1. The first E1-chk launch failed during dspark graph capture: rank 2 (spark-three) got a TCPStore broken pipe and the RDMA AR timed out. The retry was clean, and so were the 6 later launches with the same flags. This looks like the transient seen in r3-e4, not the stack.
2. `ssh spark-one 'bash ~/dsv41-opt-20261021/ops/dsv41-launch-all.sh'` does NOT work: spark-one cannot ssh to spark-six, seven or eight, so those 3 ranks kept the old instance and ranks 0-4 timed out. **Always run launch-all from the Mac:** `bash $S/dsv41-launch-all.sh` (same file as ops/).

**Rollback** (from the Mac):
- Env only (site files inert):
  `bash $S/conc/launch-conc.sh DSV41_L2_PREFETCH=0 DSV41_R4_MAINPROJ_SPLIT=0 DSV41_R4_CERT_HEAD=0 DSV41_EAGER_GLUE=`
  Individual flags work the same way.
- Launcher: `cp -p ~/dsv41-tp8.sh.bak-r4e1 ~/dsv41-tp8.sh` on all 8 (2971eb50), then `bash $S/dsv41-launch-all.sh`.
- Site files (stop the service first):
  - `bash $S/r4/rebased/l2-prefetch/uninstall.sh; bash $S/r4/eager-glue/uninstall.sh; bash $S/r4/model-merged/uninstall.sh`
  - Then verify on 8/8: rdma_all_reduce 3a128082, sparse_swa 2afd3136, dflash/speculator b731b09c, model.py 5fe8b4a7, with dsv41_l2_prefetch.py and dsv41_r4_small.py absent.
  - Keep model-merged if E5 prefill-sp is still pending.

Raw data: `./r4/e1/<inst>/` (c0p c1 c2 c3 s1 s2 s3 spin chk chk-try1 promoted unit). The pin file is `r4/e1/c1_av_pin.json`. Scripts are in `./ops/r4/e1/` (inst.sh, chain.sh, chk.sh, summ_e1.py).
