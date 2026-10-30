
## Round 4 / e4 verifycap-routelive: DSpark-confidence verify cap + dead-row expert skip. PROMOTED (2026-10-23 01:00 UTC, arm vcauto)

**What.** Two verify-side changes. Both are env-gated, and the site files are inert when the flags are off.
- `DSV41_ROUTE_LIVE=skip`: the target router gives padding and dead verify rows expert id -1 and weight 0. No expert weights are read for those rows.
- `DSV41_VERIFY_CAP=auto`: each step, a draft whose cumulative DSpark confidence falls below tau is marked dead. Dead drafts become padding for the router and invalid for acceptance. tau is derived from the AV calibration (scale 0.6 / N0 96 / N1 104 / FIXED_MS 3.4, unchanged since r3). Typical mean tau is 0.15 at C1 and 0.28-0.31 at C8-16; 10-25% of verified drafts are dead.
- Temperature > 0 relies on `DSV41_BLOCK_VERIFY_FRESH_U=1` and `DSV41_DRAFT_FRESH_NOISE=1`. Both are launcher defaults.

**Site files** (sha identical 8/8; backups `*.bak-r4-verify-router` on each rank):

| file | stock / backup | live |
|---|---|---|
| fused_moe/router/dsv4_topk.py | 14f7c848 | 45c5fc4f |
| fused_moe/router/fused_topk_bias_router.py | 8cf50cf2 | ea6720bc |
| v1/worker/gpu/spec_decode/adaptive_verification.py | ec240369 | 7e30eebc |
| v1/worker/gpu/model_runner.py | fc6d5d6d | 9fecbda4 |

**Launcher:** `~/dsv41-tp8.sh` f9140d38 -> **22e83204** on 8/8. It adds `export DSV41_ROUTE_LIVE=${DSV41_ROUTE_LIVE:-skip}` and `export DSV41_VERIFY_CAP=${DSV41_VERIFY_CAP:-auto}` after the GREEDY_SHARDED line. Backup: `~/dsv41-tp8.sh.bak-r4e4`.

**GPU unit tests** (spark-three, service stopped; archived in ops/r4/e4/vr_all.log and vr_moe2.log):
- `all` passed: router, live-cap and reject. The rejection-sampler distribution test passes for the stop rule; the peek negative control fails as designed.
- The `moe` test needed an API-drift fix to the test only: the expert is now built the way the moe-compact test builds it (FusedMoEConfig + make_mxfp4_moe_quant_config + workspace_shapes). Before the fix it hit a cuBLASLt alloc error.
- On the live M32 + compact path, with M 2-96 added: live rows are **bit-identical** in both skip and anchor modes at every M and dead count.
- Saving per dead row x 43 layers: about 2.9-3.5 ms per step at M 6-12, 2.1-2.4 ms at M 48 and 0.9-1.5 ms at M 96. These are eager timings with random routing; real routing gives roughly 0.6x.

**A/B** (one fresh launch per row; order c1 rl a1 c2 a2 u1 c3 u2 c4 u3 c5 u4). a = vc10 (`conf:0.1`), u = vcauto. Single = greedy, 1024 tok; matched = C1 ms/step.

| inst | single | c4 | c8 | c12 | c16 | match coding / prose ms | prose AL | t1 single | t1 c8 | dead % | GSM50 | exact |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c1 | 89.37 | 148.5 | 222.7 | 286.0 | 331.1 | 34.14 / 30.00 | 2.87 | 88.3 | 206.3 | - | 49 | a8c3394af9d2 |
| rl | 90.48 | 154.1 | 225.0 | 281.4 | 329.8 | 34.05 / 29.72 | 2.73 | - | - | - | 50 | a8c3394af9d2 |
| a1 | 91.11 | 161.0 | 220.4 | 282.8 | 329.6 | 34.17 / 29.53 | 2.87 | 86.4 | 207.3 | 4.3 | 50 | a8c3394af9d2 |
| c2 | 88.92 | 152.4 | 223.7 | 283.1 | 326.4 | 33.98 / 29.90 | 2.78 | 87.6 | 209.8 | - | 49 | a8c3394af9d2 |
| a2 | 88.45 | 157.7 | 226.6 | 277.9 | 326.7 | 34.20 / 29.76 | 2.80 | 88.5 | n/a | 4.6 | 50 | a8c3394af9d2 |
| u1 | 90.58 | 152.8 | 220.4 | 288.7 | 327.2 | 33.67 / 29.00 | 2.70 | 89.8 | 207.0 | 13.2 | 50 | a8c3394af9d2 |
| c3 | 87.65 | 149.2 | 223.0 | 277.9 | 324.7 | 33.91 / 29.82 | 2.71 | 89.4 | 209.3 | - | 49 | a8c3394af9d2 |
| u2 | 90.80 | 152.1 | 219.7 | 277.9 | 330.7 | 34.06 / 29.10 | 2.74 | 87.0 | 210.4 | 13.1 | 50 | a8c3394af9d2 |
| c4 | 86.53 | 157.4 | 221.4 | 285.6 | 327.7 | 34.06 / 29.98 | 2.79 | 92.3 | 207.3 | - | 49 | a8c3394af9d2 |
| u3 | 88.83 | 156.7 | 222.7 | 281.4 | 325.9 | 33.76 / 28.66 | 2.55 | 86.2 | 212.4 | 13.1 | 50 | a8c3394af9d2 |
| c5 | 91.12 | 154.1 | 219.3 | 281.4 | 328.3 | 33.97 / 29.88 | 2.73 | 88.6 | 213.0 | - | 50 | a8c3394af9d2 |
| u4 | 90.17 | 151.6 | 220.9 | 285.0 | 327.2 | 33.75 / 29.24 | 2.71 | 87.1 | 209.4 | 12.6 | 49 | **b7e868c9d700** |

Pooled C (5 instances) vs U (vcauto, 4 instances):
- **Greedy single:** 88.72 -> **90.10 (+1.55%)**. Adjacent pairs: +1.9 / +3.6 / +2.7 / -1.0%.
- **Concurrency:** c4 152.3 -> 153.3 (+0.6%), c8 222.0 -> 220.9 (-0.5%), c12 282.8 -> 283.2 (+0.2%), c16 327.6 -> 327.7 (0.0%).
- **Matched C1 ms/step:** prose 29.92 -> **29.00 (-0.92, -3.1%; every U is below every C)**; coding 34.01 -> 33.81.
- **Matched coding_div C8 ms:** 88.0 -> 86.0. AL drops by design (prose C1 2.78 -> 2.68).
- **Sampled t1 single:** quick (2 reps each) 89.2 -> 87.5. With 4 extra reps on c5/u4, 87.2 -> 88.3. Pooled over all reps: 88.65 -> 87.78 (-1.0%). The c4 control code run at 99.8 is an outlier.
- **Sampled t1 c8:** 209.1 -> 209.8 (+0.3%). t0.6 single: 86.9 -> 88.9.

vc10 (2 instances) is a no-gain arm: dead 4-5%, single 89.78 (+1.2% vs the first two controls), c8 -0.4%, matched coding_div C4 not better. rl alone is one instance (90.48, exact hash equal to ctl), so it is a neutrality guard only.
**Not run:** vc10ib, vc05 and vc20. The step was already about 1.5 h over budget, and they tune the fixed-tau arm that lost to auto. STEP 4 (drafter gather on RDMA) was also not run.

**Exactness: greedy-equivalent (not bitwise class X).**
- ROUTE_LIVE alone: live rows are bit-identical in the unit test and the rl hash equals control.
- VERIFY_CAP: on 5 of 6 capped instances, exact.py gives a8c3394af9d2. On u4 it gives b7e868c9d700: 2 of 6 prompts differ, and the hash is self-consistent over 3 runs.
- Cause: the cap changes the step trajectory (where acceptance stops). auto-tau depends on the per-restart AV profile (u4 tau 0.192 vs 0.203-0.209). This is the same cross-restart trajectory sensitivity as e1's 9330f03c4f97 instance. Every token is still the target argmax on this stack; no lossy kernel is involved.
- Correctness gates are unaffected.

**Gates on u4** (vcauto, same config as promoted):
- GSM8K-200 conc16: **195/200**. Misses 12, 93, 100, 119, 184: the reference set plus the known item-12 flip. 304 tok/s.
- longctx 4K-200K: **PASS**, needles 25/25, copy 1.000.
- gsm8k_sampled_gate compare c5 vs u4: **PASS** (50.0 vs 49.0).
- GSM50 on the U instances: 50 / 50 / 50 / 49.
- On every instance: no 'first call during capture', no JIT after health, no errors, MemAvailable >= 20.3 GB (14.5 GB right after longctx).
- No cross-rank confidence divergence: AL is normal and there is no garbage.

**Promoted relaunch** (from the Mac, 00:59 UTC start, health 200 by 01:05): 8/8 active on 22e83204, with the 'verify-cap: auto' line present. exact.py a8c3394af9d2; GSM8K-50 conc8 49/50 at 227 tok/s; MemAvailable 22.1 GB.

**Incidents.** c1 try 1 hung in DSpark capture (`rdma_ar rank 0: timeout waiting for 4 phase 1`), the known transient on the control arm; the retry was clean. Log: r4/e4/c1fail1/.

**Rollback** (from the Mac):
- Env only: `bash $S/conc/launch-conc.sh DSV41_VERIFY_CAP=0 DSV41_ROUTE_LIVE=0`.
- Launcher: `cp -p ~/dsv41-tp8.sh.bak-r4e4 ~/dsv41-tp8.sh` on all 8 (back to f9140d38), then `bash $S/dsv41-launch-all.sh`.
- Site files: stop the service, run `bash $S/r4/verify-router/uninstall.sh`, then verify on 8/8 that the shas are back to 14f7c848 / 8cf50cf2 / ec240369 / fc6d5d6d. Do the launcher rollback first.

Raw: `./r4/e4/{c1..c5,rl,a1,a2,u1..u4,live,c1fail1}/` and sum-all.txt (u4 also has gsm200, longctx, exact_r2/r3; c5/u4 have t1x). Scripts and unit logs: `./ops/r4/e4/`.
