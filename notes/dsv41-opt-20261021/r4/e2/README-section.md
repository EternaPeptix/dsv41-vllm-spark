
## Round 4 / greedy-sharded (r4-e2): sharded greedy drafter on all-greedy steps. PROMOTED (2026-10-22 19:21 UTC)

**What.** Since r3-e5 (`DSV41_SPEC_SAMPLED=1`, probabilistic drafting + block verify), the r3-e1 vocab-sharded greedy Markov drafter was disabled ("probabilistic drafting needs full logits"). With `DSV41_DSPARK_GREEDY_SHARDED=1`, dspark/speculator.py builds the sharded state anyway and captures a second set of FULL drafter CUDA graphs with it. It replays that set on steps where every request has temperature 0. The check runs on the CPU and is identical on every rank: `runner.sampler.sampling_states.temperature.np[idx_mapping_np]`. Mixed and sampled steps use the stock probabilistic graphs. It requires `DSV41_DSPARK_SHARDED_MARKOV=1`, which is the launcher default. The second graph set costs about 33 MB per rank (startup cuda_active 67.76 vs 67.79 GB).

**Site file:** v1/worker/gpu/spec_decode/dspark/speculator.py is 0cc6e2bf (backup `*.bak-r4-greedy-sharded`) -> **bbcee9e3** on 8/8.
**Launcher:** `~/dsv41-tp8.sh` 709abe3d -> **f9140d38** on 8/8. It adds `export DSV41_DSPARK_GREEDY_SHARDED=${DSV41_DSPARK_GREEDY_SHARDED:-1}` after the EAGER_GLUE line. Backup: `~/dsv41-tp8.sh.bak-r4e2gs`.

**Check instance:** both log lines ('active', 'capturing greedy draft graphs') appear on 8/8. There is no 'no CPU temperature mirror' and no 'first call during capture'. Counters show greedy steps 1000 and other steps 198-291, so sampled steps took the stock path.

**Timing** (ABAB, one launch per row; single = greedy, 1024 tok; matched = matched_bench C1 ms/step / AL):

| inst | single mean | c4 | c8 | c12 | c16 | coding ms / AL | prose ms / AL | coding_div C8 ms | draft_ms(1) | GSM50 | exact |
|---|---|---|---|---|---|---|---|---|---|---|---|
| c1r | 86.92 | 151.1 | 219.4 | 273.8 | 319.4 | 35.42 / 5.08 | 31.43 / 2.79 | 90.93 | 7.71 | 49 | a8c3394af9d2 |
| g1 | 88.36 | 156.7 | 223.4 | 280.5 | 326.3 | 33.96 / 5.07 | 29.97 / 2.75 | 90.65 | 7.78 | 50 | a8c3394af9d2 |
| c2 | 84.94 | 154.3 | 215.1 | 275.7 | 316.1 | 35.47 / 5.20 | 31.24 / 2.79 | 89.47 | 7.74 | 49 | a8c3394af9d2 |
| g2 | 91.94 | 149.5 | 218.7 | 278.8 | 323.8 | 34.02 / 5.20 | 30.17 / 2.68 | 88.18 | 7.66 | 49 | a8c3394af9d2 |
| c3 | 84.21 | 144.5 | 216.8 | 270.1 | 316.0 | 35.44 / 5.08 | 31.43 / 2.71 | 89.69 | 7.67 | 49 | a8c3394af9d2 |
| g3 | 89.48 | 151.6 | 222.1 | 281.6 | 330.5 | 34.11 / 5.20 | 29.81 / 2.72 | 88.15 | 7.80 | 50 | a8c3394af9d2 |

Pooled C vs G:
- **Greedy single:** 85.36 -> **89.93 (+5.4%)**. Pairs: +1.7 / +8.2 / +6.3%.
- **Concurrency:** c4 +1.8%, c8 +2.0%, c12 +2.6%, c16 317.2 -> 326.9 (+3.0%).
- **Matched C1 ms/step:** coding 35.44 -> 34.03 (-1.41) and prose 31.37 -> 29.98 (-1.39). Every pair is >= 1.07 ms lower. Coding_div C8: 90.03 -> 88.99.
- **AL:** unchanged within noise.
- **draft_ms(1):** the AV profile is calibrated on the stock path, so it stays ~7.7 (as expected). `DSV41_AV_FIXED_MS=2.4` was not tried because AL did not drop.

**Sampled** (runarm_sampled quick, c2 vs g2):
- t1 single: 87.0 -> 87.9.
- t1 c8: 213.0 -> 209.5 (-1.7%).
- t0.6 single: 88.4 -> 88.9.
- GSM8K t0.6 x2: 49.5 -> 50.0.

**Exactness: class X (exact).**
- exact.py gives a8c3394af9d2 on every C and G instance, with all 6 per-prompt hashes equal.
- The task's mixed test (`r4/e2/mixed.py`: 4 concurrent greedy requests, with or without 4 concurrent T=1 requests, must equal solo greedy) gives 0/4 on **control too**. Concurrent greedy output depends on batch composition: batch-variant GEMM numerics and request arrival timing. So "equal to solo" is not a usable criterion on this stack.
- Solo greedy hashes are identical between C and G. As a substitute check, `r4/e2/mixgsm.sh` ran GSM8K-50 greedy at conc8 while continuous t1 conc4 sampled traffic created mixed steps:
  - c2 50/50, g1 49/50 (the known item-12 flip), g2 50/50.

**Gates on the promoted live instance** (19:21 UTC, no env overrides):
- GSM8K-50 conc8: 49/50 (item 12) at 236 tok/s.
- **GSM8K-200 conc16: 196/200**. The misses (93, 100, 119, 184) are exactly the reference set. 302 tok/s.
- exact.py a8c3394af9d2.
- Lowest MemAvailable over the run: 20.3 GB.
- longctx was not rerun: the patch touches neither target numerics nor attention.

**Incidents.**
- 2 of 8 launches (c1 with the flag off; g2 try 1 with the flag on) hung in the FIRST, stock dspark capture: `rdma_ar rank 0: timeout waiting for 1 phase 1` at the first 160-row all_reduce, followed after 10 min by "CUDA error: unspecified launch failure" on all ranks.
- This is the same transient as r4-e1's E1-chk. Plain relaunches were clean. It is not caused by the patch.
- Logs: `r4/e2/c1fail/`, `r4/e2/g2fail1/`.

**Rollback** (from the Mac):
- Env only: `bash $S/conc/launch-conc.sh DSV41_DSPARK_GREEDY_SHARDED=0`. The site file is inert when the flag is off.
- Launcher: `cp -p ~/dsv41-tp8.sh.bak-r4e2gs ~/dsv41-tp8.sh` on all 8 (back to 709abe3d), then `bash $S/dsv41-launch-all.sh`.
- Site file: stop the service, then `bash $S/r4/rebased/greedy-sharded/uninstall.sh` (restores 0cc6e2bf). Do the launcher rollback first.

Raw results: `~/dsv41-opt-20261021/r4/e2/{chk,c1r,g1,c2,g2,c3,g3,live}`.
