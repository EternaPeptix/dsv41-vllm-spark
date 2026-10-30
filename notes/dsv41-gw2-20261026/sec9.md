
### 9. Final verification (17:10-18:10 UTC): config checked on 8/8, then the goal metrics on a pinned and a production instance at 2000 MHz. **Goal NOT met**

**Verdict.** Neither target is met. All numbers below were taken with telemetry off, at the 2000 MHz boot clock, on the promoted config (launcher 592793fd).
- **G1, single user: 100.39 tok/s in production** (gw2-FU-2: code 95.6 / prose 74.2 / math 131.4) against 120. **Not met: it needs +19.5%** (+19.6 tok/s). The pinned instance gave 100.21 (+19.7% needed), and gw2-FINAL1 (section 6) 100.50 (+19.4%).
- **G4, 4 concurrent sessions: not met.** Code, math and mixed were above 50 tok/s in every timed run on every instance; the lowest of them was 54.4. **The prose session was below 50 in every run:**
  - production gw2-FU-2: 46.44 / 47.45 / 45.35 (MIN 46.42), so **+7.7% is needed** (+10.2% for its worst run);
  - pinned gw2-FP: 47.98 / 46.85 / 47.74 (MIN 47.52), **+5.2%** (+6.7%);
  - production gw2-FINAL1 (section 6): 47.87 / 47.99 / 48.33 (MIN 48.06), **+4.0%** (+4.5%).

  The step-time-only estimate G4min_hat, which removes the effect of which prose text gets generated, is 47.2-47.9 on all three instances, so **+4.4..5.9%** is needed. The per-session mean is above 50 everywhere (58.8-61.3), but the goal requires every session.

#### 9.1 Config verified on 8/8
`fin/scripts/verify8.sh` ran before the step (17:12 UTC, on gw2-FINAL1) and after it (18:07 UTC, on the live gw2-FU-2). Apart from the instance start time, GPU readings and MemAvailable, the two outputs are identical.
- **Launcher** `~/dsv41-tp8.sh` **592793fd** on 8/8; backup `.bak-g2-sf` = **395bf5b2** on 8/8.
- **`--enable-prompt-tokens-details`** is in the launcher ARGS and in the running `vllm serve` cmdline on 8/8. The usage check returned `cached_tokens` 4160 on every instance.
- **Supervisor** `ops/rel/dsv41-relaunch.sh` **7bea1cf3** (`PRE_CLOCK_DEFAULT=2000`, `PRE_COMPACT_DEFAULT=0`); backup `.bak-g2-clock2300` = **30158e80**.
- **Site files**, identical on 8/8 (supervisor site manifest **0501bdcb13e0**):
  - `sm121_target_m32.py` ea4a8814 (`.bak-g2-fc2c-on-m32all` db659d80);
  - colltrim, inert: `cuda_communicator.py` 87bf77e3, `rdma_all_reduce.py` 26a5d834, `dsv41_l2_prefetch.py` 63737f9c (`.bak-g2-colltrim` 52fc8cf5 / 9056dcdf / 998af3d7);
  - stepovh: `dsv41_r4_small.py` c46e66bd, `sparse_swa.py` 854c3e5f, `model_state.py` f21ee2e6;
  - live rdma library `~/rdma_ar_dsv41/build-k1pl/rdma_ar_ext.so` 62daad6c.
- **Serving dependencies** are present on 8/8: `~/dsv41-stage/g1/m32-bw/` (661-664 MB) and `~/dsv41-stage/g2/fc2c-on-m32all/` (`PREP.ok` dcf5bc0909472121).
- **Running env**, the same on 8/8: `DSV41_MOE_FC2C=1 DSV41_L2_PREFETCH_WOA=1 DSV41_SYNC_FREE=argmax DSV41_EAGER_GLUE=all,swam DSV41_MOE_M32_G1=all DSV41_MOE_FC2_D05=1 DSV41_RDMA_PARLOAD=1 DSV41_L2_PREFETCH_PRUNE_SHARED=1 DSV41_TOPK_CANON=1 DSV41_AV_PROFILE=pinned`. `DSV41_AV_COST_JSON` is unset in production, so the startup profile is used ("pin miss").
- **Clock.**
  - The boot unit `dgx-spark-gpu-clock.service` (`nvidia-smi -lgc 0,2000`) is enabled and active on 8/8.
  - Each of the 3 launches logged `prelaunch (compact=0 clock=2000): ok on 8/8`.
  - The SM clock was read on 8/8 **before every measurement**: one `nvidia-smi` query per node in the gap before each phase, with no polling, so telemetry stayed off. That is 39 checks in total. Every reading was **1969-1989 MHz** with clock event reasons 0x0, and the GPU was at most 67 C.
  - No clock or power setting was raised in this step.

#### 9.2 Method
- **Driver** `fin/scripts/fin.sh` (Mac):
  1. Supervised launch with `--expect-launcher 592793fd` and `DSV41_GPU_CLOCK_MHZ=2000` passed explicitly, then the clock check.
  2. `gstart.sh` (exact.py first).
  3. Production only: a cold solo 1M with 1 Hz memwatch on 8/8. A launch that is not in the good class is relaunched.
  4. The `gw1suite.sh` phases, one at a time (`finsuite.sh` on the Mac, `finphase.sh` on spark-one, with the same commands as `gw1suite.sh`). The clock check runs before each phase. The phases are: matched C1 x5 (label pdab), G1 x3, G4 warm-up + 3 runs, bench conc 4/8/16/32 x2, matched C1 again, TTFT ~24K / ~198K, usage, GSM8K-50, full longctx, and exact.py #2.
- **Instances**, in launch order:
  - **gw2-FP**, pinned (`av-pin-pdfinal.json`, "pin hit"): healthy at +283 s. exact.py **a8c3394af9d2** with the pinned counters 905 / 471 / 2355 / 1373 and per-position 351/224/166/96/68 (**EXACT-OK**), and the same again after all the load.
  - **gw2-FU**, production: healthy at +366 s. exact.py a8c3394af9d2 (own profile, 911/465/2325/1373). Its cold 1M took **236.10 s, slow class** (late/early 1.873), so it was relaunched without further measurement.
  - **gw2-FU-2**, production: healthy at 17:51:44 UTC (+361 s). exact.py **a8c3394af9d2** (909/467/2335/1373, per-position 348/224/167/101/69), identical after all the load.
    - Its cold 1M took **183.75 s, good class** (1.161). The lowest MemAvailable during the prompt was 8.29 GB on spark-one and 8.9-9.5 GB on the other nodes, above the 6 GB floor.
    - **This instance is live.**
- **Outside activity.**
  - The `/metrics` sampler saw no outside traffic in any phase: the maximum running count always equalled the phase's own concurrency.
  - The Codex app kept working on the Mac. It wrapped up round 17 at 17:12-17:14 UTC, and at about 17:33 UTC made one `nice -n 19` read of source files from spark-one, during the pinned instance's longctx phase. Nothing reached the server.

#### 9.3 Results

| metric | **production gw2-FU-2** (unpinned, telemetry off) | pinned gw2-FP (telemetry off) | ref: production gw2-FINAL1 (section 6) |
|---|---|---|---|
| **G1 mean tok/s** (code / prose / math), 1024 tokens x3 each | **100.39** (95.63 / 74.16 / 131.38) | **100.21** (94.18 / 74.83 / 131.63) | **100.50** (94.90 / 75.56 / 131.05) |
| G1 per repetition | 100.23, 100.49, 100.45 | 100.27, 100.20, 100.18 | 100.40, 100.54, 100.57 |
| G1 AL code / prose / math | 2.603 / 1.792 / 3.607 | 2.558 / 1.807 / 3.607 | 2.590 / 1.847 / 3.594 |
| G1 output digest (3 reps; reps identical) | 939bb576cc1aad16 (yes) | bc1e8386878bd2b7 (yes) | bc1e8386878bd2b7 (yes) |
| **G4 per-session mean / prose MIN** (3 runs) | **60.34 / 46.42** | **58.84 / 47.52** | **61.30 / 48.06** |
| G4 by session: code / prose / math / mixed | 55.5 / 46.4 / 75.0 / 64.4 | 55.9 / 47.5 / 72.5 / 59.5 | 56.4 / 48.1 / 78.1 / 62.6 |
| G4 slowest session per run (session) | 46.44 (prose), 47.45 (prose), 45.35 (prose) | 47.98 (prose), 46.85 (prose), 47.74 (prose) | 47.87 (prose), 47.99 (prose), 48.33 (prose) |
| G4 session-runs at >= 50 tok/s | 9/12 | 9/12 | 9/12 |
| G4 common window mean / min | 57.89 / 39.93 | 56.44 / 40.43 | 59.12 / 40.11 |
| G4 aggregate tok/s | 176.9 | 178.7 | 180.1 |
| matched C1 coding / prose ms/step (pdab; AL) | **30.91 / 26.90** (AL 5.224 / 2.718); 2nd pass 30.85 / 26.71 | **30.72 / 26.58** (AL 5.224 / 2.718); 2nd pass 30.73 / 26.71 | **30.50 / 26.64** (AL 5.191 / 2.725); 2nd pass 30.56 / 26.61 |
| bench conc4 aggregate tok/s (runs; per-request mean / min) | **176.3** (177.9, 174.7; 60.6 / 45.3) | **174.3** (173.6, 175.0; 59.8 / 44.1) | **176.6** (177.8, 175.4; 59.7 / 47.4) |
| bench conc8 aggregate tok/s (runs; per-request mean / min) | 263.1 (261.8, 264.4; 43.4 / 34.7) | 259.3 (260.2, 258.3; 42.7 / 33.2) | 262.6 (259.6, 265.6; 43.1 / 33.5) |
| bench conc16 aggregate tok/s (runs; per-request mean / min) | 399.7 (399.2, 400.2; 34.4 / 25.7) | 397.4 (401.1, 393.6; 34.0 / 26.1) | 397.9 (392.8, 403.0; 34.9 / 26.1) |
| bench conc32 aggregate tok/s (runs; per-request mean / min) | 577.6 (578.9, 576.4; 24.6 / 18.1) | 586.2 (582.2, 590.1; 25.1 / 19.2) | 584.0 (586.9, 581.2; 24.9 / 19.1) |
| TTFT ~24K / ~198K | 4.09 s (23,868) / 33.30 s (197,737) | 5.46 s (23,934) / 33.46 s (197,825) | 4.04 s (23,800) / 33.28 s (197,861) |
| GSM8K-50 | 50/50 | 49/50 | 49/50 |
| full longctx | PASS | PASS | PASS |
| usage cached_tokens (repeat 4.2K prompt) | 4160 | 4160 | 4160 |
| exact.py #2 after all load (acc / drafts / dtok / gen; per-pos) | a8c3394af9d2 (909 / 467 / 2335 / 1373; 348/224/167/101/69) | a8c3394af9d2 (905 / 471 / 2355 / 1373; 351/224/166/96/68) | a8c3394af9d2 (909 / 467 / 2335 / 1373; 348/232/170/91/68) |
| foreign traffic (sampler, running > own concurrency) | none | none | none |

G4 runs, **production gw2-FU-2** (unpinned, telemetry off):

| run | code | prose | math | mixed | per-session mean | slowest (session) | common-window min | aggregate | batch AL |
|---|---|---|---|---|---|---|---|---|---|
| warm-up | 54.25 | 45.66 | 72.58 | 63.02 | 58.88 | 45.66 (prose) | 38.35 | 171.4 | 2.505 |
| 1 | 55.40 | 46.44 | 74.66 | 64.56 | 60.26 | 46.44 (prose) | 39.30 | 174.4 | 2.505 |
| 2 | 56.51 | 47.45 | 76.78 | 65.22 | 61.49 | 47.45 (prose) | 40.57 | 177.9 | 2.555 |
| 3 | 54.63 | 45.35 | 73.64 | 63.41 | 59.26 | 45.35 (prose) | 39.92 | 178.5 | 2.466 |

G4 runs, pinned gw2-FP (telemetry off):

| run | code | prose | math | mixed | per-session mean | slowest (session) | common-window min | aggregate | batch AL |
|---|---|---|---|---|---|---|---|---|---|
| warm-up | 54.62 | 45.53 | 75.90 | 59.02 | 58.77 | 45.53 (prose) | 40.55 | 179.0 | 2.475 |
| 1 | 58.34 | 47.98 | 68.62 | 56.47 | 57.85 | 47.98 (prose) | 40.47 | 178.1 | 2.512 |
| 2 | 54.38 | 46.85 | 75.74 | 60.00 | 59.24 | 46.85 (prose) | 40.76 | 178.9 | 2.400 |
| 3 | 54.90 | 47.74 | 73.11 | 61.93 | 59.42 | 47.74 (prose) | 40.06 | 178.9 | 2.458 |

G4 runs, ref: production gw2-FINAL1 (section 6):

| run | code | prose | math | mixed | per-session mean | slowest (session) | common-window min | aggregate | batch AL |
|---|---|---|---|---|---|---|---|---|---|
| warm-up | 50.32 | 46.70 | 74.60 | 60.72 | 58.09 | 46.70 (prose) | 38.83 | 175.4 | 2.457 |
| 1 | 54.51 | 47.87 | 76.92 | 66.75 | 61.51 | 47.87 (prose) | 38.99 | 179.5 | 2.572 |
| 2 | 57.46 | 47.99 | 79.46 | 60.30 | 61.30 | 47.99 (prose) | 40.90 | 179.8 | 2.547 |
| 3 | 57.24 | 48.33 | 78.00 | 60.72 | 61.07 | 48.33 (prose) | 40.45 | 181.0 | 2.556 |


Other per-instance facts:

| | production gw2-FU-2 | pinned gw2-FP | ref: gw2-FINAL1 |
|---|---|---|---|
| launch to healthy | +361 s (after one slow-class launch, gw2-FU) | +283 s | +364 s |
| exact.py, first request | a8c3394af9d2 (909/467/2335/1373) | a8c3394af9d2 (905/471/2355/1373) = pin | a8c3394af9d2 (909/467/2335/1373) |
| cold solo 1M (1,024,611 tokens) | **183.75 s**, good class (gw2-FU before it: 236.10 s, slow) | not run (once only) | 183.50 s, good class |
| MemAvailable minimum during the 1M | 8.29 GB (spark-one) | - | 7.8 GB (spark-one) |
| G4min_hat (step times only) | 47.57 | 47.88 | 47.23 |
| MemAvailable at the end of the suite (min over 8 nodes) | 8.4 GB | 9.3 GB | 8.1 GB |

- **The pinned instance reproduces the A/B stack.**
  - Against the stack arms X1/X2: G1 100.21 against 100.37 / 100.40 (they used 2 repetitions), G4min_hat 47.88 against 48.08 / 48.29, matched C1 30.72 / 26.58 against 30.68 / 26.59 and 30.60 / 26.59 ms.
  - Against the 5-instance control: G1 +1.3%, raw prose MIN +1.9% (47.52 against 46.65), G4min_hat +1.3%, G4 mean +0.5%.
  - The G1 outputs over 3 repetitions match the pinned digest bc1e8386878bd2b7.
- **Production numbers vary from launch to launch.** Each production launch measures its own AV profile at startup.
  - gw2-FINAL1 produced the pinned G1 outputs. gw2-FU-2 did not: its digest is 939bb576cc1aad16, because the prose output differs (AL 1.792 against 1.807 pinned).
  - Its raw G4 prose MIN was 46.42, against 48.06 on gw2-FINAL1. Most of that gap comes from which prose text was generated, not from step time. Its prose session took 283 steps per run against 276.7, and its G4min_hat (47.57) is slightly higher than gw2-FINAL1's (47.23).
  - G1, bench concurrency and C1 agree within about 1% across the three instances.
- The ~24K TTFT on gw2-FP (5.46 s) is the first long prefill after start. On the production instances it came after the 1M probe, so it is warm (~4.1 s).
- GSM8K-50 was 50/50 on gw2-FU-2 and 49/50 on gw2-FP. The pinned miss is idx 12, the usual borderline item.

#### 9.4 Distance to the goal (2000 MHz, no clock or power increase)
| target | gw2-FU-2 (production, live) | gw2-FP (pinned) | gw2-FINAL1 (production, section 6) |
|---|---|---|---|
| G1 mean >= 120 tok/s | 100.39: **+19.5%** | 100.21: +19.7% | 100.50: +19.4% |
| G4: every session >= 50 tok/s in every run (prose MIN over 3 runs; worst run) | 46.42: **+7.7%**; 45.35: +10.2% | 47.52: +5.2%; 46.85: +6.7% | 48.06: +4.0%; 47.87: +4.5% |
| G4min_hat (step time only) | 47.57: +5.1% | 47.88: +4.4% | 47.23: +5.9% |

- **G4.** The exact levers still open (section 8) are each worth tenths of a percent: drafter MoE on the M32 path +0.5-0.9%, column gather +0.3%, fused AV bookkeeping +0.15%. Together they do not close a 4-8% prose gap. Closing it needs either the prose-priority verify width or a structural lever. The verify width is the user's decision: it is a fairness trade, it is not bitwise at 4 users, and it needs a new pin.
- **G1.** About +20% is needed, which takes an acceptance lever (prose AL 1.79-1.85) or MoE weight streaming close to the Marlin bandwidth on the same bytes (section 8). The clock path is closed by the user's decision.
- **Default AV pin (still the user's decision).** A pin would make production reproducible across launches: the raw prose MIN now moves by about ±2% between launches, and gw2-FU-2 differs from the pinned G1 outputs. It would also shorten launches (283 s against 361-366 s). It does not close the gap.

#### 9.5 Live state after this step
- **gw2-FU-2**: production, AV unpinned, launcher 592793fd, supervisor 7bea1cf3, clock 0-2000 MHz.
  - Healthy since 17:51:44 UTC on 8/8, with 0 restarts and 0 ERROR/Traceback; `/health` returns 200.
  - Its config matches 9.1, re-checked after the suite.
- No telemetry, memwatch, sampler or bench process is left on 8/8.
- Rollback is unchanged from section 7.
- **Files.**
  - Mac scratchpad `gw2/fin/`:
    - scripts: `fin.sh`, `finsuite.sh`, `finphase.sh`, `finsampler.sh`, `clk8.sh`, `verify8.sh`, `node-verify.sh`, `finsumm.py`, `fintable.py`;
    - logs and checks: `fin.log`, `verify-pre.txt`, `verify-post.txt`, `clk-*.txt`;
    - results: `runs/<instance>/` (with `finsumm.json` / `finsumm.md`), `gw2-FP/`, `gw2-FU/` and `gw2-FU-2/` (relaunch logs, exact.py, 1M with memwatch CSVs), `tables-final.md`.
  - spark-one `~/dsv41-gw2-20261026/runs/`: `fin-gw2-FP`, `fin-gw2-FU-2`, `gw2-FP-start`, `gw2-FU-start`, `gw2-FU-2-start`, `p1m-gw2-FU`, `p1m-gw2-FU-2`.
