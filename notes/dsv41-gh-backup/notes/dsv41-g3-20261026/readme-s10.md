
### 10. Final verification (23:00-23:40 UTC): config checked on 8/8, then the goal metrics on a pinned and a production instance at 2000 MHz. **Goal NOT met**

**Verdict.** Neither target is met. Every number below was taken with telemetry off, at the 2000 MHz boot clock, on the promoted config (launcher b0cf94d0).
- **G1, single user: 101.51 tok/s in production** (g3-FU: code 95.70 / prose 76.56 / math 132.28) against 120. **Not met: +18.2% is needed** (+18.5 tok/s). The pinned instance g3-FP gave 101.71 (+18.0%) and g3-FINAL (section 7) 102.26 (+17.3%).
- **G4, 4 concurrent sessions: not met.** Code, math and mixed were above 50 tok/s in every timed run on all three instances; the lowest of them was 55.2 in this step (52.7 on g3-FINAL). **The prose session was below 50 in every run:**
  - production g3-FU: 49.15 / 49.52 / 47.26 (MIN 48.64), so **+2.8% is needed** (+5.8% for its worst run);
  - pinned g3-FP: 48.82 / 47.84 / 47.22 (MIN 47.96), **+4.3%** (+5.9%);
  - production g3-FINAL (section 7): 48.70 / 45.80 / 49.10 (MIN 47.87), **+4.5%** (+9.2%).

  The step-time-only estimate G4min_hat is 47.8-48.5 on the three instances, so **+3.0..4.7%** is needed: about **-1.1..-1.7 ms on every decode step**, or -2.1..-3.2 ms on the 4-session steps alone (now 44.7-45.6 ms). The per-session mean is above 50 everywhere (60.3-61.7), but the goal requires every session.
- **Exact.** Both instances gave exact.py a8c3394af9d2 at start and again after all the load. The pinned one had the pin counters 909 / 467 / 2335 / 1373 with per-position 348/224/167/101/69, and its first two G1 repetitions gave the A/B digest 69ad872124100141.
- **Power.** No clock or power setting was raised. Energy per token is flat: the production power phase gave 1.540 J/token single user and 1.054 aggregate, against 1.561 / 1.064 for the window's controls (section 6).

#### 10.1 Config verified on 8/8
`fin/scripts/verify8.sh` ran before the step (`verify-fpre`, 23:00 UTC, on g3-FINAL) and after it (`verify-fpost`, 23:38 UTC, on the live g3-FU). Apart from the instance start time, GPU readings and MemAvailable, the two outputs are identical, and they match section 7's `verify-post`.
- **Launcher** `~/dsv41-tp8.sh` **b0cf94d0** on 8/8. Backups: `.bak-g3-xstack` = 592793fd and `.bak-g2-sf` = 395bf5b2, on 8/8.
- **`--enable-prompt-tokens-details`** is in the launcher and in the running `vllm serve` cmdline on 8/8. The usage check returned `cached_tokens` 4160 on both instances.
- **Supervisor** `ops/rel/dsv41-relaunch.sh` **7bea1cf3** (`PRE_CLOCK_DEFAULT=2000`, `PRE_COMPACT_DEFAULT=0`); backup `.bak-g2-clock2300` = 30158e80. Site manifest **b1150ecfcf49**. Both launches logged "site manifest b1150ecfcf49 and launcher b0cf94d0 identical on 8/8" and "engram check: 8/8 OK".
- **Site files**, identical on 8/8:
  - goal window 3: `sparse_attn_indexer.py` 481905c2, `flashinfer_sparse.py` b059bdf3, `mla/indexer.py` d9f4c5c9 (`.bak-g3-codex-adopt` 7112c01d / c8a2da55 / 4753707e), `dsv41_r3_dense.py` 1f1c0456 (`.bak-g3-dhtac` 993023f2); `deep_gemm_utils.py` a4a0486a (mfront not installed); pin `~/dsv41-stage/g3/pin/av-pin-g3.json` 1c98cf9b;
  - earlier windows: `sm121_target_m32.py` ea4a8814 (`.bak-g2-fc2c-on-m32all` db659d80); colltrim, inert: `cuda_communicator.py` 87bf77e3, `rdma_all_reduce.py` 26a5d834, `dsv41_l2_prefetch.py` 63737f9c (`.bak-g2-colltrim` 52fc8cf5 / 9056dcdf / 998af3d7); stepovh: `dsv41_r4_small.py` c46e66bd, `sparse_swa.py` 854c3e5f, `model_state.py` f21ee2e6; `rdma_ar_ext.so` 62daad6c.
- **Serving dependencies** are present on 8/8: `~/dsv41-stage/g1/m32-bw/` (661-664 MB) and `~/dsv41-stage/g2/fc2c-on-m32all/` (`PREP.ok` dcf5bc0909472121).
- **Running env** of g3-FU, the same on 8/8: `DSV41_AV_PROFILE=pinned DSV41_DECODE_REMAP_REUSE=1 DSV41_DRAFT_HEAD_TACTIC=1 DSV41_DRAFT_HEAD_TACTIC_MAP=17-29:1 DSV41_EAGER_GLUE=all,swam DSV41_IDX_OMIT_T2S=1 DSV41_L2_PREFETCH_PRUNE_SHARED=1 DSV41_L2_PREFETCH_WOA=1 DSV41_MOE_FC2C=1 DSV41_MOE_FC2_D05=1 DSV41_MOE_M32_G1=all DSV41_RDMA_PARLOAD=1 DSV41_SYNC_FREE=argmax DSV41_TOPK_CANON=1 DSV41_TOPK_CANON_FUSED=1`. `DSV41_AV_COST_JSON` is unset in production, so the startup profile is used ("pin miss"). On both instances the rank logs on 8/8 carry "fused CANON kernel in use" and "13 of 13 mapped row counts pinned (17-29:1)".
- **Clock.**
  - The boot unit `dgx-spark-gpu-clock.service` (`nvidia-smi -lgc 0,2000`) is enabled and active on 8/8.
  - Both launches logged `prelaunch (compact=0 clock=2000): ok on 8/8`.
  - The SM clock was read on 8/8 **before every measurement**: one `nvidia-smi` query per node, no polling, so telemetry stayed off. That is **38 checks (304 node readings), all at 1969-1989 MHz** with event reasons 0x0, and at most 66 C. The power phase's 2 s telemetry peaked at 1989 MHz and 67 C.
  - No clock or power setting was changed.

#### 10.2 Method
- **Driver** `fin/scripts/final.sh` (Mac): `verify8.sh fpre`, then an outside-client logger on spark-one, then `fin.sh g3-FP pinned`, `fin.sh g3-FU prod 2`, the power phase (`finpow.sh`) on the production instance, and `verify8.sh fpost`. `fin.sh`, `finsuite.sh` and `finphase.sh` are the section 7 drivers, i.e. goal window 2's final method:
  1. A supervised launch with `--expect-launcher b0cf94d0` and `DSV41_GPU_CLOCK_MHZ=2000`, then the clock check.
  2. `gstart.sh` (exact.py first).
  3. Production only: a cold solo 1M with 1 Hz memwatch on 8/8, and a relaunch if it is not in the good class.
  4. The `gw1suite.sh` phases one at a time, with the clock check before each: matched C1 x5 (pdab), G1 x3, G4 warm-up + 3 runs, bench conc 4/8/16/32 x2, matched C1 again, TTFT ~24K / ~198K, usage, GSM8K-50, full longctx, exact.py #2.
- **Instances**, in launch order:
  - **g3-FP**, pinned (`av-pin-g3.json`, "pin hit"): healthy at 23:05:16 UTC (+276 s, first try). exact.py **a8c3394af9d2** with the pin counters 909 / 467 / 2335 / 1373 and per-position 348/224/167/101/69 (**EXACT-OK**), the same again after all the load.
  - **g3-FU**, production: healthy at 23:22:34 UTC (+360 s, first try). exact.py **a8c3394af9d2** (own profile: 909 / 467 / 2335 / 1373, per-position 348/233/171/90/67), the same again after all the load.
    - Its cold 1M took **183.70 s, good class** (late/early 1.16), at the first try. The lowest MemAvailable during the prompt was 8.27 GB on spark-one and 8.9-10.1 GB on the other nodes, above the 6 GB floor.
    - **This instance is live.**
- **Power phase** (after g3-FU's timed suite; 2 s telemetry on 8/8; 10 s idle, G1 x1, G4 warm-up + 2 runs). It measures watts only and is never used for timing.
- **Outside activity: none reached the server.**
  - The `/metrics` sampler showed the maximum running count equal to the phase's own concurrency in every phase of both instances. The suites completed 258 (g3-FU) and 257 (g3-FP) requests against the clean reference of 258; the per-phase differences of 1-2 are phase-boundary shifts.
  - The 1 s logger on spark-one (`connlog-fin.sh`) watched for clients of 127.0.0.1:8000 other than python/curl, and for any client of the LAN or tailnet listeners. It saw no LAN or tailnet client.
  - Its three entries (23:11:31, 23:26:13, 23:31:50) were our own requests: two `/metrics` scrapes and one matched-bench POST, whose ports 55922 / 55120 / 47390 match the access log. They were listed without a process because `ss -p` races short connections.
  - The Mac logger on the Codex tunnel (127.0.0.1:18000, from 23:17) saw no client.

#### 10.3 Results

| metric | **production g3-FU** (unpinned, telemetry off) | pinned g3-FP (telemetry off) | ref: production g3-FINAL (section 7) |
|---|---|---|---|
| **G1 mean tok/s** (code / prose / math), 1024 tokens x3 each | **101.51** (95.70 / 76.56 / 132.28) | **101.71** (96.71 / 75.32 / 133.10) | **102.26** (95.93 / 77.04 / 133.82) |
| G1 per repetition | 101.56, 101.49, 101.48 | 101.80, 101.68, 101.65 | 102.19, 102.34, 102.25 |
| G1 AL code / prose / math | 2.590 / 1.847 / 3.594 | 2.603 / 1.792 / 3.607 | 2.616 / 1.889 / 3.658 |
| G1 output digest (3 reps; reps identical) | bc1e8386878bd2b7 (yes) | 939bb576cc1aad16 (yes) | eb6d32aab70a0191 (yes) |
| **G4 per-session mean / prose MIN** (3 runs) | **61.65 / 48.64** | **60.34 / 47.96** | **60.86 / 47.87** |
| G4 by session: code / prose / math / mixed | 58.8 / 48.6 / 75.3 / 63.9 | 56.3 / 48.0 / 75.6 / 61.5 | 55.2 / 47.9 / 76.0 / 64.4 |
| G4 slowest session per run (session) | 49.15 (prose), 49.52 (prose), 47.26 (prose) | 48.82 (prose), 47.84 (prose), 47.22 (prose) | 48.70 (prose), 45.80 (prose), 49.10 (prose) |
| G4 session-runs at >= 50 tok/s | 9/12 | 9/12 | 9/12 |
| G4 common window mean / min | 59.02 / 40.45 | 57.64 / 41.51 | 58.92 / 40.62 |
| G4 aggregate tok/s | 182.7 | 181.8 | 180.8 |
| matched C1 coding / prose ms/step (pdab; AL) | **30.32 / 26.33** (AL 5.191 / 2.725); 2nd pass 30.25 / 26.36 | **30.71 / 26.40** (AL 5.224 / 2.718); 2nd pass 30.52 / 26.41 | **30.59 / 26.58** (AL 5.224 / 2.759); 2nd pass 30.52 / 26.64 |
| bench conc4 aggregate tok/s (runs; per-request mean / min) | **180.8** (178.9, 182.7; 60.7 / 46.5) | **180.4** (182.6, 178.3; 60.4 / 46.5) | **180.9** (180.8, 180.9; 61.2 / 48.3) |
| bench conc8 aggregate tok/s (runs; per-request mean / min) | 264.3 (264.8, 263.8; 43.9 / 34.1) | 262.6 (263.1, 262.1; 43.0 / 33.3) | 262.8 (262.7, 262.9; 44.3 / 34.8) |
| bench conc16 aggregate tok/s (runs; per-request mean / min) | 397.8 (402.8, 392.8; 34.7 / 25.4) | 399.1 (396.8, 401.4; 34.1 / 25.0) | 394.2 (397.9, 390.5; 34.2 / 25.8) |
| bench conc32 aggregate tok/s (runs; per-request mean / min) | 580.7 (584.2, 577.2; 24.9 / 18.7) | 590.8 (590.4, 591.2; 25.5 / 18.9) | 585.8 (587.8, 583.9; 25.1 / 18.5) |
| TTFT ~24K / ~198K | 4.05 s (23,901) / 33.29 s (197,810) | 5.11 s (23,821) / 33.42 s (197,691) | 4.04 s (23,863) / 33.31 s (198,008) |
| GSM8K-50 | 50/50 | 49/50 | 49/50 |
| full longctx | PASS | PASS | PASS |
| usage cached_tokens (repeat 4.2K prompt) | 4160 | 4160 | 4160 |
| exact.py #2 after all load (acc / drafts / dtok / gen; per-pos) | a8c3394af9d2 (909 / 467 / 2335 / 1373; 348/233/171/90/67) | a8c3394af9d2 (909 / 467 / 2335 / 1373; 348/224/167/101/69) | a8c3394af9d2 (921 / 455 / 2275 / 1373; 342/234/175/100/70) |
| foreign traffic (sampler, running > own concurrency) | none | none | none |
| G4min_hat (step times only) / 4-session step ms | 48.43 / 44.77 | 48.54 / 44.69 | 47.77 / 45.59 |
| G4 per-phase step ms (4 / 3 / 2 / 1 sessions); prose steps per run | 44.77 / 40.72 / 32.52 / 24.24; 281.3 | 44.69 / 40.45 / 34.14 / 23.46; 281 | 45.59 / 39.11 / 33.03 / 25.09; 276.3 |
| G1 outputs per type (sha8 code / prose / math) | 596b478a / 923a46be / 92e5d8cf | 596b478a / e8a524f3 / 92e5d8cf | 596b478a / e0e0664e / 92e5d8cf |
| G1 digest of the first 2 repetitions (A/B pinned = 69ad872124100141) | 9a18f802d930f6e7 | 69ad872124100141 | d74940fefe15b2ef |
| exact.py, first request (acc / drafts / dtok / gen; per-pos; AV) | a8c3394af9d2 (909/467/2335/1373; 348/233/171/90/67; pin miss) | a8c3394af9d2 (909/467/2335/1373; 348/224/167/101/69; pin hit) | a8c3394af9d2 (921/455/2275/1373; 342/234/175/100/70; pin miss) |
| requests finished during the suite (clean reference 258) | 258 (extra vs ref: {'usage': 1.0, 'gsm': -2.0, 'lcfull': 1.0}) | 257 (extra vs ref: {'ttft': -1.0, 'usage': 1.0, 'exact2': -1.0}) | 258 |

G4 runs, **production g3-FU** (unpinned, telemetry off):

| run | code | prose | math | mixed | per-session mean | slowest (session) | common-window min | aggregate | batch AL |
|---|---|---|---|---|---|---|---|---|---|
| warm-up | 57.83 | 47.80 | 77.17 | 64.26 | 61.77 | 47.80 (prose) | 39.87 | 178.9 | 2.583 |
| 1 | 60.53 | 49.15 | 77.82 | 66.78 | 63.57 | 49.15 (prose) | 39.29 | 182.7 | 2.656 |
| 2 | 59.01 | 49.52 | 73.13 | 65.98 | 61.91 | 49.52 (prose) | 41.15 | 185.5 | 2.570 |
| 3 | 56.75 | 47.26 | 74.91 | 59.01 | 59.48 | 47.26 (prose) | 40.90 | 179.9 | 2.369 |

G4 runs, pinned g3-FP (telemetry off):

| run | code | prose | math | mixed | per-session mean | slowest (session) | common-window min | aggregate | batch AL |
|---|---|---|---|---|---|---|---|---|---|
| warm-up | 55.87 | 47.36 | 69.45 | 60.62 | 58.33 | 47.36 (prose) | 40.51 | 177.8 | 2.470 |
| 1 | 55.16 | 48.82 | 73.49 | 61.55 | 59.75 | 48.82 (prose) | 41.49 | 183.1 | 2.479 |
| 2 | 55.70 | 47.84 | 74.09 | 61.68 | 59.83 | 47.84 (prose) | 41.91 | 182.0 | 2.427 |
| 3 | 57.95 | 47.22 | 79.33 | 61.21 | 61.43 | 47.22 (prose) | 41.14 | 180.3 | 2.466 |

G4 runs, ref: production g3-FINAL (section 7):

| run | code | prose | math | mixed | per-session mean | slowest (session) | common-window min | aggregate | batch AL |
|---|---|---|---|---|---|---|---|---|---|
| warm-up | 56.01 | 48.13 | 75.86 | 62.33 | 60.58 | 48.13 (prose) | 39.64 | 180.3 | 2.582 |
| 1 | 52.66 | 48.70 | 76.36 | 67.31 | 61.26 | 48.70 (prose) | 42.44 | 182.7 | 2.528 |
| 2 | 55.77 | 45.80 | 73.28 | 61.78 | 59.16 | 45.80 (prose) | 39.01 | 176.0 | 2.469 |
| 3 | 57.30 | 49.10 | 78.25 | 63.98 | 62.16 | 49.10 (prose) | 40.42 | 183.8 | 2.582 |

Other per-instance facts:

| | g3-FU | g3-FP | g3-FINAL |
|---|---|---|---|
| healthy at (UTC) / launch to healthy | 23:22:34 / +360 s | 23:05:16 / +276 s | 22:36:32 / +357 s |
| cold solo 1M (1,024,611 tokens) | 183.703 s, good class (1.16) | not run (once only) | 183.071 s, good class (1.16) |
| MemAvailable minimum during the 1M | 8.27 GB (spark-one); others 8.9-10.1 GB | - | 8.32 GB (spark-one); others 8.9-9.5 GB |
| MemAvailable at the end of the suite (min over 8 nodes) | 8.3 GB | 9.4 GB | 8.4 GB |

| power phase (2 s telemetry on 8/8, after the timed suite) | production g3-FU | window controls C (n=5, section 6) | stack X (n=2, section 6) |
|---|---|---|---|
| fleet GPU W idle / G1 / G4 | 72.8 / 154.9 / 188.3 | 72.0 / 155.5 / 187.0 | 72.0 / 157.7 / 190.8 |
| spark-one whole-node W G1 / G4 | 92.2 / 101.9 | 90.4 / 101.0 | 91.0 / 100.9 |
| J/token, G1 single user / G4 aggregate | **1.540 / 1.054** (G1 100.6, G4 178.6 tok/s in that phase) | 1.561 / 1.064 | 1.565 / 1.073 |
| SM MHz max / GPU max C | 1989 / 67 | 1989-1995 / 68 | 1989 / 68 |

- **The pinned instance reproduces the A/B stack and the launch the pin was made from.**
  - Its first two G1 repetitions give the A/B digest 69ad872124100141. All three give **939bb576cc1aad16, the digest of gw2-FU-2** (goal window 2's production instance on launcher 592793fd, whose startup profile became `av-pin-g3.json`), with the same ALs (2.603 / 1.792 / 3.607). So the promoted stack under that profile produces exactly the old config's outputs.
  - With the outputs identical, the comparison with gw2-FU-2 is pure step time: G1 101.71 against 100.39 is **+1.31%** (code +1.1%, prose +1.6%, math +1.3%), and G4min_hat 48.54 against 47.57 is **+2.0%**. Both agree with the pinned A/B result for the stack X (G1 +1.17%, G4min_hat +2.07%).
- **Production g3-FU produced gw2-FINAL1's outputs** (digest bc1e8386878bd2b7, same ALs 2.590 / 1.847 / 3.594): G1 101.51 against 100.50 (+1.0%), G4min_hat 48.43 against 47.23 (+2.5%).
- **Production still moves with the startup AV profile.** The two production launches since the promotion gave G1 101.5 / 102.3 and raw prose MIN 48.6 / 47.9 (prose steps per G4 run 281.3 / 276.3). G4min_hat, which removes the text effect, is 48.4 / 47.8.
- The ~24K TTFT on g3-FP (5.11 s) is the first long prefill after start. On the production instances it came after the 1M probe, so it is warm (~4.05 s).
- GSM8K-50 was 50/50 on g3-FU and 49/50 on g3-FP and g3-FINAL. The miss is idx 12, the usual borderline item.

#### 10.4 Distance to the goal and next exact levers (2000 MHz, no clock or power increase)

| target | g3-FU | g3-FP | g3-FINAL |
|---|---|---|---|
| G1 mean >= 120 tok/s | 101.51: +18.2% | 101.71: +18.0% | 102.26: +17.3% |
| G4 every session >= 50 in every run (prose MIN over 3 runs; worst run) | 48.64: +2.8%; 47.26: +5.8% | 47.96: +4.3%; 47.22: +5.9% | 47.87: +4.5%; 45.80: +9.2% |
| G4min_hat (step time only) | 48.43: +3.2% | 48.54: +3.0% | 47.77: +4.7% |

- **G1 >= 120 tok/s: NOT met.** At 101.5-102.3 tok/s, **+17.3..18.2%** is needed. At today's acceptance (prose AL 1.79-1.89) that is about -15% single-user step time, about 3.7-4.2 ms per step. The levers below are sized for 4-user rows and give far less at one user. 120 needs either an acceptance lever (a better drafter changes the verify shapes, so it is not exact under the current definition and needs a new pin and anchor) or MoE weight streaming much closer to the bound (FC1 already streams at ~94%).
- **G4 >= 50 tok/s on every session: NOT met.** The prose session needs **+2.8..4.5%** on the 3-run MIN, up to +9.2% for the worst run, and **+3.0..4.7% on step time** (G4min_hat 47.8-48.5). That is -1.1..-1.7 ms on every decode step, or -2.1..-3.2 ms on the 4-session steps alone.
- **Next exact levers**, by size at 4 users. Each keeps the target numerics bitwise and needs its own GPU bitwise test and a pinned A/B under `av-pin-g3.json`:
  1. MoE FC1 -> FC2 per-expert pipelining: -0.4..-1.0 ms per 4-session step. At M=12-24 the live M32 `all` + FC2C path is 7-8% slower than unpadded Marlin on the same bytes (~1.2-1.7 ms per step) and 11-14% above the streaming bound (section 5).
  2. Contiguous FC2C layout: -0.5..-0.8 ms.
  3. An L2-prefetch plan that keeps q_b prefetched above 8 rows (the replicated qkv_a fills the 6 MB window): up to -0.86 ms on steps with 2 or more sessions.
  4. Drafter MoE on the M32 path: ~-0.45 ms at bs4.
  5. mfront, once the mixed live/fused-graph illegal-address crash is root-caused (`CUDA_LAUNCH_BLOCKING=1` + compute-sanitizer, bisect with `DSV41_MOE_FC2C=0`): ~-0.26..-0.30 ms per step.
  6. The dhtac fix applied to the DSpark main_proj split, which has the same autotune-cache tactic-0 choice at 17-32 rows: not measured yet; check the cache first.
  7. rdma_ar P1 doorbell / P2 fail-fast: ~-0.1..-0.2 ms.

  Together that is roughly 2.5-3.5 ms of the 44.7-45.6 ms 4-session step. On paper that covers the G4 step-time gap if most of them land; the worst prose runs need more margin. Also fix the dhtac audit reporter (run its `copy_` under `torch.inference_mode()`) so that a live 4-user audit count backs D's exactness.
- **User decisions (not exact, or policy), unchanged:** a default AV pin (makes production reproducible and launches shorter, 276 s against 357-360 s, but does not close the gap); the prose-priority verify width (a fairness trade, not bitwise at 4 users); unpadded Marlin (not exact); clocks (closed at 2000 MHz); authentication for the LAN and tailnet API (`dsv41-lanproxy` has none).

#### 10.5 Live state after this step
- **g3-FU** is live: production, AV unpinned, launcher b0cf94d0, supervisor 7bea1cf3, clock 0-2000 MHz.
  - Healthy since 23:22:34 UTC on 8/8, with 0 restarts and 0 ERROR/Traceback. `/health` returns 200 locally and on the LAN address. `dsv41-lanproxy` is active.
  - Its config matches 10.1, re-checked after the suite and the power phase.
- No telemetry, memwatch, sampler, logger or bench process is left on 8/8 or on the Mac.
- Rollback is unchanged from section 8 (the live instance is now g3-FU instead of g3-FINAL, with the same config).
- **Files.**
  - Mac scratchpad `g3w/fin/`:
    - scripts: `final.sh`, `finpow.sh`, `connlog-fin.sh`, `finforeign.py`, `finaltab.py`, plus the section 7 drivers;
    - logs and checks: `fin.log`, `final.out`, `verify-fpre.txt`, `verify-fpost.txt`, `clk-g3-FP.txt`, `clk-g3-FU.txt`, `connlog-fin.txt`;
    - results: `runs/g3-FP/`, `runs/g3-FU/`, `runs/g3-FU-pow/`, `g3-FP/` and `g3-FU/` (relaunch logs, exact.py, 1M with memwatch CSVs), `pow-final.tsv`, `tables-final.md`.
  - spark-one `~/dsv41-g3-20261026/`: `runs/fin-g3-FP`, `runs/fin-g3-FU`, `runs/g3-FP-start`, `runs/g3-FU-start`, `runs/p1m-g3-FU`, `runs/g3-FU-pow`, `connlog-fin.txt`.
