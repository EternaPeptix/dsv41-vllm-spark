
## Goal window 6 (2026-10-29 01:44-09:56 UTC): exact decode levers. X (the drafter's routed MoE on the M32 kernel, `DSV41_DRAFT_MOE_M32=1`) passed the pinned chain (G1 +0.99 / +1.06%, G4min_hat +1.48 / +1.29%, every pinned output bitwise) and was promoted (launcher afa97ae6 -> d625c443), but the Phase D production check STOPPED it: unpinned, X changes the drafter cost the boot-time AV profile measures, and the G1 prose near-tie landed on an UNKNOWN outcome (33ae975a). Launcher ROLLED BACK to **afa97ae6** (X installed, flag off); keeping X in production is a user decision. Q and H NO-GO in the stopped window, L SMALL. Goal NOT met: production G1 104.3 of 120 tok/s (+15.1%); G4 prose session 48.6 in production (+3.0%)

**Verdict: nothing new is live; one exact lever is ready for a user decision.** All numbers were taken at the 2000 MHz boot clock with nothing raised (SM 1969-1989 MHz at every check on 8/8). The window-5 foreign-traffic rule applied to every unit (30 s idle before each, a 0.25-0.5 s sampler, an overlapped unit is NOT PROVEN and rerun); the window saw no foreign request in any timed unit. `dsv41-lanproxy` was not touched and no client was blocked. The cluster lock was held by claude for the whole window; every cluster action has a line in `AGENTS_LOG.md`.
- **X under the AV pin is bitwise:** 10/10 pinned instances EXACT-PASS, shadow verify 8/8 (0 mismatching words in 7,394 x 3 drafter MoE calls per rank), drafts identical (matched C1 AL 5.191 / 2.725 on every instance). Pinned: G1 105.23 / 105.30 (controls 104.18-104.29), G4min_hat 50.43 / 50.42 (controls 49.41-49.87), 4-session step 43.10 / 43.09 ms (controls 43.62-44.08).
- **X in production (unpinned) is not output-identical:** on launcher d625c443 exact.py stayed a8c3394af9d2 (start and end) and G1 code / math stayed bitwise (596b478a / 92e5d8cf), but the G1 prose output was **33ae975a in all 3 repetitions** (digest 247b90ce9f0b492e), not one of the known unpinned tie outcomes 923a46be / e8a524f3 / a22e1d39. Mechanism: the unpinned AV profile measures the drafter at boot, and X makes it cheaper (draft_ms 4.20 -> 3.92 at 1 row, 6.79 -> 5.71 at 4, 10.51 -> 8.85 at 12, 15.61 -> 13.47 at 32), so the AV policy picks other verify layouts and the prose near-tie flips. Under the pin the cost table is fixed, so nothing moves. Every unit was clean: this is a CLEAN mismatch, not noise, and it was not retried.
- **That X-on production instance measured:** G1 104.20 (98.94 / 77.30 / 136.36: code +1.1%, math +0.8%, prose -2.0% on the new tie path), **G4 metric 49.99** (prose per run 49.13 / 50.35 / 50.48; 11/12 session-runs >= 50), G4min_hat 50.72, 4-session step 42.99 ms.
- **Production now (w6-PROD-rb, suite w6-Drb, launcher afa97ae6, X off, AV unpinned):** G1 **104.26** (code 98.36 / prose 78.47 / math 135.96), G4 metric **48.56** (prose per run 48.04 / 48.09 / 49.55), G4min_hat 50.25, 4-session step 43.68 ms; exact.py a8c3394af9d2 at start and end (910/466/2330/1373), G1 code / math bitwise and prose on the known unpinned tie outcome a22e1d39 (digest 51d0a6537e239757): EXACT-PASS.
- **Regression gate (kit 08) on the final production: PASS in 596 s, foreign-traffic audit CLEAN.**
- **API downtime: 84.3 min in total**: window A 49.5 min over its 3 attempts, Phase B 27.6 min in 11 relaunches, Phase D 7.2 min in 2 relaunches (216 + 217 s).

**User decision (X).** Options: (a) keep X in production and accept 33ae975a as a fourth known unpinned prose tie outcome (code / math bitwise, exact.py a8; the production G4 metric reached 49.99); (b) make the AV pin the production default (open since window 5): then X is bitwise in production as well (the pin's draft-cost table is slightly pessimistic under X); (c) a fresh pin that includes X's costs (a new anchor); (d) leave X off. Re-enabling X is one step: `cp ~/dsv41-stage/w6/promote/dsv41-tp8.sh.d625c443 ~/dsv41-tp8.sh` on 8/8 (or `W6_PROMOTE_OK=1 bash $W/driver/promote-w6.sh X`, which rebuilds the same d625c443), then a supervised relaunch with `--expect-launcher d625c443`, or for one launch only the supervisor argument `DSV41_DRAFT_MOE_M32=1` on afa97ae6.

### 1. What is live now (verified on 8/8 09:55 UTC)

- **Launcher** `~/dsv41-tp8.sh` **afa97ae6** on 8/8 (unchanged by the window). The promoted d625c443 = afa97ae6 + 2 lines after `export DSV41_MOE_FC2C=` was installed 08:59-09:27 and the instance launched from it served 09:06-09:28; it is kept as `~/dsv41-stage/w6/promote/dsv41-tp8.sh.d625c443` on spark-one:
  ```sh
  # goal window 6 (RUNBOOK 7, promoted X): exact levers, default on; '=0' / '=1024' as a supervisor argument turns one off for a launch (RUNBOOK 9)
  export DSV41_DRAFT_MOE_M32=${DSV41_DRAFT_MOE_M32:-1}
  ```
  `~/dsv41-tp8.sh.bak-w6` = afa97ae6 on 8/8 (new; older backups unchanged).
- **Site files installed, flag off** (window A step A6, 04:54 UTC, `w6-bundle.sh install drm32`, service stopped):
  - `vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py` 5162dc0f -> **9364f121** (backup `deep_gemm_moe.py.bak-w6` = 5162dc0f; staged base copy `~/dsv41-stage/w6/bundle/deep_gemm_moe.py.base`).
  - new `vllm/model_executor/layers/fused_moe/experts/dsv41_draft_m32.py` **a17d574a** (`DSV41_DRAFT_MOE_M32` = 0 / 1 / verify, anything else is a startup error; `DSV41_DRAFT_M32_MAX_M` accepted 1..160 only).
  - With the flag unset both behave exactly as the live base (no new code path runs): A11, the 5 pinned C instances and this section's production check are bitwise on them. Neither is a supervisor-manifest file. The A7 prejit added the 2 drafter JIT entries (`sm120_m_grouped_fp8_fp4_gemm_contiguous_1d1d.3f71c0e0...` and `.74a66dfd...`) to `~/dsv41-stage/g1/m32-bw/jit/all-base/cache` on 8/8; no existing entry changed.
- **Supervisor** 7bea1cf3, preflight manifest **0c96a727fb81** (unchanged; it was 26db72449673 while d625c443 was installed: the launcher is a manifest file). **Site tree** `vllm/` 36631b8bc6ce over 5,455 files (was 11e30c3df6b6 over 5,454 files: the one new module).
- Unchanged: `dsv41_l2_prefetch.py` 63737f9c, `fp8_utils.py` b35c0561, `sm121_target_m32.py` ea4a8814, g2 pacer `libdsv41_l2pf_g2.so` 0451009d, r4 burst `libdsv41_l2pf.so` a09f85e8, gate.sh 1889ae4a5ce3, pin `av-pin-k5-g3FU.json` eeac4f97, the running env (window 5's 63 variables, no w6 variable).
- Staged only (`~/dsv41-stage/w6/`, bundle a284a782c82ca474): qb-l2pf (`dsv41_l2_prefetch.py` 0ca63b21, a manifest file), fc1fc2 H (`fp8_utils.py` 6b466c31, `sm121_target_m32.py` ae306c6b, `deep_gemm_moe.py.hpf` / `.all`).

### 2. The window, step by step

| step (UTC) | what | result | API down | launcher after |
|---|---|---|---|---|
| lock + staging (01:44-01:52) | lock taken; `~/dsv41-stage/w6/{qb-l2pf,fc2hpf,drafter-m32,bundle}` on 8/8 (stage dirs only) | verify-stage a284a782c82ca474 on 8/8, A0 GO | 0 | afa97ae6 |
| window A attempt 1 (01:55-02:09) | stopped window for every GPU bitwise test | no test started: after the stop the window's clock gate read the idle SM clock (208 MHz) against its 1900 MHz floor (fixed: the latch is judged under load); nothing installed; A11 VERIFY8-DIFF (a stale vision expectation in the window-5 check, fixed) | 4.0 min | afa97ae6 |
| window A attempt 2 (03:44-04:19) | same | harness failures only (drafter full / lpass runner rc 1, qb bench never finished, hpf unit tests 0/8); nothing installed; A11 PASS | 17.3 min | afa97ae6 |
| re-stage (04:21-04:29) | drafter-m32 (test-harness fix; MANIFEST digest 358ed295fee6df59 -> 499089245af1e302, CPU test 42/42) and fc1fc2 (harness fix, 97c3799d07b15459; the old stage dir kept as `fc2hpf.prev`); `W6-SHAS.txt` re-pinned (126 files) | no patch under test changed | 0 | afa97ae6 |
| **window A attempt 3** (04:30-05:08) | A0-A11 | **X GO, L GO, Q NO-GO, H NO-GO**; drm32 installed flags-off; production back at T+28 min; A11 PASS | 28.2 min | afa97ae6 |
| **Phase B** (05:31-08:24) | 10 pinned instances V1 C1 X1 C2 X2 C3 S1 C4 S2 C5 | all EXACT-PASS; X PASS; S (= L) SMALL | 27.6 min (11 relaunches) | afa97ae6 |
| **Phase C** (08:58-09:01) | `promote-w6.sh X` | launcher afa97ae6 -> d625c443 on 8/8 | 0 | d625c443 |
| **Phase D** (09:02-09:56) | production relaunch on d625c443 + suite; STOP; launcher rollback; production relaunch on afa97ae6 + suite + gate + identity | X-on production: unknown G1 prose outcome 33ae975a -> STOP -> rollback; final production exact, gate PASS | 7.2 min (2 relaunches) | **afa97ae6** |

### 3. Window A (stopped, attempt 3, T0 = 04:30:56)

| lane | verdict | deciding evidence |
|---|---|---|
| **X** drafter-m32 | **GO** | A2 quick + full 8/8 passed (26 summaries clean, 0 mismatching words), the same 2 new private JIT entries on 8/8; saved per drafter step at M = 20: **697.2 us** (rank 0) / **790.4 us** (rank 7), gate >= 250; A7 prejit 8/8; A8 quick on the INSTALLED files 8/8 (32 summaries) |
| **L** LIVE_ACT 192 | **GO** (L, L_DRAFTER, L_DRAFTER_M32) | A5 M4 target tests: 350 L cases per rank, 0 failures on 8/8; A2 lpass on the 128-row and the M32 drafter paths |
| **Q** qb-l2pf | **NO-GO (not tested)** | A3 PASS (Part A bitwise + Part S co-runner; R2 pacer `RATES_OK=64,80`), A4 bf16 bench F2 BITWISE PASS (0 failures on 8 ranks); the decisive `--mx` bench never ran: `bench_qb8.py --mx` calls `vllm.utils.flashinfer.mm_mxfp8`, which exists only with the launcher's `CUDA_HOME` / `PATH`; the bench ran without them (AttributeError on 8/8 at graph capture). Harness bug: rerun A4 with the launcher's CUDA env in a short stopped window (~15 min) |
| **H** FC2-head prefetch | **NO-GO** | T1 exhaustive table FAIL (3 cases) and the 3 sanitizer runs, all on the shared check `ptx_off_identical` (synthetic layers: the live arm compiled 3-4 kernel specializations, the flag-off arm 2; the real-layer runs passed it). memcheck / synccheck 0 errors; racecheck 15 races, all inside the existing live `deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl`, none in the prefetch code. Best bench gain ~0.95 us/layer against the 2.0 gate: NO-GO on gain alone |

A11 (production on afa97ae6 with drm32 installed flags-off): exact.py a8 x2 (905/471/2355/1373 unpinned), G1 104.1 (104.151 / 103.968 / 104.181), digest bc1e8386878bd2b7, no w6 line, verify8 identical on 8/8, w6state t0:drm32 with no w6 variable: **PASS**.

### 4. Phase B: pinned A/B chain (`av-pin-k5-g3FU.json`, 05:31-08:24)

Every instance: exact.py a8c3394af9d2 at start and end with 909/467/2335/1373 and 348/233/171/90/67 and a pin hit; G1 3-rep digest bc1e8386878bd2b7 (2 reps 9a18f802d930f6e7); matched C1 AL 5.191 / 2.725 (drafts identical); L2PLAN-CMP PASS; every unit CLEAN (0 foreign units, 0 NOT PROVEN). V1 (`=verify`, gate only): VERIFY PASS 8/8, 7,394 calls on each of the 3 drafter weights per rank, 79,915 rows, 0 mismatching words, 0 capture misses.

| slot | arm | G1 | G4 mean / prose MIN / G4min_hat | 4-session step ms | C1 coding / prose ms | conc8 / conc16 |
|---|---|---|---|---|---|---|
| V1 | V (timings not used) | 102.08 | 62.13 / 48.28 / 48.465 | 44.980 | 30.32 / 26.26 | 259.2 / 396.2 |
| C1 | C | 104.18 | 64.84 / 50.31 / 49.530 | 44.082 | 29.55 / 25.63 | 269.0 / 401.7 |
| **X1** | X | **105.23** | 64.50 / 50.21 / **50.433** | **43.104** | 29.42 / 25.40 | 268.7 / 408.0 |
| C2 | C | 104.21 | 63.31 / 50.20 / 49.870 | 43.780 | 29.56 / 25.62 | 265.9 / 403.4 |
| **X2** | X | **105.30** | 63.17 / 50.12 / **50.417** | **43.092** | 29.32 / 25.31 | 272.9 / 410.9 |
| C3 | C | 104.18 | 64.69 / 49.74 / 49.679 | 43.921 | 29.55 / 25.64 | 273.2 / 396.9 |
| S1 | S = L | 104.75 | 64.22 / 50.09 / 49.611 | 43.846 | 29.53 / 25.44 | 265.3 / 400.6 |
| C4 | C | 104.29 | 63.92 / 50.53 / 49.579 | 43.623 | 29.60 / 25.76 | 269.5 / 403.9 |
| S2 | S = L | 104.85 | 64.09 / 49.32 / 49.612 | 44.031 | 29.55 / 25.48 | 265.7 / 404.3 |
| C5 | C | 104.20 | 63.75 / 49.75 / 49.408 | 43.978 | 29.64 / 25.60 | 263.3 / 407.9 |

Pairs (candidate vs the mean of its two bracketing controls, %, + = better):

| pair | G1 | G4min_hat | prose MIN | C1 coding | C1 prose | conc8 | conc16 | step |
|---|---|---|---|---|---|---|---|---|
| X1 (C1, C2) | +0.99 | **+1.48** | -0.08 | +0.46 | +0.89 | +0.47 | +1.35 | +1.92 |
| X2 (C2, C3) | +1.06 | **+1.29** | +0.30 | +0.80 | +1.26 | +1.24 | +2.69 | +1.76 |
| S1 (C3, C4) | **+0.49** | -0.04 | -0.10 | +0.15 | +1.02 | -2.23 | +0.05 | -0.17 |
| S2 (C4, C5) | **+0.58** | +0.24 | -1.65 | +0.24 | +0.79 | -0.26 | -0.39 | -0.52 |

Noise (5-control spread, floors applied), %: G1 0.3 (measured 0.11), C1 coding 0.5, C1 prose 0.624, G4min_hat 0.931, conc8 3.692, conc16 3.0.

### 5. Phase C: the promotion rule (RUNBOOK 7), applied as written

- **X: promoted.** Exact on every instance (V1 8/8, X1 / X2 EXACT-PASS with drafts identical); primary metric G4min_hat +1.475% and +1.291% (>= +0.5% in both pinned pairs; G1 +0.99 / +1.06% as well); nothing worse beyond noise (the one negative, X1 prose MIN -0.08%, is far inside the 1.6% control spread). `promote-check X` rc 0. `promote-w6.sh X` (08:59:41): new launcher d625c443 on 8/8, `bash -n` OK, backup `.bak-w6` = afa97ae6 on 8/8; supervisor manifest 0c96a727fb81 -> 26db72449673 (the manifest recomputed with afa97ae6 in the launcher's place was 0c96a727fb81 on 8/8: the launcher was the only manifest file that changed).
- **L (S = L, H being NO-GO): not promoted, SMALL.** Primary metric G1 +0.494% (S1) and +0.579% (S2): S1 is under the +0.5% bar. At 4 sessions it costs a little (step +0.17 / +0.52%, prose MIN -0.10 / -1.65%). `promote-check X L` is REFUSED anyway: no arm measured X and L together.
- **Q, H:** never reached Phase B (NO-GO in A). **A** (combined arm): not run, fewer than 2 passing items.
- X's 4-session step gain (-0.83 / -0.76 ms against the bracket means, -1.9 / -1.8%) beat the review's -0.30..-0.60 ms; the measured slowest session did not move beyond noise on the pinned instances (prose MIN 50.21 / 50.12 against controls 49.74-50.54).

### 6. Phase D: production check (RUNBOOK 8)

| instance (UTC) | launcher / X | exact.py start / end | G1 (code / prose / math) | G1 outputs (code / prose / math), digest | G4 metric (prose per run) / G4min_hat / step ms | verdict |
|---|---|---|---|---|---|---|
| w6-PROD-D + w6-D1b (09:02-09:23) | d625c443 / on (default) | a8 OK / a8 OK (915/461/2305/1373; 346/238/177/89/65) | 104.20 (98.94 / 77.30 / 136.36) | 596b478a / **33ae975a** / 92e5d8cf, 247b90ce9f0b492e (all 3 reps identical) | 49.99 (49.13 / 50.35 / 50.48) / 50.72 / 42.99 | **STOP**: unknown prose outcome (CLEAN, 0 foreign) |
| w6-PROD-rb + w6-Drb (09:27-09:43) | afa97ae6 / off | a8 OK / a8 OK (910/466/2330/1373; 349/234/174/88/65) | 104.26 (98.36 / 78.47 / 135.96) | 596b478a / a22e1d39 / 92e5d8cf, 51d0a6537e239757 (all 3 reps identical) | 48.56 (48.04 / 48.09 / 49.55) / 50.25 / 43.68 | EXACT-PASS (a22e1d39 is a known unpinned tie outcome) |

- The STOP followed RUNBOOK 8 ("an unknown prose outcome is a stop") and RUNBOOK 2 (a clean exactness mismatch: production on launcher defaults with every w6 flag unset, all data kept, no retry): `promote-w6.sh --rollback` (09:26:56, afa97ae6 on 8/8, sha-checked), then the verified supervisor after a 30 s idle window, `--expect-launcher afa97ae6`, AV unpinned, no w6 flag: arm-P facts PASS (no w6 variable or line on 8/8), exact.py anchor a8c3394af9d2.
- A confirmation under the pin was not run: Phase B's X1 / X2 are that confirmation (the same files and the same running env; the promoted launcher only moves `DSV41_DRAFT_MOE_M32=1` from a supervisor argument into the launcher, which the arm facts verified on 8/8).
- Regression gate (kit 08, `gaterun.sh w6-D` on spark-one: 30 s idle window, 0.5 s sampler, gaudit) on the final production: **PASS in 596 s, audit CLEAN** (every step's success delta = its own requests): exact a8c3394af9d2; G1 one pass 104.36 (code 596b478a, math 92e5d8cf, prose a22e1d39); one cold 1M at TTFT 168.4 s (good class; MemAvailable minimum 8.3 GB); GSM8K-200 195/200; longctx 60/60, copy 1.000; KL verify / smoke / capture PASS (the capture's equivalence tier against the 10-28 capture reads 'FAIL (no noise floor)': not an exactness signal at its settings, as in window 5).
- Identity on 8/8: verify8 32 OK / 0 problems (`w6gates.py verify8` GO: identical on 8/8, no install change); site tree 36631b8bc6ce (5,455 files) and env digest 9626ff845dc0 (window 5's) on 8/8; w6state GO: the w6 files = t0:drm32, pacer 0451009d and r4 a09f85e8, launcher afa97ae6, no w6 variable in the running env of any rank. `/health` 200, `dsv41-rank` active on 8/8, nrestarts 0, SM 1969-1989 MHz.

### 7. Goal status

| metric | before window 6 (production / pinned) | now (production) | X-on, pinned / production (not live) | goal |
|---|---|---|---|---|
| G1 | 104.00 / 104.22 | 104.26 | 105.23-105.30 / 104.20 | 120 (+15.1% needed) |
| G4 metric (slowest session) | 47.89 / 50.55 | 48.56 | 50.21 / 50.12 pinned / 49.99 | 50 |
| G4min_hat | 49.25 / 49.42 | 50.25 | 50.43 / 50.72 | 50 |
| 4-session step | ~44 ms | 43.68 ms | 43.10 / 42.99 ms | - |

G1 120 still needs about +15%. Only acceptance closes that (drafter self-distillation, the open user decision of window 5, or R12's row-count invariance). G4: G4min_hat is above 50 on every X instance (50.42-50.72) and on this production launch (50.25), below it on the pinned controls (49.41-49.87); the measured slowest session (48.56 in production now, 49.99 with X) still depends on how the 4 sessions batch.

### 8. Rollback and re-enable

- The window's net change on 8/8 is the two flag-off site files and the new launcher backup. To remove the files (service stopped): `W6_WINDOW=go bash $W/bundle/w6-bundle.sh uninstall`, verified by `w6-bundle.sh expect live` (restores `deep_gemm_moe.py` 5162dc0f from `.bak-w6` or from the staged `~/dsv41-stage/w6/bundle/deep_gemm_moe.py.base`, removes `dsv41_draft_m32.py`). By hand on 8/8: `E=~/venvs/vllm032/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/experts; cp -p $E/deep_gemm_moe.py.bak-w6 $E/deep_gemm_moe.py && rm $E/dsv41_draft_m32.py`, then check 5162dc0f on 8/8. The 2 prejit JIT entries are inert and may stay.
- Re-enable X (only after the user's decision): see the decision paragraph above. Turn it off again for one launch with the supervisor argument `DSV41_DRAFT_MOE_M32=0`; permanently with `cp ~/dsv41-tp8.sh.bak-w6 ~/dsv41-tp8.sh` on 8/8 (= afa97ae6) and a relaunch with `--expect-launcher afa97ae6`.
- `$W` = the Mac kit `/private/tmp/claude-501/-Users-jeweled/64b45536-e9c4-4d44-9db6-63d2a8374326/scratchpad/w6` (ephemeral).

### 9. Open items after window 6

- **X:** the user decision above. If X goes live unpinned, add 33ae975a to the known prose outcomes (driver `PROSE_KNOWN`, gate notes) and re-check under the gate; if the pin becomes the production default, X needs no new outcome.
- **Q** (q_b in the L2 plan, review +0.7..1.3% G4min_hat): fix the A4 harness (run `bench_qb8.py --mx` with the launcher's `CUDA_HOME` / `PATH`), a short stopped window for A4 only (~15 min), then a pinned Q chain. Installing qb moves the supervisor manifest (0c96a727fb81 -> 3a29a454fbed on afa97ae6). Like X, Q changes a cost the unpinned AV profile measures (verify above 8 rows), so its production check can meet the same unpinned prose flip: plan its Phase D with that (or with the pin as the production default).
- **H:** NO-GO on gain (0.95 of 2.0 us/layer) independent of the `ptx_off_identical` failure; not worth a retry as designed.
- **Vision:** the staged image set ced2eab1bb8da400 stays bound to what is live (launcher afa97ae6, manifest 0c96a727fb81, `model_runner.py` 6a4f2433, all unchanged; qb not installed), so this window's net change needs no re-stage; only the kit RUNBOOK's recorded "before" site tree moves from 11e30c3df6b6 to 36631b8bc6ce. Re-promoting X (new launcher, manifest 26db72449673) would make the set stale.
- **Codex R12:** refresh its baseline: `deep_gemm_moe.py` is now 9364f121 (was 5162dc0f), new `experts/dsv41_draft_m32.py` a17d574a, site tree 36631b8bc6ce over 5,455 files, 2 new JIT entries; the launcher (afa97ae6), the l2 prefetch files and the running env are unchanged.
- **Harness fixes from Phase D** (Mac kit only): `w6-bundle.sh expect` compares the manifest with its pre-promotion value, so any promoted launcher made `expect t0:<set>` refuse (Phase D used `phaseD/bundle-expect-promoted.sh`, which accepts only the case where the promoted launcher is the one manifest file that differs); the window-5 `verify8.sh` hard-codes launcher afa97ae6 (a copy expecting d625c443 is `phaseD/verify8-w6d.sh`; not needed after the rollback); `promote-w6.sh`'s tests did not exercise Phase D's unpinned prose check.
