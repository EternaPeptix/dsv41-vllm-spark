
## Goal window 1 (2026-10-26 11:40-12:55 UTC): goal metrics on the final config, pinned and production. Goal NOT met (G1 102.8 of 120 tok/s; slowest G4 session 48.6 of 50). No config change

Scope: (1) verify that the server is up on the final config of the G1 candidates step (launcher 395bf5b2 + clock pre-step 2300); (2) measure the goal metrics on one pinned instance and on one production (unpinned) instance of that config: G1 x3, G4 x3, matched C1 (label pdab, twice), bench conc 4/8/16/32 x2, exact.py (first request, and again after all load), GSM8K-50, full longctx, usage, cold TTFT ~24K / ~198K, and the cold 1M TTFT once (production, which also serves as the relaunch-if-slow screen). Nothing was installed or edited: the launcher, the site files, the supervisor and the env defaults are all unchanged. Mac drivers and raw results are in the scratchpad `gw1/`: `scripts/gw1arm.sh` (one instance), `gw1suite.sh` (the suite), `gw1blk.sh` + `gw1teleab.sh` (telemetry on/off blocks), `vcheck.sh` / `vrun.sh` / `vsumm.py` (static check on 8/8), `gw1summ.py`, `gw1table.py`, `gw1tele_tab.py`; plus `runs/<label>/`, `final/<label>/`, `gw1.log` and `launches.tsv`. On spark-one the run dirs are in `~/dsv41-gw1-20261026/runs/` (with `scripts/` and `mac-scripts/`). Telemetry CSVs are in `~/dsv41-g1-20261026/tele/gw1-*` on all 8 nodes.

**Verdict.** Neither target is met. The production numbers below are telemetry-off figures; section 4 explains why.
- **G1, single user: 102.80 tok/s** (code 97.4 / prose 77.1 / math 133.9) against 120. **Not met: it needs +16.7%** (+17.2 tok/s).
- **G4, 4 concurrent sessions: the per-session mean is 62.35** tok/s, above 50, but the **slowest session (prose) is 48.64**. All 6 prose runs fell between 47.5 and 49.6; code, math and mixed were above 50 in every run. **Not met: the prose session needs +2.8%.** Over the common window, while all four sessions decode at once, the slowest session runs at 40.7 (+22.8% needed).
- G4 is the closer of the two targets. The pinned instance (measured with telemetry on) gave G1 102.26 and G4 59.89 / 48.17. Removing the measured telemetry cost (+1.1%) puts pinned G1 at about 103.4, in line with the candidates step's pinned M arm (103.23, measured without telemetry).

### 1. Live config verified (before: 11:42 UTC on g2-F1; after: 12:50 UTC on gw1-U-2)
`gw1/scripts/vcheck.sh` ran on 8/8 and its output was compared with the expected values (`vsumm.py`): **0 mismatches both times**. Apart from the instance label, start time and temperatures, the before and after states are identical.
- **Launcher** `~/dsv41-tp8.sh` **395bf5b2** on 8/8. It contains `export DSV41_MOE_M32_G1=${DSV41_MOE_M32_G1:-all}`, has `--enable-prompt-tokens-details` in ARGS, and has no default AV pin. Backup `.bak-g1-m32bw` = 04b6738e on 8/8. The 18 pd-window backups are unchanged (listing hash ae661a994ec7).
- **Running server, 8/8:** the `vllm serve` cmdline carries `--enable-prompt-tokens-details` and max_model_len 1048576. The env is identical on all 8 nodes: 50 DSV41_/RDMA_ variables, hash 67c2ffc6d11c, no test variables. It includes `DSV41_MOE_M32_G1=all`, `DSV41_MOE_FC2_D05=1`, `DSV41_L2_PREFETCH_PRUNE_SHARED=1`, and `DSV41_AV_PROFILE=pinned` with `DSV41_AV_COST_JSON` unset, so the startup profile is used ("pin miss").
- **Site files, identical on 8/8:**
  - `sm121_target_m32.py` db659d80 (`.bak-g1-m32bw` 5acc5fde).
  - `dsv41_l2_prefetch.py` 998af3d7, `dsv41_r4_small.py` c46e66bd, `sparse_swa.py` 854c3e5f, `model_state.py` f21ee2e6. Their `.bak-g1-stepovh` backups are 514dcb23 / 02f94f29 / 38b16c51 / 53e4fe39; the stepovh flags are off.
  - Every other file in the "Prefill+decode window" list is unchanged (model.py e7264781 ... gate_linear.py 41adaba8, scheduler 6074389f, no rgate or pd_policy file).
  - Supervisor site manifest 54ec141eb2dc. `~/dsv41-stage/g1/m32-bw/PREP.ok` bb9660e130330ec0, with the `all-{base,d05}` natives mapped on the 8 workers.
- **Supervisor** `ops/rel/dsv41-relaunch.sh` **30158e80** (defaults: clock 2300, compaction 0). Backups `.bak-g1-clock` f2f6e44c and `.bak-g1-pre` 39679a35.
- **Clock and compaction pre-steps** (documented in "G1 step", sections 2-3):
  - Every launch in this window logged `prelaunch compact=0 clock=2300 ... clock -lgc 0,2300 OK` on 8/8. Under load the SM clock ran at 2268-2288 MHz on every node.
  - The boot unit `dgx-spark-gpu-clock.service` (`-lgc 0,2000`) is still enabled, so a reboot restores 2000.
  - Compaction did not run (compact=0), and `vm.compaction_proactiveness` is 0.
- **Health:** 8/8 active, 0 restarts, 0 ERROR/Traceback, rdma fingerprint c40cab3c30f4d7f2 on 8 ranks, Engram check OK on 8/8.

### 2. Method
- **Instances.** All were started by the supervisor with its defaults and `--expect-launcher 395bf5b2`:
  - **gw1-P**, pinned (`DSV41_AV_COST_JSON=.../pdab/av-pin-pdfinal.json`, "pin hit"): healthy at +278 s.
  - **gw1-U**, production: healthy at +361 s. Its 1M probe came out in the slow class, so it was relaunched.
  - **gw1-U-2**, production: healthy at 12:23:19 (+354 s). Its 1M probe was in the good class, and it is the live instance.
- **Suite (`gw1suite.sh`).** On each instance: exact.py first (`gstart.sh`; on production, the cold 1M came right after it). Then:
  1. matched C1 coding/prose x5 (label pdab)
  2. G1: `goal_single.py --repeats 3`
  3. G4: `goal_conc4.py` (warm-up + 3 runs)
  4. bench.py conc 4/8/16/32 (512 tokens), x2
  5. matched C1 again
  6. TTFT ~24K / ~198K
  7. usage check
  8. GSM8K-50 (conc 8)
  9. full longctx
  10. exact.py again, after all the load

  A 2 s `/metrics` sampler showed no outside traffic: in every phase the maximum running count equals that phase's own concurrency.
- **Telemetry.** Both suites ran with the G1 step's 2 s telemetry on 8/8 (`g1tele.sh`: `nvidia-smi -lms 2000` plus a sysfs loop). On gw1-U-2 I then ran telemetry-off, on, off decode blocks; see section 4. Telemetry costs about 1.1% on G1 and about 2.3% on G4, so the production headline numbers come from the two telemetry-off blocks (A, C).

### 3. Results
| metric | **production gw1-U-2, telemetry off** (blocks A + C) | production gw1-U-2 suite, telemetry on | pinned gw1-P suite, telemetry on | ref: pinned M arm, candidates step (n=3, no telemetry) |
|---|---|---|---|---|
| **G1 mean tok/s** (code / prose / math), 1024 tokens x3 each | **102.80** (97.37 / 77.13 / 133.90); per rep 102.86, 102.77, 102.80, 102.80, 102.79, 102.77 | 101.75 (96.40 / 76.28 / 132.57) | 102.26 (96.24 / 76.39 / 134.15) | 103.23 (96.9 / 77.2 / 135.6) |
| G1 AL code / prose / math | 2.570 / 1.814 / 3.557 | same instance | 2.558 / 1.807 / 3.607 | - |
| **G4 per-session mean / min tok/s** (3 runs) | **62.35 / 48.64** | 61.11 / 48.16 | 59.89 / 48.17 | 60.41 / 48.00 |
| G4 by session: code / prose / math / mixed | 57.1 / 48.6 / 79.2 / 64.5 | 57.9 / 48.2 / 73.9 / 64.5 | 57.2 / 48.2 / 73.4 / 60.8 | 55.7 / 48.0 / 76.1 / 61.9 |
| G4 slowest session per run (always prose) | 49.64, 49.00, 48.38, 47.51, 48.19, 49.09 | 49.07, 48.53, 46.88 | 47.64, 48.68, 48.19 | - |
| G4 session-runs at >= 50 tok/s | 18 / 24 | 9 / 12 | 9 / 12 | - |
| G4 common window, all 4 decoding (mean / min) | 59.79 / 40.73 | 58.67 / 39.30 | 56.88 / 40.36 | 57.64 / 40.85 |
| G4 aggregate tok/s | 183.1 | 181.8 | 181.5 | 180.9 |
| matched C1 coding / prose ms/step (label pdab) | **29.94 / 25.84** (AL 5.165 / 2.706) | 30.21 / 26.09; second pass 30.27 / 26.10 | 30.33 / 26.18 (AL 5.224 / 2.718); second pass 30.42 / 26.15 | 30.08 / 25.92 |
| bench conc4 aggregate tok/s (per-request mean / min) | **181.4** (59.0 / 48.5) [block A] | 177.1 (60.5 / 46.1) | 177.0 (59.9 / 47.2) | - |
| bench conc8 | **266.3** (44.7 / 34.6) | 262.1 (44.6 / 34.7) | 259.6 (43.5 / 33.6) | - |
| bench conc16 | **395.9** (34.6 / 25.9) | 399.1 (34.1 / 26.6) | 386.6 (33.0 / 24.8; runs 373.3, 400.0) | - |
| bench conc32 | **578.4** (24.5 / 18.9) | 577.3 (24.5 / 19.0) | 579.2 (24.6 / 19.1) | - |
| cold TTFT ~24K / ~198K | - | 3.99 s (23,979 tokens; warm, after the 1M probe) / 32.62 s (197,931) | 6.90 s (first long prefill after start) / 32.61 s | - / 32.64 s |
| **cold 1M TTFT** (1,024,611 tokens) | **179.04 s**, good class (gw1-U-2) | | not run (once only) | |

- **Against the previous window's final** (mm-FINAL: 04b6738e, 2000 MHz, unpinned), the production instance now gives: conc4 162.8 -> 181.4 (+11.4%), conc8 238.9 -> 266.3 (+11.5%), conc16 354.2 -> 395.9 (+11.8%), conc32 514.1 -> 578.4 (+12.5%); matched C1 (pdab) 32.25 / 27.55 -> 29.94 / 25.84 ms (-7.2% / -6.2%). This is the 2300 MHz clock and m32-bw `all` together.
- **Production vs pinned.** The unpinned AV profile of gw1-U-2 differs a little from the pin: math AL is 3.557 vs 3.607, and the prose G1 output differs (the code and math outputs are identical). That is why production G1 math is 1.2% lower than pinned at equal telemetry (132.5 vs 134.2). G4 is the same within noise.

### 4. Telemetry check: on, off, on, off on gw1-U-2 (same instance, same prompts, consecutive blocks)
| block (in order) | telemetry | C1 coding / prose ms | G1 mean (code / prose / math) | G4 per-session mean / min | slowest session per run | G4 common window mean / min | G4 aggregate |
|---|---|---|---|---|---|---|---|
| suite | on | 30.21 / 26.09 | 101.75 (96.4 / 76.3 / 132.6) | 61.11 / 48.16 | 49.1, 48.5, 46.9 | 58.67 / 39.30 | 181.8 |
| A | off | 29.88 / 25.82 | 102.81 (97.4 / 77.1 / 133.9) | 62.68 / 49.01 | 49.6, 49.0, 48.4 | 60.29 / 40.89 | 183.2 |
| B | on | 30.28 / 26.12 | 101.60 (96.2 / 76.3 / 132.4) | 60.80 / 46.85 | 47.7, 48.4, 44.5 | 57.90 / 40.32 | 179.2 |
| C | off | 29.99 / 25.86 | 102.78 (97.3 / 77.1 / 133.9) | 62.02 / 48.26 | 47.5, 48.2, 49.1 | 59.29 / 40.57 | 183.1 |

- **Effect of turning telemetry off:**
  - G1 **+1.10%**; code, prose and math each +1.07 to +1.13%. The spread within a block is 0.02-0.10 tok/s.
  - C1 **-1.02%** for both coding and prose.
  - G4 per-session mean +2.29%, min +2.38%, aggregate +1.45%.
  - conc4 +2.4% and conc8 +1.6%; conc16 and conc32 are flat (-0.8% / +0.2%).
  - The G1 output digest 69ad872124100141 is identical in all four blocks, so telemetry does not change outputs.
- **Consequence for the G1 step's clock A/B.** All of its arms ran with this telemetry, so its relative gains stand, but its absolute figures are about 1% low. This explains the +1.3% gap between that step's pinned 2300 arms (98.97, telemetry on) and the candidates step's C arm (100.26, telemetry off, the same code apart from inert files). For future A/Bs, keep telemetry off, or on in every arm.

### 5. Exactness and gates
- **exact.py, first request on each instance** (the anchor is unchanged):
  - gw1-P: a8c3394af9d2, counters 905/471/2355/1373 (pin hit).
  - gw1-U: a8c3394af9d2, 913/463/2315/1373 (its own profile).
  - gw1-U-2: a8c3394af9d2, 905/471/2355/1373 (its own profile, pin miss).
  - **After all the load (exact #2):** a8c3394af9d2 with 905/471/2355/1373 on both gw1-P and gw1-U-2.
- **G1 outputs:** the 3 repetitions are identical on every instance. The pinned digest over the first two repetitions is **9a18f802d930f6e7**, the same as every pinned instance of the candidates step.
- **GSM8K-50:** 49/50 on both instances. The miss is idx 12 (13 vs 12), the known borderline item.
- **Full longctx:** PASS on both (needle 60/60, copy 1.000, 165 s).
- **usage** `prompt_tokens_details`: chat cached_tokens 0 -> 4160, stream 4160, /v1/completions 0 -> 4160, on both instances.
- **Cold 1M** (`xr_bench.py solo --len 1000k`):
  - gw1-U: 230.05 s, **slow class** (ratio 1.88; chunk bands 2.62 / 3.00 / 4.02 / 4.91 s), so it was relaunched.
  - gw1-U-2: **179.04 s, good class** (ratio 1.16; bands 2.61 / 2.76 / 2.90 / 3.02 s).
  - Unpinned launches at 2300 that had a 1M probe now total 5 good / 3 slow.
- 0 ERROR/Traceback on every instance.

### 6. Thermal and power at 2300 MHz (telemetry phases)
- **GPU:** max **77 C** (spark-one, full longctx on gw1-U-2; 74 C on gw1-P). The guard at 83 C never fired.
- **ACPI SoC zone:** max **94.8 C** (spark-one, conc32 on gw1-P; 93.7 C on gw1-U-2). That is 2.8 C above the G1 step's 92 C maximum and 10 C below the 104.8 C critical trip. The hottest zone is always on spark-one, and its temperature follows sustained load.
- **spark-one system power:** 98.6-124.1 W mean by phase, 148.4 W maximum (longctx). The G1-step 2300 arms gave matched 99.4 / G1 100.4 / G4 110.8 W; here matched was 99.6, G1 98.6 and G4 109.5 W.
- **Fleet GPU-rail sum:** 208-304 W by phase.

### 7. Live state after this window
Instance **gw1-U-2**, healthy since 12:23:19 UTC. It was started by the supervisor with its defaults: launcher 395bf5b2, AV unpinned (startup profile), clock pre-step 2300 on 8/8, no compaction. Its 1M is good class (179.04 s).
- Every item in section 1 was re-verified at 12:50 UTC with 0 mismatches.
- Telemetry is stopped, and 0 telemetry or nvidia-smi processes are left on 8/8.
- There is nothing to roll back: this window made no change. The rollbacks in "G1 candidates step" and "G1 step" still apply.

### 8. Where the gap is
- **G4 (prose session +2.8%).**
  - Measured exact lever not yet promoted: stepovh A+B+C, which gave +1.1% on the slowest session and +0.9% on the G4 mean in the pinned ABAB. It is installed but inert, with its flags off.
  - Staged or estimated levers not yet built: the fc2c tail on top of `all` (estimated about -1 ms/step at 4 users, about +2%), and routing M=1 calls to the live kernel (about 0.08 ms, on M=1 steps only).
  - Together these levers are roughly the size of the gap, so G4 is within reach but not assured. The common-window minimum (40.7) is much further from 50.
- **G1 (+16.7%).** The known exact levers add only a few percent: stepovh +0.7%, fc2c-on-`all` about +1% (-0.3 ms at 1 user), M=1 routing at most about +0.3%, and the next clock step (2418, about +1.3% more) broke the 25 W rule. Reaching 120 needs a lever of a different size. Two candidates: acceptance (the prose AL is 1.81 at 77 tok/s, the weakest type), or MoE weight streaming closer to the 211-234 GB/s that Marlin reaches on the same bytes (`all` reaches 199-219).
