
## k1 / 2-silu-liverows: routed FP4 MoE SiLU-mul-FP8-quant over live rows only (DSV41_MOE_LIVE_ACT) -- PROMOTED (2026-10-24 15:09 UTC)

**Problem.** `DeepGemmFP4Experts._act_mul_quant` ran the fused SiLU-mul-FP8-quant over the whole aligned workspace, `M_sum = 6M + min(6M,384)*(align-1)` rows, of which only 6M are live. At M=48 that is 9216 rows with 288 live (3%), and on the 128-row path (M > 96) it is about 49.5K rows. The fresh 1M profile (Mac scratchpad k1/profile-1m.md) measured 43-60 us per call at 16 users, x38 target layers, plus about 0.33 ms/step in the drafter.

**What.** Two site files under `fused_moe/`, flag-gated and default off. Off runs the original code paths (bitwise, unit-tested). `fp8_utils.py` (8dc7dab8) is untouched; its EXPERT_ALIGNMENT mode already existed.

| file | base (= `.bak-k1-2-silu-liverows` on 8/8) | live 8/8 |
|---|---|---|
| deep_gemm_utils.py | b7e8de4d | **a4a0486a** (`deepgemm_moe_permute(..., return_expert_ends=True)` also returns `expert_start_loc` after ep_scatter = each expert's exclusive live end) |
| experts/deep_gemm_moe.py | 45418259 | **5162dc0f** (`apply` passes `expert_ends`, `expert_alignment=align_used` into both fused UE8M0 SiLU launches, compact and plain; target E=384 top-6 and drafter E=128 top-3) |

- **Flags** (read in `DeepGemmFP4Experts.__init__`, so they are per instance):
  - `DSV41_MOE_LIVE_ACT=1` turns the path on.
  - `DSV41_MOE_LIVE_ACT_MIN_ROWS` (default 1024 aligned rows) is the gate. It is on for target M >= 6 (M_sum 1152) and drafter M >= 3.
  - `DSV41_MOE_LIVE_ACT_MAX_ROWS` (default 0 = no cap).
- **Why exact.** BM and packs_per_cta are still chosen from M_sum, so live rows run the same instructions. Padding rows of the FC2 input are simply not written. ep_gather reads only live rows, and `grouped_layout` (`torch.full(-1)`) is unchanged.

**GPU unit test** (`test_live_act.py` via `run_test.sh`, service stopped; spark-one rank-0 slice 12:47-12:50 UTC, spark-eight rank-7 slice 12:50-12:53 UTC; results in `gpu/`): **PASS on both ranks, 672 checks per rank (1344 total), 0 failures.** The CPU Triton-interpreter check also PASSED on both.
- Per rank: 576 eager cases (528 of them on the live path) plus 96 CUDA-graph replay checks.
- Each case checks, bitwise:
  - the ep_scatter invariant (`ends[e] == start[e] + count[e]`, `roundup(ends[e-1]) == start[e]`, layout);
  - dense vs live kernel on the live rows;
  - final layer output of the original file vs flag on vs flag off.
- Workspaces and the allocator are NaN-poisoned on every run. One graph per size is captured once and replayed on six route sets (valid, skewed, edge, half-padding, 30% holes, all-padding).
- Coverage: target compact and padded, the drafter, M32 (live) and M64 modes, the 1024 gate, real layer-0 / layer-39 / mtp.0 weights for the rank's TP slice.
- **Timing** (graph replay, warm L2, us per call, dense -> live; `gpu/timing-table.txt`). The live kernel is never slower, so `recommended_min_rows_by_geom` is the smallest tested size (target 192, drafter 384). The default gate of 1024 is kept, because below M=6 the saving is about 1 us per call.

| geom | M=1 | M=6 | M=48 | M=96 | M=128 (128-row path) | M=2048 | M=16384 |
|---|---|---|---|---|---|---|---|
| target compact kernel | 1.86 -> 1.74 | 3.47 -> 2.31 | 17.1 -> 4.8 | 22.8 -> 5.8 | 339 -> 9.0 | 429 -> 26.5 | 1017 -> 680 |
| target compact full MoE apply | 56 -> 56 | 534 -> 534 | 3235 -> 3175 | 4584 -> 4514 | 8649 -> 8329 | 10979 -> 10594 | 24938 -> 24530 |
| drafter kernel | 2.81 -> 1.89 | 5.35 -> 2.24 | 116 -> 5.0 | 132 -> 5.1 | 128 -> 5.3 | 190 -> 14.8 | 558 -> 417 |

In the engine the dense kernel is slower than in this warm-L2 replay (profile: 43-60 us at 9.2K-12.3K rows), so the real saving per call is larger.

**A/B** (live 1M config plus the arm env via launch-conc.sh; one fresh instance per arm; suite `scripts/ksuite.sh`):
- **Suite.** Matched C1 coding/prose x8 trials, bench single (1024 tok x2 per type), conc 8,16,32,8,16,32,16,32 (512 tok), and exact.py x2. i4-A also ran GSM8K-50 conc8, the longctx default gate and a 500K needle.
- **Order.** C A C A, then a pinned-AV pair Cp Ap, then a reversed pair A C, to counterbalance drift.
- **Startup.** Every instance came up on its first launch (385-411 s), with no rdma_ar hang and no engine-init failure.
- **Arm check.** `DSV41_MOE_LIVE_ACT=1` was in the vllm serve env on 8/8 in every armed instance, and unset on 8/8 in controls. The first armed launch compiled 3 new live-row Triton variants (`get_program_id z`) on every rank, which shows the live path really ran in the engine on 8/8.

| inst | arm | AV profile | coding ms/step (AL) | prose ms/step (AL) | single code/prose/math | conc8 | conc16 (x3 mean) | conc32 (x3 mean) | exact.py x2 |
|---|---|---|---|---|---|---|---|---|---|
| i1-C | C | startup | 34.145 (5.17) | 29.218 (2.72) | 85.9/68.6/118.6 | 225.3 | 330.5 | 485.7 | a8c3394af9d2 (x1) |
| i2-A | A | startup | 33.971 (5.08) | 29.109 (2.65) | 86.5/68.9/118.7 | 226.3 | 334.4 | 492.0 | **b7e868c9d700** (x1; see Exactness) |
| i3-C | C | startup | 34.034 (5.12) | 29.169 (2.75) | 86.0/68.5/117.5 | 222.6 | 327.9 | 478.5 | a8c3394af9d2 x2 |
| i4-A | A | startup | 34.115 (5.08) | 29.089 (2.64) | 85.8/67.8/118.8 | 227.0 | 337.9 | 491.0 | a8c3394af9d2 x2 |
| i5-Cp | C | pinned i1-C | 34.211 (5.11) | 29.234 (2.69) | 85.9/68.6/118.7 | 222.3 | 328.2 | 477.4 | a8c3394af9d2 x2 |
| i6-Ap | A | pinned i1-C | 34.125 (5.25) | 29.170 (2.74) | 86.0/68.8/119.0 | 226.9 | 334.2 | 492.7 | a8c3394af9d2 x2 |
| i7-A | A | startup | 33.994 (5.07) | 29.125 (2.68) | 86.2/68.8/119.6 | 225.8 | 335.5 | 491.7 | a8c3394af9d2 x2 |
| i8-C | C | startup | 34.025 (5.04) | 29.209 (2.73) | 86.2/67.4/118.2 | 220.7 | 326.1 | 479.3 | a8c3394af9d2 x2 |
| i9-Cx | C | pinned i2-A | - | - | 86.0/68.7/118.2 | - | - | - | **b7e868c9d700** x2 (counterfactual) |

C = control (installed, flag unset), A = `DSV41_MOE_LIVE_ACT=1`. "Pinned i1-C" means `DSV41_AV_COST_JSON=~/dsv41-stage/k1/2-silu-liverows/av-pin-i1C.json`, i1-C's raw startup AV profile, so both arms make identical verify-cap and budget decisions. Matched values are means of 8 trials (trial sd 0.14-0.49 ms). Matched AL varies per trial by construction, because each trial has a different front tag.

Per pair (% better; ms columns = fewer ms/step):

| pair | coding ms | prose ms | single c/p/m | conc8 | conc16 | conc32 |
|---|---|---|---|---|---|---|
| r1 i2-A vs i1-C | +0.51 | +0.37 | +0.65 / +0.39 / +0.08 | +0.42 | +1.17 | +1.30 |
| r2 i4-A vs i3-C | -0.24 | +0.27 | -0.23 / -1.02 / +1.09 | +1.94 | +3.02 | +2.59 |
| r3 i6-Ap vs i5-Cp (pinned) | +0.25 | +0.22 | +0.10 / +0.31 / +0.29 | +2.06 | +1.83 | +3.21 |
| r4 i7-A vs i8-C (reversed order) | +0.09 | +0.29 | -0.01 / +2.05 / +1.16 | +2.30 | +2.87 | +2.60 |
| **mean of 4 pairs** | **+0.15** | **+0.29** | +0.13 / +0.43 / +0.66 | **+1.68** | **+2.23** | **+2.43** |

- **Arm means (4 v 4).** conc8 222.7 -> 226.5 (+1.7%), **conc16 328.2 -> 335.5 (+2.2%)**, **conc32 480.2 -> 491.9 (+2.4%)**. C1 coding 34.10 -> 34.05 and prose 29.21 -> 29.12 ms/step. Single-stream is within noise.
- **The armed conc32 numbers are tight** (492.0 / 491.0 / 492.7 / 491.7). The controls spread more (485.7 / 478.5 / 477.4 / 479.3). r1 has the lowest gain because i1-C is the highest control.
- **Matches the kernel model.** At 16 users (48-64 rows), about -1.6 ms target plus -0.3 ms drafter is about -2 ms of a ~100 ms step. C1 is unchanged by design (M=6: -1.2 us per call).
- **AV profile.** The 128-row verify path got cheaper by 10-12 ms per step: verify_ms at 104/128/192 rows is C ~258/280/332 vs A ~248/271/320. The AV trimmer prices M > 96 accordingly.

**Exactness: exact (bitwise).**
- **Kernel and layer:** the unit test above.
- **Engine, AV profile held fixed (r3):** i5-Cp (flag off) and i6-Ap (flag on) both use i1-C's raw AV profile. Their bench-single greedy spec counters (steps / verified / accepted) are identical to each other and to i1-C: code 401/2005/622, prose 570/2850/453, math 287/1435/741. exact.py = a8c3394af9d2 x2 in both. So the target and the drafter produce the same tokens and the same accept decisions with the flag on (the drafter's AL is unchanged).
- **i2-A's b7e868c9d700 comes from its AV profile, not the flag.**
  - b7e868c9d700 is the known alternate VERIFY_CAP=auto trajectory: same 2 of 6 prompts, same per-prompt hashes 2228532c / b246b2a0. It is the hash r4-e4's u4 gave with no MoE change ("AV-profile dependent").
  - Counterfactual i9-Cx: flag **off** with i2-A's raw profile pinned (`av-pin-i2A.json`) reproduces i2-A bit for bit. exact.py b7e868c9d700 x2, and bench-single counters code 396/1980/627, prose 567/2835/457, math 288/1440/740, identical to i2-A.
  - So for both profiles tested, flag on == flag off (2x2: i1-C profile -> a8c3394af9d2; i2-A profile -> b7e868c9d700).
  - i2-A's raw profile had the highest 2-row verify sample seen (v2 21.97 ms; u4 21.85; 20.0-21.4 elsewhere; `rawprof.txt`, `tauproxy-r1.txt`). The flag cannot change the 2-row graph, because target M_sum 384 and drafter 768 are below the 1024 gate.
  - exact.py with a startup profile: a8c3394af9d2 in 2 of 3 armed instances (i4-A, i7-A) and 3 of 3 controls (i1-C, i3-C, i8-C); pinned: 2 of 2. This is the same AV-noise sensitivity the live config already had.
  - A high v2 alone does not decide the flip. i8-C (control, v2 21.37) and i6-Ap's raw profile (v2 21.44, pinned away) are close to i2-A; the boundary also depends on the 4- and 6-row points. The pinned 2x2 is the proof.
- **Gates on i4-A** (armed, startup profile):
  - GSM8K-50 conc8 **50/50** (247.6 tok/s).
  - longctx default gate **PASS** (needles 60/60, copy 1.000).
  - 500K needle **7/7** (497,263 tokens, copy exact, TTFT 87.3 s; LENAWARE instances: 87.2-88.3 s).

**Promotion.**
- Rule: exact, and conc16/conc32 improves >= 2% (the C1 bar is not met: +0.15 / +0.29%, as designed). conc16 is +2.23% and conc32 +2.43% as the mean of 4 pairs. Every pair is positive (min +1.17 / +1.30), and nothing is worse beyond noise.
- **Launcher** `~/dsv41-tp8.sh` cfb97923 -> **1a2f26f4** on 8/8. It adds one line after the DSV41_CAND_LENAWARE export: `export DSV41_MOE_LIVE_ACT=${DSV41_MOE_LIVE_ACT:-1}`. Backup: `~/dsv41-tp8.sh.bak-k1-2-silu-liverows` = cfb97923 on 8/8. Tool: `scripts/promote-launcher.py` (run with `python3 -I -` on each rank; it refuses unless the launcher is cfb97923). MIN_ROWS / MAX_ROWS stay at the code defaults (1024 / 0).
- **Site files** a4a0486a / 5162dc0f on 8/8 (backups b7e8de4d / 45418259).
- **Final live relaunch** (15:02:57-15:09:33 UTC, `ops/dsv41-launch-all.sh`, launcher 1a2f26f4 defaults, no env; instance `ab/i10-LIVE`):
  - Up in 390 s on the first attempt. `DSV41_MOE_LIVE_ACT=1` in the vllm serve env on 8/8 (launcher default). /v1/models max_model_len 1048576. No Traceback or ERROR lines in the 8 rank logs.
  - exact.py **a8c3394af9d2** x2.
  - Matched C1 coding **34.08** / prose **29.29** ms/step (AL 5.15 / 2.72). Bench single 86.1/69.1/118.0 tok/s. conc8 224.5/227.4, conc16 332.8/334.6/333.1, conc32 493.0/489.9/493.5.
  - GSM8K-50 conc8 **50/50** (247.9 tok/s). longctx default gate **PASS** (60/60, copy 1.000). 500K needle **7/7** (TTFT 86.8 s).
  - 1M needle **7/7** (999,411 tokens, copy exact, TTFT 182.0 s; LENAWARE final live 205.1 s). This change models only about 1 s of the 1M prefill (about -0.4 ms per MoE layer per 16K chunk), so most of that difference is run-to-run variance; do not count it as a gain.

**Rollback** (from the Mac):
- Env only, keeps every file: `ssh spark-one 'cat ~/dsv41-opt-20261021/ops/conc/launch-conc.sh' | bash -s -- DSV41_MOE_LIVE_ACT=0` (any value other than 1 is off).
- Launcher: `for h in one two three four five six seven eight; do ssh spark-$h 'cp -p ~/dsv41-tp8.sh.bak-k1-2-silu-liverows ~/dsv41-tp8.sh'; done` (back to cfb97923), then relaunch with `ssh spark-one 'cat ~/dsv41-opt-20261021/ops/dsv41-launch-all.sh' | bash`, and check that the sha is cfb97923 on 8/8.
- Site files (service stopped; undo the launcher first): on each rank run `bash ~/dsv41-stage/k1/2-silu-liverows/uninstall.sh`. It restores b7e8de4d / 45418259 from the `.bak-k1-2-silu-liverows` files and refuses on any unexpected sha. Or, from the Mac, run `bash $S/k1/2-silu-liverows/all.sh uninstall` (S = the Mac scratchpad; the same all.sh is in `~/dsv41-stage/k1/2-silu-liverows/`). It runs uninstall.sh on 8/8 and prints the 8/8 sha status.

**Notes for later experiments.**
- **Exp 5 (FC2 epilogue):** with the flag on, padding rows of the FC2 input (FP8 q and packed UE8M0 scales) hold stale bytes, so FC2's padding output rows can be NaN/Inf. Nothing reads them today (ep_gather reads live rows only; the unit test poisons them). A change that reads padding rows must handle that, or run with the flag off.
- **Exp 6 (fused MoE front):** `expert_ends` is `expert_start_loc` after `_fwd_kernel_ep_scatter_2`'s atomic_add. If exp 6 replaces that kernel, switch the source and re-run `run_test.sh` (it needs about 3 min per rank).
- **Exact A/Bs under VERIFY_CAP=auto:** the exact.py trajectory depends on the startup AV profile, so a single b7e868c9d700 is not by itself evidence of an inexact change. Pin the AV profile on both arms (`DSV41_AV_COST_JSON`, made with `scripts/mkpin.py` from a control's rank0 log) and compare the bench-single spec counters (`scripts/alstat.py`). For a suspicious instance, rerun the control pinned to that instance's profile (the i9-Cx counterfactual).

**Files** (`./k1/2-silu-liverows/`):
- `gpu/`: rank 0 / rank 7 unit-test results, `run_test.r*.out`, `timing-table.txt`.
- `ab/<instance>/`: suite outputs, `rank0.startup.log`, `env.txt`, `avprofile.txt`, `arm-check.txt`; `ab/live-before.*` is the pre-experiment live instance.
- `scripts/`: ksuite.sh, ksumm.py, kab.py, pairs.py, alstat.py, tauproxy.py, rawprof.py, mkpin.py, promote-launcher.py, plus the Mac drivers kdrv.sh, chain*.sh, launch-*.sh.
- `av-pin-i1C.json`, `av-pin-i2A.json` (also in `~/dsv41-stage/k1/2-silu-liverows/` on 8/8).
- `summ-all.txt` / `summ-all.json` (all instances, pairs, counters, tau factors), `rawprof.txt`, `tauproxy-r1.txt`.
- The patch, test and install scripts are in `~/dsv41-stage/k1/2-silu-liverows/` on 8/8 (MANIFEST 01420e3a).

**Live state after k1/2** (2026-10-24 15:10 UTC):
- Launcher `~/dsv41-tp8.sh` **1a2f26f4** on 8/8 (previous cfb97923 = `.bak-k1-2-silu-liverows`).
- `fused_moe/deep_gemm_utils.py` **a4a0486a** and `fused_moe/experts/deep_gemm_moe.py` **5162dc0f** on 8/8 (base b7e8de4d / 45418259 = `.bak-k1-2-silu-liverows`).
- Unchanged: `quantization/utils/fp8_utils.py` 8dc7dab8, `kernels/attention/dsa/candidate_blocks.py` a5816163, `sparse_attn_indexer.py` 7112c01d.
- Live env adds DSV41_MOE_LIVE_ACT=1 (launcher default). DSV41_MOE_LIVE_ACT_MIN_ROWS / _MAX_ROWS are unset (1024 / 0).
