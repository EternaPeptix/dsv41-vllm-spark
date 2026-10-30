# Takeover audit: DSv4.1 TP8 cluster after Codex goal-120 (2026-10-24, 21:15-21:45 UTC)

There was no config change and no restart. The server stays up on the live config.

- Raw results on spark-one: `~/dsv41-takeover-20261024/`. It holds `anchor/live-2055/`, `baseline/live-2055/`, `scripts/tsuite.sh`, `scripts/tsumm.py` and `avtau.py`.
- README section: appended to spark-one `~/dsv41-opt-20261021/README.md` as "Takeover audit after Codex goal-120", with a pointer line at the top. sha 65fead17 -> d7b325f2; the backup `README.md.bak-tw-audit` = 65fead17.
- Local parse outputs are in the scratchpad under `ta/`: Codex message extracts, site/env/stage audit JSON, and copies of the Codex reports. The Codex session logs were read only, never modified.

## 1. Live state and Codex idle check

**Codex is idle.** All of the following held at 21:18-21:28 UTC:
- **Mac:** no Codex or ChatGPT process, no crontab entry, no Codex automation. `~/.codex/automations` holds only a July salt file.
- **Codex session files:** the 3 rollouts from 10-24 10:46 (sub-agents) and the parent `2026/10/20/rollout-...c16e398c2a67.jsonl` were all last written at 21:13:11Z.
- **Codex goal state:** "reach 120 tok/s ... exact" is **paused** at 21:13:10Z after 11,771 s and 5.40M tokens. It is not cancelled, so resuming it in the Codex app would put Codex back in control.
- **Sparks:** the only user unit is `dsv41-rank` on 8/8. There is no tmux, screen, cron or user timer, and no bench or python job besides vLLM and the system daemons. Load average was 0.0-0.6.
- **API:** running and waiting were 0/0. The request counter stayed at 60 (2 stop, 58 length, 0 abort) from 21:18 until our runs. The only ssh connection was ours.
- **Codex's last process:** `RUNSTATE.json` live handle PID 78044 (its load screen) no longer exists anywhere. That load screen finished all 8 cells (see section 4) but was never audited.

**Live instance.**
- Started 2026-10-24 20:55:26 UTC on 8/8 by Codex's D05 lifecycle script. That is why the transient unit description reads `d05-lifecycle pooled-shadow-v2a normal rankN`.
- ExecStart is the normal `~/dsv41-tp8.sh <rank>`. The description resets at the next `ops/dsv41-launch-all.sh`.
- Launcher **2027f4d8** on 8/8, 22 launcher backups identical on 8/8.
- The env is the launcher defaults (k1/2 set + `DSV41_RDMA_PARLOAD=1`). The env key set is identical on 8/8, and PYTHONPATH is `~/dsv41-sched` only. There is no AV pin, no D05 or goal-120 variables, no new .pth or sitecustomize, and no goal-120 shared objects in the worker maps. The maps show only skinny_mx, the m32 native lib, libdsv41_l2pf and `rdma_ar_dsv41/build-k1pl/rdma_ar_ext.so`.
- 0 restarts, 0 ERROR/Traceback lines, `engram check OK` on 8/8, max_model_len 1,048,576, prefix caching on (vLLM default), scheduler `dsv41_sched` armed.

**Service history today (spark-one journal).** Startups at 15:03, 15:59, 16:12, 16:46, 17:02, 17:19, 18:16:51 (goal120-trace.sh), 18:41:15 (trace), 18:58:45, 19:28:16, 20:11:56 (D05 shadow v1), 20:20:01, 20:43:26 (shadow v2a) and 20:55:26.
- **Eight startups came up cleanly after C02 was installed at 18:09:37.** No relaunch-after-hang pattern appears; every stop was a planned transition.
- Under the old ~1-in-8 hang rate, 8 clean startups would happen by chance with P≈0.34, so this does not prove C02 works.

## 2. Site files vs the README's last documented live set (k1/2, 15:10 UTC)

Checked against the vllm-0.32.0 wheel RECORD on each rank. 33 modified or new site files and 43 backups, **all identical on 8/8**. Only these four entries differ from what the README documented:

| file (`site-packages/vllm/...`) | README | live (8/8) | installed by / when | gating, backup |
|---|---|---|---|---|
| `v1/worker/gpu/model_runner.py` | 9fecbda4 | **6f9d997f** | Codex C02, 18:09:37 | **Unconditional, not flag-gated.** Adds `torch.accelerator.synchronize()` after the decoder graph capture, before `speculator.capture()` (startup only). Backup `~/dsv41-goal120-20261024/backups/model_runner.py.9fecbda4` on 8/8 |
| `distributed/device_communicators/dsv41_l2_prefetch.py` | 7d2acdc8 | **4da7745b** | Codex D01, 18:09:37 | `DSV41_L2_PREFETCH_PRUNE_ROUTER/_SHARED/_STALE_QKV`, `_PLAN_AUDIT`, `_AUDIT_TOKENS`, all default off. By inspection, flags off gives the same events and plan. Backup `.../backups/dsv41_l2_prefetch.py.7d2acdc8` on 8/8 |
| `distributed/device_communicators/rdma_all_reduce.py` | 99102ff1 | **acbeeef7** | Codex k1/3 promotion, 15:56:56 | `DSV41_RDMA_PARLOAD` (launcher default 1) selects `~/rdma_ar_dsv41/build-k1pl/rdma_ar_ext.so` 62daad6c on 8/8. Legacy `build/` dfeeab00 kept. `.bak-k1-3-rdma-loads` = 99102ff1 on 8/8 |
| launcher `~/dsv41-tp8.sh` | 1a2f26f4 | **2027f4d8** | Codex k1/3, 17:18:40 | `.bak-k1-3-rdma-loads` = 1a2f26f4 on 8/8 |

Codex's README header at 17:35 documents only the k1/3 rows. The two 18:09 installs are documented for the first time in our new README section.

**Staged but not live:**

| item | where | 8/8 consistency |
|---|---|---|
| D05 native package (`d05-shadow-v1/native`, 889 files) | `~/dsv41-goal120-20261024/` on all 8 | identical (file-list hash 2a8a84f7e765) |
| D05 serving adapters v1 / v2 (py) | `d05-shadow-v1/serving-v1`, `d05-shadow-v2/serving-v2` | identical (b5a08273dfa5 / c807f9d1eaf0). The JIT dirs differ per rank, which is expected for cubins |
| `staging/` (C02 + l2pf sources, `goal120-trace.sh`, `install-control.py`, `av-pin-initial.json`) | all 8 | identical (29e3601e8dc8) |
| D02 qkv-pipeline v2, D04 m32-deadwalk (+ native-v1), P09 dualquant | spark-four only (component tests) | n/a |
| D01 post-budget arm, schedule gate, proposal-capture prototype (alpha·markov) | Mac only, Codex workspace `~/Documents/Codex/2026-10-20/let-s-improve-our-deployment-of-2/work/goal120-20261024/` | not on Sparks |
| AV pins `~/dsv41-stage/codex-rdma-20261024/av-pin-c0.json` (272cdc20), `~/dsv41-stage/k1/3-rdma-loads/av-pin-i1C.json` (a2f27b2b) | all 8 | identical |

Installed-but-inert flags in live files: D01 prune flags, `DSV41_CAND_LENSELECT` (off), `DSV41_MOE_LIVE_ACT_MIN_ROWS/_MAX_ROWS` (defaults).

## 3. Exactness anchor

**Live anchor: `a8c3394af9d2` x3** in the 20:55 instance (21:28-21:29 UTC), plus a 4th run after the baseline load (21:37). Per prompt: 8c6a41c0 e3aa05bc 1b2d3bb7 cd93c706 9ba77590 bb1e4093. This equals our anchor. `anchor/live-2055/` holds the AV profile line.

**Both hashes come from the same method.**
- The tool is `~/dsv41-bench/exact.py` (e50b924d): 6 fixed prompts, serial, temperature 0, `max_tokens` 256. Each prompt is hashed as sha256(reasoning + NUL + content), and the combined digest is sha256 of the concatenated per-prompt hex digests.
- Codex ran this exact script through our k1 `ksuite.sh`, on the live server config with prefix caching on.

**What 9330f03c4f97 is.**
- It is an alternate greedy trajectory, identical to the anchor except p2 "sky blue" = 2228532c and p4 "haiku" = 8c5e765e.
- The other known alternate, `b7e868c9d700` (k1/2 i2-A), has p2 2228532c and p4 b246b2a0.
- These two prompts have near-tie flip points.

**Why Codex saw it.** The cause is the startup adaptive-verification cost profile under `DSV41_VERIFY_CAP=auto`, not code. The evidence comes from Codex's own instances:

| instance (start) | code | effective AV profile | exact.py |
|---|---|---|---|
| c0-recovered (16:46) | PARLOAD **off** | own startup profile (= av-pin-c0) | a8c3394af9d2 x2 |
| a1-pinned (17:02) | PARLOAD **on** | **pinned av-pin-c0** | a8c3394af9d2 x2, identical spec counters |
| final-default (17:19) | PARLOAD on | fresh startup profile P_fd | **9330f03c4f97 x2** |
| goal120 `control` (same 17:19 instance, run at 18:02) | PARLOAD on | P_fd | **9330f03c4f97 x2** |
| profile-control-screen (18:16) | + C02 + l2pf (flags off) | **pinned P_fd** (`av-pin-initial.json`) | **9330f03c4f97 x2** |
| profile-shared-pruned-screen (18:41) | + D01 PRUNE_SHARED=1 | pinned P_fd | 9330f03c4f97 x2 |
| normal-restored (18:58) | PARLOAD + C02 + l2pf | fresh | a8c3394af9d2 x2 |
| **live (20:55)** | same | fresh (v1 16.28, v2 20.85, v4 27.20, v6 34.76, d1 4.30) | **a8c3394af9d2 x4** |

Rows 1-2 show PARLOAD is exact under a common profile. Rows 3-5 show that the same profile P_fd gives 9330 both before and after C02 and the l2pf file. So C02 and l2pf with flags off are exact under a common profile. Rows 7-8 show fresh profiles on the same code returning to a8.

The verify-cap tau factors (`avtau.py`; k6/k4 from v1-v6 and d1) do not separate a8 from 9330 or b7e8 along one axis. For example, i8-C has k4 0.059, like P_fd and i2-A, yet gives a8. The flip boundary depends on several profile points, as k1/2 already noted.

**"Prefix cache off?"** In our nondet bisect, arm A8 (`VERIFY_CAP=0`, `ROUTE_LIVE=0`, 262k, **prefix caching off**) gave 9330f03c4f97 x3. Arm A9, identical except prefix caching on, gave a8c3394af9d2. Codex's report matched its hash to these A8 runs. However, Codex's instances had prefix caching **on**: the launcher has no `--no-enable-prefix-caching` and the cmdline shows no prefix-caching flag. So 9330 in Codex's runs is the same alternate branch reached through a different perturbation, the AV-driven verify widths, rather than through prefix caching.

**Verdict.** No change since the anchor altered greedy output.
- By our definition (identical output and counters under a pinned AV profile), PARLOAD, C02 and l2pf with flags off are exact.
- By Codex's definition ("exact equivalence with common calibration; cross-restart equivalence under arbitrary fresh calibration not claimed"), they are exact too.
- **Caveat for promotion gates:** 2 of the 18 measured fresh (unpinned) startups since k1/1 landed on an alternate hash with unchanged code, about 1 in 9. They are k1/2 i2-A (b7e8) and Codex final-default (9330); r4-e4 u4 also gave b7e8. A non-a8 hash on a candidate instance needs the pinned counterfactual from k1/2 (`mkpin.py` + `DSV41_AV_COST_JSON`), not an automatic reject.

## 4. Baseline on the live config (20:55 instance, `tsuite.sh`, 21:33-21:38 UTC)

| metric | live 2027f4d8 | k1/2 final live (15:09, 1a2f26f4) | Codex k1/3 pinned candidate |
|---|---|---|---|
| matched C1 coding ms/step (x5) | **33.574** (sd 0.31; trials 33.47 33.89 33.91 33.32 33.28; AL 5.13) | 34.08 | 33.449 |
| matched C1 prose ms/step (x5) | **28.620** (sd 0.40; trials 28.56 29.28 28.53 28.55 28.18; AL 2.70) | 29.29 | 28.591 |
| bench single code tok/s (1024 tok x2) | **87.86 / 87.94** (AL 2.573; counters 398/1990/625 both reps) | 86.1 | 87.02 |
| bench single prose | **69.69 / 69.62** (AL 1.816; 564/2820/459) | 69.1 | 68.72 |
| bench single math | **120.80 / 120.89** (AL 3.543; 289/1445/739) | 118.0 | 119.66 |
| conc4 agg tok/s (per-req mean / min) | **153.9** (50.2 / 40.6) | - | - |
| conc8 | **228.9, 228.7** (37.2 / 30.0) | 224.5, 227.4 | - |
| conc16 | **332.2, 339.8** (28.5 / 21.9) | 332.8, 334.6, 333.1 | - |
| conc32 | **497.8, 496.6** (20.9 / 15.6) | 493.0, 489.9, 493.5 | - |
| exact.py | a8c3394af9d2 (4th) | a8c3394af9d2 x2 | a8c3394af9d2 x2 |

GSM8K and longctx were not rerun on this instance. Codex's last gates on the same code minus the C02 sync and the inert l2pf flags (17:19 instance) were GSM8K-50 50/50, longctx 60/60 and 1M 7/7.

**Codex's load screen** (same instance, 21:11-21:13; its own prompts, thinking on, 512-token cap; per-user minimum / aggregate tok/s). It answers the user's question to Codex, "1/4/8/16":

| | c1 | c4 | c8 | c16 |
|---|---|---|---|---|
| code | 76.4 | 47.1 / 179 | 33.9 / 263 | 24.5 / 389 |
| prose | 71.6 | 40.6 / 157 | 29.9 / 230 | 22.1 / 341 |

Codex interpreted the target as **at least 120 tok/s per user on both code and prose at 1-4 users**. The user confirmed that interpretation in the Codex thread: "At least 120 per user on code and prose". On Codex's own math, C1 code needs about 7.9 ms/step less and prose about 10.8 ms less at fixed acceptance. At C4, even 6 tokens per step at the measured 50.4 ms wall gives only 119 tok/s per user.

## 5. Codex goal-120 experiments (2026-10-24 16:42-21:13 UTC)

The workspace on spark-one is `~/dsv41-goal120-20261024/` and `~/dsv41-opt-20261024-codex-rdma/`. The Mac workspace is `~/Documents/Codex/2026-10-20/let-s-improve-our-deployment-of-2/{outputs,work/goal120-20261024}`; its key files are `latest-results.md`, `README.md`, `RUNSTATE.json`, the 30-experiment roadmap `outputs/experiment-roadmap-20261024/` and `deadstores/next-decode-priority.md`.

| # | experiment (Codex ID) | hypothesis / change | key numbers | outcome |
|---|---|---|---|---|
| 1 | Recovery of my interrupted k1/3 transition | Services had been stopped at 16:25 by my driver when the limit hit | Restarted the qualified 1a2f26f4 bundle | done |
| 2 | **k1/3 rdma_ar parallel loads** (`DSV41_RDMA_PARLOAD`) | Issue all 8 peer loads before the ordered FP32 accumulate (exact) | Pinned pair c0 vs a1: C1 coding 33.804 -> 33.449 (-1.05%), prose 29.040 -> 28.591 (-1.55%), 16/16 trials lower, identical counters. My earlier unpinned pair: -1.1% / -1.9%. Single +0.5 to +1.1%. C2/C4 mixed (2u greedy 103.2 -> 101.8, 4u 153.6 -> 155.2). GSM8K 49/50 ctl, 50/50 cand. longctx 60/60, 1M 7/7 (TTFT 211.8 s) | **PROMOTED** 17:18 (launcher 2027f4d8). Two pairs, each above 0.8% on C1, so it meets our C1 bar |
| 3 | Fresh control on 2027f4d8 | Baseline | code / prose / math 87.845 / 70.270 / 120.545 (AL 2.583 / 1.827 / 3.569). exact 9330 (P_fd) | data |
| 4 | **C02** startup stream ordering | rdma_ar capture hang caused by the decoder capture stream overlapping DSpark capture | One `synchronize()` between the two captures | **INSTALLED unconditionally** 18:09. 8 clean startups since; not proven |
| 5 | Profile, nsys C1/C2/C4 (pinned P_fd) | Fresh critical path | Step wall 28.899 / 36.064 / 50.428 ms. Target span 25.08 / 31.55 / 44.28. Drafter span 2.64 / 3.24 / 4.68. FC1/FC2 kernel sums 5.34/3.11, 8.43/5.18, 14.44/9.70 ms. C1 categories (overlapping): dense 7.68, RDMA 5.94, L2 prefetch 2.94. 30 graphs per rank | artifacts `profile-control/r{0..7}.{1,2,3}.{nsys-rep,sqlite,analysis.json}`, reusable |
| 6 | **C05** RDMA one-shot / two-shot crossover | Retune 32 KiB threshold for the PARLOAD binary | 28 eager + 112 graph checks per rank pass. Current policy won **56/56** paired bf16 timings | rejected; keep 32 KiB |
| 7 | **D01** shared-expert prefetch pruning (`PRUNE_SHARED=1`) | Fewer L2 hint bytes leave more DRAM for the collectives | Pinned C1 hashes and counters identical. Code 87.76 -> 89.57, prose 70.27 -> 71.77, math 120.62 -> 122.90 (about +2%, 2-repeat screen). Target span -0.50 ms. Net selected hints only -15 MiB (165 shared removed, 150 router backfilled). C2/C4 hashes vary, but the control varies too | **HOLD**; flags installed default off. One instance pair only |
| 8 | D01 post-budget arm | Remove shared hints without backfill | 10 CPU planner tests | prepared, not run |
| 9 | **D02** qkv_a load/barrier pipeline v2 | All rows in flight, one barrier | 600 bitwise checks. Warm -18.9 / -18.6 / -16.2% at M4/6/8, rotating weights -14.0 / -10.9 / -8.3%. About 0.076-0.094 ms/step projected | not installed (small); spark-four |
| 10 | **D04** M32 dead-tail termination | Stop invalid tile walks | 187 synthetic + 1,632 (layer 0) + 85 (layer 19) bitwise cases pass. Whole layer -0.27 to +0.97% (M6 slightly worse) | **HOLD**, not installed |
| 11 | **D05** skip fully dead 8-row FC2 output-store groups | Fewer FC2 epilogue stores | 148 quick + 490 full-shape bitwise cases; memcheck, synccheck and racecheck clean. Warm routed MoE -7.16 to -9.04% (M6-96), isolated FC2 -20.3 to -22.0% (layer 19, TP3, synthetic routes). Serving adapter: 4,264 GPU checks across 8 ranks. Shadow v1 (20:12): 0 comparisons (guard false positive). Shadow v2a (20:43): **21,584 live comparisons / 900,576 rows, 0 mismatches**, layers 0 and 19 only, 4 shapes, original output served | **not in the serving path**; package staged 8/8. No serving A/B, no M4 timing, 38 layers uncovered. Codex ranks it #1 next decode candidate. Its own budget: 20% of FC2 ≈ -0.62 / -1.04 / -1.94 ms at C1/C2/C4 (2.2 / 2.9 / 3.9%) |
| 12 | **P09** prefill dual quantization | One BF16 read emits shared MXFP8 + routed FP8 | Stage A 134 / 297 checks; isolated quant -32.7% (4K) / -34.1% (16K). Stage B full shared+routed MoE -1.46% at 4K (8/12 pairs) / -3.10% at 16K (12/12) | not installed; spark-four |
| 13 | Per-user baseline + long completions | 120 tok/s per user goal | 512-cap matrix, 11 cells / 29 requests: C4 worst code 47.1, prose 40.6, mixed 37.1. C1 completions: code 111.97 tok/s (16,218 tok), prose 69.67. C4 completions: worst code 48.99, prose 38.90; code 1/4 answers finished | data |
| 14 | Concurrency exactness analysis | Can C2/C4 HTTP runs prove exactness? | **No.** C2/C4 greedy outputs and counters vary across warm repeats even in the control (batching changes operator shapes). The C1 panel is stable | finding; fixed-cohort **schedule gate** prototyped (28 CPU tests, Mac only) |
| 15 | Proposal quality: alpha·markov_bias at position 0 | Raise acceptance without a drafter pass | Capture prototype, 46 CPU tests; no data | prototype, Mac only |
| 16 | Drafter investigation (D09) | Is a compact / M32 drafter worth it? | Drafter FC1+FC2 0.55 / 0.93 / 1.50 ms (C1/C2/C4). Ideal saving 0.09-0.23 ms | deprioritised |
| 17 | Load screen c1/c4/c8/c16 | User question "1/4/8/16" | Table in section 4 | finished; Codex never audited it |

Left mid-way when Codex aborted at 21:13:
- **load-screen-c16-v1:** finished, but not audited and not reported.
- **d05-shadow-v2:** the v2a live shadow completed and normal service was restored at 21:03:59. Its next planned steps had not started: all-40-layer coverage, the largest-shape repeats, the M4 timing, and a serving A/B with AV pinned.
- **Proposal capture and the schedule gate:** local prototypes only.

Codex's own planning doc lists 30 roadmap items (P01-P10, D01-D10, C01-C10). Only D01, D02, D04, D05, P09, C02 and C05 got any GPU or serving work. The rest are untested hypotheses.

Earlier Codex stints on 10-20/21 overnight, on the M64 deployment, are already folded into our README (rounds 1-4). For context, they covered the adaptive/SM121 patch, the M64 deployment, Engram residency, the M32 trial that failed retention, the 33,280-input M64/M32 bitwise comparison, the WO-A BF16-emulation finding and a WO-A Marlin trial (about 2% slower at 1 user).

## 6. Leads: already covered by Codex vs still open

**Covered. Do not repeat; reuse the artifacts:**
- **rdma_ar consumer-loop load diet** (R1 #3): promoted (PARLOAD).
- **One-shot/two-shot RDMA threshold retune** (C05): 56/56 favour current. Closed.
- **Fresh critical-path profile on 2027f4d8 at 1/2/4 users:** done (`profile-control/`, 8 ranks x 3). Reuse it instead of a new nsys window. 1M-context and 16u profiles are not covered.
- **M32 dead-tile walk termination** (R2 #1, first part; Codex D04): bitwise, but under 1% even warm. Closed for decode.
- **skinny_mx qkv_a "all rows in flight, one barrier"** (R3 #3; Codex D02 v2): built and bitwise, 16-19% kernel gain but only about 0.08-0.09 ms/step. Low value; the code exists on spark-four if ever bundled.
- **Compact/M32 drafter MoE** (R2 #7; Codex D09): analysed, at most about 0.09-0.23 ms. Deprioritised.
- **Concurrency exactness via HTTP C2/C4 hashes:** shown invalid. Keep using C1 exact.py + GSM8K + longctx as the exactness gates, as our rules already do.

**Partially covered. Strongest next steps, with Codex's evidence attached:**
1. **D05, dead FC2 store groups** (R1 #5, first half). Component-qualified and shadow-verified on 2 layers; package staged 8/8.
   - **Open:** wire it into the real path behind a default-off flag, add M4/M6 timing and all-layer coverage, then run the ABAB serving A/B.
   - **Expected gain:** largest at concurrency (≈2-4% of the step).
   - **Interaction:** the k1/2 note says that with LIVE_ACT on, FC2 padding rows hold stale bytes. D05 skips exactly those stores, which is compatible.
2. **L2 prefetch re-plan** (R2 #5; takuz0 attention-AR window ~0.76 ms; critic C6 "q_b never prefetched at 16u").
   - Codex's `PRUNE_SHARED` arm showed about +2% C1 single with identical hashes and counters, but on one pinned pair only. The router backfill ate 150 of the 165 MiB saved.
   - **Open:** the post-budget (no-backfill) arm, the attention-AR window prune, `PRUNE_STALE_QKV` (implemented, never measured; it targets the stale qkv_a shard reads, which is the critic's q_b point), and C01 pacing.
   - The flags are already installed (inert), so `PRUNE_SHARED` / `PRUNE_STALE_QKV` are env-only arms via launch-conc.sh. That makes them cheap ABAB candidates under our rules.
3. **C02 capture hang fix** (R2 #2). One synchronize is installed unconditionally, with 8 clean startups since.
   - **Open:** a root-cause check, the fail-fast part (detect a hang in about 1 min, auto stop-all and relaunch), and folding C02 into our documented set behind a flag or keeping it as-is.
   - This should be documented, not reverted.
4. **P09 dual quant + P08 Gather-A FC1** (R3 #6 / #5). Prefill only, -1.5 to -3.1% of MoE at 4-16K. Not a decode lever; low priority for the decode goal.
5. **Proposal quality / acceptance** (Codex alpha·markov; our R2 #8 is the sampled-only λT). Codex's math says acceptance is the only path to large C1 gains (about +0.95-1.3 AL needed). Capture prototype exists; no data.

**Open. Codex did not touch these:**
- **R3 #1** router gate exact cuBLAS-replica kernel, M≤8.
- **R3 #2** PDL extension: FP4 weight prologue before `griddepcontrol.wait`.
- **R2 #4 relaunch diet:** skip the 185-run AV profile when pinned, and stop DSpark walking 48 shards. Codex always paid the full startup.
- **R1 #4** flashinfer `decode_dsv4` split-K for small-T sparse MLA.
- **R1 #6** fused deterministic MoE front.
- **R1 #7** M32 cap 192 rows.
- **R1 #8** delayed-mHC overlap.
- **R1 #10** compact-scratch long-context decode indexer.
- **R1 #5, second half:** FC2-compact v2 re-gate, which is Codex D06.
- **R2 #6** wo_a sonar 32/64-row tiles at 33-128 rows.
- **R2 #3** prefill sequence parallelism re-gate (C03).
- **R2 #9** fastokens tokenization (P02).
- **R2 #10** sparse_mla_prefill 2 tokens per tile (P06).
- **R3 #4** lone-prefill 24-32K chunks (C07).
- **R3 #7-#10:** drafter glue, hc-prenorm TF32, eager glue diet, CERT lm-head tile table.
- **R1 #9 / D03** vocab-sharded Gumbel drafter for sampled rows.
- **C09** AV calibration per path/concurrency.
- **k1/1 leads:** per_row vs persistent topk above 8 rows at 4-32K contexts; fused CANON (about -0.2 ms/step at 1u).

## 7. Notes for the next steps

- **Promotion A/Bs.** Pin AV on both arms with `DSV41_AV_COST_JSON`; pins `av-pin-c0.json` and `av-pin-i1C.json` exist on 8/8. Startup profiles flip exact.py to b7e8 or 9330 with unchanged code about 1 time in 9, so judge exactness by pinned counters and the counterfactual.
- **Hang-fix step.** C02 is already live and unflagged. Baseline its effect by counting startups, and add the fail-fast detector. Do not revert it without a reason.
- **D05 handoff.** The Codex package and the shadow tooling are on 8/8. The immutable native package is in `~/dsv41-goal120-20261024/d05-shadow-v1/native`, with the source patch and the GPU gates in Codex's Mac workspace `deadstores/`.
- **Housekeeping (none done).** The unit description string resets at the next launch-all. Codex's goal is paused in its app, so ask the user to cancel it or keep the Codex app closed while we drive.
