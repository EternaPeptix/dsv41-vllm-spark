
## pd decode A/B step (2026-10-25 20:59-23:25 UTC): D05 + L2 prune-shared PROMOTED, pdl HOLD, rgate KILLED

Prepared candidates from "Offline prep pd" (`~/dsv41-stage/pd/README.md`), run serially by expected value. Raw results are on spark-one in `~/dsv41-pdab-20261025/`: `STATE.md` (running log), `runs/<label>/` (one per instance), `scripts/asuite.sh` and `asumm.py`. The Mac scratchpad `pdab/` holds the drivers `runarm.sh`, `stopall.sh`, `pushlauncher.sh`, `mapscheck.sh` and `abcmp.py`, plus `summ-*.json`, `cmp-*.txt` and `launches.tsv`.

**Live state after this step.** Everything below is sha256-identical on 8/8.
- **Launcher** `~/dsv41-tp8.sh` **04b6738e**. It adds `export DSV41_MOE_FC2_D05=${DSV41_MOE_FC2_D05:-1}` and `export DSV41_L2_PREFETCH_PRUNE_SHARED=${DSV41_L2_PREFETCH_PRUNE_SHARED:-1}`.
  - Backups: `.bak-pd-l2pfS` = 92dbac8e (D05 only), `.bak-pd-d05` = b1e75497 (pre-step).
- **Site files** (installed with `pd-all.sh`; backups are `<file>.bak-pd-<tag>`):
  - d05: `fused_moe/experts/sm121_target_m32.py` 723fdf8d -> **5acc5fde**; new `sm121_d05_verify.py` **c04c363d**.
  - l2pf: `device_communicators/dsv41_l2_prefetch.py` 4da7745b -> **514dcb23**. The S arm was measured on this file; its PB flag is off.
  - pdl: `quantization/utils/fp8_utils.py` 8dc7dab8 -> **b35c0561**, inert with the flags off: the constexpr `PDL_TRIGGER` defaults to 0, which gives identical Triton code. Every A/B instance had it installed.
  - rgate: uninstalled. `gate_linear.py` is back to 41adaba8 and `dsv41_rgate.py` is absent.
  - Unchanged from pd base: `model_runner.py` 316fca44, `dsv41_rel.py` f373f911, `rdma_all_reduce.py` 9056dcdf, `dspark.py` 36811ca9.
- **`common/pd-all.sh` fix:** the per-host loop now uses `ssh -n`. The script is piped into `bash -s`, so an in-loop ssh without `-n` reads the rest of the script from stdin. Backup `pd-all.sh.bak-pdab`, on 8/8.
- **Live instance `pdab-FINAL`.** Started 23:01:13 UTC, healthy at 23:07:27 (+373 s). AV unpinned (startup profile, "pin miss"), as with pd-FINAL.
  - Fingerprint c2f008432bd4769f on 8 ranks; 0 restarts; 0 ERROR/Traceback lines on 8/8; Engram check OK on 8/8; xstream busy 0.
  - D05 native (`d05-shadow-v1/native`) mapped on 8/8; env `DSV41_MOE_FC2_D05=1 DSV41_L2_PREFETCH_PRUNE_SHARED=1`.
  - `--enable-prompt-tokens-details` still on. A repeated 4,205-token chat shows `cached_tokens` 0 -> 4160; streaming and /v1/completions carry it too.

### Protocol
- **Launches.** Every instance came from the supervised relaunch with `DSV41_AV_COST_JSON=/home/jeweled/dsv41-stage/rel2/av-pin-live2055.json`, so the diet was a hit and the AV profile was pinned.
  - 10/10 pinned launches were healthy on the first try, in 275-287 s (downtime 287-306 s).
  - With pdab-FINAL that is 11 clean launches in a row this step (34 since C02).
- **Suite per instance (`asuite.sh`):**
  - exact.py with spec-counter deltas on the fresh instance, then a warm-up;
  - matched C1 coding/prose x5;
  - bench single code/prose/math, 1024 tokens x2;
  - conc 8/16/32 x2, 512 tokens;
  - matched x5 again, so n = 10 C1 trials per workload per instance;
  - GSM8K-50 and full longctx on the second candidate instance.
- **Matched-bench label fix (affects older A/Bs).** `matched_bench.py` puts `--label` into every prompt's front tag (`<label>-t<i>`). A per-arm label therefore changes the prompts, the greedy text and AL.
  - On the same A1 instance, coding was 33.16 ms at AL 4.88 with label `pdab-A1-ctl`, and 33.55 ms at AL 5.15 with label `pdab`.
  - Earlier A/Bs that ran tsuite or ksuite with per-arm labels mixed about ±1% of prompt effect into C1.
  - This step uses one fixed label `pdab` (env `MLABEL`). Matched AL was identical on every pinned instance (coding 5.149, prose 2.695), so each C1 delta is pure step time and also confirms identical output.
- **Exactness.** exact.py gave a8c3394af9d2 with pinned counters 901/473/2365/1373 on all 10 pinned instances. The bench single counters were identical too (code 398/1990/625, prose 564/2820/459, math 289/1445/739).

### Per-candidate results (pinned AV; C1 = matched ms/step; deltas are B vs the A run just before; positive = better)

| # | candidate (flag) | stopped-window test | pair 1: C1 coding / prose, c16 / c32 | pair 2 | exact | GSM8K-50 / longctx | verdict |
|---|---|---|---|---|---|---|---|
| 1 | **d05** `DSV41_MOE_FC2_D05=1` | GPU unit 8/8 PASS: 883 cases/rank, all 40 layers x tokens 1/4/6/48, eager + graphs, poisoned, 0 fail; 321 D05 calls, 11,540 groups skipped; FC2 1.08-1.29x at m1-96 | A1 33.549/28.613 -> B1 32.709/28.121: **-2.50% / -1.72%**, c16 +3.09%, c32 +4.51% | A2 33.561/28.747 -> B2 32.685/28.101: **-2.61% / -2.25%**, c16 +5.40%, c32 +5.96% | identical x4 | 50/50, PASS 60/60 (wall 171 s) | **PROMOTED** |
| 2 | **l2pf S** `DSV41_L2_PREFETCH_PRUNE_SHARED=1` (on d05) | cache hints only | B2 -> L1-S 32.015/27.508: **-2.05% / -2.11%**, c16 +0.90%, c32 +0.12% | C3 32.743/28.107 -> L4-S 32.183/27.529: **-1.71% / -2.06%**, c16 -0.59%, c32 +0.27% | identical | 50/50, PASS 60/60 (174 s) | **PROMOTED** |
| 2b | l2pf PB `..._POST_BUDGET=1` (screen) | CPU 10/10 | B2 -> 32.357/27.723: -1.00% / -1.35%, c8 -2.17%, c32 -0.87% | - | identical | - | not promoted: about 1% behind S |
| 2c | l2pf SQ `S + ..._PRUNE_STALE_QKV=1` (screen) | - | B2 -> 32.062/27.509: -1.91% / -2.11% (vs S -0.15% / 0.00%) | - | identical | - | not promoted: no gain over S |
| 3 | rgate `DSV41_ROUTER_GATE_REPLICA` | probe on spark-two: bitwise at M2-8 (H1z/0, 2000 trials per M + graph replay, 40 layers); M1 not exact. Replica is **slower**: warm 0.83-0.96x at M2-8, cold 0.65x (32 vs 21 µs). m_allow empty, no config written | - | - | - | - | **KILLED**, uninstalled 8/8 |
| 4 | pdl EB `DSV41_MOE_PDL_EARLY_B=1 DSV41_MOE_PDL_TRIGGER=start` (on d05+S) | unit PASS: 80 cases, 200 bitwise graph replays per arm; racecheck 0 hazards, synccheck 0 errors | L4-S -> P1 32.078/27.484: -0.33% / -0.16%, single +0.25..0.50%, c16 +0.57%, c32 +0.19% | not run | identical | - | **HOLD**: exact and slightly positive, but at most about 0.1 ms; it cannot reach the 0.8% C1 / 2% conc gates (the prep's own ceiling is 0.75%). Stopped after pair 1; flags off |
| 5 | fc2cd05 | not built (prep: 4 host hunks to hand-merge, native rebuild, +2.5 GiB/rank at 1M) | - | - | - | - | not run |

Absolute numbers per instance (single code/prose/math tok/s; conc 8/16/32 aggregate tok/s):

| instance | config | C1 coding / prose ms | single | conc 8 / 16 / 32 |
|---|---|---|---|---|
| A1-ctl | pre-step | 33.549 / 28.613 | 87.98 / 69.72 / 120.81 | 227.2 / 339.6 / 496.9 |
| B1-d05 | +D05 | 32.709 / 28.121 | 89.62 / 70.90 / 122.94 | 230.4 / 350.1 / 519.3 |
| A2-ctl | pre-step | 33.561 / 28.747 | 87.70 / 69.61 / 120.42 | 225.1 / 337.2 / 490.3 |
| B2-d05 | +D05 | 32.685 / 28.101 | 89.59 / 70.94 / 122.89 | 239.4 / 355.4 / 519.5 |
| L1-S | D05+S | 32.015 / 27.508 | 91.64 / 72.68 / 125.72 | 238.0 / 358.6 / 520.1 |
| L2-PB | D05+PB | 32.357 / 27.723 | 90.69 / 71.93 / 124.75 | 234.2 / 356.1 / 515.0 |
| L3-SQ | D05+S+SQ | 32.062 / 27.509 | 91.48 / 72.59 / 125.44 | 238.3 / 356.7 / 520.6 |
| C3-ctl | D05 (launcher 92dbac8e) | 32.743 / 28.107 | 89.53 / 70.86 / 123.09 | 237.5 / 354.6 / 519.0 |
| L4-S | D05+S | 32.183 / 27.529 | 91.35 / 72.56 / 125.21 | 235.8 / 352.5 / 520.4 |
| P1-EB | D05+S+pdl | 32.078 / 27.484 | 91.64 / 72.74 / 125.83 | 237.3 / 354.5 / 521.4 |
| **FINAL** | launcher 04b6738e, **unpinned** | 32.585 / 27.721 (AL 5.224 / 2.718, own profile) | 90.59 / 72.73 / 126.06 | 240.9 / 356.2 / 518.3 |

- **Stacked effect, D05 + S vs pre-step,** pinned, means of 2 instances each, same session:
  - C1 coding 33.555 -> 32.099 ms (**-4.3%**); prose 28.680 -> 27.519 ms (**-4.0%**);
  - single +4.2 / +4.2 / +4.0%;
  - conc8 +4.8%, conc16 +5.1%, conc32 +5.4%.
- **The two gains add up** (S measured on top of D05). D05 helps at every concurrency. S helps at 1 user (matched C1 and single) and is neutral at 8-32 users, where the shared-expert hints are not the binding cost.
- **pdab-FINAL vs the pre-step live baseline:** single +3.1 / +4.3 / +4.3%, conc8 +5.2%, conc16 +6.0%, conc32 +4.3%.
  - exact.py on its own fresh profile: a8c3394af9d2, counters 905/471/2355/1373. GSM8K-50 50/50. Full longctx PASS 60/60, copy 1.000.
  - Its matched C1 is not comparable with the pinned rows: its own AV profile picks other verify widths, giving AL 5.224 vs 5.149.

### Notes and next steps
- **New AV pin for A/Bs on top of this config:** `~/dsv41-stage/pdab/av-pin-pdfinal.json` (0d28dcb4, on 8/8). It holds the pdab-FINAL raw curves plus the live2055 shape_key, and v1 verify is 15.56 ms vs the pin's 16.28.
  - The live2055 pin is now stale for D05+S kernels. It is fine as a common pin within one A/B, but use the new pin for future arms.
  - The default-pin proposal (pd base step) remains a **user decision**. If accepted, point it at this file.
- **l2pf PB with S on.** S is now the launcher default, so a PB arm needs `DSV41_L2_PREFETCH_PRUNE_SHARED=0 DSV41_L2_PREFETCH_PRUNE_SHARED_POST_BUDGET=1`; the module raises on mixed modes.
- **pdl.** Pair 1 sits at the noise floor. Re-test only as a bundle with the not-yet-built FC1 prologue and glue-kernel arm A (pd/pdl README "Not done"). fp8_utils.py b35c0561 is installed inert. Codex P09 dual-quant edits the same file and would need a merge; the staged prefill packages (`pf`, `sched-pd`) do not touch it.
- **D05 dependencies.** D05 lives in `sm121_target_m32.py` 5acc5fde, and the native tree is `~/dsv41-goal120-20261024/d05-shadow-v1/native`, which must stay. The prefill `pf` package edits `deep_gemm_utils.py` (a4a0486a); the d05 GPU test asserts that hash, so re-run `d05/ops/run-gpu-tests.sh` after any such install.
- **Not run:** the D05 live `verify` shadow, which is optional because the unit test covers all 40 layers; l2pf SR; pdl EBe/T; and fc2cd05.

### Rollback
- **Env only:** relaunch with `DSV41_MOE_FC2_D05=0 DSV41_L2_PREFETCH_PRUNE_SHARED=0` (supervisor `VAR=VAL`), which gives the pre-step behaviour.
- **Launcher:** on 8/8, `cp -p ~/dsv41-tp8.sh.bak-pd-l2pfS ~/dsv41-tp8.sh` (92dbac8e, D05 only) or `.bak-pd-d05` (b1e75497, pre-step).
- **Site files** (service stopped): `ssh spark-one cat ~/dsv41-stage/pd/common/pd-all.sh | bash -s uninstall <d05|l2pf|pdl>`.
  - This restores 723fdf8d (and removes sm121_d05_verify.py) / 4da7745b / 8dc7dab8 and checks the digest is identical on 8/8.
  - Restore the launcher first when uninstalling d05.
