

## Round 3 / e1: sharded Markov drafter + AV calibration (2026-10-22 UTC). PROMOTED (arm M2)

**Change:** `DSV41_DSPARK_SHARDED_MARKOV=1` (the round-2 exp4 patch to `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py`, sha 97bcfaa6 -> 774b34c2) together with `DSV41_AV_FIXED_MS` 2.0 -> 3.4. The exp5 AV cost calibration stays live (VERIFY_SCALE 0.6, N0 48, N1 56).
Round-2 exp4 was not promoted only because the uncalibrated trimmer over-trimmed code. With calibration plus FIXED_MS 3.4, the trimmer keeps its previous choices and the drafter saves 1.5 ms per step: draft_ms(1) 5.2 -> 3.73, draft_ms(16) 12.6 -> 9.94.

**A/B** (C = round-2 live config; M1 = +flag; M2 = +flag with FIXED_MS 3.4). All instances were launched with `launch-conc.sh` in the order C1 M1 M2 C2 M1b M2b C3 M2c. Each ran runarm_conc (single code/prose/math, conc4/8/12/16, exact x2, GSM8K-50 at conc 8).

| inst | code | prose | math | mean | c4 | c8 | c12 | c16 | AL code/prose/math | GSM8K | draft_ms(1) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 | 68.06 | 60.75 | 111.31 | 80.04 | 144.67 | 211.95 | 269.12 | 319.13 | 2.50/1.88/3.91 | 50/50 | 5.29 |
| M1 | 72.82 | 61.87 | 114.50 | 83.06 | 147.08 | 216.37 | 276.27 | 321.48 | 2.56/1.80/3.83 | 49/50 | 3.73 |
| M2 | 78.06 | 64.14 | 110.78 | 84.33 | 146.94 | 219.50 | 277.86 | 327.19 | 2.77/1.88/3.68 | 50/50 | 3.72 |
| C2 | 72.24 | 59.36 | 112.89 | 81.50 | 141.45 | 211.57 | 263.36 | 310.49 | 2.68/1.83/3.93 | 50/50 | 5.24 |
| M1b | 70.29 | 61.58 | 113.37 | 81.75 | 150.58 | 218.56 | 277.47 | 322.70 | 2.42/1.82/3.79 | 50/50 | 3.73 |
| M2b | 76.67 | 64.02 | 116.58 | 85.76 | 145.94 | 217.72 | 276.01 | 327.40 | 2.70/1.88/3.92 | 50/50 | 3.74 |
| C3 | 73.39 | 58.43 | 114.98 | 82.27 | 142.23 | 206.58 | 270.35 | 315.19 | 2.73/1.79/4.04 | 49/50 | 5.20 |
| M2c | 78.31 | 62.75 | 116.89 | 85.98 | 148.12 | 215.79 | 272.59 | 318.61 | 2.81/1.85/3.95 | 49/50 | 3.72 |

exact.py was self-consistent in all 8 instances.

**Pooled, control (3 instances) vs arm:**

| | single | c4 | c8 | c12 | c16 | AL code | AL prose | AL math |
|---|---|---|---|---|---|---|---|---|
| C | 81.27 | 142.78 | 210.03 | 267.61 | 314.94 | 2.63 | 1.83 | 3.96 |
| M2 | 85.36 (+5.0%) | 147.00 (+3.0%) | 217.67 (+3.6%) | 275.49 (+2.9%) | 324.40 (+3.0%) | 2.76 (+0.12) | 1.87 (+0.03) | 3.85 (-0.11) |
| M1 | 82.41 (+1.4%) | 148.83 (+4.2%) | 217.47 (+3.5%) | 276.87 (+3.5%) | 322.09 (+2.3%) | 2.49 (-0.15) | 1.81 (-0.02) | 3.81 (-0.15) |

- **M1** fails the gate (single +1.4%, code AL -0.15). With FIXED_MS 2.0 the trimmer still trims code a little too much.
- **M2** passes: single +5.0%, every concurrency level is +2.9% to +3.6%, and code AL is up. The math AL change of -0.11 comes from one instance (M2: 3.68). M2b and M2c (3.92/3.95) are inside the control range (3.91-4.04), and math single runs at drafts_per_step 5.0 in every instance, so it is not trimming.
- **GSM8K-200 at conc16 on M2c:** 197/200, misses idx 93, 119, 184 (reference 196: 93, 100, 119, 184).
- **Long-context check:** not needed. The change touches only the drafter and the AV cost model, not attention/indexer/KV/MoE numerics, and the target model still verifies every token.

**Promotion:**
- Launcher sha: e270215f -> abd49904, identical on all 8. The backup is `~/dsv41-tp8.sh.bak-r3markov` (e270215f) on every rank.
- New lines: `DSV41_AV_FIXED_MS` default 3.4, and `export DSV41_DSPARK_SHARDED_MARKOV=${DSV41_DSPARK_SHARDED_MARKOV:-1}`.
- speculator.py = 774b34c2 on all 8. The backup is `speculator.py.bak-dsmarkov` (97bcfaa6).
- After relaunching with `dsv41-launch-all.sh` (started 04:08:43 UTC): rank0.log shows `SHARDED_MARKOV: vocab shard [0, 16160)`, with no "ignored" warning and no "first call during capture" warning. draft_ms(1) is 3.75, and GSM8K-50 at conc 8 scored 50/50.

**Rollback:**
1. Restore the launcher on every rank: `cp -p ~/dsv41-tp8.sh.bak-r3markov ~/dsv41-tp8.sh`, and check that the sha is e270215f. For a temporary rollback instead, launch with `DSV41_DSPARK_SHARDED_MARKOV=0 DSV41_AV_FIXED_MS=2.0`.
2. Run `bash <Mac scratchpad>/r2/drafter/install.sh markov rollback`, which restores speculator.py to 97bcfaa6.
3. Relaunch.

**Note:** dspark/speculator.py is no longer stock, so the r3 spec-sampling install.sh, which expects sha 97bcfaa6, will refuse to run until it is rebased onto 774b34c2.

Raw results: `~/dsv41-opt-20261021/r3-e1/` (instance dirs, inst.sh, summ2.py, dsumm.py, M2c/gsm200.*, final-gsm50.*).
