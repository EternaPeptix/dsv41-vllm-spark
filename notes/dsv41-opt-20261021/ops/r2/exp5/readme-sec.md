
## Round 2 / exp5 — adaptive-verification cost calibration (PROMOTED, 2026-10-22 01:31 UTC)

**What changed**
- `vllm/v1/worker/gpu/spec_decode/adaptive_verification.py` was patched (av-calib) on all 8 ranks: sha b0526416 -> ec240369. Backup is `.bak-avcal`. Installed with `bash $D/install.sh av`, where D = Mac scratchpad r2/drafter.
  - The patch adds the env knobs DSV41_AV_VERIFY_SCALE / SCALE_N0 / SCALE_N1 / FIXED_MS / COST_JSON / DUMP_JSON / LOG_BUDGET.
  - It recalibrates the startup cost profile on rank 0 before the profile is broadcast to the other ranks.
- `~/dsv41-tp8.sh` changed from d0749eda to e270215f on all 8 ranks; backup is `~/dsv41-tp8.sh.bak-avcal`. It now exports these defaults:
  `DSV41_AV_VERIFY_SCALE=0.6 DSV41_AV_FIXED_MS=2.0 DSV41_AV_SCALE_N0=48 DSV41_AV_SCALE_N1=56`.
  - Every verify width up to 48 is priced as V(1) + 0.6*(V(n)-V(1)). The price tapers back to the profiled cost by width 56; 48->56 is the cliff where the M32 MoE path ends.
  - Draft cost gets +2.0 ms for the time spent outside the graphs.
- Rank0.log shows a line 'DSV41 adaptive-verification calibration (... scale=0.6 n0=48.0 n1=56.0 fixed_ms=2.0)'.

**Why:** the startup profile uses random tokens. Its verify slope is about 3.5 ms per extra draft at 1 user, but the measured slope is 2.39 ms per draft: (math - prose step ms) / (math - prose drafts verified) from the C1 budget histograms. That puts the refit scale at a = 0.68. The time outside the graphs refits to 2.1 ms (prose and math both).
- The planner's default taper (N0=8, N1=48) made the cost slope at widths 24-48 *steeper* than the profile. At conc12/16 the trimmer then cut verified tokens from about 30 to 15-22, and conc12/16 regressed by 2-6% (arms A1 a=.4/f2.5 and A2 a=.6/f2).
- N1=60 (A4) still regressed by 1.4%. A flat scale up to 48 (A5) fixed it.

**A/B (same window, C/A alternated, every instance got GSM8K-50 at conc 8 and an exact.py x2 self-consistency check):**

| inst | code | prose | math | single mean | c4 | c8 | c12 | c16 | GSM | accept per pos (math) |
|---|---|---|---|---|---|---|---|---|---|---|
| C1 | 68.5 | 55.5 | 106.8 | 76.95 | 135.8 | 205.4 | 264.8 | 317.2 | 50 | .84/.64/.56/.30/.26 |
| A2-1 (a.6 f2 N1 48) | 78.0 | 58.8 | 109.7 | 82.15 | 145.1 | 204.8 | 255.3 | 309.8 | 50 | .87/.72/.59/.36/.30 |
| A1-1 (a.4 f2.5 N1 48) | 76.2 | 58.9 | 103.1 | 79.40 | 142.3 | 208.8 | 253.8 | 293.4 | 49 | .82/.64/.54/.33/.27 |
| A4-1 (a.6 f2 N1 60) | 78.8 | 59.0 | 111.1 | 82.96 | 144.9 | 207.1 | 267.3 | 310.4 | 50 | .86/.72/.61/.38/.31 |
| C2 | 70.5 | 58.0 | 112.5 | 80.32 | 144.8 | 207.4 | 269.5 | 312.3 | 50 | .88/.71/.61/.36/.29 |
| A5-1 (a.6 f2 N0 48 N1 56) | 77.0 | 58.9 | 115.4 | 83.76 | 143.2 | 210.5 | 268.4 | 319.6 | 50 | .87/.75/.65/.43/.36 |
| C3 | 72.3 | 56.8 | 112.4 | 80.52 | 140.1 | 206.6 | 266.6 | 311.1 | 50 | .85/.71/.60/.38/.33 |
| A5-2 (same) | 79.9 | 60.0 | 113.3 | 84.38 | 144.4 | 212.0 | 271.8 | 313.2 | 50 | .87/.74/.61/.41/.33 |

Mean of the 3 controls vs A5 (2 instances):
- single: code 70.4 -> 78.5 (+11.4%), prose 56.8 -> 59.4 (+4.7%), math 110.6 -> 114.3 (+3.4%); mean **79.3 -> 84.1 (+6.1%)**
- conc4 140.2 -> 143.8 (+2.5%), conc8 206.5 -> 211.3 (+2.3%), conc12 267.0 -> 270.1 (+1.2%), conc16 313.5 -> 316.4 (+0.9%)
- Every instance was self-consistent on exact.py.

Drafts verified per step at 1 user, control -> A5:
- prose 1.61-2.09 -> 2.14-2.33 (verifying 1 draft: 68% -> 37-48%)
- code 2.70 -> 3.48
- math 3.22 -> 4.04 (verifying 5 drafts: 34% -> 56-59%)

Total tokens verified per step:
- r4: 7.4-7.7 -> 8.0-8.4
- r8: 21-22 -> 24
- r16: 30 -> 31.5

Code position-2 acceptance went from .36-.40 to .51-.55; prose position-2 acceptance went from .08-.10 to .16-.20.

**Cumulative since round 1** (76.6 / 134.7 / 192.6 / 277.3): single 84.1 (+9.8%), conc4 143.8 (+6.8%), conc8 211.3 (+9.7%), conc16 316.4 (+14.1%).

**Rollback**
- Launcher only (trimmer becomes stock): `cp ~/dsv41-tp8.sh.bak-avcal ~/dsv41-tp8.sh` on all 8 ranks. Alternatively launch with DSV41_AV_VERIFY_SCALE=1 DSV41_AV_FIXED_MS=0. Patch with scale 1 / fixed 0 = stock behaviour.
- Site file: `bash $D/install.sh av rollback`, then restart with `dsv41-launch-all.sh`.

**Raw data:** ./r2-exp5/<inst>/ (bench output, rank0.log copies, av-curves JSON) plus av5_summ.py and nrhist.py.
