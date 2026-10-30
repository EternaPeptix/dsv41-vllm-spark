
## Round 2 / exp4: vocab-sharded Markov drafter (DSV41_DSPARK_SHARDED_MARKOV): NOT promoted (2026-10-21/22 UTC)

**What it is:** a patch to `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py` (sha 97bcfaa6 -> 774b34c2). When the env flag is set, each rank adds the Markov bias only to its own 16,160-token vocab slice. The ranks then exchange one 16 B (max, id) record per request per position (5 small fp32 gathers per step) instead of each reading the full 129280x256 Markov head 5 times and all-gathering full-vocab logits.
Sources: Mac scratchpad `r2/drafter/` (install.sh, markov-shard/). Install: `bash install.sh markov`. Revert: `bash install.sh markov rollback`.

**Correctness (all checks passed):**
- rank0.log has `DSV41_DSPARK_SHARDED_MARKOV: vocab shard [0, 16160) pad 0, markov rows/rank 16160 of 129280, tp 8`, with no "ignored" warning.
- RDMA_AR_AUDIT confirms the new `all_gather (n,4) float32` calls go through rdma_ar, not NCCL.
- exact.py was self-consistent in every instance. GSM8K-50 at conc 8 scored 50/50 in all 4 A/B instances.
- AV cost profile draft_ms(1): 5.10 / 5.20 (control) -> 3.72 / 3.76 (arm), i.e. -1.4 ms. draft_ms(16): 12.5 -> 9.95.

**A/B** (C = live config, A = +flag; same launcher d0749eda; order C1 A1 C2 A2). AL = mean accept length.

| inst | code | prose | math | mean | c4 | c8 | c12 | c16 | AL code/prose/math | GSM8K |
|---|---|---|---|---|---|---|---|---|---|---|
| C1 | 76.29 | 58.17 | 110.38 | 81.61 | 140.78 | 212.26 | 267.22 | 316.45 | 2.59/1.73/3.78 | 50/50 |
| A1 | 72.51 | 60.44 | 113.84 | 82.26 | 144.64 | 216.86 | 274.68 | 322.42 | 2.27/1.69/3.72 | 50/50 |
| C2 | 72.61 | 57.14 | 107.65 | 79.13 | 140.70 | 206.21 | 265.92 | 311.93 | 2.58/1.71/3.69 | 50/50 |
| A2 | 71.31 | 58.04 | 111.14 | 80.16 | 144.41 | 212.76 | 278.47 | 320.44 | 2.36/1.64/3.64 | 50/50 |

Means, control -> arm:
- single 80.37 -> 81.21 (+1.0%; gate needs +2.5%)
- conc4 140.7 -> 144.5 (+2.7%)
- conc8 209.2 -> 214.8 (+2.7%)
- conc12 266.6 -> 276.6 (+3.8%)
- conc16 314.2 -> 321.4 (+2.3%)
- code AL 2.585 -> 2.31 (-0.27, outside the +/-0.08 gate). Prose -0.06, math -0.06.

**Why code AL dropped: the drafts did not get worse.** An AV-off probe was run (`DSV41_SPEC_JSON={"method":"dspark","num_speculative_tokens":5,"enable_adaptive_verification":false}`, DC1 vs DA1, single x2 reps):
- Untrimmed acceptance matches: code 2.80 vs 2.78, prose 2.13 vs 2.10, math 3.59 vs 3.80, conc4 2.78 vs 2.82.
- exact.py hashes were identical across the two instances (47098a31f5e4).
- The cause is the adaptive-verification trimmer. It picks the draft budget k = argmax (1+sum s_i)/(draft_ms + verify_ms(k)). A cheaper draft_ms lowers the fixed cost, so the profiled verify slope (already ~3x too steep, see the drafter prep notes) trims harder. Code loses positions 2-5 and about 3% tok/s, which cancels most of the drafter saving at 1 user.
- With AV off, the flag gives +6.1% single (77.6 -> 82.3). At concurrency, the trimmer's decisions are dominated by verify cost, so the gain shows through.

**Result:** not promoted under the exp4 gate (single +1.0% < +2.5%; code AL outside +/-0.08). The patch was rolled back (speculator.py = 97bcfaa6 on all 8). The launcher is unchanged (d0749eda). Live = same config as before exp4 (launch-all, no extra env), /health 200, sanity 391.

**Follow-up:** re-test this flag together with the AV cost calibration (drafter patch B: `DSV41_AV_VERIFY_SCALE` / `DSV41_AV_COST_JSON`). Once the verify slope is calibrated, the cheaper drafter should no longer cause over-trimming at 1 user, and the +2.3-3.8% conc gain plus about +6% untrimmed single gain should both be available.
Raw results: `~/dsv41-opt-20261021/r2-exp4/` (C1 A1 C2 A2 DC1 DA1, summ2.py, dsumm.py, inst.sh, diag.sh).
