
## k1 / 1-indexer-mask: length-aware decode candidate mask + flags -- PROMOTED LENAWARE (2026-10-24 11:55 UTC); LENSELECT tested, NOT promoted (all gates pass; gain over LENAWARE within noise)

**Problem.** At live max_model_len 1,048,576 the decode indexer's candidate kernels walked the whole max_model_len-wide logits row on every step, whatever the real context. `_mask_candidates_kernel` wrote -inf over all 1M columns (grid cdiv(W,1024) = 1024 per row), `_candidate_flags_kernel` zeroed all 131,073 flag bytes per row, and the layer-20 select ran torch.topk(2048) over all 131,072 blocks. The fresh 1M profile (Mac scratchpad k1/profile-1m.md) measured the mask at 78 us per call at 4 rows and 1.09-1.46 ms at 48-64 rows, with 4 consumer layers (24/28/32/36) per step. This is most of the conc16/conc32 drop recorded under "Context 1M" (conc16 330.9 -> 311.8).

**What.** One site file, `vllm/model_executor/kernels/attention/dsa/candidate_blocks.py` 45f1eaa3 -> **a5816163** on 8/8 (backup `candidate_blocks.py.bak-k1-1-indexer-mask` = 45f1eaa3 on every rank). The diff only adds lines (every stock line is kept). Both flags default off, and off runs the unchanged stock kernels. `sparse_attn_indexer.py` (7112c01d) is untouched. Flags are read at import (each worker logs `dsv41 k1 candidate_blocks armed: LENAWARE=.. LENSELECT=.. CTAS=..`), so arms are per instance.
- **`DSV41_CAND_LENAWARE=1`** (decode only, `row_ks is None`; prefill keeps stock):
  - (b) candidate flags: zero only slots `[0, cdiv(min(end,W)-start, 8))` plus the edge slot (stock zeroes all 131,073 B per row at 1M).
  - (a) -inf mask: grid `(rows, G)`, `G = min(cdiv(W,1024), cdiv(DSV41_CAND_CTAS, rows))` (CTAS default 384), each CTA loops over the 1024-column tiles below `min(end,W)`. Every column below `end` gets exactly the stock value (edge term included); columns at or past `end` are not written. Stock writes -inf over the whole max_model_len-wide row (grid 1024 at 1M).
  - Exact: no decode reader looks at columns >= end. persistent_topk / top_k_per_row_decode read `[0, len)` per row with len <= the mask's end (both seq_lens layouts), and the layer-20 logits already reach top-k unmasked.
- **`DSV41_CAND_LENSELECT=1`** (decode only): layer-20 select = `_block_scores_bounded_kernel` over `nvalid = min(cdiv(end,8), nblocks)` blocks (same max-of-8 + pinned newest block; writes `lens[row] = nvalid` on device) -> `persistent_topk(scores, lens, out, ws, 2048, max_seq_len=nblocks)` -> -1 where the score is -inf (stock rule). No host branch on lengths (graph-safe); same 1 MiB workspace request as the decode top-k. Same candidate SET as torch.topk when nvalid <= 2048 (ctx <= 16,384 tokens; order differs, nothing reads it); above that it can differ only on an fp32 tie / NaN at the 2048th block (greedy-eq).
- `DSV41_CAND_CTAS` (default 384) is the CTA target for the looping kernels. The Phase C sweep is flat between 192 and 1536, so the default is kept.

**GPU unit test** (`bundle/test_k1_idxmask_gpu.py`, service stopped; phases A+B on spark-one, C+D on spark-two, 09:50 UTC; results in `gpu/`): **PASS, 698 checks, 0 fails**.
- **A (bitwise, eager).** rows {4,6,48,96,192} x W {262K, 1M}, ends {0,1,7,8,513,16383,16384,16385,65544,200000,300000,524288,W}, columns >= end poisoned with NaN/+inf/1e30, row_repeat 1/2/6.
  - Flag-off path == stock, bitwise.
  - LENAWARE: columns < end bitwise == stock; poisoned columns >= end untouched; edge / -1 / duplicate candidates behave the same.
  - Downstream top-512 + CANON is bitwise equal for persistent_topk (max_seq_len = W, as live), top_k_per_row_decode, and next_n > 1.
  - LENSELECT: the set equals stock for nvalid <= 2048 (112 rows). Above that, 628/628 rows gave an identical full chain (0 tie rows). Rows with NaN in real columns: equal sets 6/6.
- **B (CUDA graph).** The patched mask+select is captured once and replayed with new ends and fresh poison: W1M x 6 / 48 rows and W262K x 96 rows, all equal to eager stock.
- **C (timing, us per call, graph replay; flags+mask = apply_candidate_mask).**

| W | rows | end | flags+mask stock -> LENAWARE | select stock -> LENSELECT |
|---|---|---|---|---|
| 1M | 4 | 2K | 97.6 -> 6.2 | 86.1 -> 8.2 |
| 1M | 6 | 2K | 146.3 -> 6.2 | 97.4 -> 8.2 |
| 1M | 48 | 2K / 32K / 131K | 1097.7 -> 6.2 / 34.2 / 152.9 | 491.6 -> 11.4 / 65.1 / 210.6 |
| 1M | 96 | 2K | 2205 -> 6.2 | 1427 -> 14.9 |
| 1M | 192 | 2K | 4452 -> 8.1 | 2875 -> 22.6 |
| 1M | 48 | full (1M) | 1129 -> 1166 (+3%) | 1414 -> 1225 |
| 262K | 4 | 2K | 24.7 -> 6.2 | 65.6 -> 8.2 |

  Per step (4 masks + 1 select) at 1M and short context, this saves **0.44 ms at 4 rows, 0.65 ms at 6 rows, 4.85 ms at 48 rows, 10.2 ms at 96 rows and 20.6 ms at 192 rows**. Of that, LENSELECT is 0.08 / 0.09 / 0.48 / 1.4 / 2.85 ms. Only full-length rows (end = W) are about 3% slower on the mask, which does not matter in practice.
- **D (critic arm: decode top-512 on 1M-wide logits).** Not pursued. Rule: go ahead only if a <=262K graph set saves at least ~0.3 ms/step at rows 4-6 with length 300-4096. Measured: 25-131 us/step (4 rows: 131 / 25 us at len 300 / 4096; 6 rows: 84 / 96 us). The `max_seq_len=32768` bucket on 1M-wide logits is no better at 4-6 rows and slower at length >= 4096 (6 rows x 4096: 564 vs 482 us/step live), so it is dead too. Other findings:
  - At <= 8 rows on 1M-wide logits, live persistent_topk already takes the per_row path (20.4 us at 4 rows x 300 for both).
  - CANON alone (`_canon_topk_`: torch.sort + where + masked_fill + copy) costs a flat **~28 us per call** at every shape, x 8 index layers = **~0.22 ms/step**.
  - **Above 8 rows, persistent_topk's cost scales with the logits row width, not with the row lengths.** At 48 rows x len 4096 it takes 320.7 us at W=1M, 51.2 us at 131K, and 31.2 us for per_row. At 48 x 32768: 896 us vs per_row 145 us. Per-step model (5 ratio-1 layers at 1M + 3 ratio-2 at 512K, CANON included):

| rows | ctx (ratio-1 len) | live 1M us/step | <=262K graph set |
|---|---|---|---|
| 16 | 4096 | 963 | 438 |
| 48 | 300 | 337 | 273 |
| 48 | 4096 | 2345 | 797 |
| 48 | 32768 | 6048 | 1763 |
| 96 | 32768 | 11921 | 3291 |

  The bench conc runs use ~300-token contexts (48 rows x 300 = 0.34 ms/step), so this cost is invisible in the conc16/conc32 numbers here. Real multi-user traffic with 4K-32K contexts pays 1.5-4.3 ms/step at 48 rows (see Leads).

**A/B.** Live 1M config plus the arm env via launch-conc.sh, one fresh instance per arm, run order C A B C A B; then a pinned-AV round Cp Ap Bp. Suite `scripts/ksuite.sh`: matched C1 coding/prose x8 trials, bench single (1024 tok x 2 per type), conc8, conc16 x2, conc32 x2 (512 tok), exact.py; i5/i6 add GSM8K-50 conc8, the longctx default gate and a 500K needle. Every instance came up on its first launch, with no rdma_ar hang and no engine-init failure. The armed log line was on 8/8 ranks in every armed instance and 0/8 in controls. The new kernels appeared in the Triton cache of ranks 2/4/7 only after the matching arm, which shows both paths really run in the engine.

| inst | arm | AV profile | coding ms/step (AL) | prose ms/step (AL) | single code/prose/math | conc8 | conc16 x2 | conc32 x2 | exact.py |
|---|---|---|---|---|---|---|---|---|---|
| i1 | C | startup | 34.716 (5.17) | 29.539 (2.71) | 84.6/68.1/116.8 | 215.9 | 317.1/317.2 | 454.3/454.0 | a8c3394af9d2 |
| i2 | A | startup | 33.981 (5.08) | 28.764 (2.61) | 85.8/68.7/117.8 | 219.5 | 331.6/330.0 | 472.6/476.5 | a8c3394af9d2 |
| i3 | B | startup | 34.086 (5.14) | 29.038 (2.68) | 86.1/68.8/119.2 | 222.2 | 329.8/332.4 | 488.4/483.3 | a8c3394af9d2 |
| i4 | C | startup | 34.579 (5.24) | 29.414 (2.69) | 85.0/66.3/116.1 | 211.2 | 316.3/318.3 | 452.6/454.2 | a8c3394af9d2 |
| i5 | A | startup | 34.130 (5.19) | 29.139 (2.72) | 86.5/69.1/118.3 | 219.6 | 324.5/329.9 | 481.8/481.0 | a8c3394af9d2 |
| i6 | B | startup | 33.905 (5.16) | 28.681 (2.62) | 86.4/68.8/118.1 | 221.5 | 326.4/327.9 | 484.2/479.0 | a8c3394af9d2 |
| i7 | C | pinned | 34.606 (5.15) | 29.788 (2.64) | 84.6/68.1/116.8 | 212.8 | 312.8/317.4 | 452.2/450.1 | a8c3394af9d2 |
| i8 | A | pinned | 34.161 (5.18) | 29.215 (2.70) | 85.9/68.9/119.0 | 221.5 | 329.0/328.6 | 488.2/485.0 | a8c3394af9d2 |
| i9 | B | pinned | 34.129 (5.10) | 29.121 (2.69) | 86.0/68.9/119.0 | 217.0 | 327.8/326.9 | 485.2/481.2 | a8c3394af9d2 |

C = control (no flags), A = `DSV41_CAND_LENAWARE=1`, B = A + `DSV41_CAND_LENSELECT=1`. "Pinned" means `DSV41_AV_COST_JSON=~/dsv41-stage/k1/1-indexer-mask/av-pin-i1C.json` (i1's raw startup cost profile) on all three arms. Matched C1 values are means of 8 trials (trial sd 0.15-0.46 ms).

Per pair (% better; ms columns = fewer ms/step):

| pair | coding ms | prose ms | single code/prose/math | conc8 | conc16 | conc32 |
|---|---|---|---|---|---|---|
| A vs C r1 / r2 / r3 | +2.12 / +1.30 / +1.29 | +2.62 / +0.93 / +1.92 | all + (+0.85..+4.22) | +1.70 / +4.01 / +4.05 | +4.30 / +3.13 / +4.35 | +4.49 / +6.18 / +7.86 |
| B vs C r1 / r2 / r3 | +1.81 / +1.95 / +1.38 | +1.70 / +2.49 / +2.24 | all + (+1.09..+3.82) | +2.91 / +4.88 / +1.94 | +4.40 / +3.10 / +3.90 | +6.98 / +6.21 / +7.10 |
| B vs A r1 / r2 / r3 | -0.31 / +0.66 / +0.09 | -0.95 / +1.57 / +0.32 | -0.39..+1.20 | +1.19 / +0.84 / -2.03 | +0.10 / -0.03 / -0.44 | +2.39 / +0.03 / -0.70 |

**Means over 3 instances.**

| arm | coding ms | prose ms | single c/p/m | conc8 | conc16 | conc32 |
|---|---|---|---|---|---|---|
| C | 34.634 | 29.580 | 84.8/67.5/116.6 | 213.3 | 316.5 | 452.9 |
| A | 34.091 | 29.039 | 86.1/68.9/118.4 | 220.2 | 328.9 | 480.9 |
| B | 34.040 | 28.947 | 86.2/68.9/118.8 | 220.2 | 328.5 | 483.5 |

A vs C: **coding -0.54 ms/step (-1.57%), prose -0.54 ms (-1.83%), single +1.6% / +2.1% / +1.5%, conc8 +3.2%, conc16 +3.9%, conc32 +6.2%.**

- **Matches the kernel model.** Phase C predicts (a)+(b) saves 0.37 ms/step at 4 rows and 0.56 at 6 rows (C1 coding verifies about 6 rows). Measured A vs C coding savings: 0.74 / 0.45 / 0.45 ms. The 1M width tax is essentially gone at short context:
  - conc16 is 328.9, against 330.9 for the round-4 262K reference and 311.8 at 1M before.
  - conc32 is 480.9, against 479.0 at 262K and 456.9 at 1M before.
  - C1 coding is 34.09 ms, against 33.75 at 262K and 34.4-34.6 at 1M before.
- **The AL dip in rounds 1-2 is AV re-calibration, not the kernels.** exact.py and bench-single trajectories are identical across arms. bench-single AL: code 2.54-2.58, math 3.53-3.58 and prose 1.75-1.82, the same discrete values recurring in every arm. VERIFY_CAP=auto derives tau from the startup cost profile, and that profile is cheaper per verify row once the width-scaled mask is gone. Round 3 pins the same profile on all arms. A then shows equal-or-higher matched AL than C (coding 5.18 vs 5.15, prose 2.70 vs 2.64) and still saves 0.45 / 0.57 ms/step.
- **LENSELECT's increment over LENAWARE is within noise end-to-end** (means: coding -0.05 ms, prose -0.09 ms, conc16 -0.1%, conc32 +0.6%). That matches Phase C (0.08 ms/step at 4-6 rows, about 0.5 ms at 48 rows), which is below the run-to-run spread.

**Correctness.**
- exact.py = **a8c3394af9d2 in all 9 instances**: 3 control, 3 LENAWARE, 3 LENAWARE+LENSELECT, both AV modes. Final live instance: a8c3394af9d2 (i10-LIVE).
- i5 (A): GSM8K-50 conc8 **50/50** (244.8 tok/s); longctx default gate **PASS** (needles 60/60, copy 1.000); 500K needle **PASS** (7/7, copy exact, 497,263 tokens, TTFT 88.3 s).
- i6 (B): GSM8K-50 **50/50** (246.8 tok/s); longctx **PASS** (60/60, copy 1.000); 500K needle **PASS** (7/7, copy exact, TTFT 87.2 s). At 497K context the select's persistent_topk runs its multi-CTA path (62K valid blocks, above 32,768) inside the decode graphs, with no hang.
- Startup: 9/9 instances came up on the first attempt in 383-409 s, the same as before. There were no rdma_ar capture hangs and no engine-init failures in this window.

**Promotion.**
- **LENAWARE: PROMOTED.**
  - Exact: exact.py stays a8c3394af9d2.
  - C1 coding improved in every pair (+2.12 / +1.30 / +1.29%, bar 0.8%), and conc16/conc32 improved by +3.1 to +7.9% in every pair (bar 2%). Nothing got worse.
  - Launcher `~/dsv41-tp8.sh` f4e4e214 -> **cfb97923** on 8/8. It adds one line after the TOPK_CANON export: `export DSV41_CAND_LENAWARE=${DSV41_CAND_LENAWARE:-1}`. Backup: `~/dsv41-tp8.sh.bak-k1-1-indexer-mask` = f4e4e214 on 8/8. Tool: `scripts/promote-launcher.py 1 0`, run on each rank; it refuses unless the launcher is f4e4e214.
  - Site file `candidate_blocks.py` a5816163 on 8/8 (backup 45f1eaa3).
- **LENSELECT: NOT promoted.** It passed every gate it has: exact.py a8c3394af9d2 x3, GSM8K-50 50/50, bench-single AL unchanged, longctx PASS and 500K needle PASS. But its gain over LENAWARE alone is within noise in every pair (coding -0.31 / +0.66 / +0.09%, conc16 +0.10 / -0.03 / -0.44%), so it misses the per-flag bar. It is also greedy-eq, not bitwise, above 16K context. It stays installed and off (default 0). Enable it with `DSV41_CAND_LENSELECT=1` via launch-conc.sh if the 0.5-2.9 ms/step it saves at 48-192 rows (Phase C) is wanted for high-concurrency traffic.
- **Final live relaunch** (11:56-12:03 UTC, `ops/dsv41-launch-all.sh`, launcher cfb97923 defaults, no env; instance `ab/i10-LIVE`):
  - Up in 407 s on the first attempt. Armed line on 8/8: `LENAWARE=1 LENSELECT=0 CTAS=384`. /v1/models max_model_len 1048576.
  - exact.py **a8c3394af9d2**; matched C1 coding **34.20** / prose **29.09** ms/step (AL 5.13 / 2.69); bench single 86.0/68.4/117.6 tok/s; conc8 218.8, conc16 324.6/329.2, conc32 483.9/480.8; GSM8K-50 conc8 **49/50** (item 12, the known flip; 247.2 tok/s); longctx default gate **PASS** (60/60, copy 1.000); 500K needle **7/7** (TTFT 87.6 s); 1M needle **7/7** (999,411 tokens, copy exact, TTFT 205.1 s; stock 1M was 209.4-209.8 s).

**Rollback** (from the Mac; S = ~/dsv41-opt-20261021/ops on spark-one):
- Env only, keeps every file: `ssh spark-one 'cat ~/dsv41-opt-20261021/ops/conc/launch-conc.sh' | bash -s -- DSV41_CAND_LENAWARE=0` runs the stock kernels (the installed file is inert with the flags off).
- Launcher: `for h in one two three four five six seven eight; do ssh spark-$h 'cp -p ~/dsv41-tp8.sh.bak-k1-1-indexer-mask ~/dsv41-tp8.sh'; done` (back to f4e4e214), then relaunch with `ssh spark-one 'cat ~/dsv41-opt-20261021/ops/dsv41-launch-all.sh' | bash` and check the sha is f4e4e214 on 8/8.
- Site file (service stopped first): `scp spark-one:dsv41-opt-20261021/k1/1-indexer-mask/bundle/k1-all.sh . && bash k1-all.sh uninstall` from the Mac (it runs `~/dsv41-stage/k1/1-indexer-mask/uninstall-rank.sh` on 8/8, then checks the sha), or run that per-rank script on each rank. Either restores 45f1eaa3 from `candidate_blocks.py.bak-k1-1-indexer-mask` and refuses on any unexpected sha. Undo the launcher first, then the file.

**Leads found here (not done):**
1. **Decode top-k above 8 rows on 1M-wide logits** (Phase D): persistent_topk costs 2.3 ms/step at 48 rows x 4K context and 6.0 ms at 32K, against 0.8 / 1.8 ms on 262K-wide logits. The live per_row kernel runs the same rows in 31 / 145 us per call. Candidate fixes:
   - (a) Pick per_row over persistent for long rows above 8 rows.
   - (b) Narrow the logits row stride for steps whose max context fits.
   Both are greedy-eq: per_row and persistent can choose different sets on an exact tie at the 512th score. CANON already fixes the order. exact.py is unaffected, because a single request (<= 6 rows) already takes per_row at 1M. This needs a conc A/B at 4K-32K contexts, which the bench prompts do not exercise.
2. **Fused CANON:** sort 512 int32 per row in one kernel instead of torch.sort + where + masked_fill + copy (28 us per call, x8 per step). The output bits are identical, so it is exact. Expected about -0.2 ms/step at 1u (+0.6%).
3. **Pinned AV profile for A/B work** (round 3 above): `DSV41_AV_COST_JSON=~/dsv41-stage/k1/1-indexer-mask/av-pin-i1C.json` holds the verify-cap calibration fixed across instances, which separates kernel time from AV re-calibration in matched C1.

**Housekeeping.** `ops/r4/qwork/journal.txt:25` (also the Mac copy) said the stock candidate mask is "bounded by seq_lens", which is wrong. It is corrected in place: d66b47bd -> 0ded720b, backup `journal.txt.bak-k1-1-indexer-mask`.

**Files** (./k1/1-indexer-mask/): `bundle/` (patched + stock file, install/uninstall/k1-all/build, GPU/CPU tests, MANIFEST, offline/ results), `gpu/` (unit test outputs + csum.py/dsum.py), `ab/<instance>/` (suite outputs, rank0 startup log, env, armed), `scripts/` (ksuite.sh, kdrv.sh, ksumm.py, kab.py, chain1/2.sh, promote-launcher.py, mkpin.py, rungpu.sh), `av-pin-i1C.json` (pinned AV profile, also at ~/dsv41-stage/k1/1-indexer-mask/ on 8/8), `summary.txt`.

**Live state after k1** (2026-10-24 12:03 UTC): launcher `~/dsv41-tp8.sh` **cfb97923** on 8/8 (previous f4e4e214 = `.bak-k1-1-indexer-mask`); `vllm/model_executor/kernels/attention/dsa/candidate_blocks.py` **a5816163** on 8/8 (stock 45f1eaa3 = `.bak-k1-1-indexer-mask`); every other site file unchanged (sparse_attn_indexer.py 7112c01d). Live env adds DSV41_CAND_LENAWARE=1 (launcher default); DSV41_CAND_LENSELECT unset (= 0).
