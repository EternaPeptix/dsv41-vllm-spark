
## k1 / 1-indexer-mask: length-aware decode candidate mask + flags -- PROMOTED LENAWARE (2026-10-24 @@PROMO_TIME@@ UTC); LENSELECT @@LENSELECT_VERDICT@@

**Problem.** At live max_model_len 1,048,576 the decode indexer's candidate kernels walked the whole max_model_len-wide logits row on every step, whatever the real context. `_mask_candidates_kernel` wrote -inf over all 1M columns (grid cdiv(W,1024) = 1024 per row), `_candidate_flags_kernel` zeroed all 131,073 flag bytes per row, and the layer-20 select ran torch.topk(2048) over all 131,072 blocks. The fresh 1M profile (Mac scratchpad k1/profile-1m.md) measured the mask at 78 us per call at 4 rows and 1.09-1.46 ms at 48-64 rows, with 4 consumer layers (24/28/32/36) per step. This is most of the conc16/conc32 drop recorded under "Context 1M" (conc16 330.9 -> 311.8).

@@STATIC@@
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
- **D (critic arm: decode top-512 on 1M-wide logits).** @@PHASED@@

**A/B.** Live 1M config plus the arm env via launch-conc.sh, one fresh instance per arm, run order C A B C A B; then a pinned-AV round Cp Ap Bp. Suite `scripts/ksuite.sh`: matched C1 coding/prose x8 trials, bench single (1024 tok x 2 per type), conc8, conc16 x2, conc32 x2 (512 tok), exact.py; i5/i6 add GSM8K-50 conc8, the longctx default gate and a 500K needle. Every instance came up on its first launch, with no rdma_ar hang and no engine-init failure. The armed log line was on 8/8 ranks in every armed instance and 0/8 in controls. The new kernels appeared in the Triton cache of ranks 2/4/7 only after the matching arm, which shows both paths really run in the engine.

@@ABTABLE@@

@@ABNOTES@@

**Correctness.**
@@CORR@@

**Promotion.**
@@PROMO@@

**Rollback** (from the Mac; S = ~/dsv41-opt-20261021/ops on spark-one):
- Env only, keeps every file: `ssh spark-one 'cat ~/dsv41-opt-20261021/ops/conc/launch-conc.sh' | bash -s -- DSV41_CAND_LENAWARE=0@@ROLLBACK_LS@@` runs the stock kernels (the installed file is inert with the flags off).
- Launcher: `for h in one two three four five six seven eight; do ssh spark-$h 'cp -p ~/dsv41-tp8.sh.bak-k1-1-indexer-mask ~/dsv41-tp8.sh'; done` (back to f4e4e214), then relaunch with `ssh spark-one 'cat ~/dsv41-opt-20261021/ops/dsv41-launch-all.sh' | bash` and check the sha is f4e4e214 on 8/8.
- Site file (service stopped first): `scp spark-one:dsv41-opt-20261021/k1/1-indexer-mask/bundle/k1-all.sh . && bash k1-all.sh uninstall` from the Mac (it runs `~/dsv41-stage/k1/1-indexer-mask/uninstall-rank.sh` on 8/8, then checks the sha), or run that per-rank script on each rank. Either restores 45f1eaa3 from `candidate_blocks.py.bak-k1-1-indexer-mask` and refuses on any unexpected sha. Undo the launcher first, then the file.

**Leads found here (not done):**
@@LEADS@@

**Housekeeping.** `ops/r4/qwork/journal.txt:25` (also the Mac copy) said the stock candidate mask is "bounded by seq_lens", which is wrong. It is corrected in place: d66b47bd -> 0ded720b, backup `journal.txt.bak-k1-1-indexer-mask`.

**Files** (./k1/1-indexer-mask/): `bundle/` (patched + stock file, install/uninstall/k1-all/build, GPU/CPU tests, MANIFEST, offline/ results), `gpu/` (unit test outputs + csum.py/dsum.py), `ab/<instance>/` (suite outputs, rank0 startup log, env, armed), `scripts/` (ksuite.sh, kdrv.sh, ksumm.py, kab.py, chain1/2.sh, promote-launcher.py, mkpin.py, rungpu.sh), `av-pin-i1C.json` (pinned AV profile, also at ~/dsv41-stage/k1/1-indexer-mask/ on 8/8), `summary.txt`.
