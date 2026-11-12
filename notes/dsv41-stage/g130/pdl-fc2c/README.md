# g130/pdl-fc2c: pd-pdl FC2 early-B prologue rebased onto the live FC2C tree

Staged on 8/8. Nothing installed, no restart, no GPU job. Trees generated per-node from each
node's OWN live g2 tree (doubles as an input-integrity check — all 8 passed the source sha
asserts and produced identical outputs).

**v2+3 fix (2026-11-08 02:26-02:44 UTC):** the first staged tree (kept at `tree.bad-kisbatch/`)
did not compile — the prologue used names one scope above where the live header declares them.
Two one-line fixes, each mirroring the live main loop's own declaration verbatim in the prologue
block: (a) `constexpr bool kIsBatchedMM = (kGemmType == GemmType::Batched);` and (b) `const auto
tma_b_desc = (is_k_grouped_contiguous(kGemmType) ? gmem_tm_b : &tensor_map_b_base);` (for
the FC2C M-grouped entries both are the compile-time values the main loop uses: false and
`&tensor_map_b_base`). A static name census (`tools/census.py`) now checks every identifier the
prologue uses against declarations available at the insertion point: no remaining misses. The
02:19 and 02:37 UTC SASS windows caught both pre-serving (cc-check A/A arm); an unrelated
window-restore bug (wrong sup-prod.sh path) cost 158 s of extra downtime once, production
restored and verified 02:24:48 UTC (v2 window restored cleanly at 02:30:06 UTC). Tree history on
every node: `tree/` = v3 (current), `tree-hist/v2-kisbatch/`, `tree.bad-kisbatch/` = v1.

## Pins (verify before any use)

| artifact | sha256 |
|---|---|
| merged tree manifest | `b5879a041793b11a82ee2663f66e92de6777171b85ea6da317d629733e3d0e4c` |
| merged header | `253b2bc869b534f21ea006f8934f642c1dec76f54c9be1bb4ec929460b822857` |
| native (unchanged) | `33d4eb19f3dc7c281440afa6df80724c1f9adae0ed6bc8e57f835b9eea83889a` |
| source (live g2 tree header) | `1900579d689bf7b6...` (asserted) |
| tool | `tools/make_fc2c_pdl_tree.py` (9b5c2d43c210451d) |
| census | `tools/census.py` (582f03b6741ff16a) |
| cc-check | `ops/cc-check.sh` (2703c10f3102) |

## Location

`~/dsv41-stage/g130/pdl-fc2c/` (new kit, additive; the live g2 tree is read-only input; the
fc2c env-pinned-manifest rule keeps this out of production until a site patch says otherwise).

## The patch (5 merged subs; all anchors count==1-asserted, CPU-only)

- **M1** the up-front griddep wait: `kDsv41EarlyB = (MGroupedContiguous + FP4 B K-major +
  SHAPE_N 5120 + SHAPE_K 384 + 384 groups + split 1) and (kBMainK > 0)`; only the TMA-leader
  thread defers its wait. **The `kBMainK > 0` term is load-bearing**: the non-compact
  instantiation (kBMainK == 0) must not prologue-load from the tail descriptor, which only the
  compact host path provides.
- **M2** the prologue, placed AFTER the fc2c tail-descriptor fence and BEFORE the pf cursor:
  private scheduler copy walks the first kNumStages stages; per k-block the pre-issue follows
  the compact dispatch (`kb < kBMainK` -> main descriptor, else -> the global-memory TAIL
  descriptor), SFB + B both under `kG1BHint`, `arrive_and_expect_tx(SMEM_TMA_BYTES +
  (kb < kBMainK ? 0 : TMA_B_BYTES))` (compact-aware count); then the leader's wait.
- **M3** `dsv41_pre = kDsv41EarlyB and tma_iter_idx < dsv41_npre` in the main k-block loop.
- **M4a/M4b** SFB skipped and `continue` after the A copy for pre-issued stages (their SFB + B
  + expect already landed pre-wait).

## Verification status

- Textual (done, 8/8): diff vs the live g2 header = 4 hunks (5 subs; M4a/M4b share a hunk by
  context overlap), 48 added / 2 removed lines, 3 `pd-pdl on fc2c` markers; tree history and the
  two one-line scope fixes above; read-only source files other than the header are byte-identical
  (copytree + manifest check built into the tool; 888/888 files hash-verified).
- **NEXT, in order:**
  1. DONE 2026-11-08 (offline nvcc/SASS check, PASS): A/A + FC1 SASS-identical to the live
     serving tree (648 / 1568 lines); fc2c-compact designed prologue delta 648 -> 728 lines,
     TMA ops 12 -> 14, regs 168, spills unchanged. Artifacts `cc/` (16 files).
  2. GPU unit (stopped window) — UNIT PASS 2026-11-08 02:45Z: whole-`DeepGemmFP4Experts.apply`
     A/B, serving construction (pre-JIT cubin HIT) vs the merged tree (fresh JIT entry
     `dsv41_fc2c_d05_v1.fb6f2c85...`), tokens 1,2,4,6,48 x 7 route patterns, arms
     ctl/t0/t_start/t_end: 168 cases 0 failures, bitwise uint8 equal eager AND 200 graph
     replays per arm/size with 255-poisoned workspaces between every replay; gating matches
     serving (m=1 d05, m>=2 compact on all routes incl. all_dead, min_rows 384); call-count
     parity per route. racecheck + synccheck: re-running after the window script fix
     (/usr/local/cuda/bin/compute-sanitizer, rc-gated) — see AGENTS_LOG.
  3. DONE 2026-11-08 03:17-04:11Z (site patch): installed 8/8 (sha 38fb644bae275a5c, one-time
     `.bak-pdlfc2` of baseline 191162826c on every node); branch off by default; launcher has no
     overriding envs (verified). Offline smoke of both arms verified. Flag-off relaunch HEALTHY,
     exact.py combined `ec2f4437f8e1ed09` == anchor (PASS, results/exact-flagoff.json).
     PRIME (one flag-on serving boot, 04:04-04:11Z, `DSV41_MOE_FC2C_PDL=1 DSV41_MOE_PDL_TRIGGER=start`):
     boot HEALTHY, rank0 0 errors; exact.py == anchor; one goal_single G1 119.108 tok/s
     (code 120.64 / prose 86.31 / math 150.38), outputs_digest `2f7fdccbd388225a` == production
     digest, reps identical (results/exact-prime-t.json, g1-prime-t.jsonl). Prewarmed kit
     jit-cache (unit-verified entry `dsv41_fc2c_d05_v1.fb6f2c85...`) distributed 8/8 (base64
     over the ts-ssh shim; raw tar pipes are not binary-safe there).
  4. DONE 2026-11-08 04:49Z (serving ABAB, C-T-C-T one window, harness `we3/pair-we3.sh` +
     `we3/abab-we3.sh`, results spark-one `~/dsv41-g130ab/we3/`): c1 118.937/118.918 —
     t1 119.001/118.917 (+0.03% vs c1) — c2 118.645/118.620 — t2 118.928/118.793
     (+0.19% vs c2); T mean +0.11% vs C mean; 2/2 pairs positive; pre-registered +0.3%
     bar NOT met. EXACTNESS PERFECT in every arm (exact == anchor, G1 digest ==
     2f7fdccbd388225a, 9/9 long ALL==control, 0 errors). Verdict: bitwise-exact,
     rider-grade positive, below promotion bar. Branch stays installed-inert (off);
     owner decides. See REPORT.md.
- The pre-wait-reads exactness argument (verbatim from pd): weights, weight scales and
  grouped_layout were all written before FC1 ran, hence before this kernel's launch;
  activations and every global write stay after the wait; pre-issued copies + expect_tx are
  identical bytes/counts/order to the main loop's for the same stages.
