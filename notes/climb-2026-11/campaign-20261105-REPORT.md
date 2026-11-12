# g130 kernel campaign — 2026-11-05, agent vibe (one lock, 5 windows)

Lock `vibe-g130resume-20261105T0734Z` (taken 07:33Z, eta 13:33Z). Production launcher
`39e5c200` (official-fp8 + OBLIT) unchanged throughout; every window ended with a
supervisor restore verified HEALTHY, `/health` 200, digest `2f7fdccbd388225a` ref-ok.
Current production G1 on the latest boot: **117.1 tok/s**. Reference digest this campaign:
`2f7fdccbd388225a` (post-migration; the brief's `9a18f802`/`a8c3394af9d2` pins are stale).

## What ran and what it showed

| # | Window | Result |
|---|---|---|
| 0 | Fresh nsys trace of `39e5c200` (rank 0 + rank 4, code/prose/math, 60-step windows) | `~/dsv41-g130prof5/q{0,4}.*`. **rank4 == rank0 within 0.1 ms — no rank skew at G1**. rdma_ar 57–64 µs/AR; new `_proj_rows` (OBLIT) 0.37 ms/step; l2pf split into 2 kernels ~4.0 ms/step; idle 0.78 ms. REPORT: `g130-trace-20261105/REPORT.md` |
| 1 | FC2 barrier `DSV41_FC2_BARRIER_EVERY` 32 vs 1024, CTCT (6 arms) | c-mean 117.174 vs t-mean 117.352 → **+0.15%**, all digests identical. Inconclusive-to-positive; exposure trade-off (fail-stop up to ~22 s late). REPORT: `g130-barrier-20261105/REPORT.md` |
| 2 | **QHOIST** (`DSV41_QHOIST`, owner-designated kit) | Stopped-node GPU harness **PASS** (mode 1 + verify bitwise over M=1..24 eager + 200 replays, hostile rows, negative control OK; −2.05 µs at M=1..4). Serving CTCT: c-mean 117.148 vs t-mean 117.228 → **+0.07% ≈ dead** (bar +0.5% both pairs), but **bitwise exact** (digests + det COLD COMPARE OK 18 rows 0 mismatches). Findings: serving `verify` mode HANGS the boot (reporter thread); kit's WINDOW/install pin `84ee3093` for `dsv41_qhoist.py` is stale vs shipped `34619a17` (SHA256SUMS 18/18 OK). REPORT: `g130-qhoist-window/REPORT.md` |
| 3 | **FC1 G-aware tile deal** (`DSV41_MOE_FC1_DEAL=deal`) | Probe **GO** (prose 4.97 / code 3.88 µs per call vs 3.5/2.5 thresholds; staircase holds). exact PASS (124 cases, 0 failures, G 0..49+, both hosts), sync PASS, race rc=9 (21 reports inside the byte-identical shared `.so`; harness cannot run the `all` arm to compare — asserts). Serving CTTC: c1 117.421 / c2 117.073 vs t1 118.758 / t2 118.722 → **+1.21% G1**, all digests ref-ok, det COLD COMPARE OK. **Promotion still gated on the co-residency stress run (runner not built).** REPORT: `g130-fc1-window/REPORT.md` |
| 4 | attn probe w1 (spark-five, probe-only kit) | VERDICTS: **xv HOLD (rider), diet HOLD, q16 NO, gather DROP, xsplit GO (timing only, 2–4 wk build)**, exact_total_mismatch 0. Attention family does not clear its bars on this build. REPORT: `g130-attn-window/REPORT.md` |

## Campaign totals

- Measured, exact, unpromoted: **FC1 deal +1.21% G1** (needs stress-run build + sign-off),
  **barrier-1024 +0.15%** (needs sign-off), QHOIST +0.07% (dead, exact — bundle rider at best).
- If FC1 deal + barrier both land: ~117.1 × 1.0136 ≈ **118.7 tok/s**. Remaining levers are
  +0.05–1% each and several are now proven dead/HOLD: QHOIST, xv, diet, q16, gather, moe-ar,
  rdma two-shot.
- New evidence vs the brief: no rank skew (r0/r4); OBLIT `_proj_rows` costs 0.37 ms/step
  (~1.6% of wall) since the 11-04 migration; `g1lib` hardcodes the removed `uncensored-fp8`
  model dir (breaks any probe calling `real_layer`; fix = symlink to official-fp8 on the
  probe node); supervisor preflight flakes on transient ssh timeouts (retry passes) and on
  rapid stop/start cycles after a killed boot (space attempts ≥45 s, `--tries 4`).

## Open decisions for the owner

1. FC1 deal promotion: build the co-residency stress runner (spec in
   `g130-fc1-stress/gpu_gate.py`, `executed_on: None`) then full gate + sign-off, or bundle
   route?
2. Barrier 1024: promote to launcher default (`+0.15%`, exact, but fail-stop exposure up to
   ~22 s)?
3. Next window if continuing: g130-l2win (capped gather-window L2 prefetch, +0.2–1.0%
   modelled) — the only remaining untested serving lever with a kit.

## Addendum: l2win window (21:00–21:53 UTC) — DEAD
Bench C ceiling 64 KiB@116 → 118.7 GB/s full 30/30; phases B,W,F PASS on all 8 ranks
(bitwise clean, forced-overlap 0); verdict GO p125_c64/p150* env-only vs *unpaced* live.
Serving CTTC: c1 118.717 / c2 118.628 vs t1(p125 explicit) 118.721 / t2(p150) 118.435 →
**l2win adds nothing over the launcher default** (default already p125_c64; p150 worse).
det COLD COMPARE OK. Nothing promoted; production restored on 6d72bc33. REPORT:
`g130-l2win-window/REPORT.md`. Lock released 21:53Z.

## Campaign final state (2026-11-05 ~22:00 UTC)
Production: launcher **6d72bc33** (FC1 deal default), **G1 118.70 tok/s**, bitwise-exact
digest `2f7fdccbd388225a`, lock free, health 200.
Landed: FC1 tile deal +1.21% (promoted). Measured-exact, unpromoted: FC2 barrier 1024 +0.15%
(one launcher line, exposure trade-off — owner call). Proven dead this campaign: QHOIST
+0.07%, l2win incremental ~0, attn xv/diet HOLD (only xsplit GO, 2–4 wk build), moe-ar,
rdma two-shot. Remaining levers are sub-0.5% each or long builds.

## Addendum: barrier-1024 promoted (22:48–22:56 UTC)
Launcher `6d72bc33` → **`de7fcdee`** (default DSV41_FC2_BARRIER_EVERY 32→1024; bak
`.bak-vibe-bar1024`; rollback env `DSV41_FC2_BARRIER_EVERY=32`). Verified 8/8: **G1
119.072 tok/s**, digest ref-ok. Production now at 119.07 G1 exact, lock free.

## Addendum: final trace re-pricing (2026-11-05 ~23:00 UTC, offline, from prof5 q0.3 prose)

Per-step SUMS from the fresh `39e5c200`+deal capture (wall 23.54 ms/step under nsys):
dense_mxfp8 6.10 (448 launches), fc1_moe 5.10, rdma_ar 4.99 (**145 ar_kernel launches/step,
34.4 µs each**), indexer_attn 2.78, fc2_moe 2.42, mhc 2.28, l2pf+l2pf_paced 3.98,
dense_bf16 1.02, moe_glue 0.63, other_deepgemm 0.52, screen 0.48, other 1.27 (of which
`_proj_rows` 0.37 = OBLIT; rest is ~490 tiny launches < 0.02 ms each).

Conclusion: after the FC1 deal and barrier promotions, dense MXFP8 at M≤6 is latency-bound
(not DRAM-bound), rdma_ar is fixed-count/fixed-order by the model (per-launch 34 µs; the
reduction order must not change), and the small-kernel pool is ~0.9 ms/step spread over
hundreds of sub-0.02 ms launches. XSPLIT attention: GO on timing, 2–4 week build,
write-back unpriced.
