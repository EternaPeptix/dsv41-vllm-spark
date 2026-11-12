# W-E3 pdl-fc2c REPORT (2026-11-08 02:30-04:53 UTC, vibe)

Kit: `~/dsv41-stage/g130/pdl-fc2c/` (8/8). Work dir: `we3/` (Mac `~/Documents/dsv41-offline/ds41f-adapt-20261107/we3/`).
Serving results: spark-one `~/dsv41-g130ab/we3/` (c1, t1, c2, t2). Anchor era: exact `ec2f4437f8e1ed09`, G1 digest `2f7fdccbd388225a`, launcher `fda6d05a`.

## The change

The pd-kit FC2 PDL early-B prologue (TMA leader defers its grid-dep wait and pre-issues the
first pipeline stages' B + SFB weight loads before the wait), rebased onto the live
fc2c-on-m32all tree (the FC2C compact path, the M>=2 FC2 GEMM = ~99% of G1 FC2 value):
- merged tree v3: manifest `b5879a041793b11a...`, header `253b2bc869b534f2...`, native unchanged `33d4eb19...`; 4-hunk diff vs live (48+/2-), prologue gated `kDsv41EarlyB = MGroupedContiguous + FP4 B K-major + N5120 K384 G384 + kBMainK>0`.
- site branch installed 8/8 (sha `38fb644bae275a5c`, baseline `191162826c` backed up at `.bak-pdlfc2`): `fc2c_tree()` `DSV41_MOE_FC2C_PDL` branch, off by default, behavior-identical when off.
- T arm also sets `DSV41_MOE_PDL_TRIGGER=start` (fp8 SiLU early-signal, already shipped in installed fp8_utils `b35c0561`).

## Exactness evidence (all PASS)

1. SASS (offline, window): non-compact FC2 + FC1 m32 entries SASS-identical to the live tree (648/1568 lines); fc2c-compact designed prologue-only delta (648->728 lines, TMA ops 12->14, regs 168 flat, spills unchanged).
2. GPU unit (window, tokens 1,2,4,6,48 x 7 routes, arms ctl/t0/t_start/t_end): 168/168 whole-`DeepGemmFP4Experts.apply` bitwise (uint8) eager AND 200 CUDA-graph replays per arm/size with 255-poisoned workspaces; gating parity (m=1 d05, m>=2 compact incl. all_dead); merged-tree JIT entry `dsv41_fc2c_d05_v1.fb6f2c85...` compiled once, serving pre-JIT cubin HIT unchanged.
3. Sanitizers: racecheck (fc2c-instantiation filter, positive-controlled) 0 hazards in BOTH cubins; synccheck whole-process 0 errors. The pre-existing production FC1 m32 kernel is the source of all whole-process racecheck noise (10M cap flood, both arms, unrelated).
4. Flag-off serving: exact.py `ec2f4437f8e1ed09` == anchor with the patched site file serving.
5. Flag-on serving (prime): exact == anchor; G1 outputs digest == `2f7fdccbd388225a`.
6. ABAB (below): every arm exact == anchor, G1 digest == expected, 9/9 long prompts ALL==control including both flag-on arms, errors_rank0=0.

## Serving ABAB (pinned, one window, C-T-C-T, bars pre-registered +0.3%/pair)

| arm | G1 run1 | run2 | pair delta | G4 min (2 runs) | G4 mean |
|---|---|---|---|---|---|
| c1 C | 118.937 | 118.918 | - | 54.195 / 53.304 | 71.729 / 70.616 |
| t1 T | 119.001 | 118.917 | **+0.03% vs c1** | 53.503 / 53.941 | 71.306 / 73.287 |
| c2 C | 118.645 | 118.620 | - | 53.774 / 53.550 | 70.053 / 68.416 |
| t2 T | 118.928 | 118.793 | **+0.19% vs c2** | 53.495 / 53.469 | 69.437 / 68.731 |

T mean 118.910 vs C mean 118.780 = **+0.11%**, consistent direction (2/2 pairs positive), but the
pre-registered +0.3%/pair bar was NOT met. Boot-to-boot drift within the window (c1->c2 = -0.25%)
is the same order as the effect. G4: no clear effect. Verdict: **bitwise-exact, rider-grade
positive; not promotion-grade on its own.**

## Current state + restore path

- Production: launcher `fda6d05a`, flags-off serving, api 200, no lock. The site branch is
  installed-inert (off by default, proven identical when off).
- Enabling the arm (owner decision): relaunch with `DSV41_MOE_FC2C_PDL=1 DSV41_MOE_PDL_TRIGGER=start`.
- Removing the branch: `cp sm121_target_m32.py.bak-pdlfc2 sm121_target_m32.py` on 8/8 + relaunch.
- Kit files: `tree/`, `jit-cache/cache/dsv41_fc2c_d05_v1.fb6f2c85...` (prewarmed 8/8), `site/` (the patched site file), `tests/test_pdl_fc2c_unit.py`, `ops/` (cc-check, prep-jit, gpu-unit-window, install-window, install-followup, prime-we3), README.md (full pins + history), tree-hist/ (v1 buggy, v2) for audit.

## Addendum: bundle ABAB we3b (2026-11-08 06:03-06:44Z)

T arm = pdl-fc2c + shared-fold together (`DSV41_MOE_FC2C_PDL=1 DSV41_MOE_PDL_TRIGGER=start
DSV41_SHARED_ACT_QUANT_FOLD=1`), C-T-C-T one window (results `~/dsv41-g130ab/we3b/`):

| arm | G1 run1 | run2 | pair delta |
|---|---|---|---|
| c1 C | 119.184 | 119.039 | - |
| t1 T | 118.674 | 118.612 | **-0.45% vs c1** |
| c2 C | 118.964 | 118.889 | - |
| t2 T | 118.776 | 118.775 | **-0.15% vs c2** |

T mean 118.643 vs C mean 119.027 = **-0.32%, 2/2 pairs negative -> bundle rejected.**
Exactness perfect in all arms (exact == anchor, G1 digest == 2f7fdccbd388225a, 9/9 long
ALL==control, 0 errors). Reading: the shared fold nets <= 0 at serving scale, and with it
on, its Triton kernel carries no PDL trigger, so the fc2c early-B overlap is neutralized on
folded calls. Both flags stay OFF. pdl-fc2c alone (we3 window): +0.11% mean, 2/2 pairs
positive, bitwise-exact — installed-inert, owner's call.
