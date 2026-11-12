# r6-head: adversarial review (head:review, climb round 6, 2026-11-09)

**Verdict: FIX.** The mechanism is sound and flag-off identity holds. One gate gap must be closed before the window:
the config sweeps skip served shape buckets. Everything else is non-blocking.

## What I checked and how

| check | how | result |
|---|---|---|
| Live shas on the cluster | Read-only `sha256sum` over ssh on all 8 Sparks | `dsv41_r4_small.py` 3364d8c8, `model.py` dd46a65e, `dsv41_l2pf_r1.py` 5af558fe, `dsv41_drafter_sched.py` ff95fbad, `dsv41_dtgraph.py` 5afb75d4, launcher 166d7aac on **8/8**. No `CLUSTER_LOCK`. No r6 file present. The kit's `base/` and `live/` copies match these bytes. |
| Installer, flag-off identity | I ran `tests/test_install.py` on the Mac | **44/44 PASS** |
| Installer by hand | install + uninstall on a scratch copy of live | Diff is exactly the 5 module-level lines and the 2 dispatch lines, all tagged. Patched sha c77844fb. Uninstall restores 3364d8c8 byte for byte, removes the module, and leaves no `.bak`. |
| Dequant proof | I ran `tests/test_dequant.py` | **ALL PASS** (65,024 pairs) |
| r5 composition | Diffed `climb-r5/integrate/composed` against `integrate/live` | `model.py` r4 call lines identical. `_wrap_cert` in `dsv41_l2pf_r1.py` identical. No r5 KIT.json, and no r2/r3/r4 kit, lists `dsv41_r4_small.py`. |
| Code review | Read in full: the kit module, the live `_kernels` / `certified_logits` / `target_logits` / SF kernels, `install.py`, `device_test.py`, EXACTNESS.md | See below |
| Not run by me | `test_refine_logic.py` (512S1), `compile_check.py` / `interp_test.py` / `--smoke` (spark-eight), and any GPU or serving test | I reviewed their logs only: `results/*.log` and `results_test_refine_logic.txt` |

## Flag-off identity: holds

- With both master flags unset or in `_OFF`, the patch costs one module-level `any(...)` over two `os.environ.get` calls and one False test per `certified_logits` call.
- `dsv41_head_r6` is never imported. `_OFF` is defined at line 73, before the insertion point.
- The generator variable `_n` does not leak into the module namespace.
- No r5 kit touches the file, so the r5-composed tree gives the same patched bytes (test 44/44).
- Supervisor manifest: `dsv41_r4_small.py` is not in it, and `RDMA_AR_FINGERPRINT` covers only the rdma environment, the rdma module and .so, and the l2 prefetch module. A relaunch preflight is therefore unaffected.

## Exactness when on: argument verified

**`tile` mode.**
- The cand predicate is the live one: same masked loads, same `>=`, and the same `other=-inf` / `other=+inf` choices.
- The case where T is −inf (every row passes, including masked rows j ≥ V) behaves identically.
- `sel & ok` and the `tvalid` filter match live's per-pid tiles and its NEED count.

**`row` mode.**
- The certificate argument is correct, and stronger than the README states: every rank's local (max, id) is identical to live's.
- Reason: the local maximum l\* satisfies hi ≥ l\* ≥ T, so l\* is computed in both modes, and every tied row is too.
- Consequences: `_sf_local_argmax_kernel`, `_sf_onehot_kernel`, `_finish` + argmax, and check mode's local-argmax counter all see the same winner.
- A rank-local self-check fallback is therefore safe even when ranks disagree on the mode.

**Refine.**
- Same `tl.dot(w[BV,BK], xt[BK,BMX], acc)` shape, BK, ascending k0, fp32 accumulate and RN bf16 store.
- An MMA output row depends only on its own A row and on B, so the arbitrary list order and tile-mates do not matter.
- Masked rows load 0 and are not stored.

**Screen DQ.**
- The magic-constant f32(q) is exact, and `(a - c) * sc` is not an FMA-contraction pattern.
- `tl.reshape` keeps the [G32, 32, BN] order, and the masked-lane values (`other=0` / `other=127`) are the same.
- The post-loop text is identical on both sides, so FMA contraction there is identical too.
- The 1..254 exponent-range guard is host-checked once, at an eager call.

**Races.**
- One stream throughout: xnorm zeroes CNT, then the screen, then cand (atomics on CNT/NEED/STATS), then refine (reads CNT/IDX).
- Write sets are disjoint and complete. Buffers are per call.
- PDL (r1-pdl, r5 pdlr2) is per launch (`launch_pdl=True`), not a global Triton hook. The r6 kernels launch without it, so they get full stream serialization.

**Graphs.**
- `target_logits` returns None while capturing, so the head is always eager in production. Graph safety is moot, though tested.
- The self-check and the table check run only eagerly, by construction.

**Self-check reference.**
- `_run(st, x, False, "0")` is a copy of the live launch sequence, not a call to `r4.certified_logits`.
- I checked it line by line: same kernels, arguments, grids, BMX, `FULL=False`, num_warps and num_stages. It is identical today.

## Blocking (fix before the device window)

**B1. The ELIGIBLE sweeps skip served shape buckets.**
- **Screen sweep.** It runs at `swM = [2, 7, 32, 96]`, which covers BM 16, 32 and 128. It never runs BM = 64 (M 33..64), where `num_warps = max(w, 8)`.
- **Refine sweep.** It runs only at M ∈ {2, 7}, i.e. BMX 16. But `grid` and `spread` (and `warps`) from `DSV41_R6_HEAD_REFINE_CFG` apply at every BMX, so a non-default grid or spread is unverified on device for M 17..96.
- **Why it matters.** The README tells the lead to arm hscr / hall with the "device winner" CFG. That would serve shapes the gate never ran. The serving self-check sees only one input per M, so it does not close this.
- **Fix, either of:**
  - (a) Add one M per bucket to both sweeps, e.g. screen `swM = [2, 7, 24, 48, 96]` and refine `[2, 7, 24, 48, 96]` for grid/spread.
  - (b) Make the module ignore non-default grid/spread/warps above BMX 16, and non-default screen CFG outside the swept BM buckets.

## Non-blocking (recommended)

1. **Anchor the device reference to the serving entry points.** Add a short part that compares:
   - (a) with flags off, `r4.certified_logits(st, x)` against `r6._run(st, x, False, "0")["out"]`, bitwise, at each M;
   - (b) with `--source installed` and the flags set in a subprocess, `r4.certified_logits` dispatching to `r6.certified_logits`, plus a self-check pass at M = 2/7/33.

   The GPU gate currently exercises only `_run`. The served path (the 7 installed lines, `certified_logits`, `_done`, stats, fallback) runs only in the interpreter (`interp_test` 143/143).
2. **ELIGIBLE refine output does not match the env format.** The sweep prints `(bk, stages, grid, spread)`, but `DSV41_R6_HEAD_REFINE_CFG` is `BK,warps,stages,grid,spread`, and warps is never swept. Print ready-to-paste env strings, and keep warps at 0.
3. **The screen sweep checks only DQ2 at non-default configs.** DQ1 gives the same `wd` bits (exhaustive proof), so this is safe, but either check both or document "non-default CFG ⇒ DQ2".
4. **Estimates look optimistic, in my read of the t2 traces.**
   - Light case (code and prose): the t2 traces show about 2.8 µs between consecutive head kernels (gap1/gap2/gap3). r6 adds a kernel (cand, about 3–4 µs) in front of a refine chain that is no shorter (about 20 µs, same K chain). So expect −3 to −6 µs per step in code and prose, not neutral.
   - Maths: a central 0.080 ms assumes about 92 % of the 87 µs excess is recovered, while the rows/tiles ratio is unknown and the slowest rank is untraced.
   - My central for HEADREF is about 0.10 % G1, not 0.17 %. The +0.3 % bar for the combined arm is then unlikely without SCREENDQ landing near its high end.
   - The arms decide. The stats line is the right first measurement.
5. **Composition with r1 tooling.** While r6 is installed:
   - `climb-r1/install-all.sh status` (full kit set) reports FOREIGN on `dsv41_r4_small.py`, because c77844fb is neither c46e66bd nor 3364d8c8. Subset runs such as `R1ALL_KITS=r1-pdl` are unaffected.
   - A direct `climb-r1/r1-drafter/install.py uninstall` restores dspark.py and speculator.py first, then refuses on r4_small, leaving r1-drafter half removed.

   Add to README §7: uninstall r6-head before r1-drafter, and expect r1's full status to report FOREIGN.
6. **The self-check runs inside measured arms.** Each new M costs one extra head pass plus several host syncs, during serving on rank-local first calls, for up to 95 M values. That is negligible over a long arm but can bias short C-T-C-T segments. Warm up all common M before measuring, or run speed arms with `DSV41_R6_HEAD_SELFCHECK=0` once hcheck and the device test are clean.
7. **`install.py`.**
   - No `CLUSTER_LOCK` check and no 8/8 wrapper. That is acceptable under the r1/r2 pattern, but r5's install-all gates on the lock; mirror that if a fleet script is added.
   - `uninstall` deletes `dsv41_head_r6.py` without checking its sha.
   - `install` silently overwrites a STALE module.
8. **Document inconsistencies.**
   - KIT.json says "installer 43/43"; the run is 44/44.
   - The issue floor is "39-62 µs" in KIT.json and "47-60 µs" in the README.
   - device_test.py says `time` "never gates", but its forced-set bitwise `check()` calls do gate. That is stricter, which is fine, but say so.
9. **`row` relies on the certificate more than live does.** The README states this correctly: hcheck counts violations online, and `tile` is the fallback.

## Not run, and why

No GPU or serving test of this kit exists. My role is read-only on the cluster, and the cluster is serving.
- The device gate (`device_test.py --part all`, synthetic and `--weights real --rank 4`), every arm and the CERTPRE micro-bench are all still outstanding.
- I did not re-run `test_refine_logic.py`, `compile_check.py`, `interp_test.py` or `--smoke`, which need 512S1 and spark-eight. I reviewed their logs only.

## Fixes (head:fix, 2026-11-09)

Module `dsv41_head_r6.py` 9bd17b03 → **ad775bc1** (the earlier sha is kept as KIT.json `prior_sha256` and in
`tests/fixtures/`). The patched `dsv41_r4_small.py` is unchanged: c77844fb, the same 7 tagged lines.

**B1 (blocking): fixed both ways, (a) and (b).**
- **(a) Sweeps cover every served bucket.**
  - `device_test.py` sweeps at `SWEEP_M = [2, 7, 24, 48, 96]`: BM/BMX 16, 16, 32, 64, 128. This includes BM 64, where
    `num_warps = max(w, 8)`.
  - The screen sweep runs every shape at all five M × 4 cases.
  - The refine sweep runs every (BK, stages, grid, spread) at all five M × 4 cases, in tile and row mode. Grid and
    spread are therefore device-checked at every BMX they act on.
  - The live x per (M, case) is computed once per sweep.
- **(b) The module serves nothing outside what the sweep can run.**
  - `SWEPT_MAX_B = 128`. A non-default screen shape (`screen_cfg`), refine grid (`_grid_n`) and spread
    (`refine_spread`) apply only at BM/BMX ≤ 128. Above that, the live screen shape, grid 4 × SM and spread 1 serve.
  - Refine BK/stages apply at BMX 16 only, as before.
  - Refine warps is not swept, so a non-zero value raises ValueError at import (the CFG format is unchanged; warps is
    always the live XWARPS).
- **New test.** `tests/test_cfg_guard.py` checks these rules: **54/54 PASS** on the Mac (stub torch) and on 512S1
  (torch CPU). It includes 7 must-fail mutants, one per removed guard, and all are caught.

**Non-blocking items:**
1. **Anchor and dispatch on the GPU.** Two new gating parts.
   - `--part anchor`: with flags off, `r4.certified_logits(st, x)` vs `r6._run(st, x, False, "0")["out"]`, bitwise
     (with argmax), at every M in 2..96 × 4 cases.
   - `--part dispatch`: one subprocess per mode (row and tile) with `REFINE=<mode> SCREEN=1 SELFCHECK=1`.
     - For `--source kit`, the venv's live file is patched by `install.py`'s own `patched()` in a temp dir and must
       equal c77844fb. With `--source installed`, the venv file is used as is.
     - Through `r4.certified_logits` it checks: the self-check pass line per new M; output bitwise (tile == live,
       row == where(pred, live, −inf)); `sf_local_pairs` == live; `after_screen` called once; debug filled;
       `full=True` stays live; stats counters plus the stats line. M ∈ {2, 7, 24, 33, 48, 96} × 4 cases.
     - Must-fail: `out-flip` and `hi-flip` corruptions of the r6 pass must be caught by the serving self-check at
       every M, which then serves the live result and turns r6 off.
2. **Ready-to-paste env strings.** The sweeps print `ELIGIBLE-ENV screen: DSV41_R6_HEAD_SCREEN=1
   DSV41_R6_HEAD_SCREEN_DQ=2 DSV41_R6_HEAD_SCREEN_CFG=...` and `ELIGIBLE-ENV refine: DSV41_R6_HEAD_REFINE_CFG=BK,0,stages,grid,spread`.
   Warps is always 0.
3. **DQ.** A non-default `SCREEN_CFG` requires `SCREEN_DQ=2`: it is a ValueError at import, documented in README and
   EXACTNESS, and tested. Timing runs both DQ only on the live shape.
4. **Estimates re-based on t2 with your light-case cost.** Code and prose now lose 3–6 µs/step: cand ≈ 3–4 µs plus a
   ≈ 2.8 µs gap, and the chain is no shorter. The maths central is 45 µs (about half of the 87 µs excess), not 80.

   | | low | central | high |
   |---|---|---|---|
   | HEADREF, % G1 | −0.02 | 0.08 | 0.16 |
   | HEADREF, ms/step | −0.005 | 0.017 | 0.035 |
   | built set, % G1 | −0.04 | 0.17 | 0.53 |

   README §4 states that HEADREF alone is unlikely to clear +0.3 %, and that the combined arm needs SCREENDQ near its
   high end.
5. **r1 composition.** README §7 and KIT.json `composition_notes` say to uninstall r6-head before r1-drafter, and that
   `climb-r1/install-all.sh status` (full set) reports FOREIGN while r6 is installed.
6. **Self-check in measured arms.** hcheck runs with `SELFCHECK=1`. The speed arms (hrow, htile, hscr, hall) use
   `SELFCHECK=0`, and only after the device test exits 0 on both weight sets and hcheck is clean. README §6 says so,
   and so do the KIT.json arms.
7. **install.py.**
   - Module state is now absent / present / STALE-PRIOR (a known earlier kit sha, which install upgrades) / FOREIGN.
   - Install refuses a FOREIGN module with nothing written.
   - Uninstall checks the module sha before deleting it and refuses FOREIGN with nothing changed.
   - Still no CLUSTER_LOCK check or 8/8 wrapper (per-host, the r1/r2 pattern). This is documented in README §7.
   - Tested in `test_install.py`, now 52/52.
8. **Docs.**
   - KIT.json and README carry the same test counts.
   - The issue floor is stated once, everywhere: "at M = 7: live 47–60 µs, r6 39–50 µs (2.4–1.9 GHz) vs DRAM
     383–440 µs".
   - The device_test docstring says the forced-set bitwise `check()` calls inside `time` do gate; only the timings
     and the L2 bench are INFO.

**Tests rerun after the fixes:**

| test | where | result |
|---|---|---|
| `test_install.py` | Mac | 52/52 PASS |
| `test_cfg_guard.py` | Mac and 512S1 | 54/54 PASS |
| `test_dequant.py` | Mac | ALL PASS |
| `test_refine_logic.py` | 512S1 | 216/216 PASS |
| `compile_check.py` | spark-eight, nice 19, lock absent, mandatory env | ALL PASS (module ad775bc1) |
| `interp_test.py` | spark-eight | 143/143 PASS |
| `device_test.py --smoke` | spark-eight | ALL PASS |

- The first post-fix smoke run **failed** in the row dispatch child at M = 17, with "argmax differs" from the
  self-check.
- The cause is interpreter numerics. Its bf16 dot runs on raw bits, so the certificate does not hold, and a row-mode
  argmax can legitimately differ there.
- In smoke mode only, the child now skips a row-mode M whose live pass breaks the certificate, and prints an INFO line
  (M = 17 `flat`). The device run skips nothing. The rerun passed.

**Still not run:**
- Every GPU test: `device_test.py --part all`, synthetic and `--weights real --rank 4`, now including anchor, the
  per-bucket sweeps and dispatch.
- Every serving arm.
- The CERTPRE micro-bench.

The cluster is serving and this role is read-only. The only cluster actions were the read-only CLUSTER_LOCK check on
spark-one and the CPU-only compile, interpreter and smoke runs in `~/climb-scratch/r6-head` on spark-eight at nice 19,
with the lock absent. The scratch Triton cache and results were removed afterwards.
