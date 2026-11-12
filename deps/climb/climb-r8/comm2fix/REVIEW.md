# comm2fix adversarial review (climb round 8, 2026-11-09)

This reviews the repair of r7-comm2's pre-gate in `climb-r8/comm2fix/` (REPAIR.md). The r7 review is kept at
`climb-r7/comm2/REVIEW.md`; this file replaces the copy of it that came along with the kit copy.

**Verdict: fix. The gate repair is sound; two cheap operational items block the dw window.**
- **B1.** `ops/window-r7.sh` still prints `EXPECT=166d7aac` for both the dw and the serving window. Production is
  4f3cc1c6.
- **B2.** The device test does not set or assert `DSV41_GLUE_RQPAD=1 DSV41_GLUE_RPSPLIT=split`. The brief makes that
  mandatory for every device test.

**What holds:**
- No gate was weakened.
- Every mutant is caught for its own reason. MR4 has a criterion gap (N1 below), but the catch itself is genuine.
- The projection record matches the served `_proj_rows` cubins that are on spark-eight today.
- The gate output is real: I reproduced it from scratch.

I ran nothing on a GPU and compiled nothing. I did not touch the cluster: no ssh to any Spark, no lock, no API.

## What I ran myself (512S1, fresh copy `/tmp/r8rev/comm2fix`, empty proof cache)

| check | result |
|---|---|
| Gate files: local vs 512S1 `~/climb-r8/comm2fix` | All 11 `gate/*.py` have identical sha256 (projgate db10bc1f…, sassir 6d7b2b3b…, fpflow 395f89e3…, ordgate 1c779c7a…, run_gate7 416aa329…). The 512S1 `sass/` is byte-identical to `climb-r7/comm2/sass/` (17 dev-*.sass). |
| `gate/run_gate7.py --sass6 <r6 sass>`, fresh copy, no `.projcache` (every proof recomputed) | **RUN_GATE7 PASS: 40/40**. The verdict lines are identical to `gate/results/SUMMARY.txt` once timings are stripped (diff rc 0). Saved in `review/run_gate7-review-512S1.out`. `results/` and `results-r8/` SUMMARYs are also verdict-identical to each other. |
| `tests/run_tests7.py` | CPU_TESTS7 PASS 10/10. In my /tmp copy T8 was SKIPPED (no `../../climb-r5`). In the repair's `~/climb-r8/comm2fix` run, T8 really ran (`only ['climb-r6/comm'] touch them`). |
| `device_test7.py --cpu-selfcheck` | R7_CPU_SELFCHECK PASS |
| My own SASS mutants of one kR7Proj kernel (`…ELi525EE`, `review/mkmut.py`, `review/projgate-review-mutants.out`) | Control: PASS. **Every mutant is caught, each in the right environment:** fallback-only lane-mask swap (`[g1-trip1-div] tree level`), fallback-only SHFL clamp 0x1f→0xf via the MOV (`[g1-trip1-div] tree level`), fast-path-only xor-4/xor-2 swap (`[g1-trip1] tree level`), chunk-3 fold as `FADD.FTZ` (`modifier`), fold as `FADD -RZ, acc` (`operand modifier`), level-2 `BAR.SYNC` → NOP (`no BAR between the STS /*4920*/ and its reader`). |
| fpflow `--mutant` on the **unmutated** dev-r7, scored with each MR regex | Under mutant mode all 18 kR7Proj kernels fail, as in r7. Vacuous catches: MR1 0/18, MR2/MR3 0/18, MR5 0/18, MR8 0/18, but **MR4 /FFMA/ 18/18** (see N1). |
| O3 load set: r7 syntactic slice vs repair's proven set (`review/slicecmp.py`, `…525EE`) | The slice has 43 LDG; the proof has 15. All 15 proven loads are the projection's `LDG.E.128.STRONG.GPU` at /*38a0*/-/*3b10*/. The 28 dropped loads are all AR-phase loads (/*0120*/-/*29b0*/: staging, `STRONG.SYS` receive loads, masked `p.in`), which run before the flag by design. None of the dropped loads is a projection load. |
| Served `_proj_rows` today | The collection made today on spark-eight finds the same 3 Triton cache dirs, with cubin sha256 32d52f76… / 970be971… / 15e8328b… = `ref/triton/*/_proj_rows.cubin`. All have `launch_pdl` False. The 4f3cc1c6 env (RQPAD/RPSPLIT) does not change the served projection cubin. |

## 1. Blocking

### B1. `ops/window-r7.sh` prints `EXPECT=166d7aac` (production is 4f3cc1c6)

**Where.** `window-r7.sh:47` (dw: `WIN=r7-dw EXPECT=166d7aac … device-window.sh`) and `window-r7.sh:61` (serve-cmd: `WIN=r7-w1
EXPECT=166d7aac … climb-window.sh`).

**Why it matters.**
- Both runners pass `$EXPECT` as `--expect-launcher` to the relaunch supervisor, for every arm and for the
  `claude-$WIN-restore` relaunch.
- If the plan is pasted as printed, the most likely outcome is that the supervisor refuses. A stopped window would then
  end without production restored.
- REPAIR.md's next step says "with EXPECT=4f3cc1c6", but the script it points to says otherwise.
- `KIT.json` `live_untouched` / `composes_with` also still name 166d7aac.

**Fix.** Replace both occurrences with 4f3cc1c6, or better, require `EXPECT` from the environment with no default. Update
the two KIT.json strings.

### B2. The device test does not mirror the production glue env

**What the brief requires.** Every device test must mirror the production launcher's env, including
`DSV41_GLUE_RQPAD=1 DSV41_GLUE_RPSPLIT=split`.

**What the kit does.**
- `device_test6.ENV_PREFIX` (reused by device_test7 and `run-device8-r7.sh`) sets only the venv, PATH and CUDA_HOME.
- Neither the prefix nor the preflight sets or asserts the two glue flags. The preflight refuses only `DSV41_GLUE_PDL`
  and `DSV41_OBLIT_FOLD`.

**Risk.** The flags change the `_dsv41_rp` / quant glue, not `_proj_rows` or the AR. The risk to the proj result is low,
but this is a mandatory rule and the dw window is next.

**Fix.** Export both flags in the device prefix. Assert them at device_test7 start, next to the serving-op assertion.

Both items are inherited from r7. The repair did not touch them, but they block the next step it schedules.

## 2. Gate changes, checked one by one (no weakening found)

**G1. BRA.DIV is a two-way branch.**
- BRA.DIV is warp-uniform: taken iff the warp is not converged over the mask. Modelling it as a branch on an opaque
  `SR_WARPDIV` (both paths when unset) widens reachability. The r7 parse made the running path invisible.
- The mask operand is dropped only after its form is checked (`U?R\w+`). Any other form raises.
- In `sass/` and `ref/triton`, BRA.DIV occurs only in dev-r7 (18 = one per kR7Proj kernel). It occurs 0 times in live,
  r3, r6, the mutants' non-proj kernels and `_proj_rows`.
- It does occur widely in the t2 collection (mhc, deep_gemm, topkGating, fusedDeepseekV4QNormRopeKV, per_token_group_quant:
  ~300 sites). acqcheck therefore now sees both paths there too. That only widens A1/A2, and the A1g witness walk
  already explores undecided paths.

**G2. Only control instructions decide an edge.**
- This is correct: a predicated STS cannot redirect control.
- The list (BRA/EXIT/RET/BPT/BRX/JMP/JMX) omits KILL. There are 0 KILL in every SASS here, and treating a predicated KILL
  as a fall-through would only add paths.
- The CFG builder's `uncond` was also fixed for BRA.DIV.

**G3. HFMA2 halves.**
- `(h0 << 16) | h1` is right: `HFMA2 R, -RZ, RZ, 1.875, 0` = 0x3f800000 = 1.0f, and `0, 5.96e-08` = 1.
- The r7 order turned the arrival `+1` into 0x10000. That was a real semantic bug.
- Every r3/r6 case (G0-G5, M1-M7, S7a/b) gives the same verdict in my rerun.

**G4. Four environments, identical records required.**
- This is the main proof limit, not a defect. sassir follows loops on their first trip only, so the gate proves the
  projection for 0 and 1 AR-loop trips, 1 and 3 CTAs, and both shuffle paths, and it checks that the record does not
  depend on which of these ran.
- It does not prove trip counts ≥ 2 (production nvec). The argument that the tail is independent of the loop is sound in
  source (the tail reads only threadIdx, q and memory), and trip-0 = trip-1 equality is strong evidence. Still, the real
  shapes are covered only by device part `proj` (bitwise against the live AR + served Triton, rows 1-6).
- Removing `cb[0x360]=512 / cb[0x370]=1` is fine: `make_env` now sets ntid and nctaid itself.

**G5. Fold idioms.**
- `FFMA(+0, +0, acc)` ≡ `FADD(+0, acc)` bitwise:
  - one RN add of +0;
  - acc = −0 → +0 in both;
  - NaN, Inf and subnormals propagate alike;
  - neither flushes without `.FTZ`.
- The served record has the same (Z, Z) terms: the ref summary's last term is `(('Z',),('Z',))`, with 3072 masked terms
  in row 0. The equivalence therefore holds on both sides.
- The check refuses exactly the cases that would break bitwise equality. I confirmed both: `.FTZ`, and `-RZ`, which
  would give −0 for acc = −0. An FADD with two zero operands (len(z) ≠ 1) also fails. If nvcc ever dropped the add
  entirely, the term would be missing and the record would differ.
- `MOV Rd, RZ` as a zero arm requires an unmodified RZ.

**G6. CFG-based slot writer and BAR check.**
- The last writer must be unique per environment, and the reader must be unreachable from it without crossing a
  `BAR.SYNC`.
- `see()` follows a MOV only when it is unpredicated and unmodified, and follows a phi only by the environment's pick.
- The `nobar` mutant proves that the BAR check can fail.

**G7. fpflow exemption = exactly what was proven.**
- Without a proof, or with a failed one, the kernel fails.
- Every q.y store found by address decode must be one the proof executed.
- Proven partial-sum STS skip `check_store`.
- FP ops count as exempt only if they are in `covered` and in no C1 chain.
- The exemption is narrower than "the y slice + everything" and wider than r7's (broken) slice, and that is the
  intended set.
- A residual hole: a q.y store whose address does not decode in env0 is not matched. It then goes through the normal
  store checks and fails as an unproven FP store, so it is not silent.

**G8. O3 proven loads.**
- See the table above. The narrowing removes only AR-phase loads.
- MO3 (no acquire) is still caught 12/12 for /^O3/.
- It fails closed with no proof, or when a "load" is not LDG/LD.

**Proof cache.**
- Keyed by sha256(name, SASS text, projgate+sassir+fpflow source).
- ordgate's use of `loads` is computed in projgate, so ordgate need not be in the key.
- My rerun with no cache reproduced every verdict, so no stale proof hides anything.

## 3. Non-blocking

**N1. MR4 (m_ffma) is vacuous by criterion on the 18 kR7Proj kernels.**
- `--mutant` keeps r7's projection handling, so the unmutated kR7Proj kernels already fail with "stray FFMA …", which
  matches /FFMA/.
- The mutant is in fact caught genuinely: all 18 also carry "FFMA at … in the reduction chain (only FADD allowed)". But
  the gate cannot tell the two apart. This is the same class of defect the repair fixed for MP1-MP5.
- MR1/2/3/5/8 are not affected: 0/18 vacuous for their regexes.
- **Fix.** Load `PROJ_PROOFS` in mutant mode too, so the base kR7Proj kernels pass. Or tighten MR4/M4's regex to
  `FFMA at .* in the reduction chain`. Then add a "base passes in mutant mode" control.

**N2. The local copy has no `sass/`.**
- The gate ran on 512S1's `~/climb-r8/comm2fix/sass`, which equals `climb-r7/comm2/sass`.
- `window-r7.sh gate` rsyncs `sass` from the kit dir under `set -euo pipefail`. Run on its own from comm2fix, before a
  build creates `sass/`, it aborts.
- In `pre` and `dw`, `dev8-r7.sh` / `build8-r7.sh` run `mkdir -p sass` first, so the window path works. 512S1 keeps the
  dev-*.sass because rsync does not delete.
- **Fix.** Copy or symlink `climb-r7/comm2/sass` into the copy, or note it in REPAIR.md.

**N3. "LOCAL:0, so no spill" is imprecise.**
- The kR7Proj kernels have STACK 24/48 and 6 STL (printf argument frames before `CALL.ABS`), so local memory is not
  empty.
- Spills would also live in the stack frame. The conclusion still holds, because projgate's chain resolution refuses any
  LDL/STL in the projection dataflow.

**N4. N1/N2 (acqcheck).**
- I agree with FAIL/OPEN for `fusedDeepseekV4QNormRopeKV` (EXITs before the wait, witness undecided) and for
  `act_and_mul` (no griddepcontrol in any arch, 11/15480 overlaps, launch site unread).
- The collection log's own counters say 388 probes, 355 steps, 342 dumps and no lock. I could not verify the probes
  independently, because that would need cluster history.
- The C2 pair stays blocked and `trig` stays forbidden. C1 and HT13 do not depend on them.

**N5. Residual exactness scope.**
- Static equality is shown for rows = 6 (the record's env) and the 4 CONFIGS.
- rows 1-5, real nvec, grid sizes and 8-rank data are for device part `proj`. Do not run the q/p pair before that passes.

## 4. Before the dw window

1. B1: EXPECT=4f3cc1c6 in `window-r7.sh` (both places) and KIT.json.
2. B2: export and assert `DSV41_GLUE_RQPAD=1 DSV41_GLUE_RPSPLIT=split` in the device prefix and preflight.
3. Optional: N1's MR4 regex or mutant-mode proofs; N2's `sass/` link.

After that, the plan in REPAIR.md §5 stands as written.

## Fixes (r8 fix pass, 2026-11-09)

Every fix below was made in this copy only; climb-r7 was not edited. I ran nothing on a GPU, compiled nothing and did
not touch any Spark: no ssh to a Spark, no lock, no API. No source or device code changed, so the `dev-*.sass` gated
here is still the current source's and no dev-TU recompile was needed. All runs were on 512S1. Logs are in `build-logs/`.

### B1 (fixed): EXPECT is required and has no default
- `ops/window-r7.sh` `plan` / `serve-cmd` call `need_expect` (lines 16-24, 52, 72). It requires `EXPECT` from the
  environment as exactly 8 hex characters, refuses the stale launchers 166d7aac / 08a249da / fda6d05a, and prints
  `EXPECT=$EXPECT` into both the dw line (device-window.sh) and the serving line (climb-window.sh). Usage:
  `EXPECT=4f3cc1c6 bash ops/window-r7.sh plan`.
- What I ran (`build-logs/window-r7-expect-r8fix.txt`):
  - unset, malformed (`4f3cc1c6x`) and the three stale shas each give rc 2 for both `plan` and `serve-cmd`;
  - `EXPECT=4f3cc1c6` prints `EXPECT=4f3cc1c6` on both runner lines, with 0 occurrences of 166d7aac in the plan.
- `KIT.json`: `live_untouched` now names `$HOME/dsv41-tp8.sh (4f3cc1c6)`. `composes_with` names live 4f3cc1c6 and says
  how it differs from 166d7aac.
- Comments in `shapes.py` and `device_test6.py` that derive bounds from 166d7aac now say that the bounds are unchanged
  in 4f3cc1c6. I checked this myself: a diff of the two launcher copies (climb-r6/integrate/live = 166d7aac,
  climb-r8/integrate/live = 4f3cc1c6) shows exactly two added lines, the `DSV41_GLUE_RPSPLIT` and `DSV41_GLUE_RQPAD`
  exports. No RDMA_AR_* line differs.
- **Outside this kit (not fixed, write scope):** `climb-common/device-window.sh` still has `EXPECT=${EXPECT:-08a249da}`
  and `climb-window.sh` has `EXPECT=${EXPECT:-fda6d05a}`. Anyone who calls those runners without EXPECT gets a stale
  launcher. The printed r7 plan always passes EXPECT explicitly. The runners' defaults should become `${EXPECT:?}`.

### B2 (fixed): the device test mirrors the production glue env and asserts it
- `device_test6.py`:
  - `PROD_GLUE_ENV = (DSV41_GLUE_RQPAD=1, DSV41_GLUE_RPSPLIT=split)`.
  - `ENV_PREFIX` now exports both flags.
  - `check_glue_env(env, launcher_txt)` (line 409) refuses in each of these cases:
    - either flag is missing from the environment or differs from the production value;
    - `~/dsv41-tp8.sh` cannot be read;
    - the launcher's `${VAR:-default}` for either flag differs, which means the kit's production env is stale.
  - `preflight()` appends those problems. The r6 path and the r7 path (which calls `D.preflight`) therefore both refuse
    before any GPU work.
- `device_test7.py`:
  - asserts the env again right after the serving-op assertion (line 537), and prints `GLUE-ENV OK: ...` or
    `GLUE-ENV FAIL` + `R7_DEVICE_TEST REFUSED`;
  - `--cpu-selfcheck` gained `glue_env_selfcheck`. Its negative controls prove the check can fail. It also checks that
    `run-device8-r7.sh`'s ENVP equals `D.ENV_PREFIX`.
- `ops/run-device8-r7.sh`: ENVP exports both flags.
- `tests/importcheck7.py` takes an optional 6th argument (the launcher). It now also prints the launcher sha and the
  two glue values, so the composition check can run on the 4f3cc1c6 env.
- What I ran (512S1):

  | run | result |
  |---|---|
  | `device_test7.py --cpu-selfcheck` | R7_CPU_SELFCHECK PASS. The production env with the 4f3cc1c6 launcher is accepted. These are refused: no flags (the old r7 prefix), RQPAD only, RPSPLIT=fused, RQPAD=0, the 166d7aac launcher, an unreadable launcher. ENVP == ENV_PREFIX. |
  | `check_glue_env` on the two real launcher files (`build-logs/glue-env-real-launchers-r8fix.txt`) | Accepted only for 4f3cc1c6 + ENV_PREFIX. Refused for 4f3cc1c6 + bare env, 166d7aac + prefix, 166d7aac + bare. |
  | `device_test6.py --cpu-selfcheck` | CPU_SELFCHECK PASS |
  | `tests/mock_device7.py` | MOCK_DEVICE7 PASS (harness smoke only, no kernel) |
  | `tests/importcheck7.py` | PASS 33/33 with the 166d7aac launcher (92 vars) and 33/33 with the 4f3cc1c6 launcher (94 vars) |

- **Limits.**
  - The GPU-side assertion has never run on a node; it runs in dw.
  - The prefix mirrors the two flags the brief names. The launcher's other exports still come from the process
    environment, as before. The diff above shows that the two flags are the only launcher change since 166d7aac.

### N1 (fixed): MR4 reason tightened, plus vacuity controls
- `gate/run_gate7.py`: MR4's reason is now `FFMA at .* in the reduction chain`. The V1..V6 cases run fpflow `--mutant`
  on the **unmutated** dev-r7 with each MR1-5/MR8 reason. Each must report `caught 0/N -> SURVIVED` with rc 1, so a
  reason that matches the base kernels fails the gate. `ops/window-r7.sh pre` now also requires V1-V6 OK.
- `RUN_GATE7 PASS: 46/46` (full run, no `--skip-missing`, `build-logs/gate7-r8fix-SUMMARY.txt`). Against the
  pre-fix SUMMARY, timings stripped, the only differences are MR4's reason text and the six V lines:
  - MR4 caught 60/60 under the tight reason.
  - V1-V6 gave 0/60 ×5 and 0/54.
- Negative control: the old /FFMA/ reason on the unmutated dev-r7 gives `caught 18/60 -> SURVIVED`
  (`build-logs/V4-negative-control-old-regex-r8fix.txt`). V4 rejects that, because it needs `caught 0/`.
- The M4 r6 case keeps /FFMA/. r6 SASS has no kR7Proj kernels, so the vacuity source does not exist there.

### N2 (fixed): `sass/` is in the copy
- `sass/` is now a clone of `climb-r7/comm2/sass` (17 dev-*.sass + resusage, made with `cp -c`). `window-r7.sh gate`
  no longer depends on a build having created the directory.
- The gate run above used this copy, rsynced to 512S1.

### N3 (fixed in REPAIR.md)
- The "LOCAL:0 so no spill" sentence is corrected. My own count in `sass/dev-r7.sass` for the 18 kR7Proj kernels:
  - STACK 24 or 48 bytes;
  - 6-12 STL each (printf frames);
  - 0 LDL.
- The conclusion rests on projgate refusing LDL/STL in the projection dataflow.

### N4, N5: unchanged
- N1/N2 acqcheck stays FAIL/OPEN. The C2 pair stays blocked and `trig` stays forbidden.
- The device part `proj` (rows 1-5, real nvec, 8-rank data) must pass before the q/p pair.

**Status:** B1 and B2 are fixed and every gate above was rerun and passes. The next step is the dw window per REPAIR.md
§5, launched as `EXPECT=4f3cc1c6 bash ops/window-r7.sh plan`, and only while nothing holds CLUSTER_LOCK.
