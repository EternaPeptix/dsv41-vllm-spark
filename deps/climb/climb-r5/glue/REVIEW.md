# r5-glue review (glue:review, climb round 5, 2026-11-09 ~05:45Z)

**Verdict: FIX.** Both levers look exact by construction, flag-off is inert, and the installer is sound. One defect
blocks the kit's own first gate: `tools/compile_check.py` cannot pass as written, and it checks a Triton
specialization that production never runs. The other findings are hardening and documentation.

## What I checked, and what I ran

Read-only on the cluster. The only writes were this file, a scratch copy under my session scratchpad, and a temp
dir on 512S1 that was deleted after the run.

| item | how | result |
|---|---|---|
| Live shas, 8/8 Sparks | `sha256sum` over ssh (read-only) | `dsv41_attnout.py` 308942eb and `dsv41_routeprep.py` 842f49fb on all 8; launcher 166d7aac on all 8; no `dsv41_r5g_*` installed. Equal to `base/` and `live/` and to the r2 kits' shipped modules. |
| Other code reaching the hooked names | grep of site-packages/vllm and `~/dsv41-opt-20261021/ops` on spark-one | Nothing outside the two modules imports `_launch_rq` / `_launch` or names `_r2_reduce_quant` / `_dsv41_rp_kernel`. No sitecustomize or transform anchors touch these files. `o_proj.py` imports dsv41_attnout lazily inside `o_proj()`; `deep_gemm_utils.py` / `deep_gemm_moe.py` import dsv41_routeprep at module import, with no try/except. A bad flag value therefore fails loudly at boot. |
| `tests/test_install.py` | ran on this Mac | ALL PASS |
| `tests/test_ast_exact.py` | ran on this Mac | ALL PASS, 9/9 mutants caught |
| `tests/test_writesets.py` | ran on this Mac | ALL PASS |
| `tests/test_wrappers.py` | ran on 512S1 (exo-venv313, torch CPU), temp dir removed | ALL PASS |
| Installer, independently | install, then uninstall, on a scratch copy of `base/` | +4 tagged lines at EOF per file. The AST prefix is identical to base, plus one `Assign` and one `If`. Uninstall restores 308942eb and 842f49fb and removes both modules. |
| `tools/compile_check.py`, `tools/interp_test.py`, `device_test.py` | **not run** | CLUSTER_LOCK was still present on spark-one at 05:43Z (steering persist/eval, eta 11:32Z), and I have no GPU. |

## Blocking

**B1. compile_check fails as written, and on the wrong specialization.**
- `compile_one()` builds `ASTSource(fn, signature, constexprs)` with no `attrs`. Without `tt.divisibility`, Triton
  cannot prove that `x_ptr` or `t * x_row_stride` is aligned, so it emits scalar `ld.global.b16` loads.
- Evidence: r2-moefront's own compile on spark-eight (Triton 3.7.1, same method) emits **40 `ld.global.b16`** and no
  vector loads (`climb-r2/r2-moefront/ptx/s64_off.ptx`).
- The kit then requires `ld.global…v4.b32` loads before the first `atom.shared` (≥ 5 for hoist, ≥ 1 for split), and
  `LDG.E.128` before the first `ATOMS` in SASS. Both counts will be 0, so every hoist and split variant FAILs and
  `run_spark8_checks.sh` exits non-zero. The gate cannot pass.
- It also means the PTX/SASS "hoist order" evidence comes from a variant that production does not run. In
  production, the pointers and the strides 5120, `M_sum` and `mn_aligned` are 16-divisible and get specialized.
- Fix:
  - Pass `attrs={(i,): [["tt.divisibility", 16]]}` for the pointer args and for the args that are 16-divisible in
    production (`x_row_stride`, `out_row_stride`, `mn_aligned`, `M_sum`). r2-attnout's `scripts/compile_check.py`
    already does this.
  - Compile r2 and the kit kernel with the same attrs.
  - Also add a `M == 1` variant, because Triton specializes `M` for `_r5g_rp_kernel`.
  - Optionally count loads generically (`ld.global(.v[248])?`) so that a width change cannot turn the check into a
    false FAIL.
  - Re-run the 3 Mac tests after the edit. Only KIT.json / TEST-RESULTS change, not the shipped modules.

## Non-blocking (fix before or with the window where cheap)

**N1. Spill check is vacuous.**
- This Triton build reports `n_spills` as `"?"` (see r2-moefront `results/compile.log`: "regs ?, spills ?"), and
  `check(... meta[name][1] in (0, "?"))` always passes.
- README §7.5 says "compile_check requires 0 spills". It does not.
- Fix: grep the SASS for `STL`/`LDL`, or run `cuobjdump -res-usage`.

**N2. The runtime self-check covers almost none of the served M for RPSPLIT.**
- In today's rank0.log (spark-one, 05:40:26), the first fused route-prep call and the first r2-attnout call are both
  eager at **M=2**, and then 45 piecewise graphs are captured.
- With the kit, the eager checks would run at:
  - **RQ:** M = 1, 5, 16, 32 (r2-attnout's synthetic loop, through the rebound `_launch_rq`) plus M = 2.
  - **RP:** M = 2 only.
- The bulk of the served calls are rq at M=4/6 and rp at M=4/6 (code and maths). Those run the kit kernel inside
  graphs without any check on that rank. README risk 3 calls this "theoretical"; it is the normal path.
- Exactness then rests on `device_test.py` RP-B / RQ-B. Those do cover every M, but on one node.
- Recommendation: on the first eager call, run a synthetic sweep in the kit, the same way r2-attnout does: RQ at
  M=1..32 on hostile partials, RP at M=1..10 on synthetic ids and x. That costs about 50 launches once per process
  and makes the 8/8 boot "bit-identical" line mean every served shape.
- Note that the boot gate as written proves RP only at M=2.

**N3. Composition with the r2 integrate tooling is not recorded.**
- `climb-r2/install-all.sh status` hashes `dsv41_attnout.py` and `dsv41_routeprep.py` against the r2 kits' installed
  shas, and its `judge()` marks any other sha as `FOREIGN`. With r5-glue installed, r2 status reads **FOREIGN on
  8/8** and exits 2.
- Any window or promotion gate that runs it will refuse. That fails safe, but it will look like damage.
- r2-attnout's `install` re-copies its module unconditionally. r2-moefront's re-copies it on any sha difference.
  Both drop the hooks, and this is already documented.
- Add the FOREIGN status to KIT.json `conflicts_with` and README §9.
- Also ship an 8/8 runner like `climb-r3/install-all.sh` (per-node install, then a status agreement check). Today
  the relaunch preflight manifest does not cover these files, so a 7/8 install is caught only by the boot-line count.

**N4. `FOREIGN_TAGS = ("# r1-pdl",)` is accepted under files whose kernels the kit re-implements.**
- r1-pdl does not edit these files today. REPORT r4 #13, however, plans "r1-pdl … then rebase onto r2 kernels".
- If that rebase edits `_dsv41_rp_kernel` or `_r2_reduce_quant` (for example a trigger point), the installer would
  accept the base. The kit kernels would then silently not carry the change, and a PDL difference would not show up
  bytewise.
- Fix: refuse foreign-tagged lines inside the two kernel functions, or simply refuse any foreign tag in these two
  files until a rebase exists.

**N5. The hook-on import path is never exercised with real vllm.**
- `compile_check` and `interp_test` stub vllm. `device_test` loads the kit modules by path and wraps manually.
- So the real installed path is never run: lazy import of dsv41_attnout inside `o_proj()`, then the hook, then
  `dsv41_r5g_rq` importing its partially initialised parent; and `deep_gemm_utils` importing dsv41_routeprep, then
  the hook, then `dsv41_r5g_rp`. It should work (all names are defined before the hook, and Python 3.12 resolves
  `from pkg import mod` through sys.modules), but a failure costs a relaunch try.
- Add to device_test: install into an `R5G_VLLM` copy, import it with both flags on, and assert
  `_launch_rq.r5g` / `_launch.r5g`. Then call `fused_permute` end to end once.

**N6. Device-test realism.**
1. RQ-G and the RP x-path have no in-graph producer.
   - RQ-G replays the wrapper alone, with P written outside the graph.
   - RP-G puts `dsv4_topk` in the graph, which is good for ids, but x is produced outside the graph. A mutant that
     moves the hoisted x loads above `gdc_wait` would pass every device case. Only the PTX order check guards it,
     and that check depends on B1.
   - Fix: put a PDL producer in the graph (the sonar partial kernel for RQ, a copy kernel writing x before
     `dsv4_topk` for RP), poison its input region per replay, and add a "load before gdc_wait" mutant.
2. The S1 "serving op" assertions use env that the test sets itself (`ao.RQ is True` is tautological), and the
   profiler pass only lists kernels the test launched.
   - Add a read of the launcher: sha 166d7aac, with `DSV41_ATTNOUT_OP=rope,rq`, `DSV41_MOE_ROUTEPREP=1`, and
     `FC1_FRONT` unset (it is unset today).
3. `N_RQ = 1024` is hard-coded. t2 grid 4×M confirms n = 1024. Assert it from the model config instead.

**N7. Estimates.**
- The arithmetic re-checks against t2. For example, at RQ T_new = 2.5 µs I get code 60 / prose 99 / maths 55 µs per
  step, which is +0.32 % G1. One RP µs per call is +0.19 %.
- Caveats:
  1. **RQ:** t2 shows wo_b launching **1.2–1.8 µs after rq starts** (rq duration + succ-gap at M=1/2/4/5/6:
     1.79/1.37/1.34/1.22/1.37 µs). The on-path floor is therefore max(T_new, that offset + wo_b's pre-wait
     prologue), not T_new. The high end (T_new = 2.1, +0.40–0.45 %) needs a prologue under ~0.5 µs. The central
     value (+0.3 %) is twice round 4's 0.15. That is plausible, but only the window can confirm it; `device_test` T
     times the kernel alone and cannot see this.
  2. **RP:** r2-moefront predicted 2.5–4 µs and measured 8, so per-call savings are uncalibrated. Treat split vs
     hoist as unknown until T is measured.
  3. Promotion needs ≥ +0.3 % in both pairs. w5 control G1 spread was 129.94–130.20 (~0.2 %). RQPAD alone sits at
     the threshold, so the `glue` arm (central 0.65 %) is the one most likely to read cleanly.

**N8. Document inconsistencies.**
- install.py docstring says "+ 3 tagged lines"; there are 4.
- RP mutant counts disagree. KIT.json says "4 per mode", but hoist has 3. The summary says "6 RP mutants", but there
  are 5 definitions and 7 mode-runs. The device_test docstring says "8 mutants x 10 M".
- `run_spark8_checks.sh` does not use the mandatory environment (venv activate, `PATH`, `CUDA_HOME`). It is not a
  device test, but sourcing the environment costs nothing and matches the rule.

## Exactness (reviewed, no defect found)

- **RQPAD.**
  - Every statement before the pad store is the live kernel's (AST-checked; I read both).
  - The pad rows `M+r+j0+jj*M`, masked `< PADDED_M`, for `j0 in range(0, PADDED_M-M-r, 128*M)`, are exactly the live
    loop's rows. There are zero trips when `M+r >= PADDED_M`.
  - The offsets use the same formula. The value is constant 0, and each byte is written once (write-set test, M up
    to 384).
  - Allocations, grid, PDL position (trigger + wait first) and options are unchanged.
  - The graph-captured allocations are the same two `torch.empty` calls.
  - No defect found.
- **RPSPLIT.**
  - The chunk body is the live loop body. The only move is the load issue point: after `gdc_wait`, same address,
    no rounding.
  - x loads now also run for dead routes. They stay in bounds because `t < M`.
  - Split keeps `nprog = num_programs(0)`. The ch-0 program owns `inv_perm`, layout, segment padding, the dead-row
    fill, `ends` and the prefetch. `gdc_launch_dependents` runs in all CTAs, as required.
  - FC1 synchronises on full grid completion, as today. `fused_permute` still rejects x/out aliasing.
  - No defect found.
- **Wrappers.**
  - Fail-closed on mismatch or exception.
  - A capture uses the kit kernel only after a pass.
  - Out-of-scope M (> 10) always runs r2.
  - A precompile failure disables only the kit kernel.
  - Precompile covers only the `M == 1` Triton specialization. The M = 2..10 variant compiles at the first eager
    check, inside try/except. That is the same exposure the r2 kernel has today.

## Gate order I would use

1. Fix B1 (and N1). Re-run the Mac tests.
2. `run_spark8_checks.sh` with the lock absent: compile and interpreter.
3. `device_test.py` exit 0 on a stopped node, with the mandatory environment, ideally after N5/N6.1.
4. Install 8/8. Run the flags-off relaunch: exact.py ec2f4437f8e1ed09, digest 2f7fdccbd388225a.
5. Run the arms. Require both INFO lines 8/8 and no DISABLED line. With N2 added, those lines then cover every
   served M.

## Fixes (glue:fix, 2026-11-09 ~06:30Z)

Everything below was changed under `climb-r5/glue/` only. Nothing was written on the cluster. Reads on spark-eight were
read-only: Triton 3.7.1 sources, the launcher, and the model config. The CLUSTER_LOCK was still present at 06:25:44Z, so
**compile_check, interp_test and device_test are still NOT RUN.** Every test result below is a run from this round
(TEST-RESULTS.txt, results/*.log).

**B1 (blocking): fixed. compile_check now compiles production's specialization.** `tools/compile_check.py` was
rewritten.
- **Production arguments.** It calls the real launch functions with the kernel object replaced by a recorder:
  `dsv41_routeprep._launch`, `dsv41_attnout._launch_rq`, and the kit's `_launch_new` and `_launch_new_into`. The
  arguments are CPU tensors shaped and strided as `fused_permute` and `o_proj` pass them: x `[M,5120]` bf16, topk
  `[M,6]`, slot rows `[M_sum,5120]`, sfp `(M_sum,10)` with strides `(1, mn_aligned)`, the fused path's 32-aligned
  M_sum, and P `[32,1,M,1024]`. So the arguments, constexprs and options are the ones production builds.
- **Specialization.** (signature, constexprs, attrs) come from Triton's own launch binder
  (`runtime.jit.create_function_from_signature` plus the CUDA backend's `parse_attr`, verbatim `_pack_args` logic).
  If that internal API is missing, a rule model is the fallback, and the output says which path ran.
- **Asserted.** tt.divisibility 16 on every pointer and on x_row_stride, out_row_stride, mn_aligned and M_sum. M is
  constexpr 1 at M=1 and plain i32 at M=4. ids_stride1 and inv_stride1 are constexpr 1. The RQ M is never specialized.
  The r2 and kit kernels get the identical spec on every shared argument (plus SPLIT / PAD_J).
- **Variants.** r2, hoist and split × {M=1, M=4 (the same spec as every M 2..10), int64 ids at M=1 and M=4, pdl, pf2,
  nopdl}, plus RQ r2 and kit at M=1 and M=4. That is 25 compiles, and the runner timeout is now 2400 s.
- **Load order, counted generically.** The check counts global-load bytes per thread before the first `atom.shared`
  (PTX) and before the first `ATOMS` (SASS), over every `ld.global` / `LDG` width. It requires the kit to exceed r2 at
  the same variant by at least 80 B (hoist: 5 chunks × 8 bf16) or 16 B (split). Scalar b16 lowering can no longer
  cause a false FAIL. The load widths are printed as INFO, and v4.b32 / LDG.E.128 are expected under the production
  spec.
- **Old bug fixed.** The previous version's launch_dependents check expected the trigger in `*_nopdl`, because the
  name contains "pdl".
- **Logic test.** `tests/test_compile_check_logic.py` runs on 512S1 with torch CPU. It uses a fake `triton` package
  and runs the whole script except the binder and the compiler.
  - The production-spec assertions PASS through the fallback model.
  - Synthetic listings PASS, and so do scalar-b16 listings.
  - The must-fail modes each exit 1 with the expected FAIL line for both hoist and split: `late` (x loads after the
    histogram), `spill` (an STL) and `nodiv` (M_sum not 16-divisible).
  - On the real `climb-r2/r2-moefront/ptx/s64_off.ptx`, the counter finds 4 B (the ids `ld.global.b32`) before the
    histogram and 88 B in total.
  - Result: ALL PASS (21).
- **Not yet verified.** Triton's real binder output, the real PTX/SASS, and so the hoist/split load-order verdict
  itself, are unverified until `run_spark8_checks.sh` runs with the lock absent.

**N1: fixed.** The spill check now requires no `STL`/`LDL` in the SASS and LOCAL 0 in `cuobjdump -res-usage` (STACK
is reported). It also requires the SASS to be non-empty. This applies to every kit variant and to the RQ kit. README
§7.5 is corrected.

**N2: fixed.** Both wrappers now run a synthetic sweep at the first eager call, before the real-data check.
- **RQ:** every sonar M 1..32 on `dsv41_attnout._hostile_partials`, q + every SF byte, 0xFF poison.
- **RP:** every M 1..10 × 2 routings, with the call's ids dtype and all 5 outputs.
  - "spread": distinct experts.
  - "edge": experts 0 and 383 only, with segments longer than one 32-row block.
  - x is hostile: per-group magnitudes 2^-30..2^30, a zero group, and a tiny group.
- **Failure handling.** Any mismatch or exception disables the lever, and the triggering call is served by r2.
  Captures use the kit only after the sweep passed.
- **Boot lines.** They now read "... bit-identical to ... at every M 1..32 / 1..10 (synthetic sweep ...". README and
  KIT.json require exactly these lines 8/8.
- **Tests.** `tests/test_wrappers.py` was updated (counts, INFO lines). New mutants that are wrong only at M=6 are
  caught by the sweep when the first real call is at M=2, and a later capture at M=6 then runs r2. Result: ALL PASS
  (43) on 512S1.
- **Module shas.** rq 9ae74abc68b6c5a9, rp db2dfaf4aceb8967 (KIT.json updated). The kernels and launch functions are
  unchanged, and `test_ast_exact` still passes.

**N3: fixed.**
- New `install-all.sh status | preflight | install | uninstall`.
  - **Probe.** A read-only remote probe returns the shas of the 2 patched and 2 new modules, plus any stale
    `.tmp-r5-glue` files.
  - **Verdicts.** They are computed locally: CLEAN, INSTALLED, HOOKS-LOST, FOREIGN, ABSENT or STALE-TMP. The
    installed shas are not pinned twice: they are produced by running install.py on base copies here.
  - **Agreement.** `status` exits 0 only when 8/8 agree.
  - **Install.** It stages to `~/dsv41-stage/climb-r5/r5-glue` and installs node by node with a probe after each node.
    A failure uninstalls every touched node.
- `tests/test_install_all.py` covers 3 fake nodes: install, idempotence, refusals that write nothing (HOOKS-LOST,
  FOREIGN, stale tmp), rollback after a failure on node 3, and uninstall back to base bytes. Result: ALL PASS (17).
- KIT.json `conflicts_with`, README §9 and the install.py docstring now record two things:
  - climb-r2 `install-all.sh status` reports FOREIGN on 8/8 while r5-glue is installed.
  - The relaunch manifest does not cover these files, so `./install-all.sh status` must run before each r5-glue
    relaunch.

**N4: fixed.** `FOREIGN_TAGS = ()`. A base carrying any foreign-tagged line is refused and nothing is written.
`tests/test_install.py` now checks that refusal for both modules. The composes_with entry is updated.

**N5: fixed, written but not run.** New `tools/hook_import_test.py`, run by `device_test.py --part hook|all` once per
RPSPLIT mode.
- **Setup.** It builds a symlink tree of the venv's `vllm` package, so the venv itself is never written, and runs
  `install.py install` on it. Then, in a fresh interpreter with `PYTHONPATH=<copy>` and the flags on, it checks:
- **H1.** vllm resolves to the copy. `deep_gemm_utils` imports `dsv41_routeprep`, and its `_launch` has `.r5g`.
- **H2.** The lazy import of `dsv41_attnout` resolves the circular kit import, and `_launch_rq` has `.r5g`.
- **H3.** `dsv41_attnout._launch_rq` equals its `__wrapped__` at M=2,1,4,6.
- **H4.** `fused_permute` end to end at M=2,1,4,6 with int32 and int64 ids: wrapped `_launch` equals r2 `_launch`.
  Both sweep INFO lines arrive through vLLM's logger, with no DISABLED line.
- The child script is pyflakes-clean.

**N6: fixed, written but not run.**
1. **In-stream PDL producer.** New `tools/pdl_producer.late_copy` calls `gdc_launch_dependents` at its start, spins,
   and only then writes. Two new device cases use it:
   - RQ-P: [producer of the partials → wrapper] at M 1,2,4,5,6.
   - RP-P: [producer of x → wrapper] at M 1,2,4,6 per mode.
   - Each runs eagerly and as a CUDA graph, with the inputs NaN-poisoned before every run. The kit must equal r2.
   - New PDL-order mutants must be caught, eagerly or in the graph, at every one of those M: RQ "partial loads before
     gdc_wait", and RP "hoisted x loads before gdc_wait" per mode.
2. **S0 reads production's configuration.** It reads `~/dsv41-tp8.sh`: sha 166d7aac, or `--launcher-sha` if that
   changes. It parses the `export VAR=${VAR:-default}` lines, several per line, and was checked against the real
   launcher. It asserts DSV41_ATTNOUT_OP contains rq, MOE_ROUTEPREP=1, FC1_FRONT absent or 0, and sonar fp8/32/32. The
   test's env is then taken from the launcher.
3. **Geometry from the model config.** `N_RQ` comes from `text_config.o_lora_rank`, which is 1024 on spark-eight, read
   read-only. The check asserts o_groups/8 = 1, H 5120, E 384 and top-6.

**N7: fixed.**
- **Estimate script.** `scripts/estimate.py` now reads succ-gap from t2 and adds the floor `max(T_new, off + P)`, where
  `off` = 1.87 / 1.40 / 1.37 / 1.22 / 1.37 µs at M = 1/2/4/5/6. Output is in `scripts/estimate.txt`.
- **RQPAD:** 0.1 / 0.25 / 0.4 %, down from 0.1 / 0.3 / 0.45.
- **RPSPLIT:** the centrals are cut for the uncalibrated r2-moefront history: hoist 0.05 / 0.25 / 0.6, split
  0.1 / 0.35 / 0.8.
- **Kit:** 0.15 / 0.55 / 1.15 %.
- README §6 and KIT.json say that RQPAD alone sits below the +0.3% bar and inside the ~0.2% control noise. They mark
  `glue` as the primary arm.

**N8: fixed.**
- install.py says +4 lines.
- The mutant counts are made consistent in device_test's docstring and KIT.json:
  - RQ: 4 byte mutants plus 1 PDL-order mutant.
  - RP: 5 byte-mutant definitions run as 7 mode-runs (split 2, hoist 1, both 2), plus 1 PDL-order mutant per mode,
    for 9 mode-runs.
- `run_spark8_checks.sh` sources the mandatory environment (venv, PATH, CUDA_HOME) and echoes it into the logs.

**Gate order (unchanged otherwise).**
1. `bash tools/run_spark8_checks.sh` once the lock is gone. This is the first real test of the B1 fix.
2. `device_test.py` exit 0 on a stopped node. This includes S0, RQ-P, RP-P and H, all new and never executed.
3. `./install-all.sh install` with 8/8 agreement.
4. A flags-off relaunch: exact.py ec2f4437f8e1ed09 and digest 2f7fdccbd388225a.
5. The arms. `glue` is primary. Require both sweep INFO lines 8/8 and no DISABLED line.
