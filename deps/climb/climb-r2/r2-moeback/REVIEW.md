# r2-moeback: adversarial review (2026-11-08, review-moeback)

Verdict: **install OK (flags off are inert). G25: ship, but it may only be armed after the lead's device gates pass. G18: fix (binary not built).**

Reviewed: install.py f53a620f, KIT.json 4e615ed7, dsv41_moe_fold.py a6c66c1a, PDL kernel 13c85826 and binding 4af817b4
(the diffs against the live Athena sources in ref/ are only the tagged wait and the cudaLaunchKernelEx launch).

## Live bases (checked myself, 14:30Z)
- spark-one sha256: moe_runner.py 20b69fe5…, fc2_status_gated_gather_trial.py 733c3a87…, site/sitecustomize.py 733bdbe0….
  These are the bases KIT.json accepts. spark-two through spark-eight have the same three shas (8/8).
- The pinned files are untouched and match: athena_fc2_forty_layer_trial.py ae208972…, live FC2 .so 85d888ae…,
  deep_gemm_moe.py 9364f121…, fp8_utils.py b35c0561…, shared_experts.py 2f0bca4b…. The launcher is now 08a249da.
- The live engine has `compilation_config.mode = NONE` and `enforce_eager=False` (rank0.log), so
  MoERunner.forward runs as plain Python at enqueue/capture time and Dynamo does not trace it. This matters: under
  torch.compile the `self.__dict__.pop("_dsv41_g25_folded")` decision would be baked into the trace with the fake op
  and the shared output would be added twice. Production does not compile today. **Re-review G25 if
  compilation mode ever changes.**

## Tests I reran (512S1, torch 2.11 CPU, copy of the kit)
- test_install.py: ALL PASS (42).
- test_compose_r1.py: ALL PASS. It needs the real directory layout: R2 = the parent of the kit directory, so a
  stray copy of the kit next to it is counted as "another kit".
- test_plan_g25.py: ALL PASS (30). It depends on `scratch/tree/` (it renders from there), so do not delete scratch/.
- emul_add_exhaustive.py: 4,261,739,522 pairs, A!=B 0, A!=C 0, PASS (11.1 s).
- emul_gather_fold.py: 518,400 elements, OLD!=NEW 0, negative control 88,586 differ, PASS.
- device_test.py --plan: 11/11 OK.
- I rendered the three patched files and diffed them against base myself: only tagged lines are added, plus the
  original `if shared_output is not None:` becomes a tagged `elif`. Rendered shas: moe_runner c8b333b6,
  gather b182b57b, site 376bf156.
- NOT run, because nobody could: ptx_check_g25.py, build_pdl.sh and device_test on a GPU. CLUSTER_LOCK is
  present (13:48Z).

## Flag-off identity
- moe_runner: a module-level bool, three `if False` branches, and an `if False and …` before the original `elif`
  add. Nothing is imported; the device stream, events, allocations and graph are unchanged.
- gather: the flag is defined after the `@triton.jit` kernel, so the kernel's lines and Triton key do not move. The
  `if False` comes before the unchanged grid/launch.
- sitecustomize: the block runs only when `DSV41_FC2_PDL == "1"`.
- Identical when off: **yes.**

## G25 (DSV41_FC2_GATHER_ADD): exactness and races
- Op order: the fp32 fma accumulation is the same source, then the same cvt.rn to bf16, then
  f32(routed) + f32(shared) with one RN to bf16. torch computes `shared + 1*routed` in opmath with one RN,
  then c10 RNE. Addition is commutative, and the exhaustive proof covers fp32-add+cvt and native add.bf16x2.
  routed_scaling_factor is already in the weights, and `_runner_ok` excludes scale != 1, transforms, SP/PCP,
  naive dispatch, pre-reduced output and DBO. A padded width is declined by the exact-shape check.
- Path check: the DSv4.1 target runs `self.experts(...)`, which is MoERunner.forward through the custom op, and the
  op resolves the same MoERunner object, so the flag set and popped is on the same `self`. `forward_unfinalized`
  applies only to TrtLlmMxfp4 (not this backend) and never folds. `_dsv41_sp_ffn` goes through MoERunner.forward.
  The drafter and non-Athena layers never call take().
- Buffers: `output` is a fresh `torch.empty_like` in FusedMoEKernelModularImpl.apply, so the AR input is a fresh
  per-call tensor, as the add's output was. Shared is peeked, not consumed; `.output` is still consumed once.
- Races: one edge is added (the gather waits on `_output_ready_event[idx]`, the same event the runner still
  waits on) and none is removed. Under capture this is the same cross-stream edge, moved from the add to the
  gather. The aux block's lifetime and reuse are no worse than base (its last read is earlier). The stale
  `_dsv41_g25_folded` cannot leak, because every `_apply_quant_method` overwrites it before forward pops it.
- Exact when on: **yes by CPU proof.** The Triton lowering has not been checked on the device.

## G18 (DSV41_FC2_PDL)
- Kernel diff: there is one `griddepcontrol.wait` after `stage_288_fc2_64`. Before it the kernel only reads
  group_layout and constant weights/scales, and the only write is the error `atomicOr(status)`. A and a_scale are
  read, and out is stored, only after the wait. Early-return CTAs write nothing.
- Transitivity:
  - Today SiLU is a plain launch, so FC2 can start only after FC1 has completed.
  - With r1-pdl `silu`, SiLU runs gdc_wait before its trigger, so this still holds.
  - With r2-moefront ROUTEPREP, the layout writer becomes a PDL primary of FC1, and FC1's unconditional grid sync
    keeps this holding.
- No binary is built. There is no SASS diff (control==live or control-vs-PDL) and no device run. The installer
  copies **any** `build/*.so` while `pdl_binary_sha256` is null, and writes the sidecar from that same file, so
  the sitecustomize check passes on an unvetted binary (fail-open on provenance).

## Blocking (G18 only; nothing blocks installing with flags off)
1. Build on spark-eight with no lock. COMPILE-LOG must show live vs control `IDENTICAL`, and control vs PDL may
   differ only by the wait and its scheduling.
2. Record the sha in KIT.json `pdl_binary_sha256`. install.py must **refuse** a build/ .so while that field is
   null; today it installs it unchecked.
3. `device_test.py --part g18` must exit 0 (eager + ≥200 graph replays with the real SiLU trigger and the aliased
   workspace2), plus racecheck/synccheck `--quick`.

## Gates before arming G25 (the lead's window; not defects)
- `device_test.py --part g25` exits 0, and `ptx_check_g25.py` passes.

## Non-blocking
- ptx_check_g25 cannot tell the one known-wrong lowering apart. If shared were folded into the fp32 accumulator,
  it would still pass: cvt.rn.bf16x2 stays at 8 and add.f32/cvt.f32.bf16 are allowed. Require
  cvt.rn.bf16x2.f32 == 16, or 8 + 8 add.bf16x2. The device test does catch it.
- Composition order with r1-pdl:
  - r1-pdl's installer refuses a gather it does not recognise, so r1-pdl must be installed **before**
    r2-moeback.
  - r1-pdl's uninstall restores its `.bak` and silently wipes our gather hook. It fails safe: G25 becomes inert,
    because the runner arms, take() is never called and the add runs.
  - Uninstall LIFO. README/test_compose overstate "each kit's uninstall leaves the other kit's edits": the test
    used an inverse replace, not r1-pdl's real uninstall.
- install() patches the files one by one with no all-bases preflight. A mismatch on a later file leaves a
  partial install; it fails safe (inert), but check every base before writing, as r1-pdl does.
- build_pdl.sh has no hostname guard (spark-eight is stated only in a comment).
- sitecustomize hashes the whole .so in every Python process when G18 is on. This is cheap but runs for each
  worker/subprocess.
- Gains come from trace models only (skeptic numbers). G25 and r1-pdl gather-PDL overlap and do not stack.

## Fixes (fix-moeback, 2026-11-08 14:45Z-15:00Z)

Files after the fixes: install.py e48f8ecc, KIT.json cddc42c0, build_pdl.sh 69a1270d, tools/ptx_rule_g25.py 93ff51ac (new),
tools/ptx_check_g25.py 0476edcd, tools/g18_gate.py 3cbe3fff (new), tests/test_install.py 10640642,
tests/test_compose_r1.py eeabe128, tests/test_ptx_rule.py 2837aa74 (new). Unchanged: dsv41_moe_fold.py a6c66c1a,
device_test.py 7912f7bd, the PDL kernel and binding sources, and the three anchored edits, so the rendered files are
still moe_runner c8b333b6, gather b182b57b and site 376bf156. Live bases were rechecked on spark-one at 14:54Z:
moe_runner 20b69fe5, gather 733c3a87, site 733bdbe0, FC2 .so 85d888ae, trial ae208972, launcher 08a249da.

### G18 blocking 1: build on spark-eight, live vs control SASS IDENTICAL, control vs PDL differs only by the wait
**Not done. The lock was held for the whole session.** `CLUSTER_LOCK` was present at both checks (14:45Z and 14:54Z):
owner=claude, window w1d, token claude-w1d-20261108T143558Z, eta 18:35:59Z. Brief rule 4 forbids compiling during a
window, and I have no authority to use the lead's token. What changed so the build, once someone runs it, is
checked mechanically and not by eye:
- `build_pdl.sh:13-14` adds a host guard: any host other than spark-eight exits 4.
- The script rotates any old COMPILE-LOG, so one log holds one build.
- It dumps the whole fatbin SASS, so nothing depends on how `-fun` matches namespaced names. The address regex now
  accepts 4 or more hex digits.
- It prints the ACQBULK count for each build.
- New `install.py vet-pdl <out>` (`vet_compare` at install.py:291, `vet_pdl` at :336). It refuses, and leaves
  KIT.json unchanged, unless all of these hold:
  - live == control SASS, byte for byte;
  - the control has no `ACQBULK` (the sm_12x SASS for griddepcontrol.wait, as in production's sm_120 quant cubins
    per r1-pdl/REVIEW.md) and the PDL build has at least one;
  - with ACQBULK/NOP removed and branch targets normalised, PDL == control (STRICT), or the opcode multisets are
    equal (SCHEDULE-ONLY, the "its scheduling" case you allowed);
  - in listing order, no global store and no non-OR atomic comes before the first wait;
  - the live .so that COMPILE-LOG hashed is the pinned 85d888ae;
  - the .so on disk is the one the log hashed.
  On success it records `pdl_binary_sha256` and `pdl_vet` (mode, host, counts, log sha) and copies the evidence into
  `build/`. 10 synthetic cases in test_install.py cover the accept and refuse paths.

### G18 blocking 2: install.py copied any build/ .so while pdl_binary_sha256 was null
**Fixed.** `pdl_gate()` (install.py:177) runs inside `preflight()` before anything is written:
- A `build/` .so with a null `pdl_binary_sha256` stops the whole install (install.py:189), and nothing is written.
- A sha that differs from KIT.json also stops the whole install.
- A vetted sha without a passing device stamp for that exact sha means the binary is not installed, and any stale
  installed copy and sidecar are removed. With the flag on, sitecustomize then exits 97. "Passing" means
  `build/DEVICE-G18.json` from tools/g18_gate.py with device_test rc 0, racecheck rc 0, synccheck rc 0,
  reps >= 200 and >= 8 graph cases.
- The sidecar is written only from the vetted sha.

`pdl_binary_sha256` is still null because no binary exists. Tested: test_install.py checks "build/ .so with
pdl_binary_sha256 null: install REFUSED" and "... nothing written", the sha-mismatch refusal, and five
failed-stamp skips.

### G18 blocking 3: device_test.py --part g18 exit 0 on a stopped node (eager + >= 200 graph replays, real SiLU trigger, aliased workspace2) + racecheck/synccheck --quick
**Not run.** No GPU is available to me, there is no binary to test (blocking 1), and the brief assigns device
tests to the lead. It is now a hard install gate rather than a note. `tools/g18_gate.py` runs, in order:
1. `device_test.py --part g18 --reps 200`: 8 graph cases x 200 replays of the real SiLU with the trigger, then the
   PDL FC2 on the aliased workspace2, plus the eager cases;
2. `compute-sanitizer --tool racecheck --error-exitcode 99 ... --quick`;
3. the same with synccheck.

It writes `build/DEVICE-G18.json` and install.py refuses to place the binary without it.

Limitation, stated in README and g18_gate.py: racecheck and synccheck see only intra-kernel shared-memory and
barrier hazards. The cross-kernel PDL ordering is covered by the bitwise graph replays.

**G18 status: partial, fail-closed.** With the flag off it is inert, as before. With the flag on, the process
cannot boot (exit 97) until vet-pdl and the device gate have both passed for one exact sha.

### Non-blocking items
- **ptx_check_g25 could not catch the known-wrong lowering.** Fixed. The rule is now `tools/ptx_rule_g25.py`, used
  by ptx_check_g25.py. Counted per thread over 16 elements, it requires exactly one of:
  - A: 16 `cvt.rn.bf16x2.f32` + 16 fp32 add elements, and no bf16 add;
  - B: 8 `cvt.rn.bf16x2.f32` + 16 bf16 add elements, and no extra fp32 add.
  Any change in an fma.* count also fails. `tests/test_ptx_rule.py` has 16 PASS. It accepts A (add.f32,
  add.rn.f32x2, scalar cvt) and B (bf16x2, scalar bf16). It rejects the fp32-accumulator fold (8 cvt + 16 add.f32),
  the init variant, half roundings, a mixed epilogue, an extra fma, a scalar fma, a stray mul, .ftz, and a
  non-production live census. ptx_check itself has not been run (it compiles; lock).
- **Install order and uninstall with r1-pdl.** Fixed in tests and docs. Correction: r1-pdl's uninstall does NOT
  silently wipe our hook. With our edit on top it refuses: rc 1, "is not this kit's patch of .bak-r1-pdl ...
  Refusing", and every file is untouched, because it checks all files before touching any.
  `tests/test_compose_r1.py` section 5 now runs both REAL installers on a temp tree holding r1-pdl's six bases and
  our three:
  - (A) LIFO round trip: every step is correct, and the end state is byte-identical to base;
  - (B) ours first, then r1-pdl: r1-pdl refuses and writes nothing;
  - (C) r1-pdl uninstalled while ours is on top: refused, nothing changed.
  README states: r1-pdl first, uninstall LIFO, and `r1-pdl status` shows the gather as "TAGGED, NOT OUR EXACT
  PATCH (!)" while both are installed. The scan in section 1 now reads KIT.json `files[].path`, because r1-pdl's
  KIT.json gained a free-text note that mentions sitecustomize.py and tripped the old substring scan. 20 PASS.
- **Partial install on a later base mismatch.** Fixed. `preflight()` (install.py:212) checks every base, applies
  every edit in memory, compiles, and runs the PDL gate before the first write. The test "bad base on the last
  target refuses the whole install" confirms no target, backup, module or binary is written.
- **build_pdl.sh hostname guard.** Fixed (build_pdl.sh:14).
- **test_plan_g25 needs scratch/tree; test_compose_r1 assumes the parent holds only kits.** Kept. scratch/ is part
  of the kit and is not deleted. The scan now reads only KIT.json paths, so a stray non-kit directory is ignored.
- **sitecustomize hashes the .so in every process.** Kept on purpose as the fail-closed provenance check
  (a few ms per process, only with G18 on). Noted in README Risks.
- **Dynamo/torch.compile caveat for G25.** Added to README Risks: re-review G25 if the compilation mode changes.
- **Gains are from trace models; G25 and r1-pdl's gather PDL overlap.** Unchanged and still stated: KIT.json
  `composes_with` and README Risks say the two do not stack.

### Tests rerun (512S1, 14:52Z-14:57Z; outputs in tests/out_*.txt and TEST-RESULTS.txt)
- test_install.py: 67 PASS (was 42).
- test_ptx_rule.py: 16 PASS (new).
- test_compose_r1.py: 20 PASS (was 10).
- test_plan_g25.py: 30 PASS.
- emul_add_exhaustive.py: 4,261,739,522 pairs, 0 differences.
- emul_gather_fold.py: 518,400 elements, 0 differences; the negative control differs on 88,586.
- device_test.py --plan: 11/11 OK.

NOT run: build_pdl.sh, vet-pdl on a real build and ptx_check_g25.py, because of the lock; device_test and
g18_gate, because they need a GPU on a stopped node.
