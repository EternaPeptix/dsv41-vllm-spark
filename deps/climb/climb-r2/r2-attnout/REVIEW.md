# r2-attnout review (review-attnout, 2026-11-08)

**Verdict: FIX.** Don't install for the rq arms as shipped. Fixing it is a 3-line change plus a device test.

## Blocking

1. **`_r2_reduce_quant` (rq) does not compile on the production Triton (3.7.1, spark-eight venv).**
   - The kit's `scripts/compile_check.py`, which the builder never ran, fails at `dsv41_attnout.py:262`: `NameError("Cannot access global variable _ASM_UE8M0 from within @jit'ed function ...")`.
   - Cause: `_ASM_UE8M0`, `_ASM_INV_SCALE` and `_ASM_FP8` (lines 92-140) are plain module-level `str` globals. Triton 3.x only lets jit code read `tl.constexpr` globals.
   - Effect in serving: the first eager call compiles rq inside `_self_check_synthetic`. Its `except` then sets `_ok["partial"] = _ok["rq"] = False` and logs ERROR.
     - Every arm that contains rq (op-rope-rq, op-rope-rq-tune, op-rope-rq-tune128) silently runs the live path, so rope and the tune parts are disabled too. These arms are window time spent measuring the control.
     - The real-call `_launch_rq` (line 543) is not inside a try. If rq were ever first reached outside the synthetic check (synthetic already done, budget left), the worker would crash.
     - `device_test.py --variants all` dies with a traceback at the second variant (`rq`).
   - **Fix (verified):** wrap the three strings in `tl.constexpr("""...""")`. With exactly that change, `compile_check.py` gives **COMPILE CHECK PASS**. Scratch copy: `spark-eight:~/climb-scratch/review-attnout-fixA/out/summary.json`, sha256 637b883f1730b222…
   - Bump `kit_sha256` in KIT.json after the fix. Then run `device_test.py` on a stopped node before any rq arm. rq has never executed anywhere.

2. **`scripts/interp_check.py` cannot run.**
   - `_launch_partial` calls `_sonar()` unconditionally (line 424). That imports `vllm.models.deepseek_v4.nvidia.ops.sonar_bf16_bmm`, which the stubbed `vllm` package cannot provide, so the script fails with `ModuleNotFoundError: No module named 'vllm.models'`.
   - Fix: register the loaded base sonar module under that name in `sys.modules`.
   - With that harness fix I ran it: **INTERP CHECK PASS**, 12/12 partial buffers bit-identical to the live kernels (M 1/5/17 × rope, vs, bn128 and rope,vs,bn128). Runtime 27 s.

TEST-RESULTS.txt says both scripts were "written and syntax-checked". Neither one could run as shipped.

## What I verified (all run by me)

- **Live bases:** on all 8 Sparks (15:2x UTC), o_proj.py = 34b9e1a4 and sonar_bf16_bmm.py = 9f6f3f84. These equal `base/` and KIT.json. Kit module sha = f70a1a0a… Nothing is installed yet.
- **Tests rerun, all PASS:**
  - `tests/test_install.py` and `tests/test_flag_off.py` (Mac).
  - `tests/test_plan.py` and `tests/test_emulation.py` (512S1, torch 2.11).
  - These are real, but test_plan stubs Triton with recorders, so it could not catch blocker 1.
- **compile_check (fixed copy, sm_121):**
  - Every partial variant (rope, st1, vs, bn128, rope_st1_vs, rope_st1_vs_bn128) uses the same `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` as live, mma v2 and kWidth 4. The BN=64 variants have the same dot encoding as live.
  - Rope PTX contains `mul.rn.f32` + `fma.rn.f32` and 0 `ftz`.
  - rq PTX:
    - Two `add.rn.f32x2` chains of 15 adds, packing lane0 = splits 0,2,…,30 and lane1 = 1,3,…,31, then one `ev+od` add per element.
    - `cvt.rn.bf16.f32`, then a single `mul.rn.f32` (amax·fp32(1/448)), the verbatim asm and `cvt.rn.satfinite.e4m3x2`.
    - `griddepcontrol.launch_dependents` then `griddepcontrol.wait`, before the first `ld.global`.
- **Live op order confirmed from the live cache (ref/):**
  - Inverse RoPE PTX: `mul.f32x2` of partner × sin, then `fma.rn.f32x2(x, cos, ps)`, then `neg` and `fma(x, cos, -ps)`, then `cvt.rn.bf16`, then `selp.b16`. This matches the kit.
  - The live reduce is a sequential in-thread chain (first add = elem0 + elem1, no zero seed), then one cross-warp `shfl.bfly 1` add. That add is commutative, so it equals ev+od.
- **Flashinfer quant:** the CuTe-DSL swizzled kernel matches the kit:
  - `global_max * Float32(1/448)`, `float_to_ue8m0_fast`, `ue8m0_to_inv_scale_fast`, then per-lane `bfloat2_to_fp8x2_scaled`.
  - `compute_sf_index_swizzled_128x4_gpu` gives the same formula as the kit.
  - Output shapes match: q [m, K] fp8, sf 1-D uint8 of padded_m·padded_cols.
  - `apply_weights` passes `qa.data` and `qa.scale` unchanged to `mm_mxfp8`, so the GEMM inputs are byte-identical, including through the DTAC pin.
- **PDL race (rq triggers dependents at entry):** safe.
  - The wo_b GEMM `mxfp8_gemm_cutlass_sm120.so` (built for sm_121a with `-DCUTLASS_ENABLE_GDC_FOR_SM100=1`; template `gemm.run(..., enablePDL=true)`) has 24 `ACQBULK` (griddepcontrol.wait) across 12 functions, and 0 PREEXIT.
  - rq itself waits before any read. Its predecessor (the partial) is non-PDL.
- **Production shapes** (rank0 log): sonar fp8, M=[1,32], G=1, N=1024, K=4096, splits=32, so rq is eligible. `torch.compile` mode is NONE; o_proj runs in a captured segment after the eager MLA break.
- **Flag-off:** the diff is 1 module flag line plus 1 gated `if` in o_proj.py, and the `r2ao=None` parameter plus 1 `if r2ao is not None` in sonar.py.
  - The L2-prefetch `try_project` wrapper forwards `*a`, and no live caller passes r2ao.
  - The kit module is not imported. Kernels, launch order, allocations and graph contents are unchanged.
  - No runtime transform or preflight manifest covers these two files.
- **Composition:** no other r2 kit touches these files, and r1-attn touches attention.py only (`_wo_b_proj` is unchanged there). r1-pdl copies its kernels into its own module, so the kit's verbatim launch of `sonar._sm120_wo_a_partial` stays valid.

## Non-blocking

- **Unchecked eager runs:** `_run` lets an eager call with `_checks_left == 0` and `_ok[part] is None` run that part unchecked. Treat that case like capture (live fallback). Also wrap the real-call self-check launches in try/except so they fail closed.
- **Uninstall order:** uninstall restores `.bak-r2-attnout` blindly. If r1-pdl was installed after r2 it would silently drop r1-pdl's edits; r1-pdl's own base-sha check should refuse that order anyway. Document "install r1-pdl first, uninstall r2 first", or check that `sha(current) == sha(patched(bak))` before restoring.
- **Unrecognised flag values:** o_proj's flag treats any value outside {"", "0", "off"} as on, and the module then raises ValueError at the first o_proj call (e.g. `DSV41_ATTNOUT_OP=false`). That fails loudly at boot. Acceptable, but say so in the README.
- **FTZ corner:** the CuTe-DSL `amax*inv448` (amax < 2^-117) case stays unresolved offline. device_test reports it.
- **r1-pdl interplay:** with r1-pdl's flag on, the r2 path bypasses r1-pdl's PDL partial and reduce. Pick the window controls with that in mind.
- **Gain:** the gain estimates are unmeasured. The tune parts may give nothing.

## Per arm after the fix

- **op-rope:** works as shipped (rq is never compiled). Compile, interpreter and emulation checks pass. It still needs `device_test.py --variants rope` on a stopped node.
- **rq-containing arms:** need the constexpr fix, then a full `device_test.py` with exit 0.

## Fixes (fix-attnout, 2026-11-08 15:2x-15:3x UTC)

Module sha256 is now **308942eb**bed1fc121f74ea5b9820a941b7fed895edf7aae4d6d63480ba1d1bfc. It was f70a1a0a.
- Lines 1-483 (every kernel, the asm constants, `INV448_BITS` and the `_launch_*` helpers) are byte-identical to the reviewer's fixA copy (2402341f).
- The only other change is host logic in `_run`.

### Blocking 1: rq did not compile (non-constexpr asm globals). FIXED.

- `_ASM_UE8M0`, `_ASM_INV_SCALE` and `_ASM_FP8` (dsv41_attnout.py:92, 113, 129) are now `tl.constexpr("""...""")`. This is exactly the reviewer's change; `diff` against fixA prints nothing for the kernel section.
- The reviewer ran compile_check on that source (COMPILE CHECK PASS, Triton 3.7.1, sm_121). Their output is copied into the kit at `ref/compile-review-fixA/`: summary.json (sha 637b883f), reduce_quant.ptx, rope.ptx, live_partial.ptx and the logs.
- KIT.json `kit_sha256` is updated to 308942eb, with the history recorded.
- **New CPU guard:** `tests/test_jit_globals.py` is an AST lint. Every module global read inside an `@triton.jit` function must be `tl.constexpr(...)`.
  - It passes on the fixed module.
  - On the reviewed f70a1a0a module it reports all three `_ASM_*` names, so this class of bug is now caught without Triton.
  - `tests/test_plan.py`'s Triton stub gained `tl.constexpr`.
- **Not done by me:** I did not rerun compile_check on the final kit. CLUSTER_LOCK was held by window w2 (15:21:47Z, eta 19:21:47Z) for the whole fix session. It is gate 1 in README §9 and KIT.json `device_checks` / `window_arms[].requires`.

### Blocking 2: device_test.py exit 0 before any rq arm. Made a HARD GATE. It cannot be run by the builder, who has no GPU.

- README §9 and KIT.json `window_arms[].requires` now state:
  - no arm runs before compile_check and interp_check pass on this kit;
  - `device_test.py --variants rope` must exit 0 before op-rope;
  - `device_test.py --variants "rq;rope,rq"` must exit 0 on a stopped node before op-rope-rq, op-rope-rq-tune or op-rope-rq-tune128;
  - an `ftz-corner` mismatch drops rq.
- `device_test.py` no longer dies on the first failing variant. A variant that raises is recorded as failed (exit 1) with its traceback, and the remaining variants still run. A summary line lists the variants with a mismatch or error.
- Runtime fail-safe in case an rq arm is started anyway and rq misbehaves:
  - every self-check launch (synthetic and real-call) is now inside try/except, and an exception disables that part with an ERROR `DISABLED ... raised` line;
  - the call then uses the live kernels, so the worker does not crash;
  - a `DISABLED` line in rank0.log means the arm measured the control and must be discarded (README §9).

### Blocking 3: interp_check.py harness (ModuleNotFoundError 'vllm.models'). FIXED, and TEST-RESULTS corrected.

- `scripts/interp_check.py` now registers the loaded base sonar copy as `vllm.models.deepseek_v4.nvidia.ops.sonar_bf16_bmm`, together with its parent packages. This is the reviewer's stub fix in a tidier form.
- The reviewer's run with the equivalent fix gave INTERP CHECK PASS, 12/12. The log is at `ref/compile-review-fixA/interp.log`.
- I could not rerun it myself: the cluster lock was held, and there is no Triton on 512S1 or the Mac.
- TEST-RESULTS.txt is rewritten. It states that neither script could run as shipped, which runs were made by the reviewer, and that my own check on the scripts was py_compile only.

### Non-blocking items

- **Unchecked eager runs: FIXED.**
  - `_run` now computes `can_check = not capturing and _checks_left > 0`. When it is False (capture, *or* eager with the budget spent), a part with `_ok` not True runs the live kernels.
  - test_plan has a new case: budget 0 and a part never checked, for rope, tune and rq. Result: live partial plus live reduce, and the part stays unchecked.
  - The real-call self-check launches are wrapped in try/except and fail closed. Another new test_plan case raises in the new partial and in rq: the part is disabled, ERROR is logged, and the result is live.
- **Uninstall order: FIXED and documented.**
  - `install.py uninstall` first checks, for every file, that `sha(current) == sha(patched(.bak-r2-attnout))`. If any file fails the check, it refuses before writing anything.
  - New test_install case: a file changed after install leads to a refusal with nothing restored or removed.
  - README §8 and KIT.json `install_order`: install r1-pdl first, uninstall r2-attnout first.
- **Unknown flag values: DOCUMENTED.** README §2 and KIT.json flag meaning: `false`, `no` and similar values are treated as on by o_proj.py, then `ValueError` is raised at the first o_proj call, so the worker crashes loudly at boot. Use unset or `0` for off.
- **FTZ corner:** unchanged. device_test reports it separately, and README §9 now says an ftz-corner mismatch drops rq.
- **r1-pdl interplay: DOCUMENTED** (README §8). Controls must use the same r1-pdl flag state as the treatment. With r1-pdl on, the r2 arms replace its o-path PDL partial/reduce, so the gain over that control may be smaller.
- **Gains unmeasured:** unchanged. The estimates stay as before, and the tune parts may give nothing. device_test prints the timings.

### Tests run by me after the fixes (TEST-RESULTS.txt, 2026-11-08T15:30:53Z)

All five exited 0:
- test_install, test_flag_off and test_jit_globals on the Mac.
- test_plan, with 13 cases including 2 new ones, and test_emulation on 512S1 (torch 2.11).
