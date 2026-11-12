# r2-moefront: adversarial review (review-moefront, 2026-11-08)

**Verdict: FIX.** The kit is safe to install with both flags off. Two things block arming it:

- The new Triton kernel has never been compiled for sm_121 and has never run in the interpreter.
- The device bit-exact test has not been run.

I found no code defect in the flag-off path, the installer, the quant op order, the plan/scatter integer logic or the PDL ordering.

Reviewed: install.py d5e62b64, dsv41_routeprep.py 32425705 (= KIT.json new_sha256). All 13 files in SHA256SUMS verify.

## 1. The live base matches the kit

I checked the files with sha256 myself on spark-one, spark-four and spark-eight. All three hosts are identical:

| file | sha256 |
|---|---|
| deep_gemm_utils.py | 58e264c8 (= KIT base) |
| experts/deep_gemm_moe.py | 9364f121 (= KIT base) |
| sm121_target_m32.py | 38fb644b |
| dsv4_topk.py | 45c5fc4f |
| no_dp_ep.py | 263c30ea |
| modular_kernel.py | 82a5d100 |

`dsv41_routeprep.py` is absent on all three, and there is no `.bak-r2-moefront`. The live launcher is **08a249da** (not fda6d05a). The flags relevant here are unchanged:

- `DSV41_MOE_FUSED_PERMUTE=1`
- `DSV41_SM121_TARGET_M32=1`
- `TARGET_M32_MAX_M=96`
- `LIVE_ACT=1`, `COMPACT=1`
- `FC1_DEAL=deal`
- `ATHENA_FC2_SERVING_TRIAL=1`

rank0.log confirms `MoEPrepareAndFinalizeNoDPEPModular` + `DeepGemmFP4Experts`. CLUSTER_LOCK was present on spark-one during this review, so I compiled nothing.

## 2. Flag-off identity: yes

I installed into a scratch copy of the live tree and diffed it:

- **deep_gemm_utils.py: 5 tagged lines.** A module-level `_DSV41_RP` bool plus a gated import, and one `if _DSV41_RP and isinstance(...)` before the mfront block.
- **deep_gemm_moe.py: 7 tagged lines.** The flag plus import, an `if _DSV41_RP and ...` before the asserts in `DeepGemmFP4Experts.apply`, and a gated property assignment at EOF.

With the flag unset:

- Nothing is imported.
- No property is installed, so the base `FusedMoEExpertsModular.expects_unquantized_inputs` returns False and `prepare()` quantizes exactly as today.
- Both `if`s short-circuit on a module-level False.

There is no new kernel, stream, event or allocation, and graph contents are unchanged. Stripping the tags gives back the base bytes (rerun, PASS).

The Athena runtime transform runs `exec`s `transform_apply(getsource(apply))` in `dict(engine.__dict__)`. That namespace contains `_DSV41_RP` and `_dsv41_rp`, its 3 anchors each still match once, and its `begin` lands after `front()`. Rerun, PASS.

## 3. Exactness when on (G04): argument complete, device proof pending

**Quant.** I checked it against the live SASS (`audit-20261106/work/verify-G04-exactness/src/ptgq_sm120.sass`, offsets 0x4c0..0xa00):

| step | live SASS | kit |
|---|---|---|
| absmax | `FMNMX |x|,eps` then the max tree | `max(max|x|, eps)`: same value, order-free |
| y | IEEE div by c[0x3a8]=fp8_max (`MUFU.RCP`+2 Newton+`FCHK` slow path) | `tl.math.div_rn` → `create_precise_divf` (div.rn; not libdevice, so `enable_reflect_ftz` does not apply) |
| floor | `FMNMX |y|, 1e-10f` | same |
| scale | `LOP3 &0x7f800000`, `+0x800000` if mantissa≠0 | identical integer ops |
| reciprocal | fast path `MUFU.RCP`+Newton for exponent window | `div_rn(1, 2^k)`; both exact 2^-k for the reachable exponents 94..247 |
| quant | `FMUL` (non-FTZ) | product with `enable_fp_fusion=False` |
| clamp | `FMNMX` max fp8_min then min fp8_max | same order |
| encode | c10 software e4m3 (thresholds 0x43f00000 / 0x3c7fffff, `+0x407ffff+odd`, `FADD a+16384`) | same; the int-on-`a` vs int-on-`q` difference is confined above bit 27 |

Any FTZ in the lowering only touches values that encode to the same signed zero, or that lose to eps in the absmax.

- **Packed words.** Same formula as the scatter kernel.
- **Plan.** Same histogram and 32-aligned cumsum.
- **Slot = start[e] + prefix rank.** This is a bijection onto the same segment, and every row has a single owner.
- **Dead rows.** Layout -1 and words 0, written exactly once.
- **ends.** start+cnt.
- **inv_perm.** Unwritten for id -1, as today.

Row order inside a segment changes from atomic-arrival order to (token, k) order. FC1/FC2 are row-independent and the gather reads through inv_perm. The new order is also the most likely arrival order of the old atomics.

**Fallback.** For M>96 or a static-check failure, `apply()` calls `_quantize_input(x, quant_config)`, which is exactly what `prepare()` calls. `apply_router_weight_on_input` is already applied in `prepare()` before the deferral.

**CPU tests rerun on 512S1 (exo-venv313), all PASS:**
- test_module: ALL PASS
- test_quant_emul: 10,092,544 element cases + both absmax sweeps + 1,152 rows, ALL PASS 0.7 s
- test_plan_emul: 690 cases, ALL PASS 5.6 s
- test_device_inputs: ALL PASS

**test_install rerun on the Mac: PASS.** This covers install/idempotent/uninstall byte-exact, over r1-pdl, foreign-base refusal, and the gate AST check.

## 4. Races, PDL and graphs: OK

- **Wait before any access.** The fused kernel runs `gdc_wait` before any load or store.
- **Top-k is safe to wait on.** `_dsv4_topk_kernel` (45c5fc4f:84-88, 120-121) waits in every program and triggers before its stores, so the wait gives complete, flushed `topk_ids`. x is final before the gate GEMM.
- **No hazards inside the kernel.** Programs only read x and topk_ids, and write disjoint rows. No atomics.
- **Overlap check.** x/aq_out overlap is checked and raises. It is a static pointer check, safe under capture.
- **Allocations.** 4 `torch.empty` per call (sfp, layout, inv_perm, ends), the same kinds as mfront. There is no host sync.
- **Triton variants.** Variants per SLOTS bucket compile in the eager warm-up that vLLM runs before each capture size, as the live mfront kernels already do.
- **Drafter.** The drafter (E=128) has `expects_unquantized_inputs` False and is untouched.

**G18 `pdl`/`pfN`.** I read the deal FC1 kernel actually used (`~/dsv41-stage/g130/fc1/trees/deal-base/include/deep_gemm/impls/sm120_fp8_fp4_gemm_1d1d.cuh:100-238`):

- It has no early return.
- The only things it does before the grid dependency sync at line 238 are TMA-descriptor prefetch of `__grid_constant__` params and smem barrier init.
- So an early `launch_dependents` cannot expose our outputs before completion.
- Consumers after FC1 (SiLU with `ends`, FC2, gather) are ordered through FC1's full wait.

`prefetch.global.L2` reads only static weights and scales.

## 5. Installer: OK, with composition caveats

- **Anchors and backup.** Every anchor is checked before any write. Writes go through a tmp file plus `os.replace`, and the `.bak` is made once.
- **Uninstall** strips exactly our tag. It restores base bytes when r1-pdl was installed before or after.
- **Refusal.** A foreign base is refused.

**Composition with r2-draftmoe**, which patches the same two files, tested on scratch copies:

| order | result |
|---|---|
| moefront → draftmoe | installs, both files AST-parse; uninstall draftmoe → moefront restores all bytes |
| `uninstall moefront` while draftmoe is present | REFUSES (nothing written), message says "restore .bak by hand" |
| draftmoe → moefront | REFUSES, `UNKNOWN-BASE` |

So the two installers are order-dependent (install moefront first, uninstall it last). r2-moeback does not touch these files or the Athena apply anchors.

## Blocking (before any rp / rp-pdl / rp-pf arm)

1. **The kernel has never been compiled (sm_121 PTX/cubin) or interpreted.** The brief requires a compile log. Run `bash tools/run_spark8_checks.sh` on spark-eight when CLUSTER_LOCK is absent (it refuses while the lock is there; it was present during this review). Attach compile.log and interp.log to the kit.
   - A compile failure fails engine startup with the flag on.
   - The `pfN` inline asm (`"=r,l"` pointer operand) is the least certain part.
2. **`device_test.py` must exit 0 on a stopped node** with `DSV41_MOE_FUSED_PERMUTE=1` and UE8M0. This is the only proof of the PTX lowering of `div_rn` / `mul` / clamp / encode against the live CUDA kernel. Until then, G04 exactness when on is "unproven" (strong CPU evidence).

## Non-blocking

- **Composition with r2-draftmoe.** Add `# r2-draftmoe` to the accepted foreign tags, as r2-draftmoe already accepts `# r2-moefront`. Otherwise document the order: install moefront before draftmoe, uninstall it after. Also change the uninstall refusal text: restoring `.bak-r2-moefront` by hand would silently drop the other kit's lines.
- **Partial uninstall.** `uninstall()` validates files one at a time. If the first file passes and the second is refused, the first is already restored (partial uninstall). Validate everything first, as `install()` does.
- **Encoding.** `open(p).read()` / `open(tmp, "w")` use the locale encoding, and the base files hold 3 + 9 non-ASCII bytes. This is fine on the Sparks (UTF-8 locale; C locale is coerced), but use `encoding="utf-8", newline=""` for robustness.
- **device_test coverage.**
  - The `pdl`/`pf2` variants are only tested with a non-FC1 successor (graph case G). Add case F with `load_rp("pdl")` / `("pf2")`, so the real FC1 PDL successor runs after an early trigger.
  - There is no integration test of `DeepGemmFP4Experts.apply` with the property installed (fused path and the M>96 fallback). The window's exact.py, G1 digest and long-prompt checks cover it end to end.
- **compile_check strictness.** It asserts "no `.ftz`" on the quant path. A harmless `.ftz` would fail the check. Read a failure there against EXACTNESS §1 before dropping the arm.
- **Gain numbers disagree.** KIT.json top-level `expected_gain_pct` (0.4/0.85/1.3) differs from the per-flag table and the summary (0.4/0.8/1.2).
- **Launcher sha.** Production now runs launcher 08a249da, not fda6d05a. The flags this kit depends on are unchanged.

## Fixes (fix-moefront, 2026-11-08T17:30Z)

Kit after the fixes: dsv41_routeprep.py `842f49fb`, install.py `17b89c2b`, device_test.py `69b0f0d9`, tools/compile_check.py `df7c1faa` (full hashes in SHA256SUMS and KIT.json). Live site files rechecked on spark-one/four/eight after the 17:05Z launcher promotion (production is now **dcd09a30**): deep_gemm_utils `58e264c8`, deep_gemm_moe `9364f121`, dsv4_topk `45c5fc4f`, sm121_target_m32 `38fb644b`, unchanged. Flags this kit relies on are still defaults in dcd09a30 (FUSED_PERMUTE=1, TARGET_M32=1 MAX_M=96, ROUTE_LIVE=skip, COMPACT=1, FC1_DEAL=deal, AV_PROFILE=pinned).

### Blocking 1: kernel never compiled / interpreted. FIXED (run, logs attached)

CLUSTER_LOCK was held by windows w2 and w2b and the climb-w2 promotion until 17:12:52Z. `tools/wait_and_run_spark8.sh` polled the lock every 60 s and started `tools/run_spark8_checks.sh` at 17:14:31Z, after the lock had been absent for two polls. The script now also kills its own remote step if the lock reappears. It ran on spark-eight with Triton 3.7.1, `/usr/local/cuda/bin/ptxas`, `CUDA_VISIBLE_DEVICES=` and `nice -n 19`. The Triton cache was deleted afterwards; 332 KB of kit copy is left in `~/climb-scratch/r2-moefront`.

- **Run 1** (`results/compile.run1.log`): all 10 variants compiled to PTX and cubin for sm_121. One check failed: "no atomics". It matched `atom.shared.add.u32`, which is how `tl.histogram` lowers: integer adds into a shared-memory histogram, followed by `bar.sync`. Integer addition is exact and order-independent, so the counts are deterministic. The check was too broad. It now fails on any global `atom`/`red` and on any shared atomic other than integer add, and it reports per variant. There was no kernel change.
- **Run 2** (`results/compile.log`, sha256 `c3749904…`): `ALL PASS`, `compile_check.py exit 0`. It covers 10 variants: SLOTS 64/256/1024; off/pdl/pf2/pf4/pf8; int32/int64 ids; PDL off. Facts from the PTX:
  - `div.rn.f32` ×80, `mul.rn.f32`, `add.rn.f32`, plain `max.f32`/`min.f32`.
  - No `.ftz` anywhere (INFO line), no `fma.rn.f32`, no `div.full`/`div.approx`, no e4m3 `cvt`.
  - The constants `0f2EDBE6FF` and `0f46800000` are present.
  - `griddepcontrol.wait` comes before the first global load/store in every PDL variant, and it is absent with PDL off.
  - `launch_dependents` appears only in pdl/pf. `prefetch.global.L2 [%rdN]` appears only in pf2/pf4/pf8; the inline asm `"=r,l"` operand compiles and assembles.
  - No global atomics in any variant.
- **Interpreter** (`results/interp.log`, sha256 `ce07f23e…`): the real kernel source under `TRITON_INTERPRET=1` was compared with the emulated live chain (SASS-order quant, mfront plan and scatter). 20 cases (M 1/2/4/6/11 × random/dead/same/hostile) are bit-identical. `interp_test.py exit 0`. Run 1 and run 2 gave byte-identical logs.

**Hardening, so that a compile failure can no longer stop engine startup.** At the first eligible call, `_ready()` (dsv41_routeprep.py:330) compiles all 5 SLOTS buckets of the variant in use with `JITFunction.warmup`. The first eligible call is in the eager profile/warm-up run, which comes before any capture.
- The warmup goes through the same `_launch()` (:294) that builds the real launch's arguments.
- In Triton 3.7.1 (`triton/runtime/jit.py:358-359, 708-753` on spark-eight), warmup binds the arguments, compiles and caches, and returns before `kernel.run`. Nothing is launched and no memory is touched.
- Any exception sets `_READY=False`. Every call then takes today's `prepare()` quant plus the mfront chain, and the log says `fused route prep OFF (precompile failed: ...)`.
- test_module checks the following:
  - warmup arguments equal the real launch's (grid, constexprs, options, ints, pointer dtypes) for FC1_FRONT 0/pdl/pf2;
  - a simulated ptxas failure means no launch and the fallback path for every later call;
  - a failing static check means no precompile.
- Residual risk: a Triton specialization that first appears at a later launch, for example an unaligned pointer, is not precompiled (README Risks).

### Blocking 2: device_test.py exit 0 on a stopped node. OPEN (lead, needs a GPU)

This cannot be done under the hard rules: no GPU here, no taking the lock, no stopping a node. It remains a hard gate in KIT.json `gates_before_window`, README "Gates" and TEST-RESULTS.txt, and is required before any rp, rp-pdl or rp-pf arm. What changed in the harness:
- **Case W (new):** precompile on the GPU for FC1_FRONT 0/pdl/pf2. It requires `_READY is True`.
- **Case F, rewritten (G18 item):**
  - It requires the production target_m32 deal FC1 to be importable. A missing FC1 is now a FAIL, not a SKIP.
  - It asserts that FC1 is PDL-launched (`native.get_pdl() is True`; `get_target_m32_kernel` copies DeepGEMM's `set_pdl(True)`).
  - Then, for `load_rp("0")`, `load_rp("pdl")` and `load_rp("pf2")`:
    1. **Eager**, 8 M values: the fused kernel, with FC1 launched immediately after it (no sync, no other launch in between).
    2. **Graph**: a captured `[dsv4_topk (PDL) -> fused (PDL, early trigger for pdl/pf2) -> FC1 (PDL)]` for M 1/6/16/96, replayed 50 times. Before each replay the FC1 output (NaN), layout, scale words, inv_perm, FP8 rows and top-k ids are poisoned.
  - Every live route's FC1 output row is compared bitwise through inv_perm with the live chain's FC1 row. If FC1 read `grouped_layout`, rows or scales before the fused grid completed, it would see the poison.
  - `--no-fc1` does not clear the gate.
- Expected runtime is now about 4-8 min. The command is in the header.

### Non-blocking

- **r2-draftmoe composition: FIXED.**
  - `FOREIGN_TAGS = ("# r1-pdl", "# r2-draftmoe")` (install.py:34).
  - A patched file is recognised as ours when stripping our tag gives an accepted base and the file equals our patch of it, ignoring foreign lines.
  - test_install section 6 covers:
    - moefront → draftmoe and draftmoe → moefront, each with and without r1-pdl underneath;
    - both uninstall orders, restoring the exact bytes;
    - draftmoe's lines are kept when moefront is uninstalled first.
  - The refusal text no longer tells anyone to restore a `.bak` by hand.
  - KIT.json `composes_with` documents one order for all 8 nodes: r1-pdl → r2-moefront → r2-draftmoe, uninstalled in reverse. The composed bytes depend on the order, though both orders work.
- **Partial uninstall: FIXED.** `uninstall()` (install.py:208) checks every file before writing any, as `install()` does. Test 7: when the second file is foreign, uninstall refuses and the first file stays byte-identical.
- **Encoding: FIXED.** All reads and writes use `encoding="utf-8", newline=""` (install.py:86, 92). Test 8 runs install/status/uninstall under `LC_ALL=C PYTHONUTF8=0` and gets exact bytes back.
- **No apply() integration test:** unchanged. As noted, it is covered end to end by the window's exact.py, the G1 digest and the long-prompt hashes. Case F now runs the real FC1 behind the fused kernel.
- **compile_check strictness: FIXED.** `.ftz` is reported as INFO only, and none was found. The atomics check was narrowed as described above.
- **Gain numbers: FIXED.** KIT.json top-level `expected_gain_pct` is now 0.4 / 0.8 / 1.2, the same as the rp flag and the summary. pdl/pf stay separate per-flag entries.
- **Launcher sha: UPDATED.** README and KIT.json `production_launcher` name dcd09a30 (promoted 17:05Z) and its history (fda6d05a → 08a249da → dcd09a30). The newly promoted `L2_PREFETCH_R1_SHW2/KVSRC` add L2 traffic that the pf arms stack on (README Risks).

### Tests rerun after the fixes (TEST-RESULTS.txt)

| test | where | result |
|---|---|---|
| test_install | Mac | exit 0, 50 PASS |
| test_module (incl. precompile cases) | 512S1 | exit 0 |
| test_quant_emul (10,092,544 element cases + sweeps) | 512S1 | exit 0 |
| test_plan_emul (690 cases) | 512S1 | exit 0 |
| test_device_inputs | 512S1 | exit 0 |
| compile_check (sm_121, 10 variants) | spark-eight, lock absent | exit 0 |
| interp_test (20 cases) | spark-eight, lock absent | exit 0 |

device_test.py was **not run** (needs a GPU).

**Status:** G04 exactness-when-on is proven through PTX (compile log) and the interpreter, and on CPU exhaustively. ptxas, the SASS and the hardware are proven only once device_test exits 0. Until then, no rp arm.
