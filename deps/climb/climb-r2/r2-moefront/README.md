# r2-moefront: MoE front-chain fusion (G04 Stage A) + FC1 front (G18, FC1 part)

Climb round 2, kernel levers. Two default-off flags, one installer, one new module. Base: production launcher
dcd09a30 (promoted 2026-11-08T17:05Z; built against fda6d05a, reviewed on 08a249da; the flags this kit relies on are
unchanged), vLLM site files identical 8/8 and unchanged at 17:2xZ (checked spark-one/four/eight). Exactness map: `EXACTNESS.md`. Review answers: `REVIEW.md` "Fixes".

## Gates before any window arm (status in TEST-RESULTS.txt)

1. **Compile + interpreter on spark-eight** (rule 4; never while `CLUSTER_LOCK` exists): from the Mac, in this folder,
   `bash tools/run_spark8_checks.sh` (or `tools/wait_and_run_spark8.sh`, which polls the lock every 60 s and runs it
   once the lock has been absent for two polls). It kills its own remote step if the lock appears. Pass =
   `results/compile.log` ends with `ALL PASS` / `compile_check.py exit 0` and `results/interp.log` with `ALL PASS` /
   `interp_test.py exit 0`. Required before every rp arm. **DONE 2026-11-08T17:15Z** (lock absent): 10 variants
   compiled to PTX + cubin for sm_121 (SLOTS 64/256/1024; off / pdl / pf2 / pf4 / pf8; int32 / int64 ids; PDL off),
   every PTX check passes (IEEE `div.rn.f32`, no `.ftz` at all, no `fma`, NaN-ignoring `max`/`min`, no e4m3 `cvt`,
   `griddepcontrol.wait` before the first global access, `launch_dependents` only in pdl/pf, `prefetch.global.L2`
   only in pf, no global atomics), and the interpreter runs the real kernel source on 20 cases bit-identical to the
   emulated live chain. Logs: `results/compile.log`, `results/interp.log`, PTX in `ptx/`. Rerun if
   `dsv41_routeprep.py` changes.
2. **OPEN (lead, GPU): `device_test.py` exit 0 on a stopped node** (header has the command, ~4-8 min): the only check of the PTX lowering
   against the live CUDA quant kernel. Required before every rp arm; its case F (the real PDL-launched FC1 directly
   behind the fused kernel, for FC1_FRONT 0 / pdl / pf2, eager and 50 poisoned graph replays) is required before the
   rp-pdl / rp-pf arms. `--no-fc1` does not clear that gate.

Safety net if gate 1 was skipped: with the flag on, the first eligible call (eager profile/warm-up run) precompiles all
5 SLOTS buckets of the variant in use; a Triton/ptxas error logs `fused route prep OFF (precompile failed: ...)` and the
process keeps today's code (the arm is then inert, not broken).

## What runs today (per target MoE layer, routed stream, nsys d200 windows; `results/chain_control.txt`)

`splitKreduce -> _dsv4_topk_kernel (PDL) -> per_token_group_quant_8bit_register_kernel<bf16,e4m3,false,true,128> ->
_dsv41_fp_plan_kernel (1 CTA) -> _dsv41_fp_scatter_kernel (6M CTAs) -> FC1 sm120_fp8_fp4_gemm_1d1d_impl<0,576,5120>`

| window (rank 0, d200) | chain splitK end -> FC1 start | g_tq | quant | g_qp | plan | g_ps | scatter | g_sf | slack (routed - shared) |
|---|---|---|---|---|---|---|---|---|---|
| code  | 14.36 us (0.574 ms/step) | 0.40 | 2.71 | 0.69 | 3.67 | 0.65 | 2.45 | 0.33 | mean 98.6, min 39.9 us |
| prose | 14.04 us (0.552) | 0.43 | 2.66 | 0.68 | 3.38 | 0.67 | 2.41 | 0.35 | mean 65.2, min 27.3 us |
| maths | 14.44 us (0.577) | 0.40 | 2.69 | 0.69 | 3.75 | 0.64 | 2.47 | 0.32 | mean 86.7, min 37.5 us |

The segment `g_tq + quant + g_qp + plan + g_ps + scatter` is 10.57 / 10.23 / 10.64 us per layer = 0.423 / 0.403 /
0.425 ms per step (40 / 39.3 / 40 layers). It is serial on the routed stream, and the routed chain is the last arrival
at the MoE join in every one of the ~2,400 layers per window (slack min 27 us >> the saving), followed by the MoE
all-reduce on the same stream: **on the critical path**.

## Lever 1: `DSV41_MOE_ROUTEPREP=1` (G04 / moe:L-A, Stage A)

- `DeepGemmFP4Experts.expects_unquantized_inputs` (installed as a property only when the flag is on) returns True for
  the target experts object (E384, top-6, H5120, `_dsv41_target_m32` set). `prepare()` then hands the bf16 MoE input
  through unquantized (stock vLLM `defer_input_quant` path of `MoEPrepareAndFinalizeNoDPEPModular`).
- `DeepGemmFP4Experts.apply` (one tagged `if` before its asserts) calls `dsv41_routeprep.front()`: for eligible calls
  (static: UE8M0 e4m3fn +-448 quant, mfront serving; per call: M32 align-32 path, i.e. M <= 96, bf16 contiguous x,
  int32/int64 ids, no expert map) it passes `(x_bf16, Deferred(w1, w1_scale))`; otherwise it runs
  `_quantize_input(x, quant_config)`, the exact call `prepare()` would have made, and today's code continues.
- `deepgemm_moe_permute` (one tagged `if` before the mfront branch) sends a `Deferred` scale to
  `dsv41_routeprep.fused_permute`, which launches **one** kernel `_dsv41_rp_kernel` (grid 6M, 4 warps, PDL-launched
  like the top-k kernel) returning the mfront contract `(aq_out, packed UE8M0 sfp, grouped_layout, inv_perm, 32,
  expert_ends)`. FC1 / SiLU / Athena FC2 / gather are unchanged.
- The kernel: `gdc_wait`; recompute the plan in every program (same histogram / 32-aligned cumsum as the plan kernel);
  deterministic slot `start[e] + prefix rank`; quantize the token row with the live CUDA kernel's exact op sequence
  straight into the slot (5 chunks of 8 groups) and write the packed scale words; dead rows (segment padding, tail) get
  layout -1 / scale word 0 from exactly one owner; program 0 writes `expert_ends`. No atomics.
- Removed kernels per target layer: the quant, plan and scatter launches and their 3 gaps (and the top-k -> quant gap
  becomes a PDL edge).

**Why flag-off is identical.** Every inserted line carries `# r2-moefront`; stripping them gives the base files byte
for byte (tested). The module-level gate `_DSV41_RP` is False for unset / 0 / off / false / '' (tested); every
inserted statement is that assignment or sits under `if _DSV41_RP...:` (AST-tested). With the flag off the module is
not imported, the property is not installed (the base class property returns False as today), and no kernel, launch,
stream, event or allocation changes.

**Why flag-on is bit-exact.** Op by op in `EXACTNESS.md`: absmax (exact max), IEEE `absmax/448` (div.rn = the live
MUFU+Newton+FCHK correctly rounded division), the 1e-10 floor, bit-math ceil-log2, IEEE reciprocal of a power of two
(exact 2^-k), RN `x*inv`, NaN-ignoring clamp, and the live c10 software e4m3 encoding replicated with the same integer
and fp32-add steps (no hardware cvt). Plan/scatter are integer/copy work producing identical arrays; only the row
order inside an expert segment changes (from atomic arrival order, which is already nondeterministic in production,
to (token, k) order), and every FC1/FC2 row depends only on its own A row and scale; the gather reads through inv_perm.

**Race analysis.** Nothing is read or written before `griddepcontrol.wait`; after it, `topk_ids` and `x` are final
(top-k waited on the splitKreduce; x is final before the router GEMM starts). Programs write disjoint rows (single
ownership tested for 690 routings), no atomics, `inv_perm[t,k]` by one program, `ends` by program 0. `x` and the slot
rows must not overlap (checked on the host, RuntimeError otherwise). FC1 is a stream successor (PDL: its
`cudaGridDependencySynchronize` waits for full completion + flush before it reads anything).

**Expected saving.** New chain = top-k -> (PDL) fused kernel -> FC1. Fused kernel estimate 2.5-4 us (plan recompute
<= 1 us at SLOTS 64, 5 x 1024-element quant chunks, 5 KB row store; the plan kernel's 1-CTA fill of M_sum x 11 words
is now spread over 6M CTAs) plus a PDL gap of about -0.5..+0.4 us. Saving 6-8 us per layer of the 10.2-10.6 us
segment = 0.24-0.32 ms per step before haircut; with the audit's 0.85 transfer haircut 0.19-0.27 ms.
G1' = mean 1000*AL/(T - d): d = 0.12 -> +0.5%, 0.19 -> +0.8%, 0.24 -> +1.0%, 0.30 -> +1.3%. **Central +0.8% G1**
(low +0.4, high +1.2). Calibration: mfront (6 kernels -> 2) measured 6.5-7.4 us/layer on the bench and +1.29% G1.

## Lever 2: `DSV41_MOE_FC1_FRONT=pdl | pf1 | pf2 | pf4 | pf8` (G18 / moe:L-F, FC1 part; needs lever 1)

The G18 skeptic's corrections apply: FC1 is already PDL-launched (`utils/deep_gemm.py:289-291 set_pdl(True)`, copied
into the deal native by `sm121_target_m32.py`; scatter -> FC1 gap median 0.38 us with negatives) and a pre-wait read of
`grouped_layout` would race. So this lever does **not** touch the FC1 launch wrapper or the FC1 JIT tree. It adds:
- `pdl`: `griddepcontrol.launch_dependents` in the fused kernel right after its plan (before the stores). FC1's CTAs
  can be scheduled on SMs the fused kernel does not occupy (6M <= 36 CTAs on 48 SMs at decode M <= 6) and run their
  pre-sync prologue (descriptor prefetch, barrier init, launch latency) while the route prep finishes. FC1 still reads
  nothing before `cudaGridDependencySynchronize`, so this is scheduling only.
- `pfN`: `pdl` + `prefetch.global.L2` of the first N FC1 k-blocks (64 B of FP4 each) of all 576 weight rows and of the
  matching `w1_scale` words of every selected expert, issued once per expert (its first route) right after top-k is
  known. N=2: 74 KB + 4.6 KB per expert, about 1.5-2.6 MB per layer at decode G = 20-33 distinct experts. Cache hint
  only (no register result used, nothing written). Disabled silently (logged) if w1 / w1_scale do not have the
  expected geometry (int8 [E, N, K/2] with K contiguous, int32 [E, N, K/128]).

**Expected saving** (unmeasured, small): pdl 0.01-0.04 ms/step (+0.05% central); pf adds 0-0.08 ms (the window
before FC1 is only ~3-4 us of otherwise idle DRAM after lever 1, about 0.8 MB at ~230 GB/s; FC1's own g1-pf already
prefetches 8 k-blocks ahead inside the kernel; the shared-expert GEMM on the aux stream competes for DRAM; the ledger's
extra-prefetch rows L4/L9 were null). It can also be slightly negative (DRAM/L2 contention): measure pf2 and pf4 only
after rp and rp-pdl.

## Files

| path (relative to site-packages/vllm) | action | base sha256 |
|---|---|---|
| `model_executor/layers/fused_moe/deep_gemm_utils.py` | patch (+5 tagged lines) | 58e264c8b870196f... live; c3a05cca1b4f4393... with r1-pdl installed first |
| `model_executor/layers/fused_moe/experts/deep_gemm_moe.py` | patch (+7 tagged lines) | 9364f121aacf5061... (no round-1 kit touches it) |
| `model_executor/layers/fused_moe/dsv41_routeprep.py` | new | see `SHA256SUMS` |

Not modified (read and documented): `router/dsv4_topk.py` 45c5fc4f (top-k stays as is), `experts/sm121_target_m32.py`
38fb644b (FC1 launch wrapper), `utils/deep_gemm.py` 3fae19d4 (FC1 PDL), `~/dsv41-fc2/athena_fc2_forty_layer_trial.py`
(ae2089724597; its runtime `getsource` transform of `DeepGemmFP4Experts.apply` still matches its 3 anchors once and
places `athena_fc2_begin` after our `front()` call: tested). The supervisor's preflight manifest does not include
either patched file; installs are 8/8 anyway.

## Install / rollback (all 8 nodes)

    python3 install.py dry        # anchors + accepted base, writes nothing
    python3 install.py install    # backup <file>.bak-r2-moefront, patch, copy module, print status
    python3 install.py status
    python3 install.py uninstall  # checks both files first, then removes exactly our tagged lines

Accepted bases: the live sha, or the live sha plus `# r1-pdl` / `# r2-draftmoe` tagged insertions (rule 5; both kits
only insert tagged lines). Anything else is refused before any write; uninstall is all-or-nothing and never restores a
`.bak`. Composition with r2-draftmoe (same two files) is tested in both install orders and both uninstall orders, with
and without r1-pdl underneath. The composed bytes depend on the order (KIT.json `patched_sha256`), so use one order on
all 8 nodes: recommended r1-pdl -> r2-moefront -> r2-draftmoe, uninstall in reverse. Files are read/written as UTF-8
with `newline=''` (tested under `LC_ALL=C PYTHONUTF8=0`).

## Window arms (A/B on the production launcher, dcd09a30 now; flags-off control C)

| arm | env |
|---|---|
| C | (none) |
| rp | `DSV41_MOE_ROUTEPREP=1` |
| rp-pdl | `DSV41_MOE_ROUTEPREP=1 DSV41_MOE_FC1_FRONT=pdl` |
| rp-pf2 | `DSV41_MOE_ROUTEPREP=1 DSV41_MOE_FC1_FRONT=pf2` |
| rp-pf4 | `DSV41_MOE_ROUTEPREP=1 DSV41_MOE_FC1_FRONT=pf4` |

Gate per arm: rank0.log shows `DSV41_MOE_ROUTEPREP=1 (r2-moefront): fused route prep eligible (precompiled 5 SLOTS
variants)` and `r2-moefront: first fused call ... FC1 front: trigger ..., prefetch N k-blocks (requested N)` (else the
arm is inert: `OFF (static check failed)` = mfront off or quant constants differ, `OFF (precompile failed: ...)` =
Triton/ptxas error), then exact.py ec2f4437f8e1ed09, G1 digest 2f7fdccbd388225a, 9 long-prompt
hashes. Keep `DSV41_AV_PROFILE=pinned` (a faster step must not re-profile verify widths). Do not combine with
`DSV41_QHOIST`.

## What to look for in an nsys trace

- Kernel census per decode step: `_dsv41_rp_kernel` 40 (target layers); `per_token_group_quant_8bit_register_kernel`,
  `_dsv41_fp_plan_kernel`, `_dsv41_fp_scatter_kernel` 0 on decode steps (prefill steps with M > 96 keep them: the
  fallback). The drafter's stock chain (count_expert / ep_scatter / transpose_pack) is unchanged.
- `tools/chain_rp.py trace.sqlite=label` (run on 512S1 like the audit scripts): per layer `chain` (splitK end -> FC1
  start) should drop from 14.0-14.4 us to about 6.5-8.5 us; `g_trp` (top-k end -> rp start) should often be <= 0
  (PDL), `rp` about 2.5-4 us; in rp-pdl / pf arms `g_rpf` (rp end -> FC1 start) shifts negative more often and in pf
  arms the `fc1` mean / p10 should drop if the prefetch helps. `slack` must stay > 0 (routed chain still last).
- anat3.py's regex buckets do not know `_dsv41_rp_kernel` (it lands in "other"); add `_dsv41_rp_kernel|_dsv41_fp_plan`
  to its `moe_glue` pattern for bucket totals.
- Stream: all of this is on the routed MoE stream (the one carrying splitKreduce / FC1 / ar_kernel_pl); the shared
  expert stream is unchanged.

## Tests (CPU, run; see TEST-RESULTS.txt)

- `tests/test_install.py` (stdlib): live base shas; install/idempotent/uninstall byte-exact on copies of the live tree
  and of the r1-pdl-patched tree; foreign base refused; r1-pdl anchors intact; flag-off strip == base; every inserted
  statement gated; gate False for unset/0/off/false/''; Athena transform still applies; r2-draftmoe composition (its
  real installer) in both orders, with and without r1-pdl; all-or-nothing uninstall; C locale.
- `tests/test_module.py` (torch CPU, stubs): env parsing; `expects_unquantized_inputs` truth table; `front()` fused iff
  M <= 96 with mfront=1 and UE8M0, else exactly the prepare() call; launch contract of `fused_permute` (grid, constexprs,
  buffer shapes/strides, PDL/prefetch args); refusals (non-fused geometry, x overlapping slot rows); the one-time
  precompile (5 SLOTS buckets, same args as the real launch, once per process; a warmup error or a failed static check
  -> every call takes the prepare() quant).
- `tests/test_quant_emul.py` (numpy): the live SASS op sequence vs the fused kernel's op sequence vs a float64 + torch
  c10 reference, exhaustive over every bf16 absmax and every bf16 x at every reachable scale, plus hostile rows.
- `tests/test_plan_emul.py` (numpy): mfront plan + atomic scatter (random arrival orders) vs the fused per-program
  plan, 690 routings, incl. single ownership of every row and a row-independent FC1/FC2 + gather stand-in.
- `tests/test_device_inputs.py`: device_test.py's Q1/Q2 inputs cover what they claim and agree under emulation.

## Device / build checks (the gates above; see TEST-RESULTS.txt for what has been run)

- `device_test.py` on a stopped node (header has the exact command; ~4-8 min; exit 0 only when every case is
  bit-identical): W precompile on the GPU for FC1_FRONT 0/pdl/pf2, Q1/Q2 exhaustive quant vs the live CUDA kernel, P
  route plan vs live mfront (M 1..96, 128, 170, top-k-generated and hostile routes, dead rows, int32/int64 ids), G 200
  poisoned CUDA-graph replays of [dsv4_topk (PDL) -> fused] for FC1_FRONT 0/pdl/pf2, F the production PDL FC1 launched
  directly behind the fused kernel for 0/pdl/pf2 (eager 8 M values; graph [topk -> fused -> FC1] 4 M values x 50
  poisoned replays; rows bitwise through inv_perm; missing FC1 = FAIL), T timings (median of 200, warm and cold-L2) of
  the live 3-kernel chain vs the fused kernel, with and without the top-k kernel in the graph.
- `tools/compile_check.py` on spark-eight (no GPU use): Triton -> PTX/cubin for sm_121 for 10 variants (SLOTS 64/256/
  1024; off/pdl/pf2/pf4/pf8; int32/int64 ids; PDL off), greps the PTX for the instructions the map needs. `.ftz` ops
  are printed as INFO, not failed (EXACTNESS.md section 1 shows FTZ is harmless on this path; read the INFO line).
  `tools/interp_test.py`: the real kernel source in the Triton interpreter (CPU) vs the emulation. Both need the venv's
  Triton (not available on the Macs) and must not run during a window.

## Risks

- PTX lowering differs from the emulation: the sm_121 PTX (results/compile.log) has `div.rn.f32` (not div.full/approx),
  `mul.rn.f32`, `add.rn.f32`, plain `max.f32`/`min.f32`, no `.ftz` anywhere and no `fma`, matching the live SASS map.
  ptxas -> SASS and the hardware are only proven by device_test Q1/Q2 (open gate).
- Compile failure: caught by the startup precompile (arm inert, logged). Not caught: a Triton specialization that only
  appears at a later launch (e.g. an unaligned x pointer or an M_sum stride that is not a multiple of 16) failing to
  compile while the precompiled ones succeed. That would fail the forward, as any live Triton kernel would.
- Triton JIT per SLOTS bucket (64..1024) and FC1_FRONT variant: now compiled up front at the first eligible call, a few
  seconds of extra startup compile.
- If some other code consumed the pre-permute `a1q` / `a1q_scale` of the target layers it would now get bf16: none
  found (QHOIST/ELPQ are not installed; the shared expert quantizes its own input on the aux stream).
- Lever 2 pf can cost DRAM bandwidth at the FC1 start; it is a separate arm. Production now also runs
  DSV41_L2_PREFETCH_R1_SHW2=1 / KVSRC=1 (promoted 17:05Z), so the pf arms add L2 prefetch traffic on top of those.
- Not built: Stage C (top-k fused too). It needs the 8-lane xor-4/2/1 `weight_sum` tree, `ex2.approx.f32`, FTZ
  `__nv_logf`, `sqrt.approx.ftz.f32` and `div.full.f32` reproduced and proven on device (2^32 sweeps); about +0.1-0.2%
  more than Stage A. Left for a later round.
