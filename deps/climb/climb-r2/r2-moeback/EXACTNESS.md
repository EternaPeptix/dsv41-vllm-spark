# r2-moeback: exactness argument (op-order maps, old vs new)

Sources read: live files copied from spark-one on 2026-11-08 into `base/` and `ref/` (sha256 in KIT.json):
`~/dsv41-fc2/fc2_status_gated_gather_trial.py` (733c3a87), `moe_runner.py` (20b69fe5), `shared_experts.py` (2f0bca4b),
`modular_kernel.py`, `deep_gemm_moe.py` (9364f121), `fp8_utils.py` (b35c0561), the Athena FC2 sources under
`~/"Athena Hillclimbing stuff"/moet-fc2packed-snapshot/` (binding 1d25bbe4, kernel e179840a, built into the live
`athena_tp8_fc2_packed_graph_sm121.so` 85d888ae), and the production Triton cache on spark-one
(`~/.triton/cache/{4QI5MV3C,7GJD3IAD,B7AWUHIX}/_gated_gather.ptx`, copied to `ref/triton-cache/`).

## G25 `DSV41_FC2_GATHER_ADD`: routed+shared add folded into the Athena gather

### Old sequence (production), per output element (t, d), t < M tokens, d < 5120

Kernel 1 `_gated_gather` (fc2_status_gated_gather_trial.py:17-50; launch :77-83, grid (5, min(M,1024)), BLOCK_D 1024,
num_warps 2, so 16 elements per thread, each element owned by exactly one thread):

| step | op | dtype / rounding |
|---|---|---|
| 1 | `ok = (status[0] & 11) == 0` | int |
| 2 | `acc = +0.0` | fp32 |
| 3 | for k = 0,1,...,5 in order: if `ids[t,k] >= 0`: `v = f32(mm2[inv[t,k], d])` (exact), `w = weights[t,k]` (fp32, already includes the routed scaling factor, dsv4_topk.py:148), `acc = fma.rn(v, w, acc)` | PTX `fma.rn.f32x2` (48 per thread = 6 k x 8 pairs), no `.ftz` (0 occurrences in all 3 cached variants) |
| 4 | `r = bf16(acc)` | `cvt.rn.bf16x2.f32`, round-to-nearest-even, subnormals kept |
| 5 | store `r` into the routed output (`output = torch.empty_like(hidden_states)` in modular_kernel.py apply; the experts' fused_out aliases it) | |
| — | status failed: steps 3 skipped, `r = +0` | |

Kernel 2 `vectorized_elementwise_kernel<4, CUDAFunctor_add<c10::BFloat16>>` = `shared_output + fused_output`
(moe_runner.py:789), next on the main stream after the runner waited on the shared expert's `_output_ready_event`
(moe_runner.py:646):

| step | op | dtype / rounding |
|---|---|---|
| 6 | `a = f32(shared[t,d])`, `b = f32(r)` | exact |
| 7 | `s = a + 1.0f * b` (opmath float, alpha = 1): nvcc emits `fma.rn.f32(1.0, b, a)` or `add.rn.f32`; both are one RN of the exact sum, no FTZ (PyTorch is not built with fast-math) | fp32, one RN |
| 8 | `out = c10::BFloat16(s)`: round_to_nearest_even bit trick (NaN -> 0x7FC0) | bf16 RNE |
| 9 | `out` (a new tensor) is the all-reduce input (`_maybe_reduce_final_output` -> `ar_kernel_pl`) | |

### New sequence (flag on), per output element

One kernel `_gated_gather_add` (files/.../dsv41_moe_fold.py:34-71), same grid, BLOCK_D, num_warps, same thread per
element:

| step | op | vs old |
|---|---|---|
| 0 | `sh = shared[t,d]` (bf16 load, issued before the k loop; independent of steps 1-4) | new load, no arithmetic |
| 1-4 | **the same source lines** as `_gated_gather` lines 31-45 and the same `.to(bf16)` of line 50 (test_install.py checks the kernel body is identical and the live kernel's lines 1..51 are unmoved) | identical |
| 6-8 | `s = f32(r) + f32(sh)`, `out = bf16(s)` | see below |
| 9 | stored into the routed output, which MoERunner.forward now passes to the all-reduce (its add is skipped) | same AR input values |

Steps 6-8 equal the old steps 6-8 bit for bit for every pair of non-NaN bf16 values, whichever of the lowerings
Triton/LLVM picks:
- (B) `add.rn.f32` then `cvt.rn.bf16.f32`: same single fp32 RN as torch (addition is commutative, so the operand
  order `shared + routed` vs `routed + shared` does not matter), then IEEE RNE to bf16, which equals c10's
  round_to_nearest_even for every non-NaN fp32 (including overflow to inf and subnormal results);
- (C) LLVM may narrow `fptrunc(fadd(fpext a, fpext b))` to a native `add.rn.bf16x2` (seen in the prefill
  `_fwd_kernel_ep_gather_add` PTX): a single RN of the exact sum. Double rounding fp32 -> bf16 is innocuous for
  addition because 24 >= 2*8+2, and fp32 and bf16 share the exponent range, so subnormal results agree too.

**Proof:** `tests/emul_add_exhaustive.py` (run on 512S1, output `tests/out_emul_add_exhaustive.txt`) enumerates
every ordered pair of non-NaN bf16 (4,261,739,522 pairs whose result is not NaN) and computes A (torch: fp32 add +
c10 RNE bit trick), B (fp32 add + independent IEEE RNE from float64) and C (exact sum rounded once to bf16, computed
from float64, which is exact-then-innocuous for two 8-bit-precision operands): **A != B: 0, A != C: 0**. Its probes
check the rounding functions on ties, subnormals, max-finite and overflow, and a negative control shows truncation
would be detected.

Why the bf16 rounding of the routed value must stay before the add: adding `shared` into the fp32 accumulator
(one rounding) is NOT exact; `tests/emul_gather_fold.py` shows that variant differs on 88,586 of 518,400 elements,
while the kit's sequence differs on 0 (9 input kinds: normal, wide range, subnormal, signed zeros, max magnitude,
ties, exact cancellation, -1 expert ids, failed status; fma emulated exactly, 4000/4000 self-check against
exact-rational rounding).

Status-failed path: old `bf16(f32(shared) + f32(+0))`, new the same expression; identical (a failed step is
discarded and fail-stopped anyway).

NaN: only for NaN/inf inputs; payload may differ (0x7FC0 vs 0x7FFF), NaN positions identical. Not reachable with
finite activations; `device_test.py` compares NaN positions and all other bytes.

FTZ: the production gather PTX has no `.ftz`; `tools/ptx_check_g25.py` lowers both kernels for sm_121 with
Triton's compiler (no GPU) and fails on any `.ftz`, on a changed fma count, or on any fp op in the epilogue other
than the lowerings above. Since the review fix it also requires EXACTLY one of the two proven epilogues, counted per
thread (16 elements): A = 16 `cvt.rn.bf16x2.f32` in total (8 for the routed rounding + 8 for the sum) and 16 fp32
add elements, no bf16 add; or B = 8 `cvt.rn.bf16x2.f32` and 16 bf16 add elements, no extra fp32 add. The known-wrong
lowering (shared added into the fp32 accumulator before the single rounding: 8 cvt + 16 fp32 adds) therefore fails
the PTX check as well as the device test (`tools/ptx_rule_g25.py`, unit-tested by `tests/test_ptx_rule.py`).

### Ordering / race analysis
- The fold adds one dependency and removes none: before launching `_gated_gather_add`, `take()` makes the current
  stream wait on the shared expert's `_output_ready_event[idx]` (the same event `SharedExperts.wait()` waits on two
  host statements later, moe_runner.py:646, still executed). In graph capture this is the same cross-stream edge,
  moved from the add to the gather. On the no-overlap path (M > VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD) the
  shared expert already ran on the main stream before the routed experts: no event needed.
- The shared tensor is peeked (`_output[idx]`), not consumed; `.output` still consumes it exactly once after
  `wait()`, so the shared_experts.py:170 assert that killed the prefill version cannot fire (test_plan_g25 asserts
  `_output[0] is None` after every forward). Its Python reference stays alive in the runner until after the add
  point, i.e. longer than its last device read (the gather) in stream order.
- The routed output is a fresh `torch.empty_like(hidden_states)` per call (modular_kernel.py apply); the add's
  output allocation is removed (one fewer allocation, none added; nothing new inside graph capture).
- The fold is only armed when every step between the gather and the all-reduce is the identity on the routed
  output (dsv41_moe_fold._runner_ok): routed_scaling_factor == 1.0, no routed input/output transform, no sequence
  parallel, pcp_size == 1, no naive dispatch/combine, fused output not pre-reduced, no DBO; and only consumed when
  the shared tensor has the gather output's exact shape/dtype/device and does not overlap it (a padded routed width
  is declined). A second Athena gather under one armed call raises (fail closed). Any decline -> the old path.
- Drafter MoE layers and non-Athena layers never call `take()`: unchanged.

## G18 `DSV41_FC2_PDL`: Athena FC2 PDL launch with early weight staging

### FC2 kernel `athena_tp8_fc2_serial32packed_k288_candidate`, per output element (row, col)
Old and new execute the same instructions in the same order, plus one `griddepcontrol.wait` in the new kernel:

| step | op | old | new |
|---|---|---|---|
| 1 | read 32 `group_layout` entries, skip all-padding segment; per 8-row subgroup `prepare_group` | yes | yes (before the wait) |
| 2 | `stage_288_fc2_64`: copy the 9 x 1024 B packed weight tile + 192 scale words of `group.expert` to shared memory, `__syncthreads` | yes | yes (before the wait) |
| 2b | `griddepcontrol.wait` (returns when SiLU-mul-quant has completed and flushed) | — | **inserted** (no-op on later subgroups; a no-op when not PDL-launched) |
| 3 | read `a_scale[row]` and the 288 live fp8 values of row `arow` | yes | yes (after the wait) |
| 4 | for slice 0..2, kb 0..3 with k32 < 9 (fixed order): `qmma<kb>(acc[nt], ...)` for nt = 0,1 | yes | identical |
| 5 | `out[row, col] = __floats2bfloat162_rn(acc[nt][0], acc[nt][1])` | yes | identical |

The arithmetic is in steps 4-5, which are unchanged source compiled with the same nvcc 13.0.88 flags
(`build_pdl.sh` mirrors the live build.ninja). `build_pdl.sh` also rebuilds a CONTROL .so from the unmodified live
sources and diffs SASS instruction streams: control == live shows the toolchain reproduces production, and control
vs PDL must differ only by the wait (see `build/COMPILE-LOG.txt` once built). `install.py vet-pdl` enforces this
mechanically before any sha is recorded: live == control exactly; control has no `ACQBULK` (griddepcontrol.wait),
PDL has at least one; after removing `ACQBULK`/`NOP` and normalising branch targets, PDL == control (STRICT) or the
opcode multiset is equal (SCHEDULE-ONLY: the asm `"memory"` clobber may move loads/registers, never add or change an
arithmetic instruction); no global store or non-OR atomic precedes the first wait in listing order.

Data hazards across the early start (FC2 may start while SiLU runs):
- Pre-wait reads: `group_layout` (expert_ids) is written by the fused-permute scatter kernel, which completed before
  FC1 passed its own grid-dependency sync (FC1 is PDL-launched), and FC1 completed before SiLU started (SiLU is a
  plain launch), and FC2 cannot start before SiLU has started; the packed weights and scales are constant after
  load. Nothing the primary (SiLU) writes is read before the wait.
- Pre-wait writes: only the structural-error `atomicOr(status, 2)`; SiLU never touches status.
- Post-wait: A (written by SiLU) and the `out` stores. `out` = `mm2_out` aliases `mm1_out` (both are views of
  workspace2, deep_gemm_moe.py:812/839), which SiLU reads: storing only after the wait avoids the write-after-read
  race. `device_test.py` part (b) reproduces exactly this aliasing in a CUDA graph with the real SiLU kernel and the
  early trigger, >= 200 replays with new data, and compares with an eager live reference.
- Successor: the gather is a plain launch, so it starts after FC2 completes; FC2 issues no early trigger.

SiLU kernel with `DSV41_MOE_PDL_TRIGGER=start` (existing constexpr, fp8_utils.py:141-143, 176-177): adds
`griddepcontrol.launch_dependents` at the start; no arithmetic change. It also lets the drafter's deep_gemm FC2
(PDL-launched, waits in `cudaGridDependencySynchronize` before reading A) start early; that combination was
flag-on bitwise in pd-pdl (2026-10-25) and W-E3.

### SiLU-mul-quant fold into the FC2 prologue: not built
It would have to reproduce the Triton lowering of fp8_utils.py:213-262 bit for bit in CUDA (div.full.f32 for the
three divisions, `tl.exp` as ex2.approx of x*log2e, `tl.log2` as lg2.approx, the `up + 0.0` kept for -0, the bf16
round-trip of y, satfinite e4m3, the exponent-byte clamp) and restructure the FC2 A path (bf16 FC1 rows instead
of fp8 + packed scale, computed redundantly by all 80 N-tile CTAs). None of that can be proven on the CPU and the
device gate needs a GPU; it is a multi-week FC2 rewrite with a new failure surface, so it is out of this kit.
