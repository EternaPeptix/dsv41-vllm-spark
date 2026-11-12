# r2-moefront: exactness map (old op order vs new, element by element)

Scope: the target MoE layers (E=384, top-6, H=5120, M <= 96 on the M32 align-32 path, `DSV41_MOE_FUSED_PERMUTE=1`).
Old = three kernels run today on the routed stream after `_dsv4_topk_kernel`:
`per_token_group_quant_8bit_register_kernel<bf16, e4m3fn, false, true, 128>` (vLLM `_C_stable_libtorch.abi3.so`, sm_120
cubin; SASS in `audit-20261106/work/verify-G04-exactness/src/ptgq_sm120.sass`), `_dsv41_fp_plan_kernel` and
`_dsv41_fp_scatter_kernel` (`deep_gemm_utils.py:508-597`, live sha 58e264c8). New = `_dsv41_rp_kernel`
(`files/vllm/model_executor/layers/fused_moe/dsv41_routeprep.py`). The top-k kernel (`dsv4_topk.py`, 45c5fc4f) is not
touched, so `topk_ids` / `topk_weights` are produced by the same instructions as today; Stage C (top-k fusion) is not
built (its fp32 `weight_sum` butterfly and `ex2.approx` / FTZ `__nv_logf` / `sqrt.approx.ftz` / `div.full` would
need their own device proof, and the brief says exactness first).

Outputs that FC1 / SiLU / FC2 / the gather consume (the contract of `deepgemm_moe_permute`'s mfront branch):
`aq_out` (FP8 rows in expert segments), `sfp` (packed UE8M0 words, MN-major, `p * mn_aligned + row`), `grouped_layout`,
`inv_perm`, `expert_ends` (LIVE_ACT). Each is covered below.

## 1. FC1-input quant, per 128-element group g of token row t (40 groups per row)

| step | old: live SASS (sm_120) | new: Triton line -> instruction | equal because |
|---|---|---|---|
| load | `LDG.E.128` bf16, `IMAD.U32 x<<16` / `PRMT` -> fp32 | `tl.load(bf16).to(fp32)` -> `cvt.f32.bf16` / shift | bf16 -> fp32 is exact |
| absmax | per lane (8 lanes x 16): `FMNMX |x0|, eps` then 15 x `FMNMX m, |xi|`; `SHFL.BFLY` 4, 2, 1 + `FMNMX` | `tl.maximum(tl.max(tl.abs(x), axis=1), eps)` (`max.f32` tree, `max.f32` with eps) | max over a set is exact and order-free for non-NaN values (FMNMX and `max.f32` both return the non-NaN operand); eps first or last gives the same maximum |
| y | `absmax / fp8_max` as IEEE division (`MUFU.RCP` + 2 Newton `FFMA` + `FCHK` slow path `CALL 0x2540` = correctly rounded) | `tl.math.div_rn(amax, fp8_max)` -> `div.rn.f32` | both correctly rounded RN division of the same operands (fp8_max = 448.0f runtime arg in both) |
| floor | `FMNMX |y|, 1e-10` | `tl.maximum(tl.abs(y), 1e-10)`, constant `0f2EDBE6FF` | same op, same fp32 constant (float32(1e-10)) |
| scale | `LOP3 y & 0x7f800000`; `+0x800000` if `y & 0x7fffff` (bit-math ceil-log2); stored as fp32 bits | `sb = yb & 0x7F800000; sb = where(yb & 0x7FFFFF != 0, sb + 0x800000, sb)` | identical integer ops |
| inv | IEEE reciprocal of the scale (`MUFU.RCP` + Newton, slow path for exponents outside the normal window) | `tl.math.div_rn(1.0, scale)` -> `div.rn.f32` | the scale is exactly 2^k with biased exponent 94..247 (all reachable values, CPU-enumerated), so the correctly rounded reciprocal is exactly 2^-k in both (also checked: equals `(254 - E) << 23` for every reachable scale) |
| q | `FMUL x, inv` (RN, non-FTZ) | `xf * inv[:, None]` -> `mul.rn.f32` / `mul.f32` (non-FTZ; `enable_fp_fusion=False`, no FMA partner anyway) | same RN product. Products below 2^-126 only arise for \|q\| < 2^-9 and encode to a signed zero either way |
| clamp | `FMNMX q, fp8_min` (max), `FMNMX ., fp8_max` (min), NaN-ignoring | `tl.minimum(tl.maximum(q, fp8_min), fp8_max)` -> `max.f32`, `min.f32` | same order, same NaN-ignoring semantics, same runtime constants (-448, 448) |
| encode | c10 `Float8_e4m3fn` software path in SASS: `a = q & 0x7fffffff`, sign `(q >> 24) & 0x80`; `a >= 0x43f00000` -> 0x7f if NaN else 0x7e; `a > 0x3c7fffff` -> `t = q + 0x407ffff + ((q >> 20) & 1)`, byte `t >> 20` unless `(t & 0xff00000) == 0x7f00000` -> 0x7e; else `FADD a + 16384` low byte; `\| sign` | the same three branches with int32 ops on `a` (`tn = a + 0x0407FFFF + odd`), `fs = float(a) + 16384.0` (`add.f32`, non-FTZ), byte `(mag & 0xFF) \| sign`, stored as uint8 | bits 20..27 of `q + c` and `a + c` are equal (they differ only in bit 31 / carries above bit 27), the subnormal branch is one RN fp32 add of the same operands; no hardware `cvt` is used on either side |
| scale word | fp32 scale stored per group (`STG` by lane 0), then the scatter kernel packs `(v0 >> 23) \| (v1 >> 15) \| (v2 >> 7) \| (v3 << 1)` on uint32 | same formula on int32 over groups 4p..4p+3 of each 8-group chunk (`tl.reshape(sb, (2, 4))`, selected by column), stored at `sfp + p * mn_aligned + dest` | scale bits are positive (sign 0), so arithmetic vs logical shifts agree; `v3 << 1` wraps to the same 32-bit pattern |

Each token row is quantized once per live route (up to 6 times) by the program that owns the slot; every copy runs the
same instructions on the same input, so all copies are bitwise equal to the single row the live quant wrote and the
scatter copied.

Evidence (CPU, `tests/test_quant_emul.py`, numpy float32 = IEEE RN without FTZ on the host, asserted):
- every non-negative and every negative finite bf16 absmax (32,640 each) -> identical scale bits old vs new vs a
  float64 reference; 154 reachable scales (exponents 94..247);
- every bf16 x (65,536 patterns, incl. +-0, denormals, max, inf/NaN) at every reachable scale (10,092,544 cases) ->
  identical e4m3 bytes old vs new; identical to torch's c10 e4m3fn cast on the 7,066,866 in-domain cases;
- packed words for every scale in every position + 1e6 random 4-tuples;
- 1,152 whole rows (random, per-row and per-element dynamic range 2^-130..2^125, signed zeros, ties, exact 448*2^k,
  bf16 max, bf16 denormals, RNE midpoints; truncated and RN-rounded bf16) -> identical bytes / scale bits / words.
The same quant is checked on the device by `device_test.py` Q1/Q2 (exhaustive over finite bf16 x at each reachable
scale and every finite absmax) against the live CUDA kernel, which also settles the PTX lowering
(`tools/compile_check.py` greps the PTX: `div.rn.f32`, no `.ftz`, no `div.full`/`div.approx`, no `fma`, no `.NaN`
min/max, the `0f2EDBE6FF` and `0f46800000` constants, no `cvt` to e4m3).

## 2. Plan + scatter (integer and copy work only)

| output | old | new | equal because |
|---|---|---|---|
| counts / 32-aligned starts | plan kernel, 1 program: `tl.histogram(where(valid, ids, E), 512)`, `aligned = (cnt + 31) // 32 * 32`, `start = cumsum(aligned) - aligned` | every program computes the same three lines (SLOTS = max(64, next_pow2(6M)) instead of 1024; masked slots read as -1 -> bin E in both) | same integers |
| slot of route (t, k), expert e | `dest = atomic_add(cursor + e, 1)`: start[e] + arrival rank (nondeterministic) | `dest = start[e] + #{s' < 6t + k : ids[s'] == e}` (deterministic) | both are a bijection of e's routes onto `[start[e], start[e] + cnt[e])` |
| `grouped_layout` | plan fills -1 on all M_sum rows; scatter writes e on live rows | live rows written by their route; padding rows `[start+cnt, start+aligned)` by e's first route; tail `[total, M_sum)` strided over programs (64 rows per step) | same array (live rows = e, every other row -1); every row written by exactly one program (no write-write race), checked |
| `sfp` | plan zeroes all M_sum rows x 10 packs; scatter writes live rows | live rows: packed words; dead rows: 0 by the row's single owner | same array |
| `aq_out` | live rows = copy of x_q[t]; dead rows never written | live rows = quantized x[t] (section 1); dead rows never written | live rows bitwise equal; dead rows hold the same workspace contents as before (no kernel writes them in either path) and FC1 skips them (layout -1, scale word 0) |
| `inv_perm` | dest for live routes; dead routes (id -1) unwritten (`torch.empty`) | same | gather skips id -1 |
| `expert_ends` | cursor after the scatter = start + cnt for all 384 experts | program 0 stores start + cnt for all 384 experts | same integers (incl. experts with count 0: start) |

Why the different in-segment row order cannot change any output bit: FC1 and FC2 compute every output row from its
own A row, its own scale words and the expert's weights with a K-order that does not depend on the row's position
(the deal tile map depends only on the per-expert counts, which are identical); LIVE_ACT SiLU uses only `expert_ends`;
the gather reads through `inv_perm` in fixed top-6 order. Today's order is already the nondeterministic atomic arrival
order and the G1 digest has been stable across all serving runs, so production already relies on this. CPU evidence:
`tests/test_plan_emul.py` (690 cases: M = 1..96, 128, 170 x random / ROUTE_LIVE dead rows / same experts / all
distinct / all dead / one live / hot experts, plus 4 x 40 random atomic orders: layout, ends, dead-row words, every
live route's row and words via inv_perm, single ownership of every row, and a row-independent FC1/FC2 stand-in with
the fixed top-6 gather all identical). Device evidence: `device_test.py` P (same on the GPU against the live kernels)
and F (the production FC1 kernel's output rows through inv_perm, bitwise).

## 3. Ordering, PDL and races

- New kernel is launched with `launch_pdl=True` exactly like the top-k kernel before it. It executes
  `griddepcontrol.wait` before its first global load or store (checked in the PTX), so it reads `topk_ids` and `x`
  only after the top-k grid has completed and flushed; the top-k kernel itself waited on the router splitKreduce, and
  `x` is final before the router GEMM starts. Nothing is written before the wait.
- No atomics; no buffer is both read and written by different programs: programs read `x`, `topk_ids` and write
  disjoint rows of `aq_out` / `sfp` / `layout` (single ownership shown above), `inv_perm[t, k]` (one program each),
  `ends` (program 0). `fused_permute` refuses (RuntimeError) if `x` and `aq_out` overlap in memory.
- Lever 2 (`DSV41_MOE_FC1_FRONT=pdl|pfN`): `griddepcontrol.launch_dependents` only lets the already-PDL FC1 grid be
  scheduled earlier. FC1's first statement after descriptor prefetch / barrier init is
  `cudaGridDependencySynchronize()` (deal tree `sm120_fp8_fp4_gemm_1d1d.cuh:213-238`), which waits for full completion
  and memory flush of this grid before it reads `grouped_layout`, rows or scales. `prefetch.global.L2` reads weight /
  weight-scale bytes into L2 and changes no value. MMA order inside FC1 is untouched.
- CUDA graphs: per call the fused path allocates 4 tensors (sfp, layout, inv_perm, ends) where mfront allocated 4 (sfp,
  layout, inv_perm, cursor) and the quant 2 more (x_q, x_s): no new allocation kinds, no host sync, no streams/events.
  Triton specialises SLOTS per M bucket (64..1024); all are compiled in eager warm-up before capture.
- Fallback: every call that is not eligible (M > 96, align != 32, mfront off, another geometry, static quant constants
  not UE8M0 e4m3fn +-448) runs `_quantize_input(x, quant_config)` in `apply()` - the exact call `prepare()` makes
  today, with no kernel between them - and then today's code. Flag off: nothing imported, no property installed,
  `prepare()` quantizes as before (tests/test_install.py strips the tagged lines back to the base bytes).

**Startup precompile (fix round).** At the first eligible call, `_ready()` runs `JITFunction.warmup` for the 5 SLOTS
buckets through the same `_launch()` that builds the real launch's arguments (tested: identical constexprs, options,
int args and pointer dtypes). Warmup compiles and caches; it launches nothing and touches no memory, so it changes no
output. If it raises, the fused path is off for the whole process and every call runs today's prepare() quant +
mfront chain (tested), which is the flag-off arithmetic.

**G18 race check on device (fix round).** device_test case F now launches the production FC1 (PDL on, asserted)
directly behind the fused kernel for FC1_FRONT 0 / pdl / pf2, eagerly and as a captured [top-k -> fused -> FC1]
graph replayed 50 times over poisoned buffers, and compares every live route's FC1 output row with the live chain's
row bitwise. An FC1 that read `grouped_layout`, rows or scales before the fused grid completed would see the poison.

## 4. Not changed

Verify width (AV pin), drafter MoE (E=128, top-3: `expects_unquantized_inputs` is False for it), top-k, FC1/FC2
kernels and their tactics, SiLU-quant, the gather, the shared expert, every reduction.
