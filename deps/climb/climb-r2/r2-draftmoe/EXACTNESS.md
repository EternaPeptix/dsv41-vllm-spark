# r2-draftmoe — exactness map (G17, `DSV41_DRAFT_MOE_G17`)

Scope: the DSpark drafter routed MoE (`mtp.0-2`, 128 experts, top-3, hidden 5120, per TP8 rank 288 live intermediate
columns), 3 layers per decode step. Old = today (padded FC1 N=768, stock permute). New = `fc1` (compact FC1 N=576)
and/or `perm` (fused permute). Everything downstream of the FC2 input (FC2, gather, shared-expert add, drafter head,
drafts) is untouched code operating on byte-identical inputs. File:line refer to the live files copied under
`base/` (sha256 in KIT.json) and `live-ref/`.

## 0. What changes, per stage

| stage | old kernels (per drafter layer) | new kernels | arithmetic changed? |
|---|---|---|---|
| weight load (once) | w13 param `[128, 768, 2560]`, zero-initialised; loader fills rows 0..287 (w1) and 384..671 (w3) | w13 param `[128, 576, 2560]`; loader fills rows 0..287 and 288..575 | no (copy) |
| weight-scale pack (once) | `_pack_deepgemm_mxfp4_scales` over 768 rows | same over 576 rows | no (per-row bit pack) |
| permute (`perm`) | count, zeros(fp32 scales), full(-1), `ep_scatter_1`, `ep_scatter_2`, `transpose_and_pack_fp32_into_ue8m0` (inside FC1) | `_dsv41_fp_plan_kernel`, `_dsv41_fp_scatter_kernel` | no (integer/copy + same bit formula) |
| FC1 (`fc1`) | `sm120_fp8_fp4_gemm_1d1d_impl<0,768,5120,128,32,128,32,64,128,...,7 stages,...>` | same template, `SHAPE_N=576` | no (per-element chain identical, §2) |
| SiLU-mul-quant (`fc1`) | `_silu_mul_quant_fp8_packed_kernel` N=768 | same kernel N=576, `padded_output_size=384` | no (§3) |
| FC2, gather | unchanged kernels | unchanged, byte-identical inputs | no |

## 1. Weight load and scale pack (integer/copy)

- Loader: `routed_experts.py:_load_w13` (live-ref) narrows the checkpoint to this rank's `loaded_per_rank = 2304/8 = 288`
  rows (`start = 288*rank`), then writes `w1` into `expert_data.narrow(0, 0, shard)` and `w3` into
  `narrow(0, shard, shard)` with `shard = param_rows/2`, then `_narrow_expert_data_for_padding` trims to 288 rows.
  Padded: shard 384 → gate rows 0..287, up rows 384..671, rows 288..383 and 672..767 keep the `torch.zeros` init
  (`mxfp4.py` create_weights). Compact: shard 288 → gate 0..287, up 288..575. Same for the UE8M0 scale bytes
  (`_encode_mxfp4_weight_scale` is a dtype view). w2/w2 scales: identical tensors in both (`[128, 5120, 192]`, K tail
  bytes 144..191 zero).
- Scale pack: `deepgemm_post_process_weight_scale_block` → `transform_sf_into_required_layout` packs 4 consecutive
  K-group UE8M0 bytes of ONE row into one int32 (MN-major, TMA-aligned MN = rows, 768 and 576 both multiples of 4).
  Row-local, so compact row j's words == padded row LIVE[j]'s words.
- Emulated: `tests/test_emulation.py` L (executes the live `_load_w13` source for all 8 ranks) and W. PASS.

## 2. FC1, per output element (row r of the permuted A, column c)

Live source: `live-ref/deal-base/sm120_fp8_fp4_gemm_1d1d.cuh` (byte-identical to the deal-base tree's header; the
drafter instance has `kNumGroups = 128`, so every dsv41 g1 feature — `kG1`, `kG1FC1`, `kG1Deal`, `kG1DS`, `kG1DW`,
`kG1LA`, `kG1PF`, `kG1EF` — is compile-time false: they all require `kNumGroups == 384`, header lines 148-165, 248).

Old (N=768, column `c_old = LIVE[c]`, i.e. `c` for gate, `c + 96` for up) and new (N=576, column `c`) execute:

1. `accum = 0.0f` (line 526, one fp32 register per element, `kSplitKFactor = 1` so no split-K partials).
2. For `kb = 0 .. 39` (K = 5120, `BLOCK_K = 128`; lines 352-356 / 515-521), for k-step `0..3` (`MMA_K = 32`) in
   increasing order: `mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.f32.e4m3.e2m1.f32.ue8m0`
   (`sm120_mma::fp8_fp4_mixed_mma_block_scaled`, lines 798-807 / 947-968): `D = A[r, k32] · B[c, k32]` scaled by
   `sfa[r, k/128]` (activation UE8M0 per 128) and `sfb[c, k/32]` (weight UE8M0 per 32) `+ D`. The pair loop
   (`kKStepPairs`) updates the same `d` with `a_frag[0]` then `a_frag[1]`: still K-ascending, one accumulator.
3. Epilogue (`kUseTMAStoreEpilogue`, line 1556): `cd_dtype_t(v)` = `__float2bfloat16_rn(accum)`, staged through
   swizzled SMEM, TMA store. No `kWithAccumulation`, identity epilogue, no scaling after the MMA chain.

Inputs to the chain for (r, c): the A row (same permuted FP8 bytes, §5), the A scale bytes (same), the B row bytes
`w13[e, c_old]` == `w13c[e, c]` (§1), the B scale bytes (same, §1). The chain itself (instruction, K order, accumulator
init, rounding) does not depend on `SHAPE_N`; `SHAPE_N` only sets `num_n_blocks` (12 → 9) and therefore which CTA,
warp and lane hold column c (BLOCK_N = 64: old tile 4 holds gate 256..287 + pad, new tile 4 holds gate 256..287 + up
0..31). An `m16n8k32` MMA computes each `D[i][j]` from row i of A and column j of B only: no cross-column term. So
every compact output equals the padded output at `LIVE[c]`, bit for bit.

Old pad columns (`c_old` in 288..383, 672..767): B bytes 0 (+0 in e2m1) and scale bytes 0; every product is ±0 and
`D = (±0 sum) + C` with `C = +0` initially gives +0 under RN (x + (+0) = +0 for x = ±0), so the padded FC1 writes
bf16 `+0` (0x0000) there. The device test asserts this on hardware (any non-+0 pad word is a failure).

The new instance's tile configuration (BLOCK 32×64×128, 7 stages) is chosen by DeepGEMM's heuristic at run time. The
argument above holds for any configuration it could choose: tiles, stages, swizzles and epilogue sub-tiles change data
movement only; MMA_K = 32 and the K-ascending loop are fixed by the template; split-K is 1 for grouped contiguous
GEMMs (same statement as the live `dsv41_draft_m32.py:9-17`). Evidence that the configuration is in fact the same:
COMPILE-LOG.txt (the identical one-line N edit reproduces the live target compact entry's key).

FTZ/denormals: the new cubin is built with the source entry's flags (no `--use_fast_math`, `ftz` default off);
products of e4m3 × e2m1 × two power-of-two scales are exact in fp32 except where they underflow, identically in both.

## 3. SiLU-mul-FP8-quant (the FC2 input), per element

Live source: `live-ref/model_executor/layers/quantization/utils/fp8_utils.py:147-271` (`_silu_mul_quant_fp8_packed_kernel`)
and `:274-385` (wrapper). Old call: `N=768` (`hidden_size = 384`, no `padded_output_size`). New call (the existing
target-compact branch, `deep_gemm_moe.py:666-683`): `N=576` (`hidden_size = 288`), `padded_output_size=384`.
Both: `GROUP_SIZE 128`, `GROUPS_PER_ROW 3`, `PACKS_PER_ROW 1`, same `BLOCK_M`/`PACKS_PER_CTA`/grid (they depend on
M_sum and the expert ends, not on N), same `clamp_limit`, `alpha 1`, `beta 0`, same `expert_ends`/alignment
(LIVE_ACT at M_sum ≥ 192).

Per output column j in 0..383 of row r (fp32 unless stated):
- `g = f32(gate_in)`, `u = f32(up_in)`. Old: `gate_in = FC1[r, j]`, `up_in = FC1[r, 384 + j]`. New: for j < 288
  `FC1c[r, j]`, `FC1c[r, 288 + j]` (= the old values, §2); for j ≥ 288 the load is masked (`input_col_mask`,
  `other=0.0`) → `+0.0`; old loads the pad columns = bf16 `+0` → `+0.0`. Identical.
- clamp `g = min(g, L)`, `u = clamp(u, -L, L)`; `glu = g / (1 + exp(-g * 1.0))`; `y = (u + 0.0) * glu`;
  `y = f32(bf16_rn(y))`. For j ≥ 288: `-(+0)*1 = -0`, `exp(-0) = 1`, `+0 / 2 = +0`, `(+0 + 0) * +0 = +0`.
- per 128-group: `absmax = max |y|` (max is exact and order-free); group 2 holds the same 32 live values and 96 `+0`
  in both. `scale_raw = max(absmax / 448, 1e-10)`, `e = ceil(log2(scale_raw))`, `scale = exp2(e)`,
  `q = clamp(y / scale, -448, 448)` → e4m3 RN; scale byte `clamp(e + 127, 0, 255)`, 4 bytes per int32.
Hence identical FP8 bytes (incl. `0x00` for j ≥ 288) and identical packed scale words. The only way to differ would be
a `-0` in the old pad columns (would give `0x80`): excluded by §2 and asserted by the device test
(`tests/test_emulation.py` "S [control]" shows the check would catch it).

## 4. FC2 and the gather

FC2 (`<0,5120,384,...,128 groups,...>`, K = 384) reads the FC2 input (§3, identical on every live row; rows outside
expert segments are never gathered and under LIVE_ACT are left unwritten in both paths), the same w2 bytes and scales:
same per-element chain as §2 with K = 384. `ep_gather`: per token, fp32 `acc += f32(row[inv[t,k]]) * w[t,k]` for
k = 0, 1, 2 in that fixed order, `bf16_rn(acc)`; the rows are identical, `inv` points at the same row content (§5).

## 5. Permute (`perm`): integer and copy work only

Old (`deep_gemm_utils.py:862-930`): `count_expert_num_tokens` → cnt; `zeros` fp32 scales `[M_sum, 40]`;
`full(-1)` layout; `_fwd_kernel_ep_scatter_1` (start = exclusive cumsum of `round_up(cnt, 32)`, layout rows of e = e);
`_fwd_kernel_ep_scatter_2` (per token, k = 0..2: `dest = atomic_add(start[e], 1)`, `inv[t,k] = dest`, copy the 5120
FP8 bytes and the 40 fp32 scales); then inside FC1 DeepGEMM's `transpose_and_pack_fp32_into_ue8m0`
(`live-ref/deal-base/smxx_layout.cuh:175-190`): `packed = (v0 >> 23) | (v1 >> 15) | (v2 >> 7) | (v3 << 1)` per 4
scales, MN-major, `tma_aligned_mn = round_up(M_sum, 4)`; zero scales → 0.
New (`deep_gemm_utils.py:508-652`, live mfront kernels with `E=128, TOPK=3` constexpr): plan writes layout `-1` and
packed scales `0` over M_sum rows, `cnt = histogram(ids)`, the same aligned exclusive cumsum into `cursor`; scatter, one
program per (t, k): `dest = atomic_add(cursor[e], 1)`, `inv`, `layout[dest] = e`, the same 5120-byte row copy, and the
same pack formula written directly (`sfp + p * mn_aligned + dest`).
Equal: grouped layout, segment starts/ends (cursor after scatter == `expert_start_loc` after `ep_scatter_2`, the
LIVE_ACT expert ends), M_sum, per-slot row bytes and packed scale words, dead-row scale words (0). Different: only the
order of rows inside an expert segment, which is atomic-arrival order in both (already nondeterministic today). FC1,
SiLU-quant and FC2 are row-independent and the gather reads through `inv`, so the MoE output is unchanged. The eligibility
gate is the target's (`dsv41_fp_eligible`) with (E, topk) = (128, 3): M32 path only (align 32), no expert map, no psum
layout, fp32 per-128 scales, `M*3 ≤ 1024` (M ≤ 160 gives ≤ 480).
Emulated: `tests/test_emulation.py` P (M = 1..160 and hostile routings, independent random atomic orders). PASS.

## 6. Not built: FC2 K 384 → 288

Dropping FC2's three trailing all-zero k32 chunks removes MMA steps from every output element's chain. The value is the
same except that an accumulator equal to `-0` before the tail would become `+0` (and a subnormal corner). That is a
change of the per-element op sequence, which this round's rules exclude; it also needs an E=128 rebuild of the FC2C or
Athena native (both host-assert `num_groups == 384`) and is worth about 0.05% G1 (FC2C ran 0.956x at M=1 on the
target). Skipped.

## 7. Evidence

| check | where | result |
|---|---|---|
| CPU emulation, old vs new op sequences, random + hostile (max, zeros, -0 products, cancellation, denormal-range scales, ties) | tests/test_emulation.py | 14/14 PASS (TEST-RESULTS.txt) |
| patched control flow with stubbed kernels: flag off = base call log; flag on only changes the compact drafter | tests/test_plan.py | 22/22 PASS |
| flag-off AST of every patched file == base AST (flags folded to off), anchors, Athena transform, r1-pdl composition | tests/test_install.py | 37/37 PASS |
| hardware bitwise: FC1 columns, pad +0, FC2 input bytes + scale words, FC2 rows, MoE output; M32 and 128-row; eager + graph replay; real mtp.0-2 weights | device_test.py | to be run by the lead on a stopped node |
| serving shadow of the permute (`fc1,permverify`) and exact.py / G1 digest with the flag on | window | to be run by the lead |
