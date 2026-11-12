# r2-skinny: exactness proof (op-order map, old vs new)

Live kernel: `~/dsv41-r2/skinny_mx/skinny_mx.cu` (sha256 `0cf2d309…3653`, copy in `base/skinny_mx.cu`), built into
`skinny_mx_ext.so` (sha256 `c054de3a…dadc`, md5 `d9f28f0d…`). It is called from `dsv41_r3_dense.py:259`
(`QkvAShard.gemm`) as `skinny_mx_a8(xq, xs, w, ws)` with split = 0 and grid = 0, which means grid = 48 SMs and
`pick_split(224, 5120, 48)` = **8**.

Live quant: `dsv41_r3_dense.py:225` → `mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)` → FlashInfer 0.7.0.post1
`mxfp8_quantize(backend="cute-dsl")` → `MXFP8QuantizeSwizzledKernel`. With M·160 < 65536 it runs the **4 threads per
scale block** variant and the small-K path (640 threads = 20 warps; 160 SF columns per row; `rows_per_block = 1`).
The source is copied in `ref/flashinfer/`.

New code: `kernel/skinny_mx_v2.cu`. Line numbers below refer to that file and to `base/skinny_mx.cu` (v1).

## 1. Shape class and indices (fixed by v1)

| quantity | v1 (skinny_mx.cu) | v2 |
|---|---|---|
| warps / block | `WARPS = 8` (:31) | `WARPS = 8` (:44) |
| K split | `pick_split(224, 5120, 48) = 8` (:205-217) | `SPLIT = 8` (:45); `supports()` (:459-464) admits only shapes where v1's own `pick_split` (copied as `v1_pick_split`, :441-452) gives 8 at v1's grid (#SMs) |
| part of warp w | `part = w % SPLIT = w`; `group = 0` (:111) | `kb = warp * KVP` (:241) |
| K slice (16-byte vectors) | `kv_per = (320 + 7) / 8 = 40`; `kb = 40·part`, `ke = kb + 40` (:113-114) | `KVP = 40` (:46) |
| rows | `row = it` for `it = blockIdx.x, +gridDim.x, …` (`ROWS_PER_IT = 1`, :116-119) | `row = blockIdx.x + i·gridDim.x`, i < MAXR (:251). Which block computes a row does **not** enter that row's op sequence, so any grid with ⌈224/grid⌉ ≤ 8 is admissible (`DSV41_SKINNY_V2_GRID`, default 0 = 48 as v1). |

## 2. Per output element y[m, row]: v1 sequence

All fp32 operations are round-to-nearest-even with denormals kept: the .so is built with `-O3 -lineinfo` and no
`--use_fast_math`. Every FMA is an explicit `fmaf`. No multiply can be contracted into an add, because each add has
no multiply operand.

1. **Activation operand** (A8 prologue, :89-104). For element k of row m:
   `x̂[m][k] = bf16_rn( float(half(e4m3 q[m][k])) * sc[m][k/32] )`, where `sc = e8m0_to_float(xs[swz(m, k/32)])`
   (:41-44: e = 0 gives 2^-127 subnormal, e = 255 gives NaN, otherwise 2^(e-127)). The product is exact (4-bit
   significand times a power of two), so the bf16 rounding only matters for subnormal results.
2. **Weight operand**: `float(half(e4m3 w[row][k]))` (exact), per-block scale `s = e8m0_to_float(ws[row][k/32])`.
3. **dot16** of one 16-byte weight vector v against x̂[m][16v .. 16v+15] (:47-61):
   `s = 0; for i in 0..7: s = fmaf(w[2i], x̂[2i], s); s = fmaf(w[2i+1], x̂[2i+1], s)`.
4. **Lane accumulator** of lane L in part p: a K walk over vectors `k = kb + L, kb + L + 32, …` while `k < ke`
   (:126-148). With 40 vectors the UNROLL = 4 loop (:127) never runs (`kb + L + 96 < kb + 40` is false), so:
   - `a = fmaf(s(kb+L), dot16(kb+L), 0.f)` on every lane;
   - `a = fmaf(s(kb+32+L), dot16(kb+32+L), a)` on lanes L < 8 only.
5. **Part sum**: butterfly `v += __shfl_xor_sync(v, o)` for o = 16, 8, 4, 2, 1 (:150-156); lane 0's value goes to
   `part_sums[p][m]` (:165).
6. **Combine** after `__syncthreads` (:167-175): `t = 0.f; t += part_sums[q][m]` for q = 0..7, then
   `y = __float2bfloat16(t)` (round to nearest even).

## 3. Per output element: v2 sequence (A8 entry `skinny_v2_a8`, lever G10)

| step | v2 code | identical to v1 because |
|---|---|---|
| 1 activation | `deq8` (:76-87) is v1's loop body (:96-101) for 8 values; A8 prologue :274-301 uses the same flat index `i = m·kv + k`, the same `swz_off` (copied, :70-73), the same `e8m0_to_float` (copied, :65-68) and the same smem layout (`smem[2i]`, `smem[2i+1]`, :298-299). Each warp writes only vectors [kb, kb+40) of every m and reads only those (:347, :357). | same function of the same bytes; only the set of threads and the order in which smem cells are written change. `__syncwarp` (:335) orders the warp's own writes before its reads. |
| 2 weight | `cvt_w16` (:90-99) runs v1's conversion once per vector, not once per (vector, m) | conversion is exact and deterministic; hoisting changes nothing |
| 3 dot16 | `dot16_cvt` (:102-113): `s = 0; i = 0..7: fmaf(wf[2i], x.x, s); fmaf(wf[2i+1], x.y, s)` | same operands, same fmaf chain, same order |
| 4a first vector | :337-348: `acc[i][m] = fmaf(s0, dot16(kb+L), 0.f)`, s0 from `ws[row][(kb+L)>>1]` (:257) | v1 :143-147, first tail iteration |
| 4b second vector | dot computed on lane `(i % 4)·8 + L` in pass i/4 (:349-357, weights loaded at :261-268): `dx = dot16(kb+32+L)`, a self-contained fmaf chain starting at 0. It is moved to lane L with `__shfl_sync` (:367), an exact 32-bit move, and lane L < 8 applies `acc = fmaf(s1, dx, acc)` with s1 = `ws[row][(kb+32+L)>>1]` loaded by lane L itself (:258, :368). | dot16 does not depend on acc, so where it runs is free. The fmaf that consumes it runs on the owning lane, with the same operands in the same order as v1 :143-147 (second tail iteration). |
| 5 part sum | :372-382: the same butterfly o = 16..1 on `acc[i][m]`; lane 0 stores `part_sums[i][warp][m]` | same adds in the same tree. (A butterfly also leaves every lane with bit-identical sums, because each step adds a commutative pair.) |
| 6 combine | :383-395, after **one** `__syncthreads` for all rows: `s = 0.f; s += part_sums[i][q][m]`, q = 0..7, `__float2bfloat16(s)` | same order and the same rounding. One thread per (row, m) instead of lane 0 of warp 0. |

Nothing else differs. There are no atomics. No value is read before its writer is ordered before the read: smem
x is warp-private behind `__syncwarp`, and `part_sums` sits behind `__syncthreads`. Rows ≥ N compute on zero
weights and are never stored.

**PDL (`DSV41_SKINNY_V2=pdl`).** Only weights and weight scales are read before `griddepcontrol.wait` (:245-271).
These are static parameters, written once at load. x, xq and xs are read only after the wait, and y is written
only after it. Launched without the attribute (`=1`), `griddepcontrol.wait` is a no-op.

## 4. QF entry `skinny_v2_q` (lever G09 part D2): the fused quant

The live path produces (q, e) bytes with the FlashInfer kernel. The v1 prologue then turns them into x̂ (step 1).
The QF prologue (:302-334) produces the same (q, e) bytes in registers, then applies the same `deq8` with
`e8m0_to_float(e)`. The fp8 byte order matches: `cvt.rn.satfinite.e4m3x2.f32 d, f1, f0` puts element 0 in the low
byte, which is the byte `__nv_cvt_fp8x2_to_halfraw2` reads as `.x`, exactly as for the stored uint4.

The FlashInfer op sequence for one 32-wide block, as CuTe-DSL PTX, and its copy in v2:

| FlashInfer (ref/flashinfer/…) | v2 |
|---|---|
| 4 consecutive threads × 8 bf16 (`ld_global_v4_u32`), `elem = sf_col·32 + thread_in_unit·8` | 4 consecutive lanes (`t>>2` = block, `t&3` = eighth), :310-318 |
| `bfloat2_max_abs_4`: `and.b32 x, 0x7FFF7FFF` ×4, `max.bf16x2` (0,1), (2,3), then (01,23) (fp4_common.py:710-738, utils :645-656) | `fi_bf2_abs`, `fi_bf2_max` in the same tree (:120-129, :207-208) |
| `bfloat2_hmax_reduce_to_f32` (utils :121-153) | `fi_bf2_hmax_to_f32`, same PTX (:130-137) |
| `reduce_max_4threads`: shuffle bfly 1 then 2, `max.f32` (utils :527-538, fp4_common :514) | :211-214 |
| `global_max * Float32(1/448)` (mxfp8_quantize.py:664-666), a DSL fp32 multiply | `fi_mul_amax` (:145-153), constant bits `0x3b124925` = fp32(1/448). The FTZ mode is selectable (see §5). |
| `float_to_ue8m0_fast` (utils :156-203) | same PTX (:154-175) |
| `ue8m0_to_inv_scale_fast` (utils :206-245) | same PTX (:176-187) |
| `bfloat2_to_fp8x2_scaled`: `mul.f32`, clamp `min/max ±448`, `cvt.rn.satfinite.e4m3x2.f32 d, f1, f0` (utils :291-334) | same PTX (:188-200) |
| `pack_fp8x8_to_u64`, `st_global_u64` (byte i = element i) | `q.x`, `q.y` packing (:219-221) |
| scale byte `scale_ue8m0_u32.to(Uint8)` at `swizzled_128x4(row, col)` | `e`, used directly. v1 would read the same byte back from the swizzled offset. |

max is exact, so the amax does not depend on the reduction tree at all. NaN handling is the same too, because the
same PTX `max` ops are used: a non-NaN operand wins, so a NaN is ignored unless the whole block is NaN, and then
ue8m0 = 254. Inf gives ue8m0 = 254 and inv = 0, so x·0 = NaN, which is then clamped to 448 by `min.f32`. These
cases follow from running the same instructions, not from a separate argument.

## 5. The one ambiguity: FTZ of `amax * (1/448)`

The DSL source does not show whether this multiply is lowered as `mul.rn.f32` or as `mul.rn.ftz.f32`. The two
differ only when the product is an fp32 subnormal above 2^-127, that is amax ∈ (448·2^-127, 448·2^-126), roughly
(2^-118.2, 2^-117.2). In that band FTZ gives ue8m0 = 0 and non-FTZ gives 1.

The CPU emulation shows that the band is the only difference (`[PASS] FTZ-mode scale differences confined to the
amax*(1/448) subnormal band`). v2 has both forms (`ftz` argument). Which one serves is never assumed:
- **Serving:** `dsv41_r2_skinny._pick_ftz` quantizes 256 hostile rows that include this band with both modes. It
  keeps the first mode whose codes and scales are byte-identical to the live quant. If neither mode matches,
  QFUSE stays off.
- **Device test:** section Q checks both modes exhaustively and requires the self-check's pick to be a mode that
  matches exhaustively.

Real activations never reach the band (a whole 32-block with amax ≈ 1e-36). The point is to keep the proof
complete. Optional: `cuobjdump -sass` of the CuTe-DSL kernel would settle the question statically. The kernel is
JIT-compiled in-process, and no cubin of it was found in `~/.cache` on spark-one.

## 6. Proof obligations and where they are discharged

| obligation | CPU (done, see TEST-RESULTS.txt) | device (`device_test.py`, run by the lead) |
|---|---|---|
| v2 index mapping (slices, second-vector lanes and shuffle sources, rows per block, grids) gives v1's fp32 sums | `tests/test_emulation.py`: lane-level simulators of v1 and v2 over M = 1..8, random/hostile weights and activations, grids 48 and 28 (MAXR 8), compared on **fp32 pre-rounding sums and bf16 outputs** | section G: M = 1..8 × 3+ weight sets × 4-5 activation sets × 8 grids × pdl on/off |
| fused quant == live quant | emulator: fused-slice mapping == FlashInfer 4T structure (both FTZ modes), codes == torch e4m3 cast | section Q: exhaustive, every bf16 value × every reachable scale byte, every bf16 amax, NaN/Inf, both FTZ modes |
| harness detects a reorder | negative controls: wrong second-vector mapping, and SPLIT 4 instead of 8, both MISMATCH | section X: one flipped weight bit must MISMATCH |
| PDL / graph capture | n/a | section C: graph replays of [quant→v1], [quant→v2 pdl], [producer→v2_q pdl] == eager v1 |
| serving-time guard | routing test: an M whose v2 output is wrong stays on v1 | the self-check runs at first eager build on the layer's real shard weights, for every M |
