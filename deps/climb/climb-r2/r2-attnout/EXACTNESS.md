# r2-attnout: op-order map, old against new (lever G19, attn:L-OP)

Production (launcher fda6d05a): `DSV41_SONAR_OPROJ=fp8`, `SPLITS=32`, `MAX_M=32`. TP8 gives 8 local heads, 1 local
o-group (G=1, heads_per_group=8), head_dim 512 (nope 448, rope 64), o_lora_rank N=1024, K=4096. Every decode step
runs this chain 43 times (40 target layers + 3 DSpark layers), all on the main stream. The call order is
`deep_gemm_fp8_o_proj` (deepseek_v4/nvidia/ops/o_proj.py:66-112) → `sonar_bf16_bmm.try_project` → `grouped_bmm`
(sonar_bf16_bmm.py:121-158) → `wo_b(z)` → FlashInferCutlassMxfp8LinearKernel.apply_weights (quant + mm_mxfp8).

Sources read: the live .py files (sha256 values in KIT.json) and the live Triton 3.7.1 cache on spark-one, copied
to `ref/triton_cache/`:

- inverse RoPE `KJJWUMP6…/kernel.{ttgir,ptx}`: enable_fp_fusion=True, launch_pdl=True, num_warps 1.
- partial `22V4OHNF…`, `LXA2QRFS…`: num_warps 4, stages 2, fusion off, shared 20480 B.
- reduce `7DYTRQIR…`, `GKMPNFWQ…`, `KLVEOBFX…`: the M runtime, M==1 and M%16 specialisations.

The flashinfer quant source is in `ref/flashinfer/quantization/`.

Notation: RN = round to nearest even. f32(·) is an IEEE fp32 result. bf16(·) is cvt.rn.bf16.f32. fma is
fma.rn.f32 (a single rounding).

## 1. Inverse RoPE: element A[m, k] of the wo_a input (part `rope`)

**Old.** `fused_inv_rope_fp8_quant.kernel`, quantize=False, grid (token, head). For o[m, h, d] (bf16), with h = g·8
+ k//512 and d = k%512, it writes `out_buf[g, m, k]` (bf16). The partial kernel then loads that value as A[m, k]
through `out.transpose(0,1)` strides. Per element:

- nope lane (d < 448): `selp` returns the original bf16 bits (PTX `selp.b16 %rs24, %rs22, %rs19, %p3`). It is not
  re-rounded.
- rope lane (d ≥ 448, rl = d−448):
  - x = f32(o[m,h,d]) (exact widening).
  - p = f32(o[m,h,d^1]).
  - c = cs[pos[m], rl>>1] and s = cs[pos[m], 32 + (rl>>1)] (fp32 loads).
  - `ps = mul.f32(p, s)`. In the PTX this is `mul.f32x2`, RN. Its result feeds an fma addend, which ptxas cannot
    contract.
  - even rl: `fma.rn(x, c, ps)`.
  - odd rl: `fma.rn(x, c, neg(ps))`. The PTX has `neg.f32` and then `fma.rn.f32x2`: LLVM contracted
    `x*c ± p*s` with x*c as the fused product.
  - One bf16 RN (`cvt.rn.bf16.f32`).
- Rows m ≥ M: the partial's masked load gives bf16 +0.

**New.** `_r2_wo_a_partial`, ROPE=True. It loads o directly, with the same h, d, partner d^1, cos/sin indices and
masks, and computes:

- `ps = xp * s`. Fusion is off, so this is `mul.rn.f32`, the same RN product.
- `fma(x, c, ps)` and `fma(x, c, -ps)` (tl.fma gives fma.rn.f32; the negation is exact).
- `.to(bfloat16)` (RN).
- `where(rope2, rot, xb)`: nope lanes and masked rows keep the loaded bf16 bits.

The value is the same bf16 bit pattern for every element, so the A tile that enters `local_alloc → local_load →
tt.dot` is identical. Masked rows hold +0 in both, because rope2 includes the row mask.

**Why it matches:** the same 4 IEEE operations in the same order, on the same inputs, with the same single
rounding to bf16. Elementwise fp ops do not depend on layout or thread mapping.

**Proof status:** `tests/test_emulation.py` part A (512 tiles, hostile inputs, two G/HPG shapes, exact fp32 FMA
checked against rational arithmetic). `scripts/compile_check.py` checks the PTX op kinds. `device_test.py` and the
on-device self-check check the bits.

## 2. wo_a partial: P[split, g, m, n] (parts `rope`, `st1`, `vs`, `bn128`)

**Old.** One CTA per (64 columns, 16 rows, split). CHUNK = 4096/32 = 128, so there is one K step of 128.

- W tile: e4m3 codes q[n, k] and E8M0 e[n, k//32]. Dequant: `w = bf16(mul.rn.f32(f32(q), bits(e<<23)))`.
- Dot: `tt.dot(a[16,128] bf16, w[128,64] bf16, acc=0)`, lowered to mma v2 `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`.
  - warpsPerCTA [1,4]. Dot operand kWidth = 4 for both operands, forced by the fp8 upcast of B.
  - Per element: 8 chained MMAs over k16 blocks 0..7 in ascending order. The first one starts from acc = 0.
- Store fp32.

**New, same body.** The dot, dequant and store are unchanged in each part:

- `st1`: num_stages=1. The loop has a single iteration, so there is no software pipelining in either version.
  Only the shared-memory allocation count changes.
- `vs`: e is loaded as [4, BN] and broadcast to [128, BN]. Element (kk, n) is the same byte e[n, k//32], and the
  dequant is elementwise.
- `bn128`: the tile is 16x128 with 4 warps (warpsPerCTA [1,4], 32 columns per warp). Each output element still
  sees the same 8 m16n8k16 MMAs in the same k order with the same kWidth.

**Residual risk:** Triton could choose a different dot layout or kWidth for a different BN or stage count, and
that would permute K inside an MMA. r3-e4 recorded exactly this hazard: bf16 against fp8 operands gave
kWidth 8 against 4. `scripts/compile_check.py` asserts kWidth=4 on both operands, mma v2, the same PTX mma
instruction, the same count, and for BN=64 a dot encoding identical to the live kernel's. `device_test.py`
compares the partial buffer bitwise for every M = 1..32. The production self-check compares the real weights and
activations bitwise on the first 16 eager calls, and disables the part on any difference.

## 3. Split reduce: z[m, g, n] (part `rq`)

**Old.** `_sm120_wo_a_reduce`, tile [32 splits, 256 positions], blocked sizePerThread [1,4], threadsPerWarp
[1,32], warpsPerCTA [2,2], order [1,0]. This layout is the same in all three cached builds.

- Each thread holds column j for the 16 rows w0, w0+2, …, w0+30, where w0 is the warp row.
- The PTX shows 16 `ld.global.v4` loads at offsets i·2·M·N. The add chain is `add.f32 r125, r1, r6`, then
  `+r10`, `+r14`, and so on: left to right in register order, starting from the first element with no zero seed.
- The two warp-row partials are then combined with one `add.f32` (via `st.shared`, `bar.sync`, `ld.shared`,
  `shfl.bfly 1`). Addition is commutative, so the operand order of this last add does not matter.
- Result: `z = bf16((((p0+p2)+p4)+…+p30) + (((p1+p3)+…)+p31))`.

**New.** `_r2_reduce_quant` writes exactly that order as explicit adds: `ev = p0; od = p1; ev += p[2i]; od +=
p[2i+1]` for i = 1..15, then `ev + od`, with fusion off. LLVM does not reassociate fp adds without fast-math
flags. The kernel then applies one bf16 RN.

**Proof status:** `tests/test_emulation.py` part B. It derives the order from the layout parameters and asserts
evens/odds. On hostile data it is bit-identical, and it shows that a sequential or pairwise order differs on 2246
of 4096 columns. The device test compares the bits.

## 4. MXFP8 activation quant of z (part `rq`)

**Old.** flashinfer `MXFP8QuantizeSwizzledKernel`, cute-dsl, swizzled 128x4, alignment 32. M·32 SF blocks is below
65536, so it takes the 4-threads-per-SF path. Per row r and per 32-block b:

1. `amax`: |bf16| with a bf16x2 abs/max tree, widened to f32, then shuffle-xor fmax. Max is exact and does not
   depend on order for NaN-free input.
2. `normalized = amax * Float32(1/448)`.
3. `ue8m0 = float_to_ue8m0_fast(normalized)`, which is inline PTX.
4. `inv = ue8m0_to_inv_scale_fast(ue8m0)`, which is inline PTX and gives 2^(127−ue8m0) built from bits.
5. Per element: `mul.f32 f, f32(z), inv`, `min.f32 448`, `max.f32 −448`, `cvt.rn.satfinite.e4m3x2.f32`.
6. Scale byte at `(c%4) + (c//4)·512 + (r%32)·16 + ((r%128)//32)·4 + (r//128)·128·padded_cols`.
7. Rows M..127 get scale 0. There is no padding of the data.

**New.** The same z bits (section 3), with:

- amax computed as `tl.max(|f32(z)|)` over each 32-block. This is the same exact max.
- `normalized` multiplied by the fp32 constant built from the bit pattern of RN(1/448).
- Steps 3, 4 and 5 use flashinfer's PTX text verbatim in `tl.inline_asm_elementwise`. Step 5 is applied per lane.
  The e4m3x2 convert treats the two lanes independently, and the partner lane is +0.
- The same swizzle formula.
- A padding loop that writes scale 0 to each row in [M, 128) exactly once.

The output dtypes and shapes match flashinfer's: q [M,1024] uint8 viewed as e4m3, and sf [128·32] uint8.
`apply_weights` then takes the same (q, sf) pair it would have produced itself (`as_quantized_activation`) and
calls the same `mm_mxfp8`, through the dense-tactic pin as well. So wo_b's GEMM inputs are byte-identical, and so
is its output.

**Known corner:** step 2 when amax/448 is an fp32 denormal, which needs amax < 2^-117. If the CuTe-DSL compiles the
multiply as `mul.ftz`, old and new can differ in that block's scale. That is the only place our multiply's flags
could matter. Real attention outputs never come near it. `device_test.py` reports this case as `ftz-corner`, and
the synthetic boot self-check avoids it on purpose. Steps 3–5 are the same PTX, so they are not affected.

**Proof status:** `tests/test_emulation.py` parts C and D (bytes identical; every scale byte written exactly once
for M = 1..32). The device test and the boot self-check compare the real bytes.

## 5. What does not change

- K split, splits=32, the weights (the same `_dsv41_sonar_fp8` re-encode built by `_fp8_copy` on the first eager
  call).
- The wo_b GEMM, its tactic, the all-reduce, draft and verify widths.
- The L2-prefetch recorder: `try_project` is still called exactly once per projection.
- Every other kernel in the step.

## 6. Flag off

With `DSV41_ATTNOUT_OP` unset, 0 or off:

- `o_proj.py` evaluates `if False and …` and never imports the kit module.
- `try_project` gets one extra defaulted parameter that no live caller passes, plus one `if r2ao is not None`
  host check. The kernels, launch order, streams, allocations and graph contents are those of today.
- `tests/test_flag_off.py` checks this by AST: each function equals the live one once the gated statements are
  removed.
