// skinny_mx_v2.cu (climb r2-skinny, 2026-11-08): latency-restructured qkv_a shard GEMV for GB10 (sm_121),
// bit-identical to the live skinny_mx_ext.so (skinny_mx.cu sha256 0cf2d309..., .so md5 d9f28f0d...).
//
//   y[M, N] (bf16) = x[M, K] @ (w[N, K] fp8-e4m3 * 2^(ws[N, K/32] - 127))^T, fp32 accumulation.
//
// Scope: the shape class the live kernel runs for the qkv_a shard: N rows (224), K = SPLIT * KVP * 16 = 5120,
// with v1's pick_split(N, K, #SMs) == 8, so ROWS_PER_IT == 1, warp w == K part w, kv_per == KVP == 40 vectors.
// The host refuses (supports() == false) every other shape; the caller then keeps the live kernel.
//
// Per output (row, m) the floating-point operations are exactly v1's (see EXACTNESS.md):
//   lane L of part p:  a_L = fmaf(s0, dot16(w[row][kb+L], x[m]), 0.f)                      (all lanes)
//                      a_L = fmaf(s1, dot16(w[row][kb+32+L], x[m]), a_L)                    (L < E = 8)
//     dot16(v, x) = s=0; for i in 0..7: s = fmaf(w_2i, x_2i, s); s = fmaf(w_2i+1, x_2i+1, s)
//   part sum      : butterfly v += shfl_xor(v, o), o = 16, 8, 4, 2, 1, value of lane 0
//   output        : t = 0.f; t += part[q] for q = 0..7; y = bf16_rn(t)
// What changes is only WHEN and WHERE each operation runs:
//   * every weight vector of every row of the block is loaded up front (v1: 2 dependent loads per row,
//     4-5 rows in sequence), before griddepcontrol.wait, so with DSV41_SKINNY_V2=pdl the weight stream
//     overlaps the producer kernel's tail;
//   * each warp builds only its own K slice of the activations (warp-private smem, __syncwarp) instead of a
//     block-wide prologue + __syncthreads;
//   * the fp8 -> float weight conversion is done once per vector instead of once per (vector, m);
//   * the 8 second-vector dots of each row (v1: lanes 0..7 only, a second serial step per row) are computed
//     by all 32 lanes, 4 rows per pass, and handed to the owning lane with __shfl_sync (exact data move);
//     the owning lane then applies the same fmaf it applies in v1;
//   * all rows' part sums are stored, then ONE __syncthreads, then the per-(row, m) combines run in parallel
//     (v1: two __syncthreads and a serial combine per row).
// QF (fused quant, DSV41_SKINNY_QFUSE): the warp quantises its slice of the bf16 x with FlashInfer's CuTe-DSL
// MXFP8QuantizeSwizzled op sequence (bf16x2 abs/max tree over 4 lanes x 8 values, two xor-shuffle max steps,
// amax * fp32(1/448), float_to_ue8m0_fast, ue8m0_to_inv_scale_fast, x * inv, clamp +-448,
// cvt.rn.satfinite.e4m3x2) copied as PTX, then dequantises exactly as v1's A8 prologue does from the stored
// (e4m3, ue8m0) bytes. The one instruction whose FTZ mode is not visible in the DSL source (amax * 1/448) is
// a runtime switch (ftz) that the Python side pins by a bitwise self-check against the live quant kernel.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr int WARPS = 8;            // 256 threads, as v1
constexpr int SPLIT = 8;            // v1's pick_split for (224, 5120, 48); == WARPS, so ROWS_PER_IT == 1
constexpr int KVP = 40;             // v1 kv_per = (K/16 + SPLIT - 1) / SPLIT for K = 5120
constexpr int E = KVP - 32;         // v1 lanes 0..E-1 take a second vector (kb + 32 + lane)
constexpr int RPP = 32 / E;         // rows whose E second vectors fit one 32-lane pass
constexpr int BPP = KVP / 2;        // 32-wide scale blocks per part slice
constexpr unsigned FULL = 0xffffffffu;
constexpr uint32_t INV448_BITS = 0x3b124925u;  // fp32(1.0 / 448.0) == FlashInfer INV_FLOAT8_E4M3_MAX
static_assert(SPLIT == WARPS, "v2 assumes one row per block iteration (v1 ROWS_PER_IT == 1)");
static_assert(E > 0 && E <= 32 && 32 % E == 0, "second-vector lanes must tile a warp");
static_assert(KVP % 2 == 0, "a part slice must hold whole 32-wide scale blocks");

// ------------------------------------------------------------------ copied verbatim from skinny_mx.cu (v1)
__device__ __forceinline__ uint4 ld_stream(const uint4* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

__device__ __forceinline__ float e8m0_to_float(uint32_t e) {
  // 2^(e-127); e = 0 is 2^-127 (subnormal), e = 255 is NaN per OCP MX.
  return e == 0 ? 5.877471754111438e-39f : (e == 255 ? __int_as_float(0x7fc00000) : __int_as_float(e << 23));
}

__device__ __forceinline__ int swz_off(int m, int kb, int k_tiles) {
  const int mt = m >> 7, r = m & 127, i4 = r >> 5, i32 = r & 31, kt = kb >> 2, k4 = kb & 3;
  return ((((mt * k_tiles + kt) * 32 + i32) * 4 + i4) * 4) + k4;
}

// v1 A8 prologue body for 8 e4m3 values (4 fp8x2 pairs, little-endian in q) with scale sc.
__device__ __forceinline__ uint4 deq8(uint2 q, float sc) {
  uint4 o;
  __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(&o);
  const __nv_fp8x2_storage_t* qp = reinterpret_cast<const __nv_fp8x2_storage_t*>(&q);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2(qp[j], __NV_E4M3);
    float2 f = __half22float2(*reinterpret_cast<__half2*>(&hr));
    op[j] = __floats2bfloat162_rn(f.x * sc, f.y * sc);  // exact: 4-bit significand * 2^e
  }
  return o;
}

// v1 dot16_fp8 split in two: the weight conversion (exact, done once per vector) ...
__device__ __forceinline__ void cvt_w16(const uint4& w, float (&wf)[16]) {
  const __nv_fp8x2_storage_t* wp = reinterpret_cast<const __nv_fp8x2_storage_t*>(&w);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2(wp[i], __NV_E4M3);
    float2 f = __half22float2(*reinterpret_cast<__half2*>(&hr));
    wf[2 * i] = f.x;
    wf[2 * i + 1] = f.y;
  }
}

// ... and v1's FMA chain, same order: s = 0; i = 0..7: fmaf(w.x, x.x, s), fmaf(w.y, x.y, s).
__device__ __forceinline__ float dot16_cvt(const float (&wf)[16], const uint4& xa, const uint4& xb) {
  const __nv_bfloat162* xpa = reinterpret_cast<const __nv_bfloat162*>(&xa);
  const __nv_bfloat162* xpb = reinterpret_cast<const __nv_bfloat162*>(&xb);
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    float2 xf = __bfloat1622float2(i < 4 ? xpa[i] : xpb[i - 4]);
    s = fmaf(wf[2 * i], xf.x, s);
    s = fmaf(wf[2 * i + 1], xf.y, s);
  }
  return s;
}

// ------------------------------------------------------------------ FlashInfer CuTe-DSL MXFP8 quant, as PTX
// Sources (flashinfer 0.7.0.post1): cute_dsl/fp4_common.py bfloat2_habs2 / bfloat2_hmax2 / fmax_f32;
// quantization/quantization_cute_dsl_utils.py bfloat2_hmax_reduce_to_f32 / float_to_ue8m0_fast /
// ue8m0_to_inv_scale_fast / bfloat2_to_fp8x2_scaled / pack_fp8x8_to_u64 / reduce_max_4threads;
// quantization/kernels/mxfp8_quantize.py MXFP8QuantizeSwizzledKernel small-K 4-threads-per-SF path.
__device__ __forceinline__ uint32_t fi_bf2_abs(uint32_t x) {
  uint32_t r;
  asm("and.b32 %0, %1, 0x7FFF7FFF;" : "=r"(r) : "r"(x));
  return r;
}
__device__ __forceinline__ uint32_t fi_bf2_max(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("max.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}
__device__ __forceinline__ float fi_bf2_hmax_to_f32(uint32_t x) {
  float r;
  asm("{\n\t.reg .b32 lo, hi;\n\t.reg .f32 f0, f1;\n\t"
      "and.b32 lo, %1, 0xFFFF;\n\tshr.b32 hi, %1, 16;\n\tshl.b32 lo, lo, 16;\n\tshl.b32 hi, hi, 16;\n\t"
      "mov.b32 f0, lo;\n\tmov.b32 f1, hi;\n\tmax.f32 %0, f0, f1;\n\t}"
      : "=f"(r) : "r"(x));
  return r;
}
__device__ __forceinline__ float fi_fmax(float a, float b) {
  float r;
  asm("max.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
// normalized_max = global_max * Float32(INV_FLOAT8_E4M3_MAX): a plain DSL multiply. ftz selects .ftz; the
// Python side picks the mode that reproduces the live kernel bit for bit (self-check), or disables QFUSE.
__device__ __forceinline__ float fi_mul_amax(float a, int ftz) {
  float r;
  const float c = __int_as_float(INV448_BITS);
  if (ftz)
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(c));
  else
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(c));
  return r;
}
__device__ __forceinline__ uint32_t fi_float_to_ue8m0_fast(float v) {
  uint32_t r;
  asm("{\n\t.reg .pred p_zero, p_has_mant, p_exp_zero, p_tiny_sub, p_ovf;\n\t"
      ".reg .u32 bits, exp_biased, mantissa, bump, result;\n\t"
      "setp.le.f32 p_zero, %1, 0f00000000;\n\t"
      "mov.b32 bits, %1;\n\t"
      "shr.b32 exp_biased, bits, 23;\n\t"
      "and.b32 exp_biased, exp_biased, 255;\n\t"
      "and.b32 mantissa, bits, 0x7FFFFF;\n\t"
      "setp.ne.u32 p_has_mant, mantissa, 0;\n\t"
      "selp.u32 bump, 1, 0, p_has_mant;\n\t"
      "setp.eq.u32 p_exp_zero, exp_biased, 0;\n\t"
      "setp.le.u32 p_tiny_sub, mantissa, 0x400000;\n\t"
      "and.pred p_tiny_sub, p_exp_zero, p_tiny_sub;\n\t"
      "@p_tiny_sub mov.u32 bump, 0;\n\t"
      "add.u32 result, exp_biased, bump;\n\t"
      "setp.gt.u32 p_ovf, result, 254;\n\t"
      "selp.u32 result, 254, result, p_ovf;\n\t"
      "selp.u32 %0, 0, result, p_zero;\n\t}"
      : "=r"(r) : "f"(v));
  return r;
}
__device__ __forceinline__ float fi_ue8m0_to_inv_scale_fast(uint32_t e) {
  float r;
  asm("{\n\t.reg .s32 new_exp;\n\t.reg .b32 float_bits;\n\t.reg .pred p_zero;\n\t"
      "setp.eq.u32 p_zero, %1, 0;\n\t"
      "sub.s32 new_exp, 254, %1;\n\t"
      "max.s32 new_exp, new_exp, 0;\n\t"
      "shl.b32 float_bits, new_exp, 23;\n\t"
      "mov.b32 %0, float_bits;\n\t"
      "@p_zero mov.b32 %0, 0;\n\t}"
      : "=f"(r) : "r"(e));
  return r;
}
__device__ __forceinline__ uint32_t fi_bf2_to_fp8x2_scaled(uint32_t bf2, float inv) {
  uint32_t r;
  asm("{\n\t.reg .b32 lo, hi;\n\t.reg .f32 f0, f1;\n\t.reg .b16 fp8_pair;\n\t"
      "and.b32 lo, %1, 0xFFFF;\n\tshr.b32 hi, %1, 16;\n\tshl.b32 lo, lo, 16;\n\tshl.b32 hi, hi, 16;\n\t"
      "mov.b32 f0, lo;\n\tmov.b32 f1, hi;\n\t"
      "mul.f32 f0, f0, %2;\n\tmul.f32 f1, f1, %2;\n\t"
      "min.f32 f0, f0, 0f43E00000;\n\tmax.f32 f0, f0, 0fC3E00000;\n\t"
      "min.f32 f1, f1, 0f43E00000;\n\tmax.f32 f1, f1, 0fC3E00000;\n\t"
      "cvt.rn.satfinite.e4m3x2.f32 fp8_pair, f1, f0;\n\t"
      "cvt.u32.u16 %0, fp8_pair;\n\t}"
      : "=r"(r) : "r"(bf2), "f"(inv));
  return r;
}

// One lane of a 4-lane group (lanes 4g..4g+3, the same 32-wide block): 8 bf16 values in v. All 32 lanes of
// the warp must call it together (FULL-mask shuffles). Returns the 8 e4m3 bytes (element order, as stored by
// FlashInfer's st_global_u64) and the block's ue8m0 byte.
__device__ __forceinline__ void fi_quant8(const uint4& v, int ftz, uint2& q, uint32_t& ue8m0) {
  // bfloat2_max_abs_4: abs0..3, max01 = max(abs0, abs1), max23 = max(abs2, abs3), max(max01, max23)
  const uint32_t mx = fi_bf2_max(fi_bf2_max(fi_bf2_abs(v.x), fi_bf2_abs(v.y)),
                                 fi_bf2_max(fi_bf2_abs(v.z), fi_bf2_abs(v.w)));
  float g = fi_bf2_hmax_to_f32(mx);
  // reduce_max_4threads: shuffle_sync_bfly offset 1 then 2, fmax_f32(val, other)
  float other = __shfl_xor_sync(FULL, g, 1);
  g = fi_fmax(g, other);
  other = __shfl_xor_sync(FULL, g, 2);
  g = fi_fmax(g, other);
  ue8m0 = fi_float_to_ue8m0_fast(fi_mul_amax(g, ftz));
  const float inv = fi_ue8m0_to_inv_scale_fast(ue8m0);
  const uint32_t p0 = fi_bf2_to_fp8x2_scaled(v.x, inv), p1 = fi_bf2_to_fp8x2_scaled(v.y, inv);
  const uint32_t p2 = fi_bf2_to_fp8x2_scaled(v.z, inv), p3 = fi_bf2_to_fp8x2_scaled(v.w, inv);
  // pack_fp8x8_to_u64: lo = (p0 & 0xFFFF) | (p1 & 0xFFFF) << 16, hi = (p2 & 0xFFFF) | (p3 & 0xFFFF) << 16
  q.x = (p0 & 0xFFFFu) | ((p1 & 0xFFFFu) << 16);
  q.y = (p2 & 0xFFFFu) | ((p3 & 0xFFFFu) << 16);
}

// ------------------------------------------------------------------ the v2 GEMV
// grid: any size G with ceil(N / G) <= MAXR (v1's partition of rows over blocks does not enter any output's
// op sequence); block b handles rows b, b + G, b + 2G, ... (v1: it = b, b + G, ...; row = it).
template <int M, int MAXR, bool QF>
__global__ void __launch_bounds__(WARPS * 32, 1)
skinny_v2_kernel(const __nv_bfloat16* __restrict__ x,   // QF: [M, K] bf16 (else nullptr)
                 const uint8_t* __restrict__ xq,        // A8: [M, K] fp8 e4m3
                 const uint8_t* __restrict__ xs,        // A8: swizzled e8m0 scales
                 const uint8_t* __restrict__ w,         // [N, K] fp8 e4m3, row-major
                 const uint8_t* __restrict__ ws,        // [N, K/32] e8m0, row-major
                 __nv_bfloat16* __restrict__ y,         // [M, ldy]
                 int N, int K, int ldy, int ftz) {
  extern __shared__ uint4 smem[];  // [M, K] bf16 (v1 layout); warp p only touches its K slice
  __shared__ float part_sums[MAXR][WARPS][M];
  constexpr int NPASS = (MAXR + RPP - 1) / RPP;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int kv = K / 16, kv8 = K / 8, ksc = K / 32;
  const int kb = warp * KVP;  // v1: part = warp % SPLIT, kb = part * kv_per, ke = kb + KVP
  const int G = gridDim.x, b0 = blockIdx.x;
  const uint4* wv = reinterpret_cast<const uint4*>(w);

  // ---- A. weight stream: every vector of every row this warp needs, issued before anything waits.
  uint4 wa[MAXR];          // first vector of row i: kb + lane (v1 tail-loop iteration 1, all lanes)
  uint32_t sa[MAXR];       // its scale byte
  uint32_t so[MAXR];       // lanes < E: scale byte of this lane's own second vector kb + 32 + lane
#pragma unroll
  for (int i = 0; i < MAXR; ++i) {
    const int row = b0 + i * G;
    wa[i] = make_uint4(0u, 0u, 0u, 0u);
    sa[i] = 0u;
    so[i] = 0u;
    if (row < N) {
      wa[i] = ld_stream(wv + (size_t)row * kv + kb + lane);
      sa[i] = __ldg(ws + (size_t)row * ksc + ((kb + lane) >> 1));
      if (lane < E) so[i] = __ldg(ws + (size_t)row * ksc + ((kb + 32 + lane) >> 1));
    }
  }
  uint4 wx[NPASS];         // pass p: second vector kb + 32 + (lane % E) of row p * RPP + lane / E
#pragma unroll
  for (int p = 0; p < NPASS; ++p) {
    const int i = p * RPP + lane / E;
    const int row = b0 + i * G;
    wx[p] = make_uint4(0u, 0u, 0u, 0u);
    if (i < MAXR && row < N) wx[p] = ld_stream(wv + (size_t)row * kv + kb + 32 + (lane % E));
  }

  // ---- wait for the producer of x / xq / xs (a no-op unless launched with programmatic serialization).
  asm volatile("griddepcontrol.wait;" ::: "memory");

  // ---- B. this warp's activation slice [kb, kb + KVP) x M, bf16, v1 smem layout; warp-private.
  if constexpr (!QF) {
    // v1 A8 prologue body per 16-fp8 vector (same deq8 arithmetic); all of the lane's loads are issued first.
    constexpr int IT = (M * KVP + 31) / 32;  // <= 10
    const int k_tiles = (K / 32 + 3) / 4;
    uint4 qv[IT];
    uint32_t sv[IT];
#pragma unroll
    for (int u = 0; u < IT; ++u) {
      const int t = u * 32 + lane;
      qv[u] = make_uint4(0u, 0u, 0u, 0u);
      sv[u] = 0u;
      if (t < M * KVP) {
        const int m = t / KVP, k = kb + (t - m * KVP);
        qv[u] = reinterpret_cast<const uint4*>(xq)[m * kv + k];  // v1's flat (m, 16-fp8 vector) index
        sv[u] = xs[swz_off(m, k >> 1, k_tiles)];
      }
    }
#pragma unroll
    for (int u = 0; u < IT; ++u) {
      const int t = u * 32 + lane;
      if (t < M * KVP) {
        const int m = t / KVP, k = kb + (t - m * KVP);
        const int i = m * kv + k;
        const float sc = e8m0_to_float(sv[u]);
        smem[2 * i] = deq8(make_uint2(qv[u].x, qv[u].y), sc);
        smem[2 * i + 1] = deq8(make_uint2(qv[u].z, qv[u].w), sc);
      }
    }
  } else {
    constexpr int TASKS = M * BPP * 4;  // 4 lanes x 8 values per 32-wide block (FlashInfer 4T/SF)
    constexpr int ITERS = (TASKS + 31) / 32;
    constexpr int CH = ITERS <= 8 ? ITERS : (ITERS + 1) / 2;  // loads in flight per lane (<= 10): 1-2 chunks
#pragma unroll 1
    for (int c0 = 0; c0 < ITERS; c0 += CH) {
      uint4 v[CH];
#pragma unroll
      for (int u = 0; u < CH; ++u) {
        const int t = (c0 + u) * 32 + lane;
        v[u] = make_uint4(0u, 0u, 0u, 0u);
        if (c0 + u < ITERS && t < TASKS) {
          const int bq = t >> 2, m = bq / BPP;
          const int elem = kb * 16 + (bq - m * BPP) * 32 + (t & 3) * 8;
          v[u] = *reinterpret_cast<const uint4*>(x + (size_t)m * K + elem);
        }
      }
#pragma unroll
      for (int u = 0; u < CH; ++u) {
        if (c0 + u < ITERS) {  // warp-uniform: every lane runs the shuffles
          const int t = (c0 + u) * 32 + lane;
          uint2 q;
          uint32_t e;
          fi_quant8(v[u], ftz, q, e);
          if (t < TASKS) {
            const int bq = t >> 2, m = bq / BPP;
            const int elem = kb * 16 + (bq - m * BPP) * 32 + (t & 3) * 8;
            smem[(size_t)m * kv8 + (elem >> 3)] = deq8(q, e8m0_to_float(e));
          }
        }
      }
    }
  }
  __syncwarp();

  // ---- C. dots. First vectors: v1 acc = fmaf(s0, dot16(w0, x), 0.f) on every lane.
  float acc[MAXR][M];
#pragma unroll
  for (int i = 0; i < MAXR; ++i) {
    float wf[16];
    cvt_w16(wa[i], wf);
    const float s = e8m0_to_float(sa[i]);
    const int k = kb + lane;
#pragma unroll
    for (int m = 0; m < M; ++m)
      acc[i][m] = fmaf(s, dot16_cvt(wf, smem[m * kv8 + 2 * k], smem[m * kv8 + 2 * k + 1]), 0.f);
  }
  // Second vectors: all 32 lanes compute dot16 for RPP rows per pass (a self-contained fmaf chain from 0) ...
  float dx[NPASS][M];
#pragma unroll
  for (int p = 0; p < NPASS; ++p) {
    float wf[16];
    cvt_w16(wx[p], wf);
    const int k = kb + 32 + (lane % E);
#pragma unroll
    for (int m = 0; m < M; ++m) dx[p][m] = dot16_cvt(wf, smem[m * kv8 + 2 * k], smem[m * kv8 + 2 * k + 1]);
  }
  // ... and lane L < E of row i takes its own dot from lane (i % RPP) * E + L of pass i / RPP, then applies
  // v1's second fmaf: acc = fmaf(s1, d1, acc).
#pragma unroll
  for (int i = 0; i < MAXR; ++i) {
    const int src = (i % RPP) * E + (lane % E);
    const float s1 = e8m0_to_float(so[i]);
#pragma unroll
    for (int m = 0; m < M; ++m) {
      const float d = __shfl_sync(FULL, dx[i / RPP][m], src);
      if (lane < E) acc[i][m] = fmaf(s1, d, acc[i][m]);
    }
  }

  // ---- D. v1 xor tree per (row, m); lane 0 publishes the part sum.
#pragma unroll
  for (int i = 0; i < MAXR; ++i) {
#pragma unroll
    for (int m = 0; m < M; ++m) {
      float v = acc[i][m];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL, v, o);
      if (lane == 0) part_sums[i][warp][m] = v;
    }
  }
  __syncthreads();

  // ---- E. v1 combine (t = 0; t += part[q], q = 0..7; bf16 rn), all (row, m) of the block in parallel.
  for (int t = threadIdx.x; t < MAXR * M; t += blockDim.x) {
    const int i = t / M, m = t - (t / M) * M;
    const int row = b0 + i * G;
    if (row < N) {
      float s = 0.f;
#pragma unroll
      for (int q = 0; q < SPLIT; ++q) s += part_sums[i][q][m];
      y[(size_t)m * ldy + row] = __float2bfloat16(s);
    }
  }
}

// ------------------------------------------------------------------ debug: the fused quant alone
// Same fi_quant8 / deq8 device code as the QF prologue, over an arbitrary [R, K] bf16 tensor (K % 32 == 0).
// Outputs e4m3 bytes [R, K], row-major ue8m0 [R, K/32] and the v1-dequantised bf16 [R, K].
__global__ void __launch_bounds__(256) quant_debug_kernel(const __nv_bfloat16* __restrict__ x,
                                                         uint8_t* __restrict__ q8, uint8_t* __restrict__ s8,
                                                         __nv_bfloat16* __restrict__ deq, long total8, int ftz) {
  const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;  // one 8-value chunk per thread
  const bool ok = t < total8;
  uint4 v = make_uint4(0u, 0u, 0u, 0u);
  if (ok) v = reinterpret_cast<const uint4*>(x)[t];
  uint2 q;
  uint32_t e;
  fi_quant8(v, ftz, q, e);
  if (ok) {
    reinterpret_cast<uint2*>(q8)[t] = q;
    if ((t & 3) == 0) s8[t >> 2] = (uint8_t)e;
    reinterpret_cast<uint4*>(deq)[t] = deq8(q, e8m0_to_float(e));
  }
}

// ------------------------------------------------------------------ host
int num_sms() {
  static int v = -1;
  if (v < 0) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev);
  }
  return v;
}

int max_smem() {
  static int v = -1;
  if (v < 0) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    v -= 2048;  // static part_sums (<= 8 * 8 * 8 * 4 B)
  }
  return v;
}

// v1 pick_split (skinny_mx.cu:205-217), forced = 0. v2 only runs where this returns SPLIT for v1's grid.
int v1_pick_split(int N, int K, int grid) {
  int best = 1;
  double best_eff = -1.0;
  for (int s = 1; s <= 8; s *= 2) {
    if (s > 1 && (K / 16) / s < 32) break;
    const long iters = (N + (WARPS / s) - 1) / (WARPS / s);
    const long waves = (iters + grid - 1) / grid;
    const double eff = (double)iters / (double)(waves * grid);
    if (eff > best_eff + 0.02) { best_eff = eff; best = s; }
  }
  return best;
}

int maxr_for(int N, int grid) {
  const int r = (N + grid - 1) / grid;
  return r <= 5 ? 5 : (r <= 8 ? 8 : 0);
}

bool supports(int64_t M, int64_t N, int64_t K, int64_t grid) {
  const int g = grid > 0 ? (int)grid : num_sms();
  return M >= 1 && M <= 8 && K == (int64_t)SPLIT * KVP * 16 && N >= 1 &&
         v1_pick_split((int)N, (int)K, num_sms()) == SPLIT && maxr_for((int)N, g) > 0 &&
         M * K * 2 <= max_smem();
}

template <int M, int MAXR, bool QF>
void launch(const __nv_bfloat16* x, const uint8_t* xq, const uint8_t* xs, const uint8_t* w, const uint8_t* ws,
            __nv_bfloat16* y, int N, int K, int ldy, int grid, bool pdl, int ftz, cudaStream_t st) {
  auto kern = skinny_v2_kernel<M, MAXR, QF>;
  static bool attr_set = false;
  if (!attr_set) {
    TORCH_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, max_smem()) == cudaSuccess,
                "skinny_mx_v2: cannot raise shared memory limit");
    attr_set = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid);
  cfg.blockDim = dim3(WARPS * 32);
  cfg.dynamicSmemBytes = (size_t)M * K * sizeof(__nv_bfloat16);
  cfg.stream = st;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = pdl ? attr : nullptr;
  cfg.numAttrs = pdl ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, x, xq, xs, w, ws, y, N, K, ldy, ftz));
}

template <int MAXR, bool QF>
void dispatch_m(int M, const __nv_bfloat16* x, const uint8_t* xq, const uint8_t* xs, const uint8_t* w,
                const uint8_t* ws, __nv_bfloat16* y, int N, int K, int ldy, int grid, bool pdl, int ftz,
                cudaStream_t st) {
  switch (M) {
    case 1: launch<1, MAXR, QF>(x, xq, xs, w, ws, y, N, K, ldy, grid, pdl, ftz, st); break;
    case 2: launch<2, MAXR, QF>(x, xq, xs, w, ws, y, N, K, ldy, grid, pdl, ftz, st); break;
    case 3: launch<3, MAXR, QF>(x, xq, xs, w, ws, y, N, K, ldy, grid, pdl, ftz, st); break;
    case 4: launch<4, MAXR, QF>(x, xq, xs, w, ws, y, N, K, ldy, grid, pdl, ftz, st); break;
    case 5: launch<5, MAXR, QF>(x, xq, xs, w, ws, y, N, K, ldy, grid, pdl, ftz, st); break;
    case 6: launch<6, MAXR, QF>(x, xq, xs, w, ws, y, N, K, ldy, grid, pdl, ftz, st); break;
    case 7: launch<7, MAXR, QF>(x, xq, xs, w, ws, y, N, K, ldy, grid, pdl, ftz, st); break;
    default: launch<8, MAXR, QF>(x, xq, xs, w, ws, y, N, K, ldy, grid, pdl, ftz, st); break;
  }
}

template <bool QF>
void dispatch(int M, const __nv_bfloat16* x, const uint8_t* xq, const uint8_t* xs, const uint8_t* w,
              const uint8_t* ws, __nv_bfloat16* y, int N, int K, int grid, bool pdl, int ftz) {
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  if (maxr_for(N, grid) == 5)
    dispatch_m<5, QF>(M, x, xq, xs, w, ws, y, N, K, N, grid, pdl, ftz, st);
  else
    dispatch_m<8, QF>(M, x, xq, xs, w, ws, y, N, K, N, grid, pdl, ftz, st);
}

void check_w(const torch::Tensor& w, const torch::Tensor& ws) {
  TORCH_CHECK(w.is_cuda() && w.dim() == 2 && w.element_size() == 1 && w.is_contiguous(),
              "skinny_mx_v2: w must be contiguous [N, K] fp8");
  TORCH_CHECK(ws.is_cuda() && ws.dim() == 2 && ws.element_size() == 1 && ws.is_contiguous(),
              "skinny_mx_v2: ws must be contiguous [N, K/32] e8m0/uint8");
  TORCH_CHECK(ws.size(0) == w.size(0) && ws.size(1) == w.size(1) / 32 && w.size(1) % 32 == 0,
              "skinny_mx_v2: scale shape");
}

// A8 (G10): same operands and call contract as v1 skinny_mx_a8 (split chosen by v1's rule == 8).
torch::Tensor skinny_v2_a8(torch::Tensor xq, torch::Tensor xs, torch::Tensor w, torch::Tensor ws, int64_t grid,
                           bool pdl) {
  check_w(w, ws);
  TORCH_CHECK(xq.is_cuda() && xq.dim() == 2 && xq.element_size() == 1 && xq.is_contiguous(), "skinny_v2_a8: xq");
  TORCH_CHECK(xs.is_cuda() && xs.element_size() == 1 && xs.is_contiguous(), "skinny_v2_a8: xs");
  const int M = (int)xq.size(0), K = (int)xq.size(1), N = (int)w.size(0);
  const int g = grid > 0 ? (int)grid : num_sms();
  TORCH_CHECK(K == w.size(1) && supports(M, N, K, g), "skinny_v2_a8: unsupported shape/M/grid");
  const int64_t need = (int64_t)((M + 127) / 128) * 128 * (((K / 32) + 3) / 4) * 4;
  TORCH_CHECK(xs.numel() >= need, "skinny_v2_a8: swizzled scale too small");
  c10::cuda::CUDAGuard guard(xq.device());
  auto y = torch::empty({M, N}, xq.options().dtype(at::kBFloat16));
  dispatch<false>(M, nullptr, (const uint8_t*)xq.data_ptr(), (const uint8_t*)xs.data_ptr(),
                  (const uint8_t*)w.data_ptr(), (const uint8_t*)ws.data_ptr(),
                  reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), N, K, g, pdl, 0);
  return y;
}

// QF (G09 D2): bf16 x in, MXFP8 quant fused into the prologue; == skinny_mx_a8(mxfp8_quantize(x)).
torch::Tensor skinny_v2_q(torch::Tensor x, torch::Tensor w, torch::Tensor ws, int64_t grid, bool pdl,
                          int64_t ftz) {
  check_w(w, ws);
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.scalar_type() == at::kBFloat16 && x.is_contiguous(),
              "skinny_v2_q: x must be contiguous [M, K] bf16");
  TORCH_CHECK(((uintptr_t)x.data_ptr() & 15) == 0, "skinny_v2_q: x must be 16-byte aligned");
  const int M = (int)x.size(0), K = (int)x.size(1), N = (int)w.size(0);
  const int g = grid > 0 ? (int)grid : num_sms();
  TORCH_CHECK(K == w.size(1) && supports(M, N, K, g), "skinny_v2_q: unsupported shape/M/grid");
  c10::cuda::CUDAGuard guard(x.device());
  auto y = torch::empty({M, N}, x.options());
  dispatch<true>(M, reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), nullptr, nullptr,
                 (const uint8_t*)w.data_ptr(), (const uint8_t*)ws.data_ptr(),
                 reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), N, K, g, pdl, ftz ? 1 : 0);
  return y;
}

std::vector<torch::Tensor> quant_debug(torch::Tensor x, int64_t ftz) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.scalar_type() == at::kBFloat16 && x.is_contiguous() &&
                  x.size(1) % 32 == 0,
              "quant_debug: x must be contiguous [R, K] bf16 with K % 32 == 0");
  c10::cuda::CUDAGuard guard(x.device());
  const int64_t R = x.size(0), K = x.size(1);
  auto q = torch::empty({R, K}, x.options().dtype(at::kByte));
  auto s = torch::empty({R, K / 32}, x.options().dtype(at::kByte));
  auto d = torch::empty({R, K}, x.options());
  const long total8 = (long)(R * K / 8);
  const long blocks = (total8 + 255) / 256;
  if (blocks > 0) {
    quant_debug_kernel<<<(unsigned)blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), (uint8_t*)q.data_ptr(), (uint8_t*)s.data_ptr(),
        reinterpret_cast<__nv_bfloat16*>(d.data_ptr()), total8, ftz ? 1 : 0);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {q, s, d};
}

// Raise the dynamic smem limit of every instantiation once, eagerly (before any CUDA-graph capture).
template <int MAXR, bool QF>
void prep_m() {
#define SKV2_PREP(MM)                                                                                         \
  TORCH_CHECK(cudaFuncSetAttribute(skinny_v2_kernel<MM, MAXR, QF>, cudaFuncAttributeMaxDynamicSharedMemorySize, \
                                   max_smem()) == cudaSuccess,                                                \
              "skinny_mx_v2: cannot raise shared memory limit");
  SKV2_PREP(1) SKV2_PREP(2) SKV2_PREP(3) SKV2_PREP(4) SKV2_PREP(5) SKV2_PREP(6) SKV2_PREP(7) SKV2_PREP(8)
#undef SKV2_PREP
}

void prepare() {
  prep_m<5, false>();
  prep_m<8, false>();
  prep_m<5, true>();
  prep_m<8, true>();
}

int64_t v1_split(int64_t N, int64_t K, int64_t grid) {
  return v1_pick_split((int)N, (int)K, grid > 0 ? (int)grid : num_sms());
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("supports", &supports, "whether v2 runs [M, K] x [N, K] bit-identically to v1", py::arg("M"), py::arg("N"),
        py::arg("K"), py::arg("grid") = 0);
  m.def("v1_split", &v1_split, py::arg("N"), py::arg("K"), py::arg("grid") = 0);
  m.def("prepare", &prepare, "set the smem attribute of every instantiation (call eagerly, before capture)");
  m.def("skinny_v2_a8", &skinny_v2_a8, "== skinny_mx_ext.skinny_mx_a8(xq, xs, w, ws), latency-restructured",
        py::arg("xq"), py::arg("xs"), py::arg("w"), py::arg("ws"), py::arg("grid") = 0, py::arg("pdl") = false);
  m.def("skinny_v2_q", &skinny_v2_q, "== skinny_mx_a8(*mxfp8_quantize(x, swizzled), w, ws), quant fused",
        py::arg("x"), py::arg("w"), py::arg("ws"), py::arg("grid") = 0, py::arg("pdl") = false,
        py::arg("ftz") = 0);
  m.def("quant_debug", &quant_debug, "fused-quant device code alone: (e4m3 bytes, row-major ue8m0, deq bf16)",
        py::arg("x"), py::arg("ftz") = 0);
}
