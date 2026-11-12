// r2-oblit (climb round 2, lever G03 stage 1): the Moet mHC boundary kernel with the DSV41_OBLIT refusal-direction
// projections folded in. DEFAULT OFF: dsv41_oblit_fold.py builds/loads this extension only when DSV41_OBLIT_FOLD is set.
//
// Base: vLLM-Moet b0dc737 tools/dsv41_sm120/mhc_overlap/mhc_post_norm_sm120.cu (Apache-2.0), live sha256
// 2f18249b4b1e523ddf84986ad33b232925f4720748a4e06014da26ee6b799965, kernel mhc_post_norm_kernel<kMode, kPDL, kAR=false>.
// Every line of the Moet arithmetic below is copied verbatim (post-mix, bf16 streams, collapse, aux, cluster exchange,
// TileLang sum of squares, rsqrt, (c * rsqrt) * w); the fold only adds two blocks, both bit-replicas of the Triton
// kernel dsv41_oblit._proj_rows (Triton 3.7.1, sm_121, BLOCK 8192, num_warps 8, H = 5120; PTX cached as
// X7VLZGHTHZWIMG5PVVMPV3OLHZNQT6HFC7YTZVQV7LK5EZI2G63Q/_proj_rows.ptx):
//
//   kPro  (prologue, FFN boundary): x is the UNPROJECTED attention output (rdma_ar result). Every CTA reads the full
//         x row, computes s = sum(x * d) in Triton's exact order and writes x' = bf16(fma(-s, d, x)) for its 1024
//         columns to x_out (out of place: the peer CTAs still read x). The post-mix then uses x' - the value the
//         Triton kernel would have stored and Moet would have loaded.
//   kEpi  (epilogue, attention and FFN boundaries): after the RMSNorm every CTA rebuilds the whole bf16 normalized row
//         from the cluster's collapsed values (the same scratch row the sum of squares reads, the same rsqrt every CTA
//         already computes, the constant weight), computes s' over it in Triton's order and stores
//         bf16(fma(-s', d, ln)) as layer_input instead of ln.
//
// Triton's op order (PTX lines 217-396, see EXACTNESS.md): thread t in [0, 256) of the row's CTA owns columns
// r * 2048 + t * 8 + e (r = 0..3, e = 0..7; columns >= 5120 are masked loads giving +0 for x and d). Its chain over
// k = r * 8 + e is   acc = x[1] * d[1] (mul.f32);  acc = fma(x[0], d[0], acc);  acc = fma(x[k], d[k], acc), k = 2..31
// (fma.rn.f32, the masked k included: fma(+0, +0, acc) maps -0 to +0). Then shfl.bfly 16, 8, 4, 2, 1 with add.f32
// (own + partner), lane 0 stores warp w's sum to smem[w], threads 0..7 load and run bfly 4, 2, 1:
//   s = ((s0 + s4) + (s2 + s6)) + ((s1 + s5) + (s3 + s7)),
// broadcast through smem; out = cvt.rn.bf16(fma.rn(-s, d, x)). No .ftz anywhere (the PTX has none).
// Moet thread ct of cluster CTA b owns columns b * 1024 + ct * 8 + e. Triton thread t = ct (warp ct / 32) owns
// blocks 0, 2, 4 at the same offset (chain A, rep 3 masked); thread t = 128 + ct (warp 4 + ct / 32) owns blocks 1 and
// 3 (chain B, reps 2 and 3 masked). So Moet warp w runs Triton warps w (chain A) and w + 4 (chain B) lane for lane.
//
// Every new op uses an explicit rounding intrinsic (__fmul_rn, __fmaf_rn, __fadd_rn), so nvcc's default
// --fmad=true cannot contract or reorder it; the masked zero FMAs use an opaque zero so they cannot be folded away.

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>
#include <unordered_map>

namespace cg = cooperative_groups;

namespace {

constexpr int kHC = 4;
constexpr int kVec = 8;         // bf16 per thread
constexpr int kTLBlock = 1024;  // TileLang's hidden_block for H % 1024 == 0
constexpr int kTLThreads = 64;  // TileLang's collapse threads (96 minus warp 0)
constexpr int kMaxH = 5120;
constexpr int kFoldH = 5120;               // the fold replicates _proj_rows' layout at H = 5120 only
constexpr int kNB = kFoldH / kTLBlock;     // 5 CTAs per token
constexpr int kTritonWarps = 8;

// ---- verbatim from mhc_post_norm_sm120.cu ----
template <int kMode>
__device__ __forceinline__ float post_mix_value(const float pm, const float xf, const float (&cm)[kHC * kHC], const int j,
                                                const float (&rf)[kHC]) {
  // cm[k * kHC + j] multiplies stream k into stream j (TileLang: cm[k, j] * residual_in[k])
  float v;
  if (kMode == 0) {
    v = pm * xf;
#pragma unroll
    for (int k = 0; k < kHC; ++k) v = fmaf(cm[k * kHC + j], rf[k], v);
  } else if (kMode == 1) {
    v = fmaf(pm, xf, cm[j] * rf[0]);
#pragma unroll
    for (int k = 1; k < kHC; ++k) v = fmaf(cm[k * kHC + j], rf[k], v);
  } else {
    v = __fmul_rn(pm, xf);
#pragma unroll
    for (int k = 0; k < kHC; ++k) v = __fadd_rn(v, __fmul_rn(cm[k * kHC + j], rf[k]));
  }
  return v;
}

__device__ __forceinline__ void load_coefficients(const float* __restrict__ pm_p, const float* __restrict__ pre_p,
                                                  const float* __restrict__ cm_p, float (&pm)[kHC], float (&pre)[kHC],
                                                  float (&cm)[kHC * kHC]) {
  const float4 a = __ldg(reinterpret_cast<const float4*>(pm_p));
  const float4 b = __ldg(reinterpret_cast<const float4*>(pre_p));
  pm[0] = a.x; pm[1] = a.y; pm[2] = a.z; pm[3] = a.w;
  pre[0] = b.x; pre[1] = b.y; pre[2] = b.z; pre[3] = b.w;
#pragma unroll
  for (int q = 0; q < kHC; ++q) {
    const float4 c = __ldg(reinterpret_cast<const float4*>(cm_p) + q);
    cm[4 * q + 0] = c.x; cm[4 * q + 1] = c.y; cm[4 * q + 2] = c.z; cm[4 * q + 3] = c.w;
  }
}

constexpr int kBlockThreads = kTLBlock / kVec;  // 128
// ---- end verbatim ----

// ------------------------------------------------------------------ the _proj_rows replica (r2-oblit)
__device__ __forceinline__ float opaque_zero() {
  float z;
  asm volatile("mov.f32 %0, 0f00000000;" : "=f"(z));
  return z;
}

__device__ __forceinline__ void unpack8(const uint4 u, float (&f)[kVec]) {
  const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
  for (int q = 0; q < kVec / 2; ++q) {
    const float2 t = __bfloat1622float2(p[q]);
    f[2 * q] = t.x;
    f[2 * q + 1] = t.y;
  }
}

__device__ __forceinline__ void load_d8(const float* __restrict__ d, const int col, float (&f)[kVec]) {
  const float4 a = __ldg(reinterpret_cast<const float4*>(d + col));
  const float4 b = __ldg(reinterpret_cast<const float4*>(d + col + 4));
  f[0] = a.x; f[1] = a.y; f[2] = a.z; f[3] = a.w;
  f[4] = b.x; f[5] = b.y; f[6] = b.z; f[7] = b.w;
}

// Triton thread chain over 32 elements: v[r][e] / dv[r][e] for the nreal real reps (blocks), then 32 - 8 * nreal masked
// elements. k = 1 first (mul), k = 0 (fma), then k = 2..31 (fma).
template <int kReal>
__device__ __forceinline__ float triton_chain(const float (&v)[kReal][kVec], const float (&dv)[kReal][kVec]) {
  float acc = __fmul_rn(v[0][1], dv[0][1]);
  acc = __fmaf_rn(v[0][0], dv[0][0], acc);
#pragma unroll
  for (int e = 2; e < kVec; ++e) acc = __fmaf_rn(v[0][e], dv[0][e], acc);
#pragma unroll
  for (int r = 1; r < kReal; ++r) {
#pragma unroll
    for (int e = 0; e < kVec; ++e) acc = __fmaf_rn(v[r][e], dv[r][e], acc);
  }
  const float z = opaque_zero();
#pragma unroll
  for (int k = kReal * kVec; k < 4 * kVec; ++k) acc = __fmaf_rn(z, z, acc);
  return acc;
}

__device__ __forceinline__ float warp_bfly_sum(float v) {
  const unsigned full = 0xffffffffu;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v = __fadd_rn(v, __shfl_xor_sync(full, v, off));
  return v;
}

// s for the row whose five 1024-column blocks this thread holds at its own offset ct * 8 (vals[bb], dv[bb]).
// s_red8: 8 floats of shared memory (Triton's per-warp partials); one __syncthreads inside.
__device__ __forceinline__ float triton_row_dot(const float (&vals)[kNB][kVec], const float (&dv)[kNB][kVec],
                                               float* s_red8) {
  float va[3][kVec], da[3][kVec], vb[2][kVec], db[2][kVec];
#pragma unroll
  for (int e = 0; e < kVec; ++e) {
    va[0][e] = vals[0][e]; da[0][e] = dv[0][e];   // Triton rep 0 of thread ct: block 0
    va[1][e] = vals[2][e]; da[1][e] = dv[2][e];   // rep 1: column 2048 + ct * 8 -> block 2
    va[2][e] = vals[4][e]; da[2][e] = dv[4][e];   // rep 2: column 4096 + ct * 8 -> block 4
    vb[0][e] = vals[1][e]; db[0][e] = dv[1][e];   // thread 128 + ct, rep 0: column 1024 + ct * 8 -> block 1
    vb[1][e] = vals[3][e]; db[1][e] = dv[3][e];   // rep 1: column 3072 + ct * 8 -> block 3
  }
  const float a = warp_bfly_sum(triton_chain<3>(va, da));  // Triton warp ct / 32
  const float b = warp_bfly_sum(triton_chain<2>(vb, db));  // Triton warp 4 + ct / 32
  const int w = threadIdx.x >> 5;
  if ((threadIdx.x & 31) == 0) {
    s_red8[w] = a;
    s_red8[w + 4] = b;
  }
  __syncthreads();
  const float s0 = s_red8[0], s1 = s_red8[1], s2 = s_red8[2], s3 = s_red8[3];
  const float s4 = s_red8[4], s5 = s_red8[5], s6 = s_red8[6], s7 = s_red8[7];
  // threads 0..7: bfly 4 (own + partner), 2, 1 -> lane 0's tree (addition is commutative, every lane has this value)
  return __fadd_rn(__fadd_rn(__fadd_rn(s0, s4), __fadd_rn(s2, s6)), __fadd_rn(__fadd_rn(s1, s5), __fadd_rn(s3, s7)));
}

__device__ __forceinline__ uint4 triton_project8(const float s, const float (&v)[kVec], const float (&dv)[kVec]) {
  const float ns = -s;  // neg.f32
  uint4 out;
  __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(&out);
#pragma unroll
  for (int q = 0; q < kVec / 2; ++q) {
    o2[q] = __floats2bfloat162_rn(__fmaf_rn(ns, dv[2 * q], v[2 * q]), __fmaf_rn(ns, dv[2 * q + 1], v[2 * q + 1]));
  }
  return out;
}

// One token per cluster of 5 CTAs x 128 threads, exactly mhc_post_norm_kernel<kMode, kPDL, false> plus the fold.
template <int kMode, bool kPDL, bool kPro, bool kEpi>
__global__ void __launch_bounds__(kBlockThreads) mhc_post_norm_fold_kernel(
    const __nv_bfloat16* __restrict__ x,            // [T, H] (kPro: the unprojected sublayer output)
    const __nv_bfloat16* __restrict__ residual_in,  // [T, kHC, H]
    const float* __restrict__ post_layer_mix,       // [T, kHC]
    const float* __restrict__ comb_res_mix,         // [T, kHC, kHC]
    const float* __restrict__ pre_mix_in,           // [T, kHC]
    const __nv_bfloat16* __restrict__ norm_w,       // [H]
    __nv_bfloat16* __restrict__ residual_out,       // [T, kHC, H]
    __nv_bfloat16* __restrict__ layer_input,        // [T, H] (kEpi: projected)
    __nv_bfloat16* __restrict__ aux,                // [T, H] or nullptr
    const int H, const float norm_eps,
    const float* __restrict__ dvec,                 // [H] fp32 unit direction (constant, created before capture)
    __nv_bfloat16* __restrict__ x_out,              // kPro: [T, H] the projected x (the side stream's mhc_proj reads it)
    __nv_bfloat16* __restrict__ scratch) {          // [T, H] bf16 workspace: every block's collapsed values
  __shared__ float s_red[kTLThreads + 4];
  __shared__ float s_pro[kTritonWarps];
  __shared__ float s_epi[kTritonWarps];
  cg::cluster_group cluster = cg::this_cluster();
  const int nb = H / kTLBlock;                            // CTAs per token = cluster size (5: H == 5120 checked)
  const int b = static_cast<int>(cluster.block_rank());   // this CTA's TileLang block
  const int i = blockIdx.x / nb;                          // token
  const int ct = threadIdx.x;
  const int h0 = b * kTLBlock + ct * kVec;
  const unsigned full = 0xffffffffu;

  // constants first (never written after graph capture): the direction at this thread's column of every block and,
  // for the epilogue, the norm weight at the same columns
  float dall[kNB][kVec];
  float down[kVec];
  uint4 wall[kNB];
  if constexpr (kPro || kEpi) {
#pragma unroll
    for (int bb = 0; bb < kNB; ++bb) load_d8(dvec, bb * kTLBlock + ct * kVec, dall[bb]);
    load_d8(dvec, h0, down);
  }
  if constexpr (kEpi) {
#pragma unroll
    for (int bb = 0; bb < kNB; ++bb) wall[bb] = *reinterpret_cast<const uint4*>(norm_w + bb * kTLBlock + ct * kVec);
  }

  // All loads in flight first.  (verbatim, kAR = false)
  const __nv_bfloat16* rrow = residual_in + static_cast<int64_t>(i) * kHC * H;
  uint4 rv[kHC], wv, xv;
  float pm[kHC], pre[kHC], cm[kHC * kHC];
  uint4 xall[kNB];
  if constexpr (kPDL) {
    // the kernel may start while the one producing x still runs: only the weight before the wait
    wv = *reinterpret_cast<const uint4*>(norm_w + h0);
    asm volatile("griddepcontrol.wait;" ::: "memory");
    // a PDL-launched successor may start its own prologue now; its wait covers this grid's completion
    asm volatile("griddepcontrol.launch_dependents;");
#pragma unroll
    for (int c = 0; c < kHC; ++c) rv[c] = *reinterpret_cast<const uint4*>(rrow + c * H + h0);
    load_coefficients(post_layer_mix + i * kHC, pre_mix_in + i * kHC, comb_res_mix + i * kHC * kHC, pm, pre, cm);
    xv = *reinterpret_cast<const uint4*>(x + static_cast<int64_t>(i) * H + h0);
  } else {
#pragma unroll
    for (int c = 0; c < kHC; ++c) rv[c] = *reinterpret_cast<const uint4*>(rrow + c * H + h0);
    wv = *reinterpret_cast<const uint4*>(norm_w + h0);
    load_coefficients(post_layer_mix + i * kHC, pre_mix_in + i * kHC, comb_res_mix + i * kHC * kHC, pm, pre, cm);
    xv = *reinterpret_cast<const uint4*>(x + static_cast<int64_t>(i) * H + h0);
  }
  if constexpr (kPro) {
    // the whole unprojected row at this thread's offset in every block (L2; read-only for the whole cluster)
#pragma unroll
    for (int bb = 0; bb < kNB; ++bb) {
      xall[bb] = *reinterpret_cast<const uint4*>(x + static_cast<int64_t>(i) * H + bb * kTLBlock + ct * kVec);
    }
    float xf[kNB][kVec];
#pragma unroll
    for (int bb = 0; bb < kNB; ++bb) unpack8(xall[bb], xf[bb]);
    const float s = triton_row_dot(xf, dall, s_pro);
    float xo[kVec];
    unpack8(xv, xo);
    xv = triton_project8(s, xo, down);  // x' = what _proj_rows stores and Moet would load
    *reinterpret_cast<uint4*>(x_out + static_cast<int64_t>(i) * H + h0) = xv;
  }

  // post-mix -> bf16 streams; collapse + aux from the rounded streams  (verbatim)
  uint4 ov[kHC];
  uint4 cv;  // the eight bf16-rounded collapsed values
  float auxv[kVec];
  {
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(&xv);
    const __nv_bfloat162* r2[kHC];
#pragma unroll
    for (int c = 0; c < kHC; ++c) r2[c] = reinterpret_cast<const __nv_bfloat162*>(&rv[c]);
    __nv_bfloat162* o2[kHC];
#pragma unroll
    for (int c = 0; c < kHC; ++c) o2[c] = reinterpret_cast<__nv_bfloat162*>(&ov[c]);
    __nv_bfloat162* c2 = reinterpret_cast<__nv_bfloat162*>(&cv);
#pragma unroll
    for (int p = 0; p < kVec / 2; ++p) {
      const float2 xf = __bfloat1622float2(x2[p]);
      float rfx[kHC], rfy[kHC];
#pragma unroll
      for (int c = 0; c < kHC; ++c) {
        const float2 f = __bfloat1622float2(r2[c][p]);
        rfx[c] = f.x;
        rfy[c] = f.y;
      }
      float col_x = 0.f, col_y = 0.f, aux_x = 0.f, aux_y = 0.f;
#pragma unroll
      for (int j = 0; j < kHC; ++j) {
        const float nx = post_mix_value<kMode>(pm[j], xf.x, cm, j, rfx);
        const float ny = post_mix_value<kMode>(pm[j], xf.y, cm, j, rfy);
        const __nv_bfloat162 nb2 = __floats2bfloat162_rn(nx, ny);
        o2[j][p] = nb2;
        const float2 nf = __bfloat1622float2(nb2);
        col_x = fmaf(pre[j], nf.x, col_x);
        col_y = fmaf(pre[j], nf.y, col_y);
        aux_x = aux_x + nf.x;
        aux_y = aux_y + nf.y;
      }
      c2[p] = __floats2bfloat162_rn(col_x, col_y);
      auxv[2 * p] = aux_x;
      auxv[2 * p + 1] = aux_y;
    }
  }
  __nv_bfloat16* orow = residual_out + static_cast<int64_t>(i) * kHC * H;
#pragma unroll
  for (int c = 0; c < kHC; ++c) *reinterpret_cast<uint4*>(orow + c * H + h0) = ov[c];
  *reinterpret_cast<uint4*>(scratch + static_cast<int64_t>(i) * H + h0) = cv;  // this block's collapsed values, for every block
  if (aux != nullptr) {
    uint4 av;
    __nv_bfloat162* a2 = reinterpret_cast<__nv_bfloat162*>(&av);
#pragma unroll
    for (int p = 0; p < kVec / 2; ++p) {
      a2[p] = __floats2bfloat162_rn(auxv[2 * p] / static_cast<float>(kHC), auxv[2 * p + 1] / static_cast<float>(kHC));
    }
    *reinterpret_cast<uint4*>(aux + static_cast<int64_t>(i) * H + h0) = av;
  }
  // every block's collapsed values are in L2: the arrive's MEMBAR.ALL.GPU publishes this block's stores, the wait
  // acquires the peers'
  cluster.sync();

  // kEpi: this thread's column of every block's collapsed values (issued now, consumed after the rsqrt)
  uint4 call[kNB];
  if constexpr (kEpi) {
    const __nv_bfloat16* crow = scratch + static_cast<int64_t>(i) * H;
#pragma unroll
    for (int bb = 0; bb < kNB; ++bb) call[bb] = __ldcg(reinterpret_cast<const uint4*>(crow + bb * kTLBlock + ct * kVec));
  }

  // TileLang's sum of squares, in its order, by 64 threads  (verbatim)
  if (ct < kTLThreads) {
    float part[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) part[j] = 0.f;
    constexpr int kMaxBlocks = kMaxH / kTLBlock;  // 5
    const __nv_bfloat16* row = scratch + static_cast<int64_t>(i) * H;
    uint4 v0[kMaxBlocks], v1[kMaxBlocks];
#pragma unroll
    for (int bb = 0; bb < kMaxBlocks; ++bb) {
      if (bb < nb) {
        v0[bb] = __ldcg(reinterpret_cast<const uint4*>(row + bb * kTLBlock + ct * kVec));
        v1[bb] = __ldcg(reinterpret_cast<const uint4*>(row + bb * kTLBlock + 512 + ct * kVec));
      }
    }
#pragma unroll
    for (int bb = 0; bb < kMaxBlocks; ++bb) {
      if (bb < nb) {
        const __nv_bfloat16* e0 = reinterpret_cast<const __nv_bfloat16*>(&v0[bb]);
        const __nv_bfloat16* e1 = reinterpret_cast<const __nv_bfloat16*>(&v1[bb]);
#pragma unroll
        for (int j = 0; j < 16; ++j) {
          const float v = __bfloat162float(j < 8 ? e0[j] : e1[j - 8]);
          part[j] = fmaf(v, v, part[j]);
        }
      }
    }
    float sumsq = 0.f;
#pragma unroll
    for (int r = 0; r < 16; ++r) sumsq = sumsq + part[((r & 1) * 8) + (r >> 1)];
    s_red[ct] = sumsq;
  }
  __syncthreads();
  if (ct < kTLThreads) {
    // AllReduce<SumOp, 64>: the xor-32 step through shared memory, then xor 16..1 in the warp
    float sumsq = s_red[ct] + s_red[ct ^ 32];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) sumsq = sumsq + __shfl_xor_sync(full, sumsq, off);
    if (ct == 0) s_red[kTLThreads] = rsqrtf(sumsq / static_cast<float>(H) + norm_eps);
  }
  __syncthreads();
  const float rsqrt_norm = s_red[kTLThreads];

  // (collapsed * rsqrt) * weight  (verbatim expression)
  {
    const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&wv);
    const __nv_bfloat162* c2 = reinterpret_cast<const __nv_bfloat162*>(&cv);
    uint4 lv;
    __nv_bfloat162* l2 = reinterpret_cast<__nv_bfloat162*>(&lv);
#pragma unroll
    for (int p = 0; p < kVec / 2; ++p) {
      const float2 wf = __bfloat1622float2(w2[p]);
      const float2 cf = __bfloat1622float2(c2[p]);
      l2[p] = __floats2bfloat162_rn((cf.x * rsqrt_norm) * wf.x, (cf.y * rsqrt_norm) * wf.y);
    }
    if constexpr (kEpi) {
      // the whole normalized row at this thread's offset of every block, by the same expression from the same
      // values (the peers' scratch columns, the cluster-wide rsqrt, the constant weight) -> bitwise their ln
      float lf[kNB][kVec];
#pragma unroll
      for (int bb = 0; bb < kNB; ++bb) {
        const __nv_bfloat162* wb = reinterpret_cast<const __nv_bfloat162*>(&wall[bb]);
        const __nv_bfloat162* cb = reinterpret_cast<const __nv_bfloat162*>(&call[bb]);
#pragma unroll
        for (int p = 0; p < kVec / 2; ++p) {
          const float2 wf = __bfloat1622float2(wb[p]);
          const float2 cf = __bfloat1622float2(cb[p]);
          const float2 lf2 =
              __bfloat1622float2(__floats2bfloat162_rn((cf.x * rsqrt_norm) * wf.x, (cf.y * rsqrt_norm) * wf.y));
          lf[bb][2 * p] = lf2.x;
          lf[bb][2 * p + 1] = lf2.y;
        }
      }
      const float s2 = triton_row_dot(lf, dall, s_epi);
      float lo[kVec];
      unpack8(lv, lo);
      lv = triton_project8(s2, lo, down);
    }
    *reinterpret_cast<uint4*>(layer_input + static_cast<int64_t>(i) * H + h0) = lv;
  }
  // (no distributed-shared-memory reads: blocks exit independently)
}

}  // namespace

void mhc_post_norm_fold(const at::Tensor& x, const at::Tensor& residual_in, const at::Tensor& post_layer_mix,
                        const at::Tensor& comb_res_mix, const at::Tensor& pre_mix_in, const at::Tensor& norm_w,
                        at::Tensor residual_out, at::Tensor layer_input, const c10::optional<at::Tensor>& aux,
                        double norm_eps, int64_t mode, bool pdl, const at::Tensor& dvec, bool pro, bool epi,
                        const c10::optional<at::Tensor>& x_out) {
  TORCH_CHECK(residual_in.is_cuda() && residual_in.dim() == 3 && residual_in.scalar_type() == at::kBFloat16 &&
                  residual_in.is_contiguous(),
              "residual: bf16 [T, hc, H] contiguous");
  const int64_t T = residual_in.size(0), hc = residual_in.size(1), H = residual_in.size(2);
  TORCH_CHECK(hc == kHC, "hc_mult must be ", kHC);
  TORCH_CHECK(H == kFoldH, "the OBLIT fold replicates _proj_rows at H = ", kFoldH, " only");
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 2 && x.size(0) == T &&
                  x.size(1) == H,
              "x: bf16 [T, H]");
  TORCH_CHECK(post_layer_mix.is_cuda() && post_layer_mix.scalar_type() == at::kFloat && post_layer_mix.is_contiguous() &&
                  post_layer_mix.numel() == T * kHC,
              "post_layer_mix: fp32 [T, hc]");
  TORCH_CHECK(comb_res_mix.is_cuda() && comb_res_mix.scalar_type() == at::kFloat && comb_res_mix.is_contiguous() &&
                  comb_res_mix.numel() == T * kHC * kHC,
              "comb_res_mix: fp32 [T, hc, hc]");
  TORCH_CHECK(pre_mix_in.is_cuda() && pre_mix_in.scalar_type() == at::kFloat && pre_mix_in.is_contiguous() &&
                  pre_mix_in.numel() == T * kHC,
              "pre_mix_in: fp32 [T, hc]");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(post_layer_mix.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(comb_res_mix.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(pre_mix_in.data_ptr()) % 16 == 0,
              "coefficient tensors must be 16-byte aligned (vector loads)");
  TORCH_CHECK(norm_w.is_cuda() && norm_w.scalar_type() == at::kBFloat16 && norm_w.is_contiguous() && norm_w.numel() == H,
              "norm_weight: bf16 [H]");
  TORCH_CHECK(residual_out.is_contiguous() && residual_out.sizes() == residual_in.sizes() &&
                  residual_out.scalar_type() == at::kBFloat16,
              "residual_out: bf16 [T, hc, H]");
  TORCH_CHECK(layer_input.is_contiguous() && layer_input.numel() == T * H && layer_input.scalar_type() == at::kBFloat16,
              "layer_input: bf16 [T, H]");
  if (aux.has_value()) {
    TORCH_CHECK(aux->is_contiguous() && aux->numel() == T * H && aux->scalar_type() == at::kBFloat16, "aux: bf16 [T, H]");
  }
  TORCH_CHECK(dvec.is_cuda() && dvec.scalar_type() == at::kFloat && dvec.is_contiguous() && dvec.numel() == H &&
                  reinterpret_cast<uintptr_t>(dvec.data_ptr()) % 16 == 0,
              "d: fp32 [H] contiguous, 16-byte aligned");
  TORCH_CHECK(!pro || (x_out.has_value() && x_out->is_contiguous() && x_out->sizes() == x.sizes() &&
                       x_out->scalar_type() == at::kBFloat16 && x_out->data_ptr() != x.data_ptr()),
              "pro: x_out bf16 [T, H], out of place");
  TORCH_CHECK(pro || epi, "fold kernel without a projection: use the Moet kernel");
  if (T == 0) return;
  const c10::cuda::OptionalCUDAGuard guard(residual_in.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(residual_in.get_device()).stream();
  const int nb = static_cast<int>(H) / kTLBlock;
  // [T, H] bf16 workspace (own, not shared with the Moet extension); grown on demand - the eager warm-up
  // (dsv41_oblit_fold.prepare, 64 tokens) sizes it before any capture, so a capture sees a fixed address
  static std::unordered_map<int, at::Tensor> workspaces;
  at::Tensor& scratch = workspaces[residual_in.get_device()];
  if (!scratch.defined() || scratch.numel() < T * H) {
    scratch = at::empty({std::max<int64_t>(T, 64) * H}, residual_in.options().dtype(at::kBFloat16));
  }
  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(T * nb);
  config.blockDim = dim3(kBlockThreads);
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attr[2];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = nb;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  attr[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[1].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attr;
  config.numAttrs = pdl ? 2 : 1;
  auto launch = [&](auto kernel) {
    C10_CUDA_CHECK(cudaLaunchKernelEx(
        &config, kernel, reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(residual_in.data_ptr()), post_layer_mix.data_ptr<float>(),
        comb_res_mix.data_ptr<float>(), pre_mix_in.data_ptr<float>(), reinterpret_cast<const __nv_bfloat16*>(norm_w.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(residual_out.data_ptr()), reinterpret_cast<__nv_bfloat16*>(layer_input.data_ptr()),
        aux.has_value() ? reinterpret_cast<__nv_bfloat16*>(aux->data_ptr()) : nullptr, static_cast<int>(H),
        static_cast<float>(norm_eps), dvec.data_ptr<float>(),
        pro ? reinterpret_cast<__nv_bfloat16*>(x_out->data_ptr()) : nullptr,
        reinterpret_cast<__nv_bfloat16*>(scratch.data_ptr())));
  };
  TORCH_CHECK(mode == 1, "the fold is built for Moet's post-mix mode 1 only (the production mode)");
  if (pdl) {
    if (pro && epi) launch(mhc_post_norm_fold_kernel<1, true, true, true>);
    else if (pro) launch(mhc_post_norm_fold_kernel<1, true, true, false>);
    else launch(mhc_post_norm_fold_kernel<1, true, false, true>);
  } else {
    if (pro && epi) launch(mhc_post_norm_fold_kernel<1, false, true, true>);
    else if (pro) launch(mhc_post_norm_fold_kernel<1, false, true, false>);
    else launch(mhc_post_norm_fold_kernel<1, false, false, true>);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mhc_post_norm_fold", &mhc_post_norm_fold,
        "Moet mHC boundary (post-mix, bf16 streams, collapse, RMSNorm, aux) with the DSV41_OBLIT projections folded in",
        py::arg("x"), py::arg("residual_in"), py::arg("post_layer_mix"), py::arg("comb_res_mix"), py::arg("pre_mix_in"),
        py::arg("norm_w"), py::arg("residual_out"), py::arg("layer_input"), py::arg("aux"), py::arg("norm_eps"),
        py::arg("mode"), py::arg("pdl"), py::arg("dvec"), py::arg("pro"), py::arg("epi"), py::arg("x_out"));
}
