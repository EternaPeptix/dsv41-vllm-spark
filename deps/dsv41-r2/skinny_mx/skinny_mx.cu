// skinny_mx.cu: decode-sized (M <= 8) MXFP8 GEMM for GB10 (sm_121).
//
//   y[M, N] (bf16) = x[M, K] @ (w[N, K] fp8-e4m3 * 2^(ws[N, K/32] - 127))^T, fp32 accumulation.
//
// Why: at M = 4..6 the FlashInfer CUTLASS MXFP8 kernel (tile 128x32x128) launches only N/128 CTAs
// (14 for fused_wqa_wkv, 32 for wq_b, 5 for the shared-expert w13) on a 48-SM GB10 and has no split-K,
// so each CTA streams a 128 x K weight slab alone: 129-150 GB/s against a 238 GB/s read ceiling.
// This kernel is a persistent GEMV: one block per SM, every warp streams whole weight rows with
// coalesced 16-byte evict-first loads, so all SMs pull weight bytes regardless of N.
//
// Activation modes (both exact w.r.t. the operands they receive; only fp32 summation order differs
// from CUTLASS):
//   * A8: x arrives MXFP8-quantised (fp8 data [M, K] + F8_128x4 swizzled e8m0 scales, the layout
//         FlashInfer mm_mxfp8 consumes). It is dequantised into shared memory as bf16, which is exact
//         (3-bit mantissa times a power of two). Same operands as the current W8A8 path.
//   * A16: x is bf16 and used as is (W8A16, strictly more precise than W8A8). Used for WO-A, whose
//         live path is a BF16 bmm over a dequantised copy of the same fp8 weights.
//
// Determinism: fixed per-row summation order (lane-strided K walk, xor-shuffle tree, fixed-order
// split combine in shared memory); no atomics. Bitwise reproducible run to run for the same M.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr int WARPS = 8;  // 256 threads, one block per SM (x in smem can be up to 80 KB)

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

// 16 fp8 weights (one 16-byte vector, inside one 32-wide scale block) dotted with 16 bf16 activations.
__device__ __forceinline__ float dot16_fp8(const uint4& w, const uint4& xa, const uint4& xb) {
  const __nv_fp8x2_storage_t* wp = reinterpret_cast<const __nv_fp8x2_storage_t*>(&w);
  const __nv_bfloat162* xpa = reinterpret_cast<const __nv_bfloat162*>(&xa);
  const __nv_bfloat162* xpb = reinterpret_cast<const __nv_bfloat162*>(&xb);
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2(wp[i], __NV_E4M3);
    float2 wf = __half22float2(*reinterpret_cast<__half2*>(&hr));
    float2 xf = __bfloat1622float2(i < 4 ? xpa[i] : xpb[i - 4]);
    s = fmaf(wf.x, xf.x, s);
    s = fmaf(wf.y, xf.y, s);
  }
  return s;
}

// Swizzled (F8_128x4) activation-scale offset for row m, 32-wide k block kb.
// Layout (vllm mxfp8_utils.swizzle_mxfp8_scale): [m_tiles, k_tiles, 32, 4(m sub), 4(k sub)].
__device__ __forceinline__ int swz_off(int m, int kb, int k_tiles) {
  const int mt = m >> 7, r = m & 127, i4 = r >> 5, i32 = r & 31, kt = kb >> 2, k4 = kb & 3;
  return ((((mt * k_tiles + kt) * 32 + i32) * 4 + i4) * 4) + k4;
}

// grid = #SMs (persistent). Each block iteration covers ROWS_PER_IT = WARPS / SPLIT rows; warp w
// handles row (it * ROWS_PER_IT + w / SPLIT), K slice (w % SPLIT).
template <int M, int UNROLL, int SPLIT>
__global__ void __launch_bounds__(WARPS * 32, 1)
skinny_mx_kernel(const __nv_bfloat16* __restrict__ x,   // A16 mode: [M, K] bf16 (else nullptr)
                 const uint8_t* __restrict__ xq,        // A8 mode: [M, K] fp8 e4m3
                 const uint8_t* __restrict__ xs,        // A8 mode: swizzled e8m0 scales
                 const uint8_t* __restrict__ w,         // [N, K] fp8 e4m3, row-major
                 const uint8_t* __restrict__ ws,        // [N, K/32] e8m0, row-major
                 __nv_bfloat16* __restrict__ y,         // [M, ldy]
                 int N, int K, int ldy) {
  extern __shared__ uint4 smem[];
  __nv_bfloat16* xs_bf = reinterpret_cast<__nv_bfloat16*>(smem);  // M * K bf16
  const int kv8 = K / 8;
  if (x != nullptr) {
    for (int i = threadIdx.x; i < M * kv8; i += blockDim.x) smem[i] = reinterpret_cast<const uint4*>(x)[i];
  } else {
    const int k_tiles = (K / 32 + 3) / 4;
    const int kv16 = K / 16;
    for (int i = threadIdx.x; i < M * kv16; i += blockDim.x) {  // 16 fp8 per step
      const int m = i / kv16, v = i - m * kv16;
      const float sc = e8m0_to_float(xs[swz_off(m, v >> 1, k_tiles)]);
      const uint4 q = reinterpret_cast<const uint4*>(xq)[i];
      const __nv_fp8x2_storage_t* qp = reinterpret_cast<const __nv_fp8x2_storage_t*>(&q);
      uint4 o[2];
      __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(o);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2(qp[j], __NV_E4M3);
        float2 f = __half22float2(*reinterpret_cast<__half2*>(&hr));
        op[j] = __floats2bfloat162_rn(f.x * sc, f.y * sc);  // exact: 4-bit significand * 2^e
      }
      smem[2 * i] = o[0];
      smem[2 * i + 1] = o[1];
    }
  }
  __syncthreads();

  constexpr int ROWS_PER_IT = WARPS / SPLIT;
  __shared__ float part_sums[WARPS][M];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int group = warp / SPLIT, part = warp % SPLIT;
  const int kv = K / 16;  // 16-byte weight vectors per row
  const int kv_per = (kv + SPLIT - 1) / SPLIT;
  const int kb = part * kv_per, ke = min(kv, kb + kv_per);
  const int ksc = K / 32;
  const int iters = (N + ROWS_PER_IT - 1) / ROWS_PER_IT;

  for (int it = blockIdx.x; it < iters; it += gridDim.x) {
    const int row = it * ROWS_PER_IT + group;
    float acc[M];
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.f;
    if (row < N) {
      const uint4* wr = reinterpret_cast<const uint4*>(w + (size_t)row * K);
      const uint8_t* sr = ws + (size_t)row * ksc;
      int k = kb + lane;
      for (; k + 32 * (UNROLL - 1) < ke; k += 32 * UNROLL) {
        uint4 wv[UNROLL];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) wv[u] = ld_stream(wr + k + 32 * u);
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
          const int kk = k + 32 * u;
          const float s = e8m0_to_float(__ldg(sr + (kk >> 1)));
#pragma unroll
          for (int m = 0; m < M; ++m) {
            const uint4 xa = smem[m * kv8 + 2 * kk], xb = smem[m * kv8 + 2 * kk + 1];
            acc[m] = fmaf(s, dot16_fp8(wv[u], xa, xb), acc[m]);
          }
        }
      }
      for (; k < ke; k += 32) {
        const uint4 wv = ld_stream(wr + k);
        const float s = e8m0_to_float(__ldg(sr + (k >> 1)));
#pragma unroll
        for (int m = 0; m < M; ++m)
          acc[m] = fmaf(s, dot16_fp8(wv, smem[m * kv8 + 2 * k], smem[m * kv8 + 2 * k + 1]), acc[m]);
      }
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
      float v = acc[m];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
      acc[m] = v;
    }
    if constexpr (SPLIT == 1) {
      if (lane == 0 && row < N) {
#pragma unroll
        for (int m = 0; m < M; ++m) y[(size_t)m * ldy + row] = __float2bfloat16(acc[m]);
      }
    } else {
      if (lane == 0) {
#pragma unroll
        for (int m = 0; m < M; ++m) part_sums[warp][m] = acc[m];
      }
      __syncthreads();
      if (part == 0 && lane == 0 && row < N) {
#pragma unroll
        for (int m = 0; m < M; ++m) {
          float t = 0.f;
#pragma unroll
          for (int q = 0; q < SPLIT; ++q) t += part_sums[group * SPLIT + q][m];
          y[(size_t)m * ldy + row] = __float2bfloat16(t);
        }
      }
      __syncthreads();
    }
  }
}

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
    v -= 1024;  // static part_sums
  }
  return v;
}

// Pick SPLIT in {1,2,4,8} that minimises the last-wave waste of the persistent loop, preferring
// smaller SPLIT on ties, and requiring >= 32 vectors per (row, part) so lanes stay busy.
int pick_split(int N, int K, int grid, int forced) {
  if (forced > 0) return forced;
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

template <int M, int SPLIT>
void launch_ms(const __nv_bfloat16* x, const uint8_t* xq, const uint8_t* xs, const uint8_t* w,
               const uint8_t* ws, __nv_bfloat16* y, int N, int K, int ldy, int grid, cudaStream_t st) {
  constexpr int UNROLL = 4;
  auto kern = skinny_mx_kernel<M, UNROLL, SPLIT>;
  static bool attr_set = false;
  if (!attr_set) {
    TORCH_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, max_smem()) == cudaSuccess,
                "skinny_mx: cannot raise shared memory limit");
    attr_set = true;
  }
  const size_t smem = (size_t)M * K * sizeof(__nv_bfloat16);
  kern<<<grid, WARPS * 32, smem, st>>>(x, xq, xs, w, ws, y, N, K, ldy);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int M>
void launch_m(int split, const __nv_bfloat16* x, const uint8_t* xq, const uint8_t* xs, const uint8_t* w,
              const uint8_t* ws, __nv_bfloat16* y, int N, int K, int ldy, int grid, cudaStream_t st) {
  switch (split) {
    case 1: launch_ms<M, 1>(x, xq, xs, w, ws, y, N, K, ldy, grid, st); break;
    case 2: launch_ms<M, 2>(x, xq, xs, w, ws, y, N, K, ldy, grid, st); break;
    case 4: launch_ms<M, 4>(x, xq, xs, w, ws, y, N, K, ldy, grid, st); break;
    default: launch_ms<M, 8>(x, xq, xs, w, ws, y, N, K, ldy, grid, st); break;
  }
}

bool fits(int64_t m, int64_t k) { return m >= 1 && m <= 8 && k % 32 == 0 && m * k * 2 <= max_smem(); }

void check_w(const torch::Tensor& w, const torch::Tensor& ws) {
  TORCH_CHECK(w.is_cuda() && w.dim() == 2 && w.element_size() == 1 && w.is_contiguous(),
              "skinny_mx: w must be contiguous [N, K] fp8");
  TORCH_CHECK(ws.is_cuda() && ws.dim() == 2 && ws.element_size() == 1 && ws.is_contiguous(),
              "skinny_mx: ws must be contiguous [N, K/32] e8m0/uint8");
  TORCH_CHECK(ws.size(0) == w.size(0) && ws.size(1) == w.size(1) / 32 && w.size(1) % 32 == 0,
              "skinny_mx: scale shape");
}

void dispatch(int M, int split, const __nv_bfloat16* x, const uint8_t* xq, const uint8_t* xs,
              const uint8_t* w, const uint8_t* ws, __nv_bfloat16* y, int N, int K, int ldy, int grid,
              cudaStream_t st) {
  switch (M) {
    case 1: launch_m<1>(split, x, xq, xs, w, ws, y, N, K, ldy, grid, st); break;
    case 2: launch_m<2>(split, x, xq, xs, w, ws, y, N, K, ldy, grid, st); break;
    case 3: launch_m<3>(split, x, xq, xs, w, ws, y, N, K, ldy, grid, st); break;
    case 4: launch_m<4>(split, x, xq, xs, w, ws, y, N, K, ldy, grid, st); break;
    case 5: launch_m<5>(split, x, xq, xs, w, ws, y, N, K, ldy, grid, st); break;
    case 6: launch_m<6>(split, x, xq, xs, w, ws, y, N, K, ldy, grid, st); break;
    case 7: launch_m<7>(split, x, xq, xs, w, ws, y, N, K, ldy, grid, st); break;
    default: launch_m<8>(split, x, xq, xs, w, ws, y, N, K, ldy, grid, st); break;
  }
}

// A8: xq [M, K] fp8, xs swizzled scales (1-D, as produced by mxfp8_e4m3_quantize(is_sf_swizzled_layout=True)).
torch::Tensor skinny_mx_a8(torch::Tensor xq, torch::Tensor xs, torch::Tensor w, torch::Tensor ws,
                           int64_t split, int64_t grid) {
  check_w(w, ws);
  TORCH_CHECK(xq.is_cuda() && xq.dim() == 2 && xq.element_size() == 1 && xq.is_contiguous(), "skinny_mx_a8: xq");
  TORCH_CHECK(xs.is_cuda() && xs.element_size() == 1 && xs.is_contiguous(), "skinny_mx_a8: xs");
  const int M = (int)xq.size(0), K = (int)xq.size(1), N = (int)w.size(0);
  TORCH_CHECK(K == w.size(1) && fits(M, K), "skinny_mx_a8: shape/M");
  const int64_t need = (int64_t)((M + 127) / 128) * 128 * (((K / 32) + 3) / 4) * 4;
  TORCH_CHECK(xs.numel() >= need, "skinny_mx_a8: swizzled scale too small");
  c10::cuda::CUDAGuard guard(xq.device());
  auto y = torch::empty({M, N}, xq.options().dtype(at::kBFloat16));
  const int g = grid > 0 ? (int)grid : num_sms();
  dispatch(M, pick_split(N, K, g, (int)split), nullptr, (const uint8_t*)xq.data_ptr(),
           (const uint8_t*)xs.data_ptr(), (const uint8_t*)w.data_ptr(), (const uint8_t*)ws.data_ptr(),
           reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), N, K, N, g, at::cuda::getCurrentCUDAStream());
  return y;
}

// A16: x [M, K] bf16.
torch::Tensor skinny_mx_a16(torch::Tensor x, torch::Tensor w, torch::Tensor ws, int64_t split, int64_t grid) {
  check_w(w, ws);
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.scalar_type() == at::kBFloat16 && x.is_contiguous(),
              "skinny_mx_a16: x must be contiguous [M, K] bf16");
  const int M = (int)x.size(0), K = (int)x.size(1), N = (int)w.size(0);
  TORCH_CHECK(K == w.size(1) && fits(M, K), "skinny_mx_a16: shape/M");
  c10::cuda::CUDAGuard guard(x.device());
  auto y = torch::empty({M, N}, x.options());
  const int g = grid > 0 ? (int)grid : num_sms();
  dispatch(M, pick_split(N, K, g, (int)split), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), nullptr,
           nullptr, (const uint8_t*)w.data_ptr(), (const uint8_t*)ws.data_ptr(),
           reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), N, K, N, g, at::cuda::getCurrentCUDAStream());
  return y;
}

int64_t chosen_split(int64_t N, int64_t K, int64_t grid) {
  return pick_split((int)N, (int)K, grid > 0 ? (int)grid : num_sms(), 0);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fits", &fits, "whether skinny_mx handles [m, k] activations");
  m.def("chosen_split", &chosen_split, py::arg("N"), py::arg("K"), py::arg("grid") = 0);
  m.def("skinny_mx_a8", &skinny_mx_a8, "y = deq(xq, xs) @ deq(w, ws).T for M <= 8", py::arg("xq"), py::arg("xs"),
        py::arg("w"), py::arg("ws"), py::arg("split") = 0, py::arg("grid") = 0);
  m.def("skinny_mx_a16", &skinny_mx_a16, "y = x @ deq(w, ws).T for M <= 8", py::arg("x"), py::arg("w"),
        py::arg("ws"), py::arg("split") = 0, py::arg("grid") = 0);
}
