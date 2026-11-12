// dsv41 climb r3 G02: C ABI around the XSPLIT kernel (loaded with ctypes; no torch ABI).
//
// Serving build:  -DXS_ARM=0 -DXS_ROPE_IN_WS=1, no -DXS_BENCH  -> libdsv41_xsplit.so
// Bench builds:   -DXS_BENCH plus XS_ARM / XS_ROPE_IN_WS variants -> libdsv41_xsplit_bench_*.so
//                 (also exports the live dual kernel, instantiated from the live headers with the
//                  live flags, and the co-runner kernels used by the microbenchmark).
#include <cuda_runtime.h>
#include <stdint.h>

#include "dsv41_xsplit_kernel.cuh"

using namespace flashinfer::sparse_mla_sm120;
using namespace flashinfer::sparse_mla_sm120::dsv41_xsplit;

namespace {

constexpr int kMaxDev = 32;

template <typename K>
cudaError_t ensure_smem(K kernel, size_t bytes, bool (&done)[kMaxDev]) {
  int dev = 0;
  cudaError_t e = cudaGetDevice(&dev);
  if (e != cudaSuccess) return e;
  if (dev >= 0 && dev < kMaxDev && done[dev]) return cudaSuccess;
  e = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes);
  if (e == cudaSuccess && dev >= 0 && dev < kMaxDev) done[dev] = true;
  return e;
}

size_t align256(size_t x) { return (x + 255) / 256 * 256; }

struct WsOffsets {
  size_t tmpub, done, rows, total;
};
WsOffsets ws_offsets(int cap_units) {
  WsOffsets o;
  o.tmpub = align256(CTL_WORDS * sizeof(unsigned));
  o.done = align256(o.tmpub + (size_t)cap_units * HPB * sizeof(unsigned long long));
  o.rows = align256(o.done + (size_t)cap_units * sizeof(unsigned));
  o.total = align256(o.rows + (size_t)cap_units * ROW_WORDS * sizeof(float));
  return o;
}

template <int PBSX>
int launch_xsplit(const void* q, const void* kv, const void* idx, const void* kvx, const void* idxx,
                  const void* sink, void* out, void* lse, float sm_scale, int T, int topk,
                  int topk_extra, unsigned long long stride_kv, unsigned long long stride_kvx,
                  unsigned long long stride_lse, const void* len, const void* lenx, void* ws_base,
                  int cap_units, int grid, cudaStream_t stream) {
  using XL = XsSmemLayout<ModelType::DSV4, ComputeMode::BF16>;
  constexpr size_t smem = XL::TOTAL;
  auto kernel = dsv41_xsplit_mg_dual_kernel<PBSX>;
  static bool configured[kMaxDev] = {};
  cudaError_t e = ensure_smem(kernel, smem, configured);
  if (e != cudaSuccess) return (int)e;

  const WsOffsets o = ws_offsets(cap_units);
  char* b = static_cast<char*>(ws_base);
  XsWs ws{reinterpret_cast<unsigned*>(b), reinterpret_cast<unsigned long long*>(b + o.tmpub),
          reinterpret_cast<unsigned*>(b + o.done), reinterpret_cast<float*>(b + o.rows),
          cap_units, 0};
  PrefillColdParams cold{sm_scale,
                         T,
                         (size_t)stride_kv,
                         (size_t)stride_kvx,
                         (size_t)stride_lse,
                         topk,
                         topk_extra,
                         static_cast<const float*>(sink),
                         static_cast<const int*>(len),
                         static_cast<const int*>(lenx)};
  const bf16* Q = static_cast<const bf16*>(q);
  const uint8_t* KV = static_cast<const uint8_t*>(kv);
  const int32_t* I = static_cast<const int32_t*>(idx);
  const uint8_t* KVX = static_cast<const uint8_t*>(kvx);
  const int32_t* IX = static_cast<const int32_t*>(idxx);
  bf16* O = static_cast<bf16*>(out);
  float* L = static_cast<float*>(lse);
  const float* S = static_cast<const float*>(sink);
  cudaLaunchConfig_t config{dim3(grid), dim3(BLOCK_THREADS), smem, stream, nullptr, 0};
  void* args[] = {(void*)&Q, (void*)&KV, (void*)&I, (void*)&KVX, (void*)&IX,
                  (void*)&O, (void*)&L,  (void*)&S, (void*)&cold, (void*)&ws};
  e = cudaLaunchKernelExC(&config, (const void*)kernel, args);
  return (int)e;
}

}  // namespace

extern "C" {

int dsv41_xsplit_abi_version() { return 1; }
int dsv41_xsplit_row_words() { return ROW_WORDS; }
int dsv41_xsplit_arm() { return XS_ARM; }
int dsv41_xsplit_rope_in_ws() { return XS_ROPE_IN_WS; }
unsigned long long dsv41_xsplit_ws_bytes(int cap_units) { return ws_offsets(cap_units).total; }
unsigned long long dsv41_xsplit_ws_offset(int cap_units, int which) {
  const WsOffsets o = ws_offsets(cap_units);
  return which == 0 ? 0 : which == 1 ? o.tmpub : which == 2 ? o.done : o.rows;
}
unsigned long long dsv41_xsplit_smem_bytes() {
  return XsSmemLayout<ModelType::DSV4, ComputeMode::BF16>::TOTAL;
}

// Returns 0 on a successful launch, a cudaError_t value otherwise (nothing launched), or -1 for
// an unsupported extra page block size.
int dsv41_xsplit_launch(int pbsx, const void* q, const void* kv, const void* idx, const void* kvx,
                        const void* idxx, const void* sink, void* out, void* lse, float sm_scale,
                        int T, int topk, int topk_extra, unsigned long long stride_kv,
                        unsigned long long stride_kvx, unsigned long long stride_lse,
                        const void* len, const void* lenx, void* ws_base, int cap_units, int grid,
                        void* stream) {
  cudaStream_t st = static_cast<cudaStream_t>(stream);
  if (pbsx == 64)
    return launch_xsplit<64>(q, kv, idx, kvx, idxx, sink, out, lse, sm_scale, T, topk, topk_extra,
                             stride_kv, stride_kvx, stride_lse, len, lenx, ws_base, cap_units,
                             grid, st);
  if (pbsx == 32)
    return launch_xsplit<32>(q, kv, idx, kvx, idxx, sink, out, lse, sm_scale, T, topk, topk_extra,
                             stride_kv, stride_kvx, stride_lse, len, lenx, ws_base, cap_units,
                             grid, st);
  return -1;
}

#ifdef XS_BENCH
// ---- Bench-only exports ------------------------------------------------------------------

// The live kernel, instantiated from the live headers exactly as launch_prefill_mg_dual does.
int dsv41_live_mg_dual_launch(int pbsx, const void* q, const void* kv, const void* idx,
                              const void* kvx, const void* idxx, const void* sink, void* out,
                              void* lse, float sm_scale, int T, int topk, int topk_extra,
                              unsigned long long stride_kv, unsigned long long stride_kvx,
                              unsigned long long stride_lse, const void* len, const void* lenx,
                              void* stream) {
  constexpr size_t smem = SmemLayoutMG<ModelType::DSV4, ComputeMode::BF16>::TOTAL;
  PrefillColdParams cold{sm_scale,
                         T,
                         (size_t)stride_kv,
                         (size_t)stride_kvx,
                         (size_t)stride_lse,
                         topk,
                         topk_extra,
                         static_cast<const float*>(sink),
                         static_cast<const int*>(len),
                         static_cast<const int*>(lenx)};
  const bf16* Q = static_cast<const bf16*>(q);
  const uint8_t* KV = static_cast<const uint8_t*>(kv);
  const int32_t* I = static_cast<const int32_t*>(idx);
  const uint8_t* KVX = static_cast<const uint8_t*>(kvx);
  const int32_t* IX = static_cast<const int32_t*>(idxx);
  bf16* O = static_cast<bf16*>(out);
  float* L = static_cast<float*>(lse);
  const float* S = static_cast<const float*>(sink);
  void* args[] = {(void*)&Q, (void*)&KV, (void*)&I, (void*)&KVX, (void*)&IX,
                  (void*)&O, (void*)&L,  (void*)&S, (void*)&cold};
  cudaLaunchConfig_t config{dim3(T), dim3(BLOCK_THREADS), smem, static_cast<cudaStream_t>(stream),
                            nullptr, 0};
  cudaError_t e;
  if (pbsx == 64) {
    auto k = sparse_mla_prefill_mg_dual_kernel<ModelType::DSV4, ComputeMode::BF16, 8, 64, 64, 1>;
    static bool c[kMaxDev] = {};
    e = ensure_smem(k, smem, c);
    if (e == cudaSuccess) e = cudaLaunchKernelExC(&config, (const void*)k, args);
  } else if (pbsx == 32) {
    auto k = sparse_mla_prefill_mg_dual_kernel<ModelType::DSV4, ComputeMode::BF16, 8, 64, 32, 1>;
    static bool c[kMaxDev] = {};
    e = ensure_smem(k, smem, c);
    if (e == cudaSuccess) e = cudaLaunchKernelExC(&config, (const void*)k, args);
  } else {
    return -1;
  }
  return (int)e;
}

}  // extern "C"

// Co-runner stand-ins (copied from the g130-attn harness): the WOA L2 prefetch and an SM-hogging
// spinner that forces partial residency.
__global__ void xs_l2pf_kernel(const long long* __restrict__ segs, int n, long long chunk) {
  for (int s = blockIdx.x; s < n; s += gridDim.x) {
    long long a = segs[2 * s], b = segs[2 * s + 1];
    for (long long off = (long long)threadIdx.x * chunk; off < b;
         off += (long long)blockDim.x * chunk) {
      long long sz = b - off < chunk ? b - off : chunk;
      sz &= ~15LL;
      if (sz > 0)
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(a + off),
                     "r"((unsigned)sz)
                     : "memory");
    }
  }
}
__global__ void __launch_bounds__(384, 1) xs_spin_kernel(long long ns) {
  unsigned long long t0, t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  do {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  } while ((long long)(t - t0) < ns);
}

extern "C" {
int dsv41_bench_l2pf(const void* segs, int n, long long chunk, int grid, void* stream) {
  xs_l2pf_kernel<<<grid, 64, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const long long*>(segs), n, chunk);
  return (int)cudaGetLastError();
}
// ctas CTAs of 384 threads and 96 KB smem: each occupies a whole SM like the MLA CTA does.
int dsv41_bench_spin(int ctas, long long ns, void* stream) {
  static bool c[kMaxDev] = {};
  cudaError_t e = ensure_smem(xs_spin_kernel, 96 * 1024, c);
  if (e != cudaSuccess) return (int)e;
  xs_spin_kernel<<<ctas, 384, 96 * 1024, static_cast<cudaStream_t>(stream)>>>(ns);
  return (int)cudaGetLastError();
}
#endif

}  // extern "C"
