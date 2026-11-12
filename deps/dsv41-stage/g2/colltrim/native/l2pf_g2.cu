// DSv4.1 g2 colltrim (b): L2-prefetch helpers with a PACED issuer (sm_121a, GB10).
// Build (Spark CPU, no GPU needed):
//   nvcc -O3 -arch=sm_121a -shared -Xcompiler -fPIC -o libdsv41_l2pf_g2.so l2pf_g2.cu
//
// dsv41_l2pf / dsv41_spin: byte-for-byte the r4 kernels of ~/dsv41-stage/r4/l2-prefetch/l2pf.cu (the live
//   libdsv41_l2pf.so), so this library can stand in for it. dsv41_l2pf_g2_version() identifies it.
// dsv41_l2pf_paced: NEW. One thread walks the segment table {addr, bytes} in plan order and issues one
//   cp.async.bulk.prefetch.L2 of `chunk` bytes per period (deadline-based on %globaltimer, so the average
//   rate is chunk / period whatever the SM clock), after an optional start delay, and stops issuing once
//   max_ns have passed since the start (0 = no cap). The live burst kernel instead fires every chunk of a
//   window at once from min(nseg, 8) CTAs x 64 threads, which keeps LPDDR5X saturated for the whole window
//   (~6 MB in ~25 us) while the RDMA collective's latency chain (send-slot staging, NIC DMA, sys-scope flag
//   polls, peer loads) queues behind it.
// A prefetch is only a cache hint: no data is written, every consumer stays bit-identical.
#include <cuda_runtime.h>
#include <stdint.h>

// ---- r4 kernels, unchanged -------------------------------------------------------------------
// segs: int64 [n][2] = {address, bytes}. CTA s handles segments s, s+grid, ...; each thread issues
// one bulk prefetch per `chunk` bytes. Addresses / sizes are rounded to 16 B (bulk-op contract).
__global__ void l2pf_kernel(const long long* __restrict__ segs, int n, long long chunk) {
  for (int s = blockIdx.x; s < n; s += gridDim.x) {
    long long a = segs[2 * s], b = segs[2 * s + 1];
    for (long long off = (long long)threadIdx.x * chunk; off < b; off += (long long)blockDim.x * chunk) {
      long long sz = b - off < chunk ? b - off : chunk;
      sz &= ~15LL;
      if (sz > 0)
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(a + off), "r"((unsigned)sz) : "memory");
    }
  }
}

extern "C" int dsv41_l2pf(const void* segs, int n, long long chunk, uintptr_t stream) {
  if (n <= 0) return 0;
  l2pf_kernel<<<n < 8 ? n : 8, 64, 0, (cudaStream_t)stream>>>((const long long*)segs, n, chunk);
  return (int)cudaGetLastError();
}

__global__ void spin_kernel(long long ns) {
  unsigned long long t0, t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  do {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  } while ((long long)(t - t0) < ns);
}

extern "C" int dsv41_spin(long long ns, int ctas, uintptr_t stream) {
  spin_kernel<<<ctas, 32, 0, (cudaStream_t)stream>>>(ns);
  return (int)cudaGetLastError();
}

// ---- g2 colltrim: paced issuer -------------------------------------------------------------------
__device__ __forceinline__ unsigned long long g2_now_ns() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// One warp, lane 0 issues (the other lanes exit at once). Deadlines are kept in picoseconds relative to
// the kernel start (no 64-bit overflow): chunk k of size sz is issued at delay + sum(period * sz_j / chunk).
// stats (optional, device int64[4]): {issued bytes, issued prefetches, first-issue ns, last-issue ns}
// relative to the kernel start; used by the GPU test to verify the pacing. nullptr in serving.
__global__ void __launch_bounds__(32, 1) l2pf_paced_kernel(const long long* __restrict__ segs, int n, long long chunk,
                                                           long long period_ps, long long delay_ns, long long max_ns,
                                                           long long* stats) {
  if (threadIdx.x != 0) return;
  const unsigned long long t0 = g2_now_ns();
  unsigned long long next_ps = (unsigned long long)delay_ns * 1000ull;
  const unsigned long long cap_ns = max_ns > 0 ? (unsigned long long)(delay_ns + max_ns) : ~0ull;
  long long issued = 0, count = 0;
  unsigned long long first = 0, last = 0;
  for (int s = 0; s < n; ++s) {
    const long long a = segs[2 * s], b = segs[2 * s + 1];
    for (long long off = 0; off < b; off += chunk) {
      long long sz = b - off < chunk ? b - off : chunk;
      sz &= ~15LL;
      if (sz <= 0) continue;
      unsigned long long rel;
      while ((rel = g2_now_ns() - t0) * 1000ull < next_ps) __nanosleep(32);
      if (rel > cap_ns) goto done;
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(a + off), "r"((unsigned)sz) : "memory");
      if (count == 0) first = rel;
      last = rel;
      issued += sz;
      ++count;
      next_ps += (unsigned long long)period_ps * (unsigned long long)sz / (unsigned long long)chunk;
    }
  }
done:
  if (stats != nullptr) {
    stats[0] = issued;
    stats[1] = count;
    stats[2] = (long long)first;
    stats[3] = (long long)last;
  }
}

// rate: period_ps = chunk * 1000 / GBps (1 GB/s = 1 byte/ns). Returns the CUDA launch error (0 = ok).
extern "C" int dsv41_l2pf_paced(const void* segs, int n, long long chunk, long long period_ps, long long delay_ns,
                                long long max_ns, void* stats, uintptr_t stream) {
  if (n <= 0 || chunk < 16 || period_ps < 0) return 0;
  l2pf_paced_kernel<<<1, 32, 0, (cudaStream_t)stream>>>((const long long*)segs, n, chunk & ~15LL, period_ps,
                                                         delay_ns < 0 ? 0 : delay_ns, max_ns, (long long*)stats);
  return (int)cudaGetLastError();
}

extern "C" int dsv41_l2pf_g2_version() { return 1; }
