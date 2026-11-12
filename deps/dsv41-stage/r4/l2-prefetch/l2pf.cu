// DSv4.1 r4 l2-prefetch native helpers (sm_121a, GB10). Build (Spark CPU, no GPU needed):
//   nvcc -O3 -arch=sm_121a -shared -Xcompiler -fPIC -o libdsv41_l2pf.so l2pf.cu
// dsv41_l2pf : side-stream L2 prefetch of a segment table {addr, bytes} (cache hint only; never
//              changes data, so every consumer stays bit-identical).
// dsv41_spin : test/bench only - a few CTAs busy-wait for `ns` (stands in for the RDMA ar_kernel,
//              which polls flags with 8-16 CTAs while SMs and DRAM are idle).
#include <cuda_runtime.h>
#include <stdint.h>

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
