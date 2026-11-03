// BS03 test helper (test_det_topk.py only, never loaded by the serving stack). Same build as ~/dsv41-stage/r4/l2-prefetch/l2pf.cu:
//   nvcc -O3 -arch=sm_121a -shared -Xcompiler -fPIC -o libbs03_stamp.so bs03_stamp.cu
// bs03_stamp: one thread writes %globaltimer (ns) to *out, in stream order, so a captured CUDA graph can time what runs
// between two stamps on its own stream while a co-runner occupies other SMs.
#include <cuda_runtime.h>
#include <stdint.h>

__global__ void bs03_stamp_kernel(unsigned long long* out) {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  *out = t;
}

extern "C" int bs03_stamp(void* out, uintptr_t stream) {
  bs03_stamp_kernel<<<1, 1, 0, (cudaStream_t)stream>>>((unsigned long long*)out);
  return (int)cudaGetLastError();
}
