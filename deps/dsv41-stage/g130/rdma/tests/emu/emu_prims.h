// emu_prims.h -- CPU emulation of the CUDA primitives that rdma_ar.cu's kernels use (tests/emu only).
// Model: one forked PROCESS per (rank, block) and one std::thread per CUDA thread, so a kernel's __shared__
// variables are process statics (shared by exactly that block). All "device" and "host-pinned" memory lives in one
// MAP_SHARED region created before any fork, so ranks, blocks and the proxy process see the same bytes.
#pragma once
#include <sched.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __launch_bounds__(...)
#define __shared__ static  // one process per (rank, block): process statics == that block's shared memory

struct EmuDim {
  unsigned x, y, z;
};
struct EmuCtx {
  EmuDim tid, bid, bdim, gdim;
};
extern thread_local EmuCtx emu_ctx;
#define threadIdx (emu_ctx.tid)
#define blockIdx (emu_ctx.bid)
#define blockDim (emu_ctx.bdim)
#define gridDim (emu_ctx.gdim)

struct EmuBarrier {
  std::mutex m;
  std::condition_variable cv;
  unsigned n = 0, count = 0, gen = 0;
  void wait() {
    std::unique_lock<std::mutex> l(m);
    const unsigned g = gen;
    if (++count == n) {
      count = 0;
      ++gen;
      cv.notify_all();
    } else {
      cv.wait(l, [&] { return gen != g; });
    }
  }
};
extern EmuBarrier* emu_bar;  // this process's block
inline void __syncthreads() { emu_bar->wait(); }
inline void __threadfence() { __atomic_thread_fence(__ATOMIC_SEQ_CST); }
inline void __threadfence_system() { __atomic_thread_fence(__ATOMIC_SEQ_CST); }
inline unsigned int atomicAdd(unsigned int* p, unsigned int v) { return __atomic_fetch_add(p, v, __ATOMIC_SEQ_CST); }
inline unsigned long long atomicAdd(unsigned long long* p, unsigned long long v) {
  return __atomic_fetch_add(p, v, __ATOMIC_SEQ_CST);
}

struct alignas(16) uint4 {
  unsigned int x, y, z, w;
};
inline uint4 make_uint4(unsigned a, unsigned b, unsigned c, unsigned d) {
  uint4 r;
  r.x = a;
  r.y = b;
  r.z = c;
  r.w = d;
  return r;
}
// uncached volatile load (the kernels only read peer data after an acquire of that peer's flag)
inline uint4 __ldcv(const uint4* p) {
  const volatile unsigned* q = reinterpret_cast<const volatile unsigned*>(p);
  uint4 r;
  r.x = q[0];
  r.y = q[1];
  r.z = q[2];
  r.w = q[3];
  return r;
}

// replacements for the three inline-PTX helpers extract.py removes
inline uint64_t ld_acquire_sys(const uint64_t* p) {
  usleep(5);  // g130: every spin iteration passes here: keep the emulator gentle on the host
  return __atomic_load_n(p, __ATOMIC_ACQUIRE);
}
inline void st_release_sys(uint64_t* p, uint64_t v) { __atomic_store_n(p, v, __ATOMIC_RELEASE); }
inline unsigned long long now_ns() {
  return (unsigned long long)std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
#define EMU_TRAP_TIMEOUT() \
  do {                     \
    fflush(stdout);        \
    _exit(43);             \
  } while (0)
#define LR_TRAP()   \
  do {              \
    fflush(stdout); \
    _exit(42);      \
  } while (0)
#define LR_HD

// bf16 (software, bit-exact: widening is exact, narrowing is round-to-nearest-even like __floats2bfloat162_rn)
struct float2 {
  float x, y;
};
struct __nv_bfloat16 {
  unsigned short x;
};
struct __nv_bfloat162 {
  __nv_bfloat16 x, y;
};
inline float emu_bf2f(unsigned short h) {
  const unsigned u = (unsigned)h << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}
inline unsigned short emu_f2bf(float f) {
  unsigned u;
  memcpy(&u, &f, 4);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (unsigned short)((u >> 16) | 0x40u);  // NaN stays NaN
  u += 0x7fffu + ((u >> 16) & 1u);
  return (unsigned short)(u >> 16);
}
inline float2 __bfloat1622float2(__nv_bfloat162 h) { return float2{emu_bf2f(h.x.x), emu_bf2f(h.y.x)}; }
inline __nv_bfloat162 __floats2bfloat162_rn(float a, float b) {
  __nv_bfloat162 r;
  r.x.x = emu_f2bf(a);
  r.y.x = emu_f2bf(b);
  return r;
}
inline float __uint_as_float(unsigned u) {
  float f;
  memcpy(&f, &u, 4);
  return f;
}
inline unsigned __float_as_uint(float f) {
  unsigned u;
  memcpy(&u, &f, 4);
  return u;
}
inline long long max(long long a, long long b) { return a > b ? a : b; }
inline long long min(long long a, long long b) { return a < b ? a : b; }

// warp-level max: the emulator runs live-row blocks with exactly 32 threads (one warp), so a block barrier is the warp
#define LR_WARP_MAX 1
extern int emu_wmax[32];
inline int lr_warp_max(int v) {
  emu_wmax[threadIdx.x] = v;
  __syncthreads();
  int m = 0;
  for (unsigned i = 0; i < blockDim.x; ++i) m = std::max(m, emu_wmax[i]);
  __syncthreads();
  return m;
}
