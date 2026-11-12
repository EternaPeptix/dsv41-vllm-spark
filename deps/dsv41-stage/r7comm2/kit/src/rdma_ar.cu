// rdma_ar.cu: RDMA all-reduce / all-gather for small tensors, one GPU per node (DGX Spark / GB10).
//
// NCCL's net transport costs one CPU-proxy hop per ring step (14 hops for an 8-rank ring all-reduce,
// ~150-190 us inside a CUDA graph). Three modes here, all one kernel launch:
//   one-shot reduce : every rank writes its whole input into each peer's receive slot (one hop),
//                     then sums the world contributions locally in rank order
//   two-shot reduce : reduce-scatter (chunk q of my input -> rank q), rank q sums chunk q in rank
//                     order and broadcasts it (two hops, 4x fewer bytes on the wire than one-shot)
//   all-gather      : one-shot copy, out[r] = input of rank r
// Every element of a reduction is summed by one fixed rule, so all ranks see bit-identical results.
//   column all-gather (g2 colltrim, 2026-10-26): same wire protocol as all-gather; the receiver writes
//                     rank r's [outer, seg] input straight into columns [r*seg, (r+1)*seg) of the
//                     concatenated [outer, world*seg] output (no staging tensor, no movedim copy)
//
// Data path (all buffers are cudaHostAlloc'd = the same LPDDR5X on GB10, registered once per HCA):
//   GPU kernel: stage input in the send slot, last CTA publishes ctrl.send_seq
//   proxy thread: sees send_seq, posts per peer {RDMA WRITE payload, RDMA WRITE 8-byte seq flag}
//                 (same RC QP, so the flag lands after the payload)
//   GPU kernel: spins until every peer's flag for this slot reaches seq, then reduces / copies
// The sequence number lives in device memory and is advanced by the kernel itself, so the ops are
// safe to capture in CUDA graphs (kernel args never change between replays).
//
// g130 rdma (2026-11-02), on top of g2 colltrim 6173b362. Every flag defaults OFF; unset == production behaviour, and the
// production kernels (kPub == 0) are instruction-identical (tests/sass_gate.py). Wire protocol, Params, Ctrl, DevState,
// slots, sum order and rounding are untouched. Read once per instance at construction; WINDOW.md gives the commands.
//   RDMA_AR_SPLIT_POST=1  A   host only: each WR gets its own ibv_post_send (nreq == 1 qualifies for BlueFlame)
//   RDMA_AR_PUB_V1=1      B   kernel: phase-1 publish compares posted_seq with seq - 1 (no send_seq load)
//   RDMA_AR_PUB_V2=1      B   kernel: phase-2 publish has no spin at all (no send2_seq / posted2_seq load)
//   RDMA_AR_HYG_PROXY=1   C   host only: proxy spins on the one doorbell that rings next, reaps CQs after each post
//   RDMA_AR_HYG_POLL=1    C   kernel: GPU spins read the clock on the first miss and once per 1024 passes
// B and C-poll select other kernel instantiations at launch (parallel-load path only). See DESIGN notes in WINDOW.md.
//
// r3-comm (climb round 3, 2026-11-08), on top of g130 d38cfe68. Default OFF; unset == the g130 .so's behaviour: the
// existing kernels are untouched (byte-identical source, SASS-gated) and the r3 kernel ar_kernel_r3 is never launched.
//   DSV41_RDMA_TRIM=1     G06 comm:A  ar_kernel_r3: early posted_seq read, seq advance at kernel entry, single-CTA
//                                     phase-1 publish for messages <= DSV41_RDMA_TRIM_S1_BYTES (24576), single-CTA
//                                     two-shot phase-2 publish when chunk <= 512 vectors. Needs PUB_V1+V2+HYG_POLL.
//   DSV41_RDMA_SENDBUF=1  G26 comm:E  (needs TRIM) a registered send-buffer region (DSV41_RDMA_SENDBUF_MB, 64) at the
//                                     arena end; RdmaAR.send_buffer(nbytes) hands out views; an op whose input IS such
//                                     a buffer is published from it (no staging copy, no per-thread fence, no barrier).
// Same reduction (fp32 from +0.0 in r = 0..7, one bf16 RN), same Params, slots, flags, messages and wire protocol.
//
// r6-comm (climb round 6, 2026-11-09), on top of r3-comm ad0f13a8. Default OFF.
//   DSV41_RDMA_PDL=1|in   C2  (needs DSV41_RDMA_TRIM=1) ar_kernel_r3 launches with programmatic stream serialization;
//                             before griddepcontrol.wait it only reads ctrl->posted_seq (monotonic, host-written).
//   DSV41_RDMA_PDL=trig   C2  as 1, plus griddepcontrol.launch_dependents after the last flag wait.
// Same FP stream as r3 / live (gate/fpflow.py on the binary); scheduling only.
//
// r7-comm2 (climb round 7, 2026-11-09), on top of r6-comm 2bff8a89. Default OFF.
//   DSV41_RDMA_PROJ=1     HT13 (needs DSV41_RDMA_TRIM=1) RdmaAR.all_reduce_proj: the bf16 [rows <= 6, 5120] all-reduce
//                             also writes y = the dsv41_oblit _proj_rows projection of its final bf16 output, computed in
//                             the kernel tail with the served Triton program's exact op order (gate/projgate.py).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <infiniband/verbs.h>
#include <pthread.h>
#include <sched.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <cstddef>
#include <cctype>  // r3-comm: std::tolower in env_r3_norm
#include <sys/mman.h>
#include <fstream>
#include <ctime>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <optional>
#include <unordered_map>

namespace {

constexpr int kSlots = 4;
constexpr int kThreads = 512;
constexpr int kSignalEvery = 32;  // request a CQE every N posts per QP so the SQ drains

enum Mode : int { kOneShot = 0, kGather = 1, kTwoShot = 2, kGatherCols = 3 };  // kGatherCols: g2 colltrim
// g130: kernel variants, chosen per launch by instantiation and never by a kernel argument, so kPub == 0 is production
enum PubBits : int { kPubV1 = 1, kPubV2 = 2, kPollThin = 4 };
// r3-comm: bits above the g130 mask select ar_kernel_r3 instantiations (never a kernel argument)
enum R3Bits : int { kR3Trim = 8, kR3S1 = 16, kR3P2 = 32, kR3Sbuf = 64 };
enum R6Bits : int { kR6Pdl = 128, kR6Trig = 256 };  // r6-comm C2: PDL in-edge / early trigger (instantiation bits)
enum R7Bits : int { kR7Proj = 512 };  // r7-comm2 HT13: projection tail (instantiation bit, bf16 one/two-shot only)
constexpr uint64_t kModeSrc = 256;  // r3 G26: descriptor mode bit: the payload is at ctrl.src_off, not in the send slot

#define CHECK_CUDA(x)                                                                        \
  do {                                                                                       \
    cudaError_t e_ = (x);                                                                    \
    if (e_ != cudaSuccess)                                                                   \
      throw std::runtime_error(std::string("rdma_ar CUDA: ") + cudaGetErrorString(e_) +      \
                               " at " #x);                                                   \
  } while (0)
#define CHECK_IB(cond, what)                                                                 \
  do {                                                                                       \
    if (!(cond))                                                                             \
      throw std::runtime_error(std::string("rdma_ar ibverbs: ") + (what) +                   \
                               " errno=" + std::to_string(errno) + " " + strerror(errno));  \
  } while (0)

struct alignas(64) Ctrl {
  uint64_t send_seq;    // GPU -> proxy: phase-1 message `seq` is staged in send slot seq % kSlots
  uint64_t send_bytes;  // whole message bytes (written before send_seq)
  uint64_t mode;        // Mode
  uint64_t chunk_vecs;  // two-shot chunk size in 16-byte vectors
  uint64_t src_off;     // r3 G26: arena offset of the payload when mode & kModeSrc (was pad0[0]; layout unchanged)
  uint64_t pad0[3];
  uint64_t posted_seq;  // proxy -> GPU: phase-1 writes for this seq have been posted
  uint64_t pad1[7];
  uint64_t send2_seq;   // GPU -> proxy: two-shot reduced chunk staged in send2 slot
  uint64_t send2_bytes;
  uint64_t pad2[6];
  uint64_t posted2_seq;  // proxy -> GPU: phase-2 writes posted
  uint64_t pad3[7];
  uint64_t pub_time;    // GPU globaltimer at the phase-1 publish (for proxy hand-off timing)
  uint64_t pad4[7];
  uint64_t ping;        // GPU -> proxy hand-off probe
  uint64_t pad5[7];
  uint64_t pong;        // proxy echo
  uint64_t pad6[7];
};
static_assert(sizeof(Ctrl) == 448 && offsetof(Ctrl, posted_seq) == 64 && offsetof(Ctrl, src_off) == 32,
              "r3: Ctrl layout must not move");

struct DevState {
  unsigned long long seq;  // last completed op
  unsigned int arrive;     // CTA arrival counter, phase 1 staging
  unsigned int arrive2;    // CTA arrival counter, phase 2 staging
  unsigned int depart;     // CTA departure counter
  // timing (ns, globaltimer; recorded by thread 0 of CTA 0 / the publishing thread)
  unsigned long long t_start, t_pub;
  long long sum_stage, sum_net, sum_tail, sum_total;  // start->pub, pub->flags, flags->end, start->end
  unsigned long long n_timed;
  unsigned int entered;  // r3 G06 (iv): CTA entry counter of ar_kernel_r3 (appended: existing offsets unchanged)
  unsigned int arrive3;  // r7-comm2 HT13: CTA arrival counter of the projection tail (appended; back to 0 after each op)
};

struct Params {
  const uint4* in;
  uint4* out;
  long long nvec;        // message size in 16-byte vectors
  long long chunk;       // two-shot chunk size in vectors
  char* send;            // kSlots * max_bytes
  char* send2;           // kSlots * max_chunk
  const char* recv;      // kSlots * world * max_bytes
  const char* recv2;     // kSlots * world * max_chunk
  const uint64_t* flags;   // kSlots * world
  const uint64_t* flags2;  // kSlots * world
  Ctrl* ctrl;
  DevState* st;
  long long max_bytes, max_chunk;
  int rank, world;
  long long timeout_ns;
  int timing;  // record per-call stage timings (RDMA_AR_TIMING=1)
  long long seg;  // g2 colltrim kGatherCols only: input row width in 16-byte vectors (output row = world * seg)
};

// r7-comm2 HT13: the projection operands of one kR7Proj launch (a second kernel parameter; zero for every other launch)
struct R7Proj {
  const float* d;   // unit direction, fp32 [kR7H], 16-byte aligned (dsv41_oblit._vec)
  uint4* y;         // projected output, bf16 [rows, kR7H], contiguous, 16-byte aligned
  long long rows;   // 1 .. kR7MaxRows
};
constexpr int kR7H = 5120;        // DSv4.1 hidden size: _proj_rows H (BLOCK 8192, num_warps 8, 256 threads per row)
constexpr int kR7MaxRows = 6;     // two 256-thread groups x kR7RowsPerGroup rows
constexpr int kR7RowsPerGroup = 3;

__device__ __forceinline__ uint64_t ld_acquire_sys(const uint64_t* p) {
  uint64_t v;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_release_sys(uint64_t* p, uint64_t v) {
  asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
// r3-comm helpers (used by ar_kernel_r3 only)
__device__ __forceinline__ uint64_t ld_relaxed_sys(const uint64_t* p) {
  uint64_t v;
  asm volatile("ld.relaxed.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_relaxed_sys(uint64_t* p, uint64_t v) {
  asm volatile("st.relaxed.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ unsigned int atom_add_acqrel_gpu(unsigned int* p, unsigned int v) {
  unsigned int old;
  asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory");
  return old;
}
__device__ __forceinline__ void r3_trap() {
  asm volatile("trap;");
}
// r6-comm C2 (sm_90+ PTX): wait for the prerequisite grid (completion + memory flush); allow dependents to launch
__device__ __forceinline__ void r6_gdc_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void r6_gdc_launch() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
// r7-comm2 HT13: bf16 halves of a 32-bit word, widened exactly to fp32 (the served kernel's cvt.f32.bf16)
__device__ __forceinline__ float r7_lo(unsigned w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float r7_hi(unsigned w) { return __uint_as_float(w & 0xffff0000u); }
__device__ __forceinline__ void r7_unpack(const uint4 v, float* f) {
  f[0] = r7_lo(v.x); f[1] = r7_hi(v.x); f[2] = r7_lo(v.y); f[3] = r7_hi(v.y);
  f[4] = r7_lo(v.z); f[5] = r7_hi(v.z); f[6] = r7_lo(v.w); f[7] = r7_hi(v.w);
}
__device__ __forceinline__ unsigned r7_pack(float lo, float hi) {  // cvt.rn.bf16x2.f32 d, hi, lo (RN, one rounding)
  const __nv_bfloat162 b = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const unsigned*>(&b);
}
// The served dsv41_oblit._proj_rows program (Triton 3.7.1, sm_121a, H 5120, BLOCK 8192, 8 warps), row by row, on rows
// g, g + 2, g + 4 of out (g = this thread's 256-thread group). Every FP op and its order are the cubin's (ref/triton/):
//   acc = x[1] * d[1]; acc = fma(x[j], d[j], acc) for j = 0, 2..31 (register order: chunk c = 0..3 at c*2048 + t*8,
//   8 elements each; masked elements are +0.0); warp butterfly xor 16, 8, 4, 2, 1 (acc + shfl(acc)); lane 0 -> smem[w];
//   lanes 0..7 of warp 0: butterfly xor 4, 2, 1 over smem[lane]; lane 0 -> smem[0]; y = bf16_rn(fma(-s, d, x)).
// All threads of the CTA must call it (two __syncthreads).
__device__ __forceinline__ void r7_project(const uint4* __restrict__ xo, const R7Proj& q) {
  __shared__ float s_w[2][kR7RowsPerGroup][8];
  __shared__ float s_s[2][kR7RowsPerGroup];
  const int g = (int)(threadIdx.x >> 8);
  const int t = (int)(threadIdx.x & 255u);
  const int lane = t & 31, warp = t >> 5;
  float d[32];
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const int e = c * 2048 + t * 8;
    float4 a = make_float4(0.f, 0.f, 0.f, 0.f), b = a;
    if (e < kR7H) {
      a = __ldcg(reinterpret_cast<const float4*>(q.d + e));
      b = __ldcg(reinterpret_cast<const float4*>(q.d + e + 4));
    }
    d[c * 8 + 0] = a.x; d[c * 8 + 1] = a.y; d[c * 8 + 2] = a.z; d[c * 8 + 3] = a.w;
    d[c * 8 + 4] = b.x; d[c * 8 + 5] = b.y; d[c * 8 + 6] = b.z; d[c * 8 + 7] = b.w;
  }
  uint4 xv[kR7RowsPerGroup][4];
#pragma unroll
  for (int k = 0; k < kR7RowsPerGroup; ++k) {
    const long long row = 2 * k + g;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const int e = c * 2048 + t * 8;
      xv[k][c] = make_uint4(0u, 0u, 0u, 0u);
      if (row < q.rows && e < kR7H) xv[k][c] = __ldcg(xo + row * (kR7H / 8) + e / 8);
    }
  }
  float acc[kR7RowsPerGroup];
#pragma unroll
  for (int k = 0; k < kR7RowsPerGroup; ++k) {
    float x[32];
#pragma unroll
    for (int c = 0; c < 4; ++c) r7_unpack(xv[k][c], x + c * 8);
    float a = __fmul_rn(x[1], d[1]);
    a = __fmaf_rn(x[0], d[0], a);
#pragma unroll
    for (int j = 2; j < 32; ++j) a = __fmaf_rn(x[j], d[j], a);
#pragma unroll
    for (int m = 16; m >= 1; m >>= 1) a = __fadd_rn(a, __shfl_xor_sync(0xffffffffu, a, m));
    acc[k] = a;
  }
  if (lane == 0) {
#pragma unroll
    for (int k = 0; k < kR7RowsPerGroup; ++k) s_w[g][k][warp] = acc[k];
  }
  __syncthreads();
  if (warp == 0) {
#pragma unroll
    for (int k = 0; k < kR7RowsPerGroup; ++k) {
      float v = s_w[g][k][lane & 7];
#pragma unroll
      for (int m = 4; m >= 1; m >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, m));
      if (lane == 0) s_s[g][k] = v;
    }
  }
  __syncthreads();
#pragma unroll
  for (int k = 0; k < kR7RowsPerGroup; ++k) {
    const long long row = 2 * k + g;
    if (row >= q.rows) continue;
    const float ns = -s_s[g][k];
    float x[32];
#pragma unroll
    for (int c = 0; c < 4; ++c) r7_unpack(xv[k][c], x + c * 8);
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const int e = c * 2048 + t * 8;
      if (e < kR7H) {
        float y[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) y[j] = __fmaf_rn(ns, d[c * 8 + j], x[c * 8 + j]);
        q.y[row * (kR7H / 8) + e / 8] = make_uint4(r7_pack(y[0], y[1]), r7_pack(y[2], y[3]), r7_pack(y[4], y[5]),
                                                   r7_pack(y[6], y[7]));
      }
    }
  }
}

__device__ __forceinline__ unsigned long long now_ns() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__device__ __forceinline__ void spin_until(const uint64_t* f, uint64_t seq, const Params& p, int who, int phase) {
  const unsigned long long t0 = now_ns();
  while (ld_acquire_sys(f) < seq) {
    if (now_ns() - t0 > (unsigned long long)p.timeout_ns) {
      printf("rdma_ar rank %d: timeout waiting for %d phase %d (seq %llu)\n", p.rank, who, phase,
             (unsigned long long)seq);
      asm volatile("trap;");
    }
  }
}

// g130 RDMA_AR_HYG_POLL: same wait and same timeout contract; the clock is read on the first miss and on every 1024th
// pass instead of on every pass, so a timeout can be late by about 1024 polls (about 1 ms against a 600 s limit).
__device__ __forceinline__ void spin_until_thin(const uint64_t* f, uint64_t seq, const Params& p, int who, int phase) {
  if (ld_acquire_sys(f) >= seq) return;
  const unsigned long long t0 = now_ns();
  unsigned int it = 0;
  while (ld_acquire_sys(f) < seq) {
    if ((++it & 1023u) == 0 && now_ns() - t0 > (unsigned long long)p.timeout_ns) {
      printf("rdma_ar rank %d: timeout waiting for %d phase %d (seq %llu)\n", p.rank, who, phase,
             (unsigned long long)seq);
      asm volatile("trap;");
    }
  }
}
// kPub without kPollThin expands to exactly the production spin_until call.
#define G130_SPIN(f, s, who, phase)                   \
  do {                                                \
    if constexpr ((kPub & kPollThin) != 0)            \
      spin_until_thin((f), (s), p, (who), (phase));   \
    else                                              \
      spin_until((f), (s), p, (who), (phase));        \
  } while (0)

// True for exactly one thread: thread 0 of the last CTA through a grid-wide arrival counter (all CTAs'
// prior writes, fenced at system scope, are visible to it). That thread then publishes to the proxy.
__device__ __forceinline__ bool last_cta(unsigned int* arrive, int* s_last) {
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) *s_last = (atomicAdd(arrive, 1u) == gridDim.x - 1);
  __syncthreads();
  if (*s_last && threadIdx.x == 0) {
    *arrive = 0;
    __threadfence_system();
    return true;
  }
  return false;
}

__device__ __forceinline__ void acc8(float* acc, const uint4& v) {
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    float2 f = __bfloat1622float2(h[k]);
    acc[2 * k] += f.x;
    acc[2 * k + 1] += f.y;
  }
}
__device__ __forceinline__ uint4 pack8(const float* acc) {
  uint4 o;
  __nv_bfloat162* oh = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
  for (int k = 0; k < 4; ++k) oh[k] = __floats2bfloat162_rn(acc[2 * k], acc[2 * k + 1]);
  return o;
}

// dtype-generic accumulate/pack over one 16-byte vector: 8 bf16 or (kF32) 4 fp32 values.
// fp32 inputs are summed in fp32 in the same fixed rank order (DSv4.1 addition, 2026-10-21).
template <bool kF32>
__device__ __forceinline__ void accv(float* acc, const uint4& v) {
  if constexpr (kF32) {
    acc[0] += __uint_as_float(v.x);
    acc[1] += __uint_as_float(v.y);
    acc[2] += __uint_as_float(v.z);
    acc[3] += __uint_as_float(v.w);
  } else {
    acc8(acc, v);
  }
}
template <bool kF32>
__device__ __forceinline__ uint4 packv(const float* acc) {
  if constexpr (kF32) {
    uint4 o;
    o.x = __float_as_uint(acc[0]);
    o.y = __float_as_uint(acc[1]);
    o.z = __float_as_uint(acc[2]);
    o.w = __float_as_uint(acc[3]);
    return o;
  } else {
    return pack8(acc);
  }
}

// GPU<->proxy round trip through host-pinned memory, timed on the GPU clock only.
__global__ void pingpong_kernel(Ctrl* ctrl, int n, unsigned long long base, unsigned long long* out_ns) {
  const unsigned long long t0 = now_ns();
  for (int i = 1; i <= n; ++i) {
    st_release_sys(&ctrl->ping, base + i);
    while (ld_acquire_sys(&ctrl->pong) != base + i) {
    }
  }
  *out_ns = now_ns() - t0;
}

template <int kMode, bool kF32>
__global__ void __launch_bounds__(kThreads) ar_kernel(Params p) {
  __shared__ unsigned long long s_seq;
  __shared__ int s_last;
  if (threadIdx.x == 0) s_seq = *reinterpret_cast<volatile unsigned long long*>(&p.st->seq) + 1;
  if (p.timing && threadIdx.x == 0 && blockIdx.x == 0) p.st->t_start = now_ns();
  __syncthreads();
  const unsigned long long seq = s_seq;
  const int slot = static_cast<int>(seq % kSlots);
  const long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  const long long stride = (long long)gridDim.x * blockDim.x;

  // 1) stage the input in this rank's send slot and hand it to the proxy
  uint4* send4 = reinterpret_cast<uint4*>(p.send + slot * p.max_bytes);
  for (long long i = tid; i < p.nvec; i += stride) send4[i] = p.in[i];
  if (last_cta(&p.st->arrive, &s_last)) {
    // the proxy must have consumed the previous message on this channel before its descriptor is
    // overwritten (compare with what was last published, not seq-1: modes interleave)
    spin_until(&p.ctrl->posted_seq, ld_acquire_sys(&p.ctrl->send_seq), p, -1, 0);
    *reinterpret_cast<volatile uint64_t*>(&p.ctrl->send_bytes) = (uint64_t)p.nvec * 16;
    *reinterpret_cast<volatile uint64_t*>(&p.ctrl->mode) = (uint64_t)kMode;
    *reinterpret_cast<volatile uint64_t*>(&p.ctrl->chunk_vecs) = (uint64_t)p.chunk;
    if (p.timing) {
      const unsigned long long tp = now_ns();
      p.st->t_pub = tp;
      *reinterpret_cast<volatile uint64_t*>(&p.ctrl->pub_time) = tp;
    }
    st_release_sys(&p.ctrl->send_seq, seq);
  }

  // 2) wait for every peer's phase-1 contribution
  if (threadIdx.x < p.world && threadIdx.x != p.rank)
    spin_until(p.flags + slot * p.world + threadIdx.x, seq, p, threadIdx.x, 1);
  __syncthreads();
  unsigned long long t_flags = 0;
  if (p.timing && threadIdx.x == 0 && blockIdx.x == 0) t_flags = now_ns();

  const char* rbase = p.recv + (long long)slot * p.world * p.max_bytes;
  if constexpr (kMode == kGather) {
    for (int r = 0; r < p.world; ++r) {
      const uint4* src = reinterpret_cast<const uint4*>(rbase + (long long)r * p.max_bytes);
      uint4* dst = p.out + (long long)r * p.nvec;
      for (long long i = tid; i < p.nvec; i += stride) dst[i] = (r == p.rank) ? p.in[i] : __ldcv(src + i);
    }
  } else if constexpr (kMode == kGatherCols) {
    // g2 colltrim: vector i of rank r's input (row i / seg, column i % seg) goes to output row i / seg,
    // column r * seg + i % seg. Same loads as kGather; only the destination index differs.
    const unsigned useg = (unsigned)p.seg;
    const long long orow = (long long)p.world * p.seg;
    for (int r = 0; r < p.world; ++r) {
      const uint4* src = reinterpret_cast<const uint4*>(rbase + (long long)r * p.max_bytes);
      uint4* dst = p.out + (long long)r * p.seg;
      for (long long i = tid; i < p.nvec; i += stride) {
        const long long row = (long long)((unsigned)i / useg);
        dst[row * orow + (i - row * p.seg)] = (r == p.rank) ? p.in[i] : __ldcv(src + i);
      }
    }
  } else if constexpr (kMode == kOneShot) {
    for (long long i = tid; i < p.nvec; i += stride) {
      float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      for (int r = 0; r < p.world; ++r)
        accv<kF32>(acc, (r == p.rank) ? p.in[i]
                                      : __ldcv(reinterpret_cast<const uint4*>(rbase + (long long)r * p.max_bytes) + i));
      p.out[i] = packv<kF32>(acc);
    }
  } else {
    // two-shot: peers wrote chunk `rank` of their input at offset 0 of recv[slot][peer]
    const long long b = (long long)p.rank * p.chunk;
    const long long len = max(0LL, min(p.chunk, p.nvec - b));
    uint4* send2 = reinterpret_cast<uint4*>(p.send2 + slot * p.max_chunk);
    for (long long i = tid; i < len; i += stride) {
      float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      for (int r = 0; r < p.world; ++r)
        accv<kF32>(acc, (r == p.rank) ? p.in[b + i]
                                      : __ldcv(reinterpret_cast<const uint4*>(rbase + (long long)r * p.max_bytes) + i));
      const uint4 o = packv<kF32>(acc);
      p.out[b + i] = o;
      send2[i] = o;
    }
    if (last_cta(&p.st->arrive2, &s_last)) {
      spin_until(&p.ctrl->posted2_seq, ld_acquire_sys(&p.ctrl->send2_seq), p, -2, 0);
      *reinterpret_cast<volatile uint64_t*>(&p.ctrl->send2_bytes) = (uint64_t)len * 16;
      st_release_sys(&p.ctrl->send2_seq, seq);
    }
    if (threadIdx.x < p.world && threadIdx.x != p.rank)
      spin_until(p.flags2 + slot * p.world + threadIdx.x, seq, p, threadIdx.x, 2);
    __syncthreads();
    const char* r2 = p.recv2 + (long long)slot * p.world * p.max_chunk;
    for (int r = 0; r < p.world; ++r) {
      if (r == p.rank) continue;
      const long long br = (long long)r * p.chunk;
      const long long lr = max(0LL, min(p.chunk, p.nvec - br));
      const uint4* src = reinterpret_cast<const uint4*>(r2 + (long long)r * p.max_chunk);
      for (long long i = tid; i < lr; i += stride) p.out[br + i] = __ldcv(src + i);
    }
  }

  // 3) last CTA out advances the sequence for the next call
  __syncthreads();
  if (p.timing && threadIdx.x == 0 && blockIdx.x == 0) {
    __threadfence();
    const unsigned long long t_end = now_ns();
    const unsigned long long ts = p.st->t_start;
    const unsigned long long tp = *reinterpret_cast<volatile unsigned long long*>(&p.st->t_pub);
    atomicAdd(reinterpret_cast<unsigned long long*>(&p.st->sum_stage), (unsigned long long)(long long)(tp - ts));
    atomicAdd(reinterpret_cast<unsigned long long*>(&p.st->sum_net), (unsigned long long)(long long)(t_flags - tp));
    atomicAdd(reinterpret_cast<unsigned long long*>(&p.st->sum_tail), (unsigned long long)(long long)(t_end - t_flags));
    atomicAdd(reinterpret_cast<unsigned long long*>(&p.st->sum_total), (unsigned long long)(long long)(t_end - ts));
    atomicAdd(&p.st->n_timed, 1ull);
  }
  if (threadIdx.x == 0 && atomicAdd(&p.st->depart, 1u) == gridDim.x - 1) {
    p.st->depart = 0;
    __threadfence();
    p.st->seq = seq;
  }
}

// ---- k1/3-rdma-loads (2026-10-24): parallel-load variant of ar_kernel, world fixed at kW --------
// ar_kernel above is left byte-for-byte unchanged (it is the legacy path, RDMA_AR_PARLOAD=0).
// The legacy consumer loops issue one uncached sysmem load per peer and consume it before the
// next load issues (SASS: LDG.E.128.STRONG.SYS -> 8 FADD -> LDG ...), so each pass pays ~7
// serial GPU<->C2C round trips. Here every thread first loads all kW peer vectors into distinct
// registers, then accumulates them in the SAME r = 0..kW-1 fp32 order and rounds once, so results
// are bitwise identical to ar_kernel. The phase-2 copy and the gather issue all loads before any
// store; the send-slot staging copy is batched x4. Protocol (flags, seq, slots, publish) unchanged.
// __launch_bounds__(kThreads, 1): with the default (no min-blocks) ptxas caps the kernel at 64 regs to
// keep 2 CTAs/SM and re-uses load registers, so only ~4-5 peer loads are in flight before the first
// FADD (two round trips). With min 1 CTA/SM it keeps all 8 in flight (SASS gate: depth 8). The grid
// is at most 16 CTAs on 48 SMs, so one CTA per SM costs nothing.
template <int kMode, bool kF32, int kW, int kPub>
__global__ void __launch_bounds__(kThreads, 1) ar_kernel_pl(Params p) {
  __shared__ unsigned long long s_seq;
  __shared__ int s_last;
  if (threadIdx.x == 0) s_seq = *reinterpret_cast<volatile unsigned long long*>(&p.st->seq) + 1;
  if (p.timing && threadIdx.x == 0 && blockIdx.x == 0) p.st->t_start = now_ns();
  __syncthreads();
  const unsigned long long seq = s_seq;
  const int slot = static_cast<int>(seq % kSlots);
  const long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  const long long stride = (long long)gridDim.x * blockDim.x;

  // 1) stage the input in this rank's send slot (x4 batched: 4 loads in flight, then 4 stores)
  uint4* send4 = reinterpret_cast<uint4*>(p.send + slot * p.max_bytes);
  {
    long long i = tid;
    for (; i + 3 * stride < p.nvec; i += 4 * stride) {
      const uint4 a0 = p.in[i];
      const uint4 a1 = p.in[i + stride];
      const uint4 a2 = p.in[i + 2 * stride];
      const uint4 a3 = p.in[i + 3 * stride];
      send4[i] = a0;
      send4[i + stride] = a1;
      send4[i + 2 * stride] = a2;
      send4[i + 3 * stride] = a3;
    }
    for (; i < p.nvec; i += stride) send4[i] = p.in[i];
  }
  if (last_cta(&p.st->arrive, &s_last)) {
    // g130 V1: every op publishes phase 1 exactly once and st->seq counts ops, so what was last published here is
    // seq - 1 and the send_seq load adds nothing. The guard itself must stay: this rank's kernel can finish while its own
    // proxy is stalled (its flags come from the peers' proxies), and the descriptor is a single mailbox.
    if constexpr ((kPub & kPubV1) != 0)
      G130_SPIN(&p.ctrl->posted_seq, seq - 1, -1, 0);
    else
      G130_SPIN(&p.ctrl->posted_seq, ld_acquire_sys(&p.ctrl->send_seq), -1, 0);
    *reinterpret_cast<volatile uint64_t*>(&p.ctrl->send_bytes) = (uint64_t)p.nvec * 16;
    *reinterpret_cast<volatile uint64_t*>(&p.ctrl->mode) = (uint64_t)kMode;
    *reinterpret_cast<volatile uint64_t*>(&p.ctrl->chunk_vecs) = (uint64_t)p.chunk;
    if (p.timing) {
      const unsigned long long tp = now_ns();
      p.st->t_pub = tp;
      *reinterpret_cast<volatile uint64_t*>(&p.ctrl->pub_time) = tp;
    }
    st_release_sys(&p.ctrl->send_seq, seq);
  }

  // 2) wait for every peer's phase-1 contribution
  if (threadIdx.x < p.world && threadIdx.x != p.rank)
    G130_SPIN(p.flags + slot * p.world + threadIdx.x, seq, threadIdx.x, 1);
  __syncthreads();
  unsigned long long t_flags = 0;
  if (p.timing && threadIdx.x == 0 && blockIdx.x == 0) t_flags = now_ns();

  const char* rbase = p.recv + (long long)slot * p.world * p.max_bytes;
  if constexpr (kMode == kGather) {
    for (long long i = tid; i < p.nvec; i += stride) {
      uint4 v[kW];
#pragma unroll
      for (int r = 0; r < kW; ++r)
        v[r] = (r == p.rank) ? p.in[i] : __ldcv(reinterpret_cast<const uint4*>(rbase + (long long)r * p.max_bytes) + i);
#pragma unroll
      for (int r = 0; r < kW; ++r) p.out[(long long)r * p.nvec + i] = v[r];
    }
  } else if constexpr (kMode == kGatherCols) {
    // g2 colltrim: as kGather above (all kW loads in flight before any store), stores to the column slot
    const unsigned useg = (unsigned)p.seg;
    for (long long i = tid; i < p.nvec; i += stride) {
      uint4 v[kW];
#pragma unroll
      for (int r = 0; r < kW; ++r)
        v[r] = (r == p.rank) ? p.in[i] : __ldcv(reinterpret_cast<const uint4*>(rbase + (long long)r * p.max_bytes) + i);
      const long long row = (long long)((unsigned)i / useg);
      uint4* dst = p.out + row * ((long long)kW * p.seg) + (i - row * p.seg);
#pragma unroll
      for (int r = 0; r < kW; ++r) dst[(long long)r * p.seg] = v[r];
    }
  } else if constexpr (kMode == kOneShot) {
    for (long long i = tid; i < p.nvec; i += stride) {
      uint4 v[kW];
#pragma unroll
      for (int r = 0; r < kW; ++r)
        v[r] = (r == p.rank) ? p.in[i] : __ldcv(reinterpret_cast<const uint4*>(rbase + (long long)r * p.max_bytes) + i);
      float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int r = 0; r < kW; ++r) accv<kF32>(acc, v[r]);
      p.out[i] = packv<kF32>(acc);
    }
  } else {
    // two-shot: peers wrote chunk `rank` of their input at offset 0 of recv[slot][peer]
    const long long b = (long long)p.rank * p.chunk;
    const long long len = max(0LL, min(p.chunk, p.nvec - b));
    uint4* send2 = reinterpret_cast<uint4*>(p.send2 + slot * p.max_chunk);
    for (long long i = tid; i < len; i += stride) {
      uint4 v[kW];
#pragma unroll
      for (int r = 0; r < kW; ++r)
        v[r] = (r == p.rank) ? p.in[b + i] : __ldcv(reinterpret_cast<const uint4*>(rbase + (long long)r * p.max_bytes) + i);
      float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int r = 0; r < kW; ++r) accv<kF32>(acc, v[r]);
      const uint4 o = packv<kF32>(acc);
      p.out[b + i] = o;
      send2[i] = o;
    }
    if (last_cta(&p.st->arrive2, &s_last)) {
      // g130 V2: no guard is needed. This thread is past the phase-1 wait of seq, so all 7 peers published phase 1 of seq,
      // so each of them finished the previous two-shot op seq' < seq on its single stream, so each of them saw this
      // rank's phase-2 flag of seq': this rank's proxy has already read send2_bytes and posted seq' to all 7 peers. (seq - 1
      // would be WRONG here: send2_seq only moves on two-shot ops, modes interleave, and the spin would never end.)
      if constexpr ((kPub & kPubV2) == 0)
        G130_SPIN(&p.ctrl->posted2_seq, ld_acquire_sys(&p.ctrl->send2_seq), -2, 0);
      *reinterpret_cast<volatile uint64_t*>(&p.ctrl->send2_bytes) = (uint64_t)len * 16;
      st_release_sys(&p.ctrl->send2_seq, seq);
    }
    if (threadIdx.x < p.world && threadIdx.x != p.rank)
      G130_SPIN(p.flags2 + slot * p.world + threadIdx.x, seq, threadIdx.x, 2);
    __syncthreads();
    const char* r2 = p.recv2 + (long long)slot * p.world * p.max_chunk;
    // legacy: for r != rank, for i < lr(r) = clamp(nvec - r*chunk, 0, chunk): out[r*chunk+i] = recv2[r][i].
    // Same element set (i < chunk and r*chunk + i < nvec), all kW-1 loads issued before the stores.
    for (long long i = tid; i < p.chunk; i += stride) {
      uint4 v[kW];
#pragma unroll
      for (int r = 0; r < kW; ++r)
        if (r != p.rank && (long long)r * p.chunk + i < p.nvec)
          v[r] = __ldcv(reinterpret_cast<const uint4*>(r2 + (long long)r * p.max_chunk) + i);
#pragma unroll
      for (int r = 0; r < kW; ++r)
        if (r != p.rank && (long long)r * p.chunk + i < p.nvec) p.out[(long long)r * p.chunk + i] = v[r];
    }
  }

  // 3) last CTA out advances the sequence for the next call
  __syncthreads();
  if (p.timing && threadIdx.x == 0 && blockIdx.x == 0) {
    __threadfence();
    const unsigned long long t_end = now_ns();
    const unsigned long long ts = p.st->t_start;
    const unsigned long long tp = *reinterpret_cast<volatile unsigned long long*>(&p.st->t_pub);
    atomicAdd(reinterpret_cast<unsigned long long*>(&p.st->sum_stage), (unsigned long long)(long long)(tp - ts));
    atomicAdd(reinterpret_cast<unsigned long long*>(&p.st->sum_net), (unsigned long long)(long long)(t_flags - tp));
    atomicAdd(reinterpret_cast<unsigned long long*>(&p.st->sum_tail), (unsigned long long)(long long)(t_end - t_flags));
    atomicAdd(reinterpret_cast<unsigned long long*>(&p.st->sum_total), (unsigned long long)(long long)(t_end - ts));
    atomicAdd(&p.st->n_timed, 1ull);
  }
  if (threadIdx.x == 0 && atomicAdd(&p.st->depart, 1u) == gridDim.x - 1) {
    p.st->depart = 0;
    __threadfence();
    p.st->seq = seq;
  }
}

// ---- r3-comm (climb round 3, 2026-11-08): G06 critical-path trim + G26 send-buffer input ---------------------------
// A SEPARATE kernel: ar_kernel_pl above is byte-for-byte unchanged and stays the kernel of every launch unless
// DSV41_RDMA_TRIM=1. ar_kernel_r3 has the same Params, Ctrl, slots, flags, wire protocol and consumer loops (section 2
// is copied VERBATIM from ar_kernel_pl: tests/test_source.py diffs it), so every output element sees the same loads,
// the same fp32 FADD sequence r = 0..7 from +0.0 and the same single bf16 RN rounding. Only staging, publish and seq
// bookkeeping differ. It sits on the production g130 mask 7 (V1 + V2 + thin poll; V2 dropped outside two-shot).
//   (ii)  thread 0 reads posted_seq (relaxed, sys) at entry; the publisher re-spins only when that early value is below
//         seq - 1. posted_seq is monotonic, so a passing early value is a valid proof that the proxy has consumed
//         message seq - 1; the fence.sc.sys every publish path executes between that read and the descriptor stores
//         makes read + fence an acquire pattern (descriptor stores cannot pass it).
//   (iv)  st->seq advances at ENTRY: thread 0 of every CTA reads it before the entry barrier; then the last thread of
//         each CTA does atom.acq_rel.gpu on st->entered and the last CTA through writes seq (and resets entered).
//         No tail atomic, tail fence or tail barrier (a barrier only when timing). The next kernel on the stream reads
//         seq after this one has completed, exactly as with the tail update.
//   (i)   kR3S1: a message of <= DSV41_RDMA_TRIM_S1_BYTES is staged by CTA 0 alone (4 loads in flight per thread,
//         then the stores), fenced at sys scope per thread, and published by thread 0 after one bar.sync: no grid
//         atomic, no second fence, no second barrier. Other CTAs do not stage; they wait for the flags as before
//         (their local term is read from p.in, as in ar_kernel_pl).
//   (iii) kR3P2: two-shot with chunk <= blockDim: every phase-2 element (i < len <= chunk) belongs to CTA 0, so CTA 0
//         publishes phase 2 alone (fence per thread + bar.sync, as (i)), and CTAs >= 1, which have no consumer work in
//         either phase, leave right after the phase-1 publish. The host picks P2 only when chunk <= 512; the kernel
//         traps otherwise.
//   G26   kR3Sbuf: p.in IS a registered send buffer (RdmaAR::send_buffer). No staging: CTA 0 thread 0 fences and
//         publishes with mode | kModeSrc and ctrl.src_off = p.in - arena. Reuse rule: a buffer used by op s0 may be
//         rewritten only after a LATER op s' (s0 < s' < s1) has passed its phase-1 wait on this rank (then every peer
//         finished s' - 1 >= s0, so it has our s0 payload and our NIC is done reading it). The 16 bytes before each
//         buffer hold the seq of its last use; the kernel traps when s1 - s0 < 2 (fail loud, never silent).
template <int kMode, int kPub>
__device__ __forceinline__ void r3_publish1(const Params& p, unsigned long long seq, uint64_t early, uint64_t src_off) {
  if (early < seq - 1) G130_SPIN(&p.ctrl->posted_seq, seq - 1, -1, 0);
  *reinterpret_cast<volatile uint64_t*>(&p.ctrl->send_bytes) = (uint64_t)p.nvec * 16;
  if constexpr ((kPub & kR3Sbuf) != 0) {
    *reinterpret_cast<volatile uint64_t*>(&p.ctrl->src_off) = src_off;
    *reinterpret_cast<volatile uint64_t*>(&p.ctrl->mode) = (uint64_t)kMode | kModeSrc;
  } else {
    *reinterpret_cast<volatile uint64_t*>(&p.ctrl->mode) = (uint64_t)kMode;
  }
  *reinterpret_cast<volatile uint64_t*>(&p.ctrl->chunk_vecs) = (uint64_t)p.chunk;
  if (p.timing) {
    const unsigned long long tp = now_ns();
    p.st->t_pub = tp;
    *reinterpret_cast<volatile uint64_t*>(&p.ctrl->pub_time) = tp;
  }
  st_release_sys(&p.ctrl->send_seq, seq);
}

// r3-comm: the r3 bits of one launch, on top of the g130 mask (host side; the CPU emulator calls this same function)
inline int r3_bits(int mode, long long nvec, long long chunk, long long s1_bytes, bool sbuf, long long threads) {
  int b = kR3Trim;
  if (sbuf)
    b |= kR3Sbuf;
  else if (nvec * 16 <= s1_bytes)
    b |= kR3S1;
  if (mode == kTwoShot && chunk <= threads) b |= kR3P2;
  return b;
}

// r3-comm (review N5): S1 (CTA 0 stages for every CTA, so CTAs >= 1 may write p.out before p.in is staged) and Sbuf (the
// NIC reads p.in until every peer has it) are not in-place safe. True when [in, in + nvec*16) overlaps the output and the
// overlap is not the one harmless alias, a dim-0 gather whose input IS its own output segment (rewritten with identical
// bytes). The caller then drops S1/Sbuf and uses the staged r3 path, whose staging thread == consuming thread per element,
// as in ar_kernel_pl. Serving never aliases today (all_reduce: out = empty_like(x); gathers: out = torch.empty).
inline bool r3_inplace_conflict(int mode, unsigned long long in, unsigned long long out, long long nvec, int rank,
                                int world) {
  const unsigned long long nb = (unsigned long long)nvec * 16ull;
  const unsigned long long ob = (mode == kGather || mode == kGatherCols) ? nb * (unsigned long long)world : nb;
  if (nb == 0 || in >= out + ob || out >= in + nb) return false;
  return !(mode == kGather && in == out + (unsigned long long)rank * nb);
}

template <int kMode, bool kF32, int kW, int kPub>
__global__ void __launch_bounds__(kThreads, 1) ar_kernel_r3(Params p, R7Proj q) {
  static_assert((kPub & kR3Trim) != 0 && (kPub & kPubV1) != 0 && (kPub & kPollThin) != 0, "r3 sits on g130 mask 7");
  static_assert(kMode != kTwoShot || (kPub & kPubV2) != 0, "r3 two-shot needs V2 (no phase-2 guard)");
  static_assert((kPub & kR3P2) == 0 || kMode == kTwoShot, "P2 is two-shot only");
  static_assert((kPub & kR3S1) == 0 || (kPub & kR3Sbuf) == 0, "S1 and Sbuf exclude each other");
  constexpr bool kOne = (kPub & (kR3S1 | kR3Sbuf)) != 0;  // CTA 0 alone publishes phase 1
  static_assert((kPub & kR6Trig) == 0 || (kPub & kR6Pdl) != 0, "r6: trig needs the PDL launch");
  static_assert((kPub & kR6Pdl) == 0 || (kPub & kR3Sbuf) == 0, "r6: PDL is not combined with the G26 send buffers");
  static_assert((kPub & kR7Proj) == 0 || (!kF32 && (kMode == kOneShot || kMode == kTwoShot) && (kPub & kR3Sbuf) == 0),
                "r7: the projection tail is bf16 one-shot / two-shot only, never with a send buffer");
  __shared__ unsigned long long s_seq;
  __shared__ int s_last;
  uint64_t early = 0;
  if constexpr ((kPub & kR6Pdl) != 0) {
    // r6 C2: the only access before the grid-dependency wait is this host-written, monotonic word (a stale value only
    // makes r3_publish1 spin, exactly as an early value below seq - 1 does in r3 (ii)). st->seq, st->entered, timing,
    // p.in, the slots, the flags and p.out are all touched after the wait, i.e. after the predecessor grid completed.
    if (threadIdx.x == 0 && (!kOne || blockIdx.x == 0)) early = ld_relaxed_sys(&p.ctrl->posted_seq);
    r6_gdc_wait();
  }
  if (threadIdx.x == 0) {
    s_seq = *reinterpret_cast<volatile unsigned long long*>(&p.st->seq) + 1;
    if constexpr ((kPub & kR6Pdl) == 0)
      if (!kOne || blockIdx.x == 0) early = ld_relaxed_sys(&p.ctrl->posted_seq);  // (ii)
    if constexpr ((kPub & kR3P2) != 0) {
      if (p.chunk > (long long)blockDim.x) {
        printf("rdma_ar r3 rank %d: P2 kernel with chunk %lld > %d threads\n", p.rank, p.chunk, (int)blockDim.x);
        r3_trap();
      }
    }
  }
  if (p.timing && threadIdx.x == 0 && blockIdx.x == 0) p.st->t_start = now_ns();
  __syncthreads();
  const unsigned long long seq = s_seq;
  // (iv) this CTA's thread 0 read st->seq before the barrier above; the acq_rel atomic orders that read before this
  // CTA's arrival, so the last CTA through writes seq only after every CTA has read the old value.
  if (threadIdx.x == blockDim.x - 1 && atom_add_acqrel_gpu(&p.st->entered, 1u) == gridDim.x - 1) {
    *reinterpret_cast<volatile unsigned int*>(&p.st->entered) = 0u;
    *reinterpret_cast<volatile unsigned long long*>(&p.st->seq) = seq;
  }
  const int slot = static_cast<int>(seq % kSlots);
  const long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  const long long stride = (long long)gridDim.x * blockDim.x;

  // 1) phase-1 publish
  uint4* send4 = reinterpret_cast<uint4*>(p.send + slot * p.max_bytes);
  if constexpr ((kPub & kR3Sbuf) != 0) {
    if (blockIdx.x == 0) {
      const unsigned hthr = blockDim.x > 32 ? 32u : blockDim.x - 1;  // off the publishing thread
      if (threadIdx.x == hthr) {
        uint64_t* hdr = reinterpret_cast<uint64_t*>(const_cast<uint4*>(p.in) - 1);
        const uint64_t last = ld_relaxed_sys(hdr);
        if (last != 0 && seq - last < 2) {
          printf("rdma_ar r3 rank %d: send buffer reused by the next op (seq %llu after %llu)\n", p.rank,
                 (unsigned long long)seq, (unsigned long long)last);
          r3_trap();
        }
        st_relaxed_sys(hdr, seq);
      }
      if (threadIdx.x == 0) {
        __threadfence_system();
        r3_publish1<kMode, kPub>(p, seq, early,
                                 (uint64_t)(reinterpret_cast<const char*>(p.in) - reinterpret_cast<const char*>(p.ctrl)));
      }
    }
  } else if constexpr ((kPub & kR3S1) != 0) {
    if (blockIdx.x == 0) {
      const long long nb = blockDim.x;
      for (long long i0 = threadIdx.x; i0 < p.nvec; i0 += 4 * nb) {
        const bool b1 = i0 + nb < p.nvec, b2 = i0 + 2 * nb < p.nvec, b3 = i0 + 3 * nb < p.nvec;
        uint4 a0, a1, a2, a3;
        a0 = p.in[i0];
        if (b1) a1 = p.in[i0 + nb];
        if (b2) a2 = p.in[i0 + 2 * nb];
        if (b3) a3 = p.in[i0 + 3 * nb];
        send4[i0] = a0;
        if (b1) send4[i0 + nb] = a1;
        if (b2) send4[i0 + 2 * nb] = a2;
        if (b3) send4[i0 + 3 * nb] = a3;
      }
      __threadfence_system();
      __syncthreads();
      if (threadIdx.x == 0) r3_publish1<kMode, kPub>(p, seq, early, 0);
    }
  } else {
    {
      long long i = tid;
      for (; i + 3 * stride < p.nvec; i += 4 * stride) {
        const uint4 a0 = p.in[i];
        const uint4 a1 = p.in[i + stride];
        const uint4 a2 = p.in[i + 2 * stride];
        const uint4 a3 = p.in[i + 3 * stride];
        send4[i] = a0;
        send4[i + stride] = a1;
        send4[i + 2 * stride] = a2;
        send4[i + 3 * stride] = a3;
      }
      for (; i < p.nvec; i += stride) send4[i] = p.in[i];
    }
    if (last_cta(&p.st->arrive, &s_last)) r3_publish1<kMode, kPub>(p, seq, early, 0);
  }
  if constexpr ((kPub & kR3P2) != 0) {
    if (blockIdx.x != 0) return;  // (iii) no consumer work here in either phase (len <= chunk <= blockDim)
  }

  // 2) wait for every peer's phase-1 contribution
  if (threadIdx.x < p.world && threadIdx.x != p.rank)
    G130_SPIN(p.flags + slot * p.world + threadIdx.x, seq, threadIdx.x, 1);
  __syncthreads();
  if constexpr ((kPub & kR6Trig) != 0 && kMode != kTwoShot) r6_gdc_launch();  // r6 C2: the last wait is behind us
  unsigned long long t_flags = 0;
  if (p.timing && threadIdx.x == 0 && blockIdx.x == 0) t_flags = now_ns();

  const char* rbase = p.recv + (long long)slot * p.world * p.max_bytes;
  if constexpr (kMode == kGather) {
    for (long long i = tid; i < p.nvec; i += stride) {
      uint4 v[kW];
#pragma unroll
      for (int r = 0; r < kW; ++r)
        v[r] = (r == p.rank) ? p.in[i] : __ldcv(reinterpret_cast<const uint4*>(rbase + (long long)r * p.max_bytes) + i);
#pragma unroll
      for (int r = 0; r < kW; ++r) p.out[(long long)r * p.nvec + i] = v[r];
    }
  } else if constexpr (kMode == kGatherCols) {
    // g2 colltrim: as kGather above (all kW loads in flight before any store), stores to the column slot
    const unsigned useg = (unsigned)p.seg;
    for (long long i = tid; i < p.nvec; i += stride) {
      uint4 v[kW];
#pragma unroll
      for (int r = 0; r < kW; ++r)
        v[r] = (r == p.rank) ? p.in[i] : __ldcv(reinterpret_cast<const uint4*>(rbase + (long long)r * p.max_bytes) + i);
      const long long row = (long long)((unsigned)i / useg);
      uint4* dst = p.out + row * ((long long)kW * p.seg) + (i - row * p.seg);
#pragma unroll
      for (int r = 0; r < kW; ++r) dst[(long long)r * p.seg] = v[r];
    }
  } else if constexpr (kMode == kOneShot) {
    for (long long i = tid; i < p.nvec; i += stride) {
      uint4 v[kW];
#pragma unroll
      for (int r = 0; r < kW; ++r)
        v[r] = (r == p.rank) ? p.in[i] : __ldcv(reinterpret_cast<const uint4*>(rbase + (long long)r * p.max_bytes) + i);
      float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int r = 0; r < kW; ++r) accv<kF32>(acc, v[r]);
      p.out[i] = packv<kF32>(acc);
    }
  } else {
    // two-shot: peers wrote chunk `rank` of their input at offset 0 of recv[slot][peer]
    const long long b = (long long)p.rank * p.chunk;
    const long long len = max(0LL, min(p.chunk, p.nvec - b));
    uint4* send2 = reinterpret_cast<uint4*>(p.send2 + slot * p.max_chunk);
    for (long long i = tid; i < len; i += stride) {
      uint4 v[kW];
#pragma unroll
      for (int r = 0; r < kW; ++r)
        v[r] = (r == p.rank) ? p.in[b + i] : __ldcv(reinterpret_cast<const uint4*>(rbase + (long long)r * p.max_bytes) + i);
      float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int r = 0; r < kW; ++r) accv<kF32>(acc, v[r]);
      const uint4 o = packv<kF32>(acc);
      p.out[b + i] = o;
      send2[i] = o;
    }
    // phase-2 publish; g130 V2 (no guard spin) holds unchanged: see ar_kernel_pl
    if constexpr ((kPub & kR3P2) != 0) {
      __threadfence_system();
      __syncthreads();
      if (threadIdx.x == 0) {
        *reinterpret_cast<volatile uint64_t*>(&p.ctrl->send2_bytes) = (uint64_t)len * 16;
        st_release_sys(&p.ctrl->send2_seq, seq);
      }
    } else {
      if (last_cta(&p.st->arrive2, &s_last)) {
        *reinterpret_cast<volatile uint64_t*>(&p.ctrl->send2_bytes) = (uint64_t)len * 16;
        st_release_sys(&p.ctrl->send2_seq, seq);
      }
    }
    if (threadIdx.x < p.world && threadIdx.x != p.rank)
      G130_SPIN(p.flags2 + slot * p.world + threadIdx.x, seq, threadIdx.x, 2);
    __syncthreads();
    if constexpr ((kPub & kR6Trig) != 0) r6_gdc_launch();  // r6 C2: the last wait is behind us
    const char* r2 = p.recv2 + (long long)slot * p.world * p.max_chunk;
    // legacy: for r != rank, for i < lr(r) = clamp(nvec - r*chunk, 0, chunk): out[r*chunk+i] = recv2[r][i].
    // Same element set (i < chunk and r*chunk + i < nvec), all kW-1 loads issued before the stores.
    for (long long i = tid; i < p.chunk; i += stride) {
      uint4 v[kW];
#pragma unroll
      for (int r = 0; r < kW; ++r)
        if (r != p.rank && (long long)r * p.chunk + i < p.nvec)
          v[r] = __ldcv(reinterpret_cast<const uint4*>(r2 + (long long)r * p.max_chunk) + i);
#pragma unroll
      for (int r = 0; r < kW; ++r)
        if (r != p.rank && (long long)r * p.chunk + i < p.nvec) p.out[(long long)r * p.chunk + i] = v[r];
    }
  }

  // r7-comm2 HT13: projection tail. Under P2 only CTA 0 is here and it wrote all of p.out itself (block-scope
  // visibility after the barrier); otherwise the last CTA through arrive3 (acq_rel, after every CTA's fence) projects.
  if constexpr ((kPub & kR7Proj) != 0) {
    if constexpr (kMode == kTwoShot && (kPub & kR3P2) != 0) {
      __syncthreads();
      r7_project(p.out, q);
    } else {
      __shared__ int s_go;
      __threadfence();
      __syncthreads();
      if (threadIdx.x == 0) {
        const bool last = atom_add_acqrel_gpu(&p.st->arrive3, 1u) == gridDim.x - 1;
        if (last) *reinterpret_cast<volatile unsigned int*>(&p.st->arrive3) = 0u;
        s_go = last ? 1 : 0;
      }
      __syncthreads();
      if (s_go) {
        __threadfence();
        r7_project(p.out, q);
      }
    }
  }

  // 3) seq already advanced at entry (iv); the tail barrier exists only for the timing block (unchanged)
  if (p.timing) {
    __syncthreads();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      __threadfence();
      const unsigned long long t_end = now_ns();
      const unsigned long long ts = p.st->t_start;
      const unsigned long long tp = *reinterpret_cast<volatile unsigned long long*>(&p.st->t_pub);
      atomicAdd(reinterpret_cast<unsigned long long*>(&p.st->sum_stage), (unsigned long long)(long long)(tp - ts));
      atomicAdd(reinterpret_cast<unsigned long long*>(&p.st->sum_net), (unsigned long long)(long long)(t_flags - tp));
      atomicAdd(reinterpret_cast<unsigned long long*>(&p.st->sum_tail), (unsigned long long)(long long)(t_end - t_flags));
      atomicAdd(reinterpret_cast<unsigned long long*>(&p.st->sum_total), (unsigned long long)(long long)(t_end - ts));
      atomicAdd(&p.st->n_timed, 1ull);
    }
  }
}
// ---- end r3-comm kernel

constexpr int kParW = 8;  // the only world size with a parallel-load instantiation (DSv4.1 TP8)

// g130: V2 has no meaning without a phase 2, so non-two-shot modes drop the bit instead of instantiating a duplicate kernel.
template <int kMode, bool kF32, int kPub>
void launch_pl(const Params& p, int blocks, cudaStream_t stream) {
  ar_kernel_pl<kMode, kF32, kParW, (kMode == kTwoShot ? kPub : (kPub & ~kPubV2))><<<blocks, kThreads, 0, stream>>>(p);
}

// r3-comm: P2 and V2 have no meaning outside two-shot, so those bits are dropped there (as launch_pl drops V2).
template <int kMode, bool kF32, int kPub>
void launch_r3_one(const Params& p, int blocks, cudaStream_t stream, const R7Proj& q) {
  if constexpr ((kPub & kR6Pdl) != 0) {
    // r6 C2: programmatic stream serialization; inside stream capture the kernel->kernel edge into this node becomes
    // a PROGRAMMATIC edge (device_test.py asserts it). Same grid, block, smem and Params as the <<<>>> launch.
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((unsigned)blocks);
    cfg.blockDim = dim3(kThreads);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at;
    cfg.numAttrs = 1;
    CHECK_CUDA(cudaLaunchKernelEx(&cfg, ar_kernel_r3<kMode, kF32, kParW, (kMode == kTwoShot ? kPub : (kPub & ~(kPubV2 | kR3P2)))>, p, q));
  } else {
    ar_kernel_r3<kMode, kF32, kParW, (kMode == kTwoShot ? kPub : (kPub & ~(kPubV2 | kR3P2)))>
        <<<blocks, kThreads, 0, stream>>>(p, q);
  }
}
// r7-comm2 HT13: the kR7Proj instantiations (bf16 one-shot / two-shot; every r6 mask without a send buffer)
template <int kMode, bool kF32>
void launch_r7(const Params& p, int blocks, cudaStream_t stream, int pub, const R7Proj& q) {
  if constexpr (!kF32 && (kMode == kOneShot || kMode == kTwoShot)) {
    constexpr int B = kPubV1 | kPubV2 | kPollThin | kR3Trim | kR7Proj;
    switch (pub) {
      case B: launch_r3_one<kMode, kF32, B>(p, blocks, stream, q); break;
      case B | kR3S1: launch_r3_one<kMode, kF32, B | kR3S1>(p, blocks, stream, q); break;
      case B | kR3P2: launch_r3_one<kMode, kF32, B | kR3P2>(p, blocks, stream, q); break;
      case B | kR3S1 | kR3P2: launch_r3_one<kMode, kF32, B | kR3S1 | kR3P2>(p, blocks, stream, q); break;
      case B | kR6Pdl: launch_r3_one<kMode, kF32, B | kR6Pdl>(p, blocks, stream, q); break;
      case B | kR6Pdl | kR3S1: launch_r3_one<kMode, kF32, B | kR6Pdl | kR3S1>(p, blocks, stream, q); break;
      case B | kR6Pdl | kR3P2: launch_r3_one<kMode, kF32, B | kR6Pdl | kR3P2>(p, blocks, stream, q); break;
      case B | kR6Pdl | kR3S1 | kR3P2: launch_r3_one<kMode, kF32, B | kR6Pdl | kR3S1 | kR3P2>(p, blocks, stream, q); break;
      case B | kR6Pdl | kR6Trig: launch_r3_one<kMode, kF32, B | kR6Pdl | kR6Trig>(p, blocks, stream, q); break;
      case B | kR6Pdl | kR6Trig | kR3S1: launch_r3_one<kMode, kF32, B | kR6Pdl | kR6Trig | kR3S1>(p, blocks, stream, q); break;
      case B | kR6Pdl | kR6Trig | kR3P2: launch_r3_one<kMode, kF32, B | kR6Pdl | kR6Trig | kR3P2>(p, blocks, stream, q); break;
      case B | kR6Pdl | kR6Trig | kR3S1 | kR3P2:
        launch_r3_one<kMode, kF32, B | kR6Pdl | kR6Trig | kR3S1 | kR3P2>(p, blocks, stream, q);
        break;
      default: throw std::runtime_error("rdma_ar r7: unsupported projection kernel mask " + std::to_string(pub));
    }
  } else {
    throw std::runtime_error("rdma_ar r7: the projection tail is bf16 all-reduce only");
  }
}
template <int kMode, bool kF32>
void launch_r3(const Params& p, int blocks, cudaStream_t stream, int pub, const R7Proj& q) {
  constexpr int B = kPubV1 | kPubV2 | kPollThin | kR3Trim;
  if ((pub & kR7Proj) != 0) {
    launch_r7<kMode, kF32>(p, blocks, stream, pub, q);
    return;
  }
  switch (pub) {
    case B: launch_r3_one<kMode, kF32, B>(p, blocks, stream, q); break;
    case B | kR3S1: launch_r3_one<kMode, kF32, B | kR3S1>(p, blocks, stream, q); break;
    case B | kR3P2: launch_r3_one<kMode, kF32, B | kR3P2>(p, blocks, stream, q); break;
    case B | kR3S1 | kR3P2: launch_r3_one<kMode, kF32, B | kR3S1 | kR3P2>(p, blocks, stream, q); break;
    case B | kR3Sbuf: launch_r3_one<kMode, kF32, B | kR3Sbuf>(p, blocks, stream, q); break;
    case B | kR3Sbuf | kR3P2: launch_r3_one<kMode, kF32, B | kR3Sbuf | kR3P2>(p, blocks, stream, q); break;
    // r6-comm C2 (never with Sbuf)
    case B | kR6Pdl: launch_r3_one<kMode, kF32, B | kR6Pdl>(p, blocks, stream, q); break;
    case B | kR6Pdl | kR3S1: launch_r3_one<kMode, kF32, B | kR6Pdl | kR3S1>(p, blocks, stream, q); break;
    case B | kR6Pdl | kR3P2: launch_r3_one<kMode, kF32, B | kR6Pdl | kR3P2>(p, blocks, stream, q); break;
    case B | kR6Pdl | kR3S1 | kR3P2: launch_r3_one<kMode, kF32, B | kR6Pdl | kR3S1 | kR3P2>(p, blocks, stream, q); break;
    case B | kR6Pdl | kR6Trig: launch_r3_one<kMode, kF32, B | kR6Pdl | kR6Trig>(p, blocks, stream, q); break;
    case B | kR6Pdl | kR6Trig | kR3S1: launch_r3_one<kMode, kF32, B | kR6Pdl | kR6Trig | kR3S1>(p, blocks, stream, q); break;
    case B | kR6Pdl | kR6Trig | kR3P2: launch_r3_one<kMode, kF32, B | kR6Pdl | kR6Trig | kR3P2>(p, blocks, stream, q); break;
    case B | kR6Pdl | kR6Trig | kR3S1 | kR3P2:
      launch_r3_one<kMode, kF32, B | kR6Pdl | kR6Trig | kR3S1 | kR3P2>(p, blocks, stream, q);
      break;
    default: throw std::runtime_error("rdma_ar r3: unsupported kernel mask " + std::to_string(pub));
  }
}

// Launch either kernel with the live grid rule: ceil(nvec / vec_per_cta) CTAs, clamped to [1, 16].
template <int kMode, bool kF32>
void launch_kernel(const Params& p, long long vec_per_cta, bool par, cudaStream_t stream, int pub = 0,
                   const R7Proj& q = R7Proj{nullptr, nullptr, 0}) {
  long long blocks = (p.nvec + vec_per_cta - 1) / vec_per_cta;
  blocks = std::max(1LL, std::min(blocks, 16LL));
  if ((pub & kR7Proj) != 0 && (pub & kR3Trim) == 0) throw std::runtime_error("rdma_ar r7: projection needs the r3 kernels");
  if ((pub & kR3Trim) != 0) {  // r3-comm: only with DSV41_RDMA_TRIM=1 (callers check par, world 8 and g130 mask 7)
    launch_r3<kMode, kF32>(p, (int)blocks, stream, pub, q);
    CHECK_CUDA(cudaGetLastError());
    return;
  }
  if (par && p.world == kParW) {
    switch (pub & 7) {
      case 0: launch_pl<kMode, kF32, 0>(p, (int)blocks, stream); break;
      case 1: launch_pl<kMode, kF32, 1>(p, (int)blocks, stream); break;
      case 2: launch_pl<kMode, kF32, 2>(p, (int)blocks, stream); break;
      case 3: launch_pl<kMode, kF32, 3>(p, (int)blocks, stream); break;
      case 4: launch_pl<kMode, kF32, 4>(p, (int)blocks, stream); break;
      case 5: launch_pl<kMode, kF32, 5>(p, (int)blocks, stream); break;
      case 6: launch_pl<kMode, kF32, 6>(p, (int)blocks, stream); break;
      default: launch_pl<kMode, kF32, 7>(p, (int)blocks, stream); break;
    }
  } else {
    ar_kernel<kMode, kF32><<<(int)blocks, kThreads, 0, stream>>>(p);
  }
  CHECK_CUDA(cudaGetLastError());
}

int env_parload_default() {
  const char* e = getenv("RDMA_AR_PARLOAD");
  return e ? (atoi(e) != 0) : 1;  // this .so exists to run the parallel-load kernels
}

// g130: unset or 0 == off
bool env_flag(const char* name) {
  const char* e = getenv(name);
  return e && atoi(e) != 0;
}
int env_pub_mask() {
  return (env_flag("RDMA_AR_PUB_V1") ? kPubV1 : 0) | (env_flag("RDMA_AR_PUB_V2") ? kPubV2 : 0) |
         (env_flag("RDMA_AR_HYG_POLL") ? kPollThin : 0);
}
// r3-comm: one parser for the r3 flags, identical to rdma_all_reduce.py's (review N4): after trimming blanks and
// lower-casing, unset / "" / 0 / off / false / no = off, 1 / on / true / yes = on, anything else = an error (never a
// silent off). Integers: decimal digits only, in range, else an error. Read only when an r3 flag is on, except the two
// on/off flags themselves.
std::string env_r3_norm(const char* name) {
  const char* e = getenv(name);
  std::string v = e ? e : "";
  const auto a = v.find_first_not_of(" \t\r\n\v\f"), b = v.find_last_not_of(" \t\r\n\v\f");
  v = a == std::string::npos ? std::string() : v.substr(a, b - a + 1);
  for (auto& c : v) c = (char)std::tolower((unsigned char)c);
  return v;
}
bool env_r3_flag(const char* name) {
  const std::string v = env_r3_norm(name);
  if (v.empty() || v == "0" || v == "off" || v == "false" || v == "no") return false;
  if (v == "1" || v == "on" || v == "true" || v == "yes") return true;
  throw std::runtime_error(std::string("rdma_ar r3: ") + name + "='" + v + "' is not 0/1/off/on/false/true/no/yes");
}
long long env_r3_int(const char* name, long long dflt, long long lo, long long hi) {
  const std::string v = env_r3_norm(name);
  if (v.empty()) return dflt;
  if (v.size() > 12 || v.find_first_not_of("0123456789") != std::string::npos || std::stoll(v) < lo || std::stoll(v) > hi)
    throw std::runtime_error(std::string("rdma_ar r3: ") + name + "='" + v + "' is not an integer in [" +
                             std::to_string(lo) + ", " + std::to_string(hi) + "]");
  return std::stoll(v);
}
// 1 = DSV41_RDMA_TRIM, 2 = DSV41_RDMA_SENDBUF
int env_r3_mask() { return (env_r3_flag("DSV41_RDMA_TRIM") ? 1 : 0) | (env_r3_flag("DSV41_RDMA_SENDBUF") ? 2 : 0); }
long long env_r3_s1_bytes() { return env_r3_int("DSV41_RDMA_TRIM_S1_BYTES", 24576, 0, 1LL << 30); }
long long env_r3_sbuf_mb() { return env_r3_int("DSV41_RDMA_SENDBUF_MB", 64, 1, 1024); }
// r6-comm C2: DSV41_RDMA_PDL. Same normalisation as the r3 flags; off values as env_r3_flag; 1 / on / true / yes / in =
// PDL launch (in-edge); trig = PDL launch + early trigger. Anything else is a startup error. rdma_all_reduce.py has the
// same parser.
int env_r6_pdl() {
  const std::string v = env_r3_norm("DSV41_RDMA_PDL");
  if (v.empty() || v == "0" || v == "off" || v == "false" || v == "no") return 0;
  if (v == "1" || v == "on" || v == "true" || v == "yes" || v == "in") return kR6Pdl;
  if (v == "trig") return kR6Pdl | kR6Trig;
  throw std::runtime_error("rdma_ar r6: DSV41_RDMA_PDL='" + v + "' is not 0/1/off/on/false/true/no/yes/in/trig");
}

// r7-comm2 HT13: DSV41_RDMA_PROJ, the r3 on/off parser (unset / 0 / off / false / no = off; 1 / on / true / yes = on)
bool env_r7_proj() { return env_r3_flag("DSV41_RDMA_PROJ"); }

// r7-comm2 HT13: argument checks shared by RdmaAR and SelfTest; returns the rows
int64_t r7_check(const torch::Tensor& x, const torch::Tensor& out, const torch::Tensor& d, const torch::Tensor& y) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.numel() % kR7H == 0,
              "rdma_ar r7: projection needs a contiguous bf16 CUDA [rows, ", kR7H, "] input");
  const int64_t rows = x.numel() / kR7H;
  TORCH_CHECK(rows >= 1 && rows <= kR7MaxRows, "rdma_ar r7: projection rows ", rows, " not in [1, ", kR7MaxRows, "]");
  TORCH_CHECK(d.is_cuda() && d.scalar_type() == at::kFloat && d.is_contiguous() && d.numel() == kR7H &&
                  d.get_device() == x.get_device() && ((uintptr_t)d.data_ptr() % 16) == 0,
              "rdma_ar r7: direction must be a 16-byte aligned contiguous fp32 [", kR7H, "] on the input's device");
  TORCH_CHECK(y.is_cuda() && y.scalar_type() == at::kBFloat16 && y.is_contiguous() && y.numel() == x.numel() &&
                  y.get_device() == x.get_device() && ((uintptr_t)y.data_ptr() % 16) == 0,
              "rdma_ar r7: y must be a 16-byte aligned contiguous bf16 tensor like the input");
  const auto yb = (uintptr_t)y.data_ptr(), ob = (uintptr_t)out.data_ptr(), xb = (uintptr_t)x.data_ptr();
  const auto nb = (uintptr_t)(x.numel() * 2);
  TORCH_CHECK((yb + nb <= ob || ob + nb <= yb) && (yb + nb <= xb || xb + nb <= yb),
              "rdma_ar r7: y must not overlap the input or the output");
  return rows;
}

struct Hca {
  ibv_context* ctx = nullptr;
  ibv_pd* pd = nullptr;
  ibv_cq* cq = nullptr;
  ibv_mr* mr = nullptr;
  ibv_gid gid{};
  ibv_mtu mtu = IBV_MTU_1024;
};

struct Peer {
  int hca = -1;
  ibv_qp* qp = nullptr;
  uint64_t remote_base = 0;
  uint32_t rkey = 0;
  uint64_t posts = 0;
};

class RdmaAR {
 public:
  RdmaAR(int rank, int world, int64_t max_bytes, int64_t max_reduce_bytes, std::vector<std::string> hca_names,
         int gid_index, int ib_port, int proxy_core, int64_t timeout_ms, int device)
      : rank_(rank), world_(world), max_bytes_(max_bytes), gid_index_(gid_index), ib_port_(ib_port),
        proxy_core_(proxy_core), timeout_ns_(timeout_ms * 1000000LL), device_(device) {
    if (max_bytes % 16 || max_reduce_bytes % 16) throw std::runtime_error("rdma_ar: sizes must be multiples of 16");
    if (world < 2 || world > kThreads) throw std::runtime_error("rdma_ar: bad world size");
    max_reduce_bytes_ = std::min(max_reduce_bytes, max_bytes);
    max_chunk_ = align((size_t)((max_reduce_bytes_ / 16 + world - 1) / world) * 16, 64);
    c10::cuda::CUDAGuard guard(device_);
    // one host-pinned arena: [ctrl | flags | flags2 | send | send2 | recv | recv2], registered on every HCA
    const size_t fl = align(sizeof(uint64_t) * kSlots * world, 4096);
    flags_off_ = 4096;
    flags2_off_ = flags_off_ + fl;
    send_off_ = flags2_off_ + fl;
    send2_off_ = send_off_ + align(kSlots * max_bytes, 4096);
    recv_off_ = send2_off_ + align(kSlots * max_chunk_, 4096);
    recv2_off_ = recv_off_ + align((size_t)kSlots * world * max_bytes, 4096);
    arena_bytes_ = recv2_off_ + align((size_t)kSlots * world * max_chunk_, 4096);
    // r3-comm: fail closed at construction (rdma_all_reduce.py turns a construction error into a refusal to start)
    if (r3_ == 2) throw std::runtime_error("rdma_ar r3: DSV41_RDMA_SENDBUF=1 needs DSV41_RDMA_TRIM=1");
    if (r6_pdl_ != 0 && (r3_ & 1) == 0)  // r6-comm C2: PDL lives in the r3 kernels only
      throw std::runtime_error("rdma_ar r6: DSV41_RDMA_PDL needs DSV41_RDMA_TRIM=1");
    if (r7_proj_ && (r3_ & 1) == 0)  // r7-comm2 HT13: the projection tail lives in the r3 kernels only
      throw std::runtime_error("rdma_ar r7: DSV41_RDMA_PROJ needs DSV41_RDMA_TRIM=1");
    if (r3_ != 0 && (pub_mask_ != (kPubV1 | kPubV2 | kPollThin) || !parload()))
      throw std::runtime_error("rdma_ar r3: DSV41_RDMA_TRIM needs the parallel-load kernels, world 8 and RDMA_AR_PUB_V1=1 "
                               "RDMA_AR_PUB_V2=1 RDMA_AR_HYG_POLL=1");
    if ((r3_ & 2) != 0) {  // r3 G26 (DSV41_RDMA_SENDBUF=1): registered send-buffer region at the end of the arena
      sbuf_off_ = align(arena_bytes_, 4096);
      sbuf_bytes_ = (size_t)env_r3_sbuf_mb() << 20;
      arena_bytes_ = sbuf_off_ + sbuf_bytes_;
      r3_sbuf_ = true;
    }
    // Back the arena with 2 MB transparent huge pages (THP is in madvise mode on DGX OS, so a user
    // process may ask for it) and pin it with cudaHostRegister. cudaHostAlloc takes whatever pages
    // the allocator has; on a fragmented node that is 4 KB pages, and the GPU's staging copies and
    // peer reads then miss the TLB constantly (measured 3x slower stage/reduce on such ranks, and
    // every other rank waits for them). RDMA_AR_THP=0 falls back to cudaHostAlloc.
    const size_t kHuge = 2u << 20;
    arena_bytes_ = align(arena_bytes_, kHuge);
    const char* thp = getenv("RDMA_AR_THP");
    if (!thp || atoi(thp) != 0) {
      void* raw = mmap(nullptr, arena_bytes_ + kHuge, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      if (raw != MAP_FAILED) {
        char* a = reinterpret_cast<char*>(align((size_t)(uintptr_t)raw, kHuge));
        madvise(a, arena_bytes_, MADV_HUGEPAGE);
        // vLLM/inductor may fork helper processes after init; keep the RDMA-registered arena out of
        // children so copy-on-write can never move the parent's pinned pages (DSv4.1 addition).
        madvise(a, arena_bytes_, MADV_DONTFORK);
        memset(a, 0, arena_bytes_);  // fault in now, as huge pages where the kernel can
        if (cudaHostRegister(a, arena_bytes_, cudaHostRegisterMapped | cudaHostRegisterPortable) == cudaSuccess) {
          arena_ = a;
          arena_registered_ = true;
        } else {
          cudaGetLastError();
          munmap(raw, arena_bytes_ + kHuge);
        }
      }
    }
    if (!arena_registered_) {
      CHECK_CUDA(cudaHostAlloc(&arena_, arena_bytes_, cudaHostAllocMapped | cudaHostAllocPortable));
      memset(arena_, 0, arena_bytes_);
    }
    CHECK_CUDA(cudaHostGetDevicePointer(reinterpret_cast<void**>(&arena_dev_), arena_, 0));
    thp_bytes_ = huge_backed_bytes(arena_, arena_bytes_);
    CHECK_CUDA(cudaMalloc(&st_, sizeof(DevState)));
    CHECK_CUDA(cudaMemset(st_, 0, sizeof(DevState)));
    ctrl_ = reinterpret_cast<Ctrl*>(arena_);

    int ndev = 0;
    ibv_device** list = ibv_get_device_list(&ndev);
    CHECK_IB(list != nullptr, "ibv_get_device_list");
    for (auto& name : hca_names) {
      Hca h;
      for (int i = 0; i < ndev; ++i)
        if (name == ibv_get_device_name(list[i])) h.ctx = ibv_open_device(list[i]);
      CHECK_IB(h.ctx != nullptr, "open device " + name);
      h.pd = ibv_alloc_pd(h.ctx);
      CHECK_IB(h.pd, "alloc_pd");
      h.cq = ibv_create_cq(h.ctx, 4096, nullptr, nullptr, 0);
      CHECK_IB(h.cq, "create_cq");
      h.mr = ibv_reg_mr(h.pd, arena_, arena_bytes_, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
      CHECK_IB(h.mr, "reg_mr " + name);
      CHECK_IB(ibv_query_gid(h.ctx, ib_port_, gid_index_, &h.gid) == 0, "query_gid");
      ibv_port_attr pa{};
      CHECK_IB(ibv_query_port(h.ctx, ib_port_, &pa) == 0, "query_port");
      h.mtu = pa.active_mtu;
      hcas_.push_back(h);
    }
    ibv_free_device_list(list);

    peers_.resize(world);
    for (int p = 0; p < world; ++p) {
      if (p == rank) continue;
      Peer& pr = peers_[p];
      pr.hca = (rank + p) % (int)hcas_.size();  // both ends of a pair agree on the HCA
      Hca& h = hcas_[pr.hca];
      ibv_qp_init_attr ia{};
      ia.send_cq = h.cq;
      ia.recv_cq = h.cq;
      ia.qp_type = IBV_QPT_RC;
      ia.cap.max_send_wr = 512;
      ia.cap.max_recv_wr = 1;
      ia.cap.max_send_sge = 1;
      ia.cap.max_recv_sge = 1;
      ia.cap.max_inline_data = 64;
      ia.sq_sig_all = 0;
      pr.qp = ibv_create_qp(h.pd, &ia);
      CHECK_IB(pr.qp, "create_qp");
      ibv_qp_attr a{};
      a.qp_state = IBV_QPS_INIT;
      a.pkey_index = 0;
      a.port_num = ib_port_;
      a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
      CHECK_IB(ibv_modify_qp(pr.qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) == 0,
               "qp INIT");
    }
  }

  ~RdmaAR() {
    stop_.store(true);
    if (proxy_.joinable()) proxy_.join();
    // QPs/MRs are left to process teardown; the arena may still be referenced by in-flight graphs.
  }

  // [gid_hi, gid_lo] per HCA, then rkey per HCA, then arena addr, then qpn per rank (0 for self)
  std::vector<int64_t> local_info() const {
    std::vector<int64_t> v;
    for (auto& h : hcas_) {
      uint64_t hi, lo;
      memcpy(&hi, h.gid.raw, 8);
      memcpy(&lo, h.gid.raw + 8, 8);
      v.push_back((int64_t)hi);
      v.push_back((int64_t)lo);
    }
    for (auto& h : hcas_) v.push_back((int64_t)h.mr->rkey);
    v.push_back((int64_t)(uintptr_t)arena_);
    for (int p = 0; p < world_; ++p) v.push_back(p == rank_ ? 0 : (int64_t)peers_[p].qp->qp_num);
    return v;
  }

  void connect(const std::vector<std::vector<int64_t>>& infos) {
    const int nh = (int)hcas_.size();
    for (int p = 0; p < world_; ++p) {
      if (p == rank_) continue;
      Peer& pr = peers_[p];
      const auto& r = infos[p];
      Hca& h = hcas_[pr.hca];
      ibv_gid rgid{};
      uint64_t hi = (uint64_t)r[2 * pr.hca], lo = (uint64_t)r[2 * pr.hca + 1];
      memcpy(rgid.raw, &hi, 8);
      memcpy(rgid.raw + 8, &lo, 8);
      pr.rkey = (uint32_t)r[2 * nh + pr.hca];
      pr.remote_base = (uint64_t)r[3 * nh];
      uint32_t rqpn = (uint32_t)r[3 * nh + 1 + rank_];

      ibv_qp_attr a{};
      a.qp_state = IBV_QPS_RTR;
      a.path_mtu = h.mtu;
      a.dest_qp_num = rqpn;
      a.rq_psn = 0;
      a.max_dest_rd_atomic = 1;
      a.min_rnr_timer = 12;
      a.ah_attr.is_global = 1;
      a.ah_attr.grh.dgid = rgid;
      a.ah_attr.grh.sgid_index = gid_index_;
      a.ah_attr.grh.hop_limit = 64;
      a.ah_attr.port_num = ib_port_;
      CHECK_IB(ibv_modify_qp(pr.qp, &a,
                             IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                                 IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) == 0,
               "qp RTR");
      ibv_qp_attr b{};
      b.qp_state = IBV_QPS_RTS;
      b.timeout = 14;
      b.retry_cnt = 7;
      b.rnr_retry = 7;
      b.sq_psn = 0;
      b.max_rd_atomic = 1;
      CHECK_IB(ibv_modify_qp(pr.qp, &b,
                             IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                                 IBV_QP_MAX_QP_RD_ATOMIC) == 0,
               "qp RTS");
    }
    proxy_ = std::thread([this] { proxy_loop(); });
  }

  torch::Tensor all_reduce(torch::Tensor x, bool two_shot) {
    const bool f32 = x.scalar_type() == at::kFloat;
    TORCH_CHECK(x.is_cuda() && (x.scalar_type() == at::kBFloat16 || f32) && x.is_contiguous(),
                "rdma_ar: need contiguous bf16/fp32 CUDA");
    TORCH_CHECK(x.numel() * x.element_size() <= max_reduce_bytes_, "rdma_ar: all-reduce too large");
    auto out = torch::empty_like(x);
    if (f32) {
      if (two_shot)
        launch<kTwoShot, true>(x, out);
      else
        launch<kOneShot, true>(x, out);
    } else {
      if (two_shot)
        launch<kTwoShot, false>(x, out);
      else
        launch<kOneShot, false>(x, out);
    }
    return out;
  }

  // r7-comm2 HT13: out = the all-reduce of x (exactly all_reduce's bytes) and y = _proj_rows(out, d), in one kernel.
  // Needs DSV41_RDMA_PROJ=1 (and TRIM) at construction; bf16 [rows <= 6, 5120]; d fp32 [5120]; y bf16 like x.
  torch::Tensor all_reduce_proj(torch::Tensor x, bool two_shot, torch::Tensor d, torch::Tensor y) {
    TORCH_CHECK(r7_proj_ && (r3_ & 1) != 0, "rdma_ar r7: all_reduce_proj needs DSV41_RDMA_PROJ=1 and DSV41_RDMA_TRIM=1");
    TORCH_CHECK(x.numel() * x.element_size() <= max_reduce_bytes_, "rdma_ar: all-reduce too large");
    auto out = torch::empty_like(x);
    const int64_t rows = r7_check(x, out, d, y);
    const R7Proj q{reinterpret_cast<const float*>(d.data_ptr()), reinterpret_cast<uint4*>(y.data_ptr()), rows};
    if (two_shot)
      launch<kTwoShot, false>(x, out, 0, &q);
    else
      launch<kOneShot, false>(x, out, 0, &q);
    return out;
  }
  bool r7() const { return r7_proj_; }

  // out = concat over ranks (dim 0) of x; any dtype, out must be contiguous with world * x.numel() elements
  void all_gather_into(torch::Tensor out, torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && out.is_contiguous() && out.scalar_type() == x.scalar_type() &&
                    out.numel() == x.numel() * world_,
                "rdma_ar: bad all_gather args");
    launch<kGather, false>(x, out);
  }

  // g2 colltrim: column all-gather. x = [outer, seg_bytes] bytes (any dtype, contiguous), out = [outer,
  // world * seg_bytes] bytes (contiguous): out row o, bytes [r*seg_bytes, (r+1)*seg_bytes) = row o of rank r's x.
  // seg_bytes == x bytes is the plain dim-0 gather. Same wire protocol and sequence as all_gather_into.
  void all_gather_cols(torch::Tensor out, torch::Tensor x, int64_t seg_bytes) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && out.is_contiguous() && out.scalar_type() == x.scalar_type() &&
                    out.numel() == x.numel() * world_,
                "rdma_ar: bad all_gather_cols args");
    const int64_t nbytes = x.numel() * x.element_size();
    TORCH_CHECK(seg_bytes > 0 && seg_bytes % 16 == 0 && nbytes % seg_bytes == 0,
                "rdma_ar: all_gather_cols seg_bytes ", seg_bytes, " must be a positive multiple of 16 dividing ", nbytes);
    launch<kGatherCols, false>(x, out, seg_bytes / 16);
  }

  int64_t max_bytes() const { return max_bytes_; }
  // average GPU->proxy->GPU round trip (ns) through the control block
  double pingpong(int n) {
    unsigned long long* d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_out, sizeof(unsigned long long)));
    static unsigned long long base = 1ull << 40;
    base += 1ull << 20;
    pingpong_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<Ctrl*>(arena_dev_), n, base, d_out);
    unsigned long long h = 0;
    CHECK_CUDA(cudaMemcpyAsync(&h, d_out, sizeof(h), cudaMemcpyDeviceToHost, at::cuda::getCurrentCUDAStream()));
    CHECK_CUDA(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream()));
    CHECK_CUDA(cudaFree(d_out));
    return (double)h / n;
  }

  // [arena bytes, bytes backed by huge pages, 1 if THP+cudaHostRegister path was used]
  std::vector<int64_t> arena_info() const { return {(int64_t)arena_bytes_, thp_bytes_, arena_registered_ ? 1 : 0}; }
  int64_t max_reduce_bytes() const { return max_reduce_bytes_; }

 private:
  static size_t align(size_t x, size_t a) { return (x + a - 1) / a * a; }

  // bytes of [p, p+n) backed by anonymous huge pages, from /proc/self/smaps
  static int64_t huge_backed_bytes(void* p, size_t n) {
    std::ifstream f("/proc/self/smaps");
    std::string line;
    const uintptr_t lo = (uintptr_t)p, hi = lo + n;
    bool in = false;
    int64_t kb = 0;
    while (std::getline(f, line)) {
      unsigned long a, b;
      if (sscanf(line.c_str(), "%lx-%lx", &a, &b) == 2) {  // mapping header line
        in = a < hi && b > lo;
        continue;
      }
      long v;
      if (in && sscanf(line.c_str(), "AnonHugePages: %ld kB", &v) == 1) kb += v;
    }
    return kb * 1024;
  }

  template <int kMode, bool kF32>
  void launch(const torch::Tensor& x, torch::Tensor& out, int64_t seg_vec = 0, const R7Proj* q7 = nullptr) {
    const int64_t nbytes = x.numel() * x.element_size();
    TORCH_CHECK(nbytes % 16 == 0 && nbytes <= max_bytes_, "rdma_ar: bad size ", nbytes);
    TORCH_CHECK(!proxy_error_.load(), "rdma_ar: proxy reported an RDMA error");
    if (nbytes == 0) return;
    // g130: kernel variants exist on the parallel-load path only; never let a flag be silently ignored
    TORCH_CHECK(pub_mask_ == 0 || parload(), "rdma_ar: RDMA_AR_PUB_V1/V2 and RDMA_AR_HYG_POLL need the parallel-load kernels");
    Params p;
    p.in = reinterpret_cast<const uint4*>(x.data_ptr());
    p.out = reinterpret_cast<uint4*>(out.data_ptr());
    p.nvec = nbytes / 16;
    p.chunk = (p.nvec + world_ - 1) / world_;
    p.send = arena_dev_ + send_off_;
    p.send2 = arena_dev_ + send2_off_;
    p.recv = arena_dev_ + recv_off_;
    p.recv2 = arena_dev_ + recv2_off_;
    p.flags = reinterpret_cast<const uint64_t*>(arena_dev_ + flags_off_);
    p.flags2 = reinterpret_cast<const uint64_t*>(arena_dev_ + flags2_off_);
    p.ctrl = reinterpret_cast<Ctrl*>(arena_dev_);
    p.st = st_;
    p.max_bytes = max_bytes_;
    p.max_chunk = (long long)max_chunk_;
    p.rank = rank_;
    p.world = world_;
    p.timeout_ns = timeout_ns_;
    p.timing = timing_;
    p.seg = seg_vec;  // g2 colltrim: read by kGatherCols only
    // a few CTAs are enough: the cost is latency, not bandwidth (RDMA_AR_VEC_PER_CTA tunes it)
    // k1/3-rdma-loads: parload_ picks ar_kernel_pl (world 8) over the legacy ar_kernel
    int pub = r3_ ? r3_pub(kMode, x, p) : pub_mask_;
    if (q7 != nullptr) {  // r7-comm2 HT13 (never with a send buffer: all_reduce_proj inputs are never sbuf views)
      TORCH_CHECK((pub & kR3Trim) != 0 && (pub & kR3Sbuf) == 0, "rdma_ar r7: projection needs the staged r3 kernel");
      pub |= kR7Proj;
    }
    launch_kernel<kMode, kF32>(p, vec_per_cta_, parload_ != 0, at::cuda::getCurrentCUDAStream(), pub,
                               q7 != nullptr ? *q7 : R7Proj{nullptr, nullptr, 0});
  }

  // r3-comm: kernel mask of one launch with an r3 flag on (never reached with r3_ == 0)
  int r3_pub(int mode, const torch::Tensor& x, const Params& p) {
    TORCH_CHECK(parload() && pub_mask_ == (kPubV1 | kPubV2 | kPollThin),
                "rdma_ar r3: DSV41_RDMA_TRIM/SENDBUF need the parallel-load kernels and RDMA_AR_PUB_V1=1 "
                "RDMA_AR_PUB_V2=1 RDMA_AR_HYG_POLL=1 (pub mask 7), got mask ", pub_mask_);
    bool sb = false;
    if ((r3_ & 2) != 0 && !sbuf_allocs_.empty()) {
      const auto it = sbuf_allocs_.find((uintptr_t)x.data_ptr());
      sb = it != sbuf_allocs_.end() && (size_t)p.nvec * 16 <= it->second;
    }
    int b = r3_bits(mode, p.nvec, p.chunk, r3_s1_bytes_, sb, kThreads);
    if ((b & (kR3S1 | kR3Sbuf)) != 0 &&
        r3_inplace_conflict(mode, (unsigned long long)(uintptr_t)p.in, (unsigned long long)(uintptr_t)p.out, p.nvec,
                            rank_, world_))
      b &= ~(kR3S1 | kR3Sbuf);  // review N5: in-place call -> the staged r3 path
    return pub_mask_ | b | ((b & kR3Sbuf) != 0 ? 0 : r6_pdl_);  // r6-comm C2: PDL bits (never with a send buffer)
  }

  void post(Peer& pr, uint64_t src, uint32_t len, uint64_t dst, uint64_t flag_dst, uint64_t flag) {
    Hca& h = hcas_[pr.hca];
    ibv_sge sge{};
    sge.addr = src;
    sge.length = len;
    sge.lkey = h.mr->lkey;
    ibv_sge fsge{};
    fsge.addr = (uint64_t)(uintptr_t)&flag;
    fsge.length = 8;
    fsge.lkey = 0;
    ibv_send_wr wr[2]{};
    wr[0].opcode = IBV_WR_RDMA_WRITE;
    wr[0].sg_list = &sge;
    wr[0].num_sge = 1;
    wr[0].wr.rdma.remote_addr = dst;
    wr[0].wr.rdma.rkey = pr.rkey;
    wr[0].next = &wr[1];
    wr[1].opcode = IBV_WR_RDMA_WRITE;
    wr[1].sg_list = &fsge;
    wr[1].num_sge = 1;
    wr[1].send_flags = IBV_SEND_INLINE | ((++pr.posts % kSignalEvery) == 0 ? IBV_SEND_SIGNALED : 0);
    wr[1].wr.rdma.remote_addr = flag_dst;
    wr[1].wr.rdma.rkey = pr.rkey;
    ibv_send_wr* bad = nullptr;
    if (len && split_post_.load(std::memory_order_relaxed)) {
      // g130 A: the same two WRs on the same QP in the same order, as two nreq == 1 posts. If the payload post fails the
      // flag is NOT posted, as in the chain: a flag without its payload would be silent wrong data at the peer.
      wr[0].next = nullptr;
      if (ibv_post_send(pr.qp, &wr[0], &bad) || ibv_post_send(pr.qp, &wr[1], &bad)) {
        fprintf(stderr, "rdma_ar rank %d: ibv_post_send failed errno=%d\n", rank_, errno);
        proxy_error_.store(true);
      }
    } else if (ibv_post_send(pr.qp, len ? &wr[0] : &wr[1], &bad)) {
      fprintf(stderr, "rdma_ar rank %d: ibv_post_send failed errno=%d\n", rank_, errno);
      proxy_error_.store(true);
    }
  }

  void proxy_loop() {
    if (proxy_core_ >= 0) {
      cpu_set_t set;
      CPU_ZERO(&set);
      CPU_SET(proxy_core_, &set);
      pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    }
    const uint64_t base = (uint64_t)(uintptr_t)arena_;
    uint64_t last1 = 0, last2 = 0;
    ibv_wc wc[32];
    // Spin while traffic flows (and for RDMA_AR_SPIN_US after the last message, which covers the
    // CPU gaps between decode steps); after that, back off to short sleeps so an idle server does
    // not pin a 3.9 GHz core (on GB10 CPU heat is shared with the GPU). Worst case the first
    // all-reduce after an idle period waits RDMA_AR_MAX_SLEEP_US.
    const char* e1 = getenv("RDMA_AR_SPIN_US");
    const char* e2 = getenv("RDMA_AR_MAX_SLEEP_US");
    const auto spin_window = std::chrono::microseconds(e1 ? atol(e1) : 20000);
    const long max_sleep_us = e2 ? atol(e2) : 1000;
    auto last_active = std::chrono::steady_clock::now();
    long backoff_us = 8;
    unsigned idle_iters = 0;
    uint64_t last_ping = 0;
    bool last_two = false;  // g130 hygiene: the newest phase-1 message was two-shot, so its phase 2 rings next
    const auto reap = [&]() {
      for (auto& h : hcas_) {
        int n = ibv_poll_cq(h.cq, 32, wc);
        for (int i = 0; i < n; ++i)
          if (wc[i].status != IBV_WC_SUCCESS) {
            fprintf(stderr, "rdma_ar rank %d: completion error %s (qp %u)\n", rank_,
                    ibv_wc_status_str(wc[i].status), wc[i].qp_num);
            proxy_error_.store(true);
          }
      }
    };
    while (!stop_.load(std::memory_order_relaxed)) {
      // g130 C (gr8 fabric_order.h walk_step): spin on the one word that rings next. Only a hint: after kHygSpin loads the
      // full pass below runs anyway, so a wrong guess costs a few ns, never a hang, and ping, stop and backoff still work.
      if (hyg_proxy_.load(std::memory_order_relaxed)) {
        const bool e2 = last_two && last2 != last1;
        const uint64_t* db = e2 ? &ctrl_->send2_seq : &ctrl_->send_seq;
        const uint64_t old = e2 ? last2 : last1;
        for (int k = 0; k < kHygSpin && __atomic_load_n(db, __ATOMIC_ACQUIRE) == old; ++k) {
        }
      }
      bool idle = true;
      const uint64_t pg = __atomic_load_n(&ctrl_->ping, __ATOMIC_ACQUIRE);
      if (pg != last_ping) {
        __atomic_store_n(&ctrl_->pong, pg, __ATOMIC_RELEASE);
        last_ping = pg;
        idle = false;
      }
      const uint64_t s1 = __atomic_load_n(&ctrl_->send_seq, __ATOMIC_ACQUIRE);
      if (s1 != last1) {
        idle = false;
        timespec ts0;
        clock_gettime(CLOCK_REALTIME, &ts0);
        const int64_t t_seen = (int64_t)ts0.tv_sec * 1000000000LL + ts0.tv_nsec;
        const int64_t t_pub = (int64_t)__atomic_load_n(&ctrl_->pub_time, __ATOMIC_ACQUIRE);
        const uint64_t nbytes = __atomic_load_n(&ctrl_->send_bytes, __ATOMIC_ACQUIRE);
        uint64_t mode = __atomic_load_n(&ctrl_->mode, __ATOMIC_ACQUIRE);
        const uint64_t cv = __atomic_load_n(&ctrl_->chunk_vecs, __ATOMIC_ACQUIRE);
        const uint64_t nvec = nbytes / 16;
        const int slot = (int)(s1 % kSlots);
        uint64_t src0 = base + send_off_ + (uint64_t)slot * max_bytes_;
        if (r3_sbuf_ && (mode & kModeSrc) != 0) {  // r3 G26: the payload is a registered send buffer
          src0 = base + __atomic_load_n(&ctrl_->src_off, __ATOMIC_ACQUIRE);
          mode &= ~kModeSrc;
        }
        for (int i = 1; i < world_; ++i) {
          const int q = (rank_ + i) % world_;
          Peer& pr = peers_[q];
          uint64_t src = src0, len = nbytes;
          if (mode == kTwoShot) {  // peer q gets chunk q of my input
            const uint64_t b = (uint64_t)q * cv;
            len = b < nvec ? std::min(cv, nvec - b) * 16 : 0;
            src = src0 + b * 16;
          }
          post(pr, src, (uint32_t)len, pr.remote_base + recv_off_ + ((uint64_t)slot * world_ + rank_) * max_bytes_,
               pr.remote_base + flags_off_ + ((uint64_t)slot * world_ + rank_) * 8, s1);
        }
        __atomic_store_n(&ctrl_->posted_seq, s1, __ATOMIC_RELEASE);
        last_two = mode == kTwoShot;
        timespec ts1;
        clock_gettime(CLOCK_REALTIME, &ts1);
        const int64_t t_done = (int64_t)ts1.tv_sec * 1000000000LL + ts1.tv_nsec;
        px_n_.fetch_add(1, std::memory_order_relaxed);
        px_handoff_.fetch_add(t_seen - t_pub, std::memory_order_relaxed);
        px_post_.fetch_add(t_done - t_seen, std::memory_order_relaxed);
        if (t_done - t_seen > px_post_max_.load(std::memory_order_relaxed))
          px_post_max_.store(t_done - t_seen, std::memory_order_relaxed);
        last1 = s1;
        if (hyg_proxy_.load(std::memory_order_relaxed)) reap();
      }
      const uint64_t s2 = __atomic_load_n(&ctrl_->send2_seq, __ATOMIC_ACQUIRE);
      if (s2 != last2) {
        idle = false;
        const uint64_t len = __atomic_load_n(&ctrl_->send2_bytes, __ATOMIC_ACQUIRE);
        const int slot = (int)(s2 % kSlots);
        const uint64_t src = base + send2_off_ + (uint64_t)slot * max_chunk_;
        for (int i = 1; i < world_; ++i) {
          const int q = (rank_ + i) % world_;
          Peer& pr = peers_[q];
          post(pr, src, (uint32_t)len, pr.remote_base + recv2_off_ + ((uint64_t)slot * world_ + rank_) * max_chunk_,
               pr.remote_base + flags2_off_ + ((uint64_t)slot * world_ + rank_) * 8, s2);
        }
        __atomic_store_n(&ctrl_->posted2_seq, s2, __ATOMIC_RELEASE);
        last2 = s2;
        if (hyg_proxy_.load(std::memory_order_relaxed)) reap();
      }
      if (idle) {
        // g130 C: with hygiene on, the post path reaps; this slow reap stays so that a completion error with no traffic
        // behind it is still reported (gr8 has no such path, its walk ends on a timeout instead).
        if (!hyg_proxy_.load(std::memory_order_relaxed) || (idle_iters & 1023) == 0) reap();
        if ((++idle_iters & 255) == 0 && std::chrono::steady_clock::now() - last_active > spin_window) {
          std::this_thread::sleep_for(std::chrono::microseconds(backoff_us));
          backoff_us = std::min(backoff_us * 2, max_sleep_us);
        }
      } else {
        idle_iters = 0;
        backoff_us = 8;
        last_active = std::chrono::steady_clock::now();
      }
    }
  }

  int rank_, world_;
  int64_t max_bytes_, max_reduce_bytes_ = 0;
  int gid_index_, ib_port_, proxy_core_;
  long long timeout_ns_;
  int device_;
  size_t max_chunk_ = 0;
  void* arena_ = nullptr;
  char* arena_dev_ = nullptr;
  size_t arena_bytes_ = 0, flags_off_ = 0, flags2_off_ = 0, send_off_ = 0, send2_off_ = 0, recv_off_ = 0,
         recv2_off_ = 0;
  Ctrl* ctrl_ = nullptr;
  DevState* st_ = nullptr;
  std::vector<Hca> hcas_;
  std::vector<Peer> peers_;
  std::thread proxy_;
  std::atomic<bool> stop_{false};
  std::atomic<bool> proxy_error_{false};
  std::atomic<int64_t> px_n_{0}, px_handoff_{0}, px_post_{0}, px_post_max_{0};
  bool arena_registered_ = false;
  int64_t thp_bytes_ = 0;
  int timing_ = getenv("RDMA_AR_TIMING") ? atoi(getenv("RDMA_AR_TIMING")) : 0;
  long long vec_per_cta_ = getenv("RDMA_AR_VEC_PER_CTA") ? std::max(64LL, atoll(getenv("RDMA_AR_VEC_PER_CTA"))) : 512;
  int parload_ = env_parload_default();
  // g130 flags: see the file header. Off when unset. pub_mask_ acts at the next launch, the other two at the next message.
  static constexpr int kHygSpin = 64;
  int pub_mask_ = env_pub_mask();
  std::atomic<bool> split_post_{env_flag("RDMA_AR_SPLIT_POST")};
  std::atomic<bool> hyg_proxy_{env_flag("RDMA_AR_HYG_PROXY")};
  // r3-comm (off when unset; read once per instance). r3_sbuf_ is fixed before the proxy starts (connect()).
  int r3_ = env_r3_mask();
  long long r3_s1_bytes_ = r3_ ? env_r3_s1_bytes() : 24576;
  int r6_pdl_ = env_r6_pdl();  // r6-comm C2: 0, kR6Pdl or kR6Pdl | kR6Trig (DSV41_RDMA_PDL; needs TRIM)
  bool r7_proj_ = env_r7_proj();  // r7-comm2 HT13: all_reduce_proj allowed (DSV41_RDMA_PROJ; needs TRIM)
  bool r3_sbuf_ = false;
  size_t sbuf_off_ = 0, sbuf_bytes_ = 0, sbuf_used_ = 0;
  std::unordered_map<uintptr_t, size_t> sbuf_allocs_;  // device ptr of each send buffer -> its bytes

 public:
  // g130: [split_post, hyg_proxy, pub_mask]. set_variant is for benches: call it with the stream drained and the same
  // values on every rank's quiet point. A CUDA graph keeps the kernel variant it was captured with.
  std::vector<int64_t> variant() const {
    return {split_post_.load() ? 1 : 0, hyg_proxy_.load() ? 1 : 0, pub_mask_};
  }
  void set_variant(bool split_post, bool hyg_proxy, int pub_mask) {
    TORCH_CHECK(pub_mask >= 0 && pub_mask <= 7, "rdma_ar: bad pub_mask ", pub_mask);
    split_post_.store(split_post);
    hyg_proxy_.store(hyg_proxy);
    pub_mask_ = pub_mask;
  }
  // k1/3-rdma-loads: 1 = parallel-load kernels (default of this .so), 0 = legacy ar_kernel. Takes effect
  // at the next launch, so a CUDA graph keeps the variant it was captured with. All ranks must agree
  // only on the op order, not on the variant (the protocol is identical), but keep them equal anyway.
  void set_parload(bool on) { parload_ = on ? 1 : 0; }
  // r3-comm: [mask (1 trim, 2 sendbuf), S1 bytes, send-buffer region bytes, bytes handed out, buffers]
  std::vector<int64_t> r3() const {
    return {r3_, r3_s1_bytes_, (int64_t)sbuf_bytes_, (int64_t)sbuf_used_, (int64_t)sbuf_allocs_.size()};
  }
  // r3-comm, benches only: call with the stream drained, same values on every rank (a graph keeps what it captured)
  void set_r3(int mask, int64_t s1_bytes) {
    TORCH_CHECK(mask == 0 || mask == 1 || (mask == 3 && r3_sbuf_), "rdma_ar r3: bad mask ", mask,
                " (2 needs 1; 3 needs a DSV41_RDMA_SENDBUF=1 instance)");
    r3_ = mask;
    r3_s1_bytes_ = std::max<int64_t>(0, s1_bytes);
  }
  // r6-comm C2: the PDL bits of the r3 launches (0, 128 = PDL, 384 = PDL + trigger); set_r6 for benches only (stream
  // drained, same value on every rank; a graph keeps what it captured)
  int64_t r6() const { return r6_pdl_; }
  void set_r6(int bits) {
    TORCH_CHECK(bits == 0 || bits == kR6Pdl || bits == (kR6Pdl | kR6Trig), "rdma_ar r6: bad PDL bits ", bits);
    r6_pdl_ = bits;
  }
  // r3 G26: a registered send buffer of nbytes (uint8 CUDA view of the arena; never freed), or None when the region
  // is absent or full. An all_reduce / all_gather whose input starts at such a buffer is published from it.
  std::optional<torch::Tensor> send_buffer(int64_t nbytes) {
    if ((r3_ & 2) == 0 || !r3_sbuf_ || nbytes <= 0 || nbytes % 16 || nbytes > max_bytes_) return std::nullopt;
    const size_t need = align((size_t)nbytes, 256) + 256;  // 256-byte prefix: the 16-byte reuse header ends it
    if (sbuf_used_ + need > sbuf_bytes_) return std::nullopt;
    char* ptr = arena_dev_ + sbuf_off_ + sbuf_used_ + 256;
    sbuf_used_ += need;
    sbuf_allocs_[(uintptr_t)ptr] = (size_t)nbytes;
    return torch::from_blob(ptr, {nbytes}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, device_));
  }
  bool parload() const { return parload_ != 0 && world_ == kParW; }
  // [n_kernel, avg stage, avg pub->flags, avg flags->end, avg total, n_proxy, avg handoff, avg post, max post]
  // all in ns; GPU globaltimer vs CPU CLOCK_REALTIME for the hand-off (aligned on Linux drivers).
  std::vector<double> stats(bool reset) {
    DevState h{};
    CHECK_CUDA(cudaMemcpy(&h, st_, sizeof(DevState), cudaMemcpyDeviceToHost));
    const double n = h.n_timed ? (double)h.n_timed : 1.0;
    const double pn_raw = (double)px_n_.load();
    const double pn = pn_raw > 0 ? pn_raw : 1.0;
    std::vector<double> v{(double)h.n_timed, h.sum_stage / n, h.sum_net / n, h.sum_tail / n, h.sum_total / n,
                          pn_raw, px_handoff_.load() / pn, px_post_.load() / pn, (double)px_post_max_.load()};
    if (reset) {
      h.sum_stage = h.sum_net = h.sum_tail = h.sum_total = 0;
      h.n_timed = 0;
      CHECK_CUDA(cudaMemcpy(st_, &h, sizeof(DevState), cudaMemcpyHostToDevice));
      px_n_ = 0;
      px_handoff_ = 0;
      px_post_ = 0;
      px_post_max_ = 0;
    }
    return v;
  }
};

// ---- k1/3-rdma-loads single-GPU self-test (no HCA, no proxy, no network) -----------------------
// Same arena layout as RdmaAR (2 MiB THP-backed, cudaHostRegister'd host memory), but every peer flag
// and both posted_seq words are preset far in the future, so the kernel never waits: it stages,
// "publishes" to nobody, and goes straight to the consumer loops. The test writes what the peers
// would have RDMA-written into recv / recv2 (same image in all kSlots slots) and compares
// legacy vs parallel-load kernels bitwise. Used only by selftest_k1.py; not touched in serving.
class SelfTest {
 public:
  SelfTest(int rank, int world, int64_t max_bytes, int64_t max_reduce_bytes, int device)
      : rank_(rank), world_(world), max_bytes_(max_bytes), device_(device) {
    if (max_bytes % 16 || max_reduce_bytes % 16) throw std::runtime_error("selftest: sizes must be multiples of 16");
    if (world < 2 || world > kThreads || rank < 0 || rank >= world) throw std::runtime_error("selftest: bad rank/world");
    max_reduce_bytes_ = std::min(max_reduce_bytes, max_bytes);
    max_chunk_ = align((size_t)((max_reduce_bytes_ / 16 + world - 1) / world) * 16, 64);
    c10::cuda::CUDAGuard guard(device_);
    const size_t fl = align(sizeof(uint64_t) * kSlots * world, 4096);
    flags_off_ = 4096;
    flags2_off_ = flags_off_ + fl;
    send_off_ = flags2_off_ + fl;
    send2_off_ = send_off_ + align(kSlots * max_bytes, 4096);
    recv_off_ = send2_off_ + align(kSlots * max_chunk_, 4096);
    recv2_off_ = recv_off_ + align((size_t)kSlots * world * max_bytes, 4096);
    const size_t kHuge = 2u << 20;
    arena_bytes_ = align(recv2_off_ + align((size_t)kSlots * world * max_chunk_, 4096), kHuge);
    void* raw = mmap(nullptr, arena_bytes_ + kHuge, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw != MAP_FAILED) {
      char* a = reinterpret_cast<char*>(align((size_t)(uintptr_t)raw, kHuge));
      madvise(a, arena_bytes_, MADV_HUGEPAGE);
      memset(a, 0, arena_bytes_);
      if (cudaHostRegister(a, arena_bytes_, cudaHostRegisterMapped | cudaHostRegisterPortable) == cudaSuccess) {
        arena_ = a;
        raw_ = raw;
      } else {
        cudaGetLastError();
        munmap(raw, arena_bytes_ + kHuge);
      }
    }
    if (!arena_) {
      CHECK_CUDA(cudaHostAlloc(&arena_, arena_bytes_, cudaHostAllocMapped | cudaHostAllocPortable));
      memset(arena_, 0, arena_bytes_);
      host_alloc_ = true;
    }
    CHECK_CUDA(cudaHostGetDevicePointer(reinterpret_cast<void**>(&arena_dev_), arena_, 0));
    CHECK_CUDA(cudaMalloc(&st_, sizeof(DevState)));
    CHECK_CUDA(cudaMemset(st_, 0, sizeof(DevState)));
    char* a = reinterpret_cast<char*>(arena_);
    const uint64_t far = 1ull << 62;
    uint64_t* f1 = reinterpret_cast<uint64_t*>(a + flags_off_);
    uint64_t* f2 = reinterpret_cast<uint64_t*>(a + flags2_off_);
    for (int i = 0; i < kSlots * world; ++i) f1[i] = f2[i] = far;
    Ctrl* c = reinterpret_cast<Ctrl*>(a);
    c->posted_seq = far;
    c->posted2_seq = far;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
  }
  ~SelfTest() {
    if (st_) cudaFree(st_);
    if (host_alloc_) {
      cudaFreeHost(arena_);
    } else if (arena_) {
      cudaHostUnregister(arena_);
      munmap(raw_, arena_bytes_ + (2u << 20));
    }
  }

  // img: CPU uint8 [world, max_bytes] -> recv[s][r] for every slot s (row `rank` is ignored by the kernels)
  void set_recv(torch::Tensor img) { copy_img(img, recv_off_, (size_t)max_bytes_); }
  // img: CPU uint8 [world, max_chunk] -> recv2[s][r] for every slot s
  void set_recv2(torch::Tensor img) { copy_img(img, recv2_off_, max_chunk_); }

  void all_reduce_into(torch::Tensor out, torch::Tensor x, bool two_shot, bool par, int64_t vec_per_cta, bool timing) {
    const bool f32 = x.scalar_type() == at::kFloat;
    TORCH_CHECK(x.is_cuda() && (x.scalar_type() == at::kBFloat16 || f32) && x.is_contiguous() && out.is_contiguous() &&
                    out.scalar_type() == x.scalar_type() && out.numel() == x.numel(),
                "selftest: bad all_reduce args");
    TORCH_CHECK(x.numel() * x.element_size() <= max_reduce_bytes_, "selftest: all-reduce too large");
    if (f32) {
      if (two_shot) go<kTwoShot, true>(x, out, par, vec_per_cta, timing);
      else go<kOneShot, true>(x, out, par, vec_per_cta, timing);
    } else {
      if (two_shot) go<kTwoShot, false>(x, out, par, vec_per_cta, timing);
      else go<kOneShot, false>(x, out, par, vec_per_cta, timing);
    }
  }
  void all_gather_into(torch::Tensor out, torch::Tensor x, bool par, int64_t vec_per_cta, bool timing) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && out.is_contiguous() && out.scalar_type() == x.scalar_type() &&
                    out.numel() == x.numel() * world_,
                "selftest: bad all_gather args");
    go<kGather, false>(x, out, par, vec_per_cta, timing);
  }
  // r7-comm2 HT13: all_reduce_into + the projection tail (needs set_r3(1)); y = _proj_rows(out, d)
  void all_reduce_proj_into(torch::Tensor out, torch::Tensor x, bool two_shot, torch::Tensor d, torch::Tensor y, bool par,
                            int64_t vec_per_cta, bool timing) {
    TORCH_CHECK(r3_ == 1, "selftest r7: the projection tail needs set_r3(1)");
    TORCH_CHECK(out.is_contiguous() && out.scalar_type() == x.scalar_type() && out.numel() == x.numel(),
                "selftest: bad all_reduce args");
    TORCH_CHECK(x.numel() * x.element_size() <= max_reduce_bytes_, "selftest: all-reduce too large");
    const int64_t rows = r7_check(x, out, d, y);
    const R7Proj q{reinterpret_cast<const float*>(d.data_ptr()), reinterpret_cast<uint4*>(y.data_ptr()), rows};
    if (two_shot) go<kTwoShot, false>(x, out, par, vec_per_cta, timing, 0, &q);
    else go<kOneShot, false>(x, out, par, vec_per_cta, timing, 0, &q);
  }
  // r7-comm2 HT13, test only: st->arrive3 (must be back at 0 after every kernel)
  int64_t arrive3() {
    DevState h{};
    CHECK_CUDA(cudaMemcpy(&h, st_, sizeof(DevState), cudaMemcpyDeviceToHost));
    return (int64_t)h.arrive3;
  }
  // g2 colltrim: column all-gather through the same kernels (see RdmaAR::all_gather_cols)
  void all_gather_cols_into(torch::Tensor out, torch::Tensor x, int64_t seg_bytes, bool par, int64_t vec_per_cta,
                            bool timing) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && out.is_contiguous() && out.scalar_type() == x.scalar_type() &&
                    out.numel() == x.numel() * world_,
                "selftest: bad all_gather_cols args");
    const int64_t nbytes = x.numel() * x.element_size();
    TORCH_CHECK(seg_bytes > 0 && seg_bytes % 16 == 0 && nbytes % seg_bytes == 0, "selftest: bad seg_bytes ", seg_bytes);
    go<kGatherCols, false>(x, out, par, vec_per_cta, timing, seg_bytes / 16);
  }
  // [n_timed, avg stage, avg pub->flags, avg flags->end (tail), avg total] in ns
  std::vector<double> stats(bool reset) {
    DevState h{};
    CHECK_CUDA(cudaMemcpy(&h, st_, sizeof(DevState), cudaMemcpyDeviceToHost));
    const double n = h.n_timed ? (double)h.n_timed : 1.0;
    std::vector<double> v{(double)h.n_timed, h.sum_stage / n, h.sum_net / n, h.sum_tail / n, h.sum_total / n};
    if (reset) {
      h.sum_stage = h.sum_net = h.sum_tail = h.sum_total = 0;
      h.n_timed = 0;
      CHECK_CUDA(cudaMemcpy(st_, &h, sizeof(DevState), cudaMemcpyHostToDevice));
    }
    return v;
  }
  int64_t max_chunk() const { return (int64_t)max_chunk_; }
  int64_t max_bytes() const { return max_bytes_; }
  int64_t max_reduce_bytes() const { return max_reduce_bytes_; }
  bool thp() const { return !host_alloc_; }
  // g130: kernel variant for the following calls (par = true only); default 0 == production kernels
  void set_pub(int pub_mask) {
    TORCH_CHECK(pub_mask >= 0 && pub_mask <= 7, "selftest: bad pub_mask ", pub_mask);
    pub_ = pub_mask;
  }
  // r3-comm: 0 = off, 1 = trim kernels (needs set_pub(7) and par); s1_bytes as DSV41_RDMA_TRIM_S1_BYTES
  void set_r3(int mask, int64_t s1_bytes) {
    TORCH_CHECK(mask == 0 || mask == 1, "selftest: bad r3 mask ", mask);
    r3_ = mask;
    r3_s1_bytes_ = std::max<int64_t>(0, s1_bytes);
  }
  // r6-comm C2: PDL bits of the r3 launches (0, 128, 384); only with set_r3(1)
  void set_r6(int bits) {
    TORCH_CHECK(bits == 0 || bits == kR6Pdl || bits == (kR6Pdl | kR6Trig), "selftest: bad r6 bits ", bits);
    r6_ = bits;
  }
  // r3-comm, test only: bytes [0, nbytes) of send slot `slot` (which = 0) or send2 slot (which = 1), and st->seq /
  // st->entered / st->arrive / st->arrive2 / st->depart (the counters must be back at 0 after every kernel)
  torch::Tensor peek_send(int which, int slot, int64_t nbytes) {
    TORCH_CHECK(slot >= 0 && slot < kSlots && nbytes >= 0 && nbytes <= (which ? (int64_t)max_chunk_ : max_bytes_),
                "selftest: bad peek");
    CHECK_CUDA(cudaDeviceSynchronize());
    auto t = torch::empty({nbytes}, torch::TensorOptions().dtype(torch::kUInt8));
    const char* a = reinterpret_cast<const char*>(arena_) + (which ? send2_off_ + (size_t)slot * max_chunk_
                                                                       : send_off_ + (size_t)slot * max_bytes_);
    memcpy(t.data_ptr(), a, (size_t)nbytes);
    return t;
  }
  std::vector<int64_t> counters() {
    DevState h{};
    CHECK_CUDA(cudaMemcpy(&h, st_, sizeof(DevState), cudaMemcpyDeviceToHost));
    return {(int64_t)h.seq, (int64_t)h.entered, (int64_t)h.arrive, (int64_t)h.arrive2, (int64_t)h.depart};
  }

 private:
  static size_t align(size_t x, size_t a) { return (x + a - 1) / a * a; }
  // img [world, w] with w <= row: writes the first w bytes of row r of every slot (the rest is left alone)
  void copy_img(const torch::Tensor& img, size_t off, size_t row) {
    TORCH_CHECK(img.device().is_cpu() && img.scalar_type() == at::kByte && img.is_contiguous() && img.dim() == 2 &&
                    img.size(0) == world_ && (size_t)img.size(1) <= row,
                "selftest: image must be CPU uint8 [world, <= row bytes]");
    CHECK_CUDA(cudaDeviceSynchronize());
    const size_t w = (size_t)img.size(1);
    const char* src = reinterpret_cast<const char*>(img.data_ptr());
    char* a = reinterpret_cast<char*>(arena_) + off;
    for (int s = 0; s < kSlots; ++s)
      for (int r = 0; r < world_; ++r) memcpy(a + ((size_t)s * world_ + r) * row, src + (size_t)r * w, w);
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
  }
  template <int kMode, bool kF32>
  void go(const torch::Tensor& x, torch::Tensor& out, bool par, int64_t vec_per_cta, bool timing, int64_t seg_vec = 0,
          const R7Proj* q7 = nullptr) {
    const int64_t nbytes = x.numel() * x.element_size();
    TORCH_CHECK(nbytes % 16 == 0 && nbytes > 0 && nbytes <= max_bytes_, "selftest: bad size ", nbytes);
    c10::cuda::CUDAGuard guard(device_);
    Params p;
    p.in = reinterpret_cast<const uint4*>(x.data_ptr());
    p.out = reinterpret_cast<uint4*>(out.data_ptr());
    p.nvec = nbytes / 16;
    p.chunk = (p.nvec + world_ - 1) / world_;
    p.send = arena_dev_ + send_off_;
    p.send2 = arena_dev_ + send2_off_;
    p.recv = arena_dev_ + recv_off_;
    p.recv2 = arena_dev_ + recv2_off_;
    p.flags = reinterpret_cast<const uint64_t*>(arena_dev_ + flags_off_);
    p.flags2 = reinterpret_cast<const uint64_t*>(arena_dev_ + flags2_off_);
    p.ctrl = reinterpret_cast<Ctrl*>(arena_dev_);
    p.st = st_;
    p.max_bytes = max_bytes_;
    p.max_chunk = (long long)max_chunk_;
    p.rank = rank_;
    p.world = world_;
    p.timeout_ns = 5000000000LL;  // 5 s: nothing should ever wait here
    p.timing = timing ? 1 : 0;
    p.seg = seg_vec;
    TORCH_CHECK(pub_ == 0 || (par && world_ == kParW), "selftest: kernel variants need par = true and world 8");
    TORCH_CHECK(r3_ == 0 || pub_ == (kPubV1 | kPubV2 | kPollThin), "selftest: r3 needs set_pub(7)");
    const int pub = (r3_ ? (pub_ | r3_bits(kMode, p.nvec, p.chunk, r3_s1_bytes_, false, kThreads) | r6_) : pub_) |
                    (q7 != nullptr ? kR7Proj : 0);
    launch_kernel<kMode, kF32>(p, std::max<long long>(64, vec_per_cta), par, at::cuda::getCurrentCUDAStream(), pub,
                               q7 != nullptr ? *q7 : R7Proj{nullptr, nullptr, 0});
  }

  int rank_, world_;
  int pub_ = 0;  // g130
  int r3_ = 0;    // r3-comm: 1 = ar_kernel_r3 (trim), selected per launch like RdmaAR
  int r6_ = 0;    // r6-comm C2: PDL bits of the r3 launches
  long long r3_s1_bytes_ = 24576;
  int64_t max_bytes_, max_reduce_bytes_ = 0;
  int device_;
  size_t max_chunk_ = 0;
  void* arena_ = nullptr;
  void* raw_ = nullptr;
  bool host_alloc_ = false;
  char* arena_dev_ = nullptr;
  size_t arena_bytes_ = 0, flags_off_ = 0, flags2_off_ = 0, send_off_ = 0, send2_off_ = 0, recv_off_ = 0,
         recv2_off_ = 0;
  DevState* st_ = nullptr;
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<RdmaAR>(m, "RdmaAR")
      .def(py::init<int, int, int64_t, int64_t, std::vector<std::string>, int, int, int, int64_t, int>(),
           py::arg("rank"), py::arg("world"), py::arg("max_bytes"), py::arg("max_reduce_bytes"), py::arg("hcas"),
           py::arg("gid_index"), py::arg("ib_port"), py::arg("proxy_core"), py::arg("timeout_ms"), py::arg("device"))
      .def("local_info", &RdmaAR::local_info)
      .def("connect", &RdmaAR::connect)
      .def("all_reduce", &RdmaAR::all_reduce, py::arg("x"), py::arg("two_shot") = false)
      .def("all_gather_into", &RdmaAR::all_gather_into)
      .def("all_gather_cols", &RdmaAR::all_gather_cols, py::arg("out"), py::arg("x"), py::arg("seg_bytes"))
      .def("stats", &RdmaAR::stats, py::arg("reset") = false)
      .def("arena_info", &RdmaAR::arena_info)
      .def("pingpong", &RdmaAR::pingpong, py::arg("n") = 1000)
      .def("max_bytes", &RdmaAR::max_bytes)
      .def("max_reduce_bytes", &RdmaAR::max_reduce_bytes)
      .def("set_parload", &RdmaAR::set_parload, py::arg("on"))
      .def("parload", &RdmaAR::parload)
      .def("variant", &RdmaAR::variant)
      .def("set_variant", &RdmaAR::set_variant, py::arg("split_post"), py::arg("hyg_proxy"), py::arg("pub_mask"))
      .def("r3", &RdmaAR::r3)
      .def("set_r3", &RdmaAR::set_r3, py::arg("mask"), py::arg("s1_bytes") = 24576)
      .def("send_buffer", &RdmaAR::send_buffer, py::arg("nbytes"))
      .def("r6", &RdmaAR::r6)
      .def("set_r6", &RdmaAR::set_r6, py::arg("bits"))
      .def("all_reduce_proj", &RdmaAR::all_reduce_proj, py::arg("x"), py::arg("two_shot"), py::arg("d"), py::arg("y"))
      .def("r7", &RdmaAR::r7);
  py::class_<SelfTest>(m, "SelfTest")
      .def(py::init<int, int, int64_t, int64_t, int>(), py::arg("rank"), py::arg("world"), py::arg("max_bytes"),
           py::arg("max_reduce_bytes"), py::arg("device"))
      .def("set_recv", &SelfTest::set_recv)
      .def("set_recv2", &SelfTest::set_recv2)
      .def("all_reduce_into", &SelfTest::all_reduce_into, py::arg("out"), py::arg("x"), py::arg("two_shot"),
           py::arg("par"), py::arg("vec_per_cta") = 512, py::arg("timing") = false)
      .def("all_gather_into", &SelfTest::all_gather_into, py::arg("out"), py::arg("x"), py::arg("par"),
           py::arg("vec_per_cta") = 512, py::arg("timing") = false)
      .def("all_gather_cols_into", &SelfTest::all_gather_cols_into, py::arg("out"), py::arg("x"),
           py::arg("seg_bytes"), py::arg("par"), py::arg("vec_per_cta") = 512, py::arg("timing") = false)
      .def("stats", &SelfTest::stats, py::arg("reset") = false)
      .def("max_chunk", &SelfTest::max_chunk)
      .def("max_bytes", &SelfTest::max_bytes)
      .def("max_reduce_bytes", &SelfTest::max_reduce_bytes)
      .def("thp", &SelfTest::thp)
      .def("set_pub", &SelfTest::set_pub, py::arg("pub_mask"))
      .def("set_r3", &SelfTest::set_r3, py::arg("mask"), py::arg("s1_bytes") = 24576)
      .def("set_r6", &SelfTest::set_r6, py::arg("bits"))
      .def("peek_send", &SelfTest::peek_send, py::arg("which"), py::arg("slot"), py::arg("nbytes"))
      .def("counters", &SelfTest::counters)
      .def("all_reduce_proj_into", &SelfTest::all_reduce_proj_into, py::arg("out"), py::arg("x"), py::arg("two_shot"),
           py::arg("d"), py::arg("y"), py::arg("par"), py::arg("vec_per_cta") = 512, py::arg("timing") = false)
      .def("arrive3", &SelfTest::arrive3);
  m.attr("K1_VARIANT") = "3-rdma-loads-v1";
  m.attr("G2_VARIANT") = "colltrim-cols-v1";  // g2 colltrim: all_gather_cols / all_gather_cols_into present
  m.attr("G130_VARIANT") = "rdma-flags-v1";   // g130: RdmaAR.variant / set_variant, SelfTest.set_pub
  m.attr("R3_VARIANT") = "r3comm-v1";         // r3-comm: RdmaAR.r3 / set_r3 / send_buffer, SelfTest.set_r3
  m.attr("R6_VARIANT") = "r6comm-c2-v1";      // r6-comm C2: RdmaAR.r6 / set_r6, SelfTest.set_r6 (DSV41_RDMA_PDL)
  m.attr("R7_VARIANT") = "r7comm2-ht13-v1";   // r7-comm2 HT13: RdmaAR.all_reduce_proj / r7, SelfTest.all_reduce_proj_into
}
