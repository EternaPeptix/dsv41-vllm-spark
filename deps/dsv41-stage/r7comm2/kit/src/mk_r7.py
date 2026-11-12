#!/usr/bin/env python3
"""mk_r7.py <r6 rdma_ar.cu 2bff8a89> <out rdma_ar.cu>: the r7-comm2 source (climb round 7, 2026-11-09).

r7-comm2 = r6-comm (C1 DSV41_RDMA_TRIM, C2 DSV41_RDMA_PDL, both unchanged) + HT13: the dsv41_oblit `_proj_rows`
projection folded into the collective tail (DSV41_RDMA_PROJ). Count-asserted edits on the r6 source:
  * ar_kernel_pl, ar_kernel, every helper they use, Params, Ctrl and every existing DevState offset keep their text.
    DevState gains one appended counter (arrive3); the r3 kernel gains a second parameter (R7Proj q) that only the new
    kR7Proj instantiations read, so every r3 / r6 instantiation without bit 512 keeps its code (gated: SASS identity).
  * kR7Proj (= 512, bf16 one-shot / two-shot only): after the final reduce / copy of p.out, the CTA that holds the whole
    output (CTA 0 under P2; otherwise the last CTA through a grid-wide acq_rel arrival counter, as last_cta) computes,
    for every row r < q.rows of the bf16 [rows, 5120] output, EXACTLY the served Triton program
        dsv41_oblit._proj_rows[(rows,)](x2=out, y, d, 5120, 5120, H=5120, BLOCK=8192, num_warps=8)
    i.e. thread t (0..255 of a 256-thread group = one Triton program) holds elements c*2048 + t*8 + j (c = 0..3, j = 0..7,
    masked to 0 at >= 5120), forms acc = x1*d1, then fma(x_j, d_j, acc) for j = 0, 2, 3, ..., 31 (fp32, RN), reduces
    acc over the warp with the xor-16/8/4/2/1 butterfly, stores warp sums to shared memory, reduces the 8 warp sums in
    lanes 0..7 of warp 0 with the xor-4/2/1 butterfly, and writes y_e = bf16_rn(fma(-s, d_e, x_e)) for e < 5120. Rows
    go to two groups (rows g, g+2, g+4): q.rows <= 6. The op order is read from the served cubin's PTX/SASS
    (ref/triton/), and gate/projgate.py proves it on both binaries. p.out is still written unchanged.
  * Host: RdmaAR.all_reduce_proj(x, two_shot, d, y) (needs DSV41_RDMA_PROJ=1 and DSV41_RDMA_TRIM=1 at construction,
    bf16 [rows, 5120], rows <= 6, d fp32 [5120], y bf16 like x); SelfTest.all_reduce_proj_into(...) and arrive3() for
    the device test. Module attr R7_VARIANT.
Every other r3 code path (staging, publish, flags, the reduction chains) is untouched.
"""
import hashlib
import sys

src, out = sys.argv[1], sys.argv[2]
s = open(src).read()
h = hashlib.sha256(s.encode()).hexdigest()
if not h.startswith("2bff8a89"):
    sys.exit(f"mk_r7.py: base sha {h[:8]} is not 2bff8a89 (climb-r6/comm/src/rdma_ar.cu)")


def rep(old, new, count=1):
    global s
    n = s.count(old)
    if n != count:
        sys.exit(f"mk_r7.py: expected {count} matches, found {n}:\n{old[:300]}")
    s = s.replace(old, new)


# ------------------------------------------------------------------------------------------------------- header ---
rep("""// Same FP stream as r3 / live (gate/fpflow.py on the binary); scheduling only.
""", """// Same FP stream as r3 / live (gate/fpflow.py on the binary); scheduling only.
//
// r7-comm2 (climb round 7, 2026-11-09), on top of r6-comm 2bff8a89. Default OFF.
//   DSV41_RDMA_PROJ=1     HT13 (needs DSV41_RDMA_TRIM=1) RdmaAR.all_reduce_proj: the bf16 [rows <= 6, 5120] all-reduce
//                             also writes y = the dsv41_oblit _proj_rows projection of its final bf16 output, computed in
//                             the kernel tail with the served Triton program's exact op order (gate/projgate.py).
""")
rep("""enum R6Bits : int { kR6Pdl = 128, kR6Trig = 256 };  // r6-comm C2: PDL in-edge / early trigger (instantiation bits)""",
    """enum R6Bits : int { kR6Pdl = 128, kR6Trig = 256 };  // r6-comm C2: PDL in-edge / early trigger (instantiation bits)
enum R7Bits : int { kR7Proj = 512 };  // r7-comm2 HT13: projection tail (instantiation bit, bf16 one/two-shot only)""")
rep("""  unsigned int entered;  // r3 G06 (iv): CTA entry counter of ar_kernel_r3 (appended: existing offsets unchanged)
};""", """  unsigned int entered;  // r3 G06 (iv): CTA entry counter of ar_kernel_r3 (appended: existing offsets unchanged)
  unsigned int arrive3;  // r7-comm2 HT13: CTA arrival counter of the projection tail (appended; back to 0 after each op)
};""")

# ----------------------------------------------------------------------------------------------- R7Proj + tail ---
rep("""__device__ __forceinline__ uint64_t ld_acquire_sys(const uint64_t* p) {""",
    """// r7-comm2 HT13: the projection operands of one kR7Proj launch (a second kernel parameter; zero for every other launch)
struct R7Proj {
  const float* d;   // unit direction, fp32 [kR7H], 16-byte aligned (dsv41_oblit._vec)
  uint4* y;         // projected output, bf16 [rows, kR7H], contiguous, 16-byte aligned
  long long rows;   // 1 .. kR7MaxRows
};
constexpr int kR7H = 5120;        // DSv4.1 hidden size: _proj_rows H (BLOCK 8192, num_warps 8, 256 threads per row)
constexpr int kR7MaxRows = 6;     // two 256-thread groups x kR7RowsPerGroup rows
constexpr int kR7RowsPerGroup = 3;

__device__ __forceinline__ uint64_t ld_acquire_sys(const uint64_t* p) {""")

rep("""__device__ __forceinline__ unsigned long long now_ns() {""",
    """// r7-comm2 HT13: bf16 halves of a 32-bit word, widened exactly to fp32 (the served kernel's cvt.f32.bf16)
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

__device__ __forceinline__ unsigned long long now_ns() {""")

rep("""template <int kMode, bool kF32, int kW, int kPub>
__global__ void __launch_bounds__(kThreads, 1) ar_kernel_r3(Params p) {""",
    """template <int kMode, bool kF32, int kW, int kPub>
__global__ void __launch_bounds__(kThreads, 1) ar_kernel_r3(Params p, R7Proj q) {""")
rep("""  static_assert((kPub & kR6Pdl) == 0 || (kPub & kR3Sbuf) == 0, "r6: PDL is not combined with the G26 send buffers");
""", """  static_assert((kPub & kR6Pdl) == 0 || (kPub & kR3Sbuf) == 0, "r6: PDL is not combined with the G26 send buffers");
  static_assert((kPub & kR7Proj) == 0 || (!kF32 && (kMode == kOneShot || kMode == kTwoShot) && (kPub & kR3Sbuf) == 0),
                "r7: the projection tail is bf16 one-shot / two-shot only, never with a send buffer");
""")
rep("""  // 3) seq already advanced at entry (iv); the tail barrier exists only for the timing block (unchanged)
""", """  // r7-comm2 HT13: projection tail. Under P2 only CTA 0 is here and it wrote all of p.out itself (block-scope
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
""")

# ------------------------------------------------------------------------------------------------------ launches ---
rep("""template <int kMode, bool kF32, int kPub>
void launch_r3_one(const Params& p, int blocks, cudaStream_t stream) {""",
    """template <int kMode, bool kF32, int kPub>
void launch_r3_one(const Params& p, int blocks, cudaStream_t stream, const R7Proj& q) {""")
rep("""    CHECK_CUDA(cudaLaunchKernelEx(&cfg, ar_kernel_r3<kMode, kF32, kParW, (kMode == kTwoShot ? kPub : (kPub & ~(kPubV2 | kR3P2)))>, p));""",
    """    CHECK_CUDA(cudaLaunchKernelEx(&cfg, ar_kernel_r3<kMode, kF32, kParW, (kMode == kTwoShot ? kPub : (kPub & ~(kPubV2 | kR3P2)))>, p, q));""")
rep("""        <<<blocks, kThreads, 0, stream>>>(p);
  }
}
template <int kMode, bool kF32>
void launch_r3(const Params& p, int blocks, cudaStream_t stream, int pub) {
  constexpr int B = kPubV1 | kPubV2 | kPollThin | kR3Trim;
  switch (pub) {""", """        <<<blocks, kThreads, 0, stream>>>(p, q);
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
  switch (pub) {""")
# the 14 existing r3 cases pass q (zero for them)
n_old = s.count("(p, blocks, stream); break;")
if n_old != 13:
    sys.exit(f"mk_r7.py: expected 13 r3 launch cases, found {n_old}")
s = s.replace("(p, blocks, stream); break;", "(p, blocks, stream, q); break;")
rep("""      launch_r3_one<kMode, kF32, B | kR6Pdl | kR6Trig | kR3S1 | kR3P2>(p, blocks, stream);
      break;
    default: throw std::runtime_error("rdma_ar r3: unsupported kernel mask " + std::to_string(pub));""",
    """      launch_r3_one<kMode, kF32, B | kR6Pdl | kR6Trig | kR3S1 | kR3P2>(p, blocks, stream, q);
      break;
    default: throw std::runtime_error("rdma_ar r3: unsupported kernel mask " + std::to_string(pub));""")
rep("""void launch_kernel(const Params& p, long long vec_per_cta, bool par, cudaStream_t stream, int pub = 0) {""",
    """void launch_kernel(const Params& p, long long vec_per_cta, bool par, cudaStream_t stream, int pub = 0,
                   const R7Proj& q = R7Proj{nullptr, nullptr, 0}) {""")
rep("""    launch_r3<kMode, kF32>(p, (int)blocks, stream, pub);""",
    """    launch_r3<kMode, kF32>(p, (int)blocks, stream, pub, q);""")
rep("""  if ((pub & kR3Trim) != 0) {  // r3-comm: only with DSV41_RDMA_TRIM=1 (callers check par, world 8 and g130 mask 7)""",
    """  if ((pub & kR7Proj) != 0 && (pub & kR3Trim) == 0) throw std::runtime_error("rdma_ar r7: projection needs the r3 kernels");
  if ((pub & kR3Trim) != 0) {  // r3-comm: only with DSV41_RDMA_TRIM=1 (callers check par, world 8 and g130 mask 7)""")

# ------------------------------------------------------------------------------------------------------ env flag ---
rep("""struct Hca {""", """// r7-comm2 HT13: DSV41_RDMA_PROJ, the r3 on/off parser (unset / 0 / off / false / no = off; 1 / on / true / yes = on)
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

struct Hca {""")

# ------------------------------------------------------------------------------------------------------- RdmaAR ---
rep("""    if (r6_pdl_ != 0 && (r3_ & 1) == 0)  // r6-comm C2: PDL lives in the r3 kernels only
      throw std::runtime_error("rdma_ar r6: DSV41_RDMA_PDL needs DSV41_RDMA_TRIM=1");""",
    """    if (r6_pdl_ != 0 && (r3_ & 1) == 0)  // r6-comm C2: PDL lives in the r3 kernels only
      throw std::runtime_error("rdma_ar r6: DSV41_RDMA_PDL needs DSV41_RDMA_TRIM=1");
    if (r7_proj_ && (r3_ & 1) == 0)  // r7-comm2 HT13: the projection tail lives in the r3 kernels only
      throw std::runtime_error("rdma_ar r7: DSV41_RDMA_PROJ needs DSV41_RDMA_TRIM=1");""")
rep("""  // out = concat over ranks (dim 0) of x; any dtype, out must be contiguous with world * x.numel() elements
  void all_gather_into(torch::Tensor out, torch::Tensor x) {""",
    """  // r7-comm2 HT13: out = the all-reduce of x (exactly all_reduce's bytes) and y = _proj_rows(out, d), in one kernel.
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
  void all_gather_into(torch::Tensor out, torch::Tensor x) {""")
rep("""  template <int kMode, bool kF32>
  void launch(const torch::Tensor& x, torch::Tensor& out, int64_t seg_vec = 0) {""",
    """  template <int kMode, bool kF32>
  void launch(const torch::Tensor& x, torch::Tensor& out, int64_t seg_vec = 0, const R7Proj* q7 = nullptr) {""")
rep("""    launch_kernel<kMode, kF32>(p, vec_per_cta_, parload_ != 0, at::cuda::getCurrentCUDAStream(),
                               r3_ ? r3_pub(kMode, x, p) : pub_mask_);
  }""", """    int pub = r3_ ? r3_pub(kMode, x, p) : pub_mask_;
    if (q7 != nullptr) {  // r7-comm2 HT13 (never with a send buffer: all_reduce_proj inputs are never sbuf views)
      TORCH_CHECK((pub & kR3Trim) != 0 && (pub & kR3Sbuf) == 0, "rdma_ar r7: projection needs the staged r3 kernel");
      pub |= kR7Proj;
    }
    launch_kernel<kMode, kF32>(p, vec_per_cta_, parload_ != 0, at::cuda::getCurrentCUDAStream(), pub,
                               q7 != nullptr ? *q7 : R7Proj{nullptr, nullptr, 0});
  }""")
rep("""  int r6_pdl_ = env_r6_pdl();  // r6-comm C2: 0, kR6Pdl or kR6Pdl | kR6Trig (DSV41_RDMA_PDL; needs TRIM)
""", """  int r6_pdl_ = env_r6_pdl();  // r6-comm C2: 0, kR6Pdl or kR6Pdl | kR6Trig (DSV41_RDMA_PDL; needs TRIM)
  bool r7_proj_ = env_r7_proj();  // r7-comm2 HT13: all_reduce_proj allowed (DSV41_RDMA_PROJ; needs TRIM)
""")

# ----------------------------------------------------------------------------------------------------- SelfTest ---
rep("""  void all_gather_into(torch::Tensor out, torch::Tensor x, bool par, int64_t vec_per_cta, bool timing) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && out.is_contiguous() && out.scalar_type() == x.scalar_type() &&
                    out.numel() == x.numel() * world_,
                "selftest: bad all_gather args");
    go<kGather, false>(x, out, par, vec_per_cta, timing);
  }""", """  void all_gather_into(torch::Tensor out, torch::Tensor x, bool par, int64_t vec_per_cta, bool timing) {
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
  }""")
rep("""  void go(const torch::Tensor& x, torch::Tensor& out, bool par, int64_t vec_per_cta, bool timing, int64_t seg_vec = 0) {""",
    """  void go(const torch::Tensor& x, torch::Tensor& out, bool par, int64_t vec_per_cta, bool timing, int64_t seg_vec = 0,
          const R7Proj* q7 = nullptr) {""")
rep("""    const int pub = r3_ ? (pub_ | r3_bits(kMode, p.nvec, p.chunk, r3_s1_bytes_, false, kThreads) | r6_) : pub_;
    launch_kernel<kMode, kF32>(p, std::max<long long>(64, vec_per_cta), par, at::cuda::getCurrentCUDAStream(), pub);""",
    """    const int pub = (r3_ ? (pub_ | r3_bits(kMode, p.nvec, p.chunk, r3_s1_bytes_, false, kThreads) | r6_) : pub_) |
                    (q7 != nullptr ? kR7Proj : 0);
    launch_kernel<kMode, kF32>(p, std::max<long long>(64, vec_per_cta), par, at::cuda::getCurrentCUDAStream(), pub,
                               q7 != nullptr ? *q7 : R7Proj{nullptr, nullptr, 0});""")

# ------------------------------------------------------------------------------------------------------ pybind ---
rep("""      .def("r6", &RdmaAR::r6)
      .def("set_r6", &RdmaAR::set_r6, py::arg("bits"));""",
    """      .def("r6", &RdmaAR::r6)
      .def("set_r6", &RdmaAR::set_r6, py::arg("bits"))
      .def("all_reduce_proj", &RdmaAR::all_reduce_proj, py::arg("x"), py::arg("two_shot"), py::arg("d"), py::arg("y"))
      .def("r7", &RdmaAR::r7);""")
rep("""      .def("counters", &SelfTest::counters);""",
    """      .def("counters", &SelfTest::counters)
      .def("all_reduce_proj_into", &SelfTest::all_reduce_proj_into, py::arg("out"), py::arg("x"), py::arg("two_shot"),
           py::arg("d"), py::arg("y"), py::arg("par"), py::arg("vec_per_cta") = 512, py::arg("timing") = false)
      .def("arrive3", &SelfTest::arrive3);""")
rep("""  m.attr("R6_VARIANT") = "r6comm-c2-v1";      // r6-comm C2: RdmaAR.r6 / set_r6, SelfTest.set_r6 (DSV41_RDMA_PDL)
""", """  m.attr("R6_VARIANT") = "r6comm-c2-v1";      // r6-comm C2: RdmaAR.r6 / set_r6, SelfTest.set_r6 (DSV41_RDMA_PDL)
  m.attr("R7_VARIANT") = "r7comm2-ht13-v1";   // r7-comm2 HT13: RdmaAR.all_reduce_proj / r7, SelfTest.all_reduce_proj_into
""")

open(out, "w").write(s)
print(f"mk_r7.py: wrote {out} sha256 {hashlib.sha256(s.encode()).hexdigest()}")
