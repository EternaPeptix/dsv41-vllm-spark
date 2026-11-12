#!/usr/bin/env python3
"""mk_mutants7.py <r7 src/rdma_ar.cu> <outdir> [names...]: must-fail mutants of the r7-comm2 source (one edit each).

The r6 set, re-based onto the r7 kernel signature (edits inside ar_kernel_r3 only; ar_kernel_pl stays the reference):
  m_rev m_seed0 m_negzero m_ffma m_tree   C1 reduction mutants (gate/fpflow.py FAIL; device: all but m_ffma)
  m_waitlate                              C2 wait after the stage (gate/pdlcheck.py W2; device: poison replay)
New in r7 (review item: the gate's concrete environment must enter the unrolled staging loop):
  m_stage    the 4x-unrolled grid staging loop and the S1 staging loop store a2 / a3 to each other's slots
             (send[i + 2s] = in[i + 3s] and vice versa)   gate: FAIL (staging copy proof, only reachable with an
             environment that enters the unrolled loop)   device: caught (send-slot bytes at bf16 M >= 39 / S1 M >= 2)
HT13 projection-tail mutants (edits inside r7_project only):
  mp_first   the chain starts x0*d0 then fma(x1, d1) (Triton starts x1*d1 then fma(x0, d0))   projgate FAIL, device
  mp_warp    warp butterfly in the order xor 1, 2, 4, 8, 16 (served: 16 .. 1)                projgate FAIL, device
  mp_l2      second-level butterfly xor 1, 2, 4 (served: 4, 2, 1)                             projgate FAIL, device
  mp_unfused y = x - s*d with two roundings (served: one FFMA)                                 projgate FAIL, device
  mp_chunk   chunk 2 masked one 8-element vector early (e + 8 < 5120): drops elements 5112..5119 of every row's
             x and d (a mask bug)                                                               projgate FAIL, device
HT13 last-CTA ordering mutants (review item 4; edits inside ar_kernel_r3's projection tail only):
  mo_nofence the per-thread __threadfence() before the barrier that precedes the arrive3 atomic is dropped (each CTA's
             p.out stores are no longer fenced at gpu scope before its arrival is counted)   ordgate FAIL (release)
  mo_first   the FIRST CTA through arrive3 projects (`== 0u` instead of `== gridDim.x - 1`): it reads p.out before
             the other CTAs have written their parts                                          ordgate FAIL (last-CTA)
  mo_noacq   the projecting CTA's __threadfence() after the s_go barrier is dropped (its p.out loads are no longer
             ordered after the other CTAs' fenced stores)                                    ordgate FAIL (acquire)
  The device test cannot catch these reliably (only by timing luck); gate/ordgate.py is the check that must.
The device test must catch every mutant marked "device" on its hostile inputs; m_ffma is gate-only (x*1 is exact)."""
import hashlib
import os
import sys

src, outdir = sys.argv[1], sys.argv[2]
base = open(src).read()
R3K = ("template <int kMode, bool kF32, int kW, int kPub>\n__global__ void __launch_bounds__(kThreads, 1) ar_kernel_r3(Params p, R7Proj q) {",
       "// ---- end r3-comm kernel")
PROJ = ("__device__ __forceinline__ void r7_project(", "__device__ __forceinline__ unsigned long long now_ns() {")
LOOP = """      float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int r = 0; r < kW; ++r) accv<kF32>(acc, v[r]);
"""
HELPERS = """// ---- r7 MUTANT helpers (never in a shipped source) ----
template <bool kF32>
__device__ __forceinline__ void mut_init(float* acc, const uint4& v) {
  if constexpr (kF32) {
    acc[0] = __uint_as_float(v.x);
    acc[1] = __uint_as_float(v.y);
    acc[2] = __uint_as_float(v.z);
    acc[3] = __uint_as_float(v.w);
  } else {
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      float2 f = __bfloat1622float2(h[k]);
      acc[2 * k] = f.x;
      acc[2 * k + 1] = f.y;
    }
  }
}
template <bool kF32>
__device__ __forceinline__ void mut_accv_fma(float* acc, const uint4& v, float one) {
  float x[8];
  mut_init<kF32>(x, v);
#pragma unroll
  for (int j = 0; j < (kF32 ? 4 : 8); ++j) acc[j] = __fmaf_rn(x[j], one, acc[j]);
}
template <bool kF32, int kW>
__device__ __forceinline__ void mut_tree(float* acc, const uint4* v) {
  float x[kW][8];
#pragma unroll
  for (int r = 0; r < kW; ++r) mut_init<kF32>(x[r], v[r]);
#pragma unroll
  for (int j = 0; j < (kF32 ? 4 : 8); ++j) {
    const float t01 = (0.f + x[0][j]) + x[1][j];
    const float t23 = x[2][j] + x[3][j];
    const float t45 = x[4][j] + x[5][j];
    const float t67 = x[6][j] + x[7][j];
    acc[j] = (t01 + t23) + (t45 + t67);
  }
}
// ---- end r7 MUTANT helpers ----
"""
K = "k"  # region tag: ar_kernel_r3
P = "p"  # region tag: r7_project
MUT = {
    "m_rev": [(K, LOOP, """      float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int r = kW - 1; r >= 0; --r) accv<kF32>(acc, v[r]);  // MUTANT m_rev
""", 2)],
    "m_seed0": [(K, LOOP, """      float acc[8];
      mut_init<kF32>(acc, v[0]);  // MUTANT m_seed0
#pragma unroll
      for (int r = 1; r < kW; ++r) accv<kF32>(acc, v[r]);
""", 2)],
    "m_negzero": [(K, LOOP, """      float acc[8] = {-0.f, -0.f, -0.f, -0.f, -0.f, -0.f, -0.f, -0.f};  // MUTANT m_negzero
#pragma unroll
      for (int r = 0; r < kW; ++r) accv<kF32>(acc, v[r]);
""", 2)],
    "m_ffma": [(K, LOOP, """      float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      const float mut_one = (float)(p.world / kW);  // MUTANT m_ffma: 1.0f at world 8, not foldable
#pragma unroll
      for (int r = 0; r < kW; ++r) mut_accv_fma<kF32>(acc, v[r], mut_one);
""", 2)],
    "m_tree": [(K, LOOP, """      float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      mut_tree<kF32, kW>(acc, v);  // MUTANT m_tree
""", 2)],
    "m_waitlate": [
        (K, """    if (threadIdx.x == 0 && (!kOne || blockIdx.x == 0)) early = ld_relaxed_sys(&p.ctrl->posted_seq);
    r6_gdc_wait();
  }
""", """    if (threadIdx.x == 0 && (!kOne || blockIdx.x == 0)) early = ld_relaxed_sys(&p.ctrl->posted_seq);
  }  // MUTANT m_waitlate: no wait here
""", 1),
        (K, """  if constexpr ((kPub & kR3P2) != 0) {
    if (blockIdx.x != 0) return;  // (iii) no consumer work here in either phase (len <= chunk <= blockDim)
  }
""", """  if constexpr ((kPub & kR6Pdl) != 0) r6_gdc_wait();  // MUTANT m_waitlate: after the stage + publish
  if constexpr ((kPub & kR3P2) != 0) {
    if (blockIdx.x != 0) return;  // (iii) no consumer work here in either phase (len <= chunk <= blockDim)
  }
""", 1)],
    "m_stage": [
        (K, """        send4[i + 2 * stride] = a2;
        send4[i + 3 * stride] = a3;
""", """        send4[i + 2 * stride] = a3;  // MUTANT m_stage
        send4[i + 3 * stride] = a2;  // MUTANT m_stage
""", 1),
        (K, """        if (b2) send4[i0 + 2 * nb] = a2;
        if (b3) send4[i0 + 3 * nb] = a3;
""", """        if (b2 && b3) send4[i0 + 2 * nb] = a3;  // MUTANT m_stage
        if (b2 && !b3) send4[i0 + 2 * nb] = a2;
        if (b3) send4[i0 + 3 * nb] = a2;  // MUTANT m_stage
""", 1)],
    "mp_first": [(P, """    float a = __fmul_rn(x[1], d[1]);
    a = __fmaf_rn(x[0], d[0], a);
""", """    float a = __fmul_rn(x[0], d[0]);  // MUTANT mp_first
    a = __fmaf_rn(x[1], d[1], a);
""", 1)],
    "mp_warp": [(P, """    for (int m = 16; m >= 1; m >>= 1) a = __fadd_rn(a, __shfl_xor_sync(0xffffffffu, a, m));""",
                 """    for (int m = 1; m <= 16; m <<= 1) a = __fadd_rn(a, __shfl_xor_sync(0xffffffffu, a, m));  // MUTANT mp_warp""", 1)],
    "mp_l2": [(P, """      for (int m = 4; m >= 1; m >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, m));""",
               """      for (int m = 1; m <= 4; m <<= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, m));  // MUTANT mp_l2""", 1)],
    "mp_unfused": [(P, """        for (int j = 0; j < 8; ++j) y[j] = __fmaf_rn(ns, d[c * 8 + j], x[c * 8 + j]);""",
                    """        for (int j = 0; j < 8; ++j) y[j] = __fadd_rn(x[c * 8 + j], __fmul_rn(ns, d[c * 8 + j]));  // MUTANT mp_unfused""", 1)],
    "mp_chunk": [(P, """      if (row < q.rows && e < kR7H) xv[k][c] = __ldcg(xo + row * (kR7H / 8) + e / 8);""",
                  """      if (row < q.rows && e + 8 < kR7H) xv[k][c] = __ldcg(xo + row * (kR7H / 8) + e / 8);  // MUTANT mp_chunk""", 1)],
    "mo_nofence": [(K, """      __shared__ int s_go;
      __threadfence();
      __syncthreads();
""", """      __shared__ int s_go;
      __syncthreads();  // MUTANT mo_nofence: no fence before the arrival barrier
""", 1)],
    "mo_first": [(K, """        const bool last = atom_add_acqrel_gpu(&p.st->arrive3, 1u) == gridDim.x - 1;""",
                  """        const bool last = atom_add_acqrel_gpu(&p.st->arrive3, 1u) == 0u;  // MUTANT mo_first""", 1)],
    "mo_noacq": [(K, """      if (s_go) {
        __threadfence();
        r7_project(p.out, q);
""", """      if (s_go) {
        r7_project(p.out, q);  // MUTANT mo_noacq: no fence before the projection's loads
""", 1)],
}


def mutate(name):
    s = base
    for tag, old, new, cnt in MUT[name]:
        a0, b0 = (R3K if tag == K else PROJ)
        a = s.index(a0)
        b = s.index(b0, a)
        reg = s[a:b]
        n = reg.count(old)
        if n != cnt:
            sys.exit(f"mk_mutants7.py {name}: expected {cnt} matches inside {'ar_kernel_r3' if tag == K else 'r7_project'}, found {n}")
        s = s[:a] + reg.replace(old, new) + s[b:]
    a = s.index(R3K[0])
    s = s[:a] + HELPERS + s[a:]
    old = 'm.attr("R7_VARIANT") = "r7comm2-ht13-v1";'
    if s.count(old) != 1:
        sys.exit("mk_mutants7.py: R7_VARIANT attr not found")
    s = s.replace(old, old + f'\n  m.attr("R7_MUTANT") = "{name}";\n  m.attr("R6_MUTANT") = "{name}";  // read by device_test6.py')
    return s


if __name__ == "__main__":
    os.makedirs(outdir, exist_ok=True)
    names = sys.argv[3:] or list(MUT)
    for nm in names:
        t = mutate(nm)
        p = os.path.join(outdir, f"rdma_ar.{nm}.cu")
        open(p, "w").write(t)
        print(f"{nm}: {p} sha256 {hashlib.sha256(t.encode()).hexdigest()[:16]}")
