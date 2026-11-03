// SPDX-License-Identifier: Apache-2.0
// Experimental FC2 kernel, derived from the Apache-2.0 MoET donor's
// moe_fused_sm121_wide.cu. No host launch, build, execution or serving hook.
// Requires explicit compile-time opt-in and caller-owned preconditions below.
#if defined(ATHENA_EXPERIMENTAL_TP8_NATIVE_FC2)
#include "stage_native_fc2.cuh"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace athena_tp8_fc2_direct {

__device__ __forceinline__ void spread_fp4(uint32_t packed, uint32_t& lo, uint32_t& hi) {
  asm("{\n"
      ".reg .b32 s, t;\n"
      "shl.b32 s, %2, 2;\n"
      "shr.u32 t, %2, 2;\n"
      "prmt.b32 %0, s, t, 0x5140;\n"
      "prmt.b32 %1, s, t, 0x7362;\n"
      "and.b32 %0, %0, 0x3C3C3C3C;\n"
      "and.b32 %1, %1, 0x3C3C3C3C;\n"
      "}\n" : "=r"(lo), "=r"(hi) : "r"(packed));
}

template <int KB>
__device__ __forceinline__ void qmma(float (&d)[4], uint32_t a0, uint32_t a1,
                                      uint32_t b0, uint32_t b1, uint32_t sfa,
                                      uint32_t sfb) {
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m1.f32.ue8m0 "
      "{%0, %1, %2, %3}, {%4, %4, %5, %5}, {%6, %7}, {%0, %1, %2, %3}, {%8}, {%10, %11}, {%9}, {%10, %11};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(sfa), "r"(sfb), "n"(KB), "n"(0));
}

// Prospective layout: grid=(80 output blocks, M_sum rows), block=(128,1,1).
// One CTA handles one grouped-layout row and 64 outputs, avoiding any
// cross-CTA producer wait or duplicated-row write within this kernel.
//
// Caller MUST ensure: group_layout [M_sum] int32 with -1 for padding;
// aq [M_sum,384] contiguous E4M3, 8-byte aligned; a_scale [M_sum,1]
// MN-major packed K128 UE8M0 int32; w2 and w2_scale satisfy the source
// contracts in stage_native_fc2.cuh; output [M_sum,5120] BF16 contiguous and
// 4-byte aligned; every source/output/status allocation is non-overlapping;
// status is a caller-zeroed device int32 and must be checked before using any
// output; and dependencies and stream lifetime are established externally.
// Grouped rows may be padding and are not written. Bad positive expert IDs
// set status bit 1, but other CTAs may already have written: no partial result may
// be consumed on failure. This kernel itself cannot guarantee error handling.
//
// This is deliberately not registered to PyTorch/vLLM and is not executable
// from the current repository. The exact scale encoding, nibble layout, MMA
// selector semantics, BF16 rounding and numerical behavior remain unproved.
extern "C" __global__ __launch_bounds__(kThreads) void athena_tp8_fc2_row_candidate(
    const uint32_t* aq_words, const int32_t* a_scale,
    const int32_t* group_layout, int m_sum,
    const uint32_t* native_weight, int64_t weight_words,
    const int32_t* native_scale, int64_t scale_words,
    __nv_bfloat16* out, int32_t* status) {
  if (blockDim.x != kThreads || blockDim.y != 1 || blockDim.z != 1 ||
      !aq_words || !a_scale || !group_layout || !native_weight || !native_scale || !out || !status ||
      m_sum <= 0 || blockIdx.x >= kOutput / kColumns || blockIdx.y >= static_cast<unsigned>(m_sum))
    return;

  const int row = static_cast<int>(blockIdx.y);
  const int expert = group_layout[row];
  if (expert < 0) return;  // padding: gather must not consume this row
  if (expert >= kExperts) {
    if (threadIdx.x == 0) atomicOr(status, 1);
    return;
  }

  __shared__ __align__(16) uint32_t shared_weights[kWeightWords];
  __shared__ __align__(16) int32_t shared_scales[kScaleWords];
  if (!stage_384_fc2_64(native_weight, weight_words, native_scale, scale_words,
                        shared_weights, kWeightWords, shared_scales, kScaleWords,
                        expert, static_cast<int>(blockIdx.x))) {
    if (threadIdx.x == 0) atomicOr(status, 2);
    return;
  }

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int q = lane >> 2;
  const int c = lane & 3;
  const uint32_t packed_a_scale = static_cast<uint32_t>(a_scale[row]);
  const uint32_t* arow = aq_words + static_cast<int64_t>(row) * 96;
  const uint32_t sb = static_cast<uint32_t>(__cvta_generic_to_shared(shared_weights));
  const uint32_t scale_sb = static_cast<uint32_t>(__cvta_generic_to_shared(shared_scales));
  const uint32_t bbase = sb + ((warp >> 1) * 32 + lane) * 16 + (warp & 1) * 8;
  const int lc0 = warp * 16 + q;
  float acc[2][4] = {};

#pragma unroll
  for (int slice = 0; slice < 3; ++slice) {
    // Installed activation quantization produces one UE8M0 byte per K128.
    // The MoET MMA selects a byte per K32; repeat this byte four times.
    const uint32_t exp = (packed_a_scale >> (slice * 8)) & 0xffu;
    const uint32_t sfa = exp * 0x01010101u;
#pragma unroll
    for (int kb = 0; kb < 4; ++kb) {
      const int k32 = slice * 4 + kb;
      const int a_offset = slice * 32 + (kb * 2 + (c >> 1)) * 4 + (c & 1) * 2;
      const uint2 av = *reinterpret_cast<const uint2*>(arow + a_offset);
      uint32_t w0, w1;
      asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];" : "=r"(w0), "=r"(w1) : "r"(bbase + k32 * 1024));
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        uint32_t b0, b1, sfb;
        spread_fp4(nt ? w1 : w0, b0, b1);
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(sfb)
                     : "r"(scale_sb + slice * 256 + (lc0 + nt * 8) * 4));
        if (kb == 0) qmma<0>(acc[nt], av.x, av.y, b0, b1, sfa, sfb);
        else if (kb == 1) qmma<1>(acc[nt], av.x, av.y, b0, b1, sfa, sfb);
        else if (kb == 2) qmma<2>(acc[nt], av.x, av.y, b0, b1, sfa, sfb);
        else qmma<3>(acc[nt], av.x, av.y, b0, b1, sfa, sfb);
      }
    }
  }

  // MoET TP4 weights routes here; installed DeepGEMM instead emits BF16
  // unweighted rows, leaving router weighting to the later gather.
  if (q == 0) {
    const int col_base = static_cast<int>(blockIdx.x) * kColumns + warp * 16 + c * 2;
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      const int col = col_base + nt * 8;
      auto* dst = reinterpret_cast<__nv_bfloat162*>(out + static_cast<int64_t>(row) * kOutput + col);
      *dst = __floats2bfloat162_rn(acc[nt][0], acc[nt][1]);
    }
  }
}
}  // namespace athena_tp8_fc2_direct
#endif  // ATHENA_EXPERIMENTAL_TP8_NATIVE_FC2
