// SPDX-License-Identifier: Apache-2.0
// Experimental, unbuilt TP8 FC2 grouped-eight-row kernel. No serving hook.
// Reuses the Apache-2.0 MoET-derived row kernel's MMA/staging helpers.
#if defined(ATHENA_EXPERIMENTAL_TP8_NATIVE_FC2) && defined(ATHENA_EXPERIMENTAL_TP8_FC2_GROUPED8)
#include "../tp8_fc2_direct/fc2_row_candidate.cu"
#include "group_layout8.cuh"

namespace athena_tp8_fc2_direct {
// Grid (80 output blocks, M_sum/8 row groups), block 128. The host must check
// every tensor shape/stride, pointer range, ownership and stream dependency.
// Status is caller-zeroed and all candidate output is invalid on any error.
// No host binding or safe all-rank status-consumption path exists yet.
extern "C" __global__ __launch_bounds__(kThreads) void athena_tp8_fc2_grouped8_candidate(
    const uint32_t* aq_words, const int32_t* a_scale,
    const int32_t* group_layout, int m_sum,
    const uint32_t* native_weight, int64_t weight_words,
    const int32_t* native_scale, int64_t scale_words,
    __nv_bfloat16* out, int32_t* status) {
  if (!aq_words || !a_scale || !group_layout || !native_weight || !native_scale || !out || !status ||
      m_sum <= 0 || m_sum % 32 != 0 || blockIdx.x >= kOutput / kColumns ||
      blockIdx.y >= static_cast<unsigned>(m_sum / 8)) {
    if (threadIdx.x == 0 && status) atomicOr(status, 2);
    return;
  }
  const int base = static_cast<int>(blockIdx.y) * 8;
  __shared__ athena_tp8_grouped8::Group group;
  if (!athena_tp8_grouped8::prepare_group(group_layout, m_sum, base, group, status)) return;

  __shared__ __align__(16) uint32_t shared_weights[kWeightWords];
  __shared__ __align__(16) int32_t shared_scales[kScaleWords];
  if (!stage_384_fc2_64(native_weight, weight_words, native_scale, scale_words,
                        shared_weights, kWeightWords, shared_scales, kScaleWords,
                        group.expert, static_cast<int>(blockIdx.x))) {
    if (threadIdx.x == 0) atomicOr(status, 2);
    return;
  }

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int q = lane >> 2, c = lane & 3;
  // Non-live lanes use an in-range row to participate in warp MMA but never
  // store it; scale and activation of each live row remain independent.
  const int row = base + (q < group.rows ? q : 0);
  const uint32_t packed_a_scale = static_cast<uint32_t>(a_scale[row]);
  const uint32_t* arow = aq_words + static_cast<int64_t>(row) * 96;
  const uint32_t sb = static_cast<uint32_t>(__cvta_generic_to_shared(shared_weights));
  const uint32_t sf = static_cast<uint32_t>(__cvta_generic_to_shared(shared_scales));
  const uint32_t bbase = sb + ((warp >> 1) * 32 + lane) * 16 + (warp & 1) * 8;
  const int lc0 = warp * 16 + q;
  float acc[2][4] = {};
#pragma unroll
  for (int slice = 0; slice < 3; ++slice) {
    const uint32_t exponent = (packed_a_scale >> (slice * 8)) & 0xffu;
    const uint32_t sfa = exponent * 0x01010101u;
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
                     : "r"(sf + slice * 256 + (lc0 + nt * 8) * 4));
        if (kb == 0) qmma<0>(acc[nt], av.x, av.y, b0, b1, sfa, sfb);
        else if (kb == 1) qmma<1>(acc[nt], av.x, av.y, b0, b1, sfa, sfb);
        else if (kb == 2) qmma<2>(acc[nt], av.x, av.y, b0, b1, sfa, sfb);
        else qmma<3>(acc[nt], av.x, av.y, b0, b1, sfa, sfb);
      }
    }
  }
  if (q < group.rows) {
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
#endif
