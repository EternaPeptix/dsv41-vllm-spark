// SPDX-License-Identifier: Apache-2.0
// Experimental, unbuilt TP8 FC1 grouped-eight-row kernel. No serving hook.
// Reuses the Apache-2.0 MoET-derived row kernel's MMA/staging helpers.
#if defined(ATHENA_EXPERIMENTAL_TP8_NATIVE_FC1) && defined(ATHENA_EXPERIMENTAL_TP8_FC1_GROUPED8)
#include "../tp8_fc1_direct/fc1_row_candidate.cu"
#include "group_layout8.cuh"

namespace athena_tp8_fc1_direct {
// Grid (10 gate/up output tiles, M_sum/8 groups), block 128. Input FP32 scales
// are per row and K128, NOT shared across the group. All status errors make
// the entire candidate output unsafe to consume. Host binding is not supplied.
extern "C" __global__ __launch_bounds__(kThreads) void athena_tp8_fc1_grouped8_candidate(
    const uint32_t* aq_words, const float* a_scale, const int32_t* group_layout, int m_sum,
    const uint32_t* native_w, int64_t weight_words, const int32_t* native_s, int64_t scale_words,
    __nv_bfloat16* out, int32_t* status) {
  if (!aq_words || !a_scale || !group_layout || !native_w || !native_s || !out || !status ||
      m_sum <= 0 || m_sum % 32 != 0 || blockIdx.x >= 10 ||
      blockIdx.y >= static_cast<unsigned>(m_sum / 8) ||
      weight_words != static_cast<int64_t>(kExperts) * kRows * kWordsPerRow ||
      scale_words != static_cast<int64_t>(kExperts) * kRows * kGroups) {
    if (threadIdx.x == 0 && status) atomicOr(status, 2);
    return;
  }
  const int base = static_cast<int>(blockIdx.y) * 8;
  __shared__ athena_tp8_grouped8::Group group;
  if (!athena_tp8_grouped8::prepare_group(group_layout, m_sum, base, group, status)) return;
  const int projection = static_cast<int>(blockIdx.x) / 5;
  const int tile = static_cast<int>(blockIdx.x) % 5;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int q = lane >> 2, c = lane & 3;
  const int row = base + (q < group.rows ? q : 0);
  const uint32_t* arow = aq_words + static_cast<int64_t>(row) * (kHidden / 4);
  __shared__ __align__(16) uint32_t shared_w[kTileWeightWords];
  __shared__ __align__(16) int32_t shared_s[kTileScaleWords];
  __shared__ uint32_t scale_bits[8];
  __shared__ int bad_scale;
  const uint32_t sb = static_cast<uint32_t>(__cvta_generic_to_shared(shared_w));
  const uint32_t sf = static_cast<uint32_t>(__cvta_generic_to_shared(shared_s));
  const uint32_t bbase = sb + ((warp >> 1) * 32 + lane) * 16 + (warp & 1) * 8;
  const int lc0 = warp * 16 + q;
  float acc[2][4] = {};
#pragma unroll 1
  for (int slice = 0; slice < kGroups; ++slice) {
    if (threadIdx.x == 0) bad_scale = 0;
    __syncthreads();
    if (threadIdx.x < 8) {
      const int slot = threadIdx.x;
      const int source_row = base + (slot < group.rows ? slot : 0);
      const uint32_t bits = __float_as_uint(a_scale[static_cast<int64_t>(source_row) * kGroups + slice]);
      const uint32_t exponent = (bits >> 23) & 0xffu;
      scale_bits[slot] = bits;
      if (slot < group.rows && ((bits & 0x807fffffu) != 0 || exponent == 0 || exponent == 255))
        atomicOr(&bad_scale, 1);
    }
    __syncthreads();
    if (bad_scale) {
      if (threadIdx.x == 0) atomicOr(status, 4);
      return;  // CTA-uniform, before any later barrier
    }
    const uint32_t exponent = (scale_bits[q] >> 23) & 0xffu;
    const uint32_t sfa = exponent * 0x01010101u;
    stage_fc1_slice(native_w, native_s, shared_w, shared_s, group.expert, projection, tile, slice);
#pragma unroll
    for (int kb = 0; kb < 4; ++kb) {
      const int a_offset = slice * 32 + (kb * 2 + (c >> 1)) * 4 + (c & 1) * 2;
      const uint2 av = *reinterpret_cast<const uint2*>(arow + a_offset);
      uint32_t w0, w1;
      asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];" : "=r"(w0), "=r"(w1) : "r"(bbase + kb * 1024));
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        uint32_t b0, b1, sfb;
        spread_fp4(nt ? w1 : w0, b0, b1);
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(sfb) : "r"(sf + (lc0 + nt * 8) * 4));
        if (kb == 0) qmma<0>(acc[nt], av.x, av.y, b0, b1, sfa, sfb);
        else if (kb == 1) qmma<1>(acc[nt], av.x, av.y, b0, b1, sfa, sfb);
        else if (kb == 2) qmma<2>(acc[nt], av.x, av.y, b0, b1, sfa, sfb);
        else qmma<3>(acc[nt], av.x, av.y, b0, b1, sfa, sfb);
      }
    }
    __syncthreads();
  }
  if (q < group.rows) {
    const int col_base = tile * kOutputs + warp * 16 + c * 2;
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      const int col = col_base + nt * 8;
      if (col < kLive) {
        auto* dst = reinterpret_cast<__nv_bfloat162*>(out + static_cast<int64_t>(row) * kRows + projection * kLive + col);
        *dst = __floats2bfloat162_rn(acc[nt][0], acc[nt][1]);
      }
    }
  }
}
}  // namespace athena_tp8_fc1_direct
#endif
