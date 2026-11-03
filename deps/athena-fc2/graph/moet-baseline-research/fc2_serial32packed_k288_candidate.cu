// SPDX-License-Identifier: Apache-2.0
// Experimental, unbuilt TP8 FC2 serial four-group K288 prepacked-tile and empty-subgroup fast-path kernel. No serving hook.
// Reuses the Apache-2.0 MoET-derived row kernel's MMA/staging helpers.
#if defined(ATHENA_EXPERIMENTAL_TP8_NATIVE_FC2) && defined(ATHENA_EXPERIMENTAL_TP8_FC2_SERIAL32PACKED_K288)
#include "../moet-sm121-port/tools/dsv41_sm121/tp8_fc2_direct/fc2_row_candidate.cu"
#include "../moet-sm121-port/tools/dsv41_sm121/tp8_grouped8/group_layout8.cuh"

namespace athena_tp8_fc2_direct {
// Prepacked FC2 representation [384,80,9,1024] bytes, in exactly the shared
// tile layout expected by this consumer. This is NOT the native weight ABI.
// Preparation and storage costs are outside this kernel and unproven here.
constexpr int kK288WeightWords = 9 * 256;
__device__ __forceinline__ bool stage_288_fc2_64(
    const uint32_t* native_weight, int64_t weight_words,
    const int32_t* native_scale, int64_t scale_words,
    uint32_t* dst_weights, int64_t dst_weight_words,
    int32_t* dst_scales, int64_t dst_scale_words,
    int expert, int output_item) {
  if (blockDim.x != kThreads || blockDim.y != 1 || blockDim.z != 1 ||
      !native_weight || !native_scale || !dst_weights || !dst_scales ||
      reinterpret_cast<uintptr_t>(native_weight) % 16 != 0 ||
      reinterpret_cast<uintptr_t>(native_scale) % 4 != 0 ||
      reinterpret_cast<uintptr_t>(dst_weights) % 4 != 0 ||
      reinterpret_cast<uintptr_t>(dst_scales) % 4 != 0 ||
      weight_words != static_cast<int64_t>(kExperts) * (kOutput / kColumns) * kK288WeightWords ||
      scale_words != static_cast<int64_t>(kExperts) * kOutput * 3 ||
      dst_weight_words < kK288WeightWords || dst_scale_words < kScaleWords ||
      expert < 0 || expert >= kExperts || output_item < 0 || output_item >= kOutput / kColumns)
    return false;
  const int output_base = output_item * kColumns;
  // Each warp accesses contiguous 16-byte vectors in the staged-tile order.
  // An original native weight tensor must be transformed into this layout by
  // a separate pack step; no reinterpretation of native weights is permitted.
  const int64_t tile = static_cast<int64_t>(expert) * (kOutput / kColumns) + output_item;
  const uint4* src_vec = reinterpret_cast<const uint4*>(native_weight) + tile * (kK288WeightWords / 4);
  uint4* dst_vec = reinterpret_cast<uint4*>(dst_weights);
  for (int i = threadIdx.x; i < kK288WeightWords / 4; i += kThreads)
    dst_vec[i] = src_vec[i];
  for (int i = threadIdx.x; i < kScaleWords; i += kThreads) {
    const int k128 = i / kColumns;
    const int row = output_base + i % kColumns;
    const int64_t source = static_cast<int64_t>(expert) * kOutput * 3 + k128 * kOutput + row;
    dst_scales[i] = native_scale[source];
  }
  __syncthreads();
  return true;
}
// Grid (80 output blocks, M_sum/32 groups of 32 rows), block 128. The host must check
// every tensor shape/stride, pointer range, ownership and stream dependency.
// Status is caller-zeroed and all candidate output is invalid on any error.
// No host binding or safe all-rank status-consumption path exists yet.
extern "C" __global__ __launch_bounds__(kThreads) void athena_tp8_fc2_serial32packed_k288_candidate(
    const uint32_t* aq_words, const int32_t* a_scale,
    const int32_t* group_layout, int m_sum,
    const uint32_t* native_weight, int64_t weight_words,
    const int32_t* native_scale, int64_t scale_words,
    __nv_bfloat16* out, int32_t* status) {
  if (blockDim.x != kThreads || blockDim.y != 1 || blockDim.z != 1 ||
      !aq_words || !a_scale || !group_layout || !native_weight || !native_scale || !out || !status ||
      blockDim.x != kThreads || blockDim.y != 1 || blockDim.z != 1 ||
      m_sum <= 0 || m_sum % 32 != 0 || blockIdx.x >= kOutput / kColumns ||
      blockIdx.y >= static_cast<unsigned>(m_sum / 32)) {
    if (threadIdx.x == 0 && status) atomicOr(status, 2);
    return;
  }
  // Every warp reads the same 32 layout entries and makes the same decision.
  // All-negative has exactly the existing padding semantics, including any
  // negative ID. Reading all 32 avoids dropping a live/invalid positive ID
  // after leading padding. The caller must not concurrently mutate layout.
  const int segment_base = static_cast<int>(blockIdx.y) * 32;
  const int segment_lane = threadIdx.x & 31;
  if (__all_sync(0xffffffffu, group_layout[segment_base + segment_lane] < 0)) return;
  __shared__ athena_tp8_grouped8::Group group;
  // CTA-uniform loop. Each eight-row segment is validated separately so this
  // supports arbitrary expert changes at subgroup boundaries and >8 rows.
#pragma unroll 1
  for (int subgroup = 0; subgroup < 4; ++subgroup) {
  const int base = static_cast<int>(blockIdx.y) * 32 + subgroup * 8;
  // Skip a subgroup only when every one of its eight IDs is negative. Each
  // warp reads the same IDs; positive invalid/malformed entries still reach
  // prepare_group and retain its status behavior. The previous live subgroup
  // finished with a CTA barrier before any shared storage can be reused.
  const int check_lane = threadIdx.x & 31;
  const int check_id = check_lane < 8 ? group_layout[base + check_lane] : -1;
  if (__all_sync(0xffffffffu, check_id < 0)) continue;
  if (!athena_tp8_grouped8::prepare_group(group_layout, m_sum, base, group, status)) {
    if (group.error) return;  // status already set; all threads agree
    __syncthreads();          // all-padding subgroup; no tile staging
    continue;
  }

  __shared__ __align__(16) uint32_t shared_weights[kK288WeightWords];
  __shared__ __align__(16) int32_t shared_scales[kScaleWords];
  if (!stage_288_fc2_64(native_weight, weight_words, native_scale, scale_words,
                        shared_weights, kK288WeightWords, shared_scales, kScaleWords,
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
      if (k32 >= 9) continue;  // TP8 logical K=288; remaining 96 values are padding
      const int a_offset = slice * 32 + (kb * 2 + (c >> 1)) * 4 + (c & 1) * 2;
      const uint2 av = *reinterpret_cast<const uint2*>(arow + a_offset);
      uint32_t w0, w1;
      // The shared-memory address is passed as an integer register; declare
      // the hidden memory read to the compiler as well as using CTA barriers.
      asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];" : "=r"(w0), "=r"(w1) : "r"(bbase + k32 * 1024) : "memory");
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        uint32_t b0, b1, sfb;
        spread_fp4(nt ? w1 : w0, b0, b1);
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(sfb)
                     : "r"(sf + slice * 256 + (lc0 + nt * 8) * 4) : "memory");
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
  // All threads finish reading this subgroup's shared tiles before the next
  // subgroup can overwrite them. No cross-CTA synchronization is implied.
  __syncthreads();
  }  // subgroup
}
}  // namespace athena_tp8_fc2_direct
#endif
