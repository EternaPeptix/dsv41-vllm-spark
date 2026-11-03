// SPDX-License-Identifier: Apache-2.0
// Experimental TP8 FC2 direct staging component; NOT a complete GEMM.
// Address permutation derived from the b12x_next and MoET sources documented
// in docs/tp8-integration-audit.md and tp8_direct_tiles/stage_native_tile.py.
// No build or execution has established the assumed encoding/layout contracts.
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace athena_tp8_fc2_direct {

// Opt-in at compile time; no runtime or live-engine hook is provided.
#if defined(ATHENA_EXPERIMENTAL_TP8_NATIVE_FC2)
constexpr int kExperts = 384;
constexpr int kOutput = 5120;
constexpr int kK32 = 12;              // actual DeepGEMM FC2 K=384
constexpr int kColumns = 64;
constexpr int kThreads = 128;
constexpr int kWordsPerRow = 48;       // 384 FP4 values / 8 values per int32
constexpr int kWeightWords = kK32 * 256;
constexpr int kScaleWords = 3 * kColumns;

// Produce the FC2 shared-memory B tile expected by the donor's k32/n32/lane
// load sequence: [12 K32][2 N32][128 int32 words]. The scale buffer is
// [3 K128 slices][64 output rows], each word containing four exponent bytes.
//
// Required source contract (NOT proved by this helper): native contiguous
// packed E2M1 w2 viewed as uint32 [384,5120,48]; packed UE8M0 scales viewed
// as int32 [384,5120,3] with strides (5120*3,1,5120); encodings and the
// b12x_next row/word permutation must match the consuming MoET MMA code.
//
// All 128 threads in the CTA must call with identical arguments. False means
// the uniformly checked shape/range/pointer contract failed before writes.
// `dst_weights` and `dst_scales` must refer to non-overlapping correctly sized
// shared-memory buffers; caller must establish source/destination ownership,
// graph/stream dependencies, compilation architecture and scale semantics.
// This CTA-wide barrier makes its stores visible to later work in this CTA;
// it is NOT a cross-CTA or cross-stream synchronization mechanism.
__device__ __forceinline__ bool stage_384_fc2_64(
    const uint32_t* native_weight, int64_t weight_words,
    const int32_t* native_scale, int64_t scale_words,
    uint32_t* dst_weights, int64_t dst_weight_words,
    int32_t* dst_scales, int64_t dst_scale_words,
    int expert, int output_item) {
  if (blockDim.x != kThreads || blockDim.y != 1 || blockDim.z != 1 ||
      !native_weight || !native_scale || !dst_weights || !dst_scales ||
      reinterpret_cast<uintptr_t>(native_weight) % 4 != 0 ||
      reinterpret_cast<uintptr_t>(native_scale) % 4 != 0 ||
      reinterpret_cast<uintptr_t>(dst_weights) % 4 != 0 ||
      reinterpret_cast<uintptr_t>(dst_scales) % 4 != 0 ||
      weight_words != static_cast<int64_t>(kExperts) * kOutput * kWordsPerRow ||
      scale_words != static_cast<int64_t>(kExperts) * kOutput * 3 ||
      dst_weight_words < kWeightWords || dst_scale_words < kScaleWords ||
      expert < 0 || expert >= kExperts || output_item < 0 || output_item >= kOutput / kColumns)
    return false;

  const int output_base = output_item * kColumns;
  for (int i = threadIdx.x; i < kWeightWords; i += kThreads) {
    const int k32 = i / 256;
    const int within = i % 256;
    const int n32 = within / 128;
    const int inner = within % 128;
    const int row = output_base + n32 * 32 + (inner % 4) * 8 + inner / 16;
    const int word = k32 * 4 + (inner / 4) % 4;
    const int64_t source = (static_cast<int64_t>(expert) * kOutput + row) * kWordsPerRow + word;
    dst_weights[i] = native_weight[source];
  }
  for (int i = threadIdx.x; i < kScaleWords; i += kThreads) {
    const int k128 = i / kColumns;
    const int row = output_base + i % kColumns;
    const int64_t source = static_cast<int64_t>(expert) * kOutput * 3 + k128 * kOutput + row;
    dst_scales[i] = native_scale[source];
  }
  __syncthreads();
  return true;
}
#endif  // ATHENA_EXPERIMENTAL_TP8_NATIVE_FC2
}  // namespace athena_tp8_fc2_direct
