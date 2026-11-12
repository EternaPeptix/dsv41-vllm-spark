// SPDX-License-Identifier: Apache-2.0
// Detached source-only contiguous grouped-row validator. No host hook.
#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace athena_tp8_grouped8 {
struct Group {
  int ids[8];
  int expert;
  int rows;
  int error;
};

// All 128 threads must enter with identical arguments. Return false uniformly
// for padding or invalid groups. Set status bit 1 for an invalid expert ID,
// bit 8 for a non-prefix/mixed-expert layout or out-of-range group. Other CTAs
// can have written already; caller must not consume any output on error.
__device__ __forceinline__ bool prepare_group(const int32_t* layout, int m_sum,
                                              int base, Group& group, int32_t* status) {
  if (blockDim.x != 128 || blockDim.y != 1 || blockDim.z != 1 ||
      !layout || !status || m_sum <= 0 || m_sum % 32 != 0 || base < 0 || base + 8 > m_sum) {
    if (threadIdx.x == 0 && status) atomicOr(status, 8);
    return false;
  }
  if (threadIdx.x < 8) group.ids[threadIdx.x] = layout[base + threadIdx.x];
  __syncthreads();
  if (threadIdx.x == 0) {
    int expert = -1, rows = 0, error = 0;
    bool padding = false;
    for (int i = 0; i < 8; ++i) {
      const int e = group.ids[i];
      if (e < 0) { padding = true; continue; }
      if (e >= 384) { error |= 1; continue; }
      if (padding || (expert >= 0 && e != expert)) error |= 8;
      if (expert < 0) expert = e;
      ++rows;
    }
    group.expert = expert;
    group.rows = rows;
    group.error = error;
    if (error) atomicOr(status, error);
  }
  __syncthreads();
  return group.error == 0 && group.rows != 0;
}
}  // namespace athena_tp8_grouped8
