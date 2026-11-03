// SPDX-License-Identifier: Apache-2.0
// Detached source-only native-FP4 to K288 shared-tile weight pack experiment.
// Not bound, built, run, or connected to the server. The caller owns allocation,
// stream/lifetime, 32/128-bit alignment as applicable, and the source encoding.
#if defined(ATHENA_EXPERIMENTAL_TP8_FC2_PACK_K288)
#include <cuda_runtime.h>
#include <cstdint>

// Input uint32 [384,5120,48], output uint32 [384,80,9,256]. Each output word
// uses the same row/word mapping as stage_288_fc2_64 in the scalar candidate.
// K288..383 is excluded. Raw data is rearranged, not numerically converted.
// Caller must supply non-overlapping storage and the documented TP8 padding
// and packing contract; this function does not inspect or validate contents.
extern "C" __global__ void athena_tp8_fc2_pack_k288(
    const uint32_t* input, uint32_t* output) {
  constexpr int64_t words_per_tile = 9 * 256;
  constexpr int64_t total_words = 384LL * 80 * words_per_tile;
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total_words) return;
  const int64_t tile = i / words_per_tile;
  const int expert = tile / 80;
  const int n_tile = tile % 80;
  const int within_tile = i % words_per_tile;
  const int k32 = within_tile / 256;
  const int within = within_tile % 256;
  const int n32 = within / 128;
  const int inner = within % 128;
  const int row = n_tile * 64 + n32 * 32 + (inner % 4) * 8 + inner / 16;
  const int word = k32 * 4 + (inner / 4) % 4;
  const int64_t source = (static_cast<int64_t>(expert) * 5120 + row) * 48 + word;
  output[i] = input[source];
}
#endif
