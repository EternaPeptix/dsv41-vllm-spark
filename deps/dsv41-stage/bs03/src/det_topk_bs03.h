// BS03: the interface between the kernels (det_topk_bs03.cu, no torch headers) and the torch glue (det_topk_bs03_py.cpp).
#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace bs03 {

struct DecodeArgs {
    const float* logits;
    int64_t stride0;
    int64_t stride1;
    int64_t width;
    const int32_t* seq_lens;
    int64_t num_seq_lens;
    int64_t next_n;
    int32_t* out;
    int64_t out_stride0;
    int64_t num_rows;
    int64_t k;
    int64_t cluster;
    int64_t small;
    int device;
    cudaStream_t stream;
};

struct ProbeResult {
    int cluster;
    int optin;
    int static_smem;
    int num_regs;
    int max_active_clusters;
    size_t dyn;
};

void launch_decode(const DecodeArgs& a);
void configure(int64_t dyn_cap_bytes);
ProbeResult probe(int cluster, int device);

}  // namespace bs03
