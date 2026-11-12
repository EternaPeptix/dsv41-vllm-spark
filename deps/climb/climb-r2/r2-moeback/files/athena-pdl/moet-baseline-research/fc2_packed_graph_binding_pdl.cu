// SPDX-License-Identifier: Apache-2.0
// Experimental standalone FC2 host binding. Not installed in vLLM.
// [r2-moeback G18] Copy of the live graph binding (fc2_packed_graph_binding.generated.cu sha256 1d25bbe4) whose
// only change is the launch: cudaLaunchKernelEx with programmatic stream serialization (PDL), same grid,
// block, smem, stream and arguments. Built as module athena_tp8_fc2_packed_graph_sm121 (the name the overlay
// loads) into athena_tp8_fc2_packed_graph_pdl_sm121.so.
#ifndef ATHENA_FC2_PDL_EARLY_B
#error "the PDL binding needs the early-B kernel (ATHENA_FC2_PDL_EARLY_B)"
#endif
// The kernel and layouts have not been compiled or executed.
#ifndef ATHENA_EXPERIMENTAL_TP8_NATIVE_FC2
#error "Explicit experimental TP8 FC2 build opt-in is required"
#endif
#ifndef ATHENA_EXPERIMENTAL_TP8_FC2_GRAPH_BINDING
#error "explicit experimental caller-reset graph binding required"
#endif
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <limits>
#include <vector>
#include "fc2_serial32packed_k288_pdl.cu"

namespace {
struct ByteRange { uintptr_t begin, end; };

ByteRange range(const at::Tensor& tensor, int64_t elements) {
  TORCH_CHECK(elements > 0, "empty range is unsupported");
  const uintptr_t begin = reinterpret_cast<uintptr_t>(tensor.data_ptr());
  const uint64_t bytes = static_cast<uint64_t>(elements) * tensor.element_size();
  TORCH_CHECK(begin <= std::numeric_limits<uintptr_t>::max() - bytes, "tensor range overflow");
  return {begin, begin + static_cast<uintptr_t>(bytes)};
}
bool overlap(ByteRange a, ByteRange b) { return a.begin < b.end && b.begin < a.end; }

void launch(const at::Tensor& aq, const at::Tensor& aq_scale,
            const at::Tensor& group_layout, const at::Tensor& weight,
            const at::Tensor& weight_scale, at::Tensor out, at::Tensor status) {
  TORCH_CHECK(aq.is_cuda() && aq.dim() == 2 && aq.scalar_type() == at::kFloat8_e4m3fn &&
              aq.is_contiguous() && aq.size(1) == 384, "aq must be contiguous CUDA E4M3 [M,384]");
  const int64_t m = aq.size(0);
  TORCH_CHECK(m > 0 && m <= 8192 && m % 32 == 0, "M must be 32-aligned in [32,8192]");
  TORCH_CHECK(aq_scale.is_cuda() && aq_scale.scalar_type() == at::kInt && aq_scale.dim() == 2 &&
              aq_scale.size(0) == m && aq_scale.size(1) == 1 && aq_scale.stride(0) == 1 &&
              aq_scale.stride(1) >= m && aq_scale.stride(1) % 4 == 0,
              "aq_scale must be packed int32 [M,1] with MN-major strides");
  TORCH_CHECK(group_layout.is_cuda() && group_layout.scalar_type() == at::kInt &&
              group_layout.dim() == 1 && group_layout.is_contiguous() && group_layout.numel() == m,
              "group_layout must be contiguous int32 [M]");
  TORCH_CHECK(weight.is_cuda() && (weight.scalar_type() == at::kByte || weight.scalar_type() == at::kChar) &&
              weight.dim() == 4 && weight.is_contiguous() && weight.size(0) == 384 &&
              weight.size(1) == 80 && weight.size(2) == 9 && weight.size(3) == 1024,
              "weight must be prepacked FP4 bytes [384,80,9,1024]");
  TORCH_CHECK(weight_scale.is_cuda() && weight_scale.scalar_type() == at::kInt && weight_scale.dim() == 3 &&
              weight_scale.size(0) == 384 && weight_scale.size(1) == 5120 && weight_scale.size(2) == 3 &&
              weight_scale.stride(0) == 5120 * 3 && weight_scale.stride(1) == 1 && weight_scale.stride(2) == 5120,
              "weight_scale must be packed int32 [384,5120,3] in the inspected MN-major layout");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kBFloat16 && out.dim() == 2 && out.is_contiguous() &&
              out.size(0) == m && out.size(1) == 5120, "out must be contiguous BF16 [M,5120]");
  TORCH_CHECK(status.is_cuda() && status.scalar_type() == at::kInt && status.dim() == 1 &&
              status.is_contiguous() && status.numel() == 1, "status must be int32 CUDA [1]");
  for (const auto& t : {aq_scale, group_layout, weight, weight_scale, out, status})
    TORCH_CHECK(t.device() == aq.device(), "all tensors must be on the same device");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(aq.data_ptr()) % 8 == 0 &&
              reinterpret_cast<uintptr_t>(weight.data_ptr()) % 4 == 0 &&
              reinterpret_cast<uintptr_t>(weight_scale.data_ptr()) % 4 == 0 &&
              reinterpret_cast<uintptr_t>(aq_scale.data_ptr()) % 4 == 0 &&
              reinterpret_cast<uintptr_t>(group_layout.data_ptr()) % 4 == 0 &&
              reinterpret_cast<uintptr_t>(status.data_ptr()) % 4 == 0 &&
              reinterpret_cast<uintptr_t>(out.data_ptr()) % 4 == 0,
              "tensor pointers do not meet kernel alignment requirements");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(weight.data_ptr()) % 16 == 0, "vector FC2 weight must be 16-byte aligned");
  const std::vector<ByteRange> reads = {
      range(aq, m * 384), range(aq_scale, m), range(group_layout, m),
      range(weight, 384LL * 80 * 9 * 1024), range(weight_scale, 384LL * 5120 * 3)};
  const auto output = range(out, m * 5120), control = range(status, 1);
  TORCH_CHECK(!overlap(output, control), "out and status overlap");
  for (const auto& input : reads)
    TORCH_CHECK(!overlap(input, output) && !overlap(input, control), "output/status overlaps an input range");

  const c10::cuda::OptionalCUDAGuard guard(aq.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(aq.get_device()).stream();
  // Experimental: capture is permitted. Caller owns status reset,
  // lifetime, stream ordering and publication; this function does not.
  // Candidate atomically accumulates structural error bits.
  // NO status reset here: the forward owner must reset before each step.
  const dim3 grid(80, static_cast<unsigned>(m / 32)), block(128);
  cudaLaunchConfig_t config = {};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attribute[1];
  attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute[0].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attribute;
  config.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&config, athena_tp8_fc2_direct::athena_tp8_fc2_serial32packed_k288_candidate,
      reinterpret_cast<const uint32_t*>(aq.data_ptr()), static_cast<const int32_t*>(aq_scale.data_ptr<int32_t>()),
      static_cast<const int32_t*>(group_layout.data_ptr<int32_t>()), static_cast<int>(m),
      reinterpret_cast<const uint32_t*>(weight.data_ptr()), static_cast<int64_t>(weight.numel() / 4),
      static_cast<const int32_t*>(weight_scale.data_ptr<int32_t>()), static_cast<int64_t>(weight_scale.numel()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), status.data_ptr<int32_t>()));
  C10_CUDA_CHECK(cudaGetLastError());
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("launch", &launch, "Experimental caller-reset TP8 FC2 graph launch (PDL early-B, r2-moeback G18)");
}
