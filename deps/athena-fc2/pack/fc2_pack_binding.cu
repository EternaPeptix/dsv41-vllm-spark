// SPDX-License-Identifier: Apache-2.0
// Detached FC2 packing binding. This does not install or dispatch an MoE kernel.
#ifndef ATHENA_EXPERIMENTAL_TP8_FC2_PACK_K288
#error "explicit experimental pack build opt-in required"
#endif
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <limits>
#include "fc2_pack_k288_source.cu"

namespace {
void pack(const at::Tensor& src, at::Tensor dst) {
  TORCH_CHECK(src.is_cuda() && src.is_contiguous() && src.dim() == 3 &&
              (src.scalar_type() == at::kByte || src.scalar_type() == at::kChar) &&
              src.size(0) == 384 && src.size(1) == 5120 && src.size(2) == 192,
              "source must be native contiguous FP4 bytes [384,5120,192]");
  TORCH_CHECK(dst.is_cuda() && dst.is_contiguous() && dst.dim() == 4 &&
              dst.scalar_type() == at::kByte && dst.size(0) == 384 &&
              dst.size(1) == 80 && dst.size(2) == 9 && dst.size(3) == 1024,
              "destination must be packed bytes [384,80,9,1024]");
  TORCH_CHECK(src.device() == dst.device(), "source/destination device mismatch");
  uintptr_t a = reinterpret_cast<uintptr_t>(src.data_ptr());
  uintptr_t b = reinterpret_cast<uintptr_t>(dst.data_ptr());
  const uint64_t asize = static_cast<uint64_t>(src.numel());
  const uint64_t bsize = static_cast<uint64_t>(dst.numel());
  TORCH_CHECK(a % 4 == 0 && b % 16 == 0, "source/destination alignment mismatch");
  TORCH_CHECK(a <= std::numeric_limits<uintptr_t>::max() - asize &&
              b <= std::numeric_limits<uintptr_t>::max() - bsize &&
              (a + asize <= b || b + bsize <= a), "source/destination ranges overlap or overflow");
  const c10::cuda::OptionalCUDAGuard guard(src.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(src.get_device()).stream();
  cudaStreamCaptureStatus capture;
  C10_CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
  TORCH_CHECK(capture == cudaStreamCaptureStatusNone, "pack must run outside CUDA graph capture");
  constexpr int64_t words = 384LL * 80 * 9 * 256;
  constexpr int threads = 256;
  athena_tp8_fc2_pack_k288<<<static_cast<unsigned>((words + threads - 1) / threads), threads, 0, stream>>>(
      reinterpret_cast<const uint32_t*>(src.data_ptr()), reinterpret_cast<uint32_t*>(dst.data_ptr()));
  C10_CUDA_CHECK(cudaGetLastError());
}
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pack", &pack, "Experimental FC2 K288 weight pack");
}
