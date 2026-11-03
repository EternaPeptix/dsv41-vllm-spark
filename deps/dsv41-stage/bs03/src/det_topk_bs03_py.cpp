// BS03: torch and pybind glue of det_topk_bs03.cu, host code only. The checks and the guard are Moet's
// det_topk_sm120.cu launch() and det_topk_decode() (vLLM-Moet 7f60bee, Apache-2.0), moved here; see that file's header.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include "det_topk_bs03.h"

void det_topk_decode(const torch::Tensor& logits, const torch::Tensor& seq_lens, int64_t next_n, torch::Tensor out,
                     int64_t num_rows, int64_t k, int64_t cluster, int64_t small) {
    TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == torch::kFloat32 && logits.dim() == 2,
                "logits: fp32 [rows, cols]");
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kInt32 && out.dim() == 2 && out.stride(1) == 1,
                "out: int32 [rows, >=k] with unit column stride");
    TORCH_CHECK(out.size(0) >= num_rows && out.size(1) >= k, "out too small");
    TORCH_CHECK(seq_lens.is_cuda() && seq_lens.scalar_type() == torch::kInt32 && seq_lens.is_contiguous() && next_n >= 1);
    const c10::cuda::CUDAGuard guard(logits.device());
    bs03::DecodeArgs a{logits.data_ptr<float>(),
                       logits.stride(0),
                       logits.stride(1),
                       logits.size(1),
                       seq_lens.data_ptr<int32_t>(),
                       seq_lens.numel(),
                       next_n,
                       out.data_ptr<int32_t>(),
                       out.stride(0),
                       num_rows,
                       k,
                       cluster,
                       small,
                       logits.get_device(),
                       c10::cuda::getCurrentCUDAStream()};
    bs03::launch_decode(a);
}

void det_topk_configure(int64_t dyn_cap_bytes) { bs03::configure(dyn_cap_bytes); }

pybind11::dict det_topk_probe(int64_t cluster, int64_t device) {
    const bs03::ProbeResult r = bs03::probe(static_cast<int>(cluster), static_cast<int>(device));
    pybind11::dict d;
    d["cluster"] = r.cluster;
    d["optin"] = r.optin;
    d["static_smem"] = r.static_smem;
    d["dyn"] = r.dyn;
    d["num_regs"] = r.num_regs;
    d["max_active_clusters"] = r.max_active_clusters;
    return d;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("det_topk_decode", &det_topk_decode);
    m.def("det_topk_configure", &det_topk_configure);
    m.def("det_topk_probe", &det_topk_probe);
}
