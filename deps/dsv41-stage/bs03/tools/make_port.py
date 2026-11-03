#!/usr/bin/env python3
"""make_port.py <moet det_topk_sm120.cu> <out det_topk_bs03.cu>: the BS03 port as exact edits of Moet's source.

Treats the Moet file as data: reads it, applies anchored replacements (each anchor must occur exactly once),
writes the result. Nothing from the Moet directory is executed. Refuses on any other input sha.
Every edit is in the host part; the check at the end proves the device code in front of it is unchanged.
"""
import hashlib
import sys

MOET_SHA256 = "7242c76acc5d9bfcf87f778017b064fb5d160781e502c995d70786a3f198e4e5"

# Moet's device code runs from this line up to KERNELS_END, and must stay byte-identical.
KERNELS_BEGIN = "namespace cg = cooperative_groups;\n"
KERNELS_END = "// Per device and cluster size: the dynamic shared memory for the key copy"

EDITS = [
    # 1. header: Apache-2.0 asks for a notice of changes
    (
        "// Deterministic exact top-k over rows of fp32 logits, for the DeepSeek-V4.1 DSA indexer on sm_120.\n",
        """// BS03 port for the DeepSeek-V4.1 TP8 stack on GB10 (sm_121, 48 SMs). Derived from vLLM-Moet (Apache-2.0)
// tools/dsv41_sm120/det_topk/det_topk_sm120.cu at commit 7f60bee1e8ef47cb5f486e8b0a757cc01487a155, modified.
// The device code is Moet's, byte for byte (tools/make_port.py checks it). Host changes, each marked "BS03:":
//   1. No torch headers here. The torch and pybind glue moved to det_topk_bs03_py.cpp, because one nvcc run over
//      torch/extension.h took MemAvailable below the 6 GB floor on a serving node. Same flags otherwise.
//   2. bs03::configure(): dynamic shared memory capped (default 64 KiB), not the opt-in maximum minus the static part.
//   3. The caller passes an explicit cluster size (1, 2, 4 or 8). No occupancy-based choice; C=16 is probed, not launched.
//   4. bs03::probe(): occupancy and resource numbers for test_det_topk.py. No kernel launch.
//   5. The prefill entry is removed: BS03 Design B replaces the decode chain only.
//
// Deterministic exact top-k over rows of fp32 logits, for the DeepSeek-V4.1 DSA indexer on sm_120.
""",
    ),
    (
        """#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cstdint>
""",
        """#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "det_topk_bs03.h"

// BS03: stand-ins for the two torch macros Moet's host part uses; pybind turns the exception into a RuntimeError.
#define C10_CUDA_CHECK(expr)                                                                            \\
    do {                                                                                                \\
        const cudaError_t bs03_e = (expr);                                                              \\
        if (bs03_e != cudaSuccess) {                                                                    \\
            (void)cudaGetLastError(); /* clears it, so a caught failure does not surface in the next kernel */ \\
            throw std::runtime_error(std::string("det_topk: ") + cudaGetErrorString(bs03_e));           \\
        }                                                                                               \\
    } while (0)
#define TORCH_CHECK(cond, msg)                            \\
    do {                                                  \\
        if (!(cond)) throw std::runtime_error("det_topk: " msg); \\
    } while (0)
""",
    ),
    # 2. dynamic smem cap
    (
        """// Per device and cluster size: the dynamic shared memory for the key copy (the opt-in maximum per block minus the static
// part) and how many such clusters the GPU holds at once.
struct ClusterInfo {
    size_t dyn = 0;
    int max_clusters = 0;
};
""",
        """// BS03: the dynamic shared memory cap. Moet used the opt-in maximum per block minus the static part; the largest shared
// memory in our decode trace is about 100 KB and the L1 reconfiguration cost on GB10 is unknown, so stay near what the
// traced kernels already use. 64 KiB holds 16384 keys, above the default 12288-column one-CTA limit. Set once, before
// the first launch (bs03::configure).
size_t g_dyn_cap = 64 * 1024;
bool g_dyn_cap_used = false;

// Per device and cluster size: the dynamic shared memory for the key copy (BS03: the opt-in maximum per block minus the
// static part, capped) and how many such clusters the GPU holds at once.
struct ClusterInfo {
    size_t dyn = 0;
    int max_clusters = 0;
    int optin = 0;
    int static_smem = 0;
    int num_regs = 0;
};
""",
    ),
    (
        """    if (device < 64 && cached[device].dyn) return cached[device];
    int optin = 0;
""",
        """    if (device < 64 && cached[device].dyn) return cached[device];
    g_dyn_cap_used = true;
    int optin = 0;
""",
    ),
    (
        """    info.dyn = (static_cast<size_t>(optin) - fa.sharedSizeBytes) & ~static_cast<size_t>(15);
""",
        """    info.dyn = std::min(static_cast<size_t>(optin) - fa.sharedSizeBytes, g_dyn_cap) & ~static_cast<size_t>(15);  // BS03
    info.optin = optin;
    info.static_smem = fa.sharedSizeBytes;
    info.num_regs = fa.numRegs;
""",
    ),
    # 3. launch functions: plain arguments instead of torch tensors, the explicit cluster size
    (
        """template <int C>
void launch_cluster(const torch::Tensor& logits, const RowBounds& b, torch::Tensor& out, int64_t k, bool relative,
                    int64_t small, cudaStream_t stream) {
    const size_t dyn = cluster_info<C>(logits.get_device()).dyn;
""",
        """template <int C>
void launch_cluster(const bs03::DecodeArgs& a, const RowBounds& b, bool relative) {
    const size_t dyn = cluster_info<C>(a.device).dyn;
""",
    ),
    (
        """    cfg.stream = stream;
""",
        """    cfg.stream = a.stream;
""",
    ),
    (
        """    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, det_topk_cluster_kernel<C>, logits.data_ptr<float>(), logits.stride(0),
                                      logits.stride(1), b, out.data_ptr<int32_t>(), out.stride(0), static_cast<int>(k),
                                      relative, small, static_cast<int64_t>(dyn / sizeof(uint32_t))));
""",
        """    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, det_topk_cluster_kernel<C>, a.logits, a.stride0, a.stride1, b, a.out,
                                      a.out_stride0, static_cast<int>(a.k), relative, a.small,
                                      static_cast<int64_t>(dyn / sizeof(uint32_t))));
""",
    ),
    (
        """void launch(const torch::Tensor& logits, const RowBounds& b, torch::Tensor& out, int64_t k, bool relative,
            int64_t cluster, int64_t small) {
    TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == torch::kFloat32 && logits.dim() == 2,
                "logits: fp32 [rows, cols]");
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kInt32 && out.dim() == 2 && out.stride(1) == 1,
                "out: int32 [rows, >=k] with unit column stride");
    TORCH_CHECK(out.size(0) >= b.num_rows && out.size(1) >= k, "out too small");
    TORCH_CHECK(k > 0 && k <= 65535, "k out of range");
    if (b.num_rows == 0) return;
    const c10::cuda::CUDAGuard guard(logits.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    // cluster < 0: the largest cluster size whose clusters for all rows run at once (by row count only: the row lengths
    // live on the device, and CUDA graphs replay one launch shape for every context length).
    if (cluster < 0) {
        const int dev = logits.get_device();
        cluster = b.num_rows <= cluster_info<16>(dev).max_clusters  ? 16
                  : b.num_rows <= cluster_info<8>(dev).max_clusters ? 8
                  : b.num_rows <= cluster_info<4>(dev).max_clusters ? 4
                  : b.num_rows <= cluster_info<2>(dev).max_clusters ? 2
                                                                    : 1;
    }
    if (cluster == 16) {
        launch_cluster<16>(logits, b, out, k, relative, small, stream);
    } else if (cluster == 8) {
        launch_cluster<8>(logits, b, out, k, relative, small, stream);
    } else if (cluster == 4) {
        launch_cluster<4>(logits, b, out, k, relative, small, stream);
    } else if (cluster == 2) {
        launch_cluster<2>(logits, b, out, k, relative, small, stream);
    } else {
        auto launch = [&](auto kernel, int threads) {
            kernel<<<b.num_rows, threads, 0, stream>>>(logits.data_ptr<float>(), logits.stride(0), logits.stride(1), b,
                                                       out.data_ptr<int32_t>(), out.stride(0), static_cast<int>(k),
                                                       relative);
        };
""",
        """void launch(const bs03::DecodeArgs& a, const RowBounds& b, bool relative) {
    TORCH_CHECK(a.k > 0 && a.k <= 65535, "k out of range");
    // BS03: an explicit map chosen by the caller (by row count only, so one CUDA graph serves every context length).
    TORCH_CHECK(a.cluster == 1 || a.cluster == 2 || a.cluster == 4 || a.cluster == 8, "cluster: 1, 2, 4 or 8");
    if (b.num_rows == 0) return;
    if (a.cluster == 8) {
        launch_cluster<8>(a, b, relative);
    } else if (a.cluster == 4) {
        launch_cluster<4>(a, b, relative);
    } else if (a.cluster == 2) {
        launch_cluster<2>(a, b, relative);
    } else {
        auto launch = [&](auto kernel, int threads) {
            kernel<<<b.num_rows, threads, 0, a.stream>>>(a.logits, a.stride0, a.stride1, b, a.out, a.out_stride0,
                                                         static_cast<int>(a.k), relative);
        };
""",
    ),
    (
        """    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace
""",
        """    C10_CUDA_CHECK(cudaGetLastError());
}

// BS03: the numbers test_det_topk.py prints for one cluster size. Queries only, no kernel launch.
template <int C>
bs03::ProbeResult probe_one(int device) {
    const ClusterInfo info = cluster_info<C>(device);
    return bs03::ProbeResult{C, info.optin, info.static_smem, info.num_regs, info.max_clusters, info.dyn};
}

}  // namespace
""",
    ),
    # 4. entry points: decode only, no pybind in this file
    (
        """// Prefill form (vLLM top_k_per_row_prefill): rows [row_starts[r], row_ends[r]); row_starts may be None.
// cluster: CTAs per row (-1 = by row count, 1 = one CTA); small: rows of at most this many columns use one CTA.
void det_topk_prefill(const torch::Tensor& logits, const c10::optional<torch::Tensor>& row_starts,
                      const torch::Tensor& row_ends, torch::Tensor out, int64_t num_rows, int64_t k, bool relative,
                      int64_t cluster, int64_t small) {
    TORCH_CHECK(row_ends.scalar_type() == torch::kInt32 && row_ends.is_contiguous() && row_ends.numel() >= num_rows);
    if (row_starts.has_value())
        TORCH_CHECK(row_starts->scalar_type() == torch::kInt32 && row_starts->is_contiguous() &&
                    row_starts->numel() >= num_rows);
    RowBounds b{row_starts.has_value() ? row_starts->data_ptr<int32_t>() : nullptr, row_ends.data_ptr<int32_t>(), nullptr,
                0, 1, num_rows, logits.size(1)};
    launch(logits, b, out, k, relative, cluster, small);
}

// Decode form (vLLM SparseIndexerTopk): row lengths from seq_lens / next_n as in SparseIndexerTopk._row_ends.
void det_topk_decode(const torch::Tensor& logits, const torch::Tensor& seq_lens, int64_t next_n, torch::Tensor out,
                     int64_t num_rows, int64_t k, int64_t cluster, int64_t small) {
    TORCH_CHECK(seq_lens.scalar_type() == torch::kInt32 && seq_lens.is_contiguous() && next_n >= 1);
    RowBounds b{nullptr, nullptr, seq_lens.data_ptr<int32_t>(), seq_lens.numel(), next_n, num_rows, logits.size(1)};
    launch(logits, b, out, k, true, cluster, small);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("det_topk_prefill", &det_topk_prefill);
    m.def("det_topk_decode", &det_topk_decode);
}
""",
        """namespace bs03 {

// Decode form (vLLM SparseIndexerTopk): row lengths from seq_lens / next_n as in SparseIndexerTopk._row_ends.
// cluster: CTAs per row (1, 2, 4 or 8); small: rows of at most this many columns use one CTA.
void launch_decode(const DecodeArgs& a) {
    RowBounds b{nullptr, nullptr, a.seq_lens, a.num_seq_lens, a.next_n, a.num_rows, a.width};
    launch(a, b, true);
}

// The dynamic shared memory cap in bytes; once, before the first launch or probe.
void configure(int64_t dyn_cap_bytes) {
    TORCH_CHECK(dyn_cap_bytes >= 16 * 1024 && dyn_cap_bytes <= 256 * 1024, "dyn_cap_bytes out of range");
    if (static_cast<size_t>(dyn_cap_bytes) == g_dyn_cap) return;
    TORCH_CHECK(!g_dyn_cap_used, "configure after the first launch");
    g_dyn_cap = static_cast<size_t>(dyn_cap_bytes);
}

ProbeResult probe(int cluster, int device) {
    if (cluster == 16) return probe_one<16>(device);
    if (cluster == 8) return probe_one<8>(device);
    if (cluster == 4) return probe_one<4>(device);
    TORCH_CHECK(cluster == 2, "cluster: 2, 4, 8 or 16");
    return probe_one<2>(device);
}

}  // namespace bs03
""",
    ),
]


def between(s):
    i = s.index(KERNELS_BEGIN)
    j = s.index(KERNELS_END, i)
    return s[i:j]


def main():
    src, dst = sys.argv[1:3]
    raw = open(src, "rb").read()
    got = hashlib.sha256(raw).hexdigest()
    if got != MOET_SHA256:
        sys.exit("refusing: %s is %s, expected %s" % (src, got, MOET_SHA256))
    orig = raw.decode("ascii")
    s = orig
    for i, (a, b) in enumerate(EDITS):
        n = s.count(a)
        if n != 1:
            sys.exit("edit %d: anchor found %d times:\n%s" % (i, n, a[:200]))
        s = s.replace(a, b)
    # Device code check. The port inserts the cap above KERNELS_END, so compare up to the first line it adds.
    moet_dev = between(orig)
    port_dev = s[s.index(KERNELS_BEGIN) :]
    if not port_dev.startswith(moet_dev):
        sys.exit("device code changed")
    with open(dst, "w") as f:
        f.write(s)
    print("device code unchanged: %d bytes, sha256 %s" % (len(moet_dev), hashlib.sha256(moet_dev.encode()).hexdigest()[:16]))
    print("wrote %s sha256 %s" % (dst, hashlib.sha256(s.encode()).hexdigest()))


main()
