
#include <deep_gemm/impls/sm120_fp8_fp4_gemm_1d1d.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {
    auto ptr = reinterpret_cast<void*>(&sm120_fp8_fp4_gemm_1d1d_impl<
        0, 5120, 384,
        128, 32,
        384,
        32, 64, 128,
        128, 128,
        128,
        7,
        128, 256,
        48,
        GemmType::MGroupedContiguous, false,
        cutlass::bfloat16_t,
        epilogue::transform::EpilogueIdentity,
        false,
        true,
        false,
        true,
        false,
        32,
        1,
        128,
        true,
        256
    >);
};
