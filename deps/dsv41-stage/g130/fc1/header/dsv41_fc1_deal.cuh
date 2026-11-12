#pragma once
// dsv41 g130 fc1: G-aware tile deal for the target routed-MoE FC1 (make_deal_tree.py; compiled in only on deal trees).
//
// Live walk (sm120_gemm.cuh get_swizzled_block_idx): 48 CTAs, swizzle group 8, so each CTA is pinned to one m residue.
// Tiles pile up on a few CTAs, whose tails run at the 396 ns per-stage floor once fewer than about 21 CTAs are active.
// Deal: K = G * N live tiles, in order, one every n CTAs: CTA c runs tiles l = c, c + n, c + 2n, ... below K.
//   q = ceil(K / 48) is the most tiles any CTA runs. The largest n that still leaves at least kTail CTAs in the final
//   round is used, so the early rounds keep as many CTAs as possible and the last one stays above the floor.
// Same tile set, each tile exactly once, same per-tile code. Only which CTA runs a tile changes.
//
// Contract used (already relied on by the live D04 dead-walk stop): grouped_layout[m * BLOCK_M] >= 0 iff m < G.
// ep_scatter_1 and the mfront plan kernel lay out the live experts' 32-aligned segments contiguously from row 0, and the
// rest of the layout stays -1. Read only after cudaGridDependencySynchronize; the layout does not change during the call.
//
// Pure functions, no state, so tests/deal_host_test.cpp compiles this same file on the host.

namespace deep_gemm::dsv41_fc1_deal {

// G = number of live m-blocks. Every lane of a warp reads the same words, so the branches are uniform.
template <uint32_t BLOCK_M>
CUTLASS_DEVICE uint32_t find_G(const int* grouped_layout, const uint32_t num_m_blocks) {
    uint32_t lo = 0, hi = num_m_blocks;
    while (lo < hi) {
        const uint32_t mid = (lo + hi) >> 1;
        if (__ldg(grouped_layout + mid * BLOCK_M) >= 0)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

// Number of CTAs that run tiles for K live tiles. 0 for K = 0.
template <uint32_t kNumSMs, uint32_t kTail>
CUTLASS_DEVICE uint32_t active_ctas(const uint32_t K) {
    if (K <= kNumSMs)
        return K;
    const uint32_t q = (K + kNumSMs - 1) / kNumSMs;         // >= 2
    uint32_t n = (K - kTail) / (q - 1);                     // final round holds K - (q - 1) * n >= kTail CTAs
    if (n > kNumSMs)
        n = kNumSMs;
    const uint32_t lo = (K + q - 1) / q;                    // n * q >= K: no CTA runs more than q tiles
    return n < lo ? lo : n;
}

// Next tile of CTA c. kOrder 0: all N tiles of an m-block on adjacent CTAs; 1: one N tile across all m-blocks first.
template <uint32_t kNumN, uint32_t kOrder>
CUTLASS_DEVICE bool next(int& iter, const uint32_t c, const uint32_t G, const uint32_t n, uint32_t& m, uint32_t& nb) {
    if (c >= n)
        return false;
    const uint32_t l = static_cast<uint32_t>(++iter) * n + c;
    if (l >= G * kNumN)
        return false;
    if constexpr (kOrder == 0) {
        m = l / kNumN;
        nb = l - m * kNumN;
    } else {
        nb = l / G;
        m = l - nb * G;
    }
    return true;
}

}  // namespace deep_gemm::dsv41_fc1_deal
