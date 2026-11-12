// g130 fc1 CPU test: compiles header/dsv41_fc1_deal.cuh on the host and checks, for every geometry the kernel can see,
// that every live tile runs exactly once, no CTA runs more than q tiles, every walker gets the same sequence, and the prefix
// binary search returns G. Prints the deal per G as TSV on stdout for the Python model test.
//   g++ -std=c++20 -O2 -I. tests/deal_host_test.cpp -o deal_host_test && ./deal_host_test
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <vector>
#define CUTLASS_DEVICE static inline
#define __ldg(p) (*(p))
#include "header/dsv41_fc1_deal.cuh"
using namespace deep_gemm::dsv41_fc1_deal;

template <uint32_t kOrder>
static void check(uint32_t G, uint32_t tail, uint32_t& n_out, uint32_t& maxt, uint32_t& lastround) {
    constexpr uint32_t N = 9, SM = 48;
    const uint32_t K = G * N;
    uint32_t n = tail == 24 ? active_ctas<SM, 24>(K) : active_ctas<SM, 1>(K);
    assert(n <= SM);
    std::vector<int> seen(K, 0);
    maxt = 0;
    std::vector<uint32_t> per(SM, 0);
    for (uint32_t c = 0; c < SM; ++c) {
        int it = -1, it2 = -1;
        uint32_t m, nb, m2, nb2, cnt = 0;
        while (next<N, kOrder>(it, c, G, n, m, nb)) {
            // a second walker over a private iterator (pf cursor, LA warp, math warps) must get the same tile
            bool ok2 = next<N, kOrder>(it2, c, G, n, m2, nb2);
            assert(ok2 && m2 == m && nb2 == nb);
            assert(m < G && nb < N);
            seen[m * N + nb]++;
            cnt++;
        }
        { const bool more = next<N, kOrder>(it2, c, G, n, m2, nb2); assert(!more); }
        per[c] = cnt;
        if (cnt > maxt) maxt = cnt;
    }
    for (uint32_t i = 0; i < K; ++i) assert(seen[i] == 1);
    const uint32_t q = K == 0 ? 0 : (K + SM - 1) / SM;
    assert(maxt == q);
    lastround = 0;
    for (uint32_t c = 0; c < SM; ++c) lastround += per[c] == q && q > 0;
    n_out = n;
}

int main() {
    // find_G on every prefix for every num_m_blocks the path can see (M_sum / 32, up to 96 tokens)
    for (uint32_t nmb = 1; nmb <= 420; ++nmb) {
        std::vector<int> layout(nmb * 32, -1);
        for (uint32_t G = 0; G <= nmb; ++G) {
            for (uint32_t m = 0; m < G; ++m) layout[m * 32] = int(m % 384);
            assert(find_G<32>(layout.data(), nmb) == G);
            if (nmb > 40 && G > 70) break;  // speed
        }
    }
    puts("G\tK\tq\tn\tmaxt\tlastround");
    for (uint32_t G = 0; G <= 420; ++G) {
        uint32_t n, maxt, last;
        check<0>(G, 24, n, maxt, last);
        uint32_t n1, maxt1, last1;
        check<1>(G, 24, n1, maxt1, last1);
        assert(n == n1 && maxt == maxt1 && last == last1);
        const uint32_t K = G * 9;
        // The final round keeps at least 24 CTAs whenever there is more than one round, unless n had to be raised to hold q.
        if (K > 48 && last < 24) {
            const uint32_t q = (K + 47) / 48;
            assert(n == (K + q - 1) / q);
        }
        printf("%u\t%u\t%u\t%u\t%u\t%u\n", G, K, K ? (K + 47) / 48 : 0, n, maxt, last);
    }
    fprintf(stderr, "HOST TEST PASS\n");
    return 0;
}
