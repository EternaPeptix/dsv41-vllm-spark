#!/usr/bin/env python3
"""g1/m32-bw: build an M32 native tree variant = an immutable DeepGEMM M32 native tree (baseline b1c4eff8 or D05
6fea5091; the .so is COPIED so dlopen gives it its own state) whose FP4 GEMM JIT header gets EXACT bandwidth
features, compile-time selected for the target routed-MoE geometry only; regenerates manifest.json. CPU only.

  python3 -I make_tree.py <variant> <src native tree> <dst dir>       -> prints TREE <dst> manifest <sha> header <sha>
  python3 -I make_tree.py --header-only <variant> <src header> <dst header>

Geometry gate kG1 (everything else compiles exactly as before): MGroupedContiguous, E384, BM32 BN64 BK128, 48 SMs,
256 math + 128 TMA threads, split-K 1, FP8 A x FP4 B (K-major), gran 128/32, swizzle A/B 128, BF16 out, identity
epilogue, no accumulation, static N/K. FC1 = K 5120 (N 576 compact / 768 padded), FC2 = N 5120 K 384.

Features (per FC1 / FC2):
  pair  FC1: before issuing an even k-block, the TMA leader also waits for the NEXT stage to be empty, so the two
        64-B halves of every 128-B weight line are requested back to back (DRAM row locality). Issue order only.
  pf    L2 prefetch cursor: a private scheduler copy walks the same live tiles kPfDist k-blocks ahead of the loads
        and issues cp.async.bulk.prefetch.tensor for B (+SFB) in groups of kPfGroup consecutive k-blocks, so each
        weight row is fetched as one 256-B burst group instead of 64-B pieces ~1 us apart. Loads unchanged.
  ef    B / SFB TMA loads carry L2::cache_hint EVICT_FIRST (weights are streamed once; keeps A/act lines in L2).
  la    live-row A: A (activations) is no longer TMA-loaded as a 32-row box; producer warp 9 cp.async's only the
        rows whose grouped_layout >= 0 into the identical 128B-swizzled smem slots and arrives on the full barrier
        with cp.async.mbarrier.arrive.noinc (init count 1 + 32). Junk rows keep stale smem -> junk rows only.
  ds    direct live-row stores: the epilogue writes each live row's BF16 pair with one 32-bit st.global (same
        values as the TMA path's smem staging) and never writes dead rows; no smem staging / named barriers.
  dw    Codex D04 dead-walk stop: break (not continue) on the first dead m-block of a FULL 8-block swizzle group
        (48 SMs = 0 mod 8 keeps m mod 8 fixed per CTA; valid m-blocks are a prefix). Same tile set, same order.
Exactness: per output element the MMA sequence (A/B fragments, scales, accumulator chain, K order) is unchanged and
every live row is stored with identical bytes; only dead rows (never read: ep_gather / LIVE_ACT read live rows) and
cache/issue timing differ.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

HDR = "include/deep_gemm/impls/sm120_fp8_fp4_gemm_1d1d.cuh"
PF_DIST, PF_GROUP = 8, 4
FEATS = ("pair", "pf", "ef", "la", "ds", "dw")
VARIANTS = {
    "off":  dict(fc1=(), fc2=()),                                   # patch mechanics only: SASS must equal live
    "pair": dict(fc1=("pair",), fc2=()),
    "pf":   dict(fc1=("pf", "ef"), fc2=()),
    "la":   dict(fc1=("la",), fc2=()),
    "ds":   dict(fc1=("ds",), fc2=("ds",)),
    "all":  dict(fc1=("pf", "ef", "la", "ds", "dw"), fc2=("ef", "ds", "dw")),
}


def patch(s: str, variant: str) -> str:
    v = VARIANTS[variant]

    def sub(old, new, count=1):
        nonlocal s
        assert s.count(old) == count, (variant, old[:90], s.count(old))
        s = s.replace(old, new)

    def flag(f):
        return "(kG1FC1 and {}) or (kG1FC2 and {})".format(
            "true" if f in v["fc1"] else "false", "true" if f in v["fc2"] else "false")

    sub("#pragma once\n", "#pragma once\n// dsv41-g1 m32-bw variant '%s': fc1=%s fc2=%s pf_dist=%d pf_group=%d (make_tree.py)\n"
        % (variant, "+".join(v["fc1"]) or "-", "+".join(v["fc2"]) or "-", PF_DIST, PF_GROUP))

    sub("""    static constexpr uint32_t kTMACopySwizzleB = kIsFP4 ? 0u : kSwizzleBMode;
""", """    static constexpr uint32_t kTMACopySwizzleB = kIsFP4 ? 0u : kSwizzleBMode;

    // dsv41-g1 m32-bw: exact bandwidth features, target M32 routed-MoE geometry only (see make_tree.py).
    static constexpr bool kG1 = (kGemmType == GemmType::MGroupedContiguous) and kNumGroups == 384 and
        BLOCK_M == 32 and BLOCK_N == 64 and BLOCK_K == 128 and kNumSMs == 48 and kNumMathThreads == 256 and
        kNumTMAThreads == 128 and kSplitKFactor == 1 and kBIsFP4 and (not kIsFP4) and (not kAIsFP4) and kBKMajor and
        (not kWithAccumulation) and kGranKA == 128 and kGranKB == 32 and kSwizzleAMode == 128 and
        kSwizzleBMode == 128 and kEpiSubM == 32 and SHAPE_N != 0 and SHAPE_K != 0 and SHAPE_N %% BLOCK_N == 0 and
        SHAPE_K %% BLOCK_K == 0 and cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t> and
        cute::is_same_v<epilogue_type_t, epilogue::transform::EpilogueIdentity>;
    static constexpr bool kG1FC1 = kG1 and SHAPE_K == 5120;
    static constexpr bool kG1FC2 = kG1 and SHAPE_N == 5120 and SHAPE_K == 384;
    static constexpr bool kG1Pair = kG1FC1 and (%(pair)s);
    static constexpr bool kG1PF = %(pf)s;
    static constexpr bool kG1EF = %(ef)s;
    static constexpr bool kG1LA = %(la)s;
    static constexpr bool kG1DS = (%(ds)s) and kUseTMAStoreEpilogue;
    static constexpr bool kG1DW = (%(dw)s) and
        sched_sm120::get_num_1d_blocks_per_group<kGemmType, BLOCK_M, BLOCK_N, kNumSMs, false>() == 8;
    static constexpr uint32_t kG1PfDist = %(dist)d, kG1PfGroup = %(group)d;
    static constexpr uint64_t kG1BHint = kG1EF ? static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_FIRST)
                                               : static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL);
    DG_STATIC_ASSERT(not kG1LA or (kNumTMAThreads >= 64 and BLOCK_M == 32 and kSMEMKBytes == 128 and
                                   kSwizzleAMode == 128 and SMEM_A == 4096), "g1-la geometry");
""" % dict(pair=flag("pair"), pf=flag("pf"), ef=flag("ef"), la=flag("la"), ds=flag("ds"), dw=flag("dw"),
           dist=PF_DIST, group=PF_GROUP))

    sub("""            full_barriers[i]->init(1);
""", """            full_barriers[i]->init(kG1LA ? 1 + 32 : 1);  // g1-la: + 32 cp.async noinc arrivals (warp 9)
""")

    # producer: prefetch cursor state
    sub("""        if (is_tma_leader) {
            uint32_t last_group_idx = kNumGroups;
            while (scheduler.get_next_block(m_block_idx, n_block_idx)) {""", """        if (is_tma_leader) {
            uint32_t last_group_idx = kNumGroups;
            // g1-pf: prefetch cursor = private scheduler copy over the same live tiles, kG1PfDist k-blocks ahead
            [[maybe_unused]] auto pf_sched = scheduler;
            [[maybe_unused]] uint32_t pf_m = 0, pf_n = 0, pf_kb = 0, pf_nkb = 0, pf_total = 0, ld_total = 0;
            [[maybe_unused]] bool pf_done = false;
            while (scheduler.get_next_block(m_block_idx, n_block_idx)) {""")

    dead_prod = """                if constexpr (kGemmType == GemmType::MGroupedContiguous) {
                    if (__ldg(grouped_layout + m_block_idx * BLOCK_M) < 0)
                        continue;
                }
                if constexpr (is_k_grouped_contiguous(kGemmType)) {"""
    sub(dead_prod, """                if constexpr (kGemmType == GemmType::MGroupedContiguous) {
                    if (__ldg(grouped_layout + m_block_idx * BLOCK_M) < 0) {
                        if constexpr (kG1DW) {  // g1-dw (Codex D04): full 8-block group => every later m is dead
                            if (scheduler.num_blocks_in_group == 8)
                                break;
                        }
                        continue;
                    }
                }
                if constexpr (is_k_grouped_contiguous(kGemmType)) {""")

    sub("""                for (uint32_t kb = kb_start; kb < kb_end; ++kb) {
                    CUTE_TIE_DECL(get_pipeline(tma_iter_idx++), s, p);
                    empty_barriers[s]->wait(p ^ 1);
""", """                for (uint32_t kb = kb_start; kb < kb_end; ++kb) {
                    CUTE_TIE_DECL(get_pipeline(tma_iter_idx++), s, p);
                    empty_barriers[s]->wait(p ^ 1);
                    if constexpr (kG1Pair) {
                        // g1-pair: also wait for the next stage, then issue both k-blocks back to back
                        if ((kb & 1u) == 0 and kb + 1 < kb_end) {
                            const uint32_t s1 = tma_iter_idx % kNumStages, p1 = (tma_iter_idx / kNumStages) & 1;
                            empty_barriers[s1]->wait(p1 ^ 1);
                        }
                    }
                    if constexpr (kG1PF) {
                        // g1-pf: keep the prefetch cursor > kG1PfDist k-blocks ahead; whole groups per issue
                        while (not pf_done and pf_total <= ld_total + kG1PfDist) {
                            if (pf_kb == pf_nkb) {
                                uint32_t pm, pn;
                                bool found = false;
                                while (pf_sched.get_next_block(pm, pn)) {
                                    if (__ldg(grouped_layout + pm * BLOCK_M) < 0) {
                                        if constexpr (kG1DW) {
                                            if (pf_sched.num_blocks_in_group == 8)
                                                break;
                                        }
                                        continue;
                                    }
                                    found = true;
                                    break;
                                }
                                if (not found) {
                                    pf_done = true;
                                    break;
                                }
                                pf_m = pm, pf_n = pn, pf_kb = 0, pf_nkb = SHAPE_K / BLOCK_K;
                            }
                            const uint32_t pf_n_idx = pf_sched.template get_global_idx<true>(shape_n, BLOCK_N, pf_n, pf_m);
                            const uint32_t pf_sfb_rows = math::ceil_div(shape_k, BLOCK_K * kNumSFBStagesPerLoad);
                            const uint32_t pf_end = (pf_kb + kG1PfGroup < pf_nkb) ? pf_kb + kG1PfGroup : pf_nkb;
                            for (; pf_kb < pf_end; ++pf_kb, ++pf_total) {
                                cute::SM90_TMA_LOAD_2D::PREFETCH::copy(&tensor_map_b_base,
                                    static_cast<int32_t>(pf_kb * BLOCK_K), static_cast<int32_t>(pf_n_idx));
                                const uint32_t pf_sfb_k = pf_sched.template get_global_idx<true, sched_sm120::IndexType::SF_K>(
                                    pf_sfb_rows, 1, pf_kb / kNumSFBStagesPerLoad, pf_m);
                                cute::SM90_TMA_LOAD_2D::PREFETCH::copy(&tensor_map_sfb,
                                    static_cast<int32_t>(pf_n * BLOCK_N), static_cast<int32_t>(pf_sfb_k));
                            }
                        }
                        ++ld_total;
                    }
""")

    sub("""                    tma::copy<BLOCK_M, BLOCK_K, 0>(&tensor_map_sfa, full_barriers[s], smem_sfa[s], m_block_idx * BLOCK_M, sfa_k, 1);
                    tma::copy<BLOCK_N, BLOCK_K, 0>(&tensor_map_sfb, full_barriers[s], smem_sfb[s], n_block_idx * BLOCK_N, sfb_k, 1);
                    tma::copy<BLOCK_K, BLOCK_M, kTMACopySwizzleA, char, kIsBatchedMM>(tma_a_desc, full_barriers[s], smem_a[s], k_idx, m_idx, 1, batch_idx);
                    if constexpr (kBKMajor) {
                        tma::copy<BLOCK_K, BLOCK_N, kTMACopySwizzleB, char, kIsBatchedMM>(tma_b_desc, full_barriers[s], smem_b[s], k_idx, n_idx, 1, batch_idx);
                    } else {""", """                    tma::copy<BLOCK_M, BLOCK_K, 0>(&tensor_map_sfa, full_barriers[s], smem_sfa[s], m_block_idx * BLOCK_M, sfa_k, 1);
                    tma::copy<BLOCK_N, BLOCK_K, 0, char, false, kG1BHint>(&tensor_map_sfb, full_barriers[s], smem_sfb[s], n_block_idx * BLOCK_N, sfb_k, 1);
                    if constexpr (not kG1LA)
                        tma::copy<BLOCK_K, BLOCK_M, kTMACopySwizzleA, char, kIsBatchedMM>(tma_a_desc, full_barriers[s], smem_a[s], k_idx, m_idx, 1, batch_idx);
                    if constexpr (kBKMajor) {
                        tma::copy<BLOCK_K, BLOCK_N, kTMACopySwizzleB, char, kIsBatchedMM, kG1BHint>(tma_b_desc, full_barriers[s], smem_b[s], k_idx, n_idx, 1, batch_idx);
                    } else {""")

    sub("""                    full_barriers[s]->arrive_and_expect_tx(SMEM_TMA_BYTES);
                }
            }
        }
    }
    // CONSUMER WARP GROUPS (math warps, 232 regs)""", """                    full_barriers[s]->arrive_and_expect_tx(kG1LA ? SMEM_TMA_BYTES - TMA_A_BYTES : SMEM_TMA_BYTES);
                }
            }
        }
        if constexpr (kG1LA) {
            // g1-la: warp kNumMathWarps + 1 walks the identical tile/stage sequence and cp.async's only the live A
            // rows (grouped_layout >= 0) into the same 128B-swizzled slots TMA would fill (chunk c of row r at
            // r * 128 + ((c ^ (r & 7)) << 4)); each lane then arrives once per stage (noinc) when its copies land.
            if (warp_idx == kNumMathWarps + 1) {
                uint32_t la_iter_idx = 0;
                while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
                    if (__ldg(grouped_layout + m_block_idx * BLOCK_M) < 0) {
                        if constexpr (kG1DW) {
                            if (scheduler.num_blocks_in_group == 8)
                                break;
                        }
                        continue;
                    }
                    const uint32_t live_mask = __ballot_sync(0xffffffffu,
                        __ldg(grouped_layout + m_block_idx * BLOCK_M + lane_idx) >= 0);
                    const auto a_tile = reinterpret_cast<const uint8_t*>(gmem_a_ptr) +
                        static_cast<uint64_t>(m_block_idx) * BLOCK_M * SHAPE_K + (lane_idx & 7u) * 16u;
                    for (uint32_t kb = 0; kb < SHAPE_K / BLOCK_K; ++kb) {
                        CUTE_TIE_DECL(get_pipeline(la_iter_idx++), s, p);
                        empty_barriers[s]->wait(p ^ 1);
                        const uint32_t dst_base = cute::cast_smem_ptr_to_uint(smem_a[s]);
                        #pragma unroll
                        for (uint32_t j = 0; j < 8; ++j) {
                            const uint32_t r = (lane_idx >> 3) + 4u * j;
                            if ((live_mask >> r) & 1u) {
                                const uint32_t dst = dst_base + r * 128u + ((((lane_idx & 7u) ^ (r & 7u))) << 4);
                                const uint8_t* src = a_tile + static_cast<uint64_t>(r) * SHAPE_K + kb * BLOCK_K;
                                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\\n" :: "r"(dst), "l"(src) : "memory");
                            }
                        }
                        asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\\n"
                                     :: "r"(cute::cast_smem_ptr_to_uint(full_barriers[s])) : "memory");
                    }
                }
            }
        }
    }
    // CONSUMER WARP GROUPS (math warps, 232 regs)""")

    sub("""            if constexpr (kGemmType == GemmType::MGroupedContiguous) {
                if (__ldg(grouped_layout + m_block_idx * BLOCK_M) < 0)
                    continue;
            }
            const uint32_t current_shape_k""", """            if constexpr (kGemmType == GemmType::MGroupedContiguous) {
                if (__ldg(grouped_layout + m_block_idx * BLOCK_M) < 0) {
                    if constexpr (kG1DW) {  // g1-dw (Codex D04), identical rule in every walker
                        if (scheduler.num_blocks_in_group == 8)
                            break;
                    }
                    continue;
                }
            }
            const uint32_t current_shape_k""")

    sub("""            if constexpr (kUseTMAStoreEpilogue) {
                #pragma unroll
                for (uint32_t ms = 0; ms < kNumEpiMSubs; ++ms) {""", """            if constexpr (kG1DS) {
                // g1-ds: each live row's BF16 pair with one 32-bit store (bytes identical to the TMA path's smem
                // staging: cd_dtype_t(v) of the same accumulators); dead rows are never written.
                #pragma unroll
                for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                    const uint32_t row0 = m_base + (m_tile_base + mt) * MMA_M + group_id;
                    const uint32_t row1 = row0 + 8;
                    const bool live0 = row0 < total_shape_m and __ldg(grouped_layout + row0) >= 0;
                    const bool live1 = row1 < total_shape_m and __ldg(grouped_layout + row1) >= 0;
                    #pragma unroll
                    for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                        const uint32_t ai = (mt * kNTilesPerWarp + nt) * MMA_ACCUM;
                        const uint32_t col = n_base + (n_tile_base + nt) * MMA_N + thread_id * 2;
                        if (live0) {
                            cd_dtype_t p0[2] = {cd_dtype_t(accum[ai + 0]), cd_dtype_t(accum[ai + 1])};
                            *reinterpret_cast<uint32_t*>(gmem_d + static_cast<int64_t>(row0) * cd_m_stride + col) =
                                *reinterpret_cast<const uint32_t*>(p0);
                        }
                        if (live1) {
                            cd_dtype_t p1[2] = {cd_dtype_t(accum[ai + 2]), cd_dtype_t(accum[ai + 3])};
                            *reinterpret_cast<uint32_t*>(gmem_d + static_cast<int64_t>(row1) * cd_m_stride + col) =
                                *reinterpret_cast<const uint32_t*>(p1);
                        }
                    }
                }
            } else if constexpr (kUseTMAStoreEpilogue) {
                #pragma unroll
                for (uint32_t ms = 0; ms < kNumEpiMSubs; ++ms) {""")
    return s


def main():
    if sys.argv[1] == "--header-only":
        variant, src, dst = sys.argv[2:5]
        Path(dst).write_text(patch(Path(src).read_text(), variant))
        print("HEADER", dst, hashlib.sha256(Path(dst).read_bytes()).hexdigest())
        return
    variant, src, dst = sys.argv[1], Path(sys.argv[2]).resolve(strict=True), Path(sys.argv[3])
    assert variant in VARIANTS, variant
    assert not dst.exists(), f"{dst} exists (trees are immutable; remove it first)"
    shutil.copytree(src, dst, symlinks=False)          # real copy: the .so gets its own inode (own dlopen state)
    h = dst / HDR
    h.write_text(patch(h.read_text(), variant))
    m = json.loads((dst / "manifest.json").read_text())
    m["files"][HDR] = hashlib.sha256(h.read_bytes()).hexdigest()
    m["g1_m32bw"] = {"variant": variant, "features": VARIANTS[variant], "pf_dist": PF_DIST, "pf_group": PF_GROUP,
                     "base_manifest_sha256": hashlib.sha256((src / "manifest.json").read_bytes()).hexdigest()}
    raw = (json.dumps(m, indent=2) + "\n").encode()
    (dst / "manifest.json").write_bytes(raw)
    for rel, want in m["files"].items():
        assert hashlib.sha256((dst / rel).read_bytes()).hexdigest() == want, rel
    print("TREE", dst, "manifest", hashlib.sha256(raw).hexdigest(), "header", m["files"][HDR],
          "native", m["native_sha256"])


if __name__ == "__main__":
    main()
