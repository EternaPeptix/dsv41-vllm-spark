#!/usr/bin/env python3
"""g130 fc1: build a variant of the m32-bw 'all' base tree (the tree that serves FC1) from anchored header patches.

  python3 -I make_deal_tree.py <arm> <src all-base tree> <dst dir> <dsv41_fc1_deal.cuh>
  python3 -I make_deal_tree.py --header-only <arm> <src header> <dst header>
  python3 -I make_deal_tree.py --list

Arms (CPU only, never touches the source tree; the .so is COPIED so dlopen gives it its own state):
  deal    FC1 G-aware deal, kOrder 0 (all N tiles of an m-block on adjacent CTAs), kTail 24     serving candidate
  dealx   FC1 G-aware deal, kOrder 1 (one N tile across all m-blocks first), kTail 24           serving candidate
  nomma   live walk, QMMAs removed from FC1                                TIMING ONLY: output is WRONG, never served
The geometry gate is kG1FC1 and SHAPE_N == 576, so FC2 and every E128 drafter instantiation keep their code.
"""
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

HDR = "include/deep_gemm/impls/sm120_fp8_fp4_gemm_1d1d.cuh"
SIDE = "include/deep_gemm/impls/dsv41_fc1_deal.cuh"
BASE_HDR_SHA = "faafb3f673f0a8860f2f5b28430182aa85365151d859a832604e0fe0cb03f43e"
BASE_MANIFEST_SHA = "74d0d4f2a09247a668f876d4d5e0993afa9e93462f4d7f6da70276738a218f5c"
ARMS = {
    "deal": dict(deal=True, order=0, tail=24, nomma=False, serve=True),
    "dealx": dict(deal=True, order=1, tail=24, nomma=False, serve=True),
    "nomma": dict(deal=False, order=0, tail=24, nomma=True, serve=False),
}


def patch(s: str, arm: str) -> str:
    v = ARMS[arm]

    def sub(old, new, count=1):
        nonlocal s
        assert s.count(old) == count, (arm, old[:90], s.count(old))
        s = s.replace(old, new)

    sub("#pragma once\n", "#pragma once\n// dsv41 g130 fc1 arm '%s': %s (make_deal_tree.py)\n" % (arm, json.dumps(v)))
    sub("    static constexpr uint32_t kG1PfDist = 8, kG1PfGroup = 4;\n",
        "    static constexpr uint32_t kG1PfDist = 8, kG1PfGroup = 4;\n"
        "    static constexpr bool kG1NoMMA = kG1FC1 and SHAPE_N == 576 and %s;\n"
        % ("true" if v["nomma"] else "false"))

    if v["nomma"]:
        n0 = len(re.findall(r"sm120_mma::fp8_fp4_mixed_mma_block_scaled\(d, a_frag\[buf\]\[mt\], b_tile\[buf\]\[nt\], sfa, sfb\);", s))
        assert n0 == 3, n0
        s = s.replace("sm120_mma::fp8_fp4_mixed_mma_block_scaled(d, a_frag[buf][mt], b_tile[buf][nt], sfa, sfb);",
                      "{ if constexpr (not kG1NoMMA) "
                      "sm120_mma::fp8_fp4_mixed_mma_block_scaled(d, a_frag[buf][mt], b_tile[buf][nt], sfa, sfb); }")

    if not v["deal"]:
        return s

    sub("#include <deep_gemm/scheduler/sm120_gemm.cuh>\n",
        "#include <deep_gemm/scheduler/sm120_gemm.cuh>\n#include <deep_gemm/impls/dsv41_fc1_deal.cuh>\n")

    sub("""    const auto get_pipeline = [=](const uint32_t& iter_idx) -> cute::tuple<uint32_t, uint32_t> {
""", """    // dsv41 g130 fc1: G-aware tile deal, FC1 only. All walkers (TMA leader, pf cursor, LA warp, math warps) take the
    // same G and n from the same layout words, so they issue the same tile sequence and the barriers stay in sync.
    static constexpr bool kG1Deal = kG1FC1 and SHAPE_N == 576;
    static constexpr uint32_t kG1DealN = SHAPE_N / BLOCK_N, kG1DealOrder = %(order)d, kG1DealTail = %(tail)d;
    [[maybe_unused]] uint32_t g1_deal_G = 0, g1_deal_n = 0;
    [[maybe_unused]] auto g1_deal_init = [&]() {
        if constexpr (kG1Deal) {
            g1_deal_G = dsv41_fc1_deal::find_G<BLOCK_M>(grouped_layout, scheduler.num_m_blocks);
            g1_deal_n = dsv41_fc1_deal::active_ctas<kNumSMs, kG1DealTail>(g1_deal_G * kG1DealN);
        }
    };
    auto g1_next_block = [&](auto& s, uint32_t& m, uint32_t& n) -> bool {
        if constexpr (kG1Deal) {
            s.num_blocks_in_group = 8;  // only read if a tile's m-block is dead, which the contract rules out
            return dsv41_fc1_deal::next<kG1DealN, kG1DealOrder>(s.current_iter, blockIdx.x, g1_deal_G, g1_deal_n, m, n);
        } else {
            return s.get_next_block(m, n);
        }
    };
    const auto get_pipeline = [=](const uint32_t& iter_idx) -> cute::tuple<uint32_t, uint32_t> {
""" % dict(order=v["order"], tail=v["tail"]))

    # TMA leader
    sub("""            uint32_t last_group_idx = kNumGroups;
            // g1-pf: prefetch cursor = private scheduler copy over the same live tiles, kG1PfDist k-blocks ahead
""", """            uint32_t last_group_idx = kNumGroups;
            g1_deal_init();
            // g1-pf: prefetch cursor = private scheduler copy over the same live tiles, kG1PfDist k-blocks ahead
""")
    sub("""            [[maybe_unused]] bool pf_done = false;
            while (scheduler.get_next_block(m_block_idx, n_block_idx)) {""",
        """            [[maybe_unused]] bool pf_done = false;
            while (g1_next_block(scheduler, m_block_idx, n_block_idx)) {""")
    # pf cursor
    sub("while (pf_sched.get_next_block(pm, pn)) {", "while (g1_next_block(pf_sched, pm, pn)) {")
    # LA warp
    sub("""                uint32_t la_iter_idx = 0;
                while (scheduler.get_next_block(m_block_idx, n_block_idx)) {""",
        """                uint32_t la_iter_idx = 0;
                g1_deal_init();
                while (g1_next_block(scheduler, m_block_idx, n_block_idx)) {""")
    # math warps
    sub("""        uint32_t iter_idx = 0;

        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {""",
        """        uint32_t iter_idx = 0;

        g1_deal_init();
        while (g1_next_block(scheduler, m_block_idx, n_block_idx)) {""")
    assert "scheduler.get_next_block" not in s and "pf_sched.get_next_block" not in s
    return s


def main():
    if sys.argv[1] == "--list":
        print(json.dumps(ARMS, indent=1))
        return
    if sys.argv[1] == "--header-only":
        arm, src, dst = sys.argv[2:5]
        Path(dst).write_text(patch(Path(src).read_text(), arm))
        print("HEADER", dst, hashlib.sha256(Path(dst).read_bytes()).hexdigest())
        return
    arm, src, dst, side = sys.argv[1], Path(sys.argv[2]).resolve(strict=True), Path(sys.argv[3]), Path(sys.argv[4])
    assert arm in ARMS, arm
    assert not dst.exists(), f"{dst} exists (trees are immutable; remove it first)"
    raw0 = (src / "manifest.json").read_bytes()
    assert hashlib.sha256(raw0).hexdigest() == BASE_MANIFEST_SHA, "source is not the pinned all-base tree"
    assert hashlib.sha256((src / HDR).read_bytes()).hexdigest() == BASE_HDR_SHA, "source header moved"
    shutil.copytree(src, dst, symlinks=False)
    h = dst / HDR
    h.write_text(patch(h.read_text(), arm))
    m = json.loads(raw0)
    m["files"][HDR] = hashlib.sha256(h.read_bytes()).hexdigest()
    if ARMS[arm]["deal"]:
        (dst / SIDE).write_bytes(side.read_bytes())
        m["files"][SIDE] = hashlib.sha256((dst / SIDE).read_bytes()).hexdigest()
    m["g130_fc1"] = dict(arm=arm, cfg=ARMS[arm], base_manifest_sha256=BASE_MANIFEST_SHA)
    raw = (json.dumps(m, indent=2) + "\n").encode()
    (dst / "manifest.json").write_bytes(raw)
    for rel, want in m["files"].items():
        assert hashlib.sha256((dst / rel).read_bytes()).hexdigest() == want, rel
    print("TREE", arm, "manifest", hashlib.sha256(raw).hexdigest(), "header", m["files"][HDR], "native", m["native_sha256"])


if __name__ == "__main__":
    main()
