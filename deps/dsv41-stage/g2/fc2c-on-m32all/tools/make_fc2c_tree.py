#!/usr/bin/env python3
"""g2 fc2c-on-m32all: rebase the g1 FC2-compact v2 tail (kBMainK) onto the live m32-bw 'all' FC2 tree. CPU only.

The live target FC2 kernel (launcher 395bf5b2, DSV41_MOE_M32_G1=all) is the Codex D05 native 6fea5091 with the
m32-bw 'all' JIT header (FC2: ef + ds + dw; ~/dsv41-stage/g1/m32-bw/trees/all-d05, manifest 073232ef, header
693cd513). The g1 fc2c-d05 native (_C 33d4eb19 = D05 host code + the opt-in dsv41_fc2c_tail_descs path, which
emits one extra template argument kBMainK = 256) carries a 5-hunk header delta on the plain D05 header (2fccad3b
-> 9f780e4b, GPU-qualified bitwise 8/8 on 2026-10-26). This tool:

  header  applies the SAME 5 hunks (text copied verbatim from the qualified header) to the live all-d05 header,
          with two adaptations that 'all' requires:
            * both B loads of the compact path carry the live B load's L2 hint kG1BHint (EVICT_FIRST under 'all';
              a cache policy, not data);
            * the non-compact expect_tx keeps the all-d05 expression; the compact one keeps fc2c's SMEM_TMA_BYTES
              base, which is right because kG1LA is false for FC2 (asserted at compile time),
          plus one compile-time guard (kBMainK > 0 only on the g1 FC2 geometry, without the FC1-only features
          la / pf / pair) and a provenance comment line.
          Self-check (always run): the same hunks applied to the plain D05 header reproduce 9f780e4b byte for byte,
          and the all-d05 input is the pinned 693cd513.
  tree    = copy of the immutable g1 fc2c-d05 native tree (its .so COPIED so dlopen gives it its own state) with
          the merged header; every other file must equal the all-d05 tree's; manifest.json regenerated.

  python3 -I make_fc2c_tree.py check  <d05 hdr> <fc2c hdr> <all-d05 hdr>
  python3 -I make_fc2c_tree.py header <all-d05 hdr> <out hdr>
  python3 -I make_fc2c_tree.py tree   <g1 fc2c-d05 native tree> <m32-bw all-d05 tree> <dst dir>
"""
import difflib
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

HDR = "include/deep_gemm/impls/sm120_fp8_fp4_gemm_1d1d.cuh"
SHA = {
    "d05_hdr": "2fccad3bd4ac36aee6e59c6f2adf2343114b8909f3c8336861133c17e5156ab1",
    "fc2c_hdr": "9f780e4ba313d0e84710a07966db9b6db9beaca5fd04bc250b4aeab71ff7fc41",
    "alld05_hdr": "693cd513f5167631d0ec05092c54b7f0075d44b81fea98c6cb088a4af77f85b0",
    "fc2c_manifest": "3b6f30bcf05bdb8d1255409cfea91dd711479e5102c5cfb1466917ad76c90cb9",
    "fc2c_native": "33d4eb19f3dc7c281440afa6df80724c1f9adae0ed6bc8e57f835b9eea83889a",
    "alld05_manifest": "073232efb28247cc03c83b4198050cd2d9c0d292f5b980819ef7403b57cb9aee",
    "d05_native": "6fea5091715fc33836ec32c9c1aabfe24f3733756d00ffab0af6f326ea479626",
}


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(p) -> str:
    return sha_bytes(Path(p).read_bytes())


# ---- the 5 fc2c hunks (verbatim from 9f780e4b); {H} = B-load cache-hint argument, {BASE} = live expect_tx
TPL_OLD = "          bool kFC2LiveStore = false>\n"
TPL_NEW = "          bool kFC2LiveStore = false,\n          uint32_t kBMainK = 0>\n"

ASSERT_ANCHOR = '        "D05 requires the pinned target FC2 scatter32 geometry");\n\n'
ASSERT_BLOCK = """\
    // g1 fc2c-d05 (FC2-compact v2 tail, kBMainK = SHAPE_K - BLOCK_K = 256): k-blocks below kBMainK load B through
    // the ordinary descriptor over the PADDED weight [G, N, SHAPE_K/2] (identical boxes and bytes to the padded
    // kernel). The last k-block loads B from a TAIL tensor [G*N, 32] uint8 that already holds the .b4x16_p64
    // shared-memory image of k kBMainK..kBMainK+31 (bytes 0-7 / 16-23 = packed FP4, pads 0) through a UINT8 TMA
    // (inner dim 32 B, box 128 B x BLOCK_N, 128B swizzle) whose descriptor lives in global memory at
    // tensor_map_buffer. TMA zero-fills bytes 32..127 of each row (OOB) exactly where the padded weight stores FP4
    // zeros (k >= kBMainK+32), so every data byte the math warps read is unchanged: same instructions, same
    // accumulation order, bitwise-identical output for every row (also rows with stale LIVE_ACT inputs). The OOB
    // fill counts toward complete_tx, so that stage expects BLOCK_N * 128 B of B instead of BLOCK_N * 64 B.
    DG_STATIC_ASSERT(kBMainK == 0 or
        (kGemmType == GemmType::MGroupedContiguous && kBIsFP4 && !kIsFP4 && !kAIsFP4 && kBKMajor &&
         kSplitKFactor == 1 && SHAPE_K != 0 && BLOCK_K == 128 && kSwizzleBMode == 128 &&
         kBMainK % BLOCK_K == 0 && SHAPE_K == kBMainK + BLOCK_K),
        "Invalid FC2-compact tail configuration");

"""

LEADER_ANCHOR = "        if (is_tma_leader) {\n"
LEADER_BLOCK = """\
            if constexpr (kBMainK > 0) {
                // g1 fc2c-d05: the tail descriptor lives in global memory, written by the host (cudaMemcpy).
                // Acquire it for the tensormap proxy in the issuing thread before first use (CUDA guide, TMA
                // tensor maps in global memory: fence.proxy.tensormap::generic.acquire.sys), then prefetch.
                asm volatile ("fence.proxy.tensormap::generic.acquire.sys [%0], 128;"
                              :: "l"(reinterpret_cast<uint64_t>(tensor_map_buffer)) : "memory");
                cute::prefetch_tma_descriptor(tensor_map_buffer);
            }
"""

BLOAD_OLD = """\
                    if constexpr (kBKMajor) {
                        tma::copy<BLOCK_K, BLOCK_N, kTMACopySwizzleB, char, kIsBatchedMM{H}>(tma_b_desc, full_barriers[s], smem_b[s], k_idx, n_idx, 1, batch_idx);
                    } else {
"""
BLOAD_NEW = """\
                    if constexpr (kBMainK > 0) {
                        // g1 fc2c-d05: tail k-block from the UINT8 tail descriptor (row n_idx of [G*N, 32]).
                        if (k_idx < kBMainK) {
                            tma::copy<BLOCK_K, BLOCK_N, kTMACopySwizzleB, char, kIsBatchedMM{H}>(tma_b_desc, full_barriers[s], smem_b[s], k_idx, n_idx, 1, batch_idx);
                        } else {
                            tma::copy<BLOCK_K, BLOCK_N, kTMACopySwizzleB, char, kIsBatchedMM{H}>(
                                tensor_map_buffer, full_barriers[s], smem_b[s], 0, n_idx, 1, batch_idx);
                        }
                    } else if constexpr (kBKMajor) {
                        tma::copy<BLOCK_K, BLOCK_N, kTMACopySwizzleB, char, kIsBatchedMM{H}>(tma_b_desc, full_barriers[s], smem_b[s], k_idx, n_idx, 1, batch_idx);
                    } else {
"""

TX_OLD = "                    full_barriers[s]->arrive_and_expect_tx({BASE});\n"
TX_NEW = """\
                    if constexpr (kBMainK > 0) {
                        full_barriers[s]->arrive_and_expect_tx(SMEM_TMA_BYTES + (k_idx < kBMainK ? 0u : TMA_B_BYTES));
                    } else {
                        full_barriers[s]->arrive_and_expect_tx({BASE});
                    }
"""

# ---- g2-only additions (all tree)
G2_TOP_ANCHOR = "// dsv41-g1 m32-bw variant 'all': fc1=pf+ef+la+ds+dw fc2=ef+ds+dw pf_dist=8 pf_group=4 (make_tree.py)\n"
G2_TOP = ("// dsv41-g2 fc2c-on-m32all: + g1 fc2c-d05 FC2-compact v2 tail (kBMainK, opt-in host path "
          "dsv41_fc2c_tail_descs) (make_fc2c_tree.py)\n")
G2_GUARD_ANCHOR = """\
    DG_STATIC_ASSERT(not kG1LA or (kNumTMAThreads >= 64 and BLOCK_M == 32 and kSMEMKBytes == 128 and
                                   kSwizzleAMode == 128 and SMEM_A == 4096), "g1-la geometry");
"""
G2_GUARD = """\
    // dsv41-g2 fc2c-on-m32all: the FC2-compact tail (kBMainK > 0) is qualified on this tree for the target FC2 only.
    // Its two B loads carry kG1BHint like the live B load (an L2 policy, not data). The FC1-only features would
    // bypass the tail (la: A path / expect_tx base; pf, pair: issue and prefetch through the padded B descriptor).
    DG_STATIC_ASSERT(kBMainK == 0 or (kG1FC2 and not kG1LA and not kG1PF and not kG1Pair),
                     "fc2c tail on a g1 tree: target FC2 geometry without the FC1-only features");
"""

LIVE = dict(H="", BASE="SMEM_TMA_BYTES")
ALL = dict(H=", kG1BHint", BASE="kG1LA ? SMEM_TMA_BYTES - TMA_A_BYTES : SMEM_TMA_BYTES")


def _sub(s: str, old: str, new: str, what: str) -> str:
    n = s.count(old)
    assert n == 1, f"anchor '{what}' found {n} times (expected 1)"
    return s.replace(old, new)


def fc2c_hunks(s: str, flavor: dict) -> str:
    """The 5 g1 fc2c-d05 hunks; flavor LIVE = plain D05 header (must give 9f780e4b), ALL = m32-bw all header."""
    s = _sub(s, TPL_OLD, TPL_NEW, "template kFC2LiveStore")
    s = _sub(s, ASSERT_ANCHOR, ASSERT_ANCHOR + ASSERT_BLOCK, "D05 static assert")
    s = _sub(s, LEADER_ANCHOR, LEADER_ANCHOR + LEADER_BLOCK, "is_tma_leader")
    s = _sub(s, BLOAD_OLD.replace("{H}", flavor["H"]), BLOAD_NEW.replace("{H}", flavor["H"]), "B load")
    s = _sub(s, TX_OLD.replace("{BASE}", flavor["BASE"]), TX_NEW.replace("{BASE}", flavor["BASE"]), "expect_tx")
    return s


def fc2c_on_all(alld05: str) -> str:
    assert sha_bytes(alld05.encode()) == SHA["alld05_hdr"], "all-d05 header is not the pinned 693cd513"
    s = _sub(alld05, G2_TOP_ANCHOR, G2_TOP_ANCHOR + G2_TOP, "g1 variant comment")
    s = _sub(s, G2_GUARD_ANCHOR, G2_GUARD_ANCHOR + G2_GUARD, "g1-la geometry assert")
    return fc2c_hunks(s, ALL)


def self_check(d05: str, fc2c: str, alld05: str) -> dict:
    assert sha_bytes(d05.encode()) == SHA["d05_hdr"], "d05 header"
    assert sha_bytes(fc2c.encode()) == SHA["fc2c_hdr"], "fc2c header"
    again = fc2c_hunks(d05, LIVE)
    assert again == fc2c, "the 5 hunks applied to the D05 header do not reproduce the qualified 9f780e4b"
    merged = fc2c_on_all(alld05)
    # Line diff all-d05 -> merged: exactly the 3 lines the fc2c hunks rewrite in the D05 case are removed (the
    # template closer, the K-major B-load `if`, the expect_tx line, which reappears inside the else branch);
    # everything else is insertion.
    removed = [l[2:] for l in difflib.ndiff(alld05.splitlines(), merged.splitlines()) if l.startswith("- ")]
    want = [TPL_OLD.rstrip("\n"), "                    if constexpr (kBKMajor) {",
            TX_OLD.replace("{BASE}", ALL["BASE"]).rstrip("\n")]
    assert removed == want, removed
    removed_d05 = [l[2:] for l in difflib.ndiff(d05.splitlines(), fc2c.splitlines()) if l.startswith("- ")]
    want_d05 = [TPL_OLD.rstrip("\n"), "                    if constexpr (kBKMajor) {",
                TX_OLD.replace("{BASE}", LIVE["BASE"]).rstrip("\n")]
    assert removed_d05 == want_d05, removed_d05
    added = sum(1 for l in difflib.ndiff(alld05.splitlines(), merged.splitlines()) if l.startswith("+ "))
    added_d05 = sum(1 for l in difflib.ndiff(d05.splitlines(), fc2c.splitlines()) if l.startswith("+ "))
    g2_lines = G2_TOP.count("\n") + G2_GUARD.count("\n")
    assert added == added_d05 + g2_lines, (added, added_d05, g2_lines)
    return dict(d05=SHA["d05_hdr"][:8], fc2c=SHA["fc2c_hdr"][:8], alld05=SHA["alld05_hdr"][:8],
                reproduced_fc2c=True, removed_lines=len(removed), added_lines=added, added_lines_d05_to_fc2c=added_d05,
                merged=sha_bytes(merged.encode()), merged_lines=merged.count("\n"))


def verify_tree(root: Path, manifest_sha: str, native_sha: str) -> dict:
    raw = (root / "manifest.json").read_bytes()
    assert sha_bytes(raw) == manifest_sha, (root, sha_bytes(raw))
    m = json.loads(raw)
    assert m["native_sha256"] == native_sha, root
    for rel, want in m["files"].items():
        p = (root / rel).resolve()
        assert p.is_relative_to(root.resolve()), rel
        assert sha_file(p) == want, (root, rel)
    return m


def build_tree(fc2c_root: Path, alld05_root: Path, dst: Path, tool_sha: str) -> dict:
    assert not dst.exists(), f"{dst} exists (trees are immutable; remove it first)"
    mf = verify_tree(fc2c_root, SHA["fc2c_manifest"], SHA["fc2c_native"])
    ma = verify_tree(alld05_root, SHA["alld05_manifest"], SHA["d05_native"])
    assert set(mf["files"]) == set(ma["files"]), set(mf["files"]) ^ set(ma["files"])
    assert mf["native"] == ma["native"]
    diff = sorted(rel for rel in mf["files"] if mf["files"][rel] != ma["files"][rel])
    assert diff == sorted([HDR, mf["native"]]), diff        # same D05 runtime tree, only header + .so differ
    assert mf["files"][HDR] == SHA["fc2c_hdr"] and ma["files"][HDR] == SHA["alld05_hdr"]
    merged = fc2c_on_all((alld05_root / HDR).read_text())
    shutil.copytree(fc2c_root, dst, symlinks=False)         # real copy: the .so gets its own inode
    for p in dst.rglob("*"):
        if p.is_file():
            p.chmod(0o644)
    (dst / HDR).write_text(merged)
    files = {}
    for p in sorted(dst.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            files[str(p.relative_to(dst))] = sha_file(p)
    assert set(files) == set(mf["files"]), set(files) ^ set(mf["files"])
    man = dict(mf)
    man["files"] = files
    man["status"] = "built_cpu_only_not_GPU_qualified_not_installed"
    man["g2_fc2c_on_m32all"] = {
        "contract": ("FC2-compact v2 tail (g1 fc2c-d05, kBMainK=256 via dsv41_fc2c_tail_descs) on the m32-bw 'all' "
                     "FC2 header (ef+ds+dw); M32 target FC2 only, opt-in DSV41_MOE_FC2C with DSV41_MOE_M32_G1=all"),
        "fc2c_manifest_sha256": SHA["fc2c_manifest"], "alld05_manifest_sha256": SHA["alld05_manifest"],
        "fc2c_header_sha256": SHA["fc2c_hdr"], "alld05_header_sha256": SHA["alld05_hdr"],
        "merged_header_sha256": files[HDR], "make_fc2c_tree_sha256": tool_sha,
    }
    raw = (json.dumps(man, indent=1, sort_keys=True) + "\n").encode()
    (dst / "manifest.json").write_bytes(raw)
    for p in dst.rglob("*"):
        if p.is_file():
            p.chmod(0o555 if p.suffix == ".so" else 0o444)
    out = verify_tree(dst, sha_bytes(raw), SHA["fc2c_native"])
    assert out["files"][HDR] == sha_bytes(merged.encode())
    return dict(tree=str(dst), manifest=sha_bytes(raw), header=files[HDR], native=man["native_sha256"],
                files=len(files))


def main():
    cmd = sys.argv[1]
    if cmd == "check":
        d05, fc2c, alld05 = (Path(x).read_text() for x in sys.argv[2:5])
        print("CHECK", json.dumps(self_check(d05, fc2c, alld05)))
    elif cmd == "header":
        src, dst = sys.argv[2:4]
        Path(dst).write_text(fc2c_on_all(Path(src).read_text()))
        print("HEADER", dst, sha_file(dst))
    elif cmd == "tree":
        fc2c_root, alld05_root, dst = (Path(x) for x in sys.argv[2:5])
        here = Path(__file__).resolve()
        d05_hdr = Path(os.path.expanduser("~/dsv41-goal120-20261024/d05-shadow-v1/native")) / HDR
        print("CHECK", json.dumps(self_check(d05_hdr.read_text(), (fc2c_root / HDR).read_text(),
                                             (alld05_root / HDR).read_text())))
        print("TREE", json.dumps(build_tree(fc2c_root.resolve(strict=True), alld05_root.resolve(strict=True), dst,
                                            sha_file(here))))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
