# pd-fc2cd05 (spark-one only): FC2-compact v2 + D05 combination -- NOT BUILT (feasibility only, 2026-10-25)
- src-fc2c-applied/: the 3 M32 source files (deepgemm-m32-hidden-v2/source) with r4-e6 fc2-deepgemm-v2.patch applied (clean).
- d05-source.patch on top: device hunks apply, but 4 host hunks fail (sm120_fp8_fp4_gemm_1d1d.hpp hunks 4/8 + a 3rd, gemm.hpp pybind hunk 4).
  Merge plan: add fc2_live_store to the compact_b2 host path + its pybind def, keep the D05 epilogue hunk, and
  relax the D05 static assert to the compact geometry (kBMainK 256 + tail). Then CPU build ~60 s (Codex build_candidate.py
  recipe), JIT nvcc check, and a payload (sm121_fc2_compact.py) wired via the shared M32 wrapper.
- Memory: the v2 tail tensors are [E*5120, 32] u8 = 63 MB/layer, +2.5 GiB/rank while the padded w2 is kept (1M MemAvailable min ~11 GB).
- Expected: FC2c v2 alone 1.10x FC2 (r4-e6), so about -0.3 ms C1 / -0.9 ms C4 on top of D05.
