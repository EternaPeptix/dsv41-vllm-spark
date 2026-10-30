
## Round 4 / e6 FC2-compact kernel fix (r4-e6-fc2c-kernel-fix): NOT PROMOTED, Step 3 timing gate failed, nothing installed (2026-10-23 ~05:00 UTC)

**Root cause of the r3-e2 715 is confirmed on the GPU** (spark-four, service stopped, probe `~/dsv41-stage/r4/fc2c-probe`).
- A `16U4_ALIGN16B` (.b4x16_p64) TMA traps with `ILLEGAL_INSTRUCTION` when the inner coordinate is not a multiple of 128 elements. It faults at 288, 320, 416, 576, 32 and 64.
- It loads correctly at 0, 128, 256, 384 and 640. The original probe's u4a16 cases hung (rc=124) only because their expect_tx was the smem size. complete_tx counts packed global bytes (2048 for a 128x32 box), and with that value they return OK.
- `16U4_ALIGN8B` (packed smem) accepts 288 and 320. UINT8 TMA needs 16 B (faults at byte 8, OK at 16).
- The r3 slot layout (slots at a k_real stride) therefore cannot work for k_real 288 or 320.

**v2 design** (tail probe `tma_tail_probe.cu`):
- A u4a16 map with inner dim 32 is rejected by `cuTensorMapEncodeTiled` (invalid argument), so the inner dim must be a multiple of 128.
- The u4a16 unpack leaves the pad bytes of each 16 B chunk untouched.
- v2 splits B into two parts:
  - MAIN: `[G, N, 256]` FP4, the standard descriptor at coordinates 0 and 128.
  - TAIL: `[G*N, 32] uint8`, already in the .b4x16_p64 smem image (data in bytes 0-7 and 16-23, pads 0).
- The tail is loaded with a UINT8 TMA (inner dim 32 B, box 128 B x BLOCK_N, 128B swizzle). Bytes 32-127 are filled with zeros as out of bounds, and that fill counts toward complete_tx, so the last stage expects +BLOCK_N*64 B.
- The tail descriptor goes in through the unused `tensor_map_buffer` argument (six descriptors for BLOCK_N 8-256). The JIT name is `..._compact_b2`, the template argument is `kBMainK`, and the API is `m_grouped_fp8_fp4_gemm_nt_contiguous_compact_b2` + `fc2_compact_tail_descs`.
- Native `~/dsv41-stage/fc2c-v2/native-fc2c-v2` exists **on spark-four only**: _C bfb55fdb, manifest 6999ecbe, hidden visibility.
- Patch `r4/r4-e6-fc2c-kernel-fix/fc2-deepgemm-v2.patch` 6129f802; payload `sm121_fc2_compact.py` cb113d58; test `test_fc2_compact_v2.py` 9bde53e3.
- Bytes per expert including scales: 1,044,480 -> 880,640 (-15.7%). FC2 weight bytes are -16.7%, not -25%, because the tail stores 32 B per row for 16 B of data.

**Step 3 offline gates** (spark-four, service stopped; `fc2c-v2/test-20261023T042959Z.*`):

| M (tok) | align | live base us | compact us | speedup |
|---|---|---|---|---|
| 1 | 32 | 17.9 | 18.7 | 0.956 |
| 6 | 32 | 236.1 | 214.3 | 1.102 |
| 32 | 32 | 1002.2 | 904.5 | 1.108 |
| 48 | 32 | 1351.3 | 1224.4 | 1.104 |
| 96 | 32 | 1884.8 | 1721.0 | 1.095 |
| 96 | 128 | 3426.2 | 3192.1 | 1.073 |
| 192 | 128 | 4244.2 | 3926.3 | 1.081 |
| 384 | 128 | 4433.3 | 4135.6 | 1.072 |

- Passed:
  - Bitwise exact vs the native padded kernel and vs the live kernel on every case, with both finite and raw NaN dead scales.
  - Finite outputs.
  - Padded vs fp32 ref rel 1.63-1.68e-3.
  - No new JIT during graph capture (27/27 cache files).
- **Failed:** the timing gate (>= 1.15x at M <= 96). The speedup is about 1.10x, which matches the -16% bytes, and M=1 is 4% slower.

**Decision:** stopped before Step 4. Site-packages, the launcher and the JIT caches are unchanged; launcher 5ccc4b79 on 8/8.
- **Next lever:** get the tail to 16 B per row (for example, 4 rows' tails packed per 64 B with a consumer-side chunk offset, or ALIGN8B packed smem for tile 2), which would give the full -25%.
- **Projected decode gain:** only about +1-2% at the measured 1.10x, so a serving A/B is low priority.

**State after:** relaunched from the Mac (`ssh spark-one cat ...dsv41-launch-all.sh | bash`; running it on spark-one fails because spark-one cannot ssh to spark-six/seven/eight, publickey). Health 200; exact.py a8c3394af9d2; GSM8K-50 conc8 49/50 (miss 12) at 229.9 tok/s.

**Rollback:** none needed.
