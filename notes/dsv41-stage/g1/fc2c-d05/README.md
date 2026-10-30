# g1 / fc2c-d05: FC2-compact v2 tail on top of the live D05 kernel (offline prep, 2026-10-26)

Staged only. Nothing is installed, no restart, no GPU job, no live site-packages or launcher edit. The native was
built CPU-only on spark-four (nice 19); the tree is identical on 8/8 (`MANIFEST.sha256`).

## What

The target FC2 grouped GEMM (E384, N5120, K384 with 96 zero-padded K columns, BF16 out, M32 path) is the live D05
kernel (dead 8-row output groups not stored). This candidate keeps the D05 kernel and changes only where the B
operand of the LAST k-block comes from:

- k-blocks 0 and 1 (k 0..255): the ordinary TMA descriptor over the **padded** w2 `[384, 5120, 192]` (unchanged
  boxes and bytes: 128 of every 192-B row);
- k-block 2 (k 256..383): a per-weight **TAIL** tensor `[384*5120, 32] uint8` = the `.b4x16_p64` shared-memory image of
  k 256..287 (bytes 0-7 / 16-23 = packed FP4, pads 0), loaded by a UINT8 TMA (inner 32 B, box 128 B x 64 rows,
  128B swizzle). TMA zero-fills bytes 32..127 (k 288..383) exactly where w2 holds FP4 zeros.

FC2 weight bytes per row 192 -> 160 (-16.7%; with scales 1,044,480 -> 880,640 B per expert, -15.7%). The kernel is
the r4-e6 FC2-compact v2 (bitwise-qualified 2026-10-23, 1.10x without D05) merged onto the D05 source, with one
change: v2 read k 0..255 from a separate contiguous MAIN copy; here it reads them from the padded w2 itself, so the
only extra memory is the tail (the padded w2 must stay: M=1, prefill and the 128-row path use it).

Gate: `DSV41_MOE_FC2C=1` uses the compact kernel when the FC2 call has `M_sum >= round_up(6t + min(6t,384)*31, 32)`
with `t = DSV41_MOE_FC2C_MIN_TOKENS` (default 2 -> 384 rows), i.e. M >= 2 tokens. M = 1 (192 rows) keeps the D05
kernel (v2 was 0.956x at M=1). Requires `DSV41_MOE_FC2_D05=1`. `DSV41_MOE_FC2C=verify` serves the D05 output and runs
the compact kernel into a shadow buffer with the D05 verify compare (device counters, `fc2c-verify.trigger`).

## Why it is exact (bitwise)

- Same instructions and accumulation order: the offline nvcc compile of the real JIT kernel (`src/offline-nvcc/`,
  flags copied from the live D05 cache `meta.json`) gives SASS that differs from the live D05 cubin only in the TMA
  producer thread: (a) before its tile loop, the tail descriptor is acquired for the tensormap proxy and
  prefetched (`fence.proxy.tensormap::generic.acquire.sys` -> `CCTL.IV` + `UTMACCTL.IV` + `UTMACCTL.PF`; the CUDA
  guide's rule for host-written tensor maps in global memory, which also rules out a stale cached descriptor if a
  descriptor table is ever freed and re-allocated at the same address) and (b) the third k-block's B `UTMALDG.2D`:
  descriptor `c[0x3a8]` (tail) instead of `tensor_map_b`, coordinate 0 instead of 256, expect_tx 0x3180 instead of
  0x2180. The consumer (ldmatrix/MMA/epilogue, D05 store loop) is instruction-identical (656 SASS lines both; the
  other diff lines are branch-target shifts). 168 registers, 0 spills, 9 barriers in both.
  The offline compile reproduces the live D05 cubin's SASS exactly (same instruction text, 656 lines), and the D05
  code compiled against the new include tree (kBMainK = 0) is SASS-identical too.
- Same shared-memory data bytes: k 256..287 bytes come from the tail (= w2 bytes 128..143, checked at build), k
  288..383 are zero in both (w2 TP padding, checked non-zero -> refuse; TMA OOB fill). Pad bytes of each 16-B chunk
  are never read by `ldmatrix .b4x16_p64`. Scales (SFB) are the same tensor.
- **Stale LIVE_ACT padding rows (the caveat).** With `DSV41_MOE_LIVE_ACT=1` the padding rows of the FC2 input (FP8 q
  and packed UE8M0 scales) are never written, so they hold stale bytes and FC2's padding output rows can be NaN/Inf.
  The compact kernel does not change that, and it does not need zero A columns: B is zero at k >= 288 in both
  kernels, so every row (live or stale) sees identical operands and produces identical bits, NaN payloads included.
  Consequences: (1) D05's contract is unchanged (dead 8-row groups skipped, partially live groups written in full,
  ep_gather reads live rows only); (2) the GPU test poisons dead-row q and scales (NaN bytes and random bytes) and
  compares every written group bitwise incl. non-finite padding rows, and checks finiteness on live rows only
  (the r4 v2 test's "finite outputs" check over all rows would fail spuriously under LIVE_ACT); (3) any future
  change that reads FC2 padding rows must not assume they are finite.

## Files (`~/dsv41-stage/g1/fc2c-d05/`, identical on 8/8 except build/ and offline-nvcc/ on spark-four)

| path | what |
|---|---|
| `FILES` | site file contract: `fused_moe/experts/sm121_target_m32.py` 5acc5fde -> **9e250455** (only site change) |
| `site/.../sm121_target_m32.py` | 5acc5fde + `FC2Compact`, `load_fc2c_native`, `fc2c_mode`, `fc2c_min_rows`, `make_fc2c_tail`; verifier `tag`. With `DSV41_MOE_FC2C` unset it issues exactly the live module's native calls (CPU test) |
| `native/` | `_C` **33d4eb19** (hidden visibility, 369 dynamic symbols like D05/base; rebuilt twice, bit-identical), manifest **3b6f30bc**, 888 files = D05 runtime tree with the new `sm120_fp8_fp4_gemm_1d1d.cuh` (9f780e4b) |
| `src/fc2c-d05-full.patch` (90e9f35e) | baseline (M32 hidden-v2, commit 4c0aac24) -> candidate; `fc2c-on-d05.patch` = the delta on top of D05 `source.patch` |
| `src/build.py`, `prepare*.sh`, `assemble.py`, `build-result.json` | CPU build provenance: baseline reconstructed from the r4-e6 v2 tree by reverse patch and verified file-by-file against the D05 provenance list (6,943 sha256, `stubs/_C.pyi` restored), build 64 s, no warnings; v0 = first build before the fence (same `_C`) |
| `src/offline-nvcc/` | nvcc check script, kernel.cu (D05 and compact), normalized SASS, `sass-diff.txt`, ptxas resource reports |
| `tests/test_fc2c_cpu.py` | CPU: staged == live call-for-call with FC2C unset (D05 0/1/verify); gate == M_sum formula t=1..120; routing; capture-miss fallback; verify ordering; tail bytes vs a `.b4x16_p64` expansion model. **PASS on spark-four** |
| `tests/test_fc2c_unit.py` | GPU bitwise test (stopped window), see below |
| `tests/bench_fc2c.py` | GPU microbench at M = 1, 6, 12, 24, 48 (+ c4rec 12/24) on recorded routes, L2-cold, graphs |
| `bench/routes/` | the 10-21 recorded decode routes used by the 10-26 Marlin measurement |
| `ops/g1-site.sh`, `ops/g1-all.sh` | install / uninstall / status (backup `<file>.bak-g1-fc2c-d05`, 8/8 digest check) |
| `ops/prep-node.sh`, `ops/run-gpu-tests.sh` | stopped-window driver: private caches, unit test on 8 ranks, bench, shared-cache diff, serving JIT seed |
| `jit-cache/` | serving JIT cache of the new native (`DSV41_MOE_FC2C_JIT_DIR` default); empty until `prep-node.sh seed` after a PASS |

## Stopped window (about 10 min of GPU time; the 10-25 D05 unit test needed ~30 s per rank for 5 layers)

From the Mac with the service stopped on 8/8:

```
ssh spark-one cat ~/dsv41-stage/g1/fc2c-d05/ops/run-gpu-tests.sh | bash          # or QUICK=1 / BENCH_RANK7=1
```

GPU unit test per rank (5 real layers per rank = all 40 layers over 8 ranks; tokens 1,2,4,6,12,24,48,96; arms d05 =
live, fc2c = serving gate, fc2c_all = compact at every M, orig = no D05):
tail bytes; direct FC2 on real scatter32 layouts with stale LIVE_ACT padding (NaN and random) -> every written group
bitwise == d05, dead groups untouched, live rows == orig and finite; full MoE apply eager (poisoned workspaces) and
CUDA-graph replay with changing routes; capture-miss fallback (no allocation inside capture, still bitwise);
`verify` mode counters clean; gating (M=1 never compact, M=97 neither); the compact JIT kernel really compiled.
Private JIT/Triton/DeepGEMM caches under `jit-test/`; `prep-node.sh post` proves the shared caches are unchanged.

Microbench (rank 0, spark-one): per-layer full MoE apply and FC2-only, live d05 vs fc2c vs fc2c_all, with the
projection `delta ms/step = per-layer delta x 40`. Gate for a serving A/B: FC2-only >= 1.15x at M = 6..48 (r4-e6
gate), or a full-MoE projection >= 0.5 ms/step at 1 user / >= 1 ms at 4 users.

Then (still stopped): `ssh spark-one cat ~/dsv41-stage/g1/fc2c-d05/ops/g1-all.sh | bash -s install` (inert with
the flag unset; `G1_ALLOW_ACTIVE=1` stages it for the next start instead), relaunch, and ABAB with the AV pin on
both arms (one fixed matched-bench label for all arms):
`... dsv41-relaunch.sh | bash -s -- --label <x> DSV41_AV_COST_JSON=/home/jeweled/dsv41-stage/pdab/av-pin-pdfinal.json [DSV41_MOE_FC2C=1]`.

## Expected effect (projection, not measured)

Live FC2 per layer (D05, 10-26 Marlin harness, recorded routes, rank 0): 106.6 us at M=6 (20.7 distinct experts,
~203 GB/s), FC2 is 34-39% of the routed MoE layer. Byte model with D05's live-group stores: 1.17x at M = 6..48.
v2 ran at 97% of the padded kernel's bandwidth, so the central kernel estimate is **1.13x (range 1.10-1.17)**: the
r4-e6 1.15x gate is borderline, the microbench decides.

| load | rows (M) | live FC2 us/layer | saving us/layer | ms/step (40 layers) | step | expected |
|---|---|---|---|---|---|---|
| 1 user, C1 coding (full width) | 6 | 106.6 | 9.7-15.5 (central 12.7) | **-0.39 to -0.62 (central -0.51)** | 32.1 ms | -1.6% (single +1.2..2.0%) |
| 1 user, C1 prose (verify width ~3-4) | 3-4 | ~62-88 | ~7-10 | **about -0.35** | 27.6 ms | -1.3% |
| 1 user, M=1 steps | 1 | ~36 | 0 (gated off) | 0 | | |
| 4 users (c4rec, real batches) | 12 / 24 | ~176 / ~293 | 16-26 / 27-43 | **-0.64 to -1.70 (central -0.84 / -1.39)** | ~50.4 ms | -1.7..-2.8% (per session +1.7..2.8%) |

In tok/s: G1 single mean about +1.2..1.8% (96.1 -> ~97.5), G4 per session about +1.7..2.8% (~40.7 -> ~41.6).
Promotion needs +1.5% on G1 mean or G4 per-session mean: 1 user is borderline, 4 users likely if the kernel reaches
>= 1.12x. Memory: **+62.9 MB per target layer = +2.52 GB (2.34 GiB) per rank** (tails, built on the first eager
eligible call during warm-up, before any capture); MemAvailable after the full longctx gate is 12-13 GiB today.

## Rollback

- Env only: `DSV41_MOE_FC2C=0` (the default; unset = 5acc5fde behaviour).
- Site file (service stopped): `ssh spark-one cat ~/dsv41-stage/g1/fc2c-d05/ops/g1-all.sh | bash -s uninstall`
  (restores 5acc5fde from `.bak-g1-fc2c-d05`, checks 8/8).
- Nothing else is touched (launcher, D05 native/JIT cache, other site files unchanged).

## Not done / next levers

- No GPU run yet (offline prep). JIT cubin is compiled in the window (seconds) and seeded to `jit-cache/` after PASS.
- 16-B tail rows (-25% FC2 bytes instead of -16.7%) need a consumer-side change (packed tail tile + matching fragment
  loads); a UINT8 TMA cannot produce the `.b4x16_p64` image from fewer than 32 B per row (16-B stride/alignment rules).
- The 128-row path (M > 96, prefill) still reads the padded w2; routing it through the compact native would let the
  padded copy go (-2.3 GiB instead of +2.3 GiB) but needs prefill-size qualification.
