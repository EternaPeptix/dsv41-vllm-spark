# g1 / m32-bw: exact bandwidth variants of the M32 routed-MoE FP8xFP4 GEMMs (staged, NOT installed)

Offline prep 2026-10-26: CPU-only builds on spark-four (nice 19, 3 GiB cap); no restart, no GPU job, no live-file edit.
Everything is in `~/dsv41-stage/g1/m32-bw/` on 8/8 (identical digests, `PREP.ok`). Default off.

## Why the live kernels stream 170-194 GB/s while Marlin streams 211-234 GB/s on the same bytes

Model below is per layer at the bench's M=6 recorded windows (~21 experts, ~1.7 live rows per 32-row block; M=24 /
4 users: ~58 experts, ~2.5 rows), live launcher 04b6738e kernels `<BM32, BN64, BK128, 7 stages, 48 CTAs persistent,
256 math + 128 TMA threads>`, FC1 576x5120, FC2 5120x384 (D05). The GB/s figures count weights+scales only.

1. **Bytes the GB/s figure does not count (about half of the gap).**
   - FC1 A: the TMA leader loads the full 32-row x 128-B A box every k-block. Only the live rows were just written by
     ep_scatter. The junk rows are stale workspace rows that the 50+ MB of weights per layer have evicted: about
     3.2 MB/layer at 1 user (+9.9% of FC1 weight bytes) and 8.8 MB at 4 users.
   - Stores: FC1 TMA-stores all 32 rows (0.76 MB/layer, +2.3%); FC2 with D05 still stores whole 8-row groups (1.70 MB
     against 0.37 MB live, +6.1% of FC2 bytes).
   - Counted in, FC1 runs at about 201 GB/s and FC2 at about 221 GB/s. Marlin moves essentially only weights+scales.
2. **FC1's DRAM access pattern (hypothesis, tested by `pair` / `pf`).** FC1 rows are 2560 B (K-major FP4). Each
   BLOCK_K=128 box reads one 64-B piece of each of 64 rows. The next piece of the same row is requested one stage
   turn later (about 0.9 us at the shared-bandwidth rate), so each 64-B burst likely needs its own DRAM row activation.
   On 16 LPDDR5X channels that is activation-rate (tFAW) bound near 200 GB/s. FC2 tiles are one contiguous 12 KB
   block (64 rows x 192 B) and Marlin's repacked tiles are contiguous runs of 512 B or more. Both stream ~10% better
   on the same bytes, which is why FC1 (179 GB/s weights-only) lags FC2 (203) inside the same kernel.
3. **Checked and not the cause.**
   - Pipeline depth: 7 stages x 4 KB of weights in flight per CTA is 1.3 MB across 48 CTAs, against ~0.4 MB needed
     by Little's law at ~1.5 us loaded latency.
   - CTA count / wave tails: valid m-blocks are a prefix, and 48 = 0 mod 8 pins each CTA to one m residue. The busiest
     CTA has 5 FC1 tiles against a mean of 4.1, and even 8 tail CTAs keep >200 GB/s in flight.
   - Scheduler dead walk: ~0 at 1 user. Codex D04 measured 4-29 us/layer at m=12-96, included here as `dw`.
   - Scale loads: SFB is a contiguous 256-B box per k-block (6.25% of B), the same bytes Marlin reads.
   - PDL / launch overlap: pd-pdl early-B measured about 0.1 ms (HOLD).
4. **Not a GEMM issue: glue.** At M=6 the live apply spends 46.7 us/layer outside FC1/FC2 against Marlin's 25.6 us:
   8 glue kernels, about 3.5 us per boundary. The variants here do not touch it.

## Variants (tools/make_tree.py; header-only, the same native .so binaries; JIT keys move, SASS of 'off' == live)

All features are compile-time gated to the target geometry (`kG1`: MGroupedContiguous E384 BM32 BN64 BK128 48 SMs
split-1 FP8xFP4 BF16 identity epilogue). Every other instantiation compiles exactly as before.

| tree | FC1 (576x5120) | FC2 (5120x384, D05 tree) | what it tests |
|---|---|---|---|
| `off` | - | - | A/A control: SASS identical to live (checked for FC1, FC2, D05-FC2) |
| `pair` | k-block pairs issued back to back | live D05 | DRAM row locality, issue order only |
| `pf` | B+SFB L2 prefetch cursor 8 k-blocks ahead, 4-k-block bursts; EVICT_FIRST weight loads | live D05 | row locality + deeper effective pipeline |
| `la` | live-row A via cp.async (warp 9), no 32-row TMA box | live D05 | junk A DRAM reads |
| `ds` | live-row direct stores (32-bit st.global), no TMA store | same | junk C writes, per-tile epilogue barriers |
| `all` | pf + ef + la + ds + D04 dead-walk stop | ef + ds + dw | the combination |

**Exactness.** For every output element the MMA instruction sequence is unchanged: the same A/B fragments, the same
scales, and the same accumulator chain in the same K order (no split-K or tile-K change). Every live row is stored
with identical bytes: the `ds` store writes the same `cd_dtype_t(v)` pair the TMA path stages in smem. What changes:
- load issue order, prefetches and L2 policy;
- junk A rows in smem (left stale with `la`): an MMA output row depends only on its own A row and scale, and junk
  outputs are never read (ep_gather and LIVE_ACT read live rows only; dense SiLU at M<=5 only turns junk into junk);
- which dead output rows get written.

The unit test checks all of this bitwise, with every workspace NaN-poisoned before each run.

SASS (ops/build-jit.sh, `cc/`, all 168 regs):
- `off` gives identical SASS (1282 / 1234 / 1314 lines);
- `pf`: 62 UTMAPF;
- `la`: 9 LDGSTS + 1 cp.async-arrive, A TMA gone;
- `ds`: 4 STG, 0 UTMASTG;
- `all` has one 4-B local slot: 1 STL in the prologue, 3 LDL on the consumer side. Harmless, but noted.

## Files

- `tools/make_tree.py` (patch + tree builder), `tools/djkey.py` (deep_jit cache key, verified on 6 live entries)
- `ops/build-jit.sh` (spark-four: trees + pre-JIT cubins + SASS report), `ops/prep-node.sh` (every node: trees,
  pinned manifests `MANIFESTS.json`, `jit.sha256`, `FILES`, writes `PREP.ok`)
- `ops/run-gpu-tests.sh` (stopped window: unit + site + bench + projection, shared-cache diff)
- `ops/g1-site.sh` / `ops/g1-site-node.sh` (install / uninstall / status of the site file on 8/8)
- `site/make_site.py`, `site/vllm/.../sm121_target_m32.py` (5acc5fde -> db659d80; `DSV41_MOE_M32_G1` default off)
- `tests/g1lib.py`, `tests/test_m32bw_unit.py`, `tests/test_site_loader.py`, `tests/test_cpu.py`
- `bench/m32bw_bench.py` (recorded routes, 40-layer graphs, ABAB), `bench/project.py`, `routes/` (10-21 recordings)
- `trees/<v>-base|-d05`, `jit/<tree>` (pre-JIT entries: FC1 / D05-FC2 / transpose), `cc/` (SASS, spark-four only)

## Stopped-window plan (per node, single GPU; all 8 can run in parallel)

1. `bash ~/dsv41-stage/g1/m32-bw/ops/run-gpu-tests.sh unit` takes about 3-5 min. It runs every variant against live
   bitwise on synthetic and real-layer weights:
   - M 1..96;
   - routes valid / spec / half / hot / skew / one, plus recorded routes;
   - eager runs plus 60 graph replays per (arm, M).
   PASS = 0 failures. Run it on at least 2 ranks; it is fast, so 8/8 is better.
2. `... run-gpu-tests.sh site` takes about 2 min: the staged site loader path for each variant, bitwise vs live.
3. `... run-gpu-tests.sh bench` takes about 6-8 min on ranks 0 and 7. It runs:
   - M 1/6/12/24/48 recorded sets plus c4rec;
   - FC1/FC2 profiles at M 6 and 24;
   - `bench/project.py`, which projects to serving ms/step at 1 and 4 users.
   Gate: `off` within +-1% of live (A/A), and a variant >= 1.03x on the M=6 or c4rec layer time.
4. Optional, once: `compute-sanitizer --tool racecheck|synccheck python tests/test_m32bw_unit.py --variants la,all
   --tokens 6 --replays 2 --real-layer -1 --force-gpu-shared`.
5. The run script refuses while dsv41-rank is active and checks that the shared caches are unchanged.

## Serving A/B (after a bench winner; flags default off)

- Install the site file on 8/8 (service stopped): `ssh spark-one cat ~/dsv41-stage/g1/m32-bw/ops/g1-site.sh | bash -s install`.
  - With the flag unset it is the live code path (exact.py must still give a8c3394af9d2).
- Arm: supervisor relaunch with `DSV41_MOE_M32_G1=<variant>` (plus the AV pin `DSV41_AV_COST_JSON=...av-pin-pdfinal.json`).
  - It requires the launcher defaults `DSV41_MOE_FC2_D05=1` and no `DSV41_MOE_PDL_EARLY_B`.
- Protocol:
  - ABAB with 2 or more instances per arm, pinned;
  - exact.py a8c3394af9d2 with identical spec counters;
  - G1 bench single code/prose/math 1024 x2; G4 4 concurrent sessions x3; matched C1 under one fixed label;
  - GSM8K-50 >= 49/50 and longctx PASS on a candidate instance.
- Rollback:
  - env: relaunch without the flag;
  - file: `ssh spark-one cat ~/dsv41-stage/g1/m32-bw/ops/g1-site.sh | bash -s uninstall` (restores 5acc5fde from `.bak-g1-m32bw` and verifies 8/8).

## Expected (serving, before the bench confirms)

| arm | 1 user (C1 coding 32.1 ms) | 4 users (step ~50.4 ms) | mechanism |
|---|---|---|---|
| pair / pf | 0 to -0.65 ms | 0 to -1.7 ms | FC1 row locality (only if the hypothesis holds) |
| la | -0.1 to -0.55 ms | -0.4 to -1.4 ms | junk A rows; ~0 if they turn out L2-resident |
| ds | -0.25 to -0.35 ms | -0.6 to -0.85 ms | junk C stores (FC1 32 rows, FC2 8-row groups) |
| all | -0.95 to -1.5 ms (+3.1..+4.9% tok/s) | -2.0 to -3.9 ms (+4..+8% per session) | sum incl. D04 dead walk |

The lower ends assume no page-locality effect and partly L2-resident junk. Bench-to-serving scale:
- 1 user: serving FC1+FC2 = 172.6 + 104.8 us at M=6 (k1 new1, pre-D05) against the bench's 181.4 + 106.6, so x0.96;
- 4 users: 472 + 259 us (k1, 4u), about x0.9.
