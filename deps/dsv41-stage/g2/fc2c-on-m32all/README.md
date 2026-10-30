# g2 / fc2c-on-m32all: FC2-compact v2 tail rebased onto the live m32-bw `all` FC2 kernel (offline prep, 2026-10-26)

Staged only. Nothing is installed, no restart, no GPU job, no live site-packages or launcher edit. CPU work ran on
spark-four (nice 19, nvcc under a 3 GiB memory cap). `tree/` is rebuilt on every node from that node's own
immutable inputs and must match the pinned manifest (`PINS`); everything else is identical 8/8 (`MANIFEST.sha256`).

## What

Live target FC2 today (launcher 395bf5b2, `DSV41_MOE_M32_G1=all`): the Codex D05 native 6fea5091 with the m32-bw
`all` JIT header (FC2: `ef` EVICT_FIRST weight loads, `ds` live-row direct stores, `dw` D04 dead-walk stop), tree
`~/dsv41-stage/g1/m32-bw/trees/all-d05` (manifest 073232ef, header 693cd513, serving pre-JIT cubin
`dsv41_fc2_live_store_v1.c9b9ccac`). The g1 fc2c-d05 candidate edited the same site file and the same FC2 header,
so it could not be stacked. This stage merges them:

- **Kernel**: the g1 fc2c-d05 header delta (5 hunks, bitwise-qualified 8/8 on the plain D05 kernel on 10-26) applied
  to the live `all-d05` header (`tools/make_fc2c_tree.py`). The k-blocks 0/1 of B still come from the padded w2
  descriptor; k-block 2 (k 256..383) comes from a per-weight TAIL `[384*5120, 32] uint8` (the `.b4x16_p64` smem
  image of k 256..287) through a UINT8 TMA whose out-of-bounds zero fill supplies k 288..383, exactly where w2 holds
  FP4 zeros. FC2 weight bytes 192 -> 160 per row (-16.7%; with scales 1,044,480 -> 880,640 B per expert). Two
  adaptations: both compact B loads carry the live load's L2 hint `kG1BHint` (EVICT_FIRST), and a compile-time guard
  allows `kBMainK > 0` only on the g1 FC2 geometry without the FC1-only features (la / pf / pair).
- **Native**: a copy of the g1 fc2c-d05 tree (its `_C` 33d4eb19 = D05 host code + the opt-in
  `dsv41_fc2c_tail_descs` path; the same .so already qualified) with the merged header: `tree/`, manifest
  **5a79c849**, header **1900579d**, 888 files. Every file other than the header and the .so equals the all-d05 tree.
- **Site file** `fused_moe/experts/sm121_target_m32.py` db659d80 -> **ea4a8814** (only site change): db659d80 + the
  9e250455 fc2c additions (source-identical: `fc2c_mode`, `fc2c_min_rows`, the tail builder, `FC2Compact`, the
  verifier) + `fc2c_tree()`: the compact tree follows `DSV41_MOE_M32_G1` (`all` -> this g2 tree, off -> the g1
  fc2c-d05 tree, so the one-launch m32 rollback `DSV41_MOE_M32_G1=0` keeps working with FC2C on; pair/pf/la/ds
  refused). With `DSV41_MOE_FC2C` unset it issues exactly db659d80's native calls (CPU test).
- **Gate**: `DSV41_MOE_FC2C=1` uses the compact kernel when the FC2 call has `M_sum >= rows(t)`,
  `t = DSV41_MOE_FC2C_MIN_TOKENS` (default **2**: M >= 2 tokens); M = 1 stays on the live all-d05 kernel.
  `DSV41_MOE_FC2C_MIN_TOKENS=1` (compact at M=1 too) is one env var away if the bench favours it (`fc2c_all` arm).
- **Memory**: +62.9 MB per target layer = **+2.52 GB (2.34 GiB) per rank**, built on each weight's first eligible
  eager call (warm-up, before capture). A weight first seen inside a capture falls back to all-d05 in that graph
  (warned once: grep `first seen during CUDA-graph capture`; still exact).

## Why it is exact (bitwise)

- **SASS** (`ops/build-jit.sh`, `src/sass/`): the live all-d05 kernel.cu compiled against `tree/include` gives SASS
  identical to the serving all-d05 cubin (648 lines): the merged header is inert for every non-compact
  instantiation. The compact kernel (kernel.cu = the D05 one + `, 256`) differs from the serving all-d05 cubin in
  20 normalized SASS lines, all in the TMA producer, and that diff is structurally the same as the qualified g1
  diff (plain D05 -> fc2c-d05, 20 lines): tail descriptor acquire + prefetch (`LDCU c[0x3a8]`, `DEPBAR`,
  `CCTL.IV`, `UTMACCTL.IV`, `UTMACCTL.PF`), third k-block B load from `c[0x3a8]` at coordinate 0 instead of
  tensor_map_b at 256, expect_tx 0x3180 instead of 0x2180, and the return-address constants shifted by the 5
  inserted instructions. The same UTMALDG (with its EVICT_FIRST operand) issues the tail load. Consumer
  (ldmatrix/MMA/`ds` epilogue) identical; 168 registers, 0 spills, both.
- **Data**: k 256..287 bytes of smem come from the tail (= w2 bytes 128..143, checked when the tail is built), k
  288..383 are zero in both (w2 TP padding, checked non-zero -> refuse; TMA OOB fill). Same scales (SFB).
- **Stores**: unchanged `ds` contract: each live row is written with the same 32-bit BF16 pairs, nothing else.
  Stale LIVE_ACT padding inputs (NaN / random) never reach memory in either kernel.
- **Verify mode**: `DSV41_MOE_FC2C=verify` with `all` first fills the served FC2 buffer with the shadow's poison
  (the all-d05 kernel stores live rows only; dead rows are never read), so the group-wise compare is meaningful.

## Files (`~/dsv41-stage/g2/fc2c-on-m32all/`)

| path | what |
|---|---|
| `FILES` | site contract: `sm121_target_m32.py` db659d80 -> ea4a8814 |
| `PINS` | tree manifest 5a79c849, pre-JIT list 11b6aab6, entry, merged header, site sha |
| `site/.../sm121_target_m32.py` | staged site file (ea4a8814) |
| `tree/` | merged native tree, rebuilt per node by `prep-node.sh build` (not in MANIFEST; pinned) |
| `jit-cache/cache/dsv41_fc2c_d05_v1.3230a7e5...` | serving pre-JIT cubin (key from `tools/djkey.py`, which reproduces the live c9b9ccac and g1 16a0ffc6 keys) |
| `tools/make_fc2c_tree.py` | header merge + tree build + self-checks (the 5 hunks on the D05 header reproduce 9f780e4b byte for byte) |
| `tools/djkey.py` | deep_jit cache key (m32-bw copy) |
| `ops/build-jit.sh` | spark-four only: key checks, A/A SASS, pre-JIT, SASS diffs |
| `ops/prep-node.sh` | `build` (tree + pins + CPU test -> PREP.ok), `pre` / `post` (private test caches, serving-cache diff), `clean` |
| `ops/run-gpu-tests.sh` | stopped-window driver (from the Mac) |
| `ops/g2-site.sh`, `ops/g2-all.sh` | install / uninstall / status on 8/8 (`.bak-g2-fc2c-on-m32all`), `g2-all.sh build` |
| `tests/test_g2_cpu.py` | CPU: provenance vs db659d80 and 9e250455, call-for-call with FC2C unset, gate, routing, verify poison, tree selection, tree pin + reproducibility, tail bytes. **92/92 PASS** |
| `tests/test_g2_unit.py` | GPU bitwise test (stopped window) |
| `tests/bench_g2.py`, `tests/bench_summary.py` | GPU microbench + summary |
| `bench/routes/` | the 10-21 recorded decode routes (same as g1) |
| `src/sass/` | SASS listings, diffs, ptxas resources, build log |

## Stopped window (~10-15 min of GPU time)

Prerequisite (any time, CPU only): `ssh spark-one cat ~/dsv41-stage/g2/fc2c-on-m32all/ops/g2-all.sh | bash -s build`
(already done: PREP.ok on 8/8). With the service stopped on 8/8, from the Mac:

```
ssh spark-one cat ~/dsv41-stage/g2/fc2c-on-m32all/ops/run-gpu-tests.sh | bash       # QUICK=1 for a short pass
```

1. `prep-node.sh pre` on 8/8: private caches under `jit-test/` (the staged loader's `_G1_ROOT` -> `jit-test/g1root`,
   whose `trees` links the real m32-bw trees and whose `jit/all-{base,d05}` are copies; `fc2c-all` = copy of the
   pre-JIT cache; `live-base`, `dg`, `triton` copies) and listings of every serving cache.
2. `tests/test_g2_unit.py` on 8 ranks in parallel (5 real layers each = all 40; tokens 1,2,4,6,12,24,48,96; routes
   valid / skewed / spec / half / holes / one / all_dead). Natives through the STAGED `get_target_m32_kernel()` with
   `DSV41_MOE_FC2C=1` (the serving construction). Arms: `live` = the installed db659d80 class on the same natives,
   `staged_off`, `fc2c` (serving gate), `fc2c_all` (compact at every M), `orig` (pristine FC2 kernel, live rows).
   Checks: tail bytes; direct FC2 with stale LIVE_ACT padding (NaN and random): whole output buffers bitwise equal
   to live with one sentinel (same rows written, same bytes), live rows == orig and finite; full MoE apply eager
   (NaN-poisoned workspaces) and CUDA-graph replay with changing routes; capture-miss fallback; verify mode (direct
   and full MoE; counters clean; a control without the poison must count the padding rows); gating (M=1 live,
   M=2 compact, M=97 neither); the compact kernel ran from the pre-JIT entry (cache hit, nothing compiled).
3. `tests/bench_g2.py` on rank 0 (spark-one) and rank 7 (spark-eight): full MoE layer and FC2-only, live vs
   `fc2c` vs `fc2c_all`, M = 1, 6, 12, 24, 48 + c4rec 12/24 (real 4-user batches), recorded routes, L2-cold,
   40-layer graphs, 25 trials; projection = per-layer delta x 40 (x0.96 / x0.90 serving scale).
4. `prep-node.sh post` on 8/8: serving caches unchanged (paths, sizes, mtimes).

Gate for a serving A/B (suggested): unit 8/8 PASS, and a full-MoE saving of >= 0.25 ms/step at M=6 or >= 0.6 ms/step
at c4rec M=24 (about +0.8% G1 code / +1.3% per 4-user session after scaling).

## Serving A/B (window rule: >= 0.5% on G4 prose-session min or G1 mean in every pinned pair)

- Install (service stopped): `ssh spark-one cat ~/dsv41-stage/g2/fc2c-on-m32all/ops/g2-all.sh | bash -s install`
  (inert with the flag unset: exact.py must still give a8c3394af9d2 with the pinned counters).
- Arm: `... dsv41-relaunch.sh | bash -s -- --label <x> DSV41_AV_COST_JSON=/home/jeweled/dsv41-stage/pdab/av-pin-pdfinal.json DSV41_MOE_FC2C=1`
  (no launcher edit needed for the A/B; promotion = one launcher line `export DSV41_MOE_FC2C=${DSV41_MOE_FC2C:-1}`
  after the `DSV41_MOE_M32_G1` line).
- Per instance: exact.py first (a8c3394af9d2, identical spec counters), G1 digest, G4 x3, matched C1; rank logs:
  the `DSV41_MOE_FC2C=1: FC2-compact tails on (min rows 384); 62.9 MB per FC2 weight` line on 8/8 and no
  `first seen during CUDA-graph capture` warning; MemAvailable (telemetry off for the measured blocks).
- Before promotion: GSM8K-50 >= 49/50, full longctx PASS, and one cold 1M probe with MemAvailable sampled.

## Memory (+2.52 GB per rank)

MemAvailable minimum during a cold 1M prompt today: 10.2-11.6 GB (pd X-ray; 1M takes 2-7 GB on top of the
post-start 17.6-18.6 GB). Current-config telemetry (gw1-U-2 incl. full longctx): spark-one 10.77 GB, others 11.5-11.8
GB. With the tails: about **7.7-9.1 GB at 1M (worst rank about 7.7 GB) >= 6 GB**, about 1.7 GB of margin on the worst
rank. Do not run CPU builds on a node during a 1M test with FC2C on (the 10-26 08:20 spark-four build dipped
MemAvailable by about 4 GB).

## Rollback

- Env: `DSV41_MOE_FC2C=0` (default; unset = db659d80 behaviour).
- Site file (service stopped): `ssh spark-one cat ~/dsv41-stage/g2/fc2c-on-m32all/ops/g2-all.sh | bash -s uninstall`
  (restores db659d80 from `.bak-g2-fc2c-on-m32all`, checks 8/8).
- Order with the m32-bw rollback: uninstall g2 first (restores db659d80), then the m32-bw site rollback
  (`~/dsv41-stage/g1/m32-bw/ops/g1-site.sh uninstall` restores 5acc5fde from its own backup whatever is installed).
  The one-launch m32 rollback `DSV41_MOE_M32_G1=0` needs no file change: FC2C then uses the g1 fc2c-d05 tree.
- After a promotion, `tree/` and `jit-cache/` become serving dependencies (like `~/dsv41-stage/g1/m32-bw/`).

## Not done / next

- No GPU run yet (offline prep). The g1 v2 tail was 0.956x at M=1 but g1 fc2c-d05 measured `fc2c_all` M=1 FC2
  x1.08: the `fc2c_all` arm decides whether `DSV41_MOE_FC2C_MIN_TOKENS=1` is worth an A/B arm.
- 16-B tail rows (-25% FC2 bytes) need a consumer-side change; routing the 128-row path through the compact native
  would let the padded w2 go (-2.3 GiB instead of +2.3 GiB) but needs prefill-size qualification.
