# g3 / dhtac: pinned CUTLASS tactic for the DSpark draft head at 4+ requests (staged, NOT installed)

Offline prep 2026-10-26 (goal window 3, task moe-drafter-bs4). No restart, GPU job or live-file edit. CPU tests ran
on spark-four (nice 19). Everything is in `~/dsv41-stage/g3/dhtac/` on 8/8 with identical digests (`MANIFEST.sha256`,
`PREP.ok`). Flag default off.

## Why

The DSpark draft head is the vocab-sharded MXFP8 GEMM `[M x 5120] x [5120 x 16160]` (85 MB of weights per rank,
`DSV41_FP8_LMHEAD=draft`, `dsv41_r3_dense.Fp8Head`). `M = 5 x requests`. FlashInfer picks the CUTLASS tactic per
M bucket from the serving autotune cache (`~/.cache/vllm/flashinfer_autotune_cache/.../8c8be689.../
autotune_configs_dp0_rank*.json`, loaded by every launch since 10-23):

| M bucket | 1 | 2 | 4 | 8 | 16 | **32** | **64** | **128** | 256 |
|---|---|---|---|---|---|---|---|---|---|
| tactic | 1 | 1 | 3 | 1 | 1 | **0** | **0** | **0** | 3 |

Tactics: `0..5 = 128x{32,64,128}` CTA tile x `{no swap, swap_ab}`. Tactic 0 (no swap, 128x32) launches 505 CTAs
whose 128-row A tiles are mostly TMA zero fill. It is what every DSpark step with 4-25 requests gets (M 20-125).
Codex's 2027f4d8 profile (rank 0) shows the cost:

| drafter graph | head kernel | head us | drafter span |
|---|---|---|---|
| C1 (M=5) | `128_32_128true` grid 1x127 | 423 | 2.66 ms |
| C2 (M=10) | `128_32_128true` grid 1x127 | 433 | 3.24 ms |
| **C4 (M=20)** | `128_32_128false` **grid 1x505** | **1104** (77 GB/s) | 4.68 ms |

So about **0.67 ms of the +1.44 ms bs2 -> bs4 drafter growth is this tactic switch**. The rest is the drafter MoE
(+0.57 ms: more distinct experts on the generic 128-row path) and collectives / mHC / glue (+0.2 ms).

## What (DSV41_DRAFT_HEAD_TACTIC, default 0)

`dsv41_r3_dense.py` 993023f2 -> **1f1c0456** (only site change; `make_patch.py` = 4 anchored hunks, nothing else):
- `0` (default): no helper object is built and `Fp8Head.__call__` issues exactly the live calls (CPU test).
- `1`: for the row counts in `DSV41_DRAFT_HEAD_TACTIC_MAP` (default `17-32:1`; set it from the bench), the head calls
  FlashInfer's own CUTLASS SM120 MXFP8 runner (`get_cutlass_mxfp8_gemm_module(12).cutlass_mxfp8_gemm_runner()`) with
  the pinned tactic and the exact inputs list `mm_mxfp8(backend="cutlass")` builds (same workspace buffer
  `mm_mxfp8_workspace`); every other M stays on the stock call.
- `audit`: the stock output is served; the pinned tactic runs as a shadow and the mismatching bf16 words are counted
  on the device. Report: `touch ~/dsv41-logs/dhtac-audit.trigger` (after 600 s) -> `dhtac-audit.<host>.<pid>.json`.

## Why it is exact (bitwise)

- The six SM120 MXFP8 configs differ only in CTA tile shape and operand swap. There is no split-K or stream-K
  (`ClusterShape_1x1x1`, one K loop per CTA), so every output element is the same block-scaled MMA chain in the same
  K order.
- **Enforced, not assumed.** The first (eager) build of the head runs every mapped M through all six tactics and the
  stock call on two synthetic MXFP8 inputs, and pins an M only if all seven outputs are bitwise equal. A failing M
  stays stock and is logged (`kept on the stock path`). The head is never built during CUDA-graph capture
  (`fp8_head_apply` already refuses), so the self-check always runs eagerly at startup (~0.2 s for 16 M values).
- Why it matters even though exact.py and G1 cannot see it: a 1-user step has M=5 (tactic 1 already), so the gates
  are blind to bs>=4. At 4 users the draft tokens AND the DSpark confidences feed the AV budget, so any bit change
  would change verify widths and therefore target numerics. Hence: bitwise or nothing.

## Tests

| test | where | result |
|---|---|---|
| `tests/test_dhtac_cpu.py` | spark-four, CPU only, nice 19 | **PASSED** (provenance, flag-off call-for-call identical to 993023f2 for M 1..160, mode-1 and audit dispatch with a fake runner, map parsing) |
| `tests/test_dhtac_gpu.py` | stopped window, 8 ranks (`ops/run-gpu-tests.sh`) | not run (needs GPU) |

The GPU test, per rank (real `head.weight` shard of that rank):
1. builds the staged head with a map covering every M 1..160: the self-check must pin all of them;
2. **equivalence classes**: for M = 5 x every DSpark graph size (1,2,3,4,5,7,8,10,...,32 requests) plus extra M,
   and 4 activation sets (block-scale spread, RMSNorm-like with large channels, half-zero rows, tiny/one-hot rows),
   all six tactics are grouped by bitwise-equal output;
3. the serving tactic for each M (parsed from the autotune file the rank log names) must share the class of the
   proposed tactic on every input set;
4. **microbench**: per (M, tactic), a CUDA graph of 8 back-to-back head GEMMs (L2-cold by size), interleaved
   replays, median us;
5. the patched `__call__` (mode 1) equals the serving tactic's output for every M;
6. audit mode: served == stock, 0 mismatching words, and a forced mismatch is counted.

Last line: `recommend` with the `DSV41_DRAFT_HEAD_TACTIC_MAP` for the drafter graph sizes (fastest tactic in the
serving tactic's class, only where >= 3% faster) and the bs4 saving in us.

## Stopped window (~5 min of GPU, all 8 ranks in parallel)

```
ssh spark-one cat ~/dsv41-stage/g3/dhtac/ops/run-gpu-tests.sh | bash          # QUICK=1 for a shorter pass
```
Refuses unless dsv41-rank is inactive, the GPU idle and PREP.ok present on 8/8. `prep-node.sh pre/post` list the
FlashInfer JIT cache and the autotune cache before/after: they must be unchanged (the test only reads them).
Gate for a serving A/B: unit PASS 8/8, every drafter M in the recommended map in one class with the serving tactic
on every rank, and a bs4 saving >= 0.3 ms.

## Serving A/B (after the bench; flags default off)

- Install (service stopped): `ssh spark-one cat ~/dsv41-stage/g3/dhtac/ops/all.sh | bash -s install`. Inert without
  the flag: exact.py must still give a8c3394af9d2 with the pinned counters.
- Arm: `... dsv41-relaunch.sh | bash -s -- --label <x> DSV41_AV_COST_JSON=<fresh g3 pin> DSV41_DRAFT_HEAD_TACTIC=1
  DSV41_DRAFT_HEAD_TACTIC_MAP=<recommend map>`. Rank logs must show `DSV41_DRAFT_HEAD_TACTIC=1: N of N mapped row
  counts pinned` on 8/8 and no `kept on the stock path`.
- Optional in-serving proof over real 4-user traffic: one arm with `=audit` (same map), run G4, touch the trigger,
  expect `"status": "OK"` and `mismatching_words: 0` on 8/8.
- Expected: the 4-request drafter step 1104 -> ~435 us head (**about -0.65 ms per 4-request step, about +0.9% on
  the G4 prose MIN** by the prose-c4 sensitivity of 1.4% per ms on 4-request steps). conc8/16 gain too if the
  bench pins buckets 64/128. No effect on 1-3 request steps (M <= 15 already uses tactic 1).
- With an UNPINNED AV profile the measured bs4 draft cost drops, so AV budgets change (that is a pricing change,
  not a numerics change of the drafter). Keep every A/B arm on the same fresh pin.

## Rollback

- Env: relaunch without `DSV41_DRAFT_HEAD_TACTIC` (default 0).
- File (service stopped): `ssh spark-one cat ~/dsv41-stage/g3/dhtac/ops/all.sh | bash -s uninstall` (restores
  993023f2 from `.bak-g3-dhtac`, checks 8/8).

## Files

`FILES` (site contract), `make_patch.py` (+ `base/dsv41_r3_dense.py.993023f2`), `site/...` (staged file),
`tests/test_dhtac_cpu.py`, `tests/test_dhtac_gpu.py`, `ops/{site,all,prep-node,run-gpu-tests}.sh`,
`results/cpu-test.txt` (spark-four CPU test record for this MANIFEST digest).
