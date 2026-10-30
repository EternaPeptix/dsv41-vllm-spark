# g3 / mfront: fused M32 MoE permute (count + fills + scatter + UE8M0 pack in 2 kernels) (staged, NOT installed)

Offline prep 2026-10-26 (goal window 3, task moe-drafter-bs4). No restart, GPU job or live-file edit. CPU tests ran on
spark-four (Triton interpreter, nice 19). Everything is in `~/dsv41-stage/g3/mfront/` on 8/8 with identical digests
(`MANIFEST.sha256`, `PREP.ok`). Flag default off.

## Why

Between the router and FC1, every target MoE layer on the M32 decode path runs 6 small launches that only move
data. Codex's 2027f4d8 4-user profile (rank 0, 40 layers, us per layer):

| kernel (live path) | C1 (M 4) | C4 (M 12-16) | m32bw bench c4rec M=24 |
|---|---|---|---|
| `count_expert_num_tokens` | 1.4 | 2.0 | 1.5 |
| 2 x `vectorized_elementwise` (torch.full(-1) layout, torch.zeros scales) | 2.2 | 4.3 | 3.0 |
| `ep_scatter_1` (384 programs, each a full cumsum) | 2.8 | 3.5 | 2.4 |
| `ep_scatter_2` (one program per token, 6 serial atomics + row copies) | 6.4 | 9.2 | 5.1 |
| DeepGEMM `transpose_and_pack_fp32_into_ue8m0` (inside FC1) | 1.9 | 3.0 | 1.9 |
| **sum** | **14.7** | **22.0** | **13.9** |

That is 0.6 ms/step at 1 user and 0.9 ms at 4 users, all launch-latency bound. (The activation quant
`per_token_group_quant_8bit`, 2.7-3.0 us, and the topk kernel stay as they are in this package.)

## What (DSV41_MOE_FUSED_PERMUTE, default 0)

`fused_moe/deep_gemm_utils.py` a4a0486a -> **58e264c8** (only site change; `make_patch.py` = 4 anchored hunks).
Eligible calls only: E 384, topk 6, M x 6 <= 1024 slots, 32-row alignment (the M32 path, M <= 96), FP8 activations
with fp32 per-128 UE8M0 scales, no expert_map / psum layout / precomputed counts. Everything else (drafter E128,
prefill, 128-row path) is untouched.
- `plan` kernel: grouped_layout = -1 and packed scale words = 0 over M_sum rows (what torch.full / torch.zeros +
  transpose_and_pack leave), plus the 32-aligned expert starts (histogram of the routed slots, the same
  round-up + exclusive cumsum as `ep_scatter_1`).
- `scatter` kernel: one program per (token, k) slot: `dest = atomic_add(cursor[e], 1)` (as `ep_scatter_2`), copy the
  FP8 row, `grouped_layout[dest] = e`, `inv_perm`, and the packed int32 UE8M0 words written straight into DeepGEMM's
  MN-major TMA-aligned layout `[M_sum, 10]`, strides `(1, round_up(M_sum, 4))`, with the exact
  `transpose_and_pack_fp32_into_ue8m0` formula `(v0 >> 23) | (v1 >> 15) | (v2 >> 7) | (v3 << 1)`.
- FC1 then receives packed int32 scales (FC2 already does, from the fused SiLU-quant kernel), so its internal
  `transpose_and_pack` launch disappears. The cursor array ends as the LIVE_ACT expert ends.
- `verify`: stock served, the fused kernels run into shadow buffers and a compare kernel counts mismatches on the
  device (row words, scale words, layout / ends / dead-row scale words). Report: `touch
  ~/dsv41-logs/mfront-verify.trigger` (after 600 s) -> `mfront-verify.<host>.<pid>.json`.

## Why it is exact (bitwise MoE output)

- Every live slot's FP8 row is a byte copy of the same quantized row; every packed scale word is the same bit formula
  on the same fp32 scales; grouped_layout, the per-expert 32-aligned segments (live rows a prefix: the D05 / ds /
  LIVE_ACT contract), the LIVE_ACT ends and the dead-row scale words are identical arrays.
- The only freedom is the order of rows INSIDE an expert segment, which is atomic-arrival order in both paths. Every
  FC1/FC2 output row depends on its own A row and scale only (same MMA chain per element), SiLU-quant is per row and
  `ep_gather` reads through `inv_perm` in topk order, so the MoE output is unchanged.
- Dead rows: A rows stay stale in both paths (the 'la' FC1 loads live rows only); dead scale words are 0 in both.

## Tests

| test | where | result |
|---|---|---|
| `tests/test_mfront_cpu.py` | spark-four, Triton interpreter, nice 19 | **PASSED**: provenance; flag-off output == a4a0486a for every case; fused vs stock on E 384 (M 1..96) and E 48: 140 cases x routes valid / skewed / spec / half / holes / one / all_dead, int32 and int64 ids (layout, ends, rows, packed words vs the formula on the stock scattered scales, dead words 0, per-expert destination sets); the gate (6 negatives); the verify compare kernel (clean = 0, and a corrupted row / scale word / layout entry / end / dead word each counted) |
| `tests/test_mfront_gpu.py` | stopped window, 8 ranks + bench on ranks 0/7 (`ops/run-gpu-tests.sh`) | not run (needs GPU) |

The GPU test runs the serving stack exactly as launcher 592793fd (M32 + COMPACT + LIVE_ACT + D05 + M32_G1=all + FC2C,
the installed sm121_target_m32.py ea4a8814 and its natives) with PRIVATE copies of every JIT cache (`jit-test/`):
- MoE output of `off` / `fused` / `verify` == `live` bitwise on all 40 real layers (5 per rank), M 1,2,3,4,5,6,8,12,
  16,24,32,48,64,96, route kinds valid / skewed / spec / half / holes / one / all_dead + recorded 4-user batches,
  NaN-poisoned workspaces;
- the direct permute check (fused packed words also against DeepGEMM's own packing when the native exposes it);
- CUDA graphs per arm replayed over changing routes == live;
- verify counters 0 over all calls, and a forced one-byte corruption counted (24 slots);
- `--bench` (ranks 0 and 7): 40-layer graphs cycling layers 0/13/26/39 (L2-cold) on recorded routes (M 1, 6, 12,
  24, 48 + c4rec 12/24), live vs fused ABAB, torch.profiler kernel list at M 6 and 24, projection to ms/step
  (x0.96 at 1 user, x0.90 at 4 users as in g1/g2).

## Stopped window (~15-25 min, 8 ranks in parallel)

```
ssh spark-one cat ~/dsv41-stage/g3/mfront/ops/run-gpu-tests.sh | bash          # QUICK=1: tokens 1,6,12,24,96
```
Refuses unless dsv41-rank is inactive, the GPU idle and PREP.ok present on 8/8. `prep-node.sh pre` makes the private
caches (m32-bw `jit/all-base`, `jit/all-d05`, the g2 fc2c `jit-cache`, empty DeepGEMM/Triton dirs) and lists every
serving cache; `post` must find them unchanged. Gate for a serving A/B: unit PASS 8/8 and a bench saving of
>= 8 us/layer at c4rec M=12 (>= 0.3 ms/step) on both ranks.
Contingency if the native refuses packed int32 A scales for FC1 (FC2 already takes them, so unlikely): the GPU test
fails on the first fused call; a fp32-scatter variant (skip only 5 of the 6 launches) is the fallback.

## Serving A/B (after the bench; flag default off)

- Install (service stopped): `ssh spark-one cat ~/dsv41-stage/g3/mfront/ops/all.sh | bash -s install`. Inert without
  the flag: exact.py a8c3394af9d2 with the pinned counters.
- Arm: `... dsv41-relaunch.sh | bash -s -- --label <x> DSV41_AV_COST_JSON=<fresh g3 pin> DSV41_MOE_FUSED_PERMUTE=1`.
  First serving start compiles 2 new Triton kernels (seconds) into the serving Triton cache.
- Per instance: exact.py first (a8 + identical pinned counters), G1 digest, G4 x3 + G4min_hat, matched C1.
  Optional proof over real traffic: one `=verify` arm, touch the trigger, expect `"status": "OK"` on 8/8.
- Expected (bench to be confirmed): about -0.4 ms/step at 1 user and -0.55..-0.65 ms at 4 users, i.e. G1 about
  +1.3% and the G4 prose MIN about +1.2..1.6% (prose-c4 sensitivity: 2.7% per ms on every step). Memory: slightly
  less (no fp32 [M_sum, 40] scale buffer, no FC1-internal packed copy).

## Rollback

- Env: relaunch without `DSV41_MOE_FUSED_PERMUTE` (default 0).
- File (service stopped): `ssh spark-one cat ~/dsv41-stage/g3/mfront/ops/all.sh | bash -s uninstall` (restores
  a4a0486a from `.bak-g3-mfront`, checks 8/8).

## Next steps in the same direction (not staged)

1. Fold the activation quant (`per_token_group_quant_8bit`, 2.7-3.0 us/layer) into the scatter kernel: needs the
   vLLM CUDA kernel's exact UE8M0 rounding (log2/ceil path) replicated; GPU-exhaustive check over amax exponents.
2. Count from the router: `_dsv4_topk_kernel` could emit the per-expert histogram (saves the plan kernel's scan).
3. FC1 -> FC2 per-expert pipelining (see the moe-drafter-bs4 README).

## Files

`FILES`, `make_patch.py` (+ `base/deep_gemm_utils.py.a4a0486a`), `site/...` (staged file), `tests/test_mfront_cpu.py`,
`tests/test_mfront_gpu.py`, `bench/routes/` (10-21 recorded decode routes, same as g1/g2),
`ops/{site,all,prep-node,run-gpu-tests}.sh`, `results/cpu-test.txt` (spark-four CPU test record).
