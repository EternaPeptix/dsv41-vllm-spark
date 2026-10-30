# pd-rgate: exact cuBLAS-replica router gate kernel for M <= 8 (round-3 #1, arm A)

Status 2026-10-25 (offline prep): extension built (CPU-only, spark-five) and staged on 8/8, NOT installed. CPU test PASS. **Bitwise equality with cuBLAS is unproven.** The GPU probe decides it.

## Mechanism

- Today GateLinear tier 5 runs `torch.mm(x, W.T, out_dtype=fp32)` (W bf16 [384, 5120]). At M <= 8 cuBLAS uses a WMMA kernel:
  - one warp per CTA, grid 8x3x16 (24 expert tiles x 16 K-splits), two waves on 48 SMs;
  - plus `splitKreduce`;
  - about 22.5 + 2.9 µs per layer, kernel-bound;
  - it blocks the shared w13 kernel for about 19 µs.
- `src/router_gate.cu` replaces this with one launch:
  - 24 CTAs (one 16-expert tile each) x 16 warps (one K-split each);
  - each warp issues the same `HMMA.16816.F32.BF16` chain (A = W rows, B = x rows, k16 slices in table order; SASS confirmed: sm_121 cubin, HMMA.16816.F32.BF16, 78 regs, 0 spills, 8 KB smem);
  - the 16 fp32 partials are reduced in smem with `__fadd_rn` in a selectable order.
- Hypotheses for the split order, in `dsv41_rgate.schedule`:
  - H1z / H1 / H1r / H1rz: CUTLASS gemm_k_size 320, residue first or last, with or without zero MMAs;
  - H2a / H2b / H2c: whole 128-wide tiles, contiguous or interleaved;
  - H3: 384 per split with two empty splits.
- Seven reduce orders (sequential, from-first, tree, reverse, 2- and 4-interleaved, reverse-from-0).
- Expected: -0.35..-0.55 ms per step at 1u if exact (+1.0-1.6% code, +1.2-1.9% prose). 0 at conc >= 8: M > 8 stays on cuBLAS/nvjet.

## Flag (staged `gate_linear.py` reads it once at import; the hook sits inside tier 5 only)

| `DSV41_ROUTER_GATE_REPLICA` | behaviour |
|---|---|
| `0` (default) | stock. The CPU test asserts the staged file equals live once the two guarded blocks are removed. |
| `1` | replica for M in the config's `m_allow` |
| `verify` | serves cuBLAS, runs the replica too, and counts mismatching fp32 words on the device (all layers, live inputs, eager + graphs). Trigger: `touch ~/dsv41-logs/rgate-verify.trigger` (≥900 s after start) -> `rgate-verify.<host>.<pid>.json`. |

`1` and `verify` require `~/dsv41-stage/pd/rgate/rgate-config.json` with status PASS. Only the probe writes it; without it, startup fails loudly. Override the path with `DSV41_ROUTER_GATE_CFG`.

## Files

| site file | live | staged |
|---|---|---|
| `fused_moe/router/gate_linear.py` | 41adaba8 (pristine wheel) | cdf9c82c |
| `fused_moe/router/dsv41_rgate.py` | absent | new |

Also `build/dsv41_rgate_ext.so` (e708c229, built from `src/router_gate.cu` by `ops/build.sh`). Install: `pd-all.sh install rgate`. No conflicts with the other pd candidates.

## Gate (stopped window, about 5 min, one node)

1. Run `ops/run-probe.sh` on spark-four. It does:
   - a cuBLAS census of kernel name and grid per M;
   - the hypothesis search on real gate weights (40 layers) and random weights, with a signed-zero tie-break;
   - a full sweep of the chosen pair: 40 layers x M 1..8 x 2000 trials, plus graph replay;
   - graph timing (warm and rotating weights).
   - On PASS it writes `rgate-config.json`.
   - **Kill** if no pair is exact, or if the speedup is below 1.8x at M=4 (kernel-ideas-3 bar). `m_allow` keeps only M values that are exact and at least 1.5x faster; adjust with `--min-speedup`.
2. Copy `rgate-config.json` to all 8 nodes. Optionally run the live shadow with `launch-conc.sh DSV41_ROUTER_GATE_REPLICA=verify` and require 0 mismatches on 8/8.
3. ABAB with pinned AV: `DSV41_ROUTER_GATE_REPLICA=0` vs `1`. exact.py must be identical. The C1 bar is ≥0.8% per pair. This item is near the noise floor alone, so bundle it with pd-pdl / D02 qkv_a (bundle α).
