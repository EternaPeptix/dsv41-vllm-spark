# pd-d05: Codex D05 dead FC2 store-group skip, wired into serving (default off)

Status 2026-10-25 (offline prep): staged on 8/8 and NOT installed. CPU dispatch test PASS. GPU unit tests written but not run (no GPU jobs during prep).

## What it does

- In the M32 target path (<=96 tokens, alignment 32), the target FC2 grouped GEMM (E384, N5120, K384, BF16 out) goes to the Codex D05 native (`_C` 6fea5091, manifest fc230f26, immutable tree `~/dsv41-goal120-20261024/d05-shadow-v1/native`) with `dsv41_fc2_live_store=True`.
- The kernel stages and computes every 32-row tile exactly as before. Its elected lane skips the TMA store of any 8-row group whose first `grouped_layout` row is -1.
  - Under the scatter32 prefix contract such a group is entirely dead, and ep_gather never reads dead rows.
  - So served bytes are unchanged. Only stale contents of dead workspace rows differ.
- Unchanged: FC1, the drafter MoE (E128), and the align-64/128 paths (prefill and >96 tokens). They stay on the original native `b1c4eff8` or the installed DeepGEMM.
- The flag is read once at worker start, in `get_target_m32_kernel`:

| `DSV41_MOE_FC2_D05` | behaviour |
|---|---|
| `0` (default) | identical calls to live `723fdf8d` (CPU test asserts call-for-call equality) |
| `1` | target FC2 -> D05 kernel |
| `verify` | serves the ORIGINAL FC2 output, then runs the D05 kernel into a shadow buffer.<br>A Triton compare counts mismatching words in written groups and touched words in dead groups, per FC2 weight. That covers **all 40 layers**, live routes, eager + graphs.<br>Report: `touch ~/dsv41-logs/d05-verify.trigger` at least 900 s after start, then read `~/dsv41-logs/d05-verify.<host>.<pid>.json` (status OK/MISMATCH, `weights_with_calls` should be 40).<br>Diagnostic only: extra FC2 + fill + compare per layer, so it is slower. Shadow 132 MB/rank. |

Optional: `DSV41_MOE_FC2_D05_NATIVE_DIR`, `DSV41_MOE_FC2_D05_JIT_DIR` (default `~/dsv41-stage/pd/d05/jit-cache`, seeded by `ops/prep-node.sh` with the per-rank cubin that the Codex v2a shadow compiled, `dsv41_fc2_live_store_v1.690c25bb…`).

## Files

`FILES` lists them. Install/uninstall: `common/pd-site.sh` (per node) or `common/pd-all.sh` (Mac, 8/8 identical-digest check).

| site file | live | staged |
|---|---|---|
| `fused_moe/experts/sm121_target_m32.py` | 723fdf8d | 5acc5fde |
| `fused_moe/experts/sm121_d05_verify.py` | absent | c04c363d (new) |

**Conflict:** `sm121_target_m32.py` is also the hook for pd-pdl (new native tree, which pins hashes here) and pd-fc2c (FC2 dispatch). Bundles need a merged file.

## Tests (stopped-service window)

1. `tests/test_d05_cpu.py`: pure-CPU dispatch test (mode parsing; flag 0 equals live call for call; only the target FC2 shape reaches D05; verify order). Passed locally and on spark-one.
2. `tests/test_d05_unit.py`: GPU, one rank per process. It refuses to run while dsv41-rank is active or the GPU is busy.
   - Real checkpoint weights of the rank's TP shard. All 40 layers across 8 ranks (`--layers all --layer-offset R --layer-stride 8 --tp-rank R`), or synthetic.
   - Tokens 1/4/6/48 (+97 fallback). Routes: valid, skewed, C1-like shared experts, half dead (ROUTE_LIVE), random holes, single-expert multi-tile, all dead.
   - Direct FC2 on real scatter32 layouts:
     - the written groups are bitwise equal to the original;
     - dead groups keep the sentinel;
     - dead inputs are poisoned;
     - the number of skipped groups is counted.
   - Whole `DeepGemmFP4Experts.apply` (COMPACT + LIVE_ACT as served), pooled workspaces poisoned (NaN bytes): eager, and graph capture + replay with changing routes. Both arms are compared bitwise to the control's eager output.
   - Fallback: M=97 never calls D05.
   - Verify-kernel negative test: one flipped bit in a live row is caught, and so is one store into a skipped group.
   - `d05_engaged`: `dsv41_fc2_live_store_v1` cubin compiled, D05 calls > 0, skipped groups > 0.
   - `--timing --rotate 4`: FC2-only graph timing ABAB.
   - Uses its own JIT dir (`jit-cache-test`).
   - Driver for all 8 ranks plus timing on spark-four: `ops/run-gpu-tests.sh`.

## Promotion plan (Base agent)

1. Install on 8/8 with the flag off (default). Behaviour is identical. `pd-all.sh install d05`.
2. Run `run-gpu-tests.sh` in a stopped window. Every rank must PASS.
3. Optional all-layer live shadow: `launch-conc.sh DSV41_MOE_FC2_D05=verify`, run load (C1/C4/C16 + 1M needle), touch the trigger, and require OK with 40 weights_with_calls on 8/8. Do not time this arm.
4. ABAB with pinned AV: `DSV41_MOE_FC2_D05=0` vs `=1` (`launch-conc.sh DSV41_AV_COST_JSON=<pin> DSV41_MOE_FC2_D05=1`).
   - exact.py must be identical under the same pin.
   - Gate: C1 at least 0.8% every pair, or conc16/32 at least 2%.
5. Rollback: `DSV41_MOE_FC2_D05=0` (env) or `pd-all.sh uninstall d05` (restores 723fdf8d from `.bak-pd-d05`).

Expected (Codex bounded numbers): isolated FC2 -20..-22% and routed MoE -7..-9% warm at M6-96. From FC2 step sums 3.1/5.2/9.7 ms at C1/C2/C4, that is about -0.6 / -1.0 / -1.9 ms per step (2-4%). Larger at C16/32.
