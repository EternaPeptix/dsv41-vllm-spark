# pd-pdl: PDL early weight loads for the FP4 FC2 + SiLU trigger (round-3 #2, arm B, FC2 part)

Status 2026-10-25 (offline prep): staged on 8/8, NOT installed. Trees built and identical on every node.
- **Done:** CPU test PASS; nvcc compile check PASS.
- **Not run:** GPU tests.
- **Not implemented:** the FC1 prologue and the glue-kernel PDL arm A (see "Not done").

## Change

1. **FC2 early-B prologue** (`tools/make_tree.py`, a JIT-header patch with no native rebuild).
   - It produces new trees from the immutable ones, using the same `.so` binaries:
     - `native-base`: b1c4eff8 + patched header, manifest 5d5f2f70;
     - `native-d05`: 6fea5091 + patched header, manifest 2009e011.
   - The patch is selected at compile time only for the target FC2 instantiation (MGroupedContiguous, E384, N5120, K384, FP4 B K-major, split 1).
   - The TMA leader walks a private copy of the persistent scheduler. It issues the B (packed FP4 weights) and SFB TMA loads for up to 7 stages (the first 2⅓ tiles), each stage with its full `arrive_and_expect_tx`. Then it runs `griddepcontrol.wait`, and only then issues A/SFA.
   - Pre-wait it reads only weights, weight scales and `grouped_layout`. `grouped_layout` is written by ep_scatter before FC1 runs.
   - Math warps keep the original wait. A full barrier completes only after all four loads.
   - Offline nvcc check on the real JIT `kernel.cu` with the JIT's own flags (`ops/cc-check.sh`):
     - FC1 SASS is **identical** (1282 lines);
     - FC2 and D05-FC2 SASS differ: the prologue adds 2 TMA ops (1234->1378 and 1314->1458 lines);
     - 168 regs and 0 spills everywhere.
2. **SiLU trigger** (`fp8_utils.py` 8dc7dab8 -> b35c0561).
   - A new `PDL_TRIGGER` constexpr (default 0 keeps the Triton code identical) calls `tl.extra.cuda.gdc_launch_dependents()` at kernel start (`start`) or after the row loop (`end`).
   - That lets the PDL-launched FC2 grid begin its weight prologue while SiLU runs.
   - Only for calls with M ≤ `DSV41_MOE_PDL_TRIGGER_MAX_ROWS` (12480 = M32 max), so prefill is untouched.
   - Without the trigger, FC2 can launch only after SiLU exits, and the early-B prologue hides almost nothing.

## Flags

All default off. The M32 wrapper half lives in **pd-d05's** `sm121_target_m32.py` (5acc5fde), so install d05 too.
- `DSV41_MOE_PDL_EARLY_B=1`: the M32 path loads `pd/pdl/native-base`, plus `native-d05` when `DSV41_MOE_FC2_D05=1`.
  - Their JIT caches start cold (`pd/pdl/jit-cache-base|-d05`), so the first start re-JITs FC1/FC2/transpose: expect about 2-3 min more.
- `DSV41_MOE_PDL_TRIGGER=start|end`.
- Arms: EB = `EARLY_B=1 TRIGGER=start`; EBe = `EARLY_B=1 TRIGGER=end`; T only = `TRIGGER=start` (measures SM stealing alone); plus the same with `DSV41_MOE_FC2_D05=1`.

## Tests (stopped window)

1. `tests/test_pdl_cpu.py`: placement, constexpr default, and staged == live after removing the pd-pdl blocks. PASS.
2. `tests/test_pdl_unit.py`: GPU.
   - Arms: ctl (live tree, no trigger), pdl with trigger 0/start/end, and pdl+d05 with trigger start.
   - Whole `DeepGemmFP4Experts.apply` (COMPACT + LIVE_ACT as served), tokens 1/4/6/48, routes valid / C1-shared / half-dead.
   - Eager, then CUDA graph with **200 replays, each compared bitwise**, because PDL races are timing-dependent.
   - Then run compute-sanitizer `racecheck` and `synccheck` once on `--tokens 6 --replays 2` (commands in the docstring).
3. Serving ABAB with pinned AV and 3 or more pairs (each arm is near the noise floor). Count at least 10 clean startups for the rdma_ar capture-hang interaction.

Expected (kernel-ideas-3, skeptic-corrected): FC2 arm -0.12..-0.24 ms/step at 1u. A trigger at start may steal SMs from the SiLU tail: an FC2 CTA uses about 64.5K registers, so it cannot co-reside. Hence the `end` variant.

## Not done (documented next steps)

- **FC1 prologue.** The predecessor chain is ep_scatter_2 -> (transpose) -> FC1, and its pre-wait `grouped_layout` read would need either proof that the scatter is complete or trigger-before-wait ordering. Not safe to do blind.
- **Arm A glue PDL** for ar_kernel, topk, count/scatter/gather, wo_a, mg_dual, q_kv_norm and skinny: Triton `launch_pdl` + `gdc_wait` per kernel. That is 7 site files and the `.so`s. Estimated -0.13..-0.2 ms. It conflicts with every other file owner. A separate window.
