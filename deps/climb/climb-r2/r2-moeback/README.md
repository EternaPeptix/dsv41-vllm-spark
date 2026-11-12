# r2-moeback: MoE back end kernel levers (climb round 2)

Two default-off levers on the target routed-MoE back end (Athena FC2 overlay, gather, shared add):

| lever | flag (env, read once at import) | status | expected G1 (low / central / high) |
|---|---|---|---|
| **G25** (moe:L-G / glue:D) fold the routed+shared add into Athena's `_gated_gather` | `DSV41_FC2_GATHER_ADD=1` | **built**, CPU-proven, device test ready | +0.26 / +0.30 / +0.35 % |
| **G18 FC2 part** (moe:L-E) Athena FC2 PDL launch, weight staging before `griddepcontrol.wait`, SiLU early trigger | `DSV41_FC2_PDL=1` | **partial, fail-closed**: sources, build script, `vet-pdl`, device gate, installer ready; **binary not built** (cluster lock held through both sessions, window w1d until ~18:36Z); the installer cannot install a PDL binary until it is vetted (SASS) **and** device-gated, so until then `DSV41_FC2_PDL=1` refuses to boot (exit 97); SiLU-mul-quant fold **skipped** | +0.17 / +0.27 / +0.43 % |

Both: +0.43 / +0.58 / +0.78 % (they act on different boundaries, tail vs mid, and stack). Gains are G1' over
the brief's 119.1 baseline; with the 2026-11-08 launcher (08a249da, +3.8 %) the steps are shorter and the relative
gains are slightly larger.

Kit layout: `install.py` (one installer, both levers), `KIT.json`, `EXACTNESS.md` (op-order maps and proofs),
`device_test.py`, `tools/ptx_check_g25.py` (+ its rule `tools/ptx_rule_g25.py`), `tools/g18_gate.py` (G18 device
gate), `files/` (new module, PDL sources, `build_pdl.sh`), `base/` (live copies
the anchors were written against), `tests/` (CPU tests + outputs), `TEST-RESULTS.txt`.

## What the installer changes (8/8, `python3 install.py install | uninstall | status | dry | vet-pdl <dir>`)

| target | action | lever | base sha256 accepted |
|---|---|---|---|
| `vllm/model_executor/layers/fused_moe/runner/moe_runner.py` | 4 anchored edits, 10 tagged lines | G25 | 20b69fe5 (live) |
| `~/dsv41-fc2/fc2_status_gated_gather_trial.py` | 2 anchored edits, 6 tagged lines (after the `@triton.jit` kernel, so its lines do not move) | G25 | 733c3a87 (live), 4fc6960b (live + r1-pdl) |
| `~/dsv41-fc2/site/sitecustomize.py` | 1 anchored edit, 9 tagged lines | G18 | 733bdbe0 (live) |
| `vllm/.../fused_moe/runner/dsv41_moe_fold.py` | new module (imported only with G25 on) | G25 | — |
| `~/dsv41-fc2/athena_tp8_fc2_packed_graph_pdl_sm121.so` + `.sha256` | new binary, copied from `build/` ONLY when vetted and device-gated (below) | G18 | — |

Not touched: `~/dsv41-fc2/athena_fc2_forty_layer_trial.py` (launcher sha pin ae2089724597; its `transform_apply`
anchors into `DeepGemmFP4Experts.apply` still match, tested), `deep_gemm_moe.py`, `fp8_utils.py`, the launcher, the
live FC2 `.so`. Every inserted line carries `# r2-moeback-g25` or `# r2-moeback-g18`; backup `<file>.bak-r2-moeback`;
idempotent; uninstall is the exact inverse replacement (so a kit installed after us keeps its edits).
**Preflight:** every target's base sha is checked and every edit applied in memory (and the PDL gate decided) before
the first write; any mismatch refuses the whole install with nothing written (no partial install).

**PDL binary gate (G18).** `install` handles `build/athena_tp8_fc2_packed_graph_pdl_sm121.so` like this:
- no `build/` binary: G18 skipped with a warning;
- binary present and `KIT.json pdl_binary_sha256` null (unvetted) or different: **install refused**, nothing written;
- vetted (sha matches) but `build/DEVICE-G18.json` missing, failed, `--quick`-only, < 200 replays, < 8 graph cases,
  or for another sha: binary **not installed** (warning) and any stale installed copy + sidecar removed;
- vetted and device-gated: copied to `~/dsv41-fc2/` with a `.sha256` sidecar written from the vetted sha.
So a binary nobody has checked can never reach sitecustomize: with the flag on and no gated binary the process exits
97; with the flag off nothing reads it.

**Order with round-1 r1-pdl (the only kit sharing a file: the gather).** Install r1-pdl FIRST, then this kit;
uninstall LIFO (this kit first). Tested with both REAL installers (`tests/test_compose_r1.py` section 5): r1-pdl's
installer refuses a gather it did not write (nothing written), and r1-pdl's uninstall with this kit still on top
refuses (rc 1, every file untouched: it only restores its `.bak` onto its own exact patch). While both are installed,
`r1-pdl status` prints the gather as `TAGGED, NOT OUR EXACT PATCH (!)`; that is expected. After our uninstall the
gather is byte-identical to r1-pdl's patch again and its status is `PATCHED`.

## G25: `DSV41_FC2_GATHER_ADD`

**Mechanism.** Today every target layer runs, on the main stream: `_gated_gather` (3.40-3.53 us) -> gap 0.55 us ->
`CUDAFunctor_add<bf16>` (shared + routed, 1.28-1.34 us) -> gap 0.72 us -> `ar_kernel_pl`. With the flag on:
- `MoERunner._apply_quant_method` arms `dsv41_moe_fold` with the shared expert's output tensor (peeked from
  `_output[idx]`, not consumed) and, on the aux-stream path, its `_output_ready_event`, but only if every runner step
  between the gather and the all-reduce is the identity (scale 1, no transforms, no SP/PCP/naive-DP, not pre-reduced,
  no DBO).
- The Athena gather (`gated_gather`) asks `take(output)`: if armed and the shapes match, the current stream waits on
  the shared event and `_gated_gather_add` runs instead: the same gather source plus one load of the shared row block
  before the k loop and the epilogue `bf16(f32(bf16(acc)) + f32(shared))`.
- `finish()` tells the runner it folded; `MoERunner.forward` uses the routed output as the all-reduce input and skips
  the add. The shared output is still consumed by `.output` exactly once (the prefill version's assert cannot fire).
- Anything else (drafter MoE, stock gathers, a declined shape) runs the old path.

**Flag off.** `_DSV41_FC2_GATHER_ADD` is a module-level False in both files; the three runner hooks are
`if False:` blocks, the original add moved into an `elif` that runs exactly as before; the gather hook is an
`if False:` before the unchanged `grid = ...` / `_gated_gather[grid](...)`. Nothing new is imported. Tested: AST check
that every tagged statement is the flag definition or flag-gated, and a CPU run of the real runner/shared-experts/
gather code (stubbed vllm, emulated kernels, logged stream/event/op order): flag-off logs and AR inputs identical to
the base files in all scenarios.

**Flag on, bit-exact.** See `EXACTNESS.md`. Per element the accumulation and its bf16 rounding are the same source
lines; the add is one fp32 RN then RNE to bf16 exactly like torch's add. Exhaustive proof over all 4.26e9 non-NaN
bf16 pairs that every lowering Triton may emit (add.f32 + cvt.rn, or native add.bf16x2) equals torch's
CUDAFunctor_add<bf16> + c10 RNE; end-to-end emulation (9 hostile input kinds) OLD == NEW on every element, and the
wrong variant (shared added into the fp32 accumulator) is detected.

**Races.** One dependency added (gather waits on the shared expert's ready event, the same edge the add had), none
removed; shared tensor lifetime extends past its last device read; one allocation fewer; graph capture records the
same event edge on the gather instead of the add. Shared-expert slack before the gather start: median 51-88 us,
minimum 18 us over all 12 traces (verify-G25 notes), so the earlier wait never stalls.

**Expected saving (trace evidence, verify-G25 on 512S1, all 12 windows, both ranks).** Gather end -> AR start is
2.59 us per layer today (gap 0.555 + add 1.31 + gap 0.72, p10-p90 within 0.07 us). After the fold: one gap
(0.55-0.74 us) plus 0.02-0.35 us of gather growth from the extra independent load = 1.50-2.02 us saved per layer x 40
layers = **0.060 / 0.071 / 0.081 ms per step** on the main stream, on the critical path (the AR is at its 26 us
floor and starts on arrival; ranks shift together) -> **G1 +0.26 / +0.30 / +0.35 %**.

## G18: `DSV41_FC2_PDL` (Athena FC2 PDL early-B)

**Mechanism.** Mid section today: FC1 -> gap 0.58 -> `_silu_mul_quant_fp8_packed_kernel` 3.06-3.26 us -> gap
0.56 us (never below 0.51 us, 0/2400 negative: the Athena FC2 is a plain `<<<grid(80, M/32), 128>>>` launch) ->
Athena FC2 (7-8 us fixed cost + 3.3 us/expert). With the flag on:
- `sitecustomize.py` (before the overlay installs) checks `~/dsv41-fc2/athena_tp8_fc2_packed_graph_pdl_sm121.so`
  against its `.sha256` sidecar (mismatch or missing -> exception -> the existing `os._exit(97)`), points
  `ATHENA_FC2_GRAPH_SO` at it (inside `~/dsv41-fc2`, so the overlay's root check passes; the overlay loads it under
  the same module name), and sets `DSV41_MOE_PDL_TRIGGER=start` unless the env already sets it.
- The PDL binary is the live binding + kernel with two changes only: the launch is `cudaLaunchKernelEx` with
  `cudaLaunchAttributeProgrammaticStreamSerialization` (same grid/block/smem/stream/args), and the kernel executes
  `griddepcontrol.wait` after `stage_288_fc2_64` (weights + scales into shared memory) and before the first read of
  A / any store.
- The SiLU kernel's existing constexpr trigger (`gdc_launch_dependents` at its start) lets FC2's CTAs launch while
  SiLU runs: they read the layout, stage their 9.2 KB weight tile, then wait for SiLU to complete.

**Flag off.** sitecustomize's tagged block is `if <env> == "1":` and nothing else changes; the overlay loads the
live `.so` as today; `DSV41_MOE_PDL_TRIGGER` unset -> SiLU constexpr 0 (unchanged kernel).

**Flag on, bit-exact.** Same arithmetic instructions in the same order (EXACTNESS.md table); pre-wait reads are
only data final before SiLU starts (layout from the scatter kernel three kernels back, constant weights); the only
pre-wait global write is the structural-error status atomicOr; `out` (aliasing FC1's output buffer workspace2, which
SiLU reads) is written only after the wait. `build_pdl.sh` proves the toolchain reproduces the live SASS (control
build) and diffs control vs PDL SASS; `install.py vet-pdl` turns that into a machine check before any sha is recorded.

**Expected saving.** Removing the SiLU->FC2 launch gap and overlapping FC2's weight ramp with SiLU: 1.0-2.5 us per
boundary (verify-G18; the closest measured analogue pd-pdl, FC2 early-B on the old deep_gemm FC2, gave 0.04-0.10 ms
per step) x 40 boundaries = **0.04 / 0.064 / 0.10 ms** on the main stream, on the critical path (side-stream work in
that window is 0.4-0.6 us; the shared expert ends >= 2 us before FC2 starts) -> **G1 +0.17 / +0.27 / +0.43 %**.

**Skipped: SiLU-mul-quant fold into the FC2 prologue.** Needs a CUDA re-implementation of the Triton lowering bit for
bit (div.full, ex2/lg2.approx, the bf16 round-trip of y, satfinite e4m3, exponent clamp) inside a rewritten FC2 A
path, computed redundantly by 80 CTAs per segment; not provable on the CPU, multi-week, new failure surface. Not
shipped (see EXACTNESS.md).

**Building, vetting and gating the binary (NOT done yet: CLUSTER_LOCK was held through both sessions; at
14:54Z window w1d holds it until ~18:36Z).** Three steps, all fail-closed:
1. Build on **spark-eight** (the script refuses any other host, exit 4), outside a window, or inside the lead's own
   window with its token (the script refuses while the lock exists unless `R2MB_LOCK_TOKEN` equals its `token=`):
   ```
   cd <kit copy on spark-eight>
   R2MB_LOCK_TOKEN=<token, only if you hold the lock> \
     bash files/athena-pdl/build_pdl.sh files/athena-pdl ~/climb-scratch/r2-moeback/build
   ```
   It builds a control `.so` from the unmodified live sources and the PDL `.so` with the live build.ninja flags (nice
   19, one job), dumps the whole SASS of live/control/PDL (`ins_*.txt`), prints `IDENTICAL` for live vs control and the
   control vs PDL diff, hashes all three into a fresh `COMPILE-LOG.txt` (an older log is rotated), deletes the objects.
2. Vet: `python3 install.py vet-pdl ~/climb-scratch/r2-moeback/build`. It refuses (KIT.json unchanged) unless:
   live == control SASS exactly; the control has no `ACQBULK` and the PDL build has at least one (`ACQBULK` is
   `griddepcontrol.wait` on sm_12x, as in production's sm_120 quant kernels); with `ACQBULK`/`NOP` removed and branch
   targets normalised, PDL == control (`STRICT`) or at least the same opcode multiset (`SCHEDULE-ONLY`: the
   `"memory"` clobber may reorder or re-register); no global store or non-OR atomic before the first wait in listing
   order; the live `.so` the log hashed is production's 85d888ae; the `.so` on disk is the one the log hashed. On
   success it copies the `.so` + `ins_*` + `COMPILE-LOG.txt` into the kit's `build/` and records
   `pdl_binary_sha256` and `pdl_vet` (mode, host, counts, log sha) in `KIT.json`.
3. Device gate, on a stopped node: `CUDA_VISIBLE_DEVICES=0 ~/venvs/vllm032/bin/python tools/g18_gate.py --kit .`
   runs `device_test.py --part g18 --reps 200` (eager bitwise + 8 graph cases x 200 replays of the real SiLU with
   the early trigger -> PDL FC2 with the aliased workspace2), then `compute-sanitizer --tool racecheck` and
   `--tool synccheck` (`--error-exitcode 99`) on `--part g18 --quick`, and writes `build/DEVICE-G18.json` (+ logs).
   Copy `build/` back into the master kit; `install` then copies the binary 8/8.

## Tests (CPU; outputs in `tests/`, summary in `TEST-RESULTS.txt`)
- `tests/test_install.py` (stdlib, 67 checks): anchors apply exactly once to the live base files; patched files compile; inverse
  edit restores the base bytes; every changed line tagged; install/second install/status/uninstall on a temp tree;
  unknown base refused; a later kit's edit survives our uninstall; AST flag-gating; trial overlay untouched and Athena
  `transform_apply` anchors still apply to the live `DeepGemmFP4Experts.apply`; `_gated_gather` kernel lines 1..51
  identical and unmoved; sitecustomize off / on / trigger override / sha mismatch -> 97 / missing .so -> 97 /
  off with missing .so boots. Review fixes: preflight refuses a bad base on the LAST target with nothing written;
  the PDL gate (null sha -> refused, wrong sha -> refused, no/failed/quick/other-sha stamp -> skipped and a stale
  installed copy removed, vetted + gated -> installed with a matching sidecar); `vet-pdl` on synthetic SASS lists
  (STRICT and SCHEDULE-ONLY accepted; live != control, extra or changed opcode, no wait, wait in control, store
  before the wait, unpinned live `.so`, `.so` not the hashed one: all refused with KIT.json unchanged).
- `tests/test_ptx_rule.py` (stdlib, 16 checks): the G25 PTX rule accepts the two proven epilogues (A: 16
  `cvt.rn.bf16x2.f32` + 16 fp32 add elements; B: 8 `cvt.rn.bf16x2.f32` + 16 bf16 add elements) and rejects the
  known-wrong lowering (shared folded into the fp32 accumulator: 8 cvt + 16 fp32 adds), its init variant, mixed
  epilogues, extra fma/mul, `.ftz`, and a live census that is not production's.
- `tests/test_plan_g25.py` (torch CPU): the real runner, SharedExperts and gather code with stubbed vllm, emulated
  kernels and logged stream/event/op order: flag off == base; flag on gives identical AR-input bits and the expected
  op log (gather+add -> wait+gather_add; no wait on the no-overlap path); declines for drafter experts, scale != 1,
  pre-reduced output, padded width; second gather raises.
- `tests/test_compose_r1.py` (20 checks): only r1-pdl touches one of our files; r1-pdl's edits then ours and ours
  then r1-pdl's give the same text; text-level removal of either kit's edits leaves the other's; with both flags
  off the gather executes the base statements. Section 5 runs both REAL installers on a temp tree with r1-pdl's six
  bases + our three: LIFO round trip ends byte-identical to base; r1-pdl after us is refused with nothing written;
  r1-pdl's uninstall while we are on top is refused with nothing changed.
- `tests/emul_add_exhaustive.py`, `tests/emul_gather_fold.py`: the exactness proofs above.
- `device_test.py --plan` builds every device case on the CPU (no kernels).

## Device test (lead, stopped node)
```
CUDA_VISIBLE_DEVICES=0 ~/venvs/vllm032/bin/python device_test.py --kit . --part g25        # ~2-4 min
CUDA_VISIBLE_DEVICES=0 ~/venvs/vllm032/bin/python tools/g18_gate.py --kit .                 # G18: needs the vetted build/…pdl….so;
                                     # device_test --part g18 (200 replays) + racecheck + synccheck --quick, ~25-60 min
CUDA_VISIBLE_DEVICES= nice -n 19 ~/venvs/vllm032/bin/python tools/ptx_check_g25.py . ~/climb-scratch/r2-moeback/ptx   # no GPU; compiles: spark-eight, no lock
```
G25 compares the live gather + torch add vs `launch_gather_add` (bytes, NaN positions) for M = 1..24, 32..96, 1500,
2048 over 9 input kinds, eager and CUDA-graph replays with new inputs and poisoned outputs, and times both (warm
in-graph and cold-L2). G18 compares the live vs PDL FC2 `.so` eager (out + status bytes) and, with the real SiLU
kernel + trigger as primary and the production workspace2 aliasing, >= 200 graph replays against an eager live
reference, plus timing of the SiLU+FC2 pair. Exit 0 only if every case is bit-identical. `ptx_check_g25.py` now
requires exactly one of the two proven epilogues (rule in `tools/ptx_rule_g25.py`), so the known-wrong lowering
fails it too, not only the device test. racecheck/synccheck see intra-kernel shared-memory and barrier hazards only;
the cross-kernel PDL ordering is covered by the bitwise graph replays.

## Window arms (env per arm; install 8/8 first, flags-off relaunch must give exact.py ec2f4437f8e1ed09 and digest 2f7fdccbd388225a)
- control: nothing
- `g25-gather-add`: `DSV41_FC2_GATHER_ADD=1`
- `g18-fc2-pdl`: `DSV41_FC2_PDL=1` (needs the vetted + device-gated binary installed; implies `DSV41_MOE_PDL_TRIGGER=start`)
- diagnostic only: `DSV41_FC2_PDL=1 DSV41_MOE_PDL_TRIGGER=0` (PDL without the early trigger)
- `g25+g18`: both

## Reading the effect in nsys (rank 0 and 4, delay-200 windows, 40 boundaries per step)
- **G25:** `_gated_gather` disappears and `_gated_gather_add` appears, 40 per step on the target (drafter MoE layers
  keep their own gather + add). `vectorized_elementwise_kernel<4, CUDAFunctor_add<c10::BFloat16>>` instances on the
  main stream drop by 40 per step. Per layer, `_gated_gather_add` (expect 3.4-3.9 us) is followed directly by
  `ar_kernel_pl` with one ~0.55-0.74 us gap; measure gather end -> AR start: 2.59 us today, expect ~0.6-1.1 us
  (gather growth included). Check no `_gated_gather_add` start waits on the aux stream: the last aux-stream kernel
  (shared down-projection GEMM) should still end well before the gather starts (today >= 18 us). Layers 19 and 39 run
  on another stream with the same pattern.
- **G18:** the Athena FC2 kernel (`athena_tp8_fc2_serial32packed_k288_candidate`) now starts before
  `_silu_mul_quant_fp8_packed_kernel` ends: the SiLU -> FC2 start gap goes negative in most of the 2400 boundaries
  (today min 0.51 us, 0 negative). FC2's own duration grows (it includes the wait), so measure FC1 end -> FC2 end
  (today 4.2-4.4 us mid section + FC2): expect 1-2.5 us shorter per boundary. With `DSV41_MOE_PDL_TRIGGER=0`, FC2 can
  only start at SiLU's end, so most of the gain should vanish (diagnostic arm).

## Risks
- G25: the fold kernel is a new Triton JIT kernel (compiled at the first eager forward, like the live gather); its
  codegen is checked by `ptx_check_g25.py` and bit-tested by `device_test.py`. Composition with r1-pdl's gather PDL:
  our path returns first (no PDL on the fold kernel); the two do not fully stack. Status-failed steps still produce a
  zero routed contribution and are fail-stopped by the existing barrier, unchanged.
- G18: PDL in CUDA graphs with a Triton primary (programmatic edge SiLU -> FC2); the analogue (pd-pdl, deep_gemm
  FC2) was clean under racecheck/synccheck and 200 replays; still to be shown for the Athena kernel by the device
  test. If the binary is missing or differs from its sidecar, `DSV41_FC2_PDL=1` refuses to boot (exit 97) rather than
  serve with the wrong FC2; flag off never touches the binary.
- G25 assumes `MoERunner.forward` runs as plain Python (live: compilation mode NONE, `enforce_eager=False`).
  Under torch.compile/Dynamo the `self.__dict__.pop("_dsv41_g25_folded")` decision would be traced once with the
  fake op and the shared output could be added twice: **re-review G25 if the compilation mode ever changes.**
- With G18 on, sitecustomize hashes the PDL `.so` once per Python process (each worker/subprocess); a few ms,
  kept as the fail-closed provenance check.
- Neither lever changes verify width, drafts, tiles, tactics or reduction orders.
