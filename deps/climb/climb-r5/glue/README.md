# r5-glue: RQPAD and RPSPLIT (climb round 5)

This kit makes two exact kernel-shortening levers for glue kernels that sit on the critical path. Both are
default-off flags. The kit was built against production launcher **166d7aac**, and every estimate below is re-based
on today's **t2** traces.

| flag | lever | what changes |
|---|---|---|
| `DSV41_GLUE_RQPAD=1` | RQPAD | `_r2_reduce_quant` (dsv41_attnout.py:272) zeroes the UE8M0 padding rows with one store per row. The new kernel does it in one masked `[128, 8]` store. |
| `DSV41_GLUE_RPSPLIT=hoist` | RPSPLIT (hoist) | `_dsv41_rp_kernel` loads the 5 token-row chunks right after `gdc_wait`, before the plan. In the r2 kernel those loads were serialised behind the plan and behind each other's stores. |
| `DSV41_GLUE_RPSPLIT=split` | RPSPLIT (split) | Grid `(M*6, 5)`: one quant chunk per program. The route's other duties stay in the ch-0 program. |

Status:

- **Built, then fixed after review** (REVIEW.md, "Fixes"). The CPU tests pass (see TEST-RESULTS.txt).
- **Compile check, interpreter run and device test: not run.** `CLUSTER_LOCK` was held from 05:32Z to about 11:32Z on
  2026-11-09, during the whole build and the fix round ("steering: persist vector capture..."), and I have no GPU.
  `tools/compile_check.py` now compiles in production's specialization (review B1); its logic is tested with a fake
  Triton (`tests/test_compile_check_logic.py`), but the real compile has not run.
- **Gates before any arm:** `tools/run_spark8_checks.sh` with the lock absent, then `device_test.py` exit 0 on a
  stopped node, then `./install-all.sh install` (8/8 agree).

## 1. Re-base on t2 (today's production, 166d7aac)

Source: `scripts/t2_glue.py` (run on 512S1 with huntA's `walkB` critical-path walker; output in `scripts/t2_glue.txt`).
p50 values are the mean of rank 0 and rank 4. "On path" means on walkB's path; all 43 rq and all 40 rp calls per step
are on it in every workload.

**`_r2_reduce_quant` (grid 4 x M):**

| M | code n/st | prose n/st | maths n/st | p50 µs | r2 pad-loop trips (r=0) |
|---|---|---|---|---|---|
| 1 | – | 0.67 | – | 8.69 | 127 |
| 2 | – | 22.0 | 0.67 | 5.22–5.44 | 63 |
| 4 | 24.0 | 17.33 | 16.67 | 4.13–4.19 | 31 |
| 5 | 3.0 | 3.0 | 3.0 | 3.46 | 25 |
| 6 | 16.0 | – | 22.67 | 3.49–3.62 | 21 |

- The successor (wo_b, PDL-launched) starts 2–7 µs **before** rq ends: succ-gap p50 is −2.1 to −6.9 µs. It then
  waits in `griddepcontrol.wait` for rq to complete.
- So every µs cut from rq's tail is cut from the path. On-path rq time is 175 µs/step for code, 214 for prose and
  171 for maths.
- A fit to t2 gives t(M) ≈ 2.03 µs + 52.5 ns × trips. That matches M=1/2 exactly and M=4/5 at p10.

**`_dsv41_rp_kernel` (grid M*6):**

- 40 calls per step in every workload, all on the path: grid 24/36 for code, 6/12/24 for prose, 12/24/36 for maths.
- p50 is 7.94–8.29 µs, flat at every grid. The kernel is a single-program latency chain, not throughput-bound.
- FC1 starts 0.38–0.45 µs after rp ends (p50), so rp's end gates FC1. On-path rp time is 326–337 µs/step.
- For calibration, r2-moefront predicted 2.5–4 µs for this kernel. It measures 8.

## 2. Mechanism

**RQPAD.** Program `(pid_c, r)` must write UE8M0 zero for the padding rows `M+r, M+2r, …` below `PADDED_M = 128`.
The r2 kernel runs a runtime loop with one 8-byte store per row: 127 trips at M=1 and 63 at M=2, about 50 ns each.
The kit computes all of these rows as a `[128]` vector and stores the `[128, 8]` zero tile once, masked by
`p < PADDED_M`. That is one trip for every sonar M ≤ 32. Every other statement is the r2 kernel verbatim.

**RPSPLIT.** Each r2 program is one route, and it runs the whole chain:

1. `gdc_wait`
2. the topk-id load, then histogram, cumsum and 4 block reductions
3. **5 ×** [load an `[8,128]` bf16 chunk of x, then amax, div, encode, store bytes, pack through shared-memory
   layout conversions, store words]
4. the dead-row fill

The loads are Triton inline-asm `ld.global` with side effects, so neither LLVM nor ptxas moves a chunk's load above
the previous chunk's stores: x and the slot rows could alias as far as the compiler knows. The five load latencies
therefore sit in series, after the plan.

- **hoist** issues the 5 loads right after `gdc_wait`. They overlap the plan and each other.
- **split** gives each chunk its own program. The critical path becomes `max(plan, one load) + one chunk`. That costs
  5× the CTAs (at most 300 at M=10), each recomputing the integer plan.

Which is faster is unknown until the device timing (`device_test.py` T). Pick the winner for the `glue` arm.

## 3. Flag off is identical

The edits to the two r2 modules are 4 tagged lines each, appended after the last function: one `os.environ.get`, plus
`if flag: import kit module; rebind _launch_rq / _launch = wrap(...)`. `tests/test_install.py` checks:

- stripping the tag gives back the exact live bytes;
- the flag expression is False for unset, `0`, `off`, `false` and `''`;
- with the flag off, executing the hook imports nothing and rebinds nothing.

The kernels, launches, allocations, streams and graph contents are those of today. The new modules are never
imported when the flag is off.

## 4. Flag on is bit-exact

See EXACTNESS.md: the op map element by element, the write-set argument, and PDL ordering. In short:

- every floating-point statement is the live statement, checked statement by statement by `test_ast_exact.py`;
- only load issue points (RPSPLIT), the program a chunk runs in (split), and the shape of a constant-zero store
  (RQPAD) change;
- `test_writesets.py` shows the same write set for each, with every byte written once;
- `device_test.py` checks every byte against the live kernel under two poisons at every gated M, and requires every
  must-fail mutant to be caught at every gated M.

**Runtime self-check (defence in depth).** Each check runs the r2 kernel and the kit kernel on poisoned copies and
compares every output byte.

- **Synthetic sweep (review N2), once per process, at the first eager call:** RQ at every sonar M = 1..32 on hostile
  partials; RP at every M = 1..10 × 2 routings ("spread" distinct experts; "edge" experts 0 and 383 only, segments
  longer than one 32-row block) on hostile x, with the call's ids dtype. Today's rank0.log shows the first eager
  route-prep and attnout calls at M=2, after which the 45 graphs are captured. The served decode sizes (M=4/6) therefore
  run only inside graphs, and without the sweep they would never be checked on the rank. With it, the 8/8 boot INFO
  lines ("… bit-identical … at every M … (synthetic sweep") cover every M a graph can contain. It costs about 64 (RQ)
  and 40 (RP) launches once, plus host syncs, in the eager profile run.
- **Real-data check:** the first eager call at each M also compares both kernels on the call's own inputs.
- On a mismatch or an exception, the lever disables itself for the process: it logs `ERROR … DISABLED for this
  process` and the r2 kernel serves.
- A capture uses the kit kernel only after the sweep passed and no check failed.
- RQPAD is also covered by r2-attnout's own checks: its first-call synthetic M = 1, 5, 16, 32 check against
  flashinfer goes through the rebound `_launch_rq`.

## 5. Races and CUDA graphs

- No new streams, events or syncs on the serving path. Host syncs happen only in eager self-checks, which are never
  inside a capture, as in r2-attnout.
- **RQPAD:** the same two `torch.empty` allocations in the same order (test_ast_exact `_alloc`), the same grid, PDL
  flags and options.
- **RPSPLIT:** no allocations. One launch as before; the grid gains a second dimension in split mode.
- PDL semantics are unchanged: the rq primary triggers at its start, rp is the secondary of top-k and waits before
  any access, and FC1 / wo_b synchronise on full grid completion.
- `test_wrappers.py` checks the capture policy (CPU, fake kernels): a capture before any check runs r2, and a mutant
  that is wrong only at M=6 is caught by the sweep when the first real call is at M=2.
- `device_test.py` RQ-P / RP-P put a PDL producer in the stream (and in the graph) that lets its dependents start at
  once and writes the kernel's input late. The kit must match r2 there, and a mutant that issues its loads before
  `griddepcontrol.wait` must be caught (review N6.1).

## 6. Expected saving

From `scripts/estimate.txt`, with production step times from the brief.

**RQPAD:**

| new kernel time | code µs/step | prose µs/step | maths µs/step | G1 |
|---|---|---|---|---|
| 3.2 µs | 30 | 69 | 25 | +0.18% |
| 2.5 µs | 60 | 99 | 55 | +0.32% |
| 2.1 µs | 77 | 117 | 72 | +0.40% |

Correction (review N7). In t2, wo_b (the PDL successor) launches `off` = 1.87 / 1.40 / 1.37 / 1.22 / 1.37 µs after rq
starts (M = 1/2/4/5/6; p50 + succ-gap), and it still has to run its pre-wait prologue P. So the on-path floor is
`max(T_new, off + P)`, not T_new (`scripts/estimate.py`):

| T_new | P = 0.5 µs | P = 1.0 µs | P = 1.5 µs |
|---|---|---|---|
| 2.5 µs | +0.32% | +0.32% | +0.25% |
| 2.1 µs | +0.40% | +0.35% | +0.25% |

Stated range: **0.1 / 0.25 / 0.4 %** (round 4 said 0.15). The device timing measures the kernel alone and cannot see
P; only the window can. Prose gains about 2× the others because it runs 22 M=2 calls per step. On its own, RQPAD sits
below the +0.3% promotion bar, and w5 control noise was about 0.2%. The `glue` arm is therefore the one expected to
read cleanly.

**RPSPLIT:** 40 calls per step on the path. Each 1 µs saved per call is +0.19% G1.

| variant | low / central / high G1 | assumed saving per call |
|---|---|---|
| hoist | 0.05 / 0.25 / 0.6 % | 4 load latencies, about 0.4–0.8 µs each |
| split | 0.1 / 0.35 / 0.8 % | 4 chunk passes, about 0.8–1 µs each, minus CTA contention |

These are uncalibrated: r2-moefront predicted 2.5–4 µs for this kernel and it measures 8. The centrals are cut
accordingly.

**Kit total** (RQPAD + the better RPSPLIT): **0.15 / 0.55 / 1.15 %**. The levers are independent: different kernels,
different layers of the step.

## 7. Risks

1. **Contention in split mode.** 5× the CTAs, each running a shared-memory histogram over 512 bins, may lengthen
   per-program latency. The device timing decides between hoist and split.
2. **Graph capture before an eager call.** If production's first rq/rp call happens inside a capture, the kit is inert
   for that graph (r2 runs). That is exact but gives no gain. The boot log shows it: no sweep INFO line.
3. **Graphs at sizes the eager run never saw.** This is the normal path: the eager profile run is at M=2 and the
   decode M=4/6 are served from graphs. The synthetic sweep checks every in-scope M on the rank before any capture
   (review N2), so a graph only ever contains a kit kernel that this process has proven at that M. A later eager
   real-data failure disables the lever for new launches, but graphs that are already captured keep the kit kernel
   (the same policy as r2-attnout). With the sweep in place, that needs an input-dependent difference that neither
   the device test nor the sweep found.
4. **Parent-kit re-install** drops the hooks. The flags then do nothing, and `status` prints `HOOKS-LOST`.
5. **Register pressure (hoist).** The hoist holds 5 × 8 bf16 per thread across the plan. compile_check requires no
   STL/LDL in the SASS and LOCAL 0 in `cuobjdump -res-usage` (review N1: `n_spills` is "?" in this Triton build).

## 8. Reading it in nsys

- RQPAD: kernel `_r5g_reduce_quant` replaces `_r2_reduce_quant`. Expect p50 of about 2.1–3.2 µs at every M, flat,
  instead of 8.7 / 5.3 / 4.2 / 3.5 µs. Use `scripts/t2_glue.py` with `NAMES` extended.
- RPSPLIT: kernel `_r5g_rp_kernel` replaces `_dsv41_rp_kernel` at M ≤ 10. Grid `(M*6, 5)` in split mode or
  `(M*6, 1)` in hoist mode. Expect p50 of about 4–6 µs.
- In both, the gap from the successor's start (wo_b / FC1) to the kernel's end should stay the same; only the
  kernel shortens.

## 9. How to run

Install on 8/8, in the window, after the parent kits, with the 8/8 runner (review N3):

    ./install-all.sh status | preflight | install | uninstall

It probes every node read-only and computes the verdict here: CLEAN, INSTALLED, HOOKS-LOST, FOREIGN or STALE-TMP.
`status` exits 0 only when 8/8 agree. `install` stages the kit in `~/dsv41-stage/climb-r5/r5-glue` and installs node
by node with a probe after each. A failure rolls back every touched node. On one node, `python3 install.py dry |
install | status | uninstall` still works, and `status` must print `OVERALL INSTALLED`.

- Uninstall r5-glue before r2-attnout or r2-moefront, and never re-run their `install` on top of it.
- While r5-glue is installed, `climb-r2/install-all.sh status` reports FOREIGN for `dsv41_attnout.py` and
  `dsv41_routeprep.py` on 8/8 and exits 2. That is expected; use `./install-all.sh status` for these files.
- The relaunch preflight manifest does not cover these files, so run `./install-all.sh status` before each relaunch
  with an r5-glue arm.
- Lines tagged by another kit (e.g. r1-pdl) in the two modules are refused (review N4). A future rebase onto these
  kernels needs a new r5-glue base.

**Compile and interpreter gate** (from this Mac, only when `CLUSTER_LOCK` is absent; spark-eight, nice 19, scratch
in `~/climb-scratch/r5-glue/`; the runner sources the mandatory environment):

    bash tools/run_spark8_checks.sh

Results go to `results/compile.log`, `results/interp.log` and `results/ptx/`. compile_check records the kernel
arguments through the real launch functions (production shapes and strides). It derives the specialization with
Triton's own launch binder and asserts it (tt.divisibility 16 on every pointer and on x_row_stride, out_row_stride,
mn_aligned and M_sum; M constexpr 1 at M=1; r2 spec == kit spec). It then checks that the x-load bytes before the
histogram exceed r2's by at least 80 (hoist) or 16 (split), in PTX and in SASS.

**Device test** (a stopped node, mandatory environment; the script asserts it and exits 2 otherwise). It reads the
serving env from `~/dsv41-tp8.sh` (sha 166d7aac) and the geometry from the model config. Expected runtime is
8–15 min; `--quick` takes about 4 min and still runs every mutant at every gated M:

    ssh spark-eight; source ~/venvs/vllm032/bin/activate; export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
    cd ~/climb-scratch/r5-glue && python3 device_test.py [--quick] [--part rq|rp|hook|all]

**Window arms** (C-T-C-T on top of production, each with exact.py, G1 digest and 9 long prompts):

| arm | env |
|---|---|
| `rqpad` | `DSV41_GLUE_RQPAD=1` |
| `rps` | `DSV41_GLUE_RPSPLIT=split` |
| `rph` | `DSV41_GLUE_RPSPLIT=hoist` |
| `glue` | both, using the device-test winner for RPSPLIT (primary arm) |

**Boot assertion, counted 8/8:** the sweep INFO lines `[r5-glue] DSV41_GLUE_RQPAD=1: _r5g_reduce_quant bit-identical
to _r2_reduce_quant at every M 1..32 (synthetic sweep` and `[r5-glue] DSV41_GLUE_RPSPLIT=<mode>: _r5g_rp_kernel
bit-identical to _dsv41_rp_kernel at every M 1..10 (synthetic sweep`, and no `DISABLED for this process` line.

## Files

| file | contents |
|---|---|
| `install.py`, `KIT.json` | installer and kit metadata |
| `install-all.sh` | 8/8 runner with the agreement check (review N3) |
| `base/` | byte copies of the two live modules (sha-checked) |
| `live/` | the copies pulled read-only from spark-one |
| `files/vllm/.../dsv41_r5g_rq.py` | RQPAD module (new) |
| `files/vllm/.../dsv41_r5g_rp.py` | RPSPLIT module (new) |
| `tests/` | CPU tests (`test_install.py`, `test_install_all.py`, `test_ast_exact.py` and `test_writesets.py` on this Mac; `test_wrappers.py` and `test_compile_check_logic.py` on 512S1 torch CPU) |
| `tools/` | compile check, interpreter test, spark-eight runner, `pdl_producer.py` (device test RQ-P / RP-P), `hook_import_test.py` (device test H) |
| `device_test.py` | device test |
| `EXACTNESS.md` | exactness map |
| `TEST-RESULTS.txt` | test results |
| `scripts/` | t2 re-base (`t2_glue.py`, `t2_glue.txt`) and the G1 estimate (`estimate.py`, `estimate.txt`) |
