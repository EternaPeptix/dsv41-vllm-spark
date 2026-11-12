> **climb-r8 comm2fix (2026-11-09):** the r7 pre-gate FAIL (36/40) was entirely gate defects; with the repaired gates RUN_GATE7 PASS 40/40 on the same SASS (46/46 after the r8 review fixes: MR4 reason tightened + V1-V6 vacuity controls). Read REPAIR.md first; run the gate from this copy. Window plans need `EXPECT=<production sha8>` (no default; 4f3cc1c6 today), and every device command runs with `DSV41_GLUE_RQPAD=1 DSV41_GLUE_RPSPLIT=split` (device_test7 refuses without them).

# r7-comm2: r6-comm's open checks (N1, N2, gate environment) and HT13 (`_proj_rows` folded into the all-reduce)

Kit `~/Documents/dsv41-offline/climb-r7/comm2/`, climb round 7, 2026-11-09. It owns `rdma_ar.cu` and its build, on top
of r6-comm (`src/rdma_ar.cu` = `src/mk_r7.py` applied to r6 2bff8a89). The r7 .so carries r6-comm's C1 (`DSV41_RDMA_TRIM`)
and C2 (`DSV41_RDMA_PDL`) unchanged and adds HT13 (`DSV41_RDMA_PROJ`). Its stopped window replaces r6-comm's window C.

| item | status |
|---|---|
| **N1** (W2 on the real AR consumers: gates `DSV41_RDMA_PDL=trig`) | **PASS for every consumer the review named**: mhc TileLang ×3, `_proj_rows` and `_q_kv_norm_quant`, plus 7 more t2 AR successors. **Still OPEN**: the flashinfer MXFP8 quant kernel, which is PDL-launched in t2 and follows an AR about 2 times per step; its cubin is not on disk. 3 ATen/vLLM successors are unverified. **trig stays forbidden.** |
| **N2** (ACQBULK in every CTA over the t2 PDL list: gates the C2 pair) | **22 of 26 kernel names checked PASS**, including two that pass only through the grid-level witness (A1g). **OPEN for 4 PDL-observed names with no SASS**: cuBLAS `Kernel2`, `act_and_mul`, flashinfer quant, `fusedDeepseekV4QNormRopeKV`. The collection was stopped when the r5-dw lock appeared (§2). **The C2 pair stays blocked.** |
| **Gate environment** (review: the concrete environment must enter the unrolled staging loop) | **Done.** fpflow now proves every staging store is a copy, in environments that enter the 4×-unrolled body once and twice, and fails a kernel whose unrolled body no environment reaches. A SASS-level m_stage is caught on 8 of 8 grid-staging kernels; with r6's environments the copy proof is blind to it. This also exposed a **sassir CFG bug that made r6's gate skip the S1 staging region** (§3). |
| **HT13** (`DSV41_RDMA_PROJ`) | **Built, not compiled.** Source, mutants, binary gates (projgate for the op order, ordgate for the last-CTA ordering), Python fold with a per-forward fold-count log, installer, device test, mock harness, import check and CPU tests are done. It refuses `DSV41_GLUE_PDL` ('proj') and `DSV41_OBLIT_FOLD` at import (§4.4). **The source has never been compiled:** CLUSTER_LOCK was held by the r5 windows for this whole session (r5-dw from 12:11Z, r5-w1 from 12:33Z). Not device-tested, not served. |

**Nothing ran on a GPU and nothing was compiled.** Every PASS below is a CPU check I ran (TEST-RESULTS.txt). §9 lists what
I could not run.

## 1. N1: do the AR's PDL consumers wait before reading its output? (`gate/acqcheck.py`, `audit/`)

Under `trig` the AR triggers its dependents before its final copy. Only a successor that is itself PDL-launched can then
start early, and that successor must execute `griddepcontrol.wait` (SASS `ACQBULK`) before any global read, in every CTA.

**Evidence chain.**
1. **Successor list.** `audit/t2_kernels.py` lists every immediate AR successor in t2 (rank 0/4 × code/prose/maths).
2. **PDL launches.** `audit/t2_pdl.py` finds the empirically PDL-launched kernels: an instance that starts before its
   same-stream predecessor ends. This test is sufficient but not necessary. In CUDA graphs it also gives rare false
   positives, because nsys maps parallel graph branches onto shared streams.
3. **SASS.** `audit/collect_sass.py` collects the SASS. It is read-only on spark-eight: Triton caches with
   `launch_pdl` metadata, the TileLang / flashinfer / deep_gemm / vllm caches, and the vllm and cuBLAS fat binaries.
4. **Checks.** `gate/acqcheck.py` runs, per kernel variant loadable on GB10 (sm_120/121):
   - **A1:** an unpredicated ACQBULK precedes every EXIT on every CFG path;
   - **A2:** A1, and no global read before it.

| consumer | result |
|---|---|
| `mhc_pre_big_fuse_with_norm_tilelang_kernel` (21 variants, PDL-launched) | PASS: waits before every global read |
| `mhc_fused_tilelang_kernel` (2), `mhc_post_tilelang_kernel` (1), `mhc_post_norm_kernel` (2) | PASS: wait first |
| `_q_kv_norm_quant_kernel` (6 variants, `launch_pdl` True) | PASS |
| `_proj_rows` (3 served cubins, `launch_pdl` False, no griddepcontrol) | PASS-NOT-PDL: starts only after the AR completes |
| `l2pf_kernel`, `_fused_engram_post_wkv`, `_gather_lookback`, `_mk_winner`, `_sf_onehot`, `ar_kernel_pl` | PASS-NOT-PDL |
| **flashinfer MXFP8 quant** (`kernel_cutlass_kernel_flashinferquantization…`; AR → quant about 2×/step; PDL-observed 9901×) | **OPEN**: CuTe DSL JIT, no on-disk cubin in the searched roots |
| `rms_norm_kernel`, ATen `elementwise` / `index_elementwise` | OPEN-WEAK: no SASS, 0 t2 overlaps, launch sites not read |

The r1-pdl / r5 v2.1 PDL variant of `_proj_rows` (`DSV41_GLUE_PDL`, not served) puts `gdc_wait()` first in its source
(`climb-r1/r1-pdl/files/.../dsv41_glue_pdl.py:97`). I checked its source, not its SASS: it is not compiled anywhere I may
read. For the same reason HT13 does not run with it: `DSV41_RDMA_PROJ=1` together with a `DSV41_GLUE_PDL` value that
enables 'proj' is refused at import (§4.4, review B1).

The selftest catches every must-fail SASS edit, built from the real `_q_kv_norm_quant` kernel:
- predicated wait;
- removed wait;
- LDG placed before the wait;
- predicated EXIT before the wait.

## 2. N2: does every PDL-launched kernel in the step wait in some CTA? (same tools)

C2's AR waits only for its immediate predecessor. Suppose a PDL-launched kernel K can complete before its own
predecessor P (because K never waits). Then the AR can read `p.in` or the previous AR's state too early. The `l2pf_kernel`
that sits between a GEMM and the AR about 39 times per step is the concrete case. It **is not PDL-launched**: it uses a
plain `<<<>>>` launch (`l2pf_g2.cu:35`), has no griddepcontrol, and shows 0 overlaps.

**Grid-level rule (A1g).** A kernel's completion implies its predecessor's completion if at least one CTA waits in every
execution. Trigger timing does not matter, because every dependent waits for full completion. Two kernels fail the strict
per-thread A1 but pass this rule:
- **cutlass MXFP8 GEMM** (`Copy_Atom`, PDL-observed 39387×). `@P0 EXIT /*09b0*/` exits warp-group ≥ 3 (`tid.x >> 7`); it is a
  role exit, not a CTA exit. The groups that remain wait at `/*0ec0*/` and `/*1450*/`.
- **`topkGatingSoftplusSqrt`.** The exit at `/*0080*/` is for warps whose row is ≥ num_rows. Assuming num_rows ≥ 1, warp 0
  of CTA 0 always has row 0.

For each, a concrete witness thread (tid 0, CTA 0) is walked on every undecided path, and it reaches the wait. Two
selftests confirm the witness is not vacuous:
- an unpredicated early EXIT fails the witness;
- dropping the `num_rows ≥ 1` assumption makes it undecided, so it fails closed.

**PASS (22 names).**
- `_dsv41_rp`, `_dsv4_topk`, `_q_kv_norm_quant`, `_r2_reduce_quant`;
- the 4 mhc kernels;
- the 3 deep_gemm sm120 kernels;
- MXFP8 GEMM and topkGating (A1g);
- 8 NONE-class kernels with `launch_pdl` False or no overlap;
- `mhc_proj_kernel`: launch-site evidence (moet `mhc_post_norm_sm120.cu:508`, plain `<<<>>>`). Its 2 of 27720 "overlaps"
  are graph-stream attribution;
- `per_token_group_quant`: r1-pdl's prior SASS audit, named, not re-run.

**OPEN (4).** These are all PDL-observed with no SASS collected: `Kernel2` (cuBLASLt), `act_and_mul_kernel`, flashinfer
quant and `fusedDeepseekV4QNormRopeKV`.
- **How the collection stopped.** At about 12:20Z I found the r5-dw CLUSTER_LOCK, which started 12:11Z. `collect_sass.py`
  checked the lock only at start; it had been running read-only `cuobjdump` at nice 19 on spark-eight's vllm `.so`. I
  stopped it at once.
- **Lock checks on a rerun (review B2).** The first fix re-checked `~/dsv41-opt-20261021/CLUSTER_LOCK` inside
  `collect_sass.py` on spark-eight, but the lock exists only on spark-one, so that check could never fire (and spark-eight
  cannot ssh to spark-one: host key verification fails in BatchMode, checked read-only). Now `audit/collect_driver.py` on
  the Mac drives the collection one step per ssh call (each scan root, each cuobjdump) and asks spark-one before every
  step and every 15 s while a step runs. Anything but `FREE` (locked, no answer) stops it with exit 3: the step's ssh is
  terminated, and the node worker kills its cuobjdump on SIGTERM/SIGHUP or at its next heartbeat write (≤ 5 s). T10 tests
  all four cases with local stand-ins for both ssh commands: a lock before the first step, between dumps and mid-dump
  (the running cuobjdump is killed), and a probe with no answer.
- **Why they are probably safe, and why that is not enough.** Each of the 4 reads its predecessor's output, so production
  is exact today only if each waits. The open risk is a CTA that exits without waiting, exactly what A1g checks.
- **Scope of the list.** The t2 step has 159 kernel names. For those with neither a PDL overlap nor collected SASS, "not
  PDL-launched" rests on the overlap test (sufficient, not necessary), not on a binary check.

**Close N2:** run `ops/collect-t2.sh` (no lock), add the flashinfer CuTe-DSL cubin (dump it from a stopped node or find its
cache), and require `ACQCHECK N2 PASS`.

## 3. Gate environment: the unrolled staging loop (`gate/fpflow.py`, `gate/stage_selftest.py`)

**What was missing.** r6's gate evaluated nvec = 4000 on 3 CTAs, so grid staging (`i + 3·stride < nvec`) never entered the
4×-unrolled body. The staging stores carry no FP, so no check looked at them at all.

**New check.** `fpflow.KernelGate.check_staging` runs in 8 environments. Besides r6's sets, it uses 16 CTAs with nvec 26112
(the unrolled body entered once) and 60000 (twice), and 1-CTA S1 sets.
- **Copy proof:** every STG to the send region copies `p.in[e]` word w to `send[slot][e]` word w, wherever it executes.
- **Coverage assertion:** a grid-staging r3 kernel whose unrolled body (a block with four send stores) runs in no
  environment FAILS. So the gate can fail.
- **Production twins:** the same check runs on the `ar_kernel_pl` twins.

**Must-fail evidence** (`stage_selftest.py`, on the real r3 .so 7ec43164 and on dev-r6):
- swapping the data registers of the 3rd and 4th unrolled send stores is caught in 8 of 8 grid-staging kernels
  ("send element 16421 holds in[24613]…");
- with r6's environments only, the copy proof is blind, and the coverage assertion fails the kernel.

The source-level twin `m_stage` (the a2/a3 slots swapped) is built in the window (gate case MR8). The device test checks
that the NIC-bound send slot differs in every case where the unrolled body runs: bf16 M ≥ 39, fp32 M ≥ 20, both modes,
PDL 0/128.

**sassir fixes this exposed.** Each fix is fail-closed. The whole r6 suite was rerun with them and gives r6's verdicts
(G0-G5, M1-M7 OK).
- **The biggest: `BRA.U UPx, target` had its condition as an operand, and sassir read it as unconditional.** The
  fall-through code was statically unreachable. In the S1 kernels (masks 29/31/…), that dropped the whole S1 staging
  region from every r6 check. The FP proofs were unaffected, because the reduction stores lie on the other path; the copy
  proof is what found it.
- `@Px BRA Py` (guard and operand) is modelled as `pand`.
- `SHFL` wrote nothing to its destination register: Rd silently kept its old value.
- `CS2R.32` clobbered Rd+1.
- The 2-operand `IADD` that newer ptxas emits is now evaluated.

## 4. HT13: `_proj_rows` folded into the collective tail

**What it removes.** 76 `_proj_rows` launches per step consume an all-reduce output directly, 2 per bound decoder layer:
- the attention output, after the `wo_b` AR;
- the FFN output, after the MoE AR.

In t2 each one starts 0.65 µs after the AR ends and runs 2.5-2.7 µs. Its consumer `mhc_post_norm` waits for it, which puts
it on the critical path (`estimates/ht13_t2b.txt`; 75.93-76.00 per step in all 6 traces). The input projections follow
`mhc_post_norm`, not an AR, and stay as they are.

### 4.1 Exactness: the served op order, proven from the cubin (`gate/projgate.py`, `proj_emu.py`)

**The served kernel.** It is the Triton `_proj_rows` (H 5120, BLOCK 8192, 8 warps, one CTA per row). The three served
cache entries on spark-eight have the same proven record, **f29c9ff9b85178f7**. Per row, from its SASS:
- **Elements.** Thread t holds elements `c·2048 + t·8 + j` (c = 0..3, j = 0..7). A vector starting at ≥ 5120 is masked to
  +0.0 for both x and d.
- **Chain.** `acc = x[1]·d[1]` (FMUL), then `fma(x[j], d[j], acc)` for j = 0, 2, 3, …, 31 (FFMA, acc in the addend).
- **Warp butterfly.** `acc + shfl.bfly(acc, m)` for m = 16, 8, 4, 2, 1.
- **Block.** Lane 0 writes smem[warp]. After a barrier, lanes 0..7 read smem[lane], butterfly with m = 4, 2, 1, and lane
  0 writes smem[0]. After another barrier, everyone reads s.
- **Output.** `y = cvt.rn.bf16x2(fma(-s, d, x))`.

**The r7 tail.** `r7_project` reproduces this with `__fmul_rn` / `__fmaf_rn` / `__fadd_rn` and `__shfl_xor_sync`.
- It runs two 256-thread groups, with rows g, g+2, g+4 per group (rows ≤ 6).
- x is reloaded from `p.out`: the AR's own final bf16 bytes, the exact `_proj_rows` input.
- Under P2, CTA 0 alone wrote all of `p.out`, so it projects after a barrier.
- Otherwise the last CTA through a new `st->arrive3` counter projects. Each CTA runs `__threadfence`, then the counter
  uses an acq_rel atomic, then the projecting CTA runs `__threadfence` and reads with `__ldcg`. The counter is reset to 0
  for the next op.

**The gate.** projgate proves from the SASS, per binary, for 6 rows × 256 threads:
1. every y store is `F2FP.PACK_AB(FFMA(-s, d[e], x[e]))` at the right element (5120 elements, each stored once);
2. x is the bf16 widening of the right element (`p.out` for r7, X for Triton), and d is the right fp32 word;
3. s is a shared word written by exactly one thread (the last STS before the reading LDS, with a BAR between), whose value
   is the m = 4, 2, 1 butterfly over lanes 0..7;
4. each lane reads the slot written by lane 0 of warp w(l), whose value is the m = 16..1 butterfly of the chain;
5. each thread's chain is FMUL then FFMAs, with the ordered (x element, d element) list (masked operands as Z);
6. no other FP op is in the projection's slice.

The r7 kernels' record must equal the Triton record. The selftest catches 6 must-fail SASS edits of the served kernel:
- chain start;
- warp butterfly order;
- block butterfly order;
- `.FTZ`;
- a lost negation;
- a d load offset by 16 B.

The 5 source mutants (`mp_*`) are gate cases MP1-MP5 once compiled.

**C1 dataflow.** fpflow treats the y stores of kR7Proj kernels as proven by projgate. Their FP slice is exempt only where no
reduction chain uses it, so the C1 reduction proof is unchanged.

**CPU emulation.** `proj_emu.py` emulates the served order exactly (fractions for FMA rounding).
`proj_emu.planted_row(0.5, 1.5)` is one row on which **every** HT13 mutant changes y; it is proven at every row scale used
(`--cpu-selfcheck`). The device test feeds it as one input family, so each device mutant is caught in every gated case,
not by chance.

### 4.2 Host side: arm, fold, take (`install.py` edits, 62 + 12 tagged lines)

- **Arm.** `dsv41_oblit._project(x, d)` first offers x to `rdma_all_reduce.r7_take(x, d)`, then arms d:
  `r7_arm(d)`. The attention-input projection therefore arms the layer's direction before `self.attn` runs its ARs.
- **Fold.** An rdma all-reduce of a contiguous bf16 `[rows ≤ 6, 5120]` tensor, while armed, calls
  `h.all_reduce_proj(x, two_shot, d, y)` (one kernel) and records a tag: the output tensor itself, its version counter, y
  and d. Any other all-reduce through this path clears the tag.
- **Take.** `project(out, d)` returns y only if all of these hold:
  - x is that output (same data_ptr, numel, contiguous bf16);
  - the version counter is unchanged (no in-place op since);
  - d is the same tensor object.

  The tag is consumed on use. Otherwise `_proj_rows` runs as today.
- **Memory.** The tag holds the output tensor, so its memory cannot be reused by another tensor while the tag lives. A
  data_ptr-only tag would have allowed a freed AR output's address to reappear as an mhc output and be "taken".
- **Disarm.** `project_head` disarms after the head projection, so the drafter's collectives never fold.
- **Waste.** A folded all-reduce whose output is not projected pays one tail. In t2 there are none to speak of: the armed
  window of every forward (layer 2 attention input to the head projection) holds exactly 76.00 rdma all-reduces per step,
  and each is followed by its `_proj_rows`. The AR before `_fused_engram_post_wkv` is `ar_kernel_pl` mode 3 (a column
  gather), which never folds. An earlier version of this README said the Engram AR wastes about 2 per step; that was
  wrong (`estimates/ht13_t2c.txt`).
- **Fold-count log (review item 1).** `project_head` → `r7_disarm` logs, for every CUDA-graph capture (and the first 8
  eager forwards that folded), `rdma_ar-r7 fwd: capturing=1 fwd_rows=R ar_rows=[..] folded=F taken=T wasted=F-T`.
  `ops/takecheck-r7.sh --watch` reads the rank logs read-only while p1 is up. It requires taken = 76 and wasted = 0 on
  every captured decode forward (fwd_rows 1..6) on all 8 ranks. This is a promote precondition: if it fails, stop
  before p2.
- **Graph capture.** All of this is host-side state decided at capture time. Graph replays carry exactly the captured
  kernels; y comes from the capture pool like `_proj_rows`' own `torch.empty`.
- **Flag off** (`DSV41_RDMA_PROJ` unset). `dsv41_oblit` imports nothing new (`_R7 = None`). In both files every r7
  statement inside an existing function is one `if _R7_PROJ:` / `if _R7 is not None:` block (AST test T5), and the .so
  never launches a kR7Proj kernel.
- **Rank consistency.** All ranks run the same Python, so they make the same fold decisions. The rdma fingerprint already
  hashes every `DSV41_RDMA*` env var.

### 4.3 Races

- **AR ordering.** The projection runs after the AR's own completion work inside the same kernel, and y is written before
  the kernel ends. Stream order, graph edges and C2 (PDL waits for full completion) all order y before every consumer.
- **Under `trig`.** The trigger fires before the tail (trig is forbidden anyway, §1).
- **Back-to-back ops.** `arrive3` is reset by the last CTA before it projects, so the next op (stream-ordered, or
  PDL-waited) sees 0. The device test asserts `arrive3 == 0` after every call.
- **Pre-wait accesses.** The PDL kernels' pre-wait region is unchanged; pdlcheck R2/D7 re-checks it.
- **Last-CTA ordering: the gate (review item 4, `gate/ordgate.py`).** projgate proves what the projection computes;
  ordgate proves when its loads may run. It checks every non-P2 kR7Proj kernel, i.e. the one-shot path that every fold
  with rows ≤ 3 takes:
  - **O1:** on every path, the `arrive3` atomic follows a CTA barrier, and every global store before it is followed by
    an unpredicated `MEMBAR` of GPU or SYS scope;
  - **O2:** the atomic's result sets the shared flag iff it equals gridDim.x − 1. This is evaluated concretely for G in
    {2, 3, 16} and v in {0, 1, G−2, G−1, G};
  - **O3:** every global load in the projection's slice is unreachable when the flag reads 0, and is preceded on every
    path by the flag read-back and by a GPU/SYS `MEMBAR` after the last barrier.

  Must-fail mutants: `mo_nofence` (O1), `mo_first` (O2), `mo_noacq` (O3), gate cases MO1-MO3; R4 runs the gate on
  dev-r7 and D7_ord on the built .so. The selftest (O7, run) catches 5 synthetic edits and 2 edits of the real r6
  `last_cta` SASS. The device test catches these mutants only by timing luck, so the gate is what decides.

### 4.4 Flag states HT13 refuses (review B1, item 3)

`dsv41_oblit` raises at import (so the launch fails closed) when `DSV41_RDMA_PROJ=1` meets either of these:
- **`DSV41_GLUE_PDL` with its 'proj' family on.** This includes `=1`, the r5 `combined-env.json` and the r5
  all1/all2/pdl1/pdl2 arms. The served projection would then be `dsv41_glue_pdl.proj_rows` (`gdc_wait`, `launch_pdl`), a
  separately compiled cubin whose projgate record nobody has compared with f29c9ff9. `device_test7.py --part proj`
  refuses the same state, so its serving-op assertion always names the op that serves. A `DSV41_GLUE_PDL` value without
  'proj' (for example `woa`) is allowed. Lifting the refusal needs the PDL cubin's record proven equal to f29c9ff9, added
  to `served_cubins()`, and a proj device run under it.
- **`DSV41_OBLIT_FOLD` (r2-oblit).** The folded layers call only `project_moe`, so the previous layer's direction stays
  armed and every take fails on `dd is not d`. That gives zero gain and about 2 wasted tails per layer. It stays exact,
  but it is never wanted. The earlier KIT.json claim that the two compose was wrong.

`tests/importcheck7.py` (run, PASS 33/33) imports the r7 files in copies of the climb-r6 integrate trees (live, r6 on
live, r6 on r5) with the 166d7aac production env (r8-comm2fix: rerun with the 4f3cc1c6 env too, PASS 33/33; see REVIEW.md Fixes). model.py imports dsv41_oblit, which imports rdma_all_reduce with the
flag on. Every refusal fires, and `DSV41_GLUE_PDL` without 'proj' is accepted. T4 has the same cases under stubs.
## 5. Expected gain, re-based on t2 (`estimates/estimates7.py`)

| lever | ms/step low / central / high | % G1 low / central / high | basis |
|---|---|---|---|
| **HT13** (on top of C1) | 0.019 / 0.098 / 0.188 | **0.09 / 0.47 / 0.90** | t2 (`ht13_t2c`, all 6 traces): 76 folds/step, gap 0.65 + duration 2.62 = 3.28 µs, minus tail, minus consumer slack, minus wasted folds (0.02/step). The tail is a **model**, 2.7 / 1.6 / 0.8 µs, not measured. |
| C1 (r6, carried) | 0.05 / 0.12 / 0.25 | 0.23 / 0.57 / 1.18 | climb-r6/comm |
| C2 in-edge (r6, carried) | 0.014 / 0.024 / 0.058 | 0.07 / 0.11 / 0.27 | blocked by N2 |

- **How the range was re-based (review item 2).** The first estimate (0.21 / 0.61 / 0.90 %) assumed every call saves
  gap + dur − tail.
  - Wasted folds are 0.02/step, so they barely matter.
  - Consumer slack matters more. The consumer (mhc_post_norm, mhc_proj or mhc_post_tilelang) starts a median 0.26 µs
    after the projection ends. In 14.8 % of pairs it starts more than 1 µs later, which means it was waiting for
    something else, and moving the projection earlier saves less there.
  - **Low** reads every delay beyond 0.26 µs as another dependency (tail 2.7 µs). **Central** counts only delays
    > 1.26 µs (tail 1.6 µs). **High** is the naive gap + dur − tail (tail 0.8 µs).
  - On this basis, HT13's central 0.47 % is still above the PLAN's 0.24 %.
- **The low end could be ≤ 0** if the tail spills registers or the counter path is slow. In that case projgate refuses
  first, because LDL/STL would appear in the dataflow. `device_test7.py --part projtime` measures the tail as
  live+Triton vs trim+Triton vs r7 fold, per M. **Do not run the serving pair if projtime shows no gain.**
- Earlier models were off 3-4× in both directions; only C-T-C-T decides.

## 6. Windows (`ops/window-r7.sh plan` prints them)

0. **`pre`** (no window; needs no lock; the service may run).
   - `ops/dev8-r7.sh` compiles device-only cubins on spark-eight: r7, the r6 control and 15 mutants. Each TU uses about
     100-200 MB; the guard floor is 8.5 GB and the lock is re-checked before every TU. It also records
     `cuobjdump -res-usage`.
   - Then the gate runs R0-R4, MR1-MR8, MP1-MP5 and MO1-MO3.
   - **This is the first compile of the r7 source.** A compile error or a spill stops here.
1. **`dw`** (stopped, about 3.5-4 h).
   - 14 full builds.
   - `RUN_GATE7 PASS` with D7 on the recorded .so, which yields the `GATED_SO_SHA256` line that `install.py` needs.
   - stage8.
   - 16 × `R7_DEVICE_TEST PASS`. No node runs more than 2 parts back to back (review item 6): spark-one r6 mutants,
     spark-four stage, spark-seven / spark-eight projmut (ranks 0 / 5):
     - r6 c1 on 8/8 nodes;
     - r6 mutants and r6 c2;
     - `stage` (m_stage);
     - `proj` on ranks 0-7 over 2 nodes: bitwise against the live AR + the **served** Triton `_proj_rows`, whose cubin
       sha must be a served one, at M 1..6, one- and two-shot, S1 24576/0/1 GiB, PDL 0/128/384; plus a 20× graph replay;
     - `projmut`: 5 mutants caught in every case. Coverage: S1 24576 only, PDL bits 0/128, M 1..6, both modes, ranks 0
       and 5. S1 = 0 and S1 = 1 GiB are covered for the mutants only by the binary MP cases, not by a device run;
     - `projtime`: **INFO only.** Its `R7_DEVICE_TEST PASS` means the timing ran and proves nothing about exactness.
       Do not run the q/p pair if it shows no gain.
2. **Serving** (`serve-cmd`). Pairs c/t (C1), then q/p (HT13 on C1), C-T-C-T.
   - Promote at ≥ +0.3 % in both pairs, with these on every T arm: exact.py `ec2f4437f8e1ed09`, digest
     `2f7fdccbd388225a`, det ALL==REF, and the boot lines counted 8/8.
   - p1 must also give `TAKECHECK_R7 PASS` (`ops/takecheck-r7.sh --watch`: taken = 76, wasted = 0 per captured decode
     forward, 8/8 ranks) before p2 runs.
   - q and p differ by `DSV41_RDMA_PROJ` only. Neither inherits `DSV41_GLUE_PDL` or `DSV41_OBLIT_FOLD` (§4.4).
   - The C2 pair waits for N2.
   - Never use `trig`.

**Integration.** `climb-r6/install-all.sh` installs r6-comm from its own .so. With this kit, window C is replaced by r7-dw:
- install r7-comm2 instead of r6-comm. Its installer applies r6-comm's Python edits itself (byte-identical, from
  `r6/install.py`) when the node has the live base, and stacks on them when r6-comm is already installed;
- `install-all.sh`'s comm entry and its pins need that swap. **Still OPEN before promotion:** this kit may write only
  under `climb-r7/comm2`. The swap changes the `SHIP` comm pins, `check_comm`, and `dsv41_oblit.py` as a target.
- **Until then, use LIFO (review item 5).** r7-comm2 installs last and uninstalls first. While r7 is installed,
  install-all.sh status, stack-install and stack-uninstall see `rdma_all_reduce.py` as FOREIGN and refuse. So run
  `ops/install8-r7.sh uninstall` before any install-all operation.
- **Never run r6-comm's own `install.py uninstall` on a node with r7 installed.** It removes the r6 lines and leaves the r7
  lines. That state imports flag-off and fails closed with PROJ=1. `install.py status` reports it as HYBRID with exit 4
  (T3 tests this).
- **Retired (review item 8).** `climb-r6/comm/gate/run_gate.py` is retired: its sassir reads a predicated `BRA.U` as
  unconditional and skips the S1 staging region. r6-comm's window C is retired too. Gate with `gate/run_gate7.py` and
  run r7-dw instead. KIT.json `retires` records this. This kit cannot write the marker into climb-r6.

## 7. Files

| path | what |
|---|---|
| `src/mk_r7.py`, `src/rdma_ar.cu` | r7 source generator (on r6 2bff8a89) and output (`14299953…`) |
| `src/mk_mutants7.py`, `src/mutants/` | 15 must-fail mutants: the r6 six rebased, plus m_stage, mp_first/warp/l2/unfused/chunk and mo_nofence/first/noacq |
| `gate/ordgate.py` | HT13 last-CTA ordering gate O1-O3 (+ `--selftest`) |
| `audit/collect_driver.py` | Mac-side SASS collection driver: spark-one lock check before and during every step |
| `ops/takecheck-r7.sh` | read-only fold-count precondition for the p arms (TAKECHECK_R7) |
| `tests/importcheck7.py` | import check of the r7 files in the climb-r6 integrate trees (stub harness of r5, real torch) |
| `estimates/ht13_t2c.py` | t2 sizing with wasted folds and consumer slack (the range in §5) |
| `gate/projgate.py` | HT13 dataflow gate (+ `--selftest`, `--records`) |
| `gate/fpflow.py`, `gate/sassir.py` | r6 gate, plus the staging-copy proof, projection-store exemption, `--mutant-skip-mask` and the 5 sassir fixes |
| `gate/stage_selftest.py`, `gate/acqcheck.py`, `gate/run_gate7.py` | staging must-fail evidence; N1/N2 checks; the suite |
| `gate/device_tu7.py`, `gate/pdlcheck.py` | device-only TU generator; r6 pdlcheck (unchanged) |
| `audit/` | t2 census (`t2_kernels`), PDL list (`t2_pdl`), SASS collection (`t2_sass.jsonl.gz`, 258 kernel variants), `acqcheck.txt` |
| `proj_emu.py` | exact CPU emulation of `_proj_rows` and its mutants; planted row |
| `device_test7.py`, `device_test6.py`, `shapes.py`, `tests/device/` | r7 device test; r6's device test repointed at r7comm2 |
| `install.py`, `KIT.json`, `r6/` | installer (both bases); manifest; r6-comm's installer, whose edits are reused |
| `ops/` | `window-r7.sh` (plan/pre/dw/serve-cmd), `dev8-r7`, `build8-r7`, `stage8-r7`, `run-device8-r7`, `install8-r7`, `verify8-r7`, `collect-t2`, `guard_run` |
| `tests/run_tests7.py`, `tests/mock_device7.py`, `tests/mock_device6.py` | CPU tests T1-T9; GPU-free harness smoke |
| `estimates/` | t2 sizing (`ht13_t2b`) and the estimate |
| `ref/` | r6 source, live and r6-patched Python bases, the 3 served Triton cache entries (cubin, PTX, SASS) |

## 8. Risks

- **The r7 source has never been compiled.** Host code (`launch_r7`, the second kernel parameter, pybind) and device code
  meet nvcc first in `pre`.
- **Register pressure.** `r7_project` keeps 3 rows × 4 uint4 of x and 32 fp32 of d per thread under
  `__launch_bounds__(512, 1)`, which allows 128 registers. A spill would make projgate fail closed: LDL in the dataflow.
  The fix would be fewer rows in flight (`kR7RowsPerGroup`). `-res-usage` is recorded.
- **Does the extra kernel parameter change the existing r3/r6 kernels' SASS?** R1 (dev-ctl_r6 vs dev-r7 identity) answers
  that. If it does, those kernels are still dataflow-proven by R0/D7. The live kernels' identity (the flag-off path) is
  checked separately (39/39).
- **The fold tag is host-side.** It relies on the PyTorch version counter for in-place changes. A custom kernel writing
  into the AR output through a raw pointer between the AR and `project` would not bump it. I found none on the two
  folded paths (`wo_b` → return; MoE AR → return), but that is a code reading, not a test.
- **N1 / N2** are open as stated. C1 and HT13 do not depend on them.

## 9. Tests run / not run

**Run** (outputs in TEST-RESULTS.txt; re-run after the review fixes):
- `gate/run_gate7.py --skip-missing` on 512S1: **19 CPU cases OK**, `RUN_GATE7 PARTIAL: 19/40`:
  - the r6 suite G0-G5 / M1-M7 with the r7 sassir;
  - S7a/S7b staging;
  - P7/P7b projgate;
  - A7 acqcheck selftest;
  - O7 ordgate selftest (9/9).

  21 cases are MISSING (they need compiled SASS).
- `tests/run_tests7.py` on 512S1: **10/10**. This includes:
  - T3: the HYBRID status;
  - T4: the refusals and the fold-count log;
  - T10: the collection stops on spark-one's lock.
- `tests/importcheck7.py` on 512S1: **IMPORTCHECK7 PASS**, 33 cases over the live, r6 and r5+r6 trees.
- `tests/mock_device7.py` on 512S1: **MOCK_DEVICE7 PASS**. It is a harness smoke with no kernel.
- `device_test7.py --cpu-selfcheck`: PASS.
- t2 sizing (`ht13_t2b`, `ht13_t2c`) on 512S1.

**Not run:**
- every compile: dev TUs and full builds, because the lock was held all session (r5-dw, r5-w1, r5-w2);
- gate R/MR/MP/MO/D7;
- ordgate on real r7 SASS;
- every GPU test, including the `projtime` measurement;
- `takecheck` (it needs a p arm);
- any install or serving arm;
- the rest of the SASS collection (§2).
