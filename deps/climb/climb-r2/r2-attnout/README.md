# r2-attnout: attention output path kernels (climb round 2)

One kit, one installer (`install.py`), one flag per lever.

| lever | flag | status |
|---|---|---|
| **G19** (attn:L-OP): wo_a partial retune, inverse-RoPE fold, reduce + MXFP8-quant fusion | `DSV41_ATTNOUT_OP` | **partial**: rope, rq, st1, vs and bn128 are built. The last-arriving-CTA reduce is not built (see below). **Gate:** no arm runs before `device_test.py` exits 0 on a stopped node for that arm's parts; rq has never executed on a GPU (§9, §11). |
| **G31** (attn:L-QR): Q-RoPE in the attention prologue, KV insert on an aux stream | (none) | **skipped**, see section 7 |

Files:

- `files/models/deepseek_v4/nvidia/ops/dsv41_attnout.py`: new module (kernels and host logic).
- `patched/…/o_proj.py` and `patched/…/sonar_bf16_bmm.py`: reference copies of the two patched live files.
- `base/`: the live files (sha256 recorded in KIT.json).
- `EXACTNESS.md`: element-by-element op-order map, old against new.
- `device_test.py`: bitwise and timing harness for a stopped node.
- `scripts/compile_check.py`: Triton compile-only check for sm_121.
- `scripts/interp_check.py`: Triton CPU interpreter on the real partial-kernel sources.
- `ref/compile-review-fixA/`: the reviewer's compile_check and interp_check output on spark-eight for the fixed kernel source (lines 1-483 of the module are byte-identical to that copy; see §11).
- `tests/`: CPU tests. `TEST-RESULTS.txt` holds their output.
- `ref/`: the live Triton cache entries (TTGIR and PTX) and the flashinfer and vLLM sources the exactness map was read from.

## 1. What runs today (per decode step, 43 times on the main stream)

40 target attention layers plus 3 DSpark layers. `deep_gemm_fp8_o_proj` (deepseek_v4/nvidia/ops/o_proj.py:66-112) runs:

1. Inverse RoPE: `fused_inv_rope_fp8_quant(quantize=False)`, trace name `kernel`, grid (M,8).
2. `_sm120_wo_a_partial`: fp8 W8A16 split-K, 32 splits, 512 CTAs.
3. `_sm120_wo_a_reduce`.
4. `wo_b(z)`, which runs flashinfer `MXFP8QuantizeSwizzledKernel` and then the CUTLASS MXFP8 GEMM.

I re-measured this with the audit's `chain.py`, mean µs over 60 steps:

| window | invrope | gap | partial | gap | reduce | gap | quant | gap→wo_b |
|---|---|---|---|---|---|---|---|---|
| prose r0 (q0.4) | 1.52 | 0.55 | 13.94 | 0.56 | 2.11 | 0.31 | 1.65 | −0.05 (PDL) |
| maths r4 (q4.6) | 1.56 | 0.56 | 14.07 | 0.55 | 2.18 | 0.34 | 1.80 | −0.21 (PDL) |

That is 0.29 ms of non-partial chain plus 0.60 ms of partial per step. The skeptic's `conc.py` found 0.0 µs of other-stream overlap, so all of it is on the critical path, before the post-attention all-reduce on every rank.

## 2. Mechanism: the parts of `DSV41_ATTNOUT_OP`

The value is a comma list. Aliases: `1|on|all` = rope,rq; `tune` = st1,vs.

**Values outside the known set fail loudly at boot.** o_proj.py treats every value other than unset, `""`, `0` or `off` as on. The kit module is imported lazily at the first o_proj call (profile run / warm-up), and `parse()` raises `ValueError` there for an unknown token. For example `DSV41_ATTNOUT_OP=false` or `=no` crashes the worker at boot with `ValueError: DSV41_ATTNOUT_OP='false': unknown part 'false'`. It never runs silently with a wrong setting. To turn the lever off, leave the variable unset or set it to `0`.

- **rope**: the partial kernel `_r2_wo_a_partial` (ROPE=True) reads the attention output `o[M, 8, 512]` directly. In the A-tile load it applies the inverse RoPE on the 64 rope lanes with the live kernel's PTX op sequence (§1 of EXACTNESS.md). The inverse-RoPE launch, its output buffer and one graph dependency disappear.
- **rq**: `_r2_reduce_quant` replaces the split reduce and the flashinfer quant.
  - It sums the 32 fp32 partials in the live reduce's order (evens chain + odds chain), rounds to bf16, and quantizes with flashinfer's own PTX for the UE8M0, inverse scale and e4m3 convert.
  - It writes the 128x4-swizzled scales and zero-fills the scale rows M..127.
  - `wo_b` then gets a `QuantizedActivation`, the same contract the SM100 fused WO-A uses. `FlashInferCutlassMxfp8LinearKernel.apply_weights` passes it straight to the same `mm_mxfp8`.
  - The kernel is launched with PDL (`griddepcontrol.launch_dependents`, then `griddepcontrol.wait` before any read), so the wo_b prologue still overlaps it as it overlaps the flashinfer quant today.
  - It only applies when: wo_b runs FlashInferCutlassMxfp8LinearKernel with a dynamic MXFP8 input, there is no GEMM-RS, G=1, splits=32, and N%256==0. Otherwise the live reduce runs.
- **st1 / vs / bn128**: exact launch retunes of the partial kernel. The tile, split, dot and dequant are unchanged (EXACTNESS.md §2).
  - st1: num_stages=1. The one-iteration loop gains nothing from stage 2, which costs 20 KB of shared memory and caps occupancy at 4 CTAs/SM.
  - vs: the E8M0 scale tile is loaded as [4,BN] and broadcast. Today each thread does 64 byte gathers of the same 256 bytes.
  - bn128: a 16x128 tile, 256 CTAs.

  Their speed is **unmeasured**. `device_test.py` times them.

How control flows:

- `deep_gemm_fp8_o_proj` gets one flag-gated call to `dsv41_attnout.o_proj`. It returns None, with nothing launched, when nothing applies.
- `o_proj` still calls `sonar.try_project` exactly once per projection, now with a context argument. So the L2-prefetch WOA recorder (which wraps `try_project`) sees the same call sequence, and the eligibility checks (M range, dtypes, splits) stay the live ones.
- `try_project` hands its validated weight and scale to `ctx.run`.

**On-device self-check:**

- The first 16 eager calls (warm-up, before graph capture) run the live kernels next to the new ones on the real weights and activations, and compare the raw bytes: the partial buffer, (q, sf) and z.
- The first of those calls also runs a synthetic hostile set: M = 1, 5, 16, 32 with a wide exponent range, ±0, bf16 denormals, ties, zero blocks and cancellation.
- On any difference, that part is disabled for the process. rank0.log gets `ERROR [r2-attnout] DSV41_ATTNOUT_OP part '…' DISABLED`, and the live kernels run.
- An exception in any self-check launch (synthetic or real-call, e.g. a Triton compile error) also disables the part, fails closed and logs `DISABLED ... raised ...`. The current call then uses the live kernels, so the worker does not crash.
- A part runs unchecked only after it has passed a check. Inside a CUDA-graph capture, and on eager calls after the 16-call budget is spent, a part still unchecked (`_ok is None`) runs the live kernels.

## 3. Why flag-off is identical

- `o_proj.py` gets a module flag `_DSV41_ATTNOUT_OP` (False for unset, "", 0 or off) and one `if _DSV41_ATTNOUT_OP and …:` statement. The flag is the first operand of the `and`, so nothing else is evaluated and the kit module is never imported.
- `sonar_bf16_bmm.try_project` gets a defaulted `r2ao=None` parameter and one `if r2ao is not None:` before the live `grouped_bmm` call. No live caller passes it, and the L2 wrapper forwards an empty `*a`.
- So the kernels, launch order, streams, allocations and graph contents are those of today.
- `tests/test_flag_off.py` proves this on the AST: with the gated statements removed, each patched function is identical to the live one, every other top-level statement is unchanged, and the flag expression evaluates False for unset, "", 0, off and OFF.

## 4. Why flag-on is bit-exact (summary; details in EXACTNESS.md)

- **rope**: the same IEEE ops in the same order: `p*s` (RN), `fma(x,c,ps)` / `fma(x,c,-ps)`, one bf16 RN. Nope lanes keep their bits, and masked rows are +0. The A tile entering the unchanged `tt.dot` is byte-identical.
- **st1/vs/bn128**: the same dot (mma.sync m16n8k16 bf16→f32, kWidth 4 from the fp8 upcast, 8 k16 steps ascending from 0) and the same elementwise dequant. Only staging, scale-load vectorisation and the CTA tile change. The residual hazard is Triton picking another dot layout. `scripts/compile_check.py` asserts the encodings, and the device tests check the bits.
- **rq**: the split sum in the live reduce's exact order, derived from its TTGIR layout and confirmed by its PTX add chain, then the same bf16 RN, then flashinfer's quant PTX verbatim, the same swizzle and the same zero padding. wo_b's GEMM sees byte-identical inputs.
- The one corner that cannot be settled offline: an MXFP8 block whose amax/448 is an fp32 denormal (amax < 2^-117), if CuTe-DSL flushes it. Real activations never reach it. `device_test.py` reports it as `ftz-corner`.

## 5. Race and graph analysis

- Every new kernel runs on the stream where the kernel it replaces ran (the main stream), in program order. No new streams or events are created.
- rq uses PDL with `griddepcontrol.wait` before its first load. The partial is not a PDL kernel, so it completes before rq reads.
- wo_b's GEMM waits on rq through its own PDL wait, exactly as it waits on the flashinfer quant today.
- Allocations inside capture are a subset of today's:
  - rope: the inverse-RoPE out_buf is gone.
  - rq: z, the flashinfer q and sf are replaced by our q and sf, of the same size.
- No host sync. Self-checks and Triton JIT happen only in eager mode, before capture (warm-up).
- `do_not_specialize=["M"]` gives one compiled variant for all M, so capture never triggers a first compile of a new M class.
- It works for every captured size M=1..32, for target and DSpark graph sets (both go through the same `deep_gemm_fp8_o_proj`), and with greedy-sharded drafting (no drafter output changes).

## 6. Expected critical-path saving (trace evidence, §1)

| part | per chain (low/central/high µs) | ×43 per step | G1 (low/central/high) |
|---|---|---|---|
| rope: −invrope 1.54 −gap 0.55, + partner/cos/sin loads on 1/4 of the partial CTAs | 0.8 / 1.4 / 2.0 | 34 / 60 / 86 µs | +0.15 / +0.26 / +0.37 % |
| rq: −reduce 2.15 −gap 0.33 −quant 1.72, + merged kernel ≈2.2–2.8 (PDL kept) | 1.4 / 1.8 / 2.2 | 60 / 77 / 95 µs | +0.26 / +0.33 / +0.41 % |
| st1+vs (+bn128), unmeasured | 0 / 1.2 / 4.5 (target only) | 0 / 48 / 180 µs | +0 / +0.20 / +0.77 % |
| **rope,rq,tune** | | **94 / 185 / 361 µs** | **+0.40 / +0.79 / +1.54 %** |

G1' = mean_i 1000·AL_i/(T_i−d) with T = 24.26/21.33/24.04 ms, which gives +0.427% per 0.1 ms. All of the removed time is on the main stream with no overlap, so it is on the critical path.

The last-arriving-CTA variant (reduce and quant inside the partial kernel) is **not built**. It needs an acq_rel tile counter and gives up the partial→reduce graph gap and PDL tail differently. The skeptic rates its extra gain at about 1 µs per chain after the merged-kernel variant. Deferred.

## 7. G31 skipped: why

- The full variant puts the Q RoPE inside the FlashInfer SM120 sparse-MLA prologue (`load_q_bf16_to_smem` / `quantize_q_to_smem`). The round-2 brief forbids modifying the sparse-MLA kernel this round (that is XSPLIT, round 3). The Q-RoPE belongs in XSPLIT's phase-A prologue.
- The exact-by-construction sub-variant (move the unchanged fused insert to an aux stream on the 8 indexer layers, about +0.12%) is already covered by **r1-attn IDXPAR (G20)**. IDXPAR moves exactly that main `wq_b → fused qnorm/rope/SWA insert` chain to its own stream on those layers, joined before `_sparse_indexer_and_attn`. A second edit of the same region of `_prepare_and_attn` would conflict with it.
- On the 32 SWA-only layers, the KV half could move to an aux stream only if a separate Q-only RoPE kernel replaced the fused one. That needs a new CUDA kernel matched to the nvcc FMA contraction of `fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu:244-268`, and it saves only the KV half of a 3.2 µs launch (about 1 µs per layer) while adding a join. I judged that not worth a risky ship this round.

## 8. Risks

- **Triton dot lowering for bn128/st1/vs.** Guarded by the compile check, the device test and the boot self-check (which disables the part).
- **rq hands wo_b a QuantizedActivation.** The SM100 path relies on the same contract, and eligibility is checked per layer: the kernel class name, the input quant key `kMxfp8Dynamic`, a default FormatScheme, and no GEMM-RS.
  - If a future change wraps wo_b in something that needs a tensor, the per-layer check fails closed and the live z path runs.
- **The boot self-check doubles the o-path work** for its 16 eager calls. This is startup only.
- **ftz-corner** (see §4).
- **Interplay with r1-pdl**, which patches `grouped_bmm` to launch the live partial and reduce with PDL. It composes (tested). With both flags on, the r2 path bypasses `grouped_bmm`, so r1-pdl's PDL hook does not apply to the r2 kernels. The rq kernel has its own PDL; the r2 partial kernel has none.
  - **Window controls:** measure this kit against a control with the **same r1-pdl flag state** as the treatment. If r1-pdl's flag is on in production, the control is "r1-pdl on, DSV41_ATTNOUT_OP unset". Each r2 arm then *replaces* r1-pdl's PDL partial/reduce on the o-path rather than adding to it, so its gain over that control can be smaller than the gain over a no-PDL control.
- **Install and uninstall order with r1-pdl.** Install r1-pdl first, then r2-attnout. Uninstall r2-attnout first, then r1-pdl. `install.py uninstall` now checks that each patched file is exactly `patched(backup)`. If another kit changed the file after r2-attnout was installed, it refuses before writing anything (tested), instead of restoring a backup that would drop that kit's edits.
- **rq has never executed on a GPU.** The shipped rq did not compile (review blocker 1, fixed: tl.constexpr asm strings). The fixed source compiles for sm_121 (the reviewer's compile_check) and its op order is emulated on CPU, but the bits on silicon are known only once `device_test.py` runs.

## 9. Window arms

Each arm needs the same env on all 8 nodes. KIT.json `window_arms`:

| arm | env |
|---|---|
| control | (none) |
| op-rope | `DSV41_ATTNOUT_OP=rope` |
| op-rope-rq | `DSV41_ATTNOUT_OP=rope,rq` |
| op-rope-rq-tune | `DSV41_ATTNOUT_OP=rope,rq,st1,vs` |
| op-rope-rq-tune128 | `DSV41_ATTNOUT_OP=rope,rq,st1,vs,bn128` |

**Gates before the window (hard).** On spark-eight with no window running (`CLUSTER_LOCK` absent):

1. `scripts/compile_check.py out` gives COMPILE CHECK PASS, and `scripts/interp_check.py` gives INTERP CHECK PASS, on this exact kit (module sha in KIT.json).
2. On a **stopped** node, `device_test.py --variants all` exits 0. At minimum, `--variants "rope"` must exit 0 before op-rope, and `--variants "rq;rope,rq"` must exit 0 before **any** arm that contains rq (op-rope-rq, op-rope-rq-tune, op-rope-rq-tune128). rq has never executed on a GPU.
3. Drop any arm whose variant prints `MISMATCH` or `ERROR`. An `ftz-corner` mismatch drops rq.

After each relaunch, grep rank0.log for `[r2-attnout]`. You should see `active` and **no** `DISABLED` line. A `DISABLED` line means the arm is measuring the control and its result must be discarded.

## 10. Reading the effect in an nsys trace

Per attention chain on the main stream:

- **control**: `kernel` (grid M×8, ~1.5 µs) → `_sm120_wo_a_partial` (grid 16×1×32) → `_sm120_wo_a_reduce` → `MXFP8QuantizeSwizzledKernel` → `device_kernel` (wo_b).
- **rope**: `_r2_wo_a_partial` (grid 16×1×32, or 8×1×32 with bn128) directly after the sparse-MLA kernel. No `kernel` launch.
- **rq**: one `_r2_reduce_quant` (grid 4×M) between the partial and `device_kernel`. Neither `_sm120_wo_a_reduce` nor `MXFP8QuantizeSwizzledKernel` appears.

Expect 43 chains per step. Measure with the audit `chain.py` adapted to the new names: sparse-MLA end → wo_b start, per chain. The control value is about 20.7 µs: 1.54+0.55+14.0+0.55+2.15+0.33+1.72−0.13.

Count the per-step kernel launches: there are 43 fewer with rope and 43 fewer with rq. Check that `_r2_wo_a_partial` duration under rope is within ~1 µs of `_sm120_wo_a_partial`.

## 11. Tests (CPU) and how to run them

| test | where | what |
|---|---|---|
| `tests/test_install.py` | any python3 | Anchors on the live files and on the files after r1-pdl; dry, install, idempotency, status, uninstall (byte restore); refusal of an unknown base; patched files compile. |
| `tests/test_flag_off.py` | any python3 | AST equality with the gated statements removed; flag expression; call sites. |
| `tests/test_plan.py` | 512S1 `~/exo-venv313/bin/python3` | Host logic with recorder kernels: gating, one `try_project` per call, the self-check order, capture gating, disable on mismatch, fallbacks. |
| `tests/test_emulation.py` | 512S1 | Exact fp32/bf16/e4m3 emulation of old against new: rope A tiles, reduce order (with order sensitivity shown), quant bytes, swizzle and padding coverage. |
| `tests/test_jit_globals.py` | any python3 | AST lint: every module global read inside an `@triton.jit` function is `tl.constexpr(...)`. This catches review blocker 1 on CPU; it flags the shipped module. |
| `scripts/compile_check.py OUTDIR` | spark-eight, no window running, `nice -n 19 ~/venvs/vllm032/bin/python` | Triton compile to TTGIR/PTX for sm_121, no GPU context; dot encodings and PTX op kinds. |
| `scripts/interp_check.py` | spark-eight, no window running | Triton CPU interpreter (`TRITON_INTERPRET=1`) on the real live and kit partial kernels; partial buffers bitwise, M 1/5/17 × rope, vs, bn128 and rope,vs,bn128. Covers only the partial; the interpreter cannot run inline asm, so rq is not covered. |
| `device_test.py` | the lead, a stopped node | Bitwise for 10 variants, M=1..32 and 4 input sets; timing warm, cold-L2 and in-graph; exit 0 only if every case is bit-identical. |

**Status of the checks (2026-11-08, after the review fixes).**

- Run by me and passing: the five CPU tests (`TEST-RESULTS.txt`).
- compile_check and interp_check were **run by the reviewer**, not by me, on spark-eight with Triton 3.7.1 (15:18 UTC, `~/climb-scratch/review-attnout-fixA`). They ran on a copy whose only change was the tl.constexpr fix: COMPILE CHECK PASS and INTERP CHECK PASS 12/12, with the interp harness fix equivalent to the one now in `scripts/interp_check.py`. Output is in `ref/compile-review-fixA/`.
  - Lines 1-483 of the current module are byte-identical to that copy: every kernel, the asm constants, `INV448_BITS` and the `_launch_*` helpers. `compile_check.py` is the same file (sha e476088a). The only later changes are in `_run` (host logic) and do not reach any compiled kernel.
- I could not rerun either script on the final kit: `CLUSTER_LOCK` was held by window w2 (15:21 to about 19:21 UTC) for this fix session. Rerunning them is gate 1 of §9.
- `device_test.py` has not run anywhere (no GPU for the builder). It is gate 2 of §9.
- The launcher moved from fda6d05a to 08a249da during the session (L2 prefetch pacing). The files this kit patches are unchanged: o_proj 34b9e1a4, sonar 9f6f3f84.
