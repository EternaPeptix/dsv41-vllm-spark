# comm2fix: repairing r7-comm2's pre-gate (climb round 8, 2026-11-09)

This is a copy of `climb-r7/comm2`. climb-r7 was not edited. Production launcher is **4f3cc1c6**.

**Verdict: every failure was in the gate. The kernel is correct.**
- The r7 source, its device-only cubins and the 15 mutants are unchanged. I did not recompile anything; the gate ran on
  the SASS that today's pre-window produced (`climb-r7/comm2/sass/dev-*.sass`).
- With the repaired gates, **`RUN_GATE7 PASS: 40/40`** on 512S1. Before the repair it was FAIL, 36/40.
  *(r8 review fixes: now 46/46, with the six V vacuity controls added; see REVIEW.md "Fixes".)*
- **HT13's fold equals the served `_proj_rows` dataflow bit for bit, as far as the static gate can prove.** projgate's
  record for all 18 kR7Proj kernels equals the served Triton record f29c9ff9b85178f7. This holds in 4 environments: one
  CTA with one loop trip, the same with the BRA.DIV divergence fallback, zero loop trips, and 3 CTAs with CTA 2 last.
- **comm2 ships with HT13.** There is no need to drop it.
- **Still not run:** any GPU test, any full `.so` build (D7), any serving arm. I ran no device test and nothing touched
  the GPU.
- **N1 and N2 stay OPEN** (§4). The read-only SASS collection now ran to the end. It moved N2 forward, but it did not
  close it.

## 1. What failed and why (pre-gate `climb-windows/r7-pre/pre.out`)

| case | r7 verdict | cause | kernel or gate? |
|---|---|---|---|
| R3 projgate | 0/18 kR7Proj kernels (`row 0: y elements stored 0/5120`) | 6 gate defects (§2 G1-G6). The first one hid every y store. | gate |
| R0 fpflow live vs dev-r7 | FPFLOW FAIL 56/74 (the 18 kR7Proj kernels) | The projection exemption covered only the y stores' register slice (§2 G7). | gate |
| MR6 m_waitlate_fp | BAD: expected PASS, got 56/74 | The same as R0 (m_waitlate leaves the projection unchanged). | gate |
| R4 ordgate | 0/12 (`O3: projection load /*0960*/ ... can run when the last-CTA flag is 0`) | O3 gated the AR's own staging loads (§2 G8). | gate |
| R1 | OK, but its FPFLOW line read FAIL 56/74 | Same as R0. R1 tests identity only (95/95). | gate |
| MP1-MP5 | "caught 18/18" | **Vacuous**: the unmutated kernel also failed every kernel. They are now re-proven, each for its own reason (§3). | gate |
| A7 / ACQCHECK | N1 OPEN, N2 OPEN | Missing SASS (§4) | open evidence |

**The r7 source was compiled for the first time today.** None of these gate paths had seen real r7 SASS before. That is
why the gate, not the kernel, broke.

## 2. Gate defects found and fixed (`gate/`, all fail-closed)

### sassir.py

These are parser and CFG bugs that affect every gate.

**G1. `BRA.DIV URx, target` was read as an unconditional BRA.**
- nvcc emits BRA.DIV before the level-2 `__shfl_xor_sync` inside `if (warp == 0)`. It is taken only when the warp is
  diverged, into a `WARPSYNC.COLLECTIVE` fallback that ends in `BRA 0x4a10`.
- sassir made the converged **fast path unreachable**. That is the path that runs, and no gate ever checked it.
- **Fix:** BRA.DIV is now a two-way branch on the environment register `SR_WARPDIV`. When it is unset the value is
  opaque, so both paths stay possible.
- BRA.DIV occurs only in the 18 r7 kR7Proj kernels. It occurs 0 times in the live, r3 and r6 SASS and the Triton
  `_proj_rows` SASS.

**G2. A block that ended in a predicated non-control instruction got the edge condition `!guard`.**
- Example: `@!P4 STS [R22+0x8]` just before the fallback's re-entry point at 0x4a10.
- As a result, the threads whose guard held (lane 0, the writer of s) never reached the rest of the kernel. Those are
  exactly the y stores of thread 0.
- **Fix:** only BRA/EXIT/RET/BPT/BRX/JMP/JMX decide an edge.

**G3. The HFMA2 move idiom had its halves swapped.**
- cuobjdump prints the high half first.
- r7 evaluated `HFMA2 R3, -RZ, RZ, 0, 5.96e-08` (the +1 of every arrival atomic) as 0x10000. It evaluated the fallback's
  lane masks 4/2/1 as 0x40000/0x20000/0x10000.
- **Fix:** the constant is now `(h0 << 16) | h1`.
- In the live, r3 and r6 SASS the idiom is only `0, 0` (value 0, either way), `0, 1.19e-07` (2) or `0, 5.96e-08` (1).
- Every r3/r6 case was rerun with the fix and gives the same verdict (G0-G5, M1-M7, S7a/S7b).

### projgate.py

**G4. Its environment could not reach the projection tail.**
- sassir follows loops on their iteration-1 path only. r7's env used nvec = 3840 on 1 CTA, so every thread entered the
  one-shot reduce loop and never left it. Every y store was unreachable: 0/5120.
- **Fix:** `CONFIGS` lists 4 environments, and each makes every AR loop run 0 or 1 times for the projecting CTA:
  - `g1-trip1`: 1 CTA, nvec 512;
  - `g1-trip1-div`: the same, with SR_WARPDIV = 1;
  - `g1-trip0`: nvec 0;
  - `g3-last2`: 3 CTAs, nvec 1536, CTA 2 projects (P2 kernels: CTA 0).
- The arrive3 atomic is pinned to gridDim.x − 1.
- The record must be identical in all 4 environments. It cannot depend on trip count, on blockIdx or on which shuffle
  path ran.

**G5. Two compiler idioms in a correct kernel were not modelled.**
- **(a) Folded masked terms.** Chunk c = 3 (elements 6144 + 8t ≥ 5120) is masked by a compile-time constant, so nvcc
  folds `FFMA(+0, +0, acc)` into `FADD(RZ, acc)`. The two are bitwise identical:
  - +0·+0 = +0 exactly, then one RN add;
  - acc = −0 gives +0 in both;
  - NaN, Inf and subnormals behave the same; neither op flushes.

  It is recorded as the masked term (Z, Z), which is exactly what the served kernel's runtime-masked FFMA records.
  `.FTZ`, a negated or `|.|` operand, and a nonzero constant are still refused.
- **(b) `MOV Rd, RZ` as the zero arm of a masked x load** (`@!P3 LDG`). Before, only `CS2R SRZ` was accepted.

**G6. The tree and slot proofs used program order.**
- The fallback stores s at /*5800*/, after its reader at /*4a70*/ in program order. Its last FADD re-joins through a
  phi, and its SHFL takes its value via `MOV` and its lane mask and clamp from MOV/HFMA2 constants.
- **Fix:**
  - `sts_writers` uses forward-CFG reachability. "Before", "last writer" and "a BAR on every path" are now CFG
    properties.
  - `butterfly` sees through phi (the incoming value under the env) and plain MOV.
- The SHFL mask must still evaluate to exactly 4/2/1 and 16..1 with clamp 0x1f.

### fpflow.py

**G7. The projection exemption was too narrow.**
- It covered only the y stores' register slice. That slice stops at the LDS of s, so the dot chain and the level-1 tree
  were "stray FP", and the partial-sum STS were sent to `check_store` ("bf16 word 0 is not an F2FP pack").
- **Fix:** fpflow runs projgate's proof first (`PROJ_PROOFS`). It exempts exactly:
  - the y stores the proof executed;
  - the STS writers it proved;
  - the FP ops it covered.
- It fails the kernel when:
  - the proof failed;
  - any q.y store found by address decode is not proven.
- The exemption still applies only to FP ops that no C1 reduction chain uses.
- `--mutant` runs keep r7's behaviour, so MR1-MR5 and MR8 are unaffected.

### ordgate.py

**G8. O3 gated the wrong loads.**
- It gated every LDG in the y stores' syntactic slice. A masked x word is `MOV Rx, RZ; @!P LDG Rx`, and that slice keeps
  Rx's older definition: the AR's own `p.in` staging load at /*0960*/. That load runs before the flag by design.
- **Fix:** O3 gates exactly the 15 global loads that projgate proved to be x/d operands (9 x + 6 d per kernel). It fails
  closed when there is no proof.
- O3's env drops SR_WARPDIV, so both paths stay possible.

### Shared plumbing

- `projgate.prove_all` caches proofs and runs them in parallel. The cache key is
  sha256(kernel name, SASS text, projgate.py + sassir.py + fpflow.py). A proof is never reused by different gate code.
- `ops/window-r7.sh` now runs the gate in `climb-r8/comm2fix` on 512S1 (`GATE_REMOTE` overrides it).

## 3. Results (512S1 `~/climb-r8/comm2fix`, `gate/run_gate7.py --sass6 ~/climb-r6/comm/sass`)

`gate/results/SUMMARY.txt` is the final-code run. `gate/results-r8/` is the identical first run.

| case | verdict |
|---|---|
| G0-G5, M1-M7, S7a/b, P7, P7b, O7, A7 | unchanged from r7: all OK. P7 is still PROJGATE_SELFTEST PASS, base record f29c9ff9b85178f7, 6/6 edits caught. |
| R0 live vs dev-r7 | IDENTITY 39/39, **FPFLOW_GATE PASS 74/74**, 8/8 twins |
| R1 dev-ctl_r6 vs dev-r7 | IDENTITY 95/95, FPFLOW PASS 74/74 |
| R2 | PDLCHECK PASS 44/44 |
| R3 | **PROJGATE PASS 18/18** (equal to served f29c9ff9 in all 4 envs) |
| R4 | **ORDGATE PASS 12/12** (O1 release, O2 last CTA, O3 15 proven loads gated + fenced) |
| MR1-MR5 | KILLED 60/60 each, for their regexes |
| MR6 m_waitlate_fp | **FPFLOW PASS 74/74** (expected) |
| MR7 | KILLED 44/44 |
| MR8 m_stage | KILLED 54/54 /staging/ |
| MP1 mp_first | KILLED 18/18 /record differs/ |
| MP2 mp_warp | KILLED 18/18 /tree level/ |
| MP3 mp_l2 | KILLED 18/18 /tree level/ |
| MP4 mp_unfused | KILLED 18/18 /not an FFMA/ |
| MP5 mp_chunk | KILLED 18/18 /y\[\d+\] = fma/ |
| MO1 mo_nofence | KILLED 12/12 /^O1/ |
| MO2 mo_first | KILLED 12/12 /^O2/ |
| MO3 mo_noacq | KILLED 12/12 /^O3/ (`not preceded by a gpu-scope MEMBAR`) |
| **RUN_GATE7** | **PASS 40/40** |

**No gate was weakened.**
- Every must-fail mutant is still caught, in every kernel.
- The MP cases now carry `--expect` regexes; r7 had none. So each projection mutant must fail for its own reason, and
  against a base that passes.
- The changes widen what the gates see (both shuffle paths, the reachable tail, 4 environments). They narrow only two
  sets, and each to what is proven:
  - O3's load set: the proven operand loads, instead of a syntactic over-approximation that included the AR's own
    loads;
  - fpflow's exemption: exactly what projgate proved.

**Other checks.**
- Register use of the kR7Proj kernels is 126-128 registers, LOCAL:0 (`sass/dev-r7.resusage`). *(r8 review N3,
  corrected:)* local memory is not empty: STACK is 24 or 48 bytes and the 18 kernels hold 6-12 STL each (printf
  argument frames before `CALL.ABS`) and 0 LDL. A spill would also live in that stack frame; the "no spill" conclusion
  holds because projgate's chain resolution refuses any LDL/STL in the projection dataflow.
- CPU tests: `tests/run_tests7.py` gives CPU_TESTS7 PASS 10/10, and `device_test7.py --cpu-selfcheck` gives PASS. Both
  ran on 512S1 against this copy.

## 4. N1 / N2 (`ops/collect-t2.sh`, lock-checked from the Mac)

**The collection.**
- It ran to completion: 388 lock probes, 355 steps, 296 kernel variants.
- spark-one had no CLUSTER_LOCK at any probe.
- It was read-only on spark-eight: cuobjdump at nice 19, streamed to the Mac. Nothing was written on any Spark.
- 7 names are still missing:
  - 3 cuBLASLt `cutlass::Kernel2<cutlass_80_wmma_...>`;
  - 4 flashinfer CuTe-DSL `MXFP8QuantizeSwizzledKernel`. These are JIT-compiled; no cubin exists in any searched root.

| item | before | now |
|---|---|---|
| `rms_norm_kernel` (N1) | OPEN-WEAK, no SASS | **PASS-NOT-PDL** (no griddepcontrol, not PDL-launched) |
| `fusedDeepseekV4QNormRopeKV` (N2) | NO-SASS | **FAIL/OPEN**: the sm_120 variants wait, but `@P0 EXIT /*00c0*/` / `@!P0 EXIT /*00b0*/` come before the wait. The A1g witness is undecided: it needs a launch-site assumption (like topkGating's num_rows ≥ 1) that I have not established. Overlap evidence: 361/16560, all after the MXFP8 GEMM. |
| `act_and_mul_kernel` (N2) | NO-SASS | **FAIL/OPEN**: the collected silu variant has **no griddepcontrol in any arch**. It "overlaps" only 11/15480 times (max 1.44 µs, all after Copy_Atom), the same signature as mhc_proj's graph-attribution false positives. It is probably not PDL-launched, but that is not proven: it needs the vLLM launch site. |
| `Kernel2` (cuBLASLt), flashinfer MXFP8 quant | NO-SASS | still NO-SASS (OPEN) |
| N1 flashinfer quant | OPEN | OPEN (JIT; dump it from a stopped node or find its cache) |

**`ACQCHECK N2 FAIL/OPEN N1 FAIL/OPEN`. The C2 pair stays blocked and `trig` stays forbidden.** C1 and HT13 do not
depend on N1 or N2.

The new collection is `audit/out/t2_sass.jsonl.gz`, with `acqcheck.txt` and `.json` next to it.

## 5. What remains before promotion (unchanged order, `ops/window-r7.sh plan`)

1. **dw** (stopped window):
   - 14 full builds;
   - RUN_GATE7 with the **D7** cases on the built .so. This yields `GATED_SO_SHA256`.
   - stage8;
   - the 16 device parts. `proj` is bitwise against the live AR + served Triton `_proj_rows`; `projmut` must catch the 5
     mutants on GPU; `projtime` is information only.
2. **Serving pairs:**
   - c/t (C1), then q/p (HT13);
   - TAKECHECK_R7 on p1;
   - the 9-case long-prompt determinism check on every T arm.

Run the gate from this copy, not from climb-r7: the climb-r7 gate gives FAIL 36/40 on the same SASS.

## 6. Files changed (vs climb-r7/comm2)

- `gate/sassir.py`: G1-G3 (31 diff lines).
- `gate/projgate.py`: G4-G6, CONFIGS, the proof cache, the `prove_all` main.
- `gate/fpflow.py`: G7 (`PROJ_PROOFS`).
- `gate/ordgate.py`: G8.
- `gate/run_gate7.py`: MP `--expect`.
- `ops/window-r7.sh`: the gate's remote directory.
- `audit/out/`: the completed collection and acqcheck.
- `gate/results/`, `gate/results-r8/`: the verdicts.
- `dbg/`: debug scripts used for the diagnosis.
- Untouched: `src/rdma_ar.cu`, the mutants, the installer, the device tests, KIT.json.
