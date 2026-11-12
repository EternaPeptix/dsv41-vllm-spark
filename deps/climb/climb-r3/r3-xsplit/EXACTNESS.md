# r3-xsplit: op-order map, live vs XSPLIT

Live kernel: `sparse_mla_prefill_mg_dual_kernel<DSV4, BF16, NH=8, PBS=64, PBSX∈{64,32}, MG_N_HG=1>`
= `prefill_mg_impl` in `flashinfer/data/include/flashinfer/attention/sparse_mla_sm120/prefill_mg_kernel.cuh`
(sha256 `0ec1603dca10…`, identical 8/8), lines 1164–2127, DUAL_CACHE, !ASSUME_FULL_TILES, VALID_HPB = 8
(rows 8–15 of the 16-row MMA tile are dead: Q = 0, never stored). Compile flags of the live JIT module
(`build.ninja` of `~/.cache/flashinfer/0.7.0.post1/121a/cached_ops/sparse_mla_sm120`): `-O3 -use_fast_math
-gencode=arch=compute_121a,code=sm_121a` (FTZ, approximate ex2/rcp/lg2, fmad contraction). The XSPLIT library is
built from the same live headers with the same flag list (`src/build.sh`).

New kernel: `dsv41_xsplit_mg_dual_kernel<PBSX>` in `src/dsv41_xsplit_kernel.cuh` (A unit = lines ~340–668,
B job = ~670–872). SASS references below are from the live dual64 kernel (`base/sass/dual64.full`, dumped from
the live JIT .so on spark-five by the g130-attn census).

## Per thread, per tile, live order (one CTA per token, tiles t = 0..n-1 in order)

Thread = math lane (mwarp 0..7, gid = lane>>2, tid = lane&3); rows gid (live) and gid+8 (dead); entries
e0 = 8·mwarp + 2·tid, e1 = e0 + 1.

| # | live op (source line, SASS) | where XSPLIT runs it | operands in XSPLIT |
|---|---|---|---|
| 1 | IO: index LDG, scale gather, cp.async.bulk 64×448 B (1261–1315) | A unit, IO warps (`claim_and_issue`) | same code, same addresses |
| 2 | QK nope: 7 blocks × 4 k-steps; KV FP8→F16 `cvt.rn.f16x2.e4m3x2`, ×scale (FMUL), `cvt.rn.bf16x2.f32`, HMMA m16n8k16 chained on qk (1553–1592) | A unit, verbatim | same smem bytes (gather), same Q smem |
| 3 | QK rope `compute_qk_rope` (1628) | A unit, verbatim | prefetched rope regs from the same entry |
| 4 | masking −1/topk_len/topk_len_extra → −1e30 (1633–1680) | A unit, verbatim | same geometry (`xs_token_geo` = 1202–1231 clamps) |
| 5 | s = qk × sm_scale_log2e (FMUL.FTZ, 1682) | A unit, verbatim | — |
| 6 | `softmax_warp_max` (FMNMX, 2 SHFL levels) → reduce_buf (1686–1690) | A unit, verbatim | — |
| 7 | threads<16: tm = −1e30; tm = fmaxf(tm, rb[w]) w=0..7 (FMNMX.FTZ ×8) (1702–1705) | A unit, verbatim | — |
| 8 | **nm = fmaxf(old_m, tm); alpha = exp2f(old_m − nm) (FMNMX, FADD, MUFU.EX2); m_smem = nm** (1706–1708) | A unit: tm published as one 64-bit word (epoch‖bits, `st.release.gpu`); old_m = fold of the published tm of tiles 0..t−1 of this token, from −1e30, `old_m = fmaxf(old_m, tm_u)` in tile order (acquire loads); then the same two lines | old_m bits = live m_smem bits: the live m chain is exactly this fold (F5); tm bits copied |
| 9 | vsc cache: ue8m0 → fp32 ×14 (1715–1729) | A unit, verbatim | same scale bytes |
| 10 | **guard: `if (alpha0<1 ‖ alpha1<1)` (FMNMX + FSETP.GEU, alpha1 = dead row) → acc_o ×alpha (28 FMUL.FTZ), acc_rope ×alpha (4), warp_l ×alpha (2)** (1739–1755) | **B job**, tile order: same predicate on alpha[gid], alpha[gid+8] read from the row (all 16 alphas written by A, F4); `__fmul_rn` = FMUL.FTZ | acc values = B's own registers carried in the same order |
| 11 | w = exp2f(s − nm) ×4 (FADD.FTZ, MUFU.EX2) (1757–1762) | A unit, verbatim | nm from #8 |
| 12 | `softmax_warp_sum`: w0+w1, SHFL xor1 + FADD, SHFL xor2 + FADD (1765) | A unit, verbatim; ls0 stored by tid==0 (the 4 quad lanes hold identical bits: each level is one commutative FADD) | — |
| 13 | **warp_l += ls (FADD.FTZ, separate from #10; SASS 1788/1789 `FADD.FTZ R15, R146, R15`)** (1766) | **B job** `__fadd_rn(warp_l, ls0)` after the guarded FMUL — never an FFMA (F2) | ls0 bits from the row |
| 14 | V-scale: ws = w×vsc (FMUL), atomicMax(int view of fmaxf(ws00,ws01)) into w_head_sc_all (zeroed per tile, 1422) (1771–1781) | A unit, verbatim | — |
| 15 | normalise: fmaxf(x, 1e−10)/448 (1786–1787) | A unit, verbatim; the 7×16 results written to the row | — |
| 16 | per vc (7): si = 1/sc (MUFU.RCP), wn = w×vsc×si, `__nv_fp8_e4m3` (F2FP.SATFINITE.E4M3), STS to W_FP8 ping-pong, bar, 2 × QMMA m16n8k32 (C chained from 0) → xv[4] (1865–1975) | A unit, verbatim; xv[0], xv[1] (live row) written to the row | same KV smem (d2_load_b_fp8), same W_FP8 |
| 17 | **acc_o[vc] += xv × sc (one FFMA.FTZ per element, SASS 1541–1793)** (1972–1975) | **B job** `__fmaf_rn(xv, sc, acc)` = FFMA.FTZ, after #10 of the same tile, vc order | xv, sc bits from the row |
| 18 | **XV rope `xv_rope_mma_mg`: STS bf16(w) (F2FP.BF16.PACK_AB) for rows gid/gid+8 at entries e0/e1; bar; 4 k-steps: ldmatrix A, B = rope V from global (idx<0 → 0), HMMA m16n8k16 with C = acc_rope carried across tiles** (1986–2003; `xv_rope_mma.cuh:139–183`) | **B job**: A unit writes fp32 w (live row) and the B operands (the same `load_rope_v` loads, `xs_rope_b`, idx<0 → 0) to the row; B stores `__float2bfloat16(w)` for the live row (dead rows 0), bar, the same 4 ldmatrix + HMMA chained on acc_rope (F3) | A: same bf16 bits for live rows (dead rows only feed dead output rows: MMA rows are independent); B: same u32 operands; C: B's acc_rope, carried in tile order |
| 19 | epilogue: tid==0 → reduce_buf; threads<16: ts = 0 + l_w0 + … + l_w7 (FADD.FTZ ×8 from RZ) (2013–2029) | B job, `__fadd_rn` from 0.f | warp_l from #13 |
| 20 | m for epilogue = m_smem after the last tile | B job: nm of tile n−1 from the row (−1e30 for n = 0) | = live m_smem |
| 21 | il: sink: FFMA(sink, LOG2E, −m) → MUFU.EX2 → FADD(l, ·) → MUFU.RCP, select d>0 (SASS 2305–2310); no sink: l>0 ? RCP(l) : 0 | B job, pinned `__fmaf_rn`, `exp2f`, `__fadd_rn`, `rcp.approx.ftz.f32` | — |
| 22 | out = bf16(acc × il) (FMUL.FTZ, F2FP.BF16.F32.PACK_AB), staging, uint4 stores of VALID_HPB rows | B job, `__fmul_rn` + `__float2bfloat16`; staging in the item's own KV buffer | — |
| 23 | lse = l>0 ? lg2(l)+m : −1e30; sink: sl = FMUL(sink, LOG2E); lse≠−1e30 ? FADD(LG2(FADD(EX2(FADD(sl,−lse)),1)), lse) : sl (SASS 2398–2420) | B job, pinned `__fmul_rn`/`__fadd_rn`, `softmax_lse` verbatim | — |

Everything the A unit computes per tile depends only on tile t's inputs and on old_m (#8); everything the B job
does is the live cross-tile recurrence in the live order on bit-identical operands. Hence every output element
sees the same FP ops, in the same order, on the same operand bits.

## The four mandatory corrections (audit `verify-G02-exactness`)

1. **Derived from the DSV4 path `prefill_mg_impl` (1164–2127)**, not the DOTS3_SWA `math_pc` path: the per-tile
   body in the A unit is that path's MG_N_HG == 1 / CM == BF16 / UE8M0 branches, verbatim (#2–#16).
2. **l replay = guarded FMUL then a separate FADD** (#10, #13): `__fmul_rn` inside the guard, `__fadd_rn` after it;
   the intrinsics are never contracted. `tools/sass_check.py` verifies every FP32 op in the XSPLIT kernels carries
   `.FTZ` (the intrinsics follow `-ftz=true` through `__nvvm_reflect`).
3. **bf16(w) is written back as the A operand of the rope MMA, and the acc_rope chain runs in phase B** (#18). The
   rope B operands are copied by A with the live loads (idx<0 → 0, never a slot-0 read for a −1 entry) — the
   `XS_ROPE_IN_WS=0` bench variant re-reads them from the KV cache in B instead (same bits; timing only).
4. **The dead-row alpha1 / guard is carried into phase B** (#10): all 16 alphas are in the row and B evaluates the
   live predicate `alpha0 < 1 || alpha1 < 1`.

Compile mode: the library is built with the live `-use_fast_math -O3` flag list; the A unit's code is the live
source text, so ex2/rcp/division/contraction choices are made on identical expressions. The explicit intrinsics
are used only for the cross-tile and epilogue ops whose live SASS is unambiguous (FMUL.FTZ, FADD.FTZ, FFMA.FTZ,
MUFU.RCP), so their form cannot depend on register pressure or CSE in the new function (F6).

## What is not reproduced, and why it cannot matter

- Dead rows 8–15 of acc_o / warp_l are not accumulated in B, and their staging rows are not written. Live never
  stores them (`VALID_HPB = 8`: il1 = 0, the coalesced store and the LSE loop cover rows < 8). Their only influence
  on live rows is alpha1 in the guard, which B evaluates.
- Rows 8–15 of the rope MMA A operand are zero in B (live: bf16 of dead-row w). An MMA output row depends only on
  its own A row, so live rows 0–7 see the same A, B and C bits.
- `m_smem` and the w_head_sc_all zeroing per tile keep their live semantics inside the A unit.

## Memory model and stale state

- Tile maxima: one 64-bit word per (unit, head) `(E << 32) | bits(tm)`, `st.release.gpu` / `ld.acquire.gpu`
  (single-copy atomic: value and epoch arrive together). E = ctl[GEN] + 1 per call.
- Rows: plain `st.global.cg` by all math threads, math barrier, thread 0 `__threadfence()` +
  `st.release.gpu done[u] = E`; reader thread 0 spins `ld.acquire.gpu` until E, math barrier, then all threads read
  with `ld.global.cg` (never `ld.global.nc`/`__ldg` on the workspace; workspace pointers are not `const
  __restrict__`).
- Epochs: every call reads GEN once; the last CTA to finish (atomic finished counter == gridDim) resets
  claim/finished and publishes GEN = E (before a 32-bit wrap it clears every flag and restarts at 0). A flag from
  any earlier call or graph replay carries an older epoch and is never accepted; the workspace can be poisoned
  between calls (device_test does) without effect.
- Deadlock: items are claimed in index order (A units token-major / tile-ascending, then B jobs, then END); every
  wait (prefix max, done flag) targets a smaller index claimed by a running CTA that publishes it without waiting
  on a larger one → acyclic, for any grid size, residency or co-runner. A 2 s spin bound traps (CUDA error) instead
  of hanging if an invariant is ever broken.

## Evidence

- CPU emulation `tests/emu_xsplit.py` (output `tests/emu_xsplit.out`): live serial loop vs A-in-any-order +
  B replay, float32 + FTZ at every step, 401 tokens over 10 input classes (normal, ties, NaN, ±Inf, ±0 Q rows,
  huge/Inf scales, FP8 NaN, rope Inf/NaN/subnormal, structural holes/empty tiles/n=0, tiny rope) plus a
  constructed dead-row-guard case: **0 mismatches**. Negative controls are detected: l as one FFMA (7 inputs),
  rope MMA split into C=0 + FADD (7), a stale epoch accepted (16), guard without alpha1 (the constructed case).
  The prefix-fold order itself is shown insensitive (fmax drops NaN, max(−0,+0)=+0), so even a parallel scan
  would be exact; the kernel keeps the sequential fold anyway.
- Device bitwise test `device_test.py` (stopped node, not run yet): live JIT kernel vs XSPLIT, raw-byte compare
  of output and LSE, normal + 6 hostile classes, T ∈ {1..6, 8, 16, 33, 64}, n ∈ {0..nmax}, dual64/dual32,
  topk_extra 512/1024, grids 1..64, poisoned workspace and stale epochs between calls, SM-hogging co-runner
  (46/47/48 SMs) and the WOA l2pf stand-in, epoch wrap, 256 CUDA-graph replays alternating 4 input sets.
- `tools/sass_check.py` on the build: C0 of the bench's live-kernel instantiation vs the live JIT SASS, FTZ on
  every FP32 op of the XSPLIT kernels, opcode histogram, LDG flavours.
