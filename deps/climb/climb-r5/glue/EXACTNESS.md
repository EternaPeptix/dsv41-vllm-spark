# r5-glue: exactness map (old vs new, element by element)

Both levers change **where and when** existing operations run. Neither changes **which** floating-point operations
produce an output element, or their order. Each claim below is tied to a check. "AST" means
`tests/test_ast_exact.py`, which compares the live and kit kernels statement by statement on the parsed source.
"WS" means `tests/test_writesets.py`. "PTX" means `tools/compile_check.py`. "DEV" means `device_test.py`.

## RQPAD: `_r5g_reduce_quant` vs live `_r2_reduce_quant` (dsv41_attnout.py:229-274)

| output | live op sequence | kit op sequence | evidence |
|---|---|---|---|
| `Q[r, col]` (e4m3 byte) | Load 32 fp32 partials. Even chain `p0+p2+..+p30` and odd chain `p1+..+p31`, each left to right, then one add. `cvt.rn.bf16.f32`, exact widen. `amax` = max of abs over the 32-block (order-free, NaN-free). `normalized = amax * fp32(1/448)`. `_ASM_UE8M0`, then `_ASM_INV_SCALE`, then `_ASM_FP8` (mul.f32, min, max, cvt.rn.satfinite.e4m3x2), then store u8. | The same statements (identical AST, same `do_not_specialize=["M"]`). The PTX is compiled with the same options (`num_warps=4`, `enable_fp_fusion=False`, `launch_pdl=True`). | AST "every statement before the padding loop"; PTX "f32 opcode multiset identical" and "inline asm present"; DEV RQ-B |
| `SF[off(r, c)]`, real row `r < M` | The UE8M0 byte from `_ASM_UE8M0`, stored at the 128x4 swizzle offset | The same statement | AST; DEV RQ-B |
| `SF[off(p, c)]`, padding row `M <= p < PADDED_M` | Loop `for p in range(M + r, PADDED_M, M)`, one `tl.store(SF + offp, zeros[8])` per trip | `p = M + r + j0 + jj*M` for `jj < 128`, masked by `p < PADDED_M`, in one `[128, 8]` store of zeros per `j0` trip (one trip for every M ≤ 32). The swizzle offset is the same formula, vectorised. | WS: the same row set for each program (r, pid_c) at every M from 1 to 384, and every SF byte written exactly once. AST "pad offset = live formula, vectorised" and "pad value 0". DEV RQ-B uses 2 poisons, so a missing or extra byte shows up. |

The padding bytes are the constant 0, and no other kernel reads SF before rq completes: wo_b's GEMM is PDL-launched
and waits in `griddepcontrol.wait` for the whole grid. The new store therefore changes no observable value and no
ordering. The FTZ corner (a 32-block with denormal `amax/448`) runs the same instructions as before. The comparison
against flashinfer is reported apart, as in the r2-attnout harness.

## RPSPLIT: `_r5g_rp_kernel` vs live `_dsv41_rp_kernel` (dsv41_routeprep.py:93-252)

### Per element

- **One quant group** (128 bf16 inputs) produces 128 e4m3 bytes and one UE8M0 exponent. Both kernels compute it
  with the same statements:
  - `xf = bf16->f32` (exact)
  - `amax = max(max|xf|, eps)` (order-free)
  - `y = div_rn(amax, 448)`, then `max(|y|, 1e-10)`
  - `sb` = exponent bits, plus one when any mantissa bit is set
  - `inv = div_rn(1, scale)` (exact power of two)
  - `q = xf * inv` (one mul.rn, no FMA: `enable_fp_fusion=False`), then clamp ±448
  - the integer e4m3 encode (`tn`, `mag_n`, `fs = a + 16384.0`, `mag_s`, `mag_o`, select), then `byte | sign`

  The kit's `_r5g_rp_chunk` body is the live `for ch in tl.static_range(5)` body statement for statement. AST checks
  that every statement after the load is identical, and that the preamble constants `g`, `c`, `col`, `fmax_v` and
  `one_v` are the same statements. Only the load moved: `xa = tl.load(xrow + offs(ch))` is issued earlier, at the
  same address (AST "`_r5g_offs(ch)` returns the live offs expression"), and `xf = xa.to(f32)` sits where the live
  `tl.load(...).to(f32)` was. A load has no rounding, so moving it changes no bit.
- **Packed scale word.** `(v0>>23) | (v1>>15) | (v2>>7) | (v3<<1)` over the chunk's two 4-group packs. Same
  statements, integer only.
- **Plan** (histogram, 32-aligned cumsum, `start_e`, `cnt_e`, `al_e`, `rank`, `dest`, `first`). The statements are
  identical (AST "plan", "route"), and all of it is integer math. The histogram is a per-CTA shared-memory integer
  add, which is order-free. In split mode, each of the 5 programs of a route computes the same plan from the same
  inputs.

### Who writes what

`WS` emulates per-program duties for live, hoist and split, for M 1..10 and 4 routings. DEV compares every byte of all
five buffers under two poisons.

- **hoist.** Grid `(M*6, 1)`, so `y == 0` and `is0` is always true. The duties are identical to live.
- **split.** Grid `(M*6, 5)`. Program `(route, ch)` writes chunk `ch`'s 1024 row bytes and its 2 packed words. Only
  the `ch == 0` program does the rest:
  - the `inv_perm` / `layout` entry
  - segment padding (`first & is0`)
  - the strided dead-row fill: same `nprog = num_programs(0)`, so the partition is identical
  - `ends` (`pid == 0 & is0`)
  - the optional FC1 prefetch

  Every element is written by exactly one program, with the value the live kernel writes.

### Ordering and PDL

`gdc_wait` stays the first statement after the program ids. Nothing is loaded or stored before it, and PTX checks
that `griddepcontrol.wait` comes before the first global ld/st. The hoisted x loads come after it, so they read the
final x and topk ids. FC1, the PDL successor, reads rows, words and layout only after `cudaGridDependencySynchronize`,
which waits for full completion and a flush of every program. That is the same contract as today.
`EARLY_TRIGGER` and the `pf<N>` prefetch are unchanged (prefetch is a cache hint and runs in the ch-0 program).

### Scope and fallback

The kit kernel serves only calls with `M*6 <= 64`. Everything else runs the r2 kernel with today's launch, and so
does any call where:

- the synthetic sweep (first eager call: RQ every M 1..32, RP every M 1..10 x 2 routings) or an eager real-data
  self-check found a mismatch or an exception, or
- the call happens before the sweep passed (for example inside a capture).
