# r6-head exactness: op-order map, live vs r6

Live source: `dsv41_r4_small.py` sha256 3364d8c8 (`base/`), kernels `_xnorm_kernel` (l.311), `_screen_kernel` (l.329),
`_refine_kernel` (l.373), launched by `certified_logits` (l.408). Kit source: `files/.../dsv41_head_r6.py`.
SASS: `results/sass/*.sass` (sm_121, production specialization, `tools/compile_check.py`).

## 0. What must be equal, per lever

| lever | output | required relation to live | consumer |
|---|---|---|---|
| r6 xnorm | XG, XN, BAD, THR init | bit-identical (same body) | screen, cand |
| screen DQ=1/2 | HI [M, V] bf16, THR [M] fp32 | bit-identical | cand / refine |
| refine `tile` | OUT [M, V] bf16 | bit-identical, every element (-inf included) | `_sf_local_argmax_kernel` / `_finish` gather |
| refine `row` | OUT [M, V] bf16 | `where(row_pred, live_OUT, -inf)`, bit for bit | same; argmax-identical by the certificate (§3) |

`row_pred[v] = any_m (HI[m, v] >= THR[m])  or  any_m BAD[m]` (the live tile predicate evaluated per row).

## 1. xnorm

`_r6_xnorm_kernel` is the live body statement for statement, plus `if m == 0: tl.store(CNT, 0)` (CNT is a fresh
r6-only int32). XG/XN/BAD/THR: identical code, identical bits. The extra store is stream-ordered before the screen and
`_r6_cand_kernel` of the same call (same stream, sequential launches).

## 2. Screen (DSV41_R6_HEAD_SCREEN)

Per HI element (m, n) the live sequence is:
1. `acc = 0`; for k0 = 0, 128, ..., 5120-128: `acc = dot(x[m, k0:k0+128] (bf16), wd[k0:k0+128, n] (bf16), acc)`,
   lowered to `HMMA.16816.F32.BF16` k16 steps in ascending k (16 per K step at BM 16, 32 at BM 32; SASS census).
2. `wd[k, n] = cvt.rn.bf16.f32( f32(q[n, k]) * bitcast_f32(e[n, k/32] << 23) )`.
3. `qb = sum_g xg[m, g] * rt[g, n]` in g order (FFMA/FMUL+FADD as compiled, same code both sides), bound, slop,
   lo/hi rounding, non-finite handling, padding mask, bf16 store of hi, `atomic_max(THR[m], max_n lo)` per CTA.

r6 changes step 2 only:
- `f32(q)` is computed as `bitcast_f32(int32(q) + 0x4B400000) - 12582912.0`. For every int8 q this is exactly f32(q)
  (sign of zero included). No I2F in the r6 K loop (SASS: live 64 I2F per K step, r6 0).
- The exponent byte is loaded once per (n, 32-group), not once per weight, and broadcast over the 32 weights
  (`tl.reshape` [BK, BN] -> [BK/32, 32, BN]). Same byte, same `<< 23` bitcast.
- The product `qf * sc` is the same fp32 multiply. DQ=1 rounds with the same `cvt.rn.bf16.f32`. DQ=2 takes the top 16
  bits, which equals RN because the product is exactly representable in bf16: |q| <= 128 has at most 8 significant
  bits, sc is a power of two, and for e in 1..254 the product is 0 or a normal f32 (no subnormal; overflow gives
  inf, whose top half is the bf16 inf).
- **Proof:** `tests/test_dequant.py`, exhaustive over q in -128..127 and e in 0..255: DQ=1 == live and DQ=2 == live
  for all 65,024 pairs with e in 1..254 (the only range `make_mxint8` emits, l.193). Outside it (e = 255, q = 0) DQ=2
  gives NaN 0x7FC0 vs 0x7FFF; the module checks `1 <= e <= 254` over the whole table once and falls back to the live
  screen otherwise.

wd is identical, so the HMMA inputs are identical; the dot is the same `tl.dot(x, wd, acc)` with the same BM, BN, BK
at the default config, and the same HMMA opcode and count per K step (compile check). Everything after the K loop is
the live text. Hence HI and THR are bit-identical at the default config (BN 64, BK 128, 4 warps, 3 stages).

**Non-default screen configs** (`DSV41_R6_HEAD_SCREEN_CFG`):
- BK changes only how many k16 HMMA steps one loop iteration issues. The MMAv2 lowering chains the accumulator through
  k16 steps in ascending k, inside a tile and across iterations, so the per-element sequence is unchanged.
- BN, warps and stages change neither the per-element K order nor the instruction.
- THR caveat: with a different BN, the per-CTA `max_n lo` partials cover different column sets. For values other
  than +-0 the max is the same whatever the partition. Triton lowers a float `atomic_max` to sign-split integer
  atomics, and `tl.max` may return either zero, so a partition change could matter only when some lo is -0.0 (not
  verified against the 3.7.1 lowering; the argument below does not need it). lo = rn_bf16((acc - B) - slop) with
  slop >= 2^-126 is never exactly -0. It is a rounded -0 only if 0 < |(acc - B) - slop| < 2^-133.
- Every non-default config is therefore admitted only if `device_test.py` prints it as ELIGIBLE: bit-identical HI and
  THR at every swept M and case.
- Coverage (round-6 review fix B1). The sweep runs one M per served BM bucket: M ∈ {2, 7, 24, 48, 96}, i.e. BM 16, 16,
  32, 64, 128. BM 64 is where `num_warps` becomes `max(w, 8)`. The module enforces the same bounds:
  - A non-default shape applies only at BM ≤ `SWEPT_MAX_B` = 128. Above that, the live shape serves.
  - Any non-zero `SCREEN_CFG` field requires `SCREEN_DQ=2` (ValueError at import), because non-default shapes are
    swept with DQ2 only. DQ1 builds the same wd bits by the exhaustive proof, but the device sweep does not run it at
    those shapes.
  - `tests/test_cfg_guard.py` checks these rules, with must-fail mutants that drop each guard.

## 3. Refine (DSV41_R6_HEAD_REFINE)

Live, per needed 16-row tile t (pid = t): rows j = 16t..16t+15, `acc[16, BMX] = 0`; for k0 ascending:
`acc = dot(W[j, k0:k0+128], x^T[k0:k0+128, :BMX], acc)` (A = W rows, B = x^T; `HMMA.16816.F32.BF16`, 8 per K step at
BMX 16); store `acc.to(bf16)` at OUT[m, j]. Not-needed tile: store -inf. Needed iff
`any_{m<M, j in tile} (HI[m, j] >= THR[m])` (masked loads: HI -inf for j >= V or m >= M, THR +inf for m >= M) or any
BAD.

r6, two kernels:
- `_r6_cand_kernel`, CTA of CB rows:
  - Same masked loads and the same `>=` compare. `hrow[j] = any_m`; `htile` is the OR over each aligned 16-row group,
    which is the live tile predicate, since CB is a multiple of 16 and the groups start at multiples of 16.
  - `tile`: sel = htile broadcast. `row`: sel = hrow. Then `sel |= any BAD`, `sel &= j < V`.
  - Not-selected (m < M, j < V) get -inf, the same bf16 constant as live. Selected j are appended to IDX: one int32
    atomic per CTA reserves a contiguous slot range, and an exclusive prefix sum orders the rows within it.
- `_r6_refine_kernel`, a persistent grid G that reads n = CNT. For each chunk c (pid, pid+G, ...), the rows are
  IDX[c*rpc : c*rpc+rpc] in a 16-row MMA tile (masked rows load 0 and are not stored), followed by the live K loop
  text verbatim: same `tl.dot(w, xt, acc)` with W rows as A and x^T as B, same BK (live 128), same BMX, same
  ascending k0, same `acc.to(bf16)` store at OUT[m, j].
  - An MMA output row depends only on its own A row and on B, so which other rows share the tile, or which CTA
    computes it, cannot change its bits.
  - SASS: identical opcode `HMMA.16816.F32.BF16`, 8 per K step, same as live (compile check at M 7 and 32).

Write sets are disjoint and complete. The cand kernel writes exactly the not-selected (m, j), and the refine kernel
writes exactly the listed (selected) j. Each selected j is listed once: one atomic range per CTA, and the coverage
test runs n = 0..16160 x grid x spread. So every (m < M, j < V) is written exactly once.
- `tile`: selected = the live needed tiles, so OUT == live OUT bit for bit.
- `row`: OUT == where(row_pred, live OUT, -inf).

**Argmax identity of `row` (certificate, `dsv41_r4_small` module doc).**
- For each m, `lo_v <= l_v <= hi_v` for every real row, so `THR[m] = max_v lo_v <= max_v l_v`.
- A row with `hi[m, v] < THR[m]` therefore has `l_v < max l` for that m. It is not the max and not tied with it.
- Every row attaining the max (every tied row included) has `hi >= l = max >= THR` and is computed with its exact
  bits.
- `_sf_local_argmax_kernel` (leftmost max per shard) and `_sf_onehot_kernel` (first rank holding the max) therefore
  return the stock leftmost argmax. Rows selected only through another m are computed exactly too, and that is
  harmless.
- With a BAD row every row is computed, as live does.
- Without SYNC_FREE (`_finish` gather) the consumer on an armed step is the same greedy argmax.

Tests:
- `tests/test_refine_logic.py`: all 9 case families x M in {1, 2, 7, 16, 17, 33, 96}, including forced ties across
  tiles, inside a tile and across shards, plus the sharded leftmost-max exchange.
- `device_test.py --part ties`: real kernels with ties at the cand-CTA boundary, the 4096 sf block boundary and the
  shard end, at every M in 2..96.

**Non-default refine configs** (BK 256/512, stages, grid, spread). The same MMAv2 argument as the screen applies; they
are admitted only from `device_test.py`'s ELIGIBLE list.
- BK and stages are used only at BMX 16; larger BMX keep the live BK/stages, because of shared memory.
- Grid and spread act at every BMX ≤ `SWEPT_MAX_B` = 128. The sweep therefore runs them at one M per BMX bucket
  (M 2, 7, 24, 48, 96, i.e. BMX 16, 16, 32, 64, 128), in tile and row mode. Above 128, grid 0 and spread 1 serve.
- Refine warps is not swept, so the module refuses any non-zero value at import: warps is always the live XWARPS.

**Reference anchor.** Every device comparison uses `_run(st, x, False, "0")`, a copy of the live launch sequence.
`device_test.py --part anchor` checks it against the serving `r4.certified_logits` (flags off), bitwise, at every M.
`--part dispatch` runs the installed 7-line dispatch, `certified_logits`, the self-check, `_done`, the stats and the
fallback on the GPU.

## 4. Flag off

With `DSV41_R6_HEAD_REFINE` and `DSV41_R6_HEAD_SCREEN` unset, `0`, `off`, `false`, `no` or empty, `_DSV41_R6_HEAD` is
False:
- `dsv41_head_r6` is never imported.
- `certified_logits` evaluates one False test and runs its unchanged body.

The patched file minus its 7 tagged lines is the live file byte for byte. The AST differs only in the flag
assignment, the gated import and the dispatch statement, and every other function is AST-identical
(`tests/test_install.py`).

## 5. Races, graphs, syncs

- One stream. xnorm (zeroes CNT) -> screen (atomic_max THR) -> after_screen hook -> cand (reads HI/THR/BAD, atomics
  on CNT/NEED/STATS, writes IDX and -inf) -> refine (reads CNT/IDX, writes listed rows). Every consumer is a later
  launch on the same stream, so no cross-stream edge is added.
- `after_screen` (r1-drafter G12 hook, off in production) is called at the same point as live: between screen and
  refine.
- Buffers (out, xg, xn, bad, thr, hi, idx, cnt) are fresh per call from the caching allocator, as live's are. STATS
  (2 x int64 per CertState) is the only persistent buffer. It is touched only by atomics and read only by the
  stats log.
- Graph capture: fixed grids, no host reads in `_run`. The first call per CertState (table check) and an unchecked
  M are never run under capture: the live kernels run instead. The stats read is skipped while capturing.
  `device_test.py --part graph` replays a captured r6 pass on 3 inputs and compares with eager.
- Host syncs: one at the first r6 call (exponent range check), one per new M while SELFCHECK=1, and one per
  STATS_EVERY calls (default 2000).
