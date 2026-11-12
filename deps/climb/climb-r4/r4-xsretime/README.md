# r4-xsretime: re-time the paced attention-window L2 prefetch for XSPLIT

This is a default-off kit for production launcher b22e4aa9. It patches the live `dsv41_l2_prefetch.py`
(sha256 `8bc9a1b5…`, which already carries the r1-pf patches, identical on 8/8) and adds one new module,
`dsv41_l2pf_xsr.py`. It only changes cache hints (L2 prefetch), so outputs are exact by construction.

## Why XSPLIT's attention saving does not reach the step

Data: t1 traces (`~/climb-traces/t1/{prod,xs}/q{0,4}.{1,2,3}.sqlite` on 512S1). For each trace, each target
dual sparse-MLA call is one sample: 60 steps × 37 layers = 2220 samples. Scripts and raw output are in
`~/Documents/dsv41-offline/climb-r4/xsretime/` (`scripts/layer.py`, `deep.py`, `model.py` → `deep.txt`, `model.txt`).
All numbers below are µs per layer: prod → xs.

| | code r0 / r4 | prose r0 / r4 | maths r0 / r4 |
|---|---|---|---|
| sparse MLA | 51.8→39.0 / 51.7→39.5 | 48.4→34.0 / 48.2→34.4 | 54.4→42.6 / 54.4→42.6 |
| attn pacer end − wo_b GEMM start | +9.1→+21.8 / +9.4→+21.1 | +13.4→+26.9 / +13.8→+26.3 | +5.9→+17.6 / +6.1→+17.6 |
| wo_b GEMM (MXFP8, 5120×1024) | 15.1→19.8 / 15.2→20.0 | 17.7→21.9 / 17.6→22.2 | 14.4→18.5 / 14.3→18.6 |
| post-attn AR pacer start − AR start | 3.4→8.3 / 3.5→7.9 | 4.4→10.1 / 4.6→9.6 | 2.1→6.7 / 2.2→6.3 |
| MLA end → post-attn AR end | +5.5 / +4.3 | +6.1 / +4.4 | +3.9 / +3.6 |
| MLA end → FC1 start | +8.2 / +8.3 | +8.5 / +8.1 | +6.4 / +7.3 |
| FC1 | +3.5 / +4.0 | +1.8 / +2.7 | +4.3 / +3.9 |
| MLA end → post-MoE AR end | +11.7 / +10.5 | +12.2 / +10.7 | +11.0 / +10.7 |
| MLA end → next MLA end (layer period) | −1.2 / −1.0 | −1.9 / −1.9 | −0.6 / −0.5 |

How the loss happens, step by step:

1. The attention window covers wo_a (4.12 MiB) and wo_b (5.16 MiB). Plan log, spark-one rank0.log
   `g2 plan tokens=4`: `[L1 …wo_a:0.12 …wo_a:4.00 …wo_b:0.16 …wo_b:5.00]`. It is paced at 125 GB/s, so one pacer
   kernel takes 78.9–79.3 µs.
2. Paced parts of every kind run on one stream, `_side_stream3()` (base `_fork_part`, live lines 1068–1076). That
   stream is a FIFO.
   - About 90% of attention pacers start ≤ 1.5 µs after the previous pacer ends (`attn_q`); that previous pacer
     is the qkv_a all-gather window: wq_b 5.16 MiB + wo_b 0.85 MiB.
   - The wait after the fork (the end of the last wq_b GEMM) averages 11.0–11.6 µs (`fifo_wait`). In
     29 of 37 layers it is 12–16 µs. In layers 0/6/12/18/22/26/30/34 it is about 1 µs (prose r0 per-layer table in
     `deep.txt`).
   - About 70–80% of post-attention AR pacers then queue behind the attention pacer (`ar_q`).
3. The pacer start lands about at the MLA start in both configs (`pace_s_vs_mlas` −1.5…0.0). With XSPLIT,
   attention is 12–14 µs shorter, but the pacer still needs 79 µs. It therefore issues wo_b while the wo_b GEMM
   runs:
   - Overrun goes from +6…+13 µs to +18…+27 µs.
   - The GEMM drops from L2 speed to DRAM-contended speed. wo_b with the pacer already done at GEMM start:
     13.0–15.6 µs in both configs. With the pacer still running: prod 14.7–18.3, xs 19.5–22.7.
4. Effects further down the layer:
   - The post-attention AR window (router gate, 3.75 MiB) starts 4–6 µs later.
   - The router, mHC projection and shared-expert w13 slip by 0.4–3.9 µs each.
   - FC1 grows 1.8–4.3 µs. No pacer overlaps FC1 (`fc1_pace` = 0). Its overlap with the side-stream shared
     expert grows by about the same amount (`fifo.txt`, FC1 columns; correlation +0.12…+0.42). So this part is
     a scheduling side effect, not a direct pacer effect.
   - Net: MLA end → post-MoE AR end grows 10.5–12.2 µs against the 11.8–14.3 µs saved in MLA, and the layer
     period drops by only 0.5–1.9 µs.
   - Kernel duration sums, prose r0, per step: MLA −549 µs; MXFP8 dense GEMMs +206; `ar_kernel_pl<0>` +212
     and `<2>` −53; FC1 +65; mhc_proj +71.
   - AR durations themselves change by ±1–3 µs with opposite signs on r0 and r4 (peer waits). The step loss is
     on the main-stream timeline, not inside the collectives.
5. Production is affected too, on a smaller scale. 76–84% of layers overrun into wo_b, and wo_b loses
   1.6–4.1 µs per such layer.

## What the kit changes

All three flags are default off. They are read once at import. Each one is a new `DSV41_L2_PREFETCH_*` flag.

| flag | effect |
|---|---|
| `DSV41_L2_PREFETCH_PACE_GBPS_ATTN=R` | Paces only the attention windows at R GB/s (bound 0 < R ≤ 400). ar/ag/post windows keep `PACE_GBPS`. |
| `DSV41_L2_PREFETCH_PACE_ATTN_STREAM=1` | Moves the paced attention windows to their own 4th side stream. The stream is created at install, outside any capture, and joined at the end of the bracketed forward, as stream 3 is. The window then starts at its fork, the last wq_b, instead of behind the all-gather pacer. The post-attention AR pacer no longer queues behind it. |
| `DSV41_L2_PREFETCH_ATTN_DEDUP=1` | Removes from each attention window the bytes that the collective window holding its fork already prefetched. Today those are wo_b scales plus 0.69 MiB of wo_b in non-indexer layers. Windows only shrink; nothing is backfilled. |

The existing knob `DSV41_L2_PREFETCH_WOB_MB=3` gives one more re-timing arm with no code: a shorter attention
window.

The prompt also suggested moving wo_b bytes to a later window. That is impossible: no window opens between the
attention fork and the wo_b GEMM; the next fork is the post-attention AR, which comes after wo_b. The useful move goes
the other way: the earlier all-gather window already holds the head of wo_b, and the dedup flag removes that
duplicate.

Modelled result. The script replays the measured per-layer timeline under each arm (`model.txt`), assuming that
main-stream kernels do not slow down. Values are the mean attention-pacer overrun into wo_b, in µs, for xs
code/prose/maths on rank 0:

| arm | overrun (code / prose / maths, rank 0) |
|---|---|
| baseline | +25.8 / +30.6 / +21.6 |
| R160 | +8.5 / +13.3 / +4.3 |
| R200 | −3.9 / +0.9 / −8.1 |
| OWN | +14.2 / +19.1 / +10.1 |
| OWN+R160 | −3.1 / +1.8 / −7.2 |
| R200+DEDUP | −8.3 / −3.6 / −12.5 |
| OWN+R160+DEDUP | −8.7 / −3.8 / −12.8 (done before wo_b in 58–95% of layers) |
| WOB3 | +7.7 / +12.5 / +3.5 |

## Patch (6 anchors, 24 lines, all tagged `# r4-xsretime`, backup `.bak-r4-xsretime`)

Anchors are on the live bytes, including the r1-pf lines:

1. `install_model_module`: `_XSR.install()` after `_R1.install(mod)`. This creates stream 4 outside capture.
2. `_bracketed` finally: `_XSR.join()` after `_join3()`. This is the end-of-forward join.
3. `make_g2_plan`: `fl = _XSR.retime_lists(fl, events)` after `g2_fork_lists`. This applies the dedup as a pure
   list edit.
4. `_parts` head: for `fork_kind == "attn"`, `_XSR.attn_parts` returns a burst head (if any) plus a 4-tuple paced
   part `"y"`, or `None`, which falls back to the base path. If the pacer library is missing, it returns `None`
   and the base burst fallback runs, as it does today.
5. `_fork_part`: `"y"` parts go to `_XSR.fork_part`. Same fork protocol: `side.wait_stream(main)` at the wq_b
   post-hook, the live pacer, the attn rate, stream 3 or 4.
6. End of module, after the r1-pf gate: the gate and import. The module is imported only when `ENABLED` and a
   master flag is set to something other than `""`/`0`/`off`/`false`. `_XSR` stays `None` if no lever is active.

## Why flag-off is identical

With every master flag unset, `dsv41_l2pf_xsr` is never imported and each inserted line is a false
`_XSR is not None` test. `tests/test_xsr.py` proves this four ways:

- The patched file minus its tagged lines equals the live base byte for byte.
- The ASTs are equal once the `_XSR` statements are stripped.
- Full launch logs (op, stream, segment table, chunk, period, delay, cap, waits, collectives; captures at sizes
  1/4/6; launcher l2pf environment including R1 KVSRC/SHW2) are identical to the live base under prod, PACE_HEAD,
  REPLAN=attn and WOB_MB=3.
- Settings that are set but not applicable (rate 0.0, invalid, above 400, attention not paced, no pacing) log
  INACTIVE and also produce identical logs.

## Why flag-on is exact

The kit changes only which bytes the live `l2pf_paced_kernel`/`l2pf_kernel` prefetch, at what period, and on
which side stream. Those kernels issue `cp.async.bulk.prefetch.L2` hints, read only their kept-alive segment
table, and write nothing (`stats` is null). Main-stream kernels, their inputs, order, tiles and reductions are
untouched. Every output bit is therefore identical. XSPLIT itself is the r3-xsplit kit, device-tested bitwise.

## Race and graph analysis

- The stream-4 branch forks with `side4.wait_stream(current)`. Under IDXPAR, "current" is the indexer-q stream
  `hi`, exactly as for the base fork (live `dsv41_idxpar.py:234-235`).
- The branch writes no memory.
- It is joined into the main stream in `_bracketed`'s `finally` before `_finish`, so every capture closes the
  branch (the same as `_join3`).
- Segment tables are created at learn time (eager) and kept in `S.keepalive`.
- No allocation or stream creation happens inside a capture.
- Target plans only. DSpark drafter plans (r1-pf "x" parts) are untouched.

## Tests

`python3 tests/test_xsr.py` passes 16/16 on 512S1 (torch 2.11 CPU); see `TEST-RESULTS.txt`. The tests cover:

- anchors on the live base and kit shas;
- the installer cycle on a temporary copy: status, dry, install, idempotent re-install, refusal on a foreign
  edit, uninstall, refusal on a drifted base;
- flag-off identity (text, AST and launch logs);
- the rate lever: only attention periods change, same bytes and order;
- the own stream: attention pacers move to side4, each launch is preceded by its fork wait, and the join comes
  at the end after side3;
- dedup: windows only shrink, removed bytes ⊆ the owner window, nothing the owner window issued is kept, and
  removed bytes = 3 layers × 0.84 MiB × 3 captures;
- all three levers combined;
- the PACE_HEAD burst head stays the base burst;
- fallback when the pacer library is missing;
- fail-closed dedup;
- `subtract` against brute force (3000 random cases);
- the owner-window mapping.

## How to read the effect in nsys

Run `scripts/deep.py` from `climb-r4/xsretime/` against one capture per arm. Check three things:

- `pace_end_vs_wobs` should be ≤ 0 in most layers.
- `wob` should return to 13–15.6 µs.
- `E_to_ar2end` (xs − prod) should go from +10.5…+12.2 µs toward 0.

With `PACE_ATTN_STREAM=1`, `fifo_wait` should be about 0 and `ar_q` should fall.

## Risks

- A faster attention pacer puts more DRAM pressure on the indexer, MLA and wo_a.
  - R200 is close to GB10's ~273 GB/s peak.
  - Burst attention prefetch was measured dead (−0.5%), and so was a global pace rate of 100/150.
  - These arms are per-kind only.
- Two pacers run concurrently with OWN. The tail of the all-gather window and the head of the attention window
  then overlap for about 13 µs, and later the attention tail overlaps the router-gate window.
- Dedup assumes the head of wo_b, prefetched about 100 µs earlier, is still in L2 (24 MiB L2; about 15 MiB is
  touched in between). If it is not, the GEMM re-reads 0.85 MiB.
- All of this is cache hints, so the worst case is a slower step. Exactness cannot be affected.
