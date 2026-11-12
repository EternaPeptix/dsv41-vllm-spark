# r2-draftmoe — G17 (audit `moe:L-D`): DSpark drafter MoE compaction

One kit, one installer, one flag: **`DSV41_DRAFT_MOE_G17`** (default off). Status: **partial** — FC1 compaction and
the fused permute are built; FC2 K 384→288 is skipped (reason below).

## What it does

The drafter (mtp.0-2: 128 routed experts, top-3) still runs the padded MoE layout and the stock permute glue; the target
MoE moved off both in October (rank0.log:221 `N768/K5120 FC1, K384 FC2`;
`dsv41_draft_m32.py:41 DRAFT_GEOMETRY = (768, 5120, 3, 128)`; mfront gate `deep_gemm_utils.py:609` requires E=384,
top-6). Per TP8 rank the drafter has 288 live intermediate columns, padded to 384 for both gate and up.

| value | effect |
|---|---|
| unset / `0` | today's code path. The new module is never imported. |
| `fc1` | drafter w13 is created unpadded `[128, 576, 2560]` (the drafter twin of the promoted target `DSV41_MOE_COMPACT`); FC1 runs at N=576 on the M32 template (M ≤ 160) or the 128-row DeepGEMM path (M > 160); the existing compact SiLU writer emits the 384-wide FC2 input with its 96 zero columns. Frees about 190 MB per rank. |
| `perm` | the g3 mfront fused permute (plan + scatter, 2 kernels) for the drafter geometry on the M32 path, instead of count / zeros / full / ep_scatter_1 / ep_scatter_2 / transpose_and_pack (6 kernels). |
| `1` | `fc1` + `perm` (treatment arm) |
| `permverify` | diagnostic: serve the stock permute, shadow-run the fused one through the existing mfront verifier (combine as `fc1,permverify`) |

Files (relative to `site-packages/vllm`; base sha256 in KIT.json):
- `model_executor/layers/quantization/mxfp4.py` — 9 tagged lines: module flag prelude; in `Mxfp4MoEMethod.create_weights`
  the drafter gate (`dsv41_r2_draftmoe.drafter_compact_eligible`: E 128, top-3, hidden 5120, padded 384 / unpadded 288,
  TP8, no EP/EPLB/bias/LoRA, SiLU, DeepGEMM-MXFP4, SM121, and `DSV41_MOE_COMPACT=1` because the compact SiLU writer
  asserts it) sets `w13_intermediate_size = 288`.
- `model_executor/layers/fused_moe/experts/deep_gemm_moe.py` — 11 lines: `apply()` sets `compact_fc1 = True` for
  the compact drafter geometry, after the target's own two asserts, with the drafter's own two (`w2` is
  `(128, 5120, 192)`, SiLU). At the top of `_workspace_m_alignment`, the compact drafter returns exactly the padded
  drafter's decision (`_workspace_m_alignment(M, 768, ...)`). `_act_mul_quant` is not edited: the gate requires
  `DSV41_MOE_COMPACT=1`, so its existing assert holds.
- `model_executor/layers/fused_moe/experts/dsv41_draft_m32.py` — 6 lines: the `DSV41_DRAFT_MOE_M32=verify` shadow
  accepts N=576.
- `model_executor/layers/fused_moe/deep_gemm_utils.py` — 17 lines: drafter branch in `deepgemm_moe_permute` calling
  the live `dsv41_fused_permute` (and the live verifier for `permverify`).
- new `model_executor/layers/fused_moe/experts/dsv41_r2_draftmoe.py` — flag parse, the two gates, log lines.
- optional pre-JIT cache entries (`install.py prejit`): the three new FC1 instances (M32 `15880fe8`, and the two
  128-row tile configs `3a6d4662` (128x128, 2 stages) and `5a63770a` (128x64, 3 stages)). See COMPILE-LOG.txt.

Every edit is a **pure insertion** of lines that end in `# r2-draftmoe`. Anchors are kept verbatim, nothing is
replaced and no untagged line is added. Removing the tagged lines therefore gives back the exact pre-install file;
`tests/test_install.py` checks this byte for byte.

## Install (window, all 8 nodes)

```
# order: r1-pdl (if used) -> r2-moefront (if used) -> r2-draftmoe
python3 install.py install          # refuses unless every file is an accepted base (below); idempotent
python3 install.py prejit           # optional, CPU-only nvcc at nice 19 (~1-3 min per kernel); else cold JIT
python3 install.py status
# uninstall: LIFO (r2-draftmoe -> r2-moefront -> r1-pdl)
python3 install.py uninstall        # strips exactly our tagged lines after verifying them; never restores a .bak
```
**Accepted base** for each patched file: the live sha256 in KIT.json, or a file whose `# r1-pdl` / `# r2-moefront`
tagged lines strip back to that sha. KIT.json also records the composed shas as examples (deep_gemm_utils.py:
r1-pdl `c3a05cca`, r2-moefront `9ce425b5`, both `97f26924`; deep_gemm_moe.py with r2-moefront `b2c3b833`).

**Uninstall** works per file:
1. It removes the lines ending in `# r2-draftmoe`.
2. It checks that the result is an accepted base and that the file equals our patch of that result. Lines of another
   kit that sits at the same anchor are allowed.
3. Only then does it write. Every file is checked before any file is written.

A file that no longer has our lines is left as it is, for example after another kit's blind restore. It is never
"restored" to our `.bak`. The `.bak-r2-draftmoe` copies are informational, and uninstall deletes them. The pre-JIT
cache entries stay; they cannot be reached with the kit off.

**Composition, tested with the real installers** (`tests/test_compose.py`):
- r1-pdl → r2-moefront → ours, then LIFO uninstall, gives every file back byte-identical.
- With our lines present, r1-pdl's and r2-moefront's uninstallers refuse and change nothing, so FIFO cannot silently
  drop our edits.
- The review's blind-restore case ends at the live `58e264c8`, not at `c3a05cca`.
- Ours first, r2-moefront on top: r2-moefront's installer *as of this fix* refuses our lines, because it strips only
  `# r1-pdl`. That is why r2-moefront must be installed first. A copy of it that also strips `# r2-draftmoe` installs
  on top of ours, and uninstall in both orders gives the exact other-kit-only files.

The flag must reach the worker environment the same way the round-1 env flags do (it is not in the launcher).
The flag must reach the worker environment the same way the round-1 env flags do (it is not in the launcher).

## Why flag-off is identical

Each patched module reads `DSV41_DRAFT_MOE_G17` once at import; unset/0 leaves `_R2DM_FC1 = False`, `_R2DM_PERM = ""`
and never imports the new module. Every inserted statement is guarded by one of those names (`_R2DM_FC1 and ...`,
`bool(_R2DM_PERM) and ...`); no original line is replaced.
`tests/test_install.py` folds the flags to off in the patched AST and checks it equals the base AST, for all four
files and for the r1-pdl-composed `deep_gemm_utils.py`; `tests/test_plan.py` runs the patched `apply()`,
`_workspace_m_alignment` and `deepgemm_moe_permute` with stubbed kernels and checks the call sequence (kernels,
shapes, arguments) equals the base's for the target, the padded drafter and M = 1..400. Weights, workspaces, streams,
events and CUDA-graph contents are therefore unchanged. The Athena FC2 overlay, which source-transforms
`DeepGemmFP4Experts.apply` at startup, still finds each of its three anchors exactly once (tested with Athena's own
`transform_apply`).

## Why flag-on is bit-exact

See EXACTNESS.md for the element-by-element map. In short: FC1 compaction removes whole zero output columns; every
remaining column runs the identical `mma ... m16n8k32 ... block_scale` chain over the same A row, B row and scale bytes
in the same K order from `+0`, rounded once to bf16; the compact SiLU writer feeds `+0` where the padded FC1 produced
`+0`, so the FC2 input is byte-identical; FC2 and everything after are untouched. The fused permute is integer and copy
work with the same scale bit-pack formula; only intra-expert row order (atomic order, nondeterministic today too)
differs, and every downstream kernel is row-independent with the gather reading through `inv_perm`. Drafts are
therefore bitwise unchanged, so AL, verify width (adaptive verification is pinned, launcher pin) and the outputs are
unchanged.

## Races and CUDA graphs

No new streams, events or host syncs. The fused permute runs on the drafter's main stream at the position of the
stock glue, exactly as the target's mfront does; its four small allocations (layout, packed scales, inv_perm, cursor)
replace the stock path's six (expert counts, start_loc, fp32 scales, layout, inv_perm, and DeepGEMM's packed-scale
output). Weights are created at load, before capture. FC1 workspace use shrinks (`M_sum x 576` instead of `x 768`),
inside the buffers `workspace_shapes` already sizes. Works for every captured drafter size (M ≤ 160 → M32 + fused
permute; M > 160 → 128-row path with compact FC1, stock permute) and is independent of the target graph set.

## Expected saving (trace evidence: audit verify-G17-combined, 512S1 traces, d200 windows, both ranks)

All drafter MoE work is serial on the drafter graph's main stream (overlap with other streams < 1 µs at rank 0,
< 7 µs at rank 4; CPU launches run 15-40 ms ahead, so the GPU is the bottleneck): **on the critical path**.

| item | today per step (3 layers) | change | saving per step |
|---|---|---|---|
| drafter FC1 `sm120_fp8_fp4_gemm_1d1d_impl<0,768,5120,…,128,…>` | code 271-277, prose 242-245, maths 231-234 µs | N 768→576 = 25% fewer weight bytes; target precedent 1.17x at M ≥ 6, 1.084x at M = 1; drafter touches 5-12 experts per call → 9-15% | 21-41 µs (central ~30) |
| drafter glue: ep_scatter_2 22.0, transpose_and_pack 11.0, ep_scatter_1 7.9, count 4.2, fills ~5 µs | ~50 µs | → plan + scatter, ~5.6 µs/layer on the target | raw ~33-36 µs |
| shared-expert overlap (stream s16 hides 21 µs/layer behind the glue today) | — | shorter glue pushes ~12 µs/layer of s16 work under FC1, which slows FC1 by 0.35-0.6 µs per µs of overlap | gives back 13-21 µs |
| **total d** | | | **low ~35, central ~55, high ~75 µs per step** |

G1' = mean 1000·AL/(T − d) with T = 24.26 / 21.33 / 24.04 ms and AL 2.926 / 1.843 / 3.613 →
**+0.15% / +0.24% / +0.32%** (low / central / high). The skeptic's +0.29% central includes ~12 µs of FC2 K-tail, which
this kit does not build.

## How to read it in nsys

Drafter MoE layers are the three MoE blocks inside the drafter graph (3 per step, 180 per 60-step window), identified
by the E=128 kernel instances. With `DSV41_DRAFT_MOE_G17=1`:
- **gone** from drafter layers: `_count_expert_num_tokens` (or the count kernel `anat3.py` buckets as glue), the
  fill kernels (`vectorized_elementwise_kernel<…FillFunctor…>` for zeros/full), `_fwd_kernel_ep_scatter_1`,
  `_fwd_kernel_ep_scatter_2`, `transpose_and_pack_fp32_into_ue8m0<512, 48, 40, 128, 1, false>`;
- **new**: `_dsv41_fp_plan_kernel` and `_dsv41_fp_scatter_kernel` per drafter layer (3 + 3 per step; the target
  already shows 40 + 40 of them);
- **shorter**: drafter FC1 `sm120_fp8_fp4_gemm_1d1d_impl<0u, 576u, 5120u, 128u, 32u, 128u, 32u, 64u, …>` replaces
  `<0u, 768u, 5120u, …, 128u, …>`; expect 9-15% less per call (bigger at larger M);
- unchanged: drafter FC2 `<0u, 5120u, 384u, …, 128u, …>`, the SiLU kernel count, the gather, the target layers;
- the topk-end → FC1-start window per drafter layer drops from ~23.5 µs to ~11-12 µs; the s16 shared-expert kernels
  (MXFP8 gate_up, act_and_mul, quantize, down) then overlap FC1 instead of the glue.
`anat3.py` buckets are regex classes: check that the drafter FC1 and the new permute kernels land in the buckets you
compare.

## Risks

- **Cold JIT**: without `install.py prejit`, the first drafter call after boot JIT-compiles the N=576 M32 instance
  (once per process, during warm-up/capture; ~1-2 min of boot time). A drafter call with M > 160 (prefill-sized)
  would JIT a 128-row instance on first use, mid-serving, unless prejit ran. The live default cache holds two
  padded-drafter 128-row configs (5559d059 128x128/2 stages, 2201c753 128x64/3 stages), and prejit builds the N576
  analogue of both (3a6d4662, 5a63770a). The analogy holds for both target configs: 0e915d63 -> ec8c0dc5 and
  4ea90a59 -> 8135464c are both live entries that DeepGEMM compiled itself. After the first long prompt, check that
  the cache has no other new E128 N576 key. Exactness does not depend on the tile config.
- **prejit vs a running JIT**: prejit never touches an entry that exists, whether committed or not. `--force` refuses
  while a vLLM process runs on the node and moves the old entry aside instead of `rmtree`. Still, do not run prejit
  while a process is JIT-compiling.
- **Interaction with r1-pdl** (`DSV41_GLUE_PDL`, 'fp' scope): its edit routes every `dsv41_fused_permute` call through
  its PDL kernels, so with both flags on the drafter permute is PDL-launched too, and those kernels were qualified
  only at E=384. Gate: on a node with r1-pdl installed, run `DSV41_GLUE_PDL=fp device_test.py` to exit 0. Then run
  the pair arm `g17+pdl` (`DSV41_DRAFT_MOE_G17=1 DSV41_GLUE_PDL=1`, KIT.json). Do not combine them without both.
- **DSV41_WEIGHT_DIGEST**: with `fc1` on, the drafter w13 params are `[128, 576, 2560]`, so their digest differs.
  This is expected and is not a regression. All other weights are unchanged.
- **Gate mismatch**: if the drafter's MoE config differs from the assumed one (unpadded 288, TP8, SiLU, ...) the
  layer stays padded and the log says `NOT compacted`; behaviour is then today's, so check the 3 `compact FC1 enabled`
  log lines.
- Memory goes down ~190 MB/rank (KV cache size at boot grows slightly; no arithmetic depends on it).
- Not compiled or GPU-tested by me (cluster lock held throughout): `device_test.py` is the gate.

## Window protocol for this kit

1. Install 8/8 in the order r1-pdl → r2-moefront → r2-draftmoe, then prejit 8/8.
2. Run `device_test.py` on a stopped node: spark-eight first (exit 0), then with `--real-weights` on **every** node,
   because each rank has different expert slices. This must pass before any timing arm:
   `~/venvs/vllm032/bin/python device_test.py --real-weights --json ~/dsv41-logs/r2dm-device.$(hostname).json`
   needs exit 0 (~3-6 min, plus JIT). Where r1-pdl is installed, also run it with `DSV41_GLUE_PDL=fp` (pair gate).
3. flags-off relaunch: exact.py `ec2f4437f8e1ed09`, G1 digest `2f7fdccbd388225a`.
4. shadow arm `DSV41_DRAFT_MOE_G17=fc1,permverify`: rank0.log shows 3 `drafter compact FC1 enabled` lines; after
   ≥ 600 s `touch ~/dsv41-logs/mfront-verify.trigger`; every `mfront-verify.<host>.<pid>.json` must say `OK`
   with calls > 0 (this verifier also counts target layers when `DSV41_MOE_FUSED_PERMUTE=verify` is set; it is not by
   default). exact.py and digest unchanged.
5. timing pairs C T C T with `DSV41_DRAFT_MOE_G17=1` (and, to attribute, `fc1` and `perm`); promote at ≥ +0.3% in
   both pairs with every exactness check identical. The central estimate (+0.24%) is below the +0.3% bar, so expect
   this lever to need stacking with others or a longer measurement.

## FC2 K 384 → 288: skipped

It drops three trailing all-zero k32 MMA steps from every FC2 output element, i.e. it changes the per-element
accumulation sequence (−0 → +0 class), which the round-2 rules exclude unless proven bitwise on device; it needs an
E=128 rebuild of the FC2C or Athena native (both host-assert `num_groups == 384`); and it is worth ~0.05% G1 with FC2C
measured slower at M=1 on the target (0.956x).
