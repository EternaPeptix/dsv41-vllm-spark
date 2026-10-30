# k1 exp 2-silu-liverows: live-row-only SiLU-mul-FP8-quant in the routed FP4 MoE

Design: kernel-ideas.md section 2, plus the critic note that expert_ends comes from
`_fwd_kernel_ep_scatter_2`'s atomic output. If exp 6 replaces that kernel, re-run this test.
Exactness class: exact. exact.py must stay a8c3394af9d2 with the flag on.

## What changes (flag-gated, default off)

`DeepGemmFP4Experts.apply` covers the target (E=384, top-6, compact N=576 or padded N=768)
and the drafter (E=128, top-3, N=768). It now asks
`deepgemm_moe_permute(..., return_expert_ends=True)` for `expert_start_loc` after ep_scatter.

How `expert_start_loc` becomes the expert ends:
- `_fwd_kernel_ep_scatter_1` stores each expert's aligned start.
- The `atomic_add` in `_fwd_kernel_ep_scatter_2` advances it once per routed row.
- So it ends at `start_e + count_e`, the exclusive end of that expert's live rows. This is
  DeepGEMM's psum layout: `roundup(end[e-1], align) == start[e]`.

It is passed as `expert_ends`, with `expert_alignment = align_used`, to the installed
`silu_mul_quant_fp8_packed_triton`. fp8_utils.py is unchanged; its EXPERT_ALIGNMENT mode
already exists. The kernel then runs only over live rows.

Why the output is unchanged:
- BM and packs_per_cta are still picked from M_sum, so live rows run the same instructions
  and stay bitwise identical.
- Padding rows of the FC2 input (q and UE8M0 scales) are no longer written. FC2 rows are
  independent, and ep_gather reads only live rows.
- `grouped_layout` (`torch.full(-1)`) is unchanged.

| env | default | meaning |
|---|---|---|
| `DSV41_MOE_LIVE_ACT` | unset (off) | `1` turns the live-row path on. It is read in `DeepGemmFP4Experts.__init__`. |
| `DSV41_MOE_LIVE_ACT_MIN_ROWS` | `1024` | The live path runs only when the aligned workspace M_sum >= this. M_sum is static per graph size. |
| `DSV41_MOE_LIVE_ACT_MAX_ROWS` | `0` (no cap) | Optional upper bound on M_sum, for use if the prefill timing shows a loss. |

M_sum values for the live config:
- Target, M32 tiles up to 96 rows: M_sum = 192/384/768/960/1152/9216/12480 at
  M = 1/2/4/5/6/48/96.
- Target, 128-row path (M > 96): M_sum is 49408 and up.
- Drafter (align 128): M_sum = 384/768/1152 at M = 1/2/3.
- All of these assume DeepGEMM's per-call alignment does not shrink below the requested
  tile. The test prints the real M_sum and align.

With the default gate, the path is on for target M >= 6 and drafter M >= 3. Pick the final
gate from `recommended_min_rows_by_geom` in the timing run.

## Files (site = ~/venvs/vllm032/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe)

| site file | base sha256 (live 8/8 on 2026-10-24) | payload sha256 |
|---|---|---|
| deep_gemm_utils.py | b7e8de4d04d8e65b9cb9fce893e80ec4e8f5e1ccb2244b47e7234f3f3491a2b2 | a4a0486a655d9c6953d4429ed3cf7fd760d2def2d56103438adca7bccdbcd580 |
| experts/deep_gemm_moe.py | 4541825946f01b57e1b2630268df2e6a4b6eef5a5971ded3d59ff633a72d3c50 | 5162dc0fdb59ea83d413411e861623d0c3a68f3fe2e87c2b20bc1ad0d937b93f |

- install.sh makes a backup of each file as `<file>.bak-k1-2-silu-liverows` on each rank.
- fp8_utils.py (8dc7dab8) is NOT touched.
- `base/` holds verbatim copies of the live files. The test loads them as its reference.

## Commands

All scripts refuse to run while dsv41-rank or `vllm serve` is running, unless FORCE=1.

    # Mac (stage dir copy or scratchpad copy of all.sh):
    bash all.sh stage | status | install | uninstall      # install/uninstall end with 8/8 status
    # GPU unit test on ONE stopped Spark (works before or after install; ~10-25 min):
    ssh spark-one 'bash ~/dsv41-stage/k1/2-silu-liverows/run_test.sh 0'
    ssh spark-eight 'bash ~/dsv41-stage/k1/2-silu-liverows/run_test.sh 7'   # optional, last TP slice
    #   QUICK=1 skips the checkpoint and m64 runs. Results go to ~/dsv41-stage/k1/2-silu-liverows/results-<UTC>/
    # Cluster arm (after install 8/8):
    bash ~/dsv41-opt-20261021/ops/conc/launch-conc.sh DSV41_MOE_LIVE_ACT=1 [DSV41_MOE_LIVE_ACT_MIN_ROWS=<n>]

CPU-only pre-check (Triton interpreter, no GPU; already PASSED on spark-eight on 2026-10-24):
`CUDA_VISIBLE_DEVICES= nice -n 19 python3 cpu_interp_test.py`.
