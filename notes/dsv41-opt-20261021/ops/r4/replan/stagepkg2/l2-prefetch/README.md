# r4 l2-prefetch: prefetch the next weights into L2 while the RDMA all-reduce spins

Status: staged, not installed. Nothing was restarted and no GPU jobs were run.
- The CPU checks pass: `test_plan_cpu.py` on spark-one, and the native lib compiled with nvcc 13.0 on spark-one's CPU (SASS contains `UBLKPF.L2`).
- The GPU test is still to be run by the experiment agent.

Flag: `DSV41_L2_PREFETCH=1`. It is off by default; when off, `rdma_all_reduce.py` never imports the new module.

## Port of knapcio `adapter/l2_prefetch.py`
- knapcio measured −1 ms/step at c1 on GB10 with 138 windows (about 7 µs per window). They also measured +3% on varied prompts and flat at c16.
- Their WOA sub-gate (prefetch wo_a during the attention core) was worth −0.4 ms. It is **not** ported:
  - Our wo_a is a bf16 bmm, which r3 sonar-port is changing (o_proj bf16 bmm).
  - It needs a linear-boundary hook. This can be added once r3 settles.

## What it does in our stack
- GB10 has 24 MiB of L2 (nsys `TARGET_INFO_GPU.l2CacheSize = 25165824`). DRAM spec is 546 GB/s; about 238 GB/s is achieved.
- Each target graph runs 85 RDMA `ar_kernel`s, 93 including the drafter's.
  - Each one is an 8–16 CTA flag poller: 47 µs mean at c1 code (about 30 µs intrinsic plus about 12 µs peer wait); 91 µs at c16.
  - During that time the SMs and DRAM are idle.
- The kernel right after each target AR (from the r2 trace, `r4/tools/ed.r0.1.txt`) is `mhc_fused_tilelang` (77 per graph at c1) or `mhc_post`.
  - After that come the weight-bound dense GEMMs: router gate bf16 3.93 MB, shared w13 3.0 MB, fused_wqa_wkv 9.46 MB (replicated, 55 µs, 72% of roofline).
- Right before each RDMA all-reduce or all-gather, `RdmaAllReduce` calls `dsv41_l2_prefetch.before_collective()`. That function:
  1. Forks a side stream (`side.wait_stream(main)`).
  2. Launches `dsv41_l2pf`: 8 CTAs × 64 threads issuing `cp.async.bulk.prefetch.L2`, 16 KB per instruction, over the first `DSV41_L2_PREFETCH_MB` (6) of the weights read after this collective.
  3. Joins the side stream at the next collective, or at the end of the forward.
- Inside a FULL CUDA-graph capture this becomes a graph branch.
- A prefetch is only a cache hint, so outputs are bit-identical by construction.

## How "the next weights" are found
Plans are learned, with no layer map. This is robust to other patches.
- **What is bracketed.** `DeepseekV4Model.forward` (the target model; the DSpark drafter class is not bracketed), for forwards of ≤128 tokens outside PIECEWISE.
- **Learning.** Each eager forward learns the sequence of events:
  - RDMA collectives, numbered by the hook.
  - Weight reads, via forward pre-hooks on every weight-holding submodule. Routed experts, embeddings and lm-head are excluded by size and name.
  - mHC `fn` tensors (1.97 MB fp32, read by `mhc_fused` right after the AR), via wrappers on `mhc_shifted_post_pre` and `mhc_pre_delayed_tilelang` in the model module.
- **Windows.** Window k is the first 6 MB of weights read after collective k.
- **Capture.** vLLM's capture runs an eager warm-up forward of the same size first (`cudagraph_utils.py` warmup then capture), so the capture replays the learned sequence.
  - The capture logs `captured tokens=N forks=.. collectives=.. (learned ..)`.
  - A mismatch is logged and the extra windows are skipped. Weights are static, so a stale plan only wastes bandwidth.
- **What a typical plan looks like.** The log prints the first windows (`plan tokens=6 ... first windows: [ar1 ...]`). Expected:
  - Post-attention windows: `ffn` hc fn 1.97 MB + router gate 3.93 MB (+0.1 MB of w13).
  - Post-MoE windows: next layer `attn` hc fn 1.97 MB + about 4 MB of fused_wqa_wkv.
  - **Check this line on the first boot.** If the gate is invoked without its module (no pre-hook fires), the window simply continues into shared w13, which is still useful.
- **Rank-local.** No communication is added. A rank whose `.so` fails to load disables itself without desynchronizing anything.

## Expected gain, grounded in the r2 step breakdown
- **c1 code (33.5 ms step):**
  - Dense GEMM family: 11.7 ms for 1.55 GB, about 132 GB/s effective.
  - 85 windows × at most 6 MB = at most 510 MB/step that could come from L2.
  - Bandwidth upper bound: about 2.5 ms. knapcio's measured efficiency on the same chip was about 7 µs per window.
  - **Estimate −0.4 to −0.9 ms/step**, i.e. +1.2 to +2.7% single-stream (code 72.5 → about 73.5–74.5 tok/s).
- **c1 prose:** the AR window is shorter (19 µs intrinsic), so expect about 60% of the code gain.
- **c16 (111 ms):** the AR windows are longer (77–90 µs), but the step is MoE-bound. Expect −0.3 to −0.8 ms (≤0.7%), roughly flat, as knapcio saw.
- **Risk:** the prefetch branch can overlap with the start of the next kernels and compete for DRAM. knapcio's v4 MOE sub-gate lost 13 µs per layer when its prefetch overlapped the MoE stream. Here the window is only the AR, so that risk is small.

## Files touched (conflict detection)
- `site-packages/vllm/distributed/device_communicators/rdma_all_reduce.py`:
  - live sha 3a128082 → patched 99102ff1. Diff: `rdma_all_reduce.diff`, +18 lines.
  - Changes: a conditional import, plus one `before_collective()` call in `all_reduce` and one in `all_gather_stacked`.
  - Any other patch to this file (rdma tuning) must be rebased.
- `site-packages/vllm/distributed/device_communicators/dsv41_l2_prefetch.py`: new file.
- Runtime monkeypatches, only when the flag is on, all in `vllm.models.deepseek_v41.nvidia.model`:
  - `DeepseekV4Model.forward` is wrapped.
  - The module-level names `mhc_shifted_post_pre` and `mhc_pre_delayed_tilelang` are wrapped.
  - Forward pre-hooks are added on target submodules.
- Native: `~/dsv41-stage/r4/l2-prefetch/libdsv41_l2pf.so` (sha a09f85e8; source `l2pf.cu`).
  - Override the path with `DSV41_L2_PREFETCH_LIB`.
  - It also contains `dsv41_spin`, which only the test uses.

## Install / test / A-B (experiment agent; restarts and GPU required)
```
S=/private/tmp/claude-501/-Users-jeweled/64b45536-e9c4-4d44-9db6-63d2a8374326/scratchpad
bash $S/r4/l2-prefetch/install.sh     # preflight sha on 8 ranks, .bak-r4-l2-prefetch, verify 8/8
# GPU unit test + microbench, ONE rank, ~1.5 GB, service may stay up if memory allows (prefer it stopped):
ssh spark-one 'source ~/venvs/vllm032/bin/activate; cd ~/dsv41-stage/r4/l2-prefetch;
  CUDA_VISIBLE_DEVICES=0 DSV41_L2_PREFETCH=1 python3 test_l2pf_gpu.py'
#   expect: "exact=True" for M=1,6,24,96, PASS; reports graph time off/on and us saved per window
#   (toy: 40 layers x [25 us spin + mhc fn + gate + w13 + w2 / spin + qkv_a + q_b + o_b]).
# serving A/B (same instance class, alternate C/A/C/A as in exp1):
bash $S/conc/launch-conc.sh DSV41_L2_PREFETCH=1
#   check rank logs: "DSV41_L2_PREFETCH armed", "plan tokens=6 ... windows=85", "captured tokens=.. forks=.."
#   with no MISMATCH; exact.py within instance; GSM8K-50; bench single + conc4/8/16.
# knobs to try if the first arm is positive: DSV41_L2_PREFETCH_MB=4 / 8, DSV41_L2_PREFETCH_SKIP_N=512
bash $S/r4/l2-prefetch/uninstall.sh   # restore; or just drop the env flag
```
