# r3 spec-sampling handoff (offline prep, 2026-10-22, no restarts or GPU jobs done)

S = /private/tmp/claude-501/-Users-jeweled/64b45536-e9c4-4d44-9db6-63d2a8374326/scratchpad, D = $S/r3/spec

## 1. What vLLM 0.32 does (code read on spark-one site-packages, all 8 ranks identical)

| question | answer | where |
|---|---|---|
| `draft_sample_method="probabilistic"` with method `dspark` | Supported. `DSparkSpeculator._sample_logits` calls `gumbel_sample(..., is_drafting=True, logits_cache=self.draft_logits, logits_cache_col=step)` inside the sequential Markov loop. DSv4.1 drafter is full-vocab (`draft_id_to_target_id is None`), so no d2t scatter buffer. | spec_decode/dspark/speculator.py:116-154, models/deepseek_v41/nvidia/dspark.py:347 |
| `rejection_sample_method="block"` | Supported for any method. It is a constexpr branch in `_rejection_kernel` plus 2 pre-kernels (`_compute_cumulative_log_p_kernel`, `_compute_local_residual_mass_kernel`, the second only with draft logits). | spec_decode/rejection_sampler_utils.py:598-630, 1143-1216 |
| adaptive verification (AV) + our calibrated adaptive_verification.py | Compatible. The block kernels handle truncated per-request draft counts ("Adaptive verification can end before num_speculative_steps"). `draft_logits` is indexed by persistent req-state slot, so AV compaction does not move it. The checkpoint has `mtp.2.confidence_head`, so AV uses the trained confidence head rather than the online estimator. The head was presumably trained on greedy acceptance, so its budget may be miscalibrated at temp>0 (risk R4). The DSV41_AV_* calibration is untouched. | spec_decode/dspark/speculator.py:106-113, rejection_sampler_utils.py:428 |
| CUDA graphs | Drafting runs under the FULL graph, including the Markov loop and gumbel sampling. temperature, seeds, draft_logits and sample_pos are persistent buffers, and the temp==0 checks happen inside the kernels, so one graph serves greedy and sampled requests. Verification and rejection run eagerly (Triton). | dspark/speculator.py docstring |
| TP8 | Every rank drafts and verifies the same requests with the same per-request seed (`np.random.randint` in the sampler states, same order on each rank). Sampled target decode already relies on this today. With probabilistic drafting the draft tokens also depend on it (risk R2). `use_local_argmax_reduction` is incompatible with probabilistic drafting but defaults to False and our JSON does not set it. | sample/states.py:54-58, speculator.py:410-427 |
| extra memory, probabilistic | `draft_logits` = max_num_seqs x n x vocab x head_dtype(bf16) = 16 x 5 x 129280 x 2 B = **20.7 MB** per rank (n=7: **29.0 MB**). Block adds `local_residual_mass` (rows x 16 fp32) and `cumulative_log_p` (rows fp32), both in KB. Negligible against the ~30 GiB MemAvailable floor. | speculator.py:179-192 |
| n=7 | Config accepts it, because the DSpark width is not capped by `dspark_block_size` (=5 in the checkpoint) when `num_speculative_tokens` is given. Positions 6-7 are beyond the trained block (extrapolation). | config/speculative.py:1481-1488 |

### Two RNG bugs that make stock block verification LOSSY (found while checking)

1. **Uniform reuse in block verification.** `_rejection_kernel` draws `u = tl_rand32(seed, pos)` for *every* draft row, and block verification evaluates all rows. Rows after the accepted prefix are re-verified in the next step at the same positions with the same u. That u is conditioned on having failed `u > h`. Token verification is not affected, because it stops at the first rejection.
2. **Drafter noise reuse.** DSpark keys draft Gumbel noise by `(seed, predicted position)`. Block's `h_i` depends on drafts past the accepted prefix, and those positions are re-drafted next step with the same noise.

Numpy model of the exact kernel semantics: `$D/sim_block_verify.py`, outputs `$D/sim_out_n5.txt` and `$D/sim_out_dnoise.txt`. Context-independent p_k/q_k, 100k streams, z>5 means the marginals differ from the target.

| config | max_z | max_TV | accepted drafts/step |
|---|---|---|---|
| standard, greedy or prob draft, stock keys | 0.6-2.2 | ~0.004 | 0.99 / 2.83 (ok, lossless) |
| block, greedy draft, stock u | **722** | **0.085** | 0.79 (also *lower* than standard 0.99) |
| block, prob draft, stock u + stock draft noise | **140** | 0.03 | (lossy) |
| block, prob draft, fresh u, stock draft noise | **152** | 0.033 | (still lossy) |
| block, greedy draft, fresh u | 1.7 | 0.006 | 0.99 (= standard; no gain) |
| block, prob draft, fresh u + fresh noise | 0.5 | 0.0045 | **3.19 vs 2.87 standard (+11%)**; with variable (AV-like) draft counts 1.73 vs 1.63 (+6%) |

**Fix** (two flag-gated patches; both default off, and with the flag off they are inert):
- `$D/patch/rejection_sampler_utils.py` (diff `$D/patch/rejection_sampler_utils.diff`). With `DSV41_BLOCK_VERIFY_FRESH_U=1`, block-verification uniforms for non-greedy rows come from murmur3(seed, pos, step start position, own domain), so each draw is fresh per (step, row). It adds a constexpr `BLOCK_FRESH_U` (default False). Greedy rows, standard and synthetic modes are untouched. **The flag-off PTX is byte-identical to stock for all 4 (draft_logits x block) variants** (debug sections stripped; `$D/ptx_cmp.py`, output `$D/ref/ptxcmp.txt`). All 6 variants compile for sm_121 CPU-only (`$D/aot_compile_check.py`, `$D/ref/aot.txt`).
- `$D/patch/dspark_speculator.py` (diff `$D/patch/dspark_speculator.diff`). With `DSV41_DRAFT_FRESH_NOISE=1` and probabilistic drafting, the draft noise key becomes `2**29 + 8*anchor_pos + step`, which is unique per draft step. After the in-kernel 2**30 salt it is disjoint from target keys (<2**30) and from stock draft keys (2**30+pos<2**31). It only touches the dense Markov path; DSv4.1 has no `dspark_draft_topk`. Greedy is unaffected because temp 0 means a pure argmax with no noise.
- Stock sha: rejection_sampler_utils.py 99556e84..., dspark/speculator.py 97bcfaa6... (identical on all 8; dspark/speculator.py also equals its `.bak-dsmarkov`). Patched: 09522889..., c9e0a2a6.... **Conflict:** if the r2 drafter-markov patch gets installed into dspark/speculator.py, rebase this 10-line change onto it. install.sh refuses to run on non-stock files.

### Greedy (temp 0) behaviour is unchanged (code read, plus PTX proof for the flag-off kernel)
- `_rejection_kernel`: the `if is_greedy:` branch comes first and is identical for every config. Under the patch, u is replaced only via `tl.where(is_greedy, u, fresh)`, and u is unused for greedy except in synthetic mode.
- The stats kernel's temp==0 branch computes only the target argmax. `_compute_cumulative_log_p_kernel` and `_compute_local_residual_mass_kernel` return immediately at temp 0. The resample kernel returns for greedy non-bonus rows, and for the bonus it takes the argmax of the raw target logits (no noise at temp 0).
- Probabilistic drafter at temp 0: `gumbel_noised_argmax` skips noise when temp==0, which gives an argmax with the same first-index tie-break as `logits.argmax`. It additionally writes the (unused) logits cache.
- The GPU test checks all of this on device: `test_spec_sampling.py greedy`.
- Caveat: greedy output is still not bit-identical across restarts (known AV/RDMA issue), so check greedy within one instance (exact.py) and with GSM8K-50.

## 2. Arms (`$D/arms.tsv`; launch with `bash $D/launch-spec-arm.sh <arm>`)

| arm | DSV41_SPEC_JSON (exact) | extra env |
|---|---|---|
| ctl5 | none (live launcher default, equivalent to `{"method":"dspark","num_speculative_tokens":5,"enable_adaptive_verification":true}`) | |
| ps5 | `{"method":"dspark","num_speculative_tokens":5,"enable_adaptive_verification":true,"draft_sample_method":"probabilistic","rejection_sample_method":"standard"}` | |
| pb5 | `{"method":"dspark","num_speculative_tokens":5,"enable_adaptive_verification":true,"draft_sample_method":"probabilistic","rejection_sample_method":"block"}` | DSV41_BLOCK_VERIFY_FRESH_U=1 DSV41_DRAFT_FRESH_NOISE=1 |
| gb5 | `{"method":"dspark","num_speculative_tokens":5,"enable_adaptive_verification":true,"draft_sample_method":"greedy","rejection_sample_method":"block"}` | DSV41_BLOCK_VERIFY_FRESH_U=1 |
| ctl7 | `{"method":"dspark","num_speculative_tokens":7,"enable_adaptive_verification":true}` | |
| ps7 | `{"method":"dspark","num_speculative_tokens":7,"enable_adaptive_verification":true,"draft_sample_method":"probabilistic","rejection_sample_method":"standard"}` | |
| pb7 | `{"method":"dspark","num_speculative_tokens":7,"enable_adaptive_verification":true,"draft_sample_method":"probabilistic","rejection_sample_method":"block"}` | DSV41_BLOCK_VERIFY_FRESH_U=1 DSV41_DRAFT_FRESH_NOISE=1 |
| gb7 | `{"method":"dspark","num_speculative_tokens":7,"enable_adaptive_verification":true,"draft_sample_method":"greedy","rejection_sample_method":"block"}` | DSV41_BLOCK_VERIFY_FRESH_U=1 |
| pb5stock / gb5stock / pb7stock / gb7stock | same JSON as pb/gb | none. Stock RNG (lossy, see the table above); for the record only, never promote. |

## 3. Bench and gate

- `$D/bench/bench.py` is a drop-in replacement for spark-one `~/dsv41-bench/bench.py`. The old CLI is unchanged: with `--top-p` unset, the old behaviour stays (top_p=1.0 is sent only at temp 0). New options are `--top-p P`, `--seed S` and `--preset t1` (temp 1.0, top_p 0.95) or `--preset t06` (temp 0.6, top_p 1.0; there is no generation_config.json, so the server default top_p is 1.0). Rows also record top_p. Diff: `$D/bench/bench.diff`.
- `$D/bench/runarm_sampled.sh <label> [quick]` runs: a warm-up request, greedy single x2, t1 single x2, t06 single x2, t1 conc4/8/16, greedy conc8, the sampled GSM8K gate, and a summary (`summ_sampled.py`). Full run about 25 min; quick (conc8 only) about 12 min.
- `$D/bench/gsm8k_sampled_gate.py run <prefix> --n 50 --conc 8 --temp 0.6 --top-p 1.0 --runs 2 --seed 1000`, then `compare <ctl_prefix> <arm_prefix>`. Seeds are `1000 + 100000*run + idx`, so they are the same across arms.
  **Sampled-quality gate:** PASS iff arm mean correct >= control mean - 2.5 (of 50, i.e. 5 items over the 2 runs), arm mean truncations <= control + 1, and missing-"Answer:" lines <= control + 1. One 50-item run at about 96% has SD 1.39 items, and the difference of two 2-run means has SD 1.39, so the threshold sits at 1.8 sigma (about 4% false fail). Promotion grade: `--n 200 --conc 16 --runs 1`, `compare --tol-mean 6` (SD 3.9). Greedy must still pass the existing gates: GSM8K-50 49-50/50 (`python3 ~/gsm8k_gate.py <out> 50 8`), the GSM8K-200 reference misses 93/100/119/184, and exact.py self-consistent.
  This gate catches garbage and gross regressions only. Small distribution bias (like the stock block bug) is invisible to GSM8K; `test_spec_sampling.py stream*` is the losslessness gate.

## 4. Install / test / run (when restarts + GPU are allowed)

```
# 0. stage bench tools on spark-one (no restart needed)
ssh spark-one 'cp ~/dsv41-bench/bench.py ~/dsv41-bench/bench.py.bak-r3spec'
scp $D/bench/bench.py $D/bench/runarm_sampled.sh $D/bench/summ_sampled.py $D/bench/gsm8k_sampled_gate.py spark-one:dsv41-bench/
scp $D/test_spec_sampling.py spark-one:dsv41-bench/
# 1. install the patches on all 8 ranks (inert until flags are set; backs up .bak-r3spec, verifies sha on all 8)
bash $D/install.sh
# 2. GPU unit test on ONE rank, ~1-2 GB, about 1 min. Only when GPU jobs are allowed (ideally while the service is down):
ssh spark-one 'source ~/venvs/vllm032/bin/activate; cd ~/dsv41-bench;
  python3 test_spec_sampling.py greedy;                                                    # expect PASS
  DSV41_BLOCK_VERIFY_FRESH_U=0 python3 test_spec_sampling.py stream;                       # expect block rows FAIL (stock)
  DSV41_BLOCK_VERIFY_FRESH_U=1 python3 test_spec_sampling.py stream;                       # expect all ok
  DSV41_BLOCK_VERIFY_FRESH_U=1 python3 test_spec_sampling.py stream-vllm-draft;            # expect block+prob FAIL
  DSV41_BLOCK_VERIFY_FRESH_U=1 DSV41_DRAFT_FRESH_NOISE=1 python3 test_spec_sampling.py stream-vllm-draft'  # all ok
# 3. arms (each restart is about 6.5 min; ready when rank0.log shows "Application startup complete")
for a in ctl5 ps5 pb5 ctl5 gb5; do bash $D/launch-spec-arm.sh $a; <wait ready>; ssh spark-one "cd ~/dsv41-bench && bash runarm_sampled.sh $a-<k>"; done
#    then pb7/ps7/ctl7 only if pb5 or ps5 wins. Gate: python3 gsm8k_sampled_gate.py compare r3spec/ctl5-1.g06 r3spec/pb5-1.g06
# rollback: bash $S/conc/launch-conc.sh  (no JSON, no flags = live config) ; bash $D/uninstall.sh ; bench.py.bak-r3spec
```
The 1st ordered pass (ctl5, ps5, pb5, ctl5, gb5) takes about 5 x (6.5 + 25) min = about 2.6 h. A promotion pass with C/A/C/A at GSM8K-200 adds about 2 h.

## 5. Expected gain (estimates; nothing has been measured yet)

Measured anchor (14:08 baseline, same prompts, AV on): greedy accept len code/prose/math 2.64/1.77/3.63 (mean 2.68), 57.1 tok/s. At temp 0.7 with greedy draft + standard: 2.25/1.74/3.36 (mean 2.45, **-8.6%**), 53.7 tok/s (**-6%**). Sampled requests lose acceptance because a one-hot draft is accepted with probability p(x_draft) instead of 1[x = argmax p].

- **Overhead per step (1 user):** draft-logits cache write 5x129280x2 B = 1.3 MB, plus its re-read in verification, about 0.01 ms at 238 GB/s. Five gumbel passes over 129280 logits (murmur3 + log) are about 0.05 ms each, 0.25 ms total. Block pre-kernels read about 6 rows x 129280 x 4 B = 3 MB, about 0.02 ms, plus launch overhead (about 0.1 ms total). That is about 0.3-0.5 ms of a 33.5 ms step, **about 1-1.5%**.
- **ctl at t1 (temp 1.0, top_p 0.95):** the -8.6% at 0.7 extrapolates to about -12..-15%, so accept len about 2.48 x 0.86 = 2.13, about 70 tok/s single (vs 80.8 greedy).
- **ps5 (probabilistic + standard):** per-position acceptance rises from E[p(argmax q)] to 1 - TV(p,q). The draft is not top-p truncated, so about 5% of its mass is proposed outside the nucleus and always rejected. Estimate: recovers half to two thirds of the loss, so accept len about 2.30-2.38 (**+8..+12% vs ctl at t1**). Net tok/s about +7..+10% (about 75-77 tok/s single). At t06 the gain is smaller (the loss is about 7%): **+3..+6%**.
- **pb5 (probabilistic + block, patched):** the simulation gives +11% accepted drafts at fixed n and +6% with AV-like variable counts. With about 1.3 accepted drafts/step at t1, that is +0.08..+0.14 tokens/step, so accept len about 2.40-2.50: **+13..+17% vs ctl at t1, about +11..+16% tok/s (about 78-81 tok/s single)**, and about +5..+9% at t06. conc16 is verify-bound with AV trimming; assume about 60-70% of the single-stream gain, i.e. **+7..+11%** (313 to about 335-345 at t1-equivalent load).
- **gb5 (greedy draft + block, patched):** about 0 (simulation 0.987 vs 0.986). Block gains need a non-degenerate q. It is kept only as a control for the block-kernel overhead.
- **n=7:** from the greedy per-position data, unconditional acceptance at positions 4-5 is already 0.002-0.3, and positions 6-7 are untrained. The drafter adds 2 queries and 2 Markov passes (about +0.6 ms), and AV trims verification. Greedy and ctl7: about 0..-3% (math possibly +2%). Block's advantage grows with draft length, so pb7 is the only n=7 arm with plausible upside, about +0..+3% over pb5. Exploratory.
- **Greedy:** 0 change by construction. Only probabilistic arms pay the about 1% gumbel/cache overhead on greedy requests too (the logits-cache write happens at temp 0). Expect greedy single within noise (about -1%). Check it in every arm (runarm_sampled.sh does).

## 6. Risks
- R1 (high, fixed behind flags): stock block verification is lossy and lowers acceptance. Never run block without `DSV41_BLOCK_VERIFY_FRESH_U=1`, and with probabilistic drafts also `DSV41_DRAFT_FRESH_NOISE=1`. The patches are new code and have only been verified by simulation and an sm_121 compile; the GPU test must pass before any serving run.
- R2 (medium): probabilistic drafts depend on identical per-request seeds across the 8 TP ranks. If a rank diverged, the draft tokens fed to the target would differ per rank and produce garbage or very low acceptance. Detect it with acceptance length and the GSM8K gate in the first ps5 run. Sampled target decode already relies on the same property, and run_b at temp 0.7 was fine.
- R3: top_p (t1) is applied to the target only. Draft tail proposals are rejected, which is lossless but costs acceptance. Nothing in vLLM 0.32 applies top_p to drafts.
- R4: AV uses the trained confidence head, probably calibrated for greedy, so it may over- or under-budget at temp>0. If pb5 underwhelms, try `DSV41_AV_VERIFY_SCALE` or disabling AV for sampled arms. That is a separate experiment and changes greedy too.
- R5: the first top_p request JIT-compiles kernels that warmup does not cover (warmup.py covers greedy and temp 0.7 only). runarm_sampled.sh sends a warm-up request first.
- R6: the patch to dspark/speculator.py conflicts with the staged r2 drafter-markov patch on the same file; rebase if that lands first.
- R7: per-request `seed` in the gate makes `seeds_set` true. Any vLLM path that falls back for explicit seeds would apply equally to control and arm, but do not compare gate tok/s with unseeded bench tok/s.

## 7. Readiness
- Config-only arms ctl/ps (n=5/7): **ready** (no code change; JSON plus restart).
- pb/gb (block): **needs-more-work, gated only on the GPU unit test** (step 2, about 1 min). The code is written, compiles for sm_121, and flag-off is PTX-identical to stock. Do not run stock block arms except as a documented negative control.
- Bench and gate tooling: ready (py_compile clean; backward-compatible CLI).
