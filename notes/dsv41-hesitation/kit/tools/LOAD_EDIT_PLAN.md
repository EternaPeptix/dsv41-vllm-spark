# load_edit: flag-gated rank-1 weight edit at load time (I7). Patch plan only

Status: **design, not installed, not run on the cluster.** The maths and the edit-file format are in `load_edit_ref.py` and are CPU-tested in `test_capture_eval.py::load_edit_reference_math`. Everything that touches vLLM below is a plan.

Tags: [V] I read it in the installed code on spark-one (read-only, 2026-11-03). [I] my inference. [?] unknown; the first edited launch must check it.

## 1. When this runs at all

- Only if an **ungated runtime ablation arm (I5, `r ← r − a (r·d) d` at L14 on every row)** passes the SPEC 5.3 gates on E (P6). Only that arm predicts what a global weight edit does. Additive steers (I3) and gated arms (I4, I5g) have no weight twin: the model has no bias tensors [V, stack.md §5], and a weight edit acts on prompt KV, answers and tool calls too.
- The direction `d` comes from `capture_spans.py direction` (`<out>/hes.npy` plus its JSON report). The report's `checks.pass` must be true and its SHA-256 recorded in `FROZEN.json`.

## 2. What is edited

Output-side ablation of residual writers, `W' = (I − a d dᵀ) W = W − a d (dᵀW)`:

| Target (module prefix) | TP layout per rank | Edit | Source |
|---|---|---|---|
| `model.layers.N.attn.wo_b` | RowParallel, `[5120, 1024]` (all 5120 output rows) | rank-local, exact under the all-reduce / reduce-scatter | prefix `f"{prefix}.wo_b"` [V attention.py:400-406] |
| `model.layers.N.ffn.shared_experts.down_proj` | RowParallel, `[5120, 288]` | rank-local, exact | `DeepseekV4MLP(prefix=f"{prefix}.shared_experts")`, `down_proj` prefix `f"{prefix}.down_proj"` [V deepseek_v4/nvidia/model.py:168-175, 784-793]; the full `model.layers.N.ffn` part is [?] until the dry run logs it |
| `model.layers.{1,14}.engram.wkv` value rows 20480-25599 | ColumnParallel, 3200 rows per rank (value rows on ranks 6 and 7) | `V_r − a d[R_r] uᵀ`, `u = Vᵀd` precomputed offline | phase 2 only: its input arrives as MXFP8 from `EngramBatch` [V stack.md §3], which the BF16 kernel below may not accept [?] |

The band is the layers at or below 14 with the largest writer score `‖dᵀW_l‖ / ‖W_l‖_F` (`load_edit_ref.writer_score`, computed offline on spark-one CPU from the checkpoint bytes with `load_edit_ref.dequant_mxfp8`). Start with 6-8 matrices.

Known gap: the 15,360 routed experts (MXFP4) keep writing along `d`, so the bake is a **partial** ablation [I]. G8 measures how partial.

## 3. Precision: keep the edited matrices in BF16

- Rounding a weak rank-1 edit back into MXFP8 loses 40-80% of it: random `d` at a=1 keeps 0.427 of the write, and a=0.5 keeps 0.821 against a 0.50 target [V, research/requant_sim.out].
- BF16 storage keeps it: on the CPU check the kept fraction is below 0.02 at a=1 (`self_check`, `ablation_kept_bf16`) [V, mock scale].
- Route: give the edited prefixes `EmulationMxfp8LinearKernel`. With `VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD` (default on) it replaces the fp8 weight with a BF16 `Parameter` in `process_weights_after_loading` and then runs `F.linear` [V emulation.py]. It also declares `supports_pre_processed_weights = True` [V].

## 4. Patch points (three files, all no-ops when the flag is unset)

Flag: `DSV41_WEIGHT_EDIT=/abs/path/edit.json` (unset means no change at all). Optional `DSV41_WEIGHT_EDIT_DRYRUN=1` logs the matched prefixes and the chosen kernels, then edits nothing.

### 4.1 `vllm/models/deepseek_v41/quant_config.py`, `DeepseekV4FP8Config.get_quant_method` [V lines 221-243]

In the MXFP8 `LinearBase` branch, after building `ModelOptLinearMethod(...)`: if the flag is set and `prefix` is in the edit file's targets, set `method._dsv41_force_kernel = "emulation"` before returning it. Read the targets once per process and cache them. Here `prefix` is the module prefix, which is how the targets are named.

### 4.2 `vllm/model_executor/layers/quantization/modelopt.py`, `ModelOptLinearMethod.create_weights` [V lines 2598-2604]

Between `self.kernel = select_linear_kernel(...)` and `expose_input_quant_key(layer, self.kernel)`:

```python
if getattr(self, "_dsv41_force_kernel", None) == "emulation" and self.spec.weight == kMxfp8Static:
    from vllm.model_executor.kernels.linear.mxfp8.emulation import EmulationMxfp8LinearKernel
    from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import Mxfp8LinearLayerConfig
    self.kernel = EmulationMxfp8LinearKernel(Mxfp8LinearLayerConfig(weight_shape=self.fmt.kernel_weight_shape(layer)))
```

- Placing it before `expose_input_quant_key` keeps the declared input format consistent with the kernel. The upstream producer then hands `wo_b` a BF16 tensor, not a `QuantizedActivation`. Whether that holds on every SM12x path is [?]: the dry run must log the kernel per prefix, and the first forward must not raise.
- `Mxfp8LinearLayerConfig` may need `bmm_batch_size=None`; the constructor fields are [?] beyond `weight_shape`/`bmm_batch_size` [V call site in kernels/linear/__init__.py:936-944].
- Side effect: `bind_gemm_rs` checks kernel eligibility before the weights load [V attention.py:724-743]. An Emulation `wo_b` will probably fail `can_run`, so long sequence-parallel prefills of those layers take the unfused `sp_reduce_scatter(self.wo_b(z))` path [I]. Prefill cost is to be measured.

### 4.3 `vllm/models/deepseek_v41/nvidia/model.py`, `DeepseekV41LLMForCausalLM` [V lines ~2731-2756]

`process_weights_after_loading()` is called twice: once from `load_weights` (before the per-module quant finalize, when the weights are still fp8) and once from `model_loader/utils.process_weights_after_loading` after every quant method has run [V utils.py: "Model-level post-load hook, after the per-layer quant finalize"]. The edit must run in the second call, when the targets are BF16:

```python
def load_weights(self, weights):
    ...
    self._dsv41_in_load = True
    try:
        self.process_weights_after_loading()
    finally:
        self._dsv41_in_load = False
    ...

def process_weights_after_loading(self) -> None:
    self.model.finalize_mega_moe_weights()
    self.model.finalize_mhc_broadcast_weights()
    if not getattr(self, "_dsv41_in_load", False):
        _dsv41_weight_edit(self)          # no-op unless DSV41_WEIGHT_EDIT is set
```

`_dsv41_weight_edit(model)` (new, about 80 lines; would live in a small `dsv41_weight_edit.py` beside the kit). It fails closed: every check raises and aborts startup.

1. Return immediately if the flag is unset, or if `model._dsv41_edit_done` is already set (idempotent).
2. Validate the file as `load_edit_ref.validate_edit_file` does (hash, version, unit `d`, `0 < a ≤ 1`, allowed targets, layer ≤ 14).
3. **Rank agreement, fail closed.** Take the first 8 bytes of the SHA-256 as an int64 tensor. `all_reduce` MIN and MAX over the TP device group, and raise unless MIN == MAX == the local value. Do **not** reuse `_dsv41_sp_agree_min`: it returns the local value on any exception, so it fails open [V model.py:372-390].
4. For each target: `mod = model.get_submodule(prefix)`. Require `mod.weight.dtype == torch.bfloat16` and `type(mod.quant_method.kernel).__name__ == "EmulationMxfp8LinearKernel"`. For `down_proj`, also require that the owning MoE layer has `num_shared_experts == 0` after finalize, i.e. the shared expert is not natively fused into MegaMoE, whose transformed copy would not see the edit [V deepseek_v4/nvidia/model.py:396-415; at TP8 without SP fusion is expected off, stack.md §3, I].
5. Edit in fp32 on the GPU: `W = mod.weight.float(); W -= a * d[:, None] * (d @ W); mod.weight.copy_(W.to(torch.bfloat16))`. Modify in place; never replace the Parameter, so that pointers stay valid.
6. Log per tensor and per rank: prefix, shape, `‖dᵀW‖/‖W‖_F` before and after, and a CRC of the bytes before and after. On rank 0, also log one summary line: `DSV41_WEIGHT_EDIT applied: <n> tensors, alpha <a>, sha <first 16 hex>`.
7. Set `model._dsv41_edit_done = True`.

CUDA graphs are captured after loading and record pointers, and the shapes do not change. So the graphs replay the edited weights with no other change [V load order, stack.md §4; I].

### 4.4 Launcher

`~/dsv41-tp8.sh` passes `DSV41_WEIGHT_EDIT` through only when the caller sets it. The edit file is rsynced to the same absolute path on all 8 nodes before the relaunch. The default launcher hash (`416e3ace`, production since 2026-11-03 09:25 UTC with incremental tokenization on; it replaced `66eaa1d9`) stays the default. The edited launch goes through `dsv41-relaunch.sh` with the variable exported for that window only.

## 5. Checks, in order

1. **Unset flag = bit-identical.** Install the patch and relaunch without the flag. `exact.py` must match the stored hash. The patch must change nothing when the flag is unset (one relaunch, about 165 s).
2. **Dry run** with `DSV41_WEIGHT_EDIT_DRYRUN=1`: the log lists every target prefix, its kernel (`EmulationMxfp8LinearKernel`), and its BF16 dtype after the finalize. Abort if any target is missing, or if a non-target changed kernel.
3. **Edited launch with capture on** (`DSV41_STEER_CAPTURE=1`). G8 fidelity: at held-out U\* boundaries, the drop in the L14 projection on `d` must be within ±20% of what runtime I5 gives. `capture_spans.py capture` on the same spans gives both numbers. Also: decode tok/s ≥ base − 3% and the DSpark acceptance rate are reported.
4. **E confirmation** of I7 against the P6 base arms with `eval_hesitation.py` (G1-G9). All traffic on that server gets the edit during this window.
5. **Rollback:** relaunch without the flag, then rerun `exact.py`.

## 6. Costs and risks

- Memory: a BF16 `wo_b` slice is `[5120, 1024]` × 2 B = 10.5 MB per rank, against 5.2 MB of fp8 plus scales. A shared `down_proj` slice is 2.9 MB. For 8 matrices that is under 70 MB per rank [I].
- Decode: roughly 2× the read bytes on those matrices, about 0.02 ms per edited `wo_b` per step at about 270 GB/s [I, unmeasured].
- Partial ablation (routed experts not edited): the effect may be much weaker than I5. G8 detects this.
- `QuantizedActivation` reaching an Emulation `wo_b` on some path [?]. This would raise at the first forward, during warmup, before any traffic.
- A shared expert that is natively fused (MegaMoE) would silently miss the edit. Step 4.3.4 guards against this.
- Engram value rows (phase 2) need an Emulation path that accepts `EngramBatch`'s MXFP8 input [?].
- The edit is global and cannot be turned off per request. Agent checking is protected only by the G5/G6 evidence, not by any mechanism.
