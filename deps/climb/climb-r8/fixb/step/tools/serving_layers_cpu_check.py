"""CPU-only check of device_test.serving_linear (the r6-review blocker-1/2 fix) against the LIVE vllm package and the
real checkpoint, without a GPU: builds mtp.0.main_proj and layers.{1,14}.engram.wkv (rank slices 0 and 7) through
the live quant-method lifecycle on CPU tensors, then checks
  - the method is ModelOptLinearMethod(kMxfp8Static, scale block 32x32), FormatScheme default;
  - the loaded scale is the live loader's repeat_interleave(32, 0) of the checkpoint block scale;
  - the processed scale is the live swizzled 1-D layout, and r4.rowmajor_scale() recovers the loaded [N, K/32];
  - the live r4.ColSplit accepts the serving layer for every rank (the crash the review reproduced with the raw
    checkpoint scale is gone) and its per-rank swizzled scale == swizzle(rows of the loaded scale);
  - the raw checkpoint scale still makes ColSplit raise (the old test's input; proves the check discriminates).
Kernel selection needs the GPU platform; with CUDA hidden it may fail, in which case the check reports that and uses
FlashInferCutlassMxfp8LinearKernel.process_weights_after_loading (the serving kernel on sm_12x) directly.

Run on spark-eight in ~/climb-scratch/r6-step, lock absent, nice 19, no GPU:
  source ~/venvs/vllm032/bin/activate; export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
  CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 nice -n 19 python3 -B tools/serving_layers_cpu_check.py
"""
import importlib.util
import os
import sys

import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("dt", os.path.join(HERE, "device_test.py"))
dt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dt)
MODEL = os.path.expanduser(os.environ.get("STEP_MODEL", "~/models/dsv41/official-fp8"))

from vllm.model_executor.kernels.linear.mxfp8.flashinfer import FlashInferCutlassMxfp8LinearKernel  # noqa: E402
from vllm.model_executor.layers.quantization import modelopt as mo  # noqa: E402
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import swizzle_mxfp8_scale  # noqa: E402
from vllm.models.deepseek_v41.nvidia import dsv41_r4_small as r4  # noqa: E402

orig_select = mo.select_linear_kernel
note = []


def select(spec_, layer, rt, weight_shape=None):
    try:
        k = orig_select(spec_, layer, rt, weight_shape=weight_shape)
        note.append(f"live select_linear_kernel -> {type(k).__name__}")
        return k
    except Exception as e:  # noqa: BLE001 - no GPU platform here
        note.append(f"live select_linear_kernel unavailable without a GPU ({type(e).__name__}); "
                    f"using {FlashInferCutlassMxfp8LinearKernel.__name__}.process_weights_after_loading")
        return FlashInferCutlassMxfp8LinearKernel.__new__(FlashInferCutlassMxfp8LinearKernel)


mo.select_linear_kernel = select

def run():
    for prefix, rows_list in (("mtp.0.main_proj", [None]),
                              ("layers.1.engram.wkv", [slice(0, 3200), slice(7 * 3200, 8 * 3200)]),
                              ("layers.14.engram.wkv", [slice(0, 3200), slice(7 * 3200, 8 * 3200)])):
        for rows in rows_list:
            lyr = dt.serving_linear(MODEL, prefix, rows, "cpu")
            qm = lyr.quant_method
            W, S = dt._CK[prefix]
            N, K = W.shape
            loaded = S.view(torch.uint8).repeat_interleave(32, 0)
            if rows is not None:
                loaded = loaded[rows]
            n = loaded.shape[0]
            dt.gate(f"{prefix} {rows}: ModelOptLinearMethod(kMxfp8Static, 32x32), FormatScheme",
                    qm.spec.weight == mo.kMxfp8Static and qm.ctx.scale_block_size == (32, 32)
                    and type(qm.fmt) is mo.FormatScheme, f"raw scale {lyr._raw}")
            dt.gate(f"{prefix} {rows}: processed scale == swizzle(repeat_interleave(raw, 32)) (1-D serving layout)",
                    lyr.weight_scale.dim() == 1
                    and torch.equal(lyr.weight_scale.data, swizzle_mxfp8_scale(loaded, M=n, K=K)),
                    f"{tuple(lyr.weight_scale.shape)} {lyr.weight_scale.dtype}")
            dt.gate(f"{prefix} {rows}: r4.rowmajor_scale(serving layer) == loaded [N, K/32]",
                    torch.equal(r4.rowmajor_scale(lyr), loaded))
            if prefix == "mtp.0.main_proj":
                for rank in range(8):
                    try:
                        st = r4.ColSplit(lyr, rank, 8)
                        rs = slice(rank * 640, (rank + 1) * 640)
                        ok = (torch.equal(st.w.view(torch.uint8), lyr.weight.data[rs].view(torch.uint8))
                              and torch.equal(st.s_swz, r4.swizzle_scale(loaded[rs].contiguous(), 640, K)))
                        dt.gate(f"main_proj: live ColSplit(serving layer, rank {rank}) builds; slice + scale exact", ok)
                    except Exception as e:  # noqa: BLE001
                        dt.gate(f"main_proj: live ColSplit(serving layer, rank {rank}) builds", False, repr(e))

                class Raw:  # the old device test's input: the raw checkpoint scale
                    weight = W
                    weight_scale = S.view(torch.uint8)

                try:
                    r4.ColSplit(Raw(), 0, 8)
                    dt.mutant("raw checkpoint scale rejected by ColSplit (old device-test input)", False)
                except RuntimeError as e:
                    dt.mutant("raw checkpoint scale rejected by ColSplit (old device-test input)", True, repr(e))
    for n_ in sorted(set(note)):
        print("      note:", n_)
    return dt.summary()


if __name__ == "__main__":
    sys.exit(run())
