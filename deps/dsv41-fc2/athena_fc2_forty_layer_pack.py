"""Default-off one-layer load-time pack probe; never dispatches the result.

This helper is intended only for a separately generated vLLM source overlay.
The native weight is retained and continues to serve all FC2 operations.
Source, native ABI and output behavior are unverified. No output checking or
hashing is performed. The runtime must reserve memory for its own future work.
"""
from __future__ import annotations
import importlib.util
import os
from pathlib import Path
import time

_CLAIMED = set()
_ATHENA = (Path.home() / "dsv41-fc2").resolve()
_INCREMENTAL_CAP = 1280 * 1024**2
_FLOOR = 6 * 1024**3


def _host_available() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable unavailable")


def prepare(expert, layer, prefix: str):
    global _CLAIMED
    if os.environ.get("ATHENA_FC2_PACK_ONE_LAYER") != "1":
        raise RuntimeError("one-layer packing explicitly disabled")
    prefixes = tuple(x for x in os.environ.get("ATHENA_FC2_PACK_PREFIXES", "").split(",") if x)
    if len(prefixes) != 40 or len(set(prefixes)) != 40 or prefix not in prefixes:
        raise RuntimeError("exactly forty distinct selected prefixes required")
    if prefix in _CLAIMED or len(_CLAIMED) >= 40:
        raise RuntimeError("duplicate or over-limit layer pack")
    _CLAIMED.add(prefix)
    import torch
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    cfg = expert.moe_config
    par = cfg.moe_parallel_config
    if not (cfg.num_experts == 384 and cfg.num_local_experts == 384 and
            cfg.hidden_dim == 5120 and cfg.experts_per_token == 6 and
            cfg.intermediate_size_per_partition == 384 and
            cfg.intermediate_size_per_partition_unpadded == 288 and
            not cfg.tp_shard_with_padding and not cfg.has_bias and not cfg.is_lora_enabled and
            cfg.activation == MoEActivation.SILU and os.environ.get("DSV41_MOE_COMPACT") == "1" and
            par.tp_size == 8 and par.dp_size == 1 and not par.use_ep and not par.enable_eplb):
        raise RuntimeError("selected layer does not match TP8 target geometry")
    src = layer.w2_weight
    if not (src.is_cuda and src.is_contiguous() and tuple(src.shape) == (384, 5120, 192)
            and src.dtype in (torch.uint8, torch.int8)):
        raise RuntimeError("selected layer has incompatible native FP4 weight metadata")
    if torch.cuda.get_device_capability(src.device) != (12, 1):
        raise RuntimeError("expected SM121 device")
    reserve_raw = os.environ.get("ATHENA_FC2_FUTURE_RESERVE_MIB")
    if reserve_raw is None or not reserve_raw.isdigit():
        raise RuntimeError("set nonnegative ATHENA_FC2_FUTURE_RESERVE_MIB explicitly")
    reserve = int(reserve_raw) * 1024**2
    raw = os.environ.get("ATHENA_FC2_PACK_SO", "")
    binary = Path(raw)
    if (not raw or not binary.is_absolute() or not binary.is_file()
            or not binary.resolve().is_relative_to(_ATHENA)):
        raise RuntimeError("pack extension must be an existing absolute file inside Athena")
    host_free = _host_available()
    gpu_free, _ = torch.cuda.mem_get_info(src.device)
    minimum = _FLOOR + _INCREMENTAL_CAP + reserve
    if host_free < minimum or gpu_free < minimum:
        raise RuntimeError(f"insufficient one-layer pack reserve: host={host_free}, cuda={gpu_free}, required={minimum}")
    spec = importlib.util.spec_from_file_location("athena_fc2_k288_pack_sm121", binary)
    if spec is None or spec.loader is None:
        raise RuntimeError("pack extension not loadable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "pack", None)):
        raise RuntimeError("pack extension missing pack entry")
    with torch.cuda.device(src.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("load-time pack must precede graph capture")
        start_wall = time.monotonic()
        dst = torch.empty((384, 80, 9, 1024), dtype=torch.uint8, device=src.device)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        module.pack(src, dst)
        end.record()
        end.synchronize()
        ms = float(start.elapsed_time(end))
        host_after = _host_available()
        cuda_after, _ = torch.cuda.mem_get_info(src.device)
        if host_after < _FLOOR + reserve or cuda_after < _FLOOR + reserve:
            raise RuntimeError("memory floor/future reserve breached after one-layer packing")
        if max(0, gpu_free - cuda_after) > _INCREMENTAL_CAP:
            raise RuntimeError("observed device-free reduction exceeded the incremental pack cap")
    scales = layer.w2_weight_scale
    print(f"athena_fc2_one_layer prefix={prefix!r} pack_ms={ms:.3f} "
          f"wall_s={time.monotonic()-start_wall:.3f} packed_bytes={dst.numel()} "
          f"native_dtype={src.dtype} native_shape={tuple(src.shape)} native_stride={tuple(src.stride())} "
          f"scale_dtype={scales.dtype} scale_shape={tuple(scales.shape)} scale_stride={tuple(scales.stride())} "
          f"host_before={host_free} host_after={host_after} cuda_before={gpu_free} cuda_after={cuda_after}",
          flush=True)
    # Only the separate copy is held; layer.w2_weight and FC2C remain unchanged.
    # The overlay does not use `dst` to produce model output.
    return dst
