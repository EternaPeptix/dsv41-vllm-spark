"""dsv41 climb r3, lever G02 (attn:L-XS): XSPLIT routing gate for the DSV4 dual-cache MLA call.

Imported only from the DSV41_ATTN_XSPLIT=1 branch of ``_sparse_mla_sm120._paged_attention``.
``try_run`` launches the fused split-tile kernel (libdsv41_xsplit.so, ctypes, no torch ABI) for
calls inside the production envelope and returns True; anything else returns False and the
caller runs the live kernel unchanged.

Envelope (all must hold, else the live path runs): planner variant PREFILL_MG_DUAL, model DSV4
(d_qk 512), 8 heads, 1 <= num_tokens <= DSV41_ATTN_XSPLIT_MAX_T (64), main page block 64,
extra page block 64 or 32, both runtime length tensors present (the live kernel is then
sparse_mla_prefill_mg_dual_kernel<DSV4, BF16, 8, 64, PBSX, 1>), dense contiguous inputs as
the live binding requires, num_tokens * max_tiles <= DSV41_ATTN_XSPLIT_CAP_UNITS, and a
pre-allocated workspace slot for the current stream.

Workspaces are allocated once, on the first eager (non-capturing) call per device, as
DSV41_ATTN_XSPLIT_NWS slots; each stream that calls gets its own slot (assigned without
allocation, so a capture stream can take one). A call made during capture with no slot left
falls back to the live kernel. The first eager call of every shape is checked bit for bit
against the live kernel (DSV41_ATTN_XSPLIT_CHECK=1, default) and a shape is never routed inside
a CUDA graph capture before it passed that check. A mismatch raises RuntimeError (fail closed;
DSV41_ATTN_XSPLIT_ON_MISMATCH=disable keeps the live result and turns XSPLIT off instead).
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import sys
import threading

import torch

_ENV = os.environ
_LIB_DIR = _ENV.get("DSV41_ATTN_XSPLIT_LIB", os.path.expanduser("~/dsv41-r3/xsplit/build"))
_LIB_NAME = "libdsv41_xsplit.so"
_MAX_T = int(_ENV.get("DSV41_ATTN_XSPLIT_MAX_T", "64"))
_CAP_UNITS = int(_ENV.get("DSV41_ATTN_XSPLIT_CAP_UNITS", "1152"))
_GRID = int(_ENV.get("DSV41_ATTN_XSPLIT_GRID", "46"))
_NWS = int(_ENV.get("DSV41_ATTN_XSPLIT_NWS", "2"))
_CHECK = _ENV.get("DSV41_ATTN_XSPLIT_CHECK", "1") == "1"
_VERBOSE = _ENV.get("DSV41_ATTN_XSPLIT_VERBOSE", "0") == "1"
# Self-check mismatch: "raise" (default, fail closed: the engine stops with a RuntimeError before
# it serves anything) or "disable" (keep the live result, XSPLIT off for the rest of the process).
_ON_MISMATCH = _ENV.get("DSV41_ATTN_XSPLIT_ON_MISMATCH", "raise")

_ABI_VERSION = 1
_BI = 64
_NUM_HEADS = 8
_D_QK = 512
_D_V = 512
_BYTES_PER_TOKEN = 584  # packed SM120 DSV4 row (448 fp8 nope + 64 bf16 rope + 8 B scale footer)
_MODEL_TYPE_DSV4 = 1
_MODEL_TYPE_AUTO = -1
_PBS_MAIN = 64
_PBSX_OK = (64, 32)

_lock = threading.Lock()
_lib = None
_lib_failed = False
_disabled = False
_pools: dict = {}  # device index -> {"slots": [tensor], "owner": [stream ptr or None]}
_checked: set = set()
_seen: set = set()
_stats = {
    "routed": 0,
    "fallback_envelope": 0,
    "fallback_no_ws": 0,
    "fallback_cap": 0,
    "fallback_lib": 0,
    "fallback_unchecked": 0,
    "checks_ok": 0,
    "checks_bad": 0,
    "launch_errors": 0,
}


def stats() -> dict:
    return dict(_stats, disabled=_disabled, lib_dir=_LIB_DIR)


def _log(msg: str) -> None:
    print(f"[dsv41_xsplit] {msg}", file=sys.stderr, flush=True)


def _load():
    """Load the prebuilt .so once; its sha256 must match the installed sidecar."""
    global _lib, _lib_failed
    if _lib is not None or _lib_failed:
        return _lib
    with _lock:
        if _lib is not None or _lib_failed:
            return _lib
        path = os.path.join(_LIB_DIR, _LIB_NAME)
        try:
            with open(path, "rb") as f:
                digest = hashlib.sha256(f.read()).hexdigest()
            with open(path + ".sha256") as f:
                want = f.read().split()[0]
            if digest != want:
                raise RuntimeError(f"sha256 {digest} != sidecar {want}")
            lib = ctypes.CDLL(path)
            if lib.dsv41_xsplit_abi_version() != _ABI_VERSION:
                raise RuntimeError("ABI version mismatch")
            if lib.dsv41_xsplit_arm() != 0:
                raise RuntimeError("timing-arm build loaded as the serving library")
            lib.dsv41_xsplit_ws_bytes.restype = ctypes.c_ulonglong
            lib.dsv41_xsplit_ws_bytes.argtypes = [ctypes.c_int]
            vp = ctypes.c_void_p
            ull = ctypes.c_ulonglong
            ci = ctypes.c_int
            lib.dsv41_xsplit_launch.restype = ci
            lib.dsv41_xsplit_launch.argtypes = [
                ci, vp, vp, vp, vp, vp, vp, vp, vp, ctypes.c_float,
                ci, ci, ci, ull, ull, ull, vp, vp, vp, ci, ci, vp,
            ]
            _lib = lib
            _log(f"loaded {path} sha256={digest[:16]} cap_units={_CAP_UNITS} grid={_GRID} "
                 f"max_t={_MAX_T} nws={_NWS} check={int(_CHECK)}")
        except Exception as e:  # noqa: BLE001 - any failure keeps the live kernel
            _lib_failed = True
            _log(f"disabled: cannot load {path}: {e!r}")
        return _lib


def _cuda(t) -> bool:
    return bool(t.is_cuda)


def _kv_layout(kv: torch.Tensor):
    """parse_paged_kv_layout of the live binding (sparse_mla_sm120.cu), prefill form."""
    if kv.dtype != torch.uint8 or not _cuda(kv) or kv.stride(-1) != 1:
        return None
    if kv.ndim == 3:
        if kv.size(-1) != _BYTES_PER_TOKEN:
            return None
        return int(kv.size(1)), int(kv.stride(0))
    if kv.ndim == 4:
        if kv.size(-1) != _BYTES_PER_TOKEN:
            return None
        if kv.size(1) == 1:
            return int(kv.size(2)), int(kv.stride(0))
        if kv.size(2) == 1:
            return int(kv.size(1)), int(kv.stride(0))
    return None


def _dense_indices(idx: torch.Tensor, num_tokens: int):
    if idx is None or idx.dtype != torch.int32 or not _cuda(idx) or not idx.is_contiguous():
        return None
    if idx.ndim not in (2, 3) or idx.size(0) != num_tokens:
        return None
    if idx.ndim == 3 and idx.size(1) != 1:
        return None
    width = int(idx.size(-1))
    if idx.stride(-1) != 1 or idx.stride(0) != width or width <= 0:
        return None
    return width


def _lengths_ok(t, num_tokens: int) -> bool:
    return (
        t is not None
        and _cuda(t)
        and t.dtype == torch.int32
        and t.ndim == 1
        and t.size(0) == num_tokens
        and t.is_contiguous()
    )


def _envelope(q, kv_cache, indices, output, out_lse, model_type, topk_length, attn_sink,
              extra_kv_cache, extra_indices, extra_topk_length):
    if model_type not in (_MODEL_TYPE_DSV4, _MODEL_TYPE_AUTO):
        return None
    if q.dtype != torch.bfloat16 or q.ndim != 3 or not _cuda(q) or not q.is_contiguous():
        return None
    T, H, d_qk = (int(s) for s in q.shape)
    if H != _NUM_HEADS or d_qk != _D_QK or T < 1 or T > _MAX_T:
        return None
    dev = q.device
    main = _kv_layout(kv_cache)
    if main is None or main[0] != _PBS_MAIN or kv_cache.device != dev:
        return None
    if extra_kv_cache is None or extra_indices is None:
        return None
    extra = _kv_layout(extra_kv_cache)
    if extra is None or extra[0] not in _PBSX_OK or extra_kv_cache.device != dev:
        return None
    topk = _dense_indices(indices, T)
    topkx = _dense_indices(extra_indices, T)
    if topk is None or topkx is None or topk % _BI != 0:
        return None
    if indices.device != dev or extra_indices.device != dev:
        return None
    if not (_lengths_ok(topk_length, T) and _lengths_ok(extra_topk_length, T)):
        return None
    if topk_length.device != dev or extra_topk_length.device != dev:
        return None
    if (
        output.dtype != torch.bfloat16
        or not output.is_contiguous()
        or tuple(output.shape) != (T, _NUM_HEADS, _D_V)
        or output.device != dev
    ):
        return None
    if (
        out_lse.dtype != torch.float32
        or out_lse.ndim != 2
        or tuple(out_lse.shape) != (T, _NUM_HEADS)
        or out_lse.stride(-1) != 1
        or out_lse.device != dev
    ):
        return None
    if attn_sink is not None:
        if (
            attn_sink.dtype != torch.float32
            or attn_sink.ndim != 1
            or attn_sink.size(0) != _NUM_HEADS
            or not attn_sink.is_contiguous()
            or attn_sink.device != dev
        ):
            return None
    nmax = topk // _BI + (topkx + _BI - 1) // _BI
    return {
        "T": T,
        "topk": topk,
        "topkx": topkx,
        "pbsx": extra[0],
        "stride_kv": main[1],
        "stride_kvx": extra[1],
        "stride_lse": int(out_lse.stride(0)),
        "nmax": nmax,
    }


def _ensure_pool(device: torch.device):
    """Allocate the workspace slots once per device, outside any capture. Called on the first
    eager MG_DUAL call whatever its shape, i.e. normally inside vLLM's memory-profiling run, so the
    bytes are accounted before the KV cache is sized."""
    dev = device.index if device.index is not None else torch.cuda.current_device()
    pool = _pools.get(dev)
    if pool is None:
        lib = _load()
        if lib is None:
            return None
        nbytes = int(lib.dsv41_xsplit_ws_bytes(_CAP_UNITS))
        slots = [torch.zeros(nbytes, dtype=torch.uint8, device=device) for _ in range(_NWS)]
        pool = {"slots": slots, "owner": [None] * _NWS}
        _pools[dev] = pool
        _log(f"workspace: {_NWS} x {nbytes / 2**20:.1f} MiB on cuda:{dev}")
    return pool


def _ws_for(device: torch.device, stream_ptr: int, capturing: bool):
    dev = device.index if device.index is not None else torch.cuda.current_device()
    pool = _pools.get(dev)
    if pool is None:
        if capturing:
            return None
        pool = _ensure_pool(device)
        if pool is None:
            return None
    owners = pool["owner"]
    for i, o in enumerate(owners):
        if o == stream_ptr:
            return pool["slots"][i]
    for i, o in enumerate(owners):
        if o is None:
            owners[i] = stream_ptr
            return pool["slots"][i]
    return None


def _launch(lib, a, q, kv_cache, indices, output, out_lse, sm_scale, topk_length, attn_sink,
            extra_kv_cache, extra_indices, extra_topk_length, ws, stream_ptr) -> int:
    vp = ctypes.c_void_p
    grid = min(_GRID, a["T"] * a["nmax"] + a["T"])
    return lib.dsv41_xsplit_launch(
        a["pbsx"],
        vp(q.data_ptr()),
        vp(kv_cache.data_ptr()),
        vp(indices.data_ptr()),
        vp(extra_kv_cache.data_ptr()),
        vp(extra_indices.data_ptr()),
        vp(attn_sink.data_ptr() if attn_sink is not None else 0),
        vp(output.data_ptr()),
        vp(out_lse.data_ptr()),
        ctypes.c_float(float(sm_scale)),
        a["T"],
        a["topk"],
        a["topkx"],
        a["stride_kv"],
        a["stride_kvx"],
        a["stride_lse"],
        vp(topk_length.data_ptr()),
        vp(extra_topk_length.data_ptr()),
        vp(ws.data_ptr()),
        _CAP_UNITS,
        grid,
        vp(stream_ptr),
    )


def try_run(module, variant: int, q, kv_cache, indices, output, out_lse, sm_scale, model_type,
            topk_length, attn_sink, extra_kv_cache, extra_indices, extra_topk_length) -> bool:
    """Run XSPLIT for an in-envelope MG_DUAL call. False means: run the live kernel."""
    global _disabled
    if _disabled:
        return False
    capturing = torch.cuda.is_current_stream_capturing()
    if not capturing and not _lib_failed:
        try:
            if _cuda(q):
                _ensure_pool(q.device)
        except Exception as e:  # noqa: BLE001 - allocation failure keeps the live kernel
            _log(f"workspace allocation failed: {e!r}")
    try:
        a = _envelope(q, kv_cache, indices, output, out_lse, model_type, topk_length, attn_sink,
                      extra_kv_cache, extra_indices, extra_topk_length)
    except Exception:  # noqa: BLE001
        a = None
    if a is None:
        _stats["fallback_envelope"] += 1
        return False
    if a["T"] * a["nmax"] > _CAP_UNITS:
        _stats["fallback_cap"] += 1
        return False
    lib = _load()
    if lib is None:
        _stats["fallback_lib"] += 1
        return False
    stream = torch.cuda.current_stream(q.device)
    stream_ptr = int(stream.cuda_stream)
    ws = _ws_for(q.device, stream_ptr, capturing)
    if ws is None:
        _stats["fallback_no_ws"] += 1
        return False
    key = (a["T"], a["topk"], a["topkx"], a["pbsx"])
    if key not in _seen:
        _seen.add(key)
        if _VERBOSE:
            _log(f"route T={key[0]} topk={key[1]} topkx={key[2]} pbsx={key[3]} nmax={a['nmax']} "
                 f"capturing={capturing}")
    args = (q, kv_cache, indices, output, out_lse, sm_scale, topk_length, attn_sink,
            extra_kv_cache, extra_indices, extra_topk_length, ws, stream_ptr)

    if _CHECK and key not in _checked:
        if capturing:
            # Never route a shape into a graph before it passed the eager self-check.
            _stats["fallback_unchecked"] += 1
            if ("unchecked",) + key not in _seen:
                _seen.add(("unchecked",) + key)
                _log(f"capture of unchecked shape T={key[0]} topk={key[1]} topkx={key[2]} "
                     f"pbsx={key[3]}: live kernel kept for this graph")
            return False
        # Bitwise self-check of this shape against the live kernel on the live inputs.
        out_ref = torch.empty_like(output)
        lse_ref = torch.empty_strided(out_lse.size(), out_lse.stride(), dtype=out_lse.dtype,
                                      device=out_lse.device)
        module.sparse_mla_sm120_paged_attention(
            q, kv_cache, indices, out_ref, lse_ref, sm_scale, model_type, int(variant),
            topk_length, attn_sink, extra_kv_cache, extra_indices, extra_topk_length)
        rc = _launch(lib, a, *args)
        if rc != 0:
            _stats["launch_errors"] += 1
            _disabled = True
            _log(f"disabled: launch error {rc} for {key}")
            output.copy_(out_ref)
            out_lse.copy_(lse_ref)
            return True
        same = torch.equal(output.view(torch.int16), out_ref.view(torch.int16)) and torch.equal(
            out_lse.view(torch.int32), lse_ref.view(torch.int32))
        _checked.add(key)
        if not same:
            _stats["checks_bad"] += 1
            _disabled = True
            _log(f"DISABLED: bitwise self-check FAILED for T={key[0]} topk={key[1]} "
                 f"topkx={key[2]} pbsx={key[3]}; live result kept, XSPLIT off for this process")
            output.copy_(out_ref)
            out_lse.copy_(lse_ref)
            if _ON_MISMATCH != "disable":
                raise RuntimeError(
                    f"dsv41_xsplit: bitwise self-check failed for T={key[0]} topk={key[1]} "
                    f"topkx={key[2]} pbsx={key[3]} (DSV41_ATTN_XSPLIT=1); refusing to serve")
            return True
        _stats["checks_ok"] += 1
        _stats["routed"] += 1
        return True

    rc = _launch(lib, a, *args)
    if rc != 0:
        _stats["launch_errors"] += 1
        _disabled = True
        _log(f"disabled: launch error {rc} for {key}")
        return False
    _stats["routed"] += 1
    return True
