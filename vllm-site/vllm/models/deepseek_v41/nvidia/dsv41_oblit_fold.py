# SPDX-License-Identifier: Apache-2.0
"""r2-oblit: fold the DSV41_OBLIT refusal-direction projections into the Moet mHC boundary kernel (climb round 2, G03).

DEFAULT OFF. model.py imports this module only when DSV41_OBLIT_FOLD is set to a non-off value; with it unset nothing
here runs, the extension is never built or loaded, and every projection stays the Triton _proj_rows launch it is today.

DSV41_OBLIT_FOLD (read once at import):
  0 / off / unset   off (default)
  ffn               attention -> FFN boundary of every bound layer (same layer, same direction d_l):
                      the attention-output projection becomes the kernel's prologue (on the all-reduce output; it also
                      writes x' out of place for the side-stream mhc_proj), the FFN-input projection its epilogue.
                      Removes 2 _proj_rows launches per bound layer (76 per target forward).
  attn              FFN -> attention boundary of every bound layer, epilogue only: the attention-input projection
                      (d_l of the layer whose attention follows). The previous layer's FFN-output projection
                      (project_moe, d_{l-1}) stays a separate launch: the cross-layer prologue is NOT folded (stage 2).
                      Layer 14 (Engram boundary, mhc_pre path) and every non-graph path keep their launches.
  1 / all / ffn,attn  both of the above (113 launches per target forward).

The fold engages only where the production Moet boundary engages: inside a FULL CUDA-graph capture of 1..MAX_TOKENS
tokens, DSV41_MHC_OVERLAP=moet, the dsv41_mhc_overlap applicability predicate true, H == 5120, the direction already on
the device (it is created by the eager forwards before any capture). Everywhere else (eager warm-up, prefill, piecewise
graphs, > MAX_TOKENS, sequence parallel, the Engram boundary) the Triton projections run exactly as today; model.py asks
ffn_ok/attn_ok BEFORE skipping a projection and dsv41_mhc_overlap fails closed (RuntimeError) if it is handed a fold
it cannot take, so a projection can never be dropped silently.

Exactness: dsv41_oblit_fold_sm120.cu replicates _proj_rows' op order bit for bit (EXACTNESS.md in the kit), the Moet
arithmetic is copied verbatim. device_test.py proves it on the device before the flag may serve.

Serving gate (prepare(), flag on only): the fold engages only when
  - the extension is PREBUILT (install.py build): prepare() never JIT-compiles inside make_stream; a missing .so or a
    stale torch build lock (BUILD_DIR/lock, left by an interrupted build; it would hang FileBaton) disables the fold;
  - DSV41_OBLIT_FOLD_SO_SHA256 equals the sha256 of the loaded .so, i.e. the binary device_test.py passed on (it prints
    the value at exit 0). Unset or different -> disabled. A binary nobody device-tested can therefore never serve.

Other knobs:
  DSV41_OBLIT_FOLD_ARCH    TORCH_CUDA_ARCH_LIST for the JIT build (default "12.0", the same target the live Moet
                           extension builds for: VLLM_MOET_MHC_ARCH default; restored after the build)
  DSV41_OBLIT_FOLD_BUILD   build directory (default ~/.cache/dsv41-oblit-fold); install.py build fills it on each node
  DSV41_OBLIT_FOLD_SO_SHA256  the device-tested .so sha256 (required when the flag is on, see above)
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

import torch

logger = logging.getLogger("vllm.dsv41_oblit_fold")

_HERE = Path(__file__).resolve().parent
SOURCE = _HERE / "dsv41_oblit_fold_sm120.cu"
EXT_NAME = "dsv41_r2o_mhc_fold_sm120"
FOLD_H = 5120


def parse_mode(raw: str) -> tuple[bool, bool]:
    """-> (ffn, attn)."""
    v = (raw or "").strip().lower()
    if v in ("", "0", "off", "false", "no"):
        return False, False
    if v in ("1", "all", "on", "true", "yes"):
        return True, True
    parts = {p.strip() for p in v.replace("+", ",").split(",") if p.strip()}
    bad = parts - {"ffn", "attn"}
    if bad:
        raise ValueError(f"DSV41_OBLIT_FOLD must be 0|ffn|attn|ffn,attn|1, got {raw!r}")
    return "ffn" in parts, "attn" in parts


FFN, ATTN = parse_mode(os.environ.get("DSV41_OBLIT_FOLD", "0"))
ARCH = os.environ.get("DSV41_OBLIT_FOLD_ARCH", "12.0").strip() or "12.0"
BUILD_DIR = os.path.expanduser(os.environ.get("DSV41_OBLIT_FOLD_BUILD", "~/.cache/dsv41-oblit-fold"))
SO_PATH = os.path.join(BUILD_DIR, EXT_NAME + ".so")
LOCK_PATH = os.path.join(BUILD_DIR, "lock")  # torch.utils.file_baton lock of cpp_extension.load
SO_SHA256 = os.environ.get("DSV41_OBLIT_FOLD_SO_SHA256", "").strip().lower()

_EXT = None
_DISABLED: str | None = None  # reason, once a precondition failed (logged once; every *_ok() is then False)


def _ext():
    """Build (or reuse) and import the extension. Never called inside a capture (prepare() runs at make_stream)."""
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load

        if os.path.exists(LOCK_PATH):
            raise RuntimeError(f"stale build lock {LOCK_PATH} (interrupted build?): torch would wait on it forever; "
                               "remove it once no build runs, then rerun install.py build")
        os.makedirs(BUILD_DIR, exist_ok=True)
        prev = os.environ.get("TORCH_CUDA_ARCH_LIST")
        os.environ["TORCH_CUDA_ARCH_LIST"] = ARCH
        try:
            _EXT = load(
                name=EXT_NAME,
                sources=[str(SOURCE)],
                extra_cuda_cflags=["-O3", "-std=c++20"],
                extra_cflags=["-O3", "-std=c++20"],
                build_directory=BUILD_DIR,
                verbose=bool(int(os.environ.get("DSV41_OBLIT_FOLD_VERBOSE", "0"))),
            )
        finally:
            if prev is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = prev
    return _EXT


def _ovl():
    from vllm.models.deepseek_v41.nvidia import dsv41_mhc_overlap as ovl

    return ovl


def _disable(reason: str) -> None:
    global _DISABLED
    if _DISABLED is None:
        _DISABLED = reason
        logger.warning("DSV41_OBLIT_FOLD disabled: %s (the Triton projections run as before)", reason)


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def serving_gate() -> str | None:
    """None when the prebuilt, device-tested binary is in place; else the reason the fold must stay off. Checked before
    the extension is loaded (no JIT build in the serving process) and again on the loaded file."""
    if os.path.exists(LOCK_PATH):
        return f"stale build lock {LOCK_PATH}"
    if not os.path.isfile(SO_PATH):
        return f"extension not prebuilt ({SO_PATH} missing: run install.py build)"
    if not SO_SHA256:
        return "DSV41_OBLIT_FOLD_SO_SHA256 unset (set it to the .so sha256 device_test.py passed on)"
    got = file_sha256(SO_PATH)
    if got != SO_SHA256:
        return f"{SO_PATH} sha256 {got[:16]} != DSV41_OBLIT_FOLD_SO_SHA256 {SO_SHA256[:16]} (not the device-tested binary)"
    return None


def prepare(hidden: int, max_tokens: int) -> None:
    """Called from dsv41_mhc_overlap._prepare (eager, before any capture): load the prebuilt extension and size its
    [T, H] workspace for max_tokens rows, so a capture never re-grows it. Disables the fold (never the model) on failure."""
    if not (FFN or ATTN) or _DISABLED is not None:
        return
    try:
        from vllm.models.deepseek_v41.nvidia import dsv41_oblit

        if not dsv41_oblit.ON:
            _disable("DSV41_OBLIT is off")
            return
        ovl = _ovl()
        if ovl.MODE != "moet":
            _disable(f"DSV41_MHC_OVERLAP={ovl.MODE} (the fold is a variant of the moet boundary)")
            return
        if ovl.MOET_MODE != 1:
            _disable(f"DSV41_MHC_OVERLAP_MOET_MODE={ovl.MOET_MODE} (built for mode 1)")
            return
        if hidden != FOLD_H:
            _disable(f"hidden {hidden} != {FOLD_H}")
            return
        why = serving_gate()
        if why is not None:
            _disable(why)
            return
        loaded = os.path.realpath(_ext().__file__)
        if loaded != os.path.realpath(SO_PATH) or file_sha256(loaded) != SO_SHA256:
            _disable(f"loaded {loaded} is not the gated binary {SO_PATH}")
            return
        dev = torch.device("cuda", torch.cuda.current_device())
        n = max(1, int(max_tokens))
        x = torch.zeros(n, hidden, dtype=torch.bfloat16, device=dev)
        res = torch.zeros(n, 4, hidden, dtype=torch.bfloat16, device=dev)
        post = torch.zeros(n, 4, 1, dtype=torch.float32, device=dev)
        comb = torch.zeros(n, 4, 4, dtype=torch.float32, device=dev)
        pre = torch.full((n, 4), 0.25, dtype=torch.float32, device=dev)
        w = torch.ones(hidden, dtype=torch.bfloat16, device=dev)
        d = torch.zeros(hidden, dtype=torch.float32, device=dev)
        d[0] = 1.0
        for pro, epi in ((True, True), (False, True)):
            post_norm(x, res, post, comb, pre, w, 1e-6, Spec(d, pro, epi), capture_aux=True)
        torch.cuda.synchronize(dev)
        logger.info("DSV41_OBLIT_FOLD: ffn=%d attn=%d (extension %s, arch %s, so sha256 %s)", FFN, ATTN, EXT_NAME, ARCH,
                    SO_SHA256[:16])
    except Exception as exc:  # noqa: BLE001
        _disable(f"warm-up failed: {exc!r}")


class Spec:
    """What one boundary folds: d = fp32 [H] unit direction on the device (the dsv41_oblit._dev cache entry),
    pro = project x (the sublayer output) before the post-mix and write x' out of place, epi = project the
    normalized layer input."""

    __slots__ = ("d", "pro", "epi")

    def __init__(self, d: torch.Tensor, pro: bool, epi: bool):
        self.d, self.pro, self.epi = d, bool(pro), bool(epi)


def _dir(layer: int, like: torch.Tensor) -> torch.Tensor | None:
    """The cached device direction, or None if it does not exist yet (never created here: no capture-time allocation)."""
    from vllm.models.deepseek_v41.nvidia import dsv41_oblit

    v = dsv41_oblit._dev.get((f"L{layer}", str(like.device)))
    if v is None or v.dtype != torch.float32 or v.numel() != FOLD_H or not v.is_contiguous() or v.data_ptr() % 16:
        return None
    return v


def _ok(enabled: bool, layer: int, x, residual, pre_mix, norm_weight, stream, fuse: bool) -> bool:
    if not enabled or _DISABLED is not None or _EXT is None or layer < 0 or stream is None or fuse:
        return False
    if not isinstance(x, torch.Tensor) or not x.is_cuda or not torch.cuda.is_current_stream_capturing():
        return False
    ovl = _ovl()
    if not ovl.ACTIVE or ovl.MODE != "moet" or ovl.MOET_MODE != 1:
        return False
    if not ovl.applicable(x, residual, pre_mix, norm_weight):
        return False
    if residual.shape[-1] != FOLD_H:
        return False
    for t in (pre_mix,):
        if t.data_ptr() % 16:
            return False
    return _dir(layer, x) is not None


def ffn_ok(layer, x, residual, pre_mix, norm_weight, stream, fuse) -> bool:
    """True iff the attention -> FFN boundary of `layer` will take the fold: model.py then skips the attention-output and
    the FFN-input projection and passes kw(...) to mhc_shifted_post_pre."""
    return _ok(FFN, layer, x, residual, pre_mix, norm_weight, stream, fuse)


def attn_ok(layer, x, residual, pre_mix, norm_weight, stream, fuse) -> bool:
    """True iff the shifted attention boundary of `layer` will take the epilogue fold (attention-input projection)."""
    return _ok(ATTN, layer, x, residual, pre_mix, norm_weight, stream, fuse)


def kw(layer: int, like: torch.Tensor, pro: bool, epi: bool) -> dict:
    d = _dir(layer, like)
    if d is None:  # ffn_ok/attn_ok checked it; never reached
        raise RuntimeError(f"DSV41_OBLIT_FOLD: direction L{layer} not on the device")
    return {"dsv41_oblit": Spec(d, pro, epi)}


def post_norm(x, residual, post_layer_mix, comb_res_mix, pre_mix, norm_weight, norm_eps, spec: Spec, *,
              capture_aux: bool = False, pdl: bool | None = None):
    """The Moet mhc_post_norm with the fold: returns (residual_out, layer_input, aux, x_used) where x_used is x' (written
    by the kernel, out of place) when spec.pro, else x. Allocations match the Moet wrapper's plus x' (which replaces the
    two Triton outputs the unfolded boundary allocates)."""
    if pdl is None:
        pdl = _ovl()._moet().PDL_DEFAULT
    T, hc, H = residual.shape
    residual_out = torch.empty_like(residual)
    layer_input = torch.empty(T, H, dtype=torch.bfloat16, device=residual.device)
    aux = torch.empty(T if capture_aux else 0, H, dtype=torch.bfloat16, device=residual.device)
    x_out = torch.empty_like(x) if spec.pro else None
    _ext().mhc_post_norm_fold(
        x, residual, post_layer_mix.reshape(T, hc).contiguous(), comb_res_mix.contiguous(), pre_mix.contiguous(),
        norm_weight.contiguous(), residual_out, layer_input, aux if capture_aux else None, float(norm_eps), 1,
        bool(pdl), spec.d, bool(spec.pro), bool(spec.epi), x_out,
    )
    return residual_out, layer_input, aux, (x_out if spec.pro else x)
