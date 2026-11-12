# SPDX-License-Identifier: Apache-2.0
"""DSv4.1 climb round 2, kit r2-skinny: skinny_mx v2 for the qkv_a column shard (default OFF).

Install path: vllm/models/deepseek_v41/nvidia/dsv41_r2_skinny.py. Imported only from dsv41_r3_dense.py and only
when DSV41_SKINNY_V2 is set, so with the flag unset nothing here runs.

Flags (every rank must see the same values):

  DSV41_SKINNY_V2=1|pdl      (lever G10 / dense:D3) qkv_a shard GEMV through skinny_mx_v2_ext.so instead of the
                             live skinny_mx_ext.so: same per-output fp32 op sequence (K split 8, lane-strided
                             fmaf walk, xor tree, q-order combine, bf16 rn), latency-restructured schedule.
                             pdl = also launch it with programmatic stream serialization: the weight loads are
                             issued before griddepcontrol.wait, overlapping the producer kernel's tail.
  DSV41_SKINNY_QFUSE=1       (lever G09 part D2) needs DSV41_SKINNY_V2: the MXFP8 activation quant of the qkv_a
                             input runs inside the v2 prologue (FlashInfer CuTe-DSL op sequence as PTX) and the
                             separate MXFP8QuantizeSwizzled launch disappears.
  DSV41_SKINNY_V2_LIB=dir    skinny_mx_v2_ext*.so location (default /home/jeweled/dsv41-r2/skinny_mx_v2/build)
  DSV41_SKINNY_V2_GRID=0     v2 grid (0 = #SMs, as v1). Any grid with ceil(224 / grid) <= 8 is exact (the block
                             a row lands in does not enter its op sequence); for timing sweeps only.
  DSV41_SKINNY_V2_CHECK=1    0 skips the build-time self-check (unit tests only, never in serving)

Enforced, not assumed: the first eager (non-capturing) build of each layer's shard runs, on that layer's real
shard weights, every M in 1..DSV41_QKV_A_SHARD_MAX_M through the live chain (mxfp8_e4m3_quantize + v1 skinny_mx_a8)
and through v2 (A8 and, with QFUSE, the fused-quant entry), on synthetic and hostile activations, and serves v2
for an M only if all outputs are bitwise equal. The fused quant is also compared byte for byte (e4m3 codes and
ue8m0 scales) with the live FlashInfer quant on hostile rows, which also pins the FTZ mode of its amax * (1/448)
multiply; if neither mode matches, QFUSE stays off. Any failure leaves that M (or QFUSE) on the live path and logs
a warning ("DSV41_SKINNY_V2 ... kept live").
"""
from __future__ import annotations

import glob
import importlib.util
import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

MODE = os.environ.get("DSV41_SKINNY_V2", "0").strip().lower()
if MODE not in ("1", "pdl"):
    raise ValueError(f"DSV41_SKINNY_V2 must be 1|pdl when this module is imported, got {MODE!r}")
PDL = MODE == "pdl"
QFUSE = os.environ.get("DSV41_SKINNY_QFUSE", "0").strip() == "1"
LIB_DIR = os.environ.get("DSV41_SKINNY_V2_LIB", "/home/jeweled/dsv41-r2/skinny_mx_v2/build")
GRID = int(os.environ.get("DSV41_SKINNY_V2_GRID", "0"))
CHECK = os.environ.get("DSV41_SKINNY_V2_CHECK", "1") != "0"

_ext = None
_warned: set[str] = set()


def _warn_once(key: str, msg: str, *args) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning(msg, *args)


def ext():
    global _ext
    if _ext is None:
        cands = sorted(glob.glob(os.path.join(LIB_DIR, "skinny_mx_v2_ext*.so")))
        if not cands:
            raise RuntimeError(f"DSV41_SKINNY_V2: no skinny_mx_v2_ext*.so in {LIB_DIR}")
        spec = importlib.util.spec_from_file_location("skinny_mx_v2_ext", cands[0])
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        mod.prepare()  # smem attributes for every instantiation, eagerly (never during capture)
        _ext = mod
        logger.info("DSV41_SKINNY_V2: skinny_mx_v2 loaded from %s (mode=%s qfuse=%s grid=%d)", cands[0], MODE,
                    QFUSE, GRID)
    return _ext


# ------------------------------------------------------------------ self-check inputs
def hostile_rows(M: int, K: int, seed: int, device) -> torch.Tensor:
    """bf16 [M, K]: per-32-block magnitude spread 2^-30..2^20, exact zeros and -0, bf16 subnormals, the
    amax * (1/448) FTZ band (amax in (448 * 2^-127, 448 * 2^-126)), values that round to e4m3 ties, power-of-two
    and just-above-power-of-two block maxima (ue8m0 ceil edge), +-448 * 2^e saturation edges, max finite bf16."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    nb = K // 32
    sh = (M, nb, 32)
    x = torch.randn(sh, generator=g) * torch.exp2(torch.randint(-30, 21, (M, nb, 1), generator=g).float())
    kind = torch.randint(0, 10, (M, nb, 1), generator=g)
    lane = torch.arange(32).view(1, 1, 32)
    e = torch.randint(-20, 10, (M, nb, 1), generator=g).float()
    pos = torch.randint(0, 32, (M, nb, 1), generator=g)
    k0 = torch.zeros(sh)
    k0[..., 1::3] = -0.0
    k1 = torch.randint(-127, 128, sh, generator=g).float() * 2.0 ** -133  # bf16 subnormals
    band = 448.0 * 2.0 ** -127 * (1.0 + torch.rand((M, nb, 1), generator=g) * 0.99)  # amax*(1/448) FTZ band
    k2 = torch.where(lane == pos, band, torch.rand(sh, generator=g) * band)
    k3 = (torch.randint(0, 240, sh, generator=g).float() + 0.5) * torch.exp2(e - 8)  # e4m3 ties
    k3 = torch.where(lane == 0, 448.0 * torch.exp2(e), k3)  # power-of-two block max (ue8m0 ceil edge)
    k4 = torch.where(lane == 0, 448.0 * torch.exp2(e) * (1 + 2.0 ** -7), x)  # just above a power of two
    k5 = torch.where(lane == 3, torch.full(sh, 3.3895313892515355e38), torch.randn(sh, generator=g) * 2.0 ** 120)
    k6 = torch.where(lane % 2 == 0, -x, x)
    out = x
    for k, alt in enumerate((k0, k1, k2, k3, k4, k5, k6)):
        out = torch.where(kind == k, alt, out)
    out = out.reshape(M, K)
    out[:, ::97] = 0.0
    return out.to(torch.bfloat16).to(device)


def _inputs(M: int, K: int, device):
    from vllm.models.deepseek_v41.nvidia.dsv41_r3_dense import _dht_inputs

    return [_dht_inputs(M, K, 1009 + M, device), hostile_rows(M, K, 7717 + M, device),
            (torch.randn(M, K, generator=torch.Generator(device="cpu").manual_seed(31 + M)) * 0.05)
            .to(torch.bfloat16).to(device)]


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


# ------------------------------------------------------------------ per-shard state
class SkinnyV2:
    def __init__(self, shard, max_m: int):
        from vllm.models.deepseek_v41.nvidia.dsv41_r3_dense import skinny_ext

        e = ext()
        self.e, self.w, self.ws, self.K, self.n = e, shard.w.view(torch.uint8), shard.s_rm, shard.K, shard.n
        self.ok_a8: set[int] = set()
        self.ok_q: set[int] = set()
        self.ftz = 0
        ms = [m for m in range(1, max(1, max_m) + 1) if e.supports(m, self.n, self.K, GRID)
              and skinny_ext().fits(m, self.K)]
        if e.v1_split(self.n, self.K, 0) != 8:
            _warn_once("v2_split", "DSV41_SKINNY_V2: v1 split for N=%d K=%d is %d, not 8; kept live", self.n, self.K,
                       e.v1_split(self.n, self.K, 0))
            return
        if not CHECK:
            self.ok_a8 = set(ms)
            self.ok_q = set(ms) if QFUSE else set()
            return
        self._self_check(ms)

    # -------- checks (eager only)
    def _self_check(self, ms) -> None:
        from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize
        from vllm.models.deepseek_v41.nvidia.dsv41_r3_dense import skinny_ext, unswizzle_scale

        assert not (torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())
        dev = self.w.device
        v1 = skinny_ext()
        bad: dict[int, str] = {}
        if QFUSE:
            self.ftz = self._pick_ftz(dev, mxfp8_e4m3_quantize, unswizzle_scale)
        for M in ms:
            a8_ok, q_ok = True, QFUSE and self.ftz >= 0
            for j, x in enumerate(_inputs(M, self.K, dev)):
                xq, xs = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
                xq8 = xq.contiguous().view(torch.uint8)
                ref = v1.skinny_mx_a8(xq8, xs.view(torch.uint8), self.w, self.ws)
                for pdl in (False, True):
                    if a8_ok and not _same(self.e.skinny_v2_a8(xq8, xs.view(torch.uint8), self.w, self.ws, GRID,
                                                               pdl), ref):
                        a8_ok, bad[M] = False, f"a8 input {j} pdl={pdl}"
                    if q_ok and not _same(self.e.skinny_v2_q(x.contiguous(), self.w, self.ws, GRID, pdl, self.ftz),
                                          ref):
                        q_ok, bad[M] = False, f"qfuse input {j} pdl={pdl}"
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            if a8_ok:
                self.ok_a8.add(M)
            if q_ok:
                self.ok_q.add(M)
        if bad:
            _warn_once("v2_bad", "DSV41_SKINNY_V2: M kept live (or QFUSE off) after a bitwise self-check mismatch: %s",
                       bad)

    def _pick_ftz(self, dev, quant, unswizzle) -> int:
        """Return the amax*(1/448) FTZ mode (0/1) whose fused quant matches the live FlashInfer quant on
        hostile rows byte for byte (codes and scales), or -1 (QFUSE off)."""
        x = hostile_rows(256, self.K, 4242, dev)
        xq, xs = quant(x, is_sf_swizzled_layout=True)
        ref_q = xq.view(torch.uint8)
        ref_s = unswizzle(xs.view(torch.uint8), x.shape[0], self.K)
        modes = []
        for ftz in (0, 1):
            q, s, _ = self.e.quant_debug(x, ftz)
            if torch.equal(q, ref_q) and torch.equal(s, ref_s):
                modes.append(ftz)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        if not modes:
            _warn_once("v2_q", "DSV41_SKINNY_QFUSE: fused quant != live MXFP8 quant in both FTZ modes; QFUSE kept off")
            return -1
        return modes[0]

    # -------- serving entry points (graph-safe: launches + one torch.empty, like v1)
    def gemm_a8(self, xq: torch.Tensor, xs: torch.Tensor) -> torch.Tensor | None:
        M = xq.shape[0]
        if M not in self.ok_a8:
            return None
        return self.e.skinny_v2_a8(xq.contiguous().view(torch.uint8), xs.view(torch.uint8), self.w, self.ws,
                                   GRID, PDL)

    def fused(self, x) -> torch.Tensor | None:
        if not self.ok_q or not isinstance(x, torch.Tensor) or x.dtype != torch.bfloat16 or x.shape[-1] != self.K:
            return None
        x2 = x.reshape(-1, self.K)
        if x2.shape[0] not in self.ok_q or not x2.is_contiguous() or x2.data_ptr() % 16:
            return None
        return self.e.skinny_v2_q(x2, self.w, self.ws, GRID, PDL, self.ftz)


def build(shard) -> SkinnyV2 | None:
    """Called from QkvAShard.__init__ (first eager call per layer). None keeps the live path."""
    from vllm.models.deepseek_v41.nvidia.dsv41_r3_dense import QKV_MAX_M

    try:
        st = SkinnyV2(shard, QKV_MAX_M)
    except Exception as e:  # noqa: BLE001 - optional feature; the live path stays
        _warn_once("v2_build", "DSV41_SKINNY_V2 disabled (kept live): %r", e)
        return None
    if not st.ok_a8 and not st.ok_q:
        return None
    _warn_once("v2_on", "DSV41_SKINNY_V2=%s active: N=%d K=%d, a8 M=%s, qfuse M=%s (ftz=%d), grid=%d, check=%s",
               MODE, st.n, st.K, sorted(st.ok_a8), sorted(st.ok_q), st.ftz, GRID, CHECK)
    return st
