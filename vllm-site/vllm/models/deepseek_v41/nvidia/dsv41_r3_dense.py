# SPDX-License-Identifier: Apache-2.0
"""DSv4.1 TP8 round-3 dense-GEMM levers (sm_121 / GB10). All flags default OFF.

Install path: vllm/models/deepseek_v41/nvidia/dsv41_r3_dense.py
Callers: attention.py (_fused_wqa_wkv_gemm, indexer weights_proj) and
logits_processor.py (_apply_head); dspark.py only tags its LogitsProcessor.

Flags (every rank must see the same values; launch through launch-conc.sh):

  DSV41_QKV_A_SHARD=1           Column-shard the replicated fused wq_a|wkv projection
                                (N 1792 x K 5120 MXFP8, 9.46 MB/layer on every rank)
                                into N/tp = 224 rows per rank, then all-gather the
                                [M, 224] bf16 output (rdma_ar all_gather, dim=-1) back
                                to the exact fused [M, 1792] layout.
  DSV41_QKV_A_SHARD_MAX_M=8     Largest token count routed to the shard path; above it
                                the stock replicated GEMM runs (the replicated weight is
                                kept, +1.15 MB/layer for the shard copy).
  DSV41_QKV_A_SHARD_GEMM=skinny skinny  : split-K persistent GEMV (r2 skinny_mx_ext.so, A8,
                                          same MXFP8 operands; fp32 summation order differs)
                                cutlass : FlashInfer CUTLASS mm_mxfp8 on the 224-row shard
                                          (bitwise == replicated output; only 2 CTAs, so
                                          this is the plumbing/exactness arm, not the fast one)
  DSV41_IDX_WPROJ=mm32|bmm<S>   Indexer weights_proj (32 x 5120 bf16, 8 layers). Stock
                                F.linear picks a 2-CTA, 1-warp cuBLAS kernel (82-102 us,
                                3.2 GB/s) that becomes the join-critical path once qkv_a
                                is sharded. mm32 = torch.mm(out_dtype=fp32) (cuBLAS split-K
                                heuristics), bmm16 = explicit 16-way split-K bmm + fp32 sum.
  DSV41_FP8_LMHEAD=draft|target|both
                                MXFP8 copy of the (shared) vocab-parallel lm_head shard
                                (16160 x 5120: 165 MB bf16 -> 85 MB fp8 + scales).
                                draft  = DSpark base logits only (lossless: target verifies)
                                target = target logits (LOSSY, needs quality gate)
  DSV41_FP8_LMHEAD_ACT=a8|a16   a8  = MXFP8 activations + FlashInfer CUTLASS (any M)
                                a16 = bf16 activations + skinny_mx_a16 (M <= 8, else a8)
  DSV41_R3_SKINNY_LIB=dir       skinny_mx_ext*.so location
                                (default /home/jeweled/dsv41-r2/skinny_mx/build)
  DSV41_DRAFT_HEAD_TACTIC=0|1|audit   (g3 dhtac 2026-10-26, default 0 = unchanged)
                                DSpark draft head only (Fp8Head, act a8): call FlashInfer's CUTLASS SM120 MXFP8
                                runner with a pinned tactic for the row counts in DSV41_DRAFT_HEAD_TACTIC_MAP
                                instead of the autotune-cache choice. The cache picks tactic 0 (128x32 tile,
                                no swap: 505 CTAs whose 128-row A tiles are mostly TMA zero fill) for M 17-128,
                                i.e. every DSpark step with 4..25 requests (M = 5 x requests): 1104 us at M=20
                                against 423-433 us for tactic 1 (128x32, swap_ab) at M=5/10 (C1/C2/C4 profile).
                                Exactness: the six SM120 MXFP8 tactics differ only in CTA tile and operand swap;
                                there is no split-K/stream-K (ClusterShape 1x1x1), so every output element is the
                                same MMA chain in the same K order. Enforced, not assumed: the first eager build
                                runs every mapped M through ALL six tactics and the stock call on synthetic
                                MXFP8 activations and pins an M only if all seven outputs are bitwise equal.
                                1     = serve the pinned tactic for self-checked M
                                audit = serve the stock (autotuned) output, also run the pinned tactic and count
                                        mismatching bf16 words on the device; report by touching
                                        ~/dsv41-logs/dhtac-audit.trigger (-> dhtac-audit.<host>.<pid>.json)
  DSV41_DRAFT_HEAD_TACTIC_MAP=17-32:1   lo-hi:tactic,... (M = head rows); tactics 0..5 = 128x{32,64,128} x
                                {no swap, swap_ab}; pick per M from the stopped-window bench (tests/test_dhtac_gpu.py)
  DSV41_DRAFT_HEAD_TACTIC_CHECK=1 0 skips the build-time self-check (unit tests only, never in serving)

Everything is built lazily on the first eager (non-capturing) call, which happens in
vLLM's profile / dummy runs before CUDA-graph capture. If a build would be needed while
capturing, the stock path is used for that graph and a warning is logged.
"""
from __future__ import annotations

import glob
import importlib.util
import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

MXFP8_BLOCK = 32


def _flag(name: str) -> bool:
    return os.environ.get(name, "") == "1"


QKV_SHARD = _flag("DSV41_QKV_A_SHARD")
QKV_MAX_M = int(os.environ.get("DSV41_QKV_A_SHARD_MAX_M", "8"))
QKV_GEMM = os.environ.get("DSV41_QKV_A_SHARD_GEMM", "skinny")
IDX_WPROJ = os.environ.get("DSV41_IDX_WPROJ", "")
FP8_HEAD = os.environ.get("DSV41_FP8_LMHEAD", "")
FP8_HEAD_ACT = os.environ.get("DSV41_FP8_LMHEAD_ACT", "a8")
_SKINNY_DIR = os.environ.get("DSV41_R3_SKINNY_LIB", "/home/jeweled/dsv41-r2/skinny_mx/build")

if QKV_GEMM not in ("skinny", "cutlass"):
    raise ValueError(f"DSV41_QKV_A_SHARD_GEMM must be skinny|cutlass, got {QKV_GEMM!r}")
if FP8_HEAD not in ("", "draft", "target", "both"):
    raise ValueError(f"DSV41_FP8_LMHEAD must be draft|target|both, got {FP8_HEAD!r}")
if FP8_HEAD_ACT not in ("a8", "a16"):
    raise ValueError(f"DSV41_FP8_LMHEAD_ACT must be a8|a16, got {FP8_HEAD_ACT!r}")
if IDX_WPROJ and not (IDX_WPROJ == "mm32" or (IDX_WPROJ.startswith("bmm") and IDX_WPROJ[3:].isdigit())):
    raise ValueError(f"DSV41_IDX_WPROJ must be mm32|bmm<S>, got {IDX_WPROJ!r}")

# g3 dhtac (DSV41_DRAFT_HEAD_TACTIC, default off): see the module docstring.
DHT_MODE = os.environ.get("DSV41_DRAFT_HEAD_TACTIC", "0").strip().lower()
if DHT_MODE in ("", "off"):
    DHT_MODE = "0"
if DHT_MODE not in ("0", "1", "audit"):
    raise ValueError(f"DSV41_DRAFT_HEAD_TACTIC must be 0|1|audit, got {DHT_MODE!r}")
DHT_MAP_RAW = os.environ.get("DSV41_DRAFT_HEAD_TACTIC_MAP", "17-32:1")
DHT_CHECK = os.environ.get("DSV41_DRAFT_HEAD_TACTIC_CHECK", "1") != "0"
DHT_MAX_M = 1024


def parse_tactic_map(raw: str, max_m: int = DHT_MAX_M) -> dict[int, int]:
    """'17-32:1,33-64:3' -> {M: tactic}. Ranges may not overlap."""
    out: dict[int, int] = {}
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        rng, sep, tac = part.partition(":")
        if not sep:
            raise ValueError(f"DSV41_DRAFT_HEAD_TACTIC_MAP entry {part!r} needs lo-hi:tactic")
        lo_s, _, hi_s = rng.partition("-")
        lo, hi, t = int(lo_s), int(hi_s or lo_s), int(tac)
        if not (1 <= lo <= hi <= max_m) or not (0 <= t < 16):
            raise ValueError(f"DSV41_DRAFT_HEAD_TACTIC_MAP entry {part!r} out of range")
        for m in range(lo, hi + 1):
            if m in out:
                raise ValueError(f"DSV41_DRAFT_HEAD_TACTIC_MAP: M={m} listed twice")
            out[m] = t
    return out


DHT_MAP = parse_tactic_map(DHT_MAP_RAW) if DHT_MODE != "0" else {}

_warned: set[str] = set()


def _warn_once(key: str, msg: str, *args) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning(msg, *args)


def _capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


# ------------------------------------------------------------------ skinny_mx extension
_ext = None


def skinny_ext():
    global _ext
    if _ext is None:
        cands = sorted(glob.glob(os.path.join(_SKINNY_DIR, "skinny_mx_ext*.so")))
        if not cands:
            raise RuntimeError(f"DSV41 r3: no skinny_mx_ext*.so in {_SKINNY_DIR}")
        spec = importlib.util.spec_from_file_location("skinny_mx_ext", cands[0])
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        _ext = mod
        logger.info("DSV41 r3: skinny_mx loaded from %s", cands[0])
    return _ext


# ------------------------------------------------------------------ MXFP8 scale layouts
def swizzle_scale(sf2d: torch.Tensor, M: int, K: int) -> torch.Tensor:
    """Row-major [M, K/32] e8m0 -> FlashInfer F8_128x4 1-D layout (same as
    vllm mxfp8_utils.swizzle_mxfp8_scale; duplicated so the CPU test needs no vllm)."""
    nm = (M + 127) // 128
    nk = (K // MXFP8_BLOCK + 3) // 4
    pad = torch.zeros((nm * 128, nk * 4), dtype=sf2d.dtype, device=sf2d.device)
    pad[:M, : K // MXFP8_BLOCK] = sf2d
    return pad.view(nm, 4, 32, nk, 4).transpose(1, 3).contiguous().view(-1)


def unswizzle_scale(sf1d: torch.Tensor, M: int, K: int) -> torch.Tensor:
    """Inverse of swizzle_scale: F8_128x4 1-D -> row-major [M, K/32]."""
    nm = (M + 127) // 128
    nk = (K // MXFP8_BLOCK + 3) // 4
    assert sf1d.numel() == nm * 128 * nk * 4, (sf1d.numel(), M, K)
    t = sf1d.reshape(nm, nk, 32, 4, 4).transpose(1, 3).contiguous().view(nm * 128, nk * 4)
    return t[:M, : K // MXFP8_BLOCK].contiguous()


def quantize_rows_mxfp8(w: torch.Tensor, chunk: int = 2048) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 [N, K] -> (fp8 e4m3 [N, K], e8m0 uint8 [N, K/32]); same math as vllm's
    _mxfp8_e4m3_quantize_torch (ceil(log2(amax/448)) + 127, clamp 0..254). Chunked to
    bound the fp32 transient (one-time, at build)."""
    N, K = w.shape
    assert K % MXFP8_BLOCK == 0
    q = torch.empty((N, K), dtype=torch.float8_e4m3fn, device=w.device)
    s = torch.empty((N, K // MXFP8_BLOCK), dtype=torch.uint8, device=w.device)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    for i in range(0, N, chunk):
        x = w[i : i + chunk].to(torch.float32).view(-1, K // MXFP8_BLOCK, MXFP8_BLOCK)
        amax = x.abs().amax(dim=-1).clamp(min=torch.finfo(torch.float32).tiny)
        e = (torch.ceil(torch.log2(amax / fp8_max)) + 127.0).clamp(0, 254)
        q[i : i + chunk] = (x / torch.exp2(e - 127.0).unsqueeze(-1)).view(-1, K).to(torch.float8_e4m3fn)
        s[i : i + chunk] = e.to(torch.uint8)
    return q, s


def _rowmajor_scale(linear: torch.nn.Module) -> torch.Tensor | None:
    """Row-major [N, K/32] uint8 view of a processed MXFP8 linear's weight scales, or None
    if the layout is not recognised (then the feature stays off for that layer)."""
    w = linear.weight
    ws = getattr(linear, "weight_scale", None)
    if ws is None or w.dtype != torch.float8_e4m3fn or w.dim() != 2 or ws.element_size() != 1:
        return None
    N, K = w.shape
    nk = (K // MXFP8_BLOCK + 3) // 4
    if ws.dim() == 1 and ws.numel() == ((N + 127) // 128) * 128 * nk * 4:
        return unswizzle_scale(ws.data.view(torch.uint8), N, K)  # FlashInfer CUTLASS layout
    if ws.dim() == 2 and ws.shape[0] >= N and ws.shape[1] >= K // MXFP8_BLOCK:
        return ws.data[:N, : K // MXFP8_BLOCK].contiguous().view(torch.uint8)
    return None


def _quantize_act(x: torch.Tensor, K: int):
    from vllm.model_executor.layers.fusion.quant_activation import as_quantized_activation
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize
    from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic

    qa = as_quantized_activation(x, kMxfp8Dynamic)
    if qa is not None:
        return qa.data, qa.scale, tuple(qa.orig_shape)
    xq, xs = mxfp8_e4m3_quantize(x.reshape(-1, K), is_sf_swizzled_layout=True)
    return xq, xs, tuple(x.shape)


def _num_tokens(x) -> int:
    shape = getattr(x, "orig_shape", None) or x.shape
    n = 1
    for d in shape[:-1]:
        n *= int(d)
    return n


# ------------------------------------------------------------------ (2) qkv_a column shard
class QkvAShard:
    def __init__(self, linear: torch.nn.Module, rank: int, tp: int):
        w = linear.weight.data
        N, K = w.shape
        s2d = _rowmajor_scale(linear)
        if s2d is None:
            raise RuntimeError("unrecognised weight/scale layout")
        if N % tp or (N // tp) % 32 or N // tp < 128:
            raise RuntimeError(f"N={N} not shardable by tp={tp} into 32-aligned >=128-row slices")
        n = N // tp
        rows = slice(rank * n, (rank + 1) * n)
        self.N, self.n, self.K, self.tp = N, n, K, tp
        self.w = w[rows].contiguous()  # [n, K] fp8
        self.s_rm = s2d[rows].contiguous()  # [n, K/32] uint8 row-major (skinny)
        self.s_swz = swizzle_scale(self.s_rm, n, K)  # 1-D F8_128x4 (cutlass)
        if QKV_GEMM == "skinny":
            skinny_ext()

    def gemm(self, xq: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
        M = xq.shape[0]
        if QKV_GEMM == "skinny" and skinny_ext().fits(M, self.K):
            return skinny_ext().skinny_mx_a8(
                xq.contiguous().view(torch.uint8), xs.view(torch.uint8),
                self.w.view(torch.uint8), self.s_rm,
            )
        from vllm.utils import flashinfer as vllm_flashinfer

        return vllm_flashinfer.mm_mxfp8(
            xq, self.w.t(), xs, self.s_swz, out_dtype=torch.bfloat16, backend="cutlass"
        )

    def __call__(self, x) -> torch.Tensor:
        from vllm.distributed import tensor_model_parallel_all_gather

        xq, xs, shape = _quantize_act(x, self.K)
        y = self.gemm(xq, xs)  # [M, n] bf16: this rank's contiguous output rows
        # all_gather(dim=-1) == concat of rank slices == the fused [M, N] layout.
        out = tensor_model_parallel_all_gather(y, dim=-1)
        return out.view(*shape[:-1], self.N)


def qkv_a_sharded(attn: torch.nn.Module, x):
    """Return the fused [M, 1792] q_lora|kv projection via the shard path, or None to
    let the caller run the stock replicated GEMM."""
    st = getattr(attn, "_dsv41_qkv_shard", None)
    if st is None:
        if _capturing():
            _warn_once("qkv_cap", "DSV41_QKV_A_SHARD: first call during capture; stock path for %s",
                       getattr(attn, "prefix", "?"))
            return None
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        tp = get_tensor_model_parallel_world_size()
        try:
            st = QkvAShard(attn.fused_wqa_wkv, get_tensor_model_parallel_rank(), tp) if tp > 1 else False
        except Exception as e:  # noqa: BLE001 - feature is optional
            _warn_once("qkv_build", "DSV41_QKV_A_SHARD disabled: %s", e)
            st = False
        attn._dsv41_qkv_shard = st
        if st:
            _warn_once("qkv_on", "DSV41_QKV_A_SHARD active: %d -> %d rows/rank, gemm=%s, max_m=%d",
                       st.N, st.n, QKV_GEMM, QKV_MAX_M)
    if not st:
        return None
    M = _num_tokens(x)
    if M == 0 or M > QKV_MAX_M:
        return None
    return st(x)


# ------------------------------------------------------------------ (5) indexer weights_proj
def indexer_weights_proj(linear: torch.nn.Module, x: torch.Tensor) -> torch.Tensor | None:
    """Faster replacement for ReplicatedLinear(5120 -> 32, bf16) at decode sizes."""
    if not IDX_WPROJ or not isinstance(x, torch.Tensor) or x.dtype != torch.bfloat16:
        return None
    w = linear.weight
    K = w.shape[1]
    x2 = x.reshape(-1, K)
    if IDX_WPROJ == "mm32":
        y = torch.mm(x2, w.t(), out_dtype=torch.float32)
    else:
        S = int(IDX_WPROJ[3:])
        if K % S:
            return None
        Nw = w.shape[0]
        ws = getattr(linear, "_dsv41_wproj_split", None)
        if ws is None or ws.shape[0] != S:
            if _capturing():
                return None
            ws = w.view(Nw, S, K // S).permute(1, 2, 0).contiguous()  # [S, K/S, Nw]
            linear._dsv41_wproj_split = ws
        xs = x2.view(-1, S, K // S).transpose(0, 1)  # [S, M, K/S] (strided view)
        y = torch.bmm(xs, ws, out_dtype=torch.float32).sum(0)
    return y.to(x.dtype).view(*x.shape[:-1], w.shape[0])


# ------------------------------------------------------------------ (3) FP8 lm-heads
def _dht_inputs(M: int, K: int, seed: int, device) -> torch.Tensor:
    """Synthetic head inputs for the self-check: post-RMSNorm-like rows with per-32-block magnitudes spread over
    ~2^-6..2^6 (so the MXFP8 block scales vary), a few exact zeros and one large outlier per row."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(M, K, generator=g)
    blk = torch.exp2(torch.randint(-6, 7, (M, K // MXFP8_BLOCK), generator=g).float())
    x = x * blk.repeat_interleave(MXFP8_BLOCK, dim=1)
    x[:, ::97] = 0.0
    x[torch.arange(M), torch.randint(0, K, (M,), generator=g)] = 300.0
    return x.to(torch.bfloat16).to(device)


class _HeadTactic:
    """g3 dhtac: pinned CUTLASS SM120 MXFP8 tactic for the DSpark draft head (DSV41_DRAFT_HEAD_TACTIC)."""

    def __init__(self, head: "Fp8Head"):
        from flashinfer.gemm.gemm_base import DEFAULT_WORKSPACE_SIZE, get_cutlass_mxfp8_gemm_module
        from flashinfer.utils import _get_cache_buf

        dev = head.w.device
        major, _ = torch.cuda.get_device_capability(dev)
        self.head = head
        self.runner = get_cutlass_mxfp8_gemm_module(major).cutlass_mxfp8_gemm_runner()
        self.tactics = list(self.runner.get_valid_tactics(None, None))
        self.ws = _get_cache_buf("mm_mxfp8_workspace", DEFAULT_WORKSPACE_SIZE, dev)
        self.ok: dict[int, int] = {}
        self.rejected: dict[int, str] = {}
        bad = sorted({t for t in DHT_MAP.values() if t not in self.tactics})
        if bad:
            raise ValueError(f"DSV41_DRAFT_HEAD_TACTIC_MAP tactics {bad} not in valid tactics {self.tactics}")
        if DHT_CHECK:
            self._self_check(dev)
        else:
            self.ok = dict(DHT_MAP)
        self.counters = None
        if DHT_MODE == "audit":
            self.counters = torch.zeros(4, dtype=torch.int64, device=dev)
            self._host = torch.zeros(4, dtype=torch.int64, pin_memory=True)
            self._side = torch.cuda.Stream(dev)
            self._device = dev
            self._start_reporter()
        pinned = sorted(self.ok)
        logger.warning(
            "DSV41_DRAFT_HEAD_TACTIC=%s: %d of %d mapped row counts pinned (%s)%s; tactics %s; self-check %s",
            DHT_MODE, len(pinned), len(DHT_MAP), _fmt_ranges(self.ok),
            f"; kept stock: {sorted(self.rejected)}" if self.rejected else "", self.tactics,
            "all tactics + stock bitwise on 2 synthetic inputs per M" if DHT_CHECK else "SKIPPED",
        )

    def run(self, xq: torch.Tensor, xs: torch.Tensor, tactic: int) -> torch.Tensor:
        """Exactly FlashInfer mm_mxfp8(backend='cutlass') minus the autotuner lookup."""
        h = self.head
        out = torch.empty((xq.shape[0], h.N), dtype=torch.bfloat16, device=xq.device)
        self.runner(inputs=[xq, h.w.t(), xs, h.s_swz, torch.bfloat16, out, self.ws], tactic=tactic)
        return out

    def _self_check(self, dev) -> None:
        from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize
        from vllm.utils import flashinfer as vllm_flashinfer

        h = self.head
        assert not _capturing()
        for M, t in sorted(DHT_MAP.items()):
            why = ""
            for seed in (M, 7919 + M):
                x = _dht_inputs(M, h.K, seed, dev)
                xq, xs = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
                ref = vllm_flashinfer.mm_mxfp8(xq, h.w.t(), xs, h.s_swz, out_dtype=torch.bfloat16, backend="cutlass")
                ref16 = ref.view(torch.int16)
                for tt in self.tactics:
                    try:
                        same = torch.equal(self.run(xq, xs, tt).view(torch.int16), ref16)
                    except Exception as e:  # noqa: BLE001 - an unusable tactic disqualifies this M
                        same, why = False, f"tactic {tt} raised {e!r}"[:200]
                    if not same:
                        why = why or f"tactic {tt} != stock (seed {seed})"
                        break
                if why:
                    break
            torch.cuda.synchronize(dev)
            if why:
                self.rejected[M] = why
                logger.warning("DSV41_DRAFT_HEAD_TACTIC: M=%d kept on the stock path: %s", M, why)
            else:
                self.ok[M] = t

    def audit(self, served: torch.Tensor, shadow: torch.Tensor) -> None:
        c = self.counters
        c[0] += 1
        c[1] += served.shape[0]
        c[2] += (served.view(torch.int16) != shadow.view(torch.int16)).sum()

    def _start_reporter(self) -> None:
        import json
        import socket
        import threading
        import time
        from pathlib import Path

        root = Path(os.environ.get("DSV41_DRAFT_HEAD_TACTIC_AUDIT_DIR", str(Path.home() / "dsv41-logs")))
        delay = float(os.environ.get("DSV41_DRAFT_HEAD_TACTIC_AUDIT_DELAY_S", "600"))
        trigger = root / "dhtac-audit.trigger"
        out = root / f"dhtac-audit.{socket.gethostname()}.{os.getpid()}.json"
        born = time.time()

        def loop():
            seen = None
            while True:
                time.sleep(2.0)
                try:
                    mtime = trigger.stat().st_mtime
                except OSError:
                    continue
                if mtime == seen or time.time() - born < delay:  # never touch CUDA during startup/capture
                    continue
                seen = mtime
                try:
                    with torch.cuda.device(self._device), torch.cuda.stream(self._side):
                        self._host.copy_(self.counters, non_blocking=True)
                        self._side.synchronize()
                    calls, rows, bad, _ = self._host.tolist()
                    rep = dict(host=socket.gethostname(), pid=os.getpid(), time=time.time(), mode="audit",
                               calls=calls, rows=rows, mismatching_words=bad, status="MISMATCH" if bad else "OK",
                               pinned=_fmt_ranges(self.ok))
                    tmp = out.with_suffix(".tmp")
                    tmp.write_text(json.dumps(rep, indent=1))
                    tmp.replace(out)
                except Exception as exc:  # diagnostics must never kill serving
                    try:
                        out.with_suffix(".err").write_text(repr(exc))
                    except OSError:
                        pass

        threading.Thread(target=loop, name="dhtac-audit-report", daemon=True).start()


def _fmt_ranges(d: dict[int, int]) -> str:
    out, run = [], None
    for m in sorted(d):
        if run and m == run[1] + 1 and d[m] == run[2]:
            run[1] = m
        else:
            if run:
                out.append(run)
            run = [m, m, d[m]]
    if run:
        out.append(run)
    return ",".join(f"{a}-{b}:{t}" if a != b else f"{a}:{t}" for a, b, t in out) or "-"


class Fp8Head:
    def __init__(self, w: torch.Tensor):
        self.N, self.K = w.shape
        self.w, self.s_rm = quantize_rows_mxfp8(w)
        self.s_swz = swizzle_scale(self.s_rm, self.N, self.K)
        # g3 dhtac: only with DSV41_DRAFT_HEAD_TACTIC=1|audit (FP8_HEAD=draft builds this head for DSpark only).
        self._dht = (_HeadTactic(self) if DHT_MODE != "0" and FP8_HEAD == "draft" and FP8_HEAD_ACT == "a8"
                     and DHT_MAP else None)

    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        h2 = h.reshape(-1, self.K)
        M = h2.shape[0]
        if FP8_HEAD_ACT == "a16" and skinny_ext().fits(M, self.K):
            y = skinny_ext().skinny_mx_a16(h2.contiguous(), self.w.view(torch.uint8), self.s_rm)
        else:
            from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
                mxfp8_e4m3_quantize,
            )
            from vllm.utils import flashinfer as vllm_flashinfer

            xq, xs = mxfp8_e4m3_quantize(h2, is_sf_swizzled_layout=True)
            dht = self._dht
            tac = dht.ok.get(M) if dht is not None else None
            if tac is not None and DHT_MODE == "1":
                y = dht.run(xq, xs, tac)
            else:
                y = vllm_flashinfer.mm_mxfp8(
                    xq, self.w.t(), xs, self.s_swz, out_dtype=torch.bfloat16, backend="cutlass"
                )
                if tac is not None:  # audit: stock output served, pinned tactic compared on the device
                    dht.audit(y, dht.run(xq, xs, tac))
        return y.view(*h.shape[:-1], self.N)


def fp8_head_apply(lp, lm_head, h, bias):
    """Hook for LogitsProcessor._apply_head. Returns None to keep the stock path."""
    if not FP8_HEAD or bias is not None or not isinstance(h, torch.Tensor):
        return None
    role = getattr(lp, "_dsv41_role", "target")
    if FP8_HEAD != "both" and FP8_HEAD != role:
        return None
    if lp.head_dtype is not None and lp.head_dtype != h.dtype:
        return None
    w = getattr(lm_head, "weight", None)
    if w is None or w.dim() != 2 or w.dtype != torch.bfloat16 or h.dtype != torch.bfloat16:
        return None
    N, K = w.shape
    # Vocab heads only (the Markov w2 shim has K=256); FlashInfer sm12x needs N%32.
    if K < 1024 or K % 32 or N % 32 or N < 128 or h.shape[-1] != K:
        return None
    st = getattr(lm_head, "_dsv41_fp8_head", None)
    if st is None:
        if _capturing():
            _warn_once("head_cap", "DSV41_FP8_LMHEAD: first call during capture; bf16 head kept")
            return None
        st = Fp8Head(w.data)
        lm_head._dsv41_fp8_head = st
        _warn_once("head_on", "DSV41_FP8_LMHEAD=%s act=%s active: lm_head shard %dx%d -> MXFP8",
                   FP8_HEAD, FP8_HEAD_ACT, N, K)
    return st(h)
