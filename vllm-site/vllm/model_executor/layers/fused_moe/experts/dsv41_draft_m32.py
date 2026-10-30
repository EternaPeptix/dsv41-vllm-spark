"""w6 drafter-m32 (default off): DSpark drafter routed MoE on the M32 target native.

The DSpark drafter (mtp.0-2: 128 routed experts, top-3, hidden 5120, per TP8 rank FC1 N=768 = padded
[gate 288+96 | up 288+96], FC2 K=384) runs today through the installed DeepGEMM with the default
128-row expert alignment (BLOCK_M=128 kernels). With this flag its decode-sized calls use 32-row expert
alignment and the M32 target native that already serves the target MoE (get_target_m32_kernel(),
DSV41_MOE_M32_G1 tree). For E=128 the m32-bw 'all' features are compile-time off (their gate is
kNumGroups == 384), so the drafter gets the plain M32 tiling: BLOCK_M 32 / BLOCK_N 64 / BLOCK_K 128,
7 stages, split-K 1 (grouped GEMMs never split K), from the SAME sm120_fp8_fp4_gemm_1d1d.cuh template
(byte-identical between the installed DeepGEMM and the M32 trees). Per output element both kernels run
the same chain: accum = 0; for every k32 chunk in K order:
  accum = mma.sync...kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.f32.e4m3.e2m1.f32.ue8m0(
          a[row, chunk], b[col, chunk], sfa[row, chunk/4], sfb[col, chunk], accum)
then bf16 = RN(accum). BLOCK_M / BLOCK_N / stages / epilogue sub-tile only decide which thread holds
which element and how bytes are staged, so every live output row is bitwise the 128-row kernel's.
The permute (row positions), the SiLU-mul-quant (per-row elementwise + exact group max) and the
gather (fp32 sum in fixed top-k order) are layout-independent per row.

  DSV41_DRAFT_MOE_M32=0 / unset  original behaviour (no new code path runs)
  DSV41_DRAFT_MOE_M32=1          drafter calls with 1 <= M <= DSV41_DRAFT_M32_MAX_M rows use the M32 path
  DSV41_DRAFT_MOE_M32=verify     serve the ORIGINAL 128-row output; also run the M32 path into private
                                 buffers and count bitwise (int16) mismatches per drafter layer on the
                                 device, eager and inside CUDA graphs (diagnostic only, ~2x drafter MoE).
                                 Report: touch $DSV41_DRAFT_M32_VERIFY_DIR/draft-m32-verify.trigger
                                 (default ~/dsv41-logs) once startup is over (not before
                                 DSV41_DRAFT_M32_VERIFY_DELAY_S, default 900 s, after worker start);
                                 each worker writes draft-m32-verify.<host>.<pid>.json.
  DSV41_DRAFT_M32_MAX_M          default 160 (= max-num-seqs 32 x DSpark block 5); larger calls
                                 (prefill chunks, mixed steps) stay on the 128-row path. Accepted 1..160 (the
                                 GPU-qualified M range); anything else is a hard startup error.
Requires DSV41_SM121_TARGET_M32=1 on SM121 (the shared M32 native); otherwise inert (one warning).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import threading

# (FC1 N, hidden K, top-k, local experts) of the DSpark drafter routed MoE on one TP8 rank (padded FC1).
DRAFT_GEOMETRY = (768, 5120, 3, 128)
DEFAULT_MAX_M = 160
_LOCK = threading.RLock()
_VERIFIER = None
_WARNED = False


def mode() -> str:
    raw = os.environ.get("DSV41_DRAFT_MOE_M32", "0").strip().lower()
    if raw in ("", "0", "off", "false"):
        return "0"
    if raw in ("1", "on", "true"):
        return "1"
    if raw == "verify":
        return "verify"
    raise ValueError(f"DSV41_DRAFT_MOE_M32={raw!r}: use 0, 1 or verify")


def max_m() -> int:
    value = int(os.environ.get("DSV41_DRAFT_M32_MAX_M", str(DEFAULT_MAX_M)))
    assert 1 <= value <= DEFAULT_MAX_M, f"DSV41_DRAFT_M32_MAX_M={value}: use 1..{DEFAULT_MAX_M} (qualified M range)"
    return value


def geometry(N: int, K: int, topk: int, experts: int) -> bool:
    return (N, K, topk, experts) == DRAFT_GEOMETRY


def aligned_rows(m: int, align: int, topk: int = 3, experts: int = 128) -> int:
    """M_sum of compute_aligned_M_and_alignment (no expert_tokens_meta): round_up(r + min(r, E)*(a-1), a)."""
    routes = m * topk
    rows = routes + min(routes, experts) * (align - 1)
    return -(-rows // align) * align


def config_eligible(moe_config) -> bool:
    """Drafter-geometry FusedMoEConfig on the qualified TP8 / SM121 / target-M32 deployment."""
    global _WARNED
    if mode() == "0":
        return False
    if moe_config.num_experts != DRAFT_GEOMETRY[3] or moe_config.experts_per_token != DRAFT_GEOMETRY[2]:
        return False
    from vllm.platforms import current_platform

    parallel = moe_config.moe_parallel_config
    ok = (
        os.environ.get("DSV41_SM121_TARGET_M32") == "1"
        and current_platform.is_device_capability(121)
        and moe_config.hidden_dim == DRAFT_GEOMETRY[1]
        and parallel.tp_size == 8
        and parallel.dp_size == 1
        and not parallel.use_ep
        and not parallel.enable_eplb
    )
    if not ok and not _WARNED:
        _WARNED = True
        from vllm.logger import init_logger

        init_logger(__name__).warning(
            "DSV41_DRAFT_MOE_M32=%s ignored: needs DSV41_SM121_TARGET_M32=1, SM121, hidden 5120, "
            "TP8, no DP/EP/EPLB (drafter MoE stays on the 128-row path)", mode())
    return ok


def announce(kind: str) -> None:
    with _LOCK:
        if getattr(announce, "done", False):
            return
        announce.done = True
    from vllm.logger import init_logger

    init_logger(__name__).info(
        "DSV41_DRAFT_MOE_M32=%s: DSpark drafter MoE (E128 top-3, N768/K5120 FC1, K384 FC2) %s for "
        "1 <= M <= %d rows (32-row alignment, M32 target native)", mode(), kind, max_m())


def get_verifier():
    global _VERIFIER
    with _LOCK:
        if _VERIFIER is None:
            _VERIFIER = DraftM32Verifier()
        return _VERIFIER


class DraftM32Verifier:
    """Shadow compare for DSV41_DRAFT_MOE_M32=verify (diagnostics only, never changes served bytes)."""

    POISON = -1  # int16 0xFFFF (bf16 NaN) in the shadow output before every shadow run
    MAX_LAYERS = 8  # drafter has 3 routed-MoE layers; counter rows are assigned in first-call order

    def __init__(self):
        import torch

        assert not torch.cuda.is_current_stream_capturing()
        self.device = torch.cuda.current_device()
        self.cap = max_m()
        # per drafter MoE weight: [calls, rows, mismatching int16 words, nonfinite served words]
        self.counters = torch.zeros((self.MAX_LAYERS, 4), dtype=torch.int64, device="cuda")
        self._host = torch.zeros((self.MAX_LAYERS, 4), dtype=torch.int64, pin_memory=True)
        self.index: dict[int, int] = {}
        self.buffers = None
        self.capture_misses = 0
        self.out_of_range = 0
        self._side = torch.cuda.Stream()
        self._start_reporter()

    def _alloc(self):
        import torch

        rows = aligned_rows(self.cap, 32)
        n, k = DRAFT_GEOMETRY[0], DRAFT_GEOMETRY[1]
        # workspace13: FP8 bytes viewed from bf16 storage (max(N/2, K) columns); workspace2: bf16 max(N, K)
        ws13 = torch.empty((rows * max(n // 2, k) + 1) // 2, dtype=torch.bfloat16, device="cuda")
        ws2 = torch.empty(rows * max(n, k), dtype=torch.bfloat16, device="cuda")
        out = torch.empty((self.cap, k), dtype=torch.bfloat16, device="cuda")
        self.buffers = (ws13, ws2, out)

    def shadow(self, expert, served_output, call: dict) -> None:
        """Re-run `expert.apply` on the same inputs with 32-row alignment into private buffers; compare."""
        import torch

        topk_ids, w1, hidden = call["topk_ids"], call["w1"], call["hidden_states"]
        m = topk_ids.size(0)
        if not geometry(w1.size(1), hidden.size(1), topk_ids.size(1), w1.size(0)):
            return
        if not (1 <= m <= self.cap):
            self.out_of_range += 1
            return
        if self.buffers is None:
            if torch.cuda.is_current_stream_capturing():
                self.capture_misses += 1  # never allocate inside a capture; that graph has no shadow
                return
            with _LOCK:
                if self.buffers is None:
                    self._alloc()
        idx = self.index.setdefault(w1.data_ptr(), len(self.index))
        assert idx < self.MAX_LAYERS, "DSV41_DRAFT_MOE_M32=verify: more drafter MoE weights than counter rows"
        ws13, ws2, out = self.buffers
        shadow = out[:m]
        shadow.view(torch.int16).fill_(self.POISON)
        expert._dsv41_force_m32 = True
        try:
            expert.apply(output=shadow, workspace13=ws13, workspace2=ws2, **call)
        finally:
            expert._dsv41_force_m32 = False
        self.compare(served_output, shadow, self.counters[idx])

    @staticmethod
    def compare(served, shadow, row) -> None:
        """row[0] += 1 call, row[1] += M, row[2] += mismatching int16 words, row[3] += non-finite served words."""
        import torch

        a, b = served.view(torch.int16), shadow.view(torch.int16)
        row[0].add_(1)
        row[1].add_(served.size(0))
        row[2].add_(torch.ne(a, b).sum())
        row[3].add_((~torch.isfinite(served)).sum())

    def _start_reporter(self) -> None:
        import socket
        import time

        root = Path(os.environ.get("DSV41_DRAFT_M32_VERIFY_DIR", str(Path.home() / "dsv41-logs")))
        delay = float(os.environ.get("DSV41_DRAFT_M32_VERIFY_DELAY_S", "900"))
        trigger = root / "draft-m32-verify.trigger"
        out = root / f"draft-m32-verify.{socket.gethostname()}.{os.getpid()}.json"
        born = time.time()

        def loop():
            import torch

            seen = None
            while True:
                time.sleep(2.0)
                try:
                    mtime = trigger.stat().st_mtime
                except OSError:
                    continue
                # Never touch CUDA before startup / graph capture is long over.
                if mtime == seen or time.time() - born < delay:
                    continue
                seen = mtime
                try:
                    with torch.cuda.device(self.device), torch.cuda.stream(self._side):
                        self._host.copy_(self.counters, non_blocking=True)
                        self._side.synchronize()
                    rows = self._host.tolist()
                    per = [dict(index=i, calls=r[0], rows=r[1], mismatch_words=r[2], nonfinite_served=r[3])
                           for i, r in enumerate(rows[: len(self.index)])]
                    bad = sum(p["mismatch_words"] for p in per)
                    report = dict(host=socket.gethostname(), pid=os.getpid(), time=time.time(), mode="verify",
                                  max_m=self.cap, weights_seen=len(self.index),
                                  weights_with_calls=sum(1 for p in per if p["calls"] > 0),
                                  capture_misses=self.capture_misses, out_of_range_calls=self.out_of_range,
                                  status="MISMATCH" if bad else "OK", per_weight=per)
                    tmp = out.with_suffix(".tmp")
                    tmp.write_text(json.dumps(report, indent=1))
                    tmp.replace(out)
                except Exception as exc:  # diagnostics must never kill serving
                    try:
                        out.with_suffix(".err").write_text(repr(exc))
                    except OSError:
                        pass

        threading.Thread(target=loop, name="draft-m32-verify-report", daemon=True).start()
