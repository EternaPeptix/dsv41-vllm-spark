# SPDX-License-Identifier: Apache-2.0
"""DSv4.1 TP8 round-4 "small kernels" bundle (sm_121 / GB10). Every flag defaults OFF.

Install path: vllm/models/deepseek_v41/nvidia/dsv41_r4_small.py
Only caller:  vllm/models/deepseek_v41/nvidia/model.py (compute_logits / compute_logits_local).
Everything else (the runner arming hook, the DSpark main_proj hook) is installed lazily from that
first call by wrapping class attributes, so no other site file is edited.

Flags (all ranks must see identical values; pass them through conc/launch-conc.sh):

  DSV41_R4_CERT_HEAD=0|1|check
      Certified (screened) target LM head. Idea from knapcio's DSV41_CERT_HEAD (re-implemented here;
      their code is AGPL, this file is not a copy). Per call, on this rank's vocab shard W [V, K]:
        xnorm   ||x_g|| per 128-column group of each hidden row (rounded up)
        screen  s = x . D^T with D an MXINT8 copy of W (int8 + one power-of-two scale per 32 weights,
                85 MB instead of 165 MB), and a proven per-logit bound
                  B = sum_g ||x_g|| * ||W_vg - D_vg||  +  2 * c_acc(K) * ||x|| * max(||W_v||, ||D_v||)
                lo = rn_bf16(s - B - slop), hi = rn_bf16(s + B + slop);  T[row] = max_v lo  (atomic)
        refine  one CTA per 16 vocab rows: if any row has hi >= T in that tile it computes the tile with
                the exact kernel (bf16 MMA, fp32 accumulate, ascending K, bf16 out), else writes -inf.
      Why the argmax is unchanged (ties included): the stock logit l_v = rn_bf16(cuBLAS fp32 sum) obeys
      lo_v <= l_v <= hi_v, so T <= max l on this rank; a cut token has l_v <= hi_v < T <= max l. Every
      token tied with the local max has hi >= T and is computed. The global argmax after the stock
      vocab all-gather is therefore the stock one, PROVIDED the exact kernel reproduces cuBLAS bits on
      the computed tiles. That is what DSV41_R4_CERT_HEAD_M qualifies: only row counts that
      gpu_test_r4.py proved bit-identical (every logit, every rank shard) are used; others stay stock.
      Masked logits are returned only when the step is "armed": every request greedy (temperature 0),
      no logprobs / logprob token ids, no logits processing of any kind (penalties, bias, bad words,
      min_tokens, thinking budget, custom processors, top-k/p are all folded into
      Sampler.needs_logits_processing), no grammar, no watermark, no synthetic acceptance, no NaN
      counting, and the step goes through the rejection sampler (num_draft_tokens > 0). Otherwise the
      stock head runs. "check" returns the stock logits, also runs the certified head, and counts on
      device: local-argmax mismatches, value-at-argmax mismatches, computed-logit bit mismatches,
      tiles computed (logged every DSV41_R4_CERT_HEAD_LOG_EVERY calls; that log costs one sync).
  DSV41_R4_CERT_HEAD_M=1-96           qualified row counts (ranges / comma list); empty = stock everywhere
  DSV41_R4_CERT_HEAD_TILE=64,128,4,3  screen BN,BK,num_warps,num_stages (BM = next pow2 >= M, >= 16)
  DSV41_R4_CERT_HEAD_EXACT=16,128,4,3 refine BV,BK,num_warps,num_stages (keep BK a multiple of 16)
  DSV41_R4_CERT_HEAD_LOG_EVERY=2000

  DSV41_R4_MAINPROJ_SPLIT=0|1
      DSpark main_proj (replicated MXFP8 ReplicatedLinear, N 5120 x K 15360, 78.6 MB on every rank)
      column-split: rank r computes output columns [r*640, (r+1)*640) with the same FlashInfer CUTLASS
      mm_mxfp8 on the same MXFP8-quantised activations, then tensor_model_parallel_all_gather(dim=-1)
      restores [M, 5120]. Self-checked: the first eager call at every row count M runs the stock layer
      too, compares bit for bit, and the ranks agree (MIN over the TP CPU group); a mismatch keeps that
      M on the stock path for good. Calls during CUDA-graph capture, M > MAX_M, or unchecked M -> stock.
  DSV41_R4_MAINPROJ_SPLIT_MAX_M=128

  DSV41_SYNC_FREE=argmax|audit (g1-stepovh part B; default off; needs DSV41_R4_CERT_HEAD=1)
      vLLM analogue of knapcio's DSV41_SPEC_SYNC_FREE. Our decode step has no rank-0 GPU broadcast (every
      rank samples from the all-gathered logits; draft noise / block-verify uniforms are counter-keyed),
      but on CERT-armed steps (above: every request greedy, nothing but an argmax reads the logits) the
      [M, V/tp] bf16 logits all-gather (RDMA, ~80-113 us at 6 rows, ~205 us at 24, ~1 ms at 48-64) only
      lets every rank compute the same argmax. "argmax" exchanges 16 B per row instead: each rank takes
      the leftmost max of its certified shard (NaN read as -inf), all-gathers (value, global id, 0, 0)
      fp32 (the sharded Markov drafter's RDMA-friendly row), every rank picks the first rank holding the
      max (ranks hold ascending vocab ranges, so this is the global leftmost max = the stock greedy
      token, ties included), and returns [M, org_vocab] bf16 logits that are -inf except 0 at that token.
      Every consumer on an armed step reads only the argmax (rejection-sampler target argmax, bonus
      argmax), so sampled tokens / counts are identical by construction (NaN rows aside: stock maps a NaN
      block max to -inf at block level; such a row means a broken forward either way). "audit" keeps the
      stock gather, runs the compact exchange too, and counts device-side mismatches against
      full.argmax(-1) (logged every DSV41_SYNC_FREE_AUDIT_EVERY calls, one sync each). Both modes add the
      same one collective on every rank and need the arming decision identical on all ranks (it is: it
      reads only replicated scheduler / sampling state, like the greedy-sharded drafter's graph choice);
      a one-time TP CPU-group agreement at CERT table build disables it everywhere if any rank lacks CERT.
  DSV41_SYNC_FREE_AUDIT_EVERY=2000

Shared-expert K padding (knapcio DSV41_SHARED_PAD_K), FP8 wo_a (DSV41_WO_A_W8), FP8 draft head
(DSV41_DRAFT_HEAD_FP8) and replicated wqkv_a / engram.wkv splits are NOT here: see the r4 handoff
(not applicable at TP8, or already prepared by r3 sonar-port / dense-fix).
"""
from __future__ import annotations

import os

import numpy as np
import torch

try:  # CPU unit tests import this file without vllm
    from vllm.logger import init_logger

    logger = init_logger(__name__)
except Exception:  # noqa: BLE001
    import logging

    logger = logging.getLogger("dsv41_r4_small")

_OFF = ("", "0", "off", "false", "no")


def _mode(name: str) -> str:
    v = os.environ.get(name, "0").strip().lower()
    if v in _OFF:
        return "0"
    return "check" if v == "check" else "1"


def parse_m(spec: str, max_m: int = 1024) -> frozenset:
    out: set[int] = set()
    for part in filter(None, (p.strip() for p in (spec or "").split(","))):
        a, _, b = part.partition("-")
        lo, hi = int(a), int(b or a)
        if not 1 <= lo <= hi <= max_m:
            raise ValueError(f"bad row-count entry {part!r} (1..{max_m})")
        out.update(range(lo, hi + 1))
    return frozenset(out)


CERT_MODE = _mode("DSV41_R4_CERT_HEAD")
CERT_M = parse_m(os.environ.get("DSV41_R4_CERT_HEAD_M", ""))
CERT_LOG_EVERY = int(os.environ.get("DSV41_R4_CERT_HEAD_LOG_EVERY", "2000") or 0)
SBN, SBK, SWARPS, SSTAGES = (int(v) for v in os.environ.get("DSV41_R4_CERT_HEAD_TILE", "64,128,4,3").split(","))
XBV, XBK, XWARPS, XSTAGES = (int(v) for v in os.environ.get("DSV41_R4_CERT_HEAD_EXACT", "16,128,4,3").split(","))
CERT_GROUP = 128
CERT_MAX_M = 128

MAINPROJ_SPLIT = _mode("DSV41_R4_MAINPROJ_SPLIT") != "0"
MAINPROJ_MAX_M = int(os.environ.get("DSV41_R4_MAINPROJ_SPLIT_MAX_M", "128"))


def parse_sync_free(raw: str) -> frozenset:
    """DSV41_SYNC_FREE -> parts. "1"/"on"/"all"/"true" = argmax; audit wins over argmax."""
    toks = {t.strip() for t in (raw or "").strip().lower().replace(";", ",").split(",") if t.strip()}
    toks -= set(_OFF)
    if toks & {"1", "on", "all", "true"}:
        toks = (toks - {"1", "on", "all", "true"}) | {"argmax"}
    bad = toks - {"argmax", "audit"}
    if bad:
        raise ValueError(f"DSV41_SYNC_FREE: unknown part(s) {sorted(bad)} (argmax, audit)")
    return frozenset(toks)


SF_PARTS = parse_sync_free(os.environ.get("DSV41_SYNC_FREE", ""))
SF_AUDIT = "audit" in SF_PARTS
SF_ARGMAX = "argmax" in SF_PARTS and not SF_AUDIT
SF_ANY = SF_AUDIT or SF_ARGMAX
SF_AUDIT_EVERY = int(os.environ.get("DSV41_SYNC_FREE_AUDIT_EVERY", "2000") or 2000)
SF_BLOCK = 4096

ANY = CERT_MODE != "0" or MAINPROJ_SPLIT

# Certificate constants. Inflations are generous: they cost nothing measurable in candidates.
INFL_X = 1.0 + 2.0 ** -14   # group / total norm of x (fp32 sum of <= 128 squares + sqrt)
INFL_Q = 1.0 + 2.0 ** -14   # fp32 sum_g ||x_g|| R_vg (40 non-negative terms)
INFL_B = 1.0 + 2.0 ** -20   # final product / add
SLOP_REL = 2.0 ** -21       # fp32 rounding of s -/+ B
SLOP_ABS = 2.0 ** -126
X_ABS = 2.0 ** -59          # underflow of x_i^2 in a group


def c_acc(K: int) -> float:
    """Assumed fp32 tensor-core accumulation error per unit ||x|| ||w|| (K/16 MMA steps, each modelled
    as 17 additions with one rounding of 2^-23; Fasi et al. 2021 style). gpu_test_r4.py measures the
    real error against float64 and reports the margin; it must stay far below this."""
    return (K / 16.0) * 17.0 * 2.0 ** -23


def next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def _capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


# =====================================================================================================
# Certified head: tables (pure torch, CPU-testable)
# =====================================================================================================
def round_up_f32(x64: torch.Tensor) -> torch.Tensor:
    """float64 (>= 0 or inf) -> smallest fp32 >= x."""
    f = x64.float()
    bump = f.double() < x64
    return torch.where(bump, torch.nextafter(f, torch.full_like(f, float("inf"))), f)


def make_mxint8(w: torch.Tensor, chunk: int = 2048) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 [V, K] -> (int8 [V, K], exponent byte uint8 [V, K/32]); D = q * 2^(e-127), |q| <= 127."""
    V, K = w.shape
    assert K % 32 == 0
    q = torch.empty((V, K), dtype=torch.int8, device=w.device)
    e = torch.empty((V, K // 32), dtype=torch.uint8, device=w.device)
    for r0 in range(0, V, chunk):
        r1 = min(V, r0 + chunk)
        wf = w[r0:r1].float().view(r1 - r0, K // 32, 32)
        amax = wf.abs().amax(-1)
        ex = torch.where(amax > 0, torch.ceil(torch.log2(amax.clamp_min(1e-38) / 127.0)), torch.zeros_like(amax))
        eb = (ex + 127.0).clamp(1.0, 254.0)
        qq = torch.round(wf / torch.exp2(eb - 127.0)[..., None]).clamp(-127, 127)
        q[r0:r1] = qq.view(r1 - r0, K).to(torch.int8)
        e[r0:r1] = eb.to(torch.uint8)
    return q, e


def deq_rows(q: torch.Tensor, e: torch.Tensor, r0: int, r1: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Rows r0:r1 of D as the screen kernel builds them: (bf16-cast values as fp32, raw fp32 values)."""
    qf = q[r0:r1].float()
    sc = (e[r0:r1].to(torch.int32) << 23).view(torch.float32).repeat_interleave(32, dim=1)
    raw = qf * sc
    return raw.to(torch.bfloat16).float(), raw


class CertState:
    pass


@torch.inference_mode()
def build_cert_tables(W: torch.Tensor, vreal: int | None = None, group: int = CERT_GROUP) -> CertState:
    """Tables for one rank's head shard W [V, K] bf16 (rows >= vreal are padding: never raise T)."""
    V, K = W.shape
    if W.dtype != torch.bfloat16 or K % group or group % 32 or K % SBK or K % XBK:
        raise ValueError(f"head shard {tuple(W.shape)} {W.dtype} does not fit group={group} BK={SBK},{XBK}")
    if not bool(torch.isfinite(W).all()):
        raise ValueError("head shard has non-finite weights")
    q, e = make_mxint8(W)
    NG = K // group
    rt = torch.empty((NG, V), dtype=torch.float32, device=W.device)
    nv = torch.empty((V,), dtype=torch.float32, device=W.device)
    n_unsafe = 0
    for r0 in range(0, V, 1024):
        r1 = min(V, r0 + 1024)
        d_bf, d_raw = deq_rows(q, e, r0, r1)
        unsafe = (d_bf != d_raw).any(dim=1)  # the kernel's bf16 cast of D must be exact
        w64 = W[r0:r1].double()
        res = (w64 - d_bf.double()).view(r1 - r0, NG, group)
        rg = round_up_f32(res.pow(2).sum(-1).sqrt() * (1.0 + 2.0 ** -40))
        rg[unsafe] = float("inf")
        rt[:, r0:r1] = rg.t()
        nrm = torch.maximum(w64.norm(dim=1), d_bf.double().norm(dim=1)) * (1.0 + 2.0 ** -40)
        nv[r0:r1] = round_up_f32(nrm)
        n_unsafe += int(unsafe.sum())
    st = CertState()
    st.W, st.q, st.e, st.rt, st.nv = W, q, e, rt, nv
    st.V, st.K, st.group, st.n_unsafe = V, K, group, n_unsafe
    st.vreal = V if vreal is None else max(0, min(V, int(vreal)))
    d0, _ = deq_rows(q, e, 0, min(V, 4096))
    w0 = W[: d0.shape[0]].float()
    st.eps = float(((w0 - d0).norm(dim=1) / w0.norm(dim=1).clamp_min(1e-30)).mean())
    st.need = torch.zeros((1,), dtype=torch.int32, device=W.device)
    st.counters = torch.zeros((8,), dtype=torch.int64, device=W.device)
    st.calls = 0
    st.armed_calls = 0
    return st


def reference_bounds(st: CertState, x: torch.Tensor, screen_acc: torch.Tensor | None = None):
    """Torch mirror of xnorm + screen: (lo, hi as fp32 holding bf16 values [M, V], T [M], bad [M])."""
    M, K = x.shape
    G, NG = st.group, K // st.group
    xf = x.float()
    xg = xf.view(M, NG, G).pow(2).sum(-1).sqrt() * INFL_X + X_ABS
    xn = xg.pow(2).sum(-1).sqrt() * INFL_X
    bad = ~torch.isfinite(xn)
    if screen_acc is None:
        d_bf, _ = deq_rows(st.q, st.e, 0, st.V)
        screen_acc = xf @ d_bf.t()
    qb = xg @ st.rt
    B = (qb * INFL_Q + 2.0 * c_acc(K) * (xn[:, None] * st.nv[None, :])) * INFL_B
    B = torch.where(torch.isnan(B), torch.full_like(B, float("inf")), B)
    slop = (screen_acc.abs() + B) * SLOP_REL + SLOP_ABS
    lo = (screen_acc - B - slop).to(torch.bfloat16).float()
    hi = (screen_acc + B + slop).to(torch.bfloat16).float()
    nonfin = ~torch.isfinite(screen_acc)
    lo = torch.where(nonfin | torch.isnan(lo), torch.full_like(lo, float("-inf")), lo)
    hi = torch.where(nonfin | torch.isnan(hi), torch.full_like(hi, float("inf")), hi)
    if st.vreal < st.V:
        lo[:, st.vreal:] = float("-inf")
        hi[:, st.vreal:] = float("-inf")
    T = lo.max(dim=1).values
    return lo, hi, T, bad


def reference_needed_tiles(hi: torch.Tensor, T: torch.Tensor, bad: torch.Tensor, bv: int = XBV) -> torch.Tensor:
    M, V = hi.shape
    nt = (V + bv - 1) // bv
    if bool(bad.any()):
        return torch.ones(nt, dtype=torch.bool)
    cand = (hi >= T[:, None]).any(dim=0)
    pad = nt * bv - V
    if pad:
        cand = torch.cat([cand, torch.zeros(pad, dtype=torch.bool, device=cand.device)])
    return cand.view(nt, bv).any(dim=1)


def reference_masked(exact_bf16: torch.Tensor, needed: torch.Tensor, bv: int = XBV) -> torch.Tensor:
    M, V = exact_bf16.shape
    col = needed.repeat_interleave(bv)[:V].to(exact_bf16.device)
    out = exact_bf16.clone()
    out[:, ~col] = float("-inf")
    return out


# =====================================================================================================
# Certified head: Triton kernels (built on first GPU use)
# =====================================================================================================
_K: dict = {}


def _kernels():
    if _K:
        return _K
    import triton
    import triton.language as tl

    @triton.jit
    def _xnorm_kernel(X, sx, XG, XN, BAD, THR, K: tl.constexpr, G: tl.constexpr, NGP: tl.constexpr,
                      INFL: tl.constexpr, XABS: tl.constexpr):
        m = tl.program_id(0)
        NG: tl.constexpr = K // G
        gi = tl.arange(0, NGP)
        gok = gi < NG
        x = tl.load(X + m * sx + gi[:, None] * G + tl.arange(0, G)[None, :], mask=gok[:, None],
                    other=0.0).to(tl.float32)
        s = tl.sum(x * x, axis=1)
        xg = tl.sqrt_rn(s) * INFL + XABS
        xg = tl.where(gok, xg, 0.0)
        tl.store(XG + m * NG + gi, xg, mask=gok)
        xn = tl.sqrt_rn(tl.sum(xg * xg, axis=0)) * INFL
        tl.store(XN + m, xn)
        tl.store(BAD + m, ((xn != xn) | (xn == float("inf"))).to(tl.int32))
        tl.store(THR + m, float("-inf"))

    @triton.jit
    def _screen_kernel(X, sx, Q, E, RT, NV, XG, XN, HI, THR, M, V, VREAL,
                       K: tl.constexpr, G: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                       BK: tl.constexpr, CACC2: tl.constexpr, INFL_Q: tl.constexpr, INFL_B: tl.constexpr,
                       SLOP_REL: tl.constexpr, SLOP_ABS: tl.constexpr):
        pid = tl.program_id(0)
        m = tl.arange(0, BM)
        n = pid * BN + tl.arange(0, BN)
        mok = m < M
        nok = n < V
        n64 = n.to(tl.int64)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            k = k0 + tl.arange(0, BK)
            x = tl.load(X + m[:, None] * sx + k[None, :], mask=mok[:, None], other=0.0)
            q = tl.load(Q + n64[None, :] * K + k[:, None], mask=nok[None, :], other=0)
            eb = tl.load(E + n64[None, :] * (K // 32) + k[:, None] // 32, mask=nok[None, :], other=127)
            sc = (eb.to(tl.int32) << 23).to(tl.float32, bitcast=True)
            wd = (q.to(tl.float32) * sc).to(tl.bfloat16)
            acc = tl.dot(x, wd, acc)
        NG: tl.constexpr = K // G
        qb = tl.zeros((BM, BN), dtype=tl.float32)
        for g in range(0, NG):
            r = tl.load(RT + g * V + n, mask=nok, other=0.0)
            xg = tl.load(XG + m * NG + g, mask=mok, other=0.0)
            qb += xg[:, None] * r[None, :]
        nv = tl.load(NV + n, mask=nok, other=0.0)
        xn = tl.load(XN + m, mask=mok, other=0.0)
        B = (qb * INFL_Q + CACC2 * (xn[:, None] * nv[None, :])) * INFL_B
        B = tl.where(B != B, float("inf"), B)
        slop = (tl.abs(acc) + B) * SLOP_REL + SLOP_ABS
        lo = (acc - B - slop).to(tl.bfloat16).to(tl.float32)
        hi = (acc + B + slop).to(tl.bfloat16).to(tl.float32)
        nonfin = (acc != acc) | (tl.abs(acc) == float("inf"))
        lo = tl.where(nonfin | (lo != lo), float("-inf"), lo)
        hi = tl.where(nonfin | (hi != hi), float("inf"), hi)
        real = n < VREAL  # padding rows never raise T and are never needed
        lo = tl.where(real[None, :], lo, float("-inf"))
        hi = tl.where(real[None, :], hi, float("-inf"))
        ok = mok[:, None] & nok[None, :]
        tl.store(HI + m[:, None] * V + n[None, :], hi.to(tl.bfloat16), mask=ok)
        lof = tl.where(ok, lo, float("-inf"))
        tl.atomic_max(THR + m, tl.max(lof, axis=1), mask=mok)

    @triton.jit
    def _refine_kernel(X, sx, W, HI, THR, BAD, OUT, so, NEED, M, V,
                       K: tl.constexpr, BV: tl.constexpr, BMX: tl.constexpr, BK: tl.constexpr,
                       FULL: tl.constexpr):
        pid = tl.program_id(0)
        j = pid * BV + tl.arange(0, BV)
        ok = j < V
        mm = tl.arange(0, BMX)
        mok = mm < M
        if FULL:
            need = True
        else:
            bad = tl.max(tl.load(BAD + mm, mask=mok, other=0), axis=0)
            h = tl.load(HI + mm[:, None] * V + j[None, :], mask=mok[:, None] & ok[None, :],
                        other=float("-inf")).to(tl.float32)
            T = tl.load(THR + mm, mask=mok, other=float("inf"))
            hit = tl.max(tl.max((h >= T[:, None]).to(tl.int32), axis=1), axis=0)
            need = (hit > 0) | (bad > 0)
        if need:
            j64 = j.to(tl.int64)
            acc = tl.zeros((BV, BMX), dtype=tl.float32)
            for k0 in range(0, K, BK):
                k = k0 + tl.arange(0, BK)
                w = tl.load(W + j64[:, None] * K + k[None, :], mask=ok[:, None], other=0.0)
                xt = tl.load(X + mm[None, :] * sx + k[:, None], mask=mok[None, :], other=0.0)
                acc = tl.dot(w, xt, acc)
            tl.store(OUT + mm[None, :] * so + j[:, None], acc.to(tl.bfloat16), mask=ok[:, None] & mok[None, :])
            tl.atomic_add(NEED, 1)
        else:
            ninf = tl.full((BV, BMX), float("-inf"), tl.float32).to(tl.bfloat16)
            tl.store(OUT + mm[None, :] * so + j[:, None], ninf, mask=ok[:, None] & mok[None, :])

    _K.update(triton=triton, xnorm=_xnorm_kernel, screen=_screen_kernel, refine=_refine_kernel)
    return _K


def certified_logits(st: CertState, x: torch.Tensor, full: bool = False, debug: dict | None = None) -> torch.Tensor:
    """x [M, K] bf16 contiguous -> bf16 [M, V]: exact logits on needed tiles, -inf elsewhere
    (full=True: exact kernel on every tile). No host sync; graph-capturable."""
    k = _kernels()
    triton = k["triton"]
    x = x.contiguous()
    M, K = x.shape
    V, dev = st.V, x.device
    NG = K // st.group
    out = torch.empty((M, V), dtype=torch.bfloat16, device=dev)
    xg = torch.empty((M, NG), dtype=torch.float32, device=dev)
    xn = torch.empty((M,), dtype=torch.float32, device=dev)
    bad = torch.empty((M,), dtype=torch.int32, device=dev)
    thr = torch.empty((M,), dtype=torch.float32, device=dev)
    hi = torch.empty((M, V), dtype=torch.bfloat16, device=dev)
    if not full:
        k["xnorm"][(M,)](x, x.stride(0), xg, xn, bad, thr, K=K, G=st.group, NGP=next_pow2(NG),
                         INFL=INFL_X, XABS=X_ABS)
        BM = max(16, next_pow2(M))
        k["screen"][(triton.cdiv(V, SBN),)](
            x, x.stride(0), st.q, st.e, st.rt, st.nv, xg, xn, hi, thr, M, V, st.vreal,
            K=K, G=st.group, BM=BM, BN=SBN, BK=SBK, CACC2=2.0 * c_acc(K), INFL_Q=INFL_Q, INFL_B=INFL_B,
            SLOP_REL=SLOP_REL, SLOP_ABS=SLOP_ABS,
            num_warps=SWARPS if BM <= 32 else max(SWARPS, 8), num_stages=SSTAGES)
    k["refine"][(triton.cdiv(V, XBV),)](
        x, x.stride(0), st.W, hi, thr, bad, out, out.stride(0), st.need, M, V,
        K=K, BV=XBV, BMX=max(16, next_pow2(M)), BK=XBK, FULL=full, num_warps=XWARPS, num_stages=XSTAGES)
    if debug is not None:
        debug.update(hi=hi, thr=thr, bad=bad, xg=xg, xn=xn)
    return out


def _check(st: CertState, stock: torch.Tensor, cert: torch.Tensor) -> None:
    """Device-side compare of this rank's shard. counters: [calls, armed calls, local-argmax mismatches,
    value-at-argmax mismatches, computed-logit mismatches, -, rows, -]."""
    a, b = stock.float(), cert.float()
    ia, ib = a.argmax(dim=-1), b.argmax(dim=-1)
    va, vb = a.gather(-1, ia[:, None]), b.gather(-1, ib[:, None])
    computed = b != float("-inf")
    upd = torch.stack([
        torch.ones((), dtype=torch.int64, device=a.device),
        torch.full((), int(_S["armed"]), dtype=torch.int64, device=a.device),
        (ia != ib).sum(), (va != vb).sum(), ((a != b) & computed).sum(),
        torch.zeros((), dtype=torch.int64, device=a.device),
        torch.full((), a.shape[0], dtype=torch.int64, device=a.device),
        torch.zeros((), dtype=torch.int64, device=a.device),
    ])
    st.counters.add_(upd)


# =====================================================================================================
# Engine integration
# =====================================================================================================
_S: dict = {"armed": False, "installed": False, "runner_hooked": False, "dspark_hooked": False}


def _tp():
    from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

    return get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size()


def _tp_agree_min(ok: bool) -> bool:
    """MIN-all-reduce a boolean over the TP CPU group (every rank must call it the same number of times)."""
    from vllm.distributed import get_tp_group

    grp = get_tp_group()
    if grp.world_size == 1:
        return ok
    t = torch.tensor([1 if ok else 0], dtype=torch.int64)
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN, group=grp.cpu_group)
    return bool(int(t.item()))


def step_armed(runner, input_batch, grammar_output) -> bool:
    """True only when the only consumer of this step's target logits is a greedy argmax.
    Everything read here is per-batch scheduler state, identical on every TP rank (and correctness
    does not even need agreement: each rank's masked shard is argmax-safe on its own)."""
    try:
        if grammar_output is not None or input_batch is None:
            return False
        rs, sp = getattr(runner, "rejection_sampler", None), getattr(runner, "sampler", None)
        if rs is None or sp is None or int(getattr(input_batch, "num_draft_tokens", 0)) <= 0:
            return False
        if getattr(rs, "watermark_key", None) is not None or getattr(rs, "synthetic_conditional_rates", None) is not None:
            return False
        if getattr(sp, "compute_nans", False) or getattr(sp, "trace_replay_state", None) is not None:
            return False
        idx = input_batch.idx_mapping_np
        if idx is None or len(idx) == 0:
            return False
        ss = sp.sampling_states
        if not bool(np.all(ss.temperature.np[idx] == 0.0)):
            return False
        if bool(np.any(sp.needs_logits_processing[idx])):
            return False
        from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS

        if ss.max_num_logprobs(idx) != NO_LOGPROBS:
            return False
        lts = getattr(sp, "logprob_token_ids_state", None)
        if lts is not None and lts.max_num_token_ids(idx) != 0:
            return False
        return True
    except Exception as e:  # noqa: BLE001 - fail closed
        if not _S.get("arm_warned"):
            _S["arm_warned"] = True
            logger.warning("DSV41_R4_CERT_HEAD: arming check failed (%r); stock head", e)
        return False


def _install_runner_hook() -> None:
    if _S["runner_hooked"]:
        return
    _S["runner_hooked"] = True
    try:
        from vllm.v1.worker.gpu import model_runner as mr

        cls = mr.GPUModelRunner
        if getattr(cls, "_dsv41_r4_cert", False):
            return
        orig = cls.sample

        def sample(self, *args, **kwargs):
            ib = kwargs.get("input_batch", args[1] if len(args) > 1 else None)
            go = kwargs.get("grammar_output", args[2] if len(args) > 2 else None)
            _S["armed"] = step_armed(self, ib, go)
            try:
                return orig(self, *args, **kwargs)
            finally:
                _S["armed"] = False

        cls.sample = sample
        cls._dsv41_r4_cert = True
        logger.info("DSV41_R4_CERT_HEAD: GPUModelRunner.sample arming hook installed")
    except Exception as e:  # noqa: BLE001
        logger.warning("DSV41_R4_CERT_HEAD: could not hook GPUModelRunner.sample (%r): never armed", e)


def _install_dspark_hook() -> None:
    if _S["dspark_hooked"]:
        return
    _S["dspark_hooked"] = True
    try:
        from vllm.models.deepseek_v41.nvidia import dspark as ds

        cls = ds.DSparkDeepseekV4Model
        if getattr(cls, "_dsv41_r4_mainproj", False):
            return
        orig = cls.combine_hidden_states

        def combine_hidden_states(self, aux_hidden_states):
            y = mainproj_split(self.main_proj, aux_hidden_states)
            if y is None:
                return orig(self, aux_hidden_states)
            return self.main_norm(y)

        cls.combine_hidden_states = combine_hidden_states
        cls._dsv41_r4_mainproj = True
        logger.info("DSV41_R4_MAINPROJ_SPLIT: DSpark combine_hidden_states hook installed")
    except Exception as e:  # noqa: BLE001
        logger.warning("DSV41_R4_MAINPROJ_SPLIT: could not hook DSpark (%r): stock main_proj", e)


def maybe_install() -> None:
    if _S["installed"]:
        return
    _S["installed"] = True
    if CERT_MODE != "0":
        _install_runner_hook()
    if MAINPROJ_SPLIT:
        _install_dspark_hook()


def _cert_state(model):
    st = getattr(model, "_dsv41_r4_cert", None)
    if st is not None:
        return st or None
    if _capturing():
        return None
    reason = None
    lp, head = model.logits_processor, model.lm_head
    W = getattr(head, "weight", None)
    qm = type(getattr(head, "quant_method", None)).__name__
    if getattr(lp, "logits_as_input", False) or getattr(lp, "soft_cap", None) is not None or lp.scale != 1.0:
        reason = "logits scale / soft cap / logits_as_input"
    elif qm != "UnquantizedEmbeddingMethod":
        reason = f"lm_head quant method {qm}"
    elif W is None or W.dtype != torch.bfloat16 or W.dim() != 2 or not W.is_contiguous() or not W.is_cuda:
        reason = "lm_head.weight is not a contiguous bf16 CUDA matrix"
    elif not CERT_M:
        reason = "DSV41_R4_CERT_HEAD_M is empty"
    st = None
    if reason is None:
        try:
            si = getattr(head, "shard_indices", None)
            vreal = None
            if si is not None:
                vreal = int(getattr(si, "num_org_elements",
                                    si.org_vocab_end_index - si.org_vocab_start_index))
            st = build_cert_tables(W.data, vreal=vreal)
            st.lm_head = head
        except Exception as e:  # noqa: BLE001
            reason = f"build failed: {e!r}"
            st = None
    model._dsv41_r4_cert = st if st is not None else False
    if SF_ANY:
        # g1 DSV41_SYNC_FREE: once, at the build every rank reaches at the same (first non-capturing)
        # target_logits call: the compact exchange changes a collective's size, so it is used only if
        # EVERY rank has CERT tables (MIN over the TP CPU group).
        _S["sf_ok"] = _tp_agree_min(st is not None and _sf_layout_ok(lp, head, st))
        logger.info("DSV41_SYNC_FREE=%s %s", ",".join(sorted(SF_PARTS)),
                    "agreed on every rank" if _S["sf_ok"]
                    else "DISABLED: some rank has no CERT tables or a padded / extended vocab shard")
        if _S["sf_ok"]:
            # JIT both kernels now (eager, before any capture / request); no collective here
            tp = int(getattr(head, "tp_size", 1) or 1)
            dummy = torch.full((2, st.V), float("-inf"), dtype=torch.bfloat16, device=W.device)
            p = sf_local_pairs(dummy, st.V, 0)
            sf_onehot(p.repeat(tp, 1).contiguous(), 2, tp, int(lp.org_vocab_size))
            torch.cuda.synchronize()
    torch.cuda.empty_cache()
    if st is None:
        logger.warning("DSV41_R4_CERT_HEAD disabled: %s", reason)
    else:
        mb = (st.q.numel() + st.e.numel() + st.rt.numel() * 4 + st.nv.numel() * 4) / 1e6
        logger.info("DSV41_R4_CERT_HEAD=%s built: MXINT8 shard %dx%d (%.1f MB screen vs %.1f MB bf16), "
                    "rel eps %.4f%%, unsafe rows %d, vreal %d, qualified M %s",
                    CERT_MODE, st.V, st.K, mb, st.V * st.K * 2 / 1e6, st.eps * 100, st.n_unsafe, st.vreal,
                    _fmt_m(CERT_M))
    return st


def _fmt_m(ms) -> str:
    s = sorted(ms)
    if not s:
        return "none"
    out, a, b = [], s[0], s[0]
    for v in s[1:]:
        if v == b + 1:
            b = v
            continue
        out.append(f"{a}-{b}" if a != b else f"{a}")
        a = b = v
    out.append(f"{a}-{b}" if a != b else f"{a}")
    return ",".join(out)


def _finish(lp, head, logits, skip_gather: bool):
    """LogitsProcessor._get_logits after _apply_head (forward's soft cap / scale are excluded by
    eligibility)."""
    if skip_gather:
        return logits
    if head.tp_size > 1:
        logits = lp._gather_logits(logits)
    if logits is not None:
        logits = logits[..., : lp.org_vocab_size]
    return logits


def _log_stats(st: CertState) -> None:
    c = [int(v) for v in st.counters.tolist()]
    need = int(st.need.item())
    try:
        rank = _tp()[0]
    except Exception:  # noqa: BLE001
        rank = 0
    msg = (f"DSV41_R4_CERT_HEAD rank {rank}: calls {st.calls} (armed {st.armed_calls}), tiles computed {need} "
           f"(full = {(st.V + XBV - 1) // XBV}/call)")
    if c[0]:
        msg += (f"; check: calls {c[0]} (armed {c[1]}), argmax mismatches {c[2]}, value mismatches {c[3]}, "
                f"computed-logit mismatches {c[4]}, rows {c[6]}")
    if rank == 0 or any(c[2:5]):
        logger.info(msg)


# =====================================================================================================
# g1-stepovh part B: DSV41_SYNC_FREE greedy logits exchange (see the module doc)
# =====================================================================================================
_SFK: dict = {}


def _sf_kernels():
    if _SFK:
        return _SFK
    import triton
    import triton.language as tl

    @triton.jit(do_not_specialize=["sl", "valid", "vstart"])
    def _sf_local_argmax_kernel(LG, sl, OUT, valid, vstart, BLOCK: tl.constexpr):
        """OUT[row, :] = (max, vstart + leftmost argmax, 0, 0) over LG[row, :valid] (bf16 read as fp32,
        NaN read as -inf). Ties: leftmost inside a block (tie_break_left), earlier block across blocks
        (strict >). An all -inf row reports index vstart + 0."""
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        x = tl.load(LG + row * sl + cols, mask=cols < valid, other=float("-inf")).to(tl.float32)
        x = tl.where(x != x, float("-inf"), x)
        best_v, best_i = tl.max(x, axis=0, return_indices=True, return_indices_tie_break_left=True)
        for v0 in range(BLOCK, valid, BLOCK):
            c = v0 + cols
            x = tl.load(LG + row * sl + c, mask=c < valid, other=float("-inf")).to(tl.float32)
            x = tl.where(x != x, float("-inf"), x)
            v, i = tl.max(x, axis=0, return_indices=True, return_indices_tie_break_left=True)
            upd = v > best_v
            best_i = tl.where(upd, v0 + i, best_i)
            best_v = tl.where(upd, v, best_v)
        base = OUT + row * 4
        tl.store(base, best_v)
        tl.store(base + 1, (vstart + best_i).to(tl.float32))
        tl.store(base + 2, 0.0)
        tl.store(base + 3, 0.0)

    @triton.jit(do_not_specialize=["M", "so", "V"])
    def _sf_onehot_kernel(G, M, TOK, OUT, so, V, TP: tl.constexpr, BLOCK: tl.constexpr):
        """G = the all-gathered [TP * M, 4] rows (rank-major). Winner = the first rank holding the
        row's max value (ranks hold ascending vocab ranges -> global leftmost max). OUT[row, :V] = -inf
        except 0 at the winner's id (bf16); TOK[row] = winner id (column block 0 writes it)."""
        row = tl.program_id(0)
        blk = tl.program_id(1)
        best_v = tl.load(G + row * 4)
        best_t = tl.load(G + row * 4 + 1)
        for r in tl.static_range(1, TP):
            v = tl.load(G + (r * M + row) * 4)
            t = tl.load(G + (r * M + row) * 4 + 1)
            upd = v > best_v
            best_t = tl.where(upd, t, best_t)
            best_v = tl.where(upd, v, best_v)
        tok = best_t.to(tl.int32)
        cols = blk * BLOCK + tl.arange(0, BLOCK)
        val = tl.where(cols == tok, 0.0, float("-inf")).to(tl.bfloat16)
        tl.store(OUT + row * so + cols, val, mask=cols < V)
        if blk == 0:
            tl.store(TOK + row, tok.to(tl.int64))

    _SFK.update(triton=triton, local=_sf_local_argmax_kernel, onehot=_sf_onehot_kernel)
    return _SFK


def sf_local_pairs(local: torch.Tensor, valid: int, vstart: int) -> torch.Tensor:
    """[M, Vs] bf16 local logits -> [M, 4] fp32 (max, global id of the leftmost max, 0, 0)."""
    assert local.dim() == 2 and local.stride(1) == 1 and local.dtype == torch.bfloat16
    assert 0 <= valid <= local.shape[1]
    k = _sf_kernels()
    M = local.shape[0]
    pair = torch.empty((M, 4), dtype=torch.float32, device=local.device)
    k["local"][(M,)](local, local.stride(0), pair, valid, vstart, BLOCK=SF_BLOCK, num_warps=8)
    return pair


def sf_onehot(gathered: torch.Tensor, M: int, tp: int, V: int) -> tuple[torch.Tensor, torch.Tensor]:
    """[tp * M, 4] gathered rows -> ([M, V] bf16 one-hot logits, [M] int64 winner ids)."""
    k = _sf_kernels()
    assert gathered.shape == (tp * M, 4) and gathered.is_contiguous() and gathered.dtype == torch.float32
    out = torch.empty((M, V), dtype=torch.bfloat16, device=gathered.device)
    tok = torch.empty((M,), dtype=torch.int64, device=gathered.device)
    k["onehot"][(M, k["triton"].cdiv(V, SF_BLOCK))](gathered, M, tok, out, out.stride(0), V,
                                                     TP=tp, BLOCK=SF_BLOCK, num_warps=4)
    return out, tok


def sync_free_logits(lp, head, local: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Armed step: certified local shard -> ([M, org_vocab] bf16 one-hot at the stock greedy token,
    [M] int64 tokens) with ONE 16 B/row all-gather instead of the [M, V/tp] bf16 logits gather."""
    from vllm.distributed import tensor_model_parallel_all_gather

    si = head.shard_indices
    valid = int(si.num_org_elements)
    pair = sf_local_pairs(local, valid, int(si.org_vocab_start_index))
    gathered = tensor_model_parallel_all_gather(pair, dim=0)
    if not _S.get("sf_logged"):
        _S["sf_logged"] = True
        logger.info("DSV41_SYNC_FREE argmax: first armed step (M=%d): 16 B/row all-gather replaces the "
                    "%d x %d bf16 logits all-gather", local.shape[0], local.shape[0], local.shape[1])
    return sf_onehot(gathered.contiguous(), local.shape[0], head.tp_size, int(lp.org_vocab_size))


def _sf_layout_ok(lp, head, st) -> bool:
    """The one-hot equals the stock [M, org_vocab] slice only when the shards tile the org vocab exactly:
    no per-shard padding, no added (LoRA) vocab, tp * shard width == org_vocab_size. Fail closed."""
    try:
        si = head.shard_indices
        tp = int(head.tp_size)
        return (int(si.num_org_vocab_padding) == 0 and int(si.num_added_elements) == 0
                and int(si.num_org_elements) == int(st.V) and tp * int(st.V) == int(lp.org_vocab_size)
                and int(si.org_vocab_start_index) == _tp()[0] * int(st.V))
    except Exception:  # noqa: BLE001
        return False


def _sf_tick() -> None:
    _S["sf_calls"] = _S.get("sf_calls", 0) + 1


def _sf_audit(lp, head, local: torch.Tensor, full: torch.Tensor) -> None:
    """audit: the stock gather already ran (full); run the compact exchange too and count, on device,
    rows whose winner differs from full.argmax(-1). One host read every SF_AUDIT_EVERY calls."""
    _out, tok = sync_free_logits(lp, head, local)
    ctr = _S.get("sf_ctr")
    if ctr is None:
        ctr = _S["sf_ctr"] = torch.zeros((3,), dtype=torch.int64, device=local.device)
    ref = full.argmax(dim=-1)
    ctr.add_(torch.stack([torch.ones((), dtype=torch.int64, device=ctr.device),
                          torch.full((), ref.shape[0], dtype=torch.int64, device=ctr.device),
                          (ref != tok).sum()]))
    _sf_tick()
    if _S["sf_calls"] % SF_AUDIT_EVERY == 0:
        c = [int(v) for v in ctr.tolist()]
        try:
            rank = _tp()[0]
        except Exception:  # noqa: BLE001
            rank = 0
        msg = f"DSV41_SYNC_FREE audit rank {rank}: calls {c[0]} rows {c[1]} winner mismatches {c[2]}"
        if c[2]:
            logger.warning(msg + "  MISMATCH: keep the stock gather")
        elif rank == 0:
            logger.info(msg)


def target_logits(model, hidden_states: torch.Tensor, skip_gather: bool = False):
    """Hook for DeepseekV41LLMForCausalLM.compute_logits[_local]. None -> caller runs the stock head."""
    maybe_install()
    if CERT_MODE == "0":
        return None
    st = _cert_state(model)
    if st is None or not isinstance(hidden_states, torch.Tensor) or hidden_states.dim() != 2:
        return None
    if hidden_states.dtype != torch.bfloat16 or model.lm_head is not st.lm_head:
        return None
    M = int(hidden_states.shape[0])
    if M == 0 or M > CERT_MAX_M or M not in CERT_M or _capturing():
        return None
    lp, head = model.logits_processor, model.lm_head
    armed = bool(_S["armed"])
    if CERT_MODE == "1" and not armed:
        return None
    st.calls += 1
    st.armed_calls += int(armed)
    if CERT_MODE == "1":
        local = certified_logits(st, hidden_states)
        if SF_ANY and armed and not skip_gather and _S.get("sf_ok") and head.tp_size > 1:
            # g1 DSV41_SYNC_FREE: the only consumer of these logits is a greedy argmax
            if SF_ARGMAX:
                out, _tok = sync_free_logits(lp, head, local)
                _sf_tick()
                return out
            full = _finish(lp, head, local, skip_gather)
            _sf_audit(lp, head, local, full)
            return full
    else:
        local = lp._apply_head(head, hidden_states, None)
        _check(st, local, certified_logits(st, hidden_states))
    if CERT_LOG_EVERY and st.calls % CERT_LOG_EVERY == 0:
        _log_stats(st)
    return _finish(lp, head, local, skip_gather)


# =====================================================================================================
# DSpark main_proj column split
# =====================================================================================================
MXFP8_BLOCK = 32


def swizzle_scale(sf2d: torch.Tensor, M: int, K: int) -> torch.Tensor:
    """Row-major [M, K/32] e8m0 -> FlashInfer F8_128x4 1-D layout (== vllm swizzle_mxfp8_scale)."""
    nm = (M + 127) // 128
    nk = (K // MXFP8_BLOCK + 3) // 4
    pad = torch.zeros((nm * 128, nk * 4), dtype=sf2d.dtype, device=sf2d.device)
    pad[:M, : K // MXFP8_BLOCK] = sf2d
    return pad.view(nm, 4, 32, nk, 4).transpose(1, 3).contiguous().view(-1)


def unswizzle_scale(sf1d: torch.Tensor, M: int, K: int) -> torch.Tensor:
    nm = (M + 127) // 128
    nk = (K // MXFP8_BLOCK + 3) // 4
    assert sf1d.numel() == nm * 128 * nk * 4, (sf1d.numel(), M, K)
    t = sf1d.reshape(nm, nk, 32, 4, 4).transpose(1, 3).contiguous().view(nm * 128, nk * 4)
    return t[:M, : K // MXFP8_BLOCK].contiguous()


def rowmajor_scale(linear) -> torch.Tensor | None:
    w = linear.weight
    ws = getattr(linear, "weight_scale", None)
    if ws is None or w.dtype != torch.float8_e4m3fn or w.dim() != 2 or ws.element_size() != 1:
        return None
    N, K = w.shape
    nk = (K // MXFP8_BLOCK + 3) // 4
    if ws.dim() == 1 and ws.numel() == ((N + 127) // 128) * 128 * nk * 4:
        return unswizzle_scale(ws.data.view(torch.uint8), N, K)
    if ws.dim() == 2 and ws.shape[0] >= N and ws.shape[1] >= K // MXFP8_BLOCK:
        return ws.data[:N, : K // MXFP8_BLOCK].contiguous().view(torch.uint8)
    return None


def split_rows(N: int, rank: int, tp: int) -> slice:
    if N % tp or (N // tp) % 32 or N // tp < 128:
        raise ValueError(f"N={N} not splittable by tp={tp} into 32-aligned >=128-row slices")
    n = N // tp
    return slice(rank * n, (rank + 1) * n)


class ColSplit:
    def __init__(self, linear, rank: int, tp: int):
        w = linear.weight.data
        N, K = w.shape
        s2d = rowmajor_scale(linear)
        if s2d is None:
            raise RuntimeError("unrecognised MXFP8 weight/scale layout")
        rows = split_rows(N, rank, tp)
        self.N, self.K, self.n = N, K, rows.stop - rows.start
        self.w = w[rows].contiguous()
        self.s_swz = swizzle_scale(s2d[rows].contiguous(), self.n, K)
        self.checked: dict[int, bool] = {}

    def local(self, x):
        """This rank's [M, n] output columns (no collective)."""
        from vllm.model_executor.layers.fusion.quant_activation import as_quantized_activation
        from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize
        from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
        from vllm.utils import flashinfer as vllm_flashinfer

        qa = as_quantized_activation(x, kMxfp8Dynamic)
        if qa is not None:
            xq, xs, out_dtype, shape = qa.data, qa.scale, qa.orig_dtype, tuple(qa.orig_shape)
        else:
            out_dtype, shape = x.dtype, tuple(x.shape)
            xq, xs = mxfp8_e4m3_quantize(x.view(-1, self.K), is_sf_swizzled_layout=True)
        y = vllm_flashinfer.mm_mxfp8(xq, self.w.t(), xs, self.s_swz, out_dtype=out_dtype, backend="cutlass")
        return y, shape

    def gather(self, y, shape):
        from vllm.distributed import tensor_model_parallel_all_gather

        # all_gather(dim=-1) == concat of the rank slices == the stock [M, N] layout.
        return tensor_model_parallel_all_gather(y, dim=-1).view(*shape[:-1], self.N)

    def __call__(self, x):
        y, shape = self.local(x)
        return self.gather(y, shape)


def mainproj_split(linear, x):
    """Split main_proj output, or None to run the stock replicated layer."""
    if not MAINPROJ_SPLIT or not isinstance(x, torch.Tensor):
        return None
    M = 1
    for d in x.shape[:-1]:
        M *= int(d)
    if M == 0 or M > MAINPROJ_MAX_M:
        return None
    st = getattr(linear, "_dsv41_r4_split", None)
    if st is None:
        if _capturing():
            return None
        try:
            rank, tp = _tp()
            st = ColSplit(linear, rank, tp) if tp > 1 else None
            ok = st is not None
        except Exception as e:  # noqa: BLE001
            logger.warning("DSV41_R4_MAINPROJ_SPLIT build failed: %r", e)
            st, ok = None, False
        if not _tp_agree_min(ok):
            st = None
        linear._dsv41_r4_split = st if st is not None else False
        if st is not None:
            logger.info("DSV41_R4_MAINPROJ_SPLIT active: %d -> %d rows/rank, K %d, max_m %d",
                        st.N, st.n, st.K, MAINPROJ_MAX_M)
    if not st:
        return None
    verdict = st.checked.get(M)
    if verdict is None:
        if _capturing():
            return None
        stock = linear(x)
        try:
            y, shape = st.local(x)
            ok = True
        except Exception as e:  # noqa: BLE001
            logger.warning("DSV41_R4_MAINPROJ_SPLIT M=%d local GEMM failed: %r", M, e)
            ok = False
        same = False
        if _tp_agree_min(ok):  # every rank reaches the gather, or none does
            mine = st.gather(y, shape)
            same = bool(torch.equal(stock.view(torch.int16), mine.view(torch.int16)))
        verdict = _tp_agree_min(same)
        st.checked[M] = verdict
        if not verdict:
            logger.warning("DSV41_R4_MAINPROJ_SPLIT M=%d: not bit-identical on some rank -> stock for this M", M)
        return stock
    return st(x) if verdict else None
