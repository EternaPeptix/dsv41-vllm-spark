# SPDX-License-Identifier: Apache-2.0
"""climb round 6, kit r6-head: default-off replacements for the certified target LM head's kernels
(dsv41_r4_small.certified_logits, DSV41_R4_CERT_HEAD=1). Imported ONLY from that function, and only when one of the
master flags below is set; with both unset this file is never imported and the live kernels run unchanged.

Master flags (read once at import of dsv41_r4_small; every TP rank must see the same values):

  DSV41_R6_HEAD_REFINE=0|tile|row
      Replaces the fixed-grid _refine_kernel (one CTA per 16 vocab rows, 1010 CTAs, the data-dependent tiles do the
      whole 5120-long dot product in whichever CTA they land in) by two kernels:
        _r6_cand_kernel    one CTA per CB vocab rows: the live tile predicate (any row m: hi[m, v] >= T[m], or any
                           bad row), writes -inf to every NOT selected (m, v), and appends the selected vocab rows to a
                           compact list (one int32 atomic per CTA + an in-CTA prefix sum);
        _r6_refine_kernel  a fixed persistent grid that walks the list in chunks and computes each listed row with
                           the live refine's exact op sequence (same tl.dot tile [16, BMX], same K order, same BK,
                           same bf16 rounding of the fp32 accumulator), writing it in place.
      tile: the selected set is the live one (every row of every tile the live kernel computes). The output
            tensor is bit-identical to the live refine's, element for element (-inf included).
      row:  only the rows whose own predicate holds (any m: hi[m, v] >= T[m], or a bad row) are computed; the other
            rows of a live-needed tile are -inf instead of their exact value. Each computed logit is bit-identical
            to the live one. Argmax-identical by the certificate (module doc of dsv41_r4_small): a row with
            hi_v < T for every m has l_v <= hi_v < T <= max l, so it is neither the max nor tied with it; every row
            tied with the max has hi >= T and is computed. Reads 16x fewer head bytes for an isolated candidate.
  DSV41_R6_HEAD_SCREEN=0|1
      _r6_screen_kernel: the live _screen_kernel with the MXINT8 dequant rebuilt from exact integer steps
      (DSV41_R6_HEAD_SCREEN_DQ). The dot product, the bound, the bf16 HI store and the THR atomic max are the live
      code, unchanged. DQ=1: float(q) via the 0x4B400000 magic (exact for |q| < 2^22) instead of I2F, one exponent
      byte per 32 weights instead of one per weight, the live f32 multiply and the live cvt.rn.bf16. DQ=2: as 1, but
      the bf16 is the top 16 bits of the exact f32 product (exact because |q| <= 128 has <= 8 significant bits and
      the product is a normal f32 for every exponent byte the table builder emits, 1..254). tests/test_dequant.py
      checks both against the live formula for every (q, e) pair.

Tuning (read once; ValueError at import on a malformed value):
  DSV41_R6_HEAD_REFINE_CFG=BK,warps,stages,grid,spread   default 0,0,0,0,1 = the live XBK / XWARPS / XSTAGES,
      grid 0 = 4 x SM count, spread 1 = rows per chunk = clamp(ceil(n / grid), 1, 16) (0 = always 16).
      Only BMX = 16 (M <= 16) uses a non-live BK/stages; larger BMX keep the live XBK / XSTAGES (shared-memory size).
  DSV41_R6_HEAD_CAND_ROWS=256    rows per _r6_cand_kernel CTA (multiple of the live XBV; halved per BMX doubling)
  DSV41_R6_HEAD_SCREEN_DQ=1      1 or 2 (above)
  DSV41_R6_HEAD_SCREEN_CFG=0,0,0,0  BN,BK,warps,stages for the r6 screen; 0 = the live SBN / SBK / SWARPS / SSTAGES
  DSV41_R6_HEAD_SELFCHECK=1     at the first eager (non-capturing) call per row count M, run the live pass too on the
      same input and compare (screen: HI and THR bits; tile: whole output bits; row: output == where(row predicate,
      live, -inf) and the list length; argmax always). One host sync per new M. A failure logs ERROR, returns the live
      result and switches r6 off for the process (rank-local: each rank's shard is argmax-safe on its own, and r6
      adds or removes no collective). Under capture an unchecked M runs the live kernels. 0 = no check.
  DSV41_R6_HEAD_STATS_EVERY=2000 every N r6 calls, log (one host read) this rank's cumulative computed rows, the
      tiles the live kernel would have computed, and calls; 0 = never (no host read at all)

Exactness, races, graph safety: see the kit README / EXACTNESS.md. No host sync on the hot path (the stats read
every N calls and a one-time table check at the first call aside); fixed grids; every buffer is per call from the
caching allocator except the stats counter; the list counter is zeroed by _r6_xnorm_kernel (the live xnorm body plus
one store) in stream order before the screen.
"""
from __future__ import annotations

import os

import torch

_OFF = ("", "0", "off", "false", "no")
TAG = "r6-head"


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip().lower()


def parse_refine(v: str) -> str:
    v = (v or "").strip().lower()
    if v in _OFF:
        return "0"
    if v in ("tile", "row"):
        return v
    raise ValueError(f"DSV41_R6_HEAD_REFINE={v!r}: expected 0, tile or row")


def parse_screen(v: str) -> bool:
    v = (v or "").strip().lower()
    if v in _OFF:
        return False
    if v in ("1", "on", "true"):
        return True
    raise ValueError(f"DSV41_R6_HEAD_SCREEN={v!r}: expected 0 or 1")


def parse_ints(name: str, v: str, n: int) -> tuple:
    parts = [p.strip() for p in (v or "").split(",")]
    if len(parts) != n or not all(p.lstrip("-").isdigit() for p in parts):
        raise ValueError(f"{name}={v!r}: expected {n} comma-separated integers")
    out = tuple(int(p) for p in parts)
    if any(p < 0 for p in out):
        raise ValueError(f"{name}={v!r}: negative value")
    return out


def _pow2(v: int) -> bool:
    return v > 0 and v & (v - 1) == 0


REFINE = parse_refine(os.environ.get("DSV41_R6_HEAD_REFINE", "0"))
SCREEN = parse_screen(os.environ.get("DSV41_R6_HEAD_SCREEN", "0"))
R_BK, R_WARPS, R_STAGES, R_GRID, R_SPREAD = parse_ints(
    "DSV41_R6_HEAD_REFINE_CFG", os.environ.get("DSV41_R6_HEAD_REFINE_CFG", "0,0,0,0,1"), 5)
CAND_ROWS = parse_ints("DSV41_R6_HEAD_CAND_ROWS", os.environ.get("DSV41_R6_HEAD_CAND_ROWS", "256"), 1)[0]
SCREEN_DQ = parse_ints("DSV41_R6_HEAD_SCREEN_DQ", os.environ.get("DSV41_R6_HEAD_SCREEN_DQ", "1"), 1)[0]
S_BN, S_BK, S_WARPS, S_STAGES = parse_ints(
    "DSV41_R6_HEAD_SCREEN_CFG", os.environ.get("DSV41_R6_HEAD_SCREEN_CFG", "0,0,0,0"), 4)
STATS_EVERY = parse_ints("DSV41_R6_HEAD_STATS_EVERY", os.environ.get("DSV41_R6_HEAD_STATS_EVERY", "2000"), 1)[0]
SELFCHECK = parse_ints("DSV41_R6_HEAD_SELFCHECK", os.environ.get("DSV41_R6_HEAD_SELFCHECK", "1"), 1)[0]
if SELFCHECK not in (0, 1):
    raise ValueError(f"DSV41_R6_HEAD_SELFCHECK={SELFCHECK}: 0 or 1")

if R_BK and (R_BK % 16 or not _pow2(R_BK)):
    raise ValueError(f"DSV41_R6_HEAD_REFINE_CFG: BK={R_BK} must be a power of two >= 16")
if R_WARPS and not _pow2(R_WARPS):
    raise ValueError(f"DSV41_R6_HEAD_REFINE_CFG: warps={R_WARPS} must be a power of two")
if R_SPREAD not in (0, 1):
    raise ValueError(f"DSV41_R6_HEAD_REFINE_CFG: spread={R_SPREAD} must be 0 or 1")
if not _pow2(CAND_ROWS) or CAND_ROWS < 16:
    raise ValueError(f"DSV41_R6_HEAD_CAND_ROWS={CAND_ROWS}: power of two >= 16")
if SCREEN_DQ not in (1, 2):
    raise ValueError(f"DSV41_R6_HEAD_SCREEN_DQ={SCREEN_DQ}: 1 or 2")
if S_BN and not _pow2(S_BN):
    raise ValueError(f"DSV41_R6_HEAD_SCREEN_CFG: BN={S_BN} must be a power of two")
if S_BK and (S_BK % 32 or not _pow2(S_BK)):
    raise ValueError(f"DSV41_R6_HEAD_SCREEN_CFG: BK={S_BK} must be a power of two >= 32")
if S_WARPS and not _pow2(S_WARPS):
    raise ValueError(f"DSV41_R6_HEAD_SCREEN_CFG: warps={S_WARPS} must be a power of two")

R4_MODULE = "vllm.models.deepseek_v41.nvidia.dsv41_r4_small"

try:  # CPU unit tests import this file without vllm
    from vllm.logger import init_logger

    logger = init_logger(__name__)
except Exception:  # noqa: BLE001
    import logging

    logger = logging.getLogger("dsv41_head_r6")


def _r4():
    import sys

    m = sys.modules.get(R4_MODULE)
    if m is None:
        import importlib

        m = importlib.import_module(R4_MODULE)
    return m


# =====================================================================================================
# Kernels (built on first GPU use)
# =====================================================================================================
_K6: dict = {}


def _kernels6():
    if _K6:
        return _K6
    import triton
    import triton.language as tl

    @triton.jit
    def _r6_xnorm_kernel(X, sx, XG, XN, BAD, THR, CNT, K: tl.constexpr, G: tl.constexpr, NGP: tl.constexpr,
                         INFL: tl.constexpr, XABS: tl.constexpr):
        # == dsv41_r4_small._xnorm_kernel, plus: program 0 zeroes the r6 list counter (stream-ordered before the
        # screen and _r6_cand_kernel of the same call).
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
        if m == 0:
            tl.store(CNT, 0)

    @triton.jit
    def _r6_screen_kernel(X, sx, Q, E, RT, NV, XG, XN, HI, THR, M, V, VREAL,
                          K: tl.constexpr, G: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                          BK: tl.constexpr, CACC2: tl.constexpr, INFL_Q: tl.constexpr, INFL_B: tl.constexpr,
                          SLOP_REL: tl.constexpr, SLOP_ABS: tl.constexpr, DQ: tl.constexpr):
        # == dsv41_r4_small._screen_kernel except the construction of wd (bit-identical, see the module doc).
        pid = tl.program_id(0)
        m = tl.arange(0, BM)
        n = pid * BN + tl.arange(0, BN)
        mok = m < M
        nok = n < V
        n64 = n.to(tl.int64)
        G32: tl.constexpr = BK // 32
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            k = k0 + tl.arange(0, BK)
            x = tl.load(X + m[:, None] * sx + k[None, :], mask=mok[:, None], other=0.0)
            q = tl.load(Q + n64[None, :] * K + k[:, None], mask=nok[None, :], other=0)
            eb = tl.load(E + n64[None, :] * (K // 32) + (k0 // 32 + tl.arange(0, G32))[:, None],
                         mask=nok[None, :], other=127)
            sc = (eb.to(tl.int32) << 23).to(tl.float32, bitcast=True)
            qf = (q.to(tl.int32) + 0x4B400000).to(tl.float32, bitcast=True) - 12582912.0
            wf = tl.reshape(tl.reshape(qf, (G32, 32, BN)) * sc[:, None, :], (BK, BN))
            if DQ == 1:
                wd = wf.to(tl.bfloat16)
            else:
                wd = (wf.to(tl.uint32, bitcast=True) >> 16).to(tl.uint16).to(tl.bfloat16, bitcast=True)
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
    def _r6_cand_kernel(HI, THR, BAD, OUT, so, IDX, CNT, NEED, STATS, M, V,
                        BMX: tl.constexpr, CB: tl.constexpr, TB: tl.constexpr, TILE: tl.constexpr):
        # The live refine predicate, evaluated per row (row) or per TB-row tile (tile, == the live kernel's tiles).
        pid = tl.program_id(0)
        j = pid * CB + tl.arange(0, CB)
        ok = j < V
        mm = tl.arange(0, BMX)
        mok = mm < M
        bad = tl.max(tl.load(BAD + mm, mask=mok, other=0), axis=0)
        h = tl.load(HI + mm[:, None] * V + j[None, :], mask=mok[:, None] & ok[None, :],
                    other=float("-inf")).to(tl.float32)
        T = tl.load(THR + mm, mask=mok, other=float("inf"))
        hrow = tl.max((h >= T[:, None]).to(tl.int32), axis=0)
        htile = tl.max(tl.reshape(hrow, (CB // TB, TB)), axis=1)
        tneed = (htile > 0) | (bad > 0)
        tvalid = (pid * CB + tl.arange(0, CB // TB) * TB) < V
        ntile = tl.sum((tneed & tvalid).to(tl.int32), axis=0)
        if TILE:
            sel = tl.reshape(tl.broadcast_to(tneed[:, None], (CB // TB, TB)), (CB,))
        else:
            sel = (hrow > 0) | (bad > 0)
        sel = sel & ok
        ni = sel.to(tl.int32)
        cnt = tl.sum(ni, axis=0)
        base = tl.atomic_add(CNT, cnt)
        pos = base + tl.cumsum(ni, axis=0) - ni
        tl.store(IDX + pos, j, mask=sel)
        ninf = tl.full((BMX, CB), float("-inf"), tl.float32).to(tl.bfloat16)
        tl.store(OUT + mm[:, None] * so + j[None, :], ninf, mask=mok[:, None] & ok[None, :] & (ni == 0)[None, :])
        tl.atomic_add(NEED, ntile)
        tl.atomic_add(STATS, cnt.to(tl.int64))
        tl.atomic_add(STATS + 1, ntile.to(tl.int64))

    @triton.jit
    def _r6_refine_kernel(X, sx, W, OUT, so, IDX, CNT, M, NPROG,
                          K: tl.constexpr, BV: tl.constexpr, BMX: tl.constexpr, BK: tl.constexpr,
                          SPREAD: tl.constexpr):
        # Each listed row j: the live _refine_kernel's computation of row j (same tl.dot tile [BV, BMX], same K order
        # and BK, fp32 accumulate, one bf16 rounding), stored in place.
        pid = tl.program_id(0)
        n = tl.load(CNT)
        if SPREAD:
            rpc = tl.minimum(tl.maximum((n + NPROG - 1) // NPROG, 1), BV)
        else:
            rpc = BV
        nch = (n + rpc - 1) // rpc
        mm = tl.arange(0, BMX)
        mok = mm < M
        rr = tl.arange(0, BV)
        for c in range(pid, nch, NPROG):
            r = c * rpc + rr
            rok = (rr < rpc) & (r < n)
            j = tl.load(IDX + r, mask=rok, other=0)
            j64 = j.to(tl.int64)
            acc = tl.zeros((BV, BMX), dtype=tl.float32)
            for k0 in range(0, K, BK):
                k = k0 + tl.arange(0, BK)
                w = tl.load(W + j64[:, None] * K + k[None, :], mask=rok[:, None], other=0.0)
                xt = tl.load(X + mm[None, :] * sx + k[:, None], mask=mok[None, :], other=0.0)
                acc = tl.dot(w, xt, acc)
            tl.store(OUT + mm[None, :] * so + j[:, None], acc.to(tl.bfloat16), mask=rok[:, None] & mok[None, :])

    _K6.update(triton=triton, xnorm=_r6_xnorm_kernel, screen=_r6_screen_kernel, cand=_r6_cand_kernel,
               refine=_r6_refine_kernel)
    return _K6


# =====================================================================================================
# Launch
# =====================================================================================================
_S6: dict = {"calls": 0, "nsm": None, "logged": False, "dead": False, "checked": set()}


def refine_cfg(BMX: int, r4) -> tuple[int, int, int]:
    """(BK, warps, stages) for the r6 refine at this BMX. Non-live BK/stages only at BMX 16."""
    bk, st = r4.XBK, r4.XSTAGES
    if BMX <= 16:
        bk = R_BK or bk
        st = R_STAGES or st
    return bk, (R_WARPS or r4.XWARPS), st


def cand_rows(BMX: int, tb: int) -> int:
    cb = CAND_ROWS
    b = 16
    while b < BMX and cb > tb:
        cb //= 2
        b *= 2
    return max(cb, tb)


def _grid_n(dev) -> int:
    if R_GRID:
        return R_GRID
    if _S6["nsm"] is None:
        _S6["nsm"] = int(torch.cuda.get_device_properties(dev).multi_processor_count)
    return 4 * _S6["nsm"]


def _state6(st, dev):
    s6 = getattr(st, "_r6", None)
    if s6 is None:
        s6 = st._r6 = {}
        s6["stats"] = torch.zeros((2,), dtype=torch.int64, device=dev)
        # DQ exactness needs every exponent byte in 1..254 (the live builder clamps to that range); one host read,
        # at the first call (the CERT table is built at a non-capturing call, so this is too).
        s6["dq_ok"] = bool(SCREEN) and int(st.e.min()) >= 1 and int(st.e.max()) <= 254
        if SCREEN and not s6["dq_ok"]:
            logger.warning("DSV41_R6_HEAD_SCREEN: exponent byte outside 1..254 in the CERT table: live screen")
    return s6


def _run(st, x: torch.Tensor, r6_screen: bool, r6_refine: str, after_screen=None) -> dict:
    """One certified head pass. With r6_screen False and r6_refine "0" this is the live
    dsv41_r4_small.certified_logits(full=False) launch sequence verbatim (same kernels, arguments, grids, options)."""
    r4 = _r4()
    k = r4._kernels()
    k6 = _kernels6()
    triton = k["triton"]
    M, K = x.shape
    V, dev = st.V, x.device
    NG = K // st.group
    out = torch.empty((M, V), dtype=torch.bfloat16, device=dev)
    xg = torch.empty((M, NG), dtype=torch.float32, device=dev)
    xn = torch.empty((M,), dtype=torch.float32, device=dev)
    bad = torch.empty((M,), dtype=torch.int32, device=dev)
    thr = torch.empty((M,), dtype=torch.float32, device=dev)
    hi = torch.empty((M, V), dtype=torch.bfloat16, device=dev)
    res = dict(out=out, hi=hi, thr=thr, bad=bad, xg=xg, xn=xn)
    BMX = max(16, r4.next_pow2(M))
    if r6_refine != "0":
        idx = torch.empty((V,), dtype=torch.int32, device=dev)
        cnt = torch.empty((1,), dtype=torch.int32, device=dev)
        res.update(idx=idx, cnt=cnt)
        k6["xnorm"][(M,)](x, x.stride(0), xg, xn, bad, thr, cnt, K=K, G=st.group, NGP=r4.next_pow2(NG),
                          INFL=r4.INFL_X, XABS=r4.X_ABS)
    else:
        k["xnorm"][(M,)](x, x.stride(0), xg, xn, bad, thr, K=K, G=st.group, NGP=r4.next_pow2(NG),
                         INFL=r4.INFL_X, XABS=r4.X_ABS)
    BM = max(16, r4.next_pow2(M))
    if r6_screen:
        sbn, sbk = S_BN or r4.SBN, S_BK or r4.SBK
        swarps, sstages = S_WARPS or r4.SWARPS, S_STAGES or r4.SSTAGES
        k6["screen"][(triton.cdiv(V, sbn),)](
            x, x.stride(0), st.q, st.e, st.rt, st.nv, xg, xn, hi, thr, M, V, st.vreal,
            K=K, G=st.group, BM=BM, BN=sbn, BK=sbk, CACC2=2.0 * r4.c_acc(K), INFL_Q=r4.INFL_Q, INFL_B=r4.INFL_B,
            SLOP_REL=r4.SLOP_REL, SLOP_ABS=r4.SLOP_ABS, DQ=SCREEN_DQ,
            num_warps=swarps if BM <= 32 else max(swarps, 8), num_stages=sstages)
    else:
        k["screen"][(triton.cdiv(V, r4.SBN),)](
            x, x.stride(0), st.q, st.e, st.rt, st.nv, xg, xn, hi, thr, M, V, st.vreal,
            K=K, G=st.group, BM=BM, BN=r4.SBN, BK=r4.SBK, CACC2=2.0 * r4.c_acc(K), INFL_Q=r4.INFL_Q,
            INFL_B=r4.INFL_B, SLOP_REL=r4.SLOP_REL, SLOP_ABS=r4.SLOP_ABS,
            num_warps=r4.SWARPS if BM <= 32 else max(r4.SWARPS, 8), num_stages=r4.SSTAGES)
    if after_screen is not None:
        after_screen()
    if r6_refine != "0":
        tb = r4.XBV
        cb = cand_rows(BMX, tb)
        k6["cand"][(triton.cdiv(V, cb),)](
            hi, thr, bad, out, out.stride(0), idx, cnt, st.need, st._r6["stats"], M, V,
            BMX=BMX, CB=cb, TB=tb, TILE=(r6_refine == "tile"), num_warps=4)
        bk, warps, stages = refine_cfg(BMX, r4)
        nprog = _grid_n(dev)
        k6["refine"][(nprog,)](
            x, x.stride(0), st.W, out, out.stride(0), idx, cnt, M, nprog,
            K=K, BV=tb, BMX=BMX, BK=bk, SPREAD=bool(R_SPREAD), num_warps=warps, num_stages=stages)
    else:
        k["refine"][(triton.cdiv(V, r4.XBV),)](
            x, x.stride(0), st.W, hi, thr, bad, out, out.stride(0), st.need, M, V,
            K=K, BV=r4.XBV, BMX=BMX, BK=r4.XBK, FULL=False, num_warps=r4.XWARPS, num_stages=r4.XSTAGES)
    return res


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16) if t.element_size() == 2 else t.view(torch.int32)


def row_select(hi: torch.Tensor, thr: torch.Tensor, bad: torch.Tensor) -> torch.Tensor:
    """[V] bool: the live refine predicate per vocab row (any m: hi[m, v] >= T[m]) or any bad row (torch mirror)."""
    sel = (hi.float() >= thr[:, None]).any(dim=0)
    return sel | bool(bad.any())


def expected_row_out(live_out: torch.Tensor, sel: torch.Tensor) -> torch.Tensor:
    return torch.where(sel[None, :], live_out, torch.full_like(live_out, float("-inf")))


def compare(live: dict, new: dict, refine: str, screen: bool) -> str:
    """'' when the r6 pass is what it must be against the live pass on the same input, else the first failure."""
    if screen:
        if not torch.equal(_bits(live["hi"]), _bits(new["hi"])):
            return "screen HI differs"
        if not torch.equal(_bits(live["thr"]), _bits(new["thr"])):
            return "screen THR differs"
    if refine == "tile" or (refine == "0"):
        if not torch.equal(_bits(live["out"]), _bits(new["out"])):
            return "refine output differs"
    elif refine == "row":
        sel = row_select(live["hi"], live["thr"], live["bad"])
        if not torch.equal(_bits(new["out"]), _bits(expected_row_out(live["out"], sel))):
            return "row-mode output differs from where(row predicate, live, -inf)"
        if int(new["cnt"][0]) != int(sel.sum()):
            return "row-mode list length differs from the row predicate count"
    if not torch.equal(live["out"].float().argmax(dim=-1), new["out"].float().argmax(dim=-1)):
        return "argmax differs"
    return ""


def certified_logits(st, x: torch.Tensor, debug: dict | None = None, after_screen=None) -> torch.Tensor:
    """== dsv41_r4_small.certified_logits(st, x, full=False) with the r6 screen and/or refine. No host sync except
    the self-check at the first eager call per row count M (DSV41_R6_HEAD_SELFCHECK=1) and the stats read."""
    x = x.contiguous()
    M = int(x.shape[0])
    if getattr(st, "_r6", None) is None and _capturing():
        # first r6 call under a capture (never in production: target_logits returns None while capturing): the
        # one-time table check needs a host read, so this call runs the live kernels and builds nothing
        return _run(st, x, False, "0", after_screen)["out"]
    s6 = _state6(st, x.device)
    use_screen = SCREEN and s6["dq_ok"] and x.shape[1] % (S_BK or _r4().SBK) == 0
    use_refine = REFINE
    if _S6["dead"]:
        use_screen, use_refine = False, "0"
    elif M not in _S6["checked"]:
        if _capturing() or not SELFCHECK:
            if not SELFCHECK:
                _S6["checked"].add(M)
            else:
                use_screen, use_refine = False, "0"  # unchecked M under capture: live kernels
        else:
            live = _run(st, x, False, "0")
            new = _run(st, x, use_screen, use_refine, after_screen)
            err = compare(live, new, use_refine, use_screen)
            _S6["checked"].add(M)
            if err:
                _S6["dead"] = True
                logger.error("[%s] self-check FAILED at M=%d (%s): r6 head DISABLED for this process, live kernels "
                             "serve", TAG, M, err)
                res = live
            else:
                logger.info("[%s] self-check passed at M=%d (refine %s, screen %d)", TAG, M, use_refine,
                            int(use_screen))
                res = new
            return _done(st, s6, res, debug, M)
    res = _run(st, x, use_screen, use_refine, after_screen)
    return _done(st, s6, res, debug, M)


def _capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _done(st, s6, res: dict, debug: dict | None, M: int) -> torch.Tensor:
    r4 = _r4()
    if debug is not None:
        debug.update({k: v for k, v in res.items() if k != "out"})
    _S6["calls"] += 1
    if not _S6["logged"]:
        _S6["logged"] = True
        bk, warps, stages = refine_cfg(16, r4)
        logger.info("[%s] DSV41_R6_HEAD_REFINE=%s SCREEN=%d (DQ %d, BN %d BK %d warps %d stages %d, table %s); refine "
                    "BK %d warps %d stages %d grid %d spread %d; cand rows %d; selfcheck %d (first call M=%d)", TAG,
                    REFINE, int(SCREEN), SCREEN_DQ, S_BN or r4.SBN, S_BK or r4.SBK, S_WARPS or r4.SWARPS,
                    S_STAGES or r4.SSTAGES, "ok" if s6["dq_ok"] else "live", bk, warps, stages,
                    _grid_n(res["out"].device), R_SPREAD, CAND_ROWS, int(SELFCHECK), M)
    if STATS_EVERY and REFINE != "0" and _S6["calls"] % STATS_EVERY == 0 and not _capturing():
        c = [int(v) for v in s6["stats"].tolist()]
        logger.info("[%s] r6 refine: calls %d, rows computed %d (%.1f/call), live-equivalent tiles %d (%.1f/call, "
                    "%d rows)", TAG, _S6["calls"], c[0], c[0] / _S6["calls"], c[1], c[1] / _S6["calls"],
                    c[1] * r4.XBV)
    return res["out"]
