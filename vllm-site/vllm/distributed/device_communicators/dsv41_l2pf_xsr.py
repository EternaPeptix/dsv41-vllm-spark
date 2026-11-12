# SPDX-License-Identifier: Apache-2.0
"""DSv4.1 climb round 4 (r4-xsretime): re-time the paced attention-window L2 prefetch (wo_a + wo_b).

Imported by dsv41_l2_prefetch.py (the "base") only when DSV41_L2_PREFETCH=1 AND at least one master flag below is
set to something other than ""/0/off/false; otherwise this file is never imported and every base edit is a dead
`if _XSR is not None:` branch (same kernels, streams, events, graph contents as without the kit).

Why (t1 traces 2026-11-08, launcher b22e4aa9, ranks 0/4, code/prose/maths): the attention window (wo_a 4.12 MiB +
wo_b 5.16 MiB, plan "L" forks) is paced at 125 GB/s (~79 us) on the base pacer stream 3, which is ONE FIFO shared
with the all-gather / all-reduce windows. In ~31 of 37 layers it starts ~13 us after its fork (queued behind the
qkv_a all-gather window's pacer) and is still issuing wo_b when the wo_b GEMM starts: +6..+13 us in production,
+18..+27 us with DSV41_ATTN_XSPLIT=1 (attention 12-14 us shorter per call). The wo_b GEMM then runs at 19.5-22.7 us
instead of 13-15.6 us, the post-attention all-reduce's paced window queues behind the attn pacer (+4-6 us), and the
router / shared-expert / FC1 work after it slips by another 3-6 us. Net per layer: XSPLIT -12..-14 us of attention,
+10.5..+12.2 us to the post-MoE all-reduce end (see ~/Documents/dsv41-offline/climb-r4/xsretime/).

Master flags (read once at import):
  DSV41_L2_PREFETCH_PACE_GBPS_ATTN=R  pace the attention windows at R GB/s (the ar/ag/post windows keep
                                      DSV41_L2_PREFETCH_PACE_GBPS). 0/unset = base rate.
  DSV41_L2_PREFETCH_PACE_ATTN_STREAM=1  issue the paced attention windows on their own (4th) side stream, joined at the
                                      end of the bracketed forward like stream 3: the window starts at its fork (the
                                      last wq_b) instead of when the all-gather pacer ahead of it in the stream-3 FIFO
                                      finishes, and the post-attention all-reduce pacer no longer waits for its tail.
  DSV41_L2_PREFETCH_ATTN_DEDUP=1      drop from every attention window the byte ranges that the collective window
                                      holding its fork (the qkv_a all-gather window of the same layer) already
                                      prefetched (today: wo_b scales + the first 0.69 MiB of wo_b, 0.85 MiB). Windows
                                      only shrink, never grow; nothing is backfilled.

Only cache hints change: the same live kernels (libdsv41_l2pf_g2.so l2pf_paced_kernel, libdsv41_l2pf.so
l2pf_kernel) walk fewer bytes / at another period / on another side stream that no main-stream kernel waits on
except the end-of-forward join. No main-stream kernel, input, order, tile or reduction changes, so every output bit is
unchanged by construction. Target plans only (the base make_g2_plan); the DSpark drafter plans (r1-pf, own parts)
are untouched. Fails closed: a planning error switches the lever off and keeps the base plan.
"""
from __future__ import annotations

import collections
import os

import torch

TAG = "r4-xsretime"
FLAGS = ("DSV41_L2_PREFETCH_PACE_GBPS_ATTN", "DSV41_L2_PREFETCH_PACE_ATTN_STREAM", "DSV41_L2_PREFETCH_ATTN_DEDUP")


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def _on(name: str) -> bool:
    return _env(name, "0").lower() in ("1", "on", "true")


def _rate(name: str) -> float:
    v = _env(name, "0").lower()
    if v in ("", "off", "false"):
        return 0.0
    try:
        r = float(v)
    except ValueError:
        return -1.0  # reported by bind() as invalid; lever stays off
    return r if r == r and r >= 0 else -1.0


GBPS_ATTN = _rate("DSV41_L2_PREFETCH_PACE_GBPS_ATTN")
STREAM = _on("DSV41_L2_PREFETCH_PACE_ATTN_STREAM")
DEDUP = _on("DSV41_L2_PREFETCH_ATTN_DEDUP")
MAX_GBPS = 400.0  # sanity bound: GB10 LPDDR5X peak is ~273 GB/s; a typo like 2000 must not become a burst


class _R:
    def __init__(self) -> None:
        self.base = None
        self.active = {"rate": False, "stream": False, "dedup": False}
        self.why: dict[str, str] = {}
        self.side4: dict[int, torch.cuda.Stream] = {}
        self.pending4 = None
        self.stats = collections.Counter()


R = _R()
B = None  # the base module (set by bind)
logger = None


def bind(base) -> None:
    """Called once by the base at the end of its import. Decides which levers are active and logs it."""
    global B, logger
    B = base
    R.base = base
    logger = base.logger
    paced_attn = base.G2 and base.PACE_GBPS > 0 and "attn" in base.PACE_KINDS
    if GBPS_ATTN < 0 or GBPS_ATTN > MAX_GBPS:
        R.why["rate"] = f"invalid DSV41_L2_PREFETCH_PACE_GBPS_ATTN={_env('DSV41_L2_PREFETCH_PACE_GBPS_ATTN', '')!r}"
    elif GBPS_ATTN > 0 and not paced_attn:
        R.why["rate"] = "attention windows are not paced (needs PACE_GBPS>0 and 'attn' in PACE_KINDS)"
    elif GBPS_ATTN > 0:
        R.active["rate"] = True
    if STREAM and not paced_attn:
        R.why["stream"] = "attention windows are not paced (needs PACE_GBPS>0 and 'attn' in PACE_KINDS)"
    elif STREAM:
        R.active["stream"] = True
    if DEDUP and not base.G2:
        R.why["dedup"] = "needs the base G2 mode (PACE_GBPS>0 or a REPLAN)"
    elif DEDUP and not (base.WOA or "attn" in base.REPLAN):
        R.why["dedup"] = "no attention windows (needs DSV41_L2_PREFETCH_WOA=1)"
    elif DEDUP:
        R.active["dedup"] = True
    if not any(R.active.values()):
        R.why.setdefault("all", "no lever active (master flags set to 0 or not applicable); base path unchanged")
    logger.info("DSV41_L2_PREFETCH r4-xsretime: active=%s attn rate=%s GB/s (base %s) own stream=%s dedup=%s%s",
                {k: v for k, v in R.active.items()}, GBPS_ATTN if R.active["rate"] else "base", base.PACE_GBPS,
                R.active["stream"], R.active["dedup"],
                ("; INACTIVE: " + "; ".join(f"{k}: {v}" for k, v in sorted(R.why.items()))) if R.why else "")


def any_active() -> bool:
    return any(R.active.values())


def _side_stream4():
    dev = torch.cuda.current_device()
    s = R.side4.get(dev)
    if s is None:
        s = R.side4[dev] = torch.cuda.Stream(device=dev)
    return s


def install() -> None:
    """Base install hook (first collective, outside any capture): create the own attn pacer stream."""
    if R.active["stream"]:
        _side_stream4()


# ----------------------------------------------------------------------------- planning (pure) ------
def _owner_windows(events) -> dict:
    """wq_b ordinal -> index of the collective whose window holds that wq_b (the last collective before it)."""
    owner, cur = {}, None
    for ev in events:
        if ev[0] == "c":
            cur = ev[1]
        elif ev[0] == "L" and cur is not None:
            owner[ev[1]] = cur
    return owner


def subtract(segs, covered) -> tuple[list, int]:
    """segs [(ptr, take, name)] minus the union of covered [(ptr, take, ...)] byte ranges, order kept, every
    piece 16 B aligned (start rounded up, length rounded down) and >= 16 B. Returns (pieces, bytes removed)."""
    iv = sorted((int(c[0]), int(c[0]) + int(c[1])) for c in covered if int(c[1]) > 0)
    merged = []
    for a, b in iv:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    out, removed = [], 0
    for ptr, take, name in segs:
        lo, hi = int(ptr), int(ptr) + int(take)
        pieces, x = [], lo
        for a, b in merged:
            if b <= x or a >= hi:
                continue
            if a > x:
                pieces.append((x, a))
            x = max(x, b)
            if x >= hi:
                break
        if x < hi:
            pieces.append((x, hi))
        kept = 0
        for a, b in pieces:
            a2 = (a + 15) & ~15
            n = (b - a2) & ~15
            if n >= 16:
                out.append((a2, n, name))
                kept += n
        removed += int(take) - kept
    return out, removed


def dedup_lists(fl: dict, events) -> tuple[dict, int]:
    """Pure: fl (g2_fork_lists result) with every attention window minus the bytes of its owner window's pre/post
    parts. Returns (new fl, bytes removed); fl itself is not modified."""
    owner = _owner_windows(events)
    attn, removed = {}, 0
    for lin, segs in fl["attn"].items():
        idx = owner.get(lin)
        cov = list(fl["pre"].get(idx, ())) + list(fl["post"].get(idx, ())) if idx is not None else []
        if cov:
            new, rm = subtract(segs, cov)
        else:
            new, rm = list(segs), 0
        if new:
            attn[lin] = new
        removed += rm
    out = dict(fl)
    out["attn"] = attn
    moved = collections.Counter(fl.get("moved", {}))
    if removed:
        moved["xsr_dedup"] += removed
    out["moved"] = dict(moved)
    return out, removed


def retime_lists(fl: dict, events) -> dict:
    """Base make_g2_plan hook (target plans): DEDUP. Fails closed (lever off, base lists kept)."""
    if not R.active["dedup"]:
        return fl
    try:
        new, removed = dedup_lists(fl, events)
    except Exception as e:  # noqa: BLE001 - a hint plan must never break the launch
        R.active["dedup"] = False
        logger.warning("DSV41_L2_PREFETCH r4-xsretime: dedup planning failed (%r); dedup off, base plan kept", e)
        return fl
    R.stats["dedup_bytes"] = removed
    return new


def attn_parts(segs, device):
    """Base _parts hook for fork_kind "attn": None = not ours (the base builds the parts as before). Otherwise the
    base burst head (PACE_HEAD_KB, mode "b", base side stream 2) and the paced tail as a 4-tuple "y" part that
    fork_part launches at the attn rate on stream 3 or the own stream."""
    if not (R.active["rate"] or R.active["stream"]):
        return None
    S = B.S
    if not (B.PACE_GBPS > 0 and S.pace_ok and "attn" in B.PACE_KINDS):
        return None  # pacer library missing: the base falls back to a burst exactly as without the kit
    if B.PACE_HEAD > 0:
        head, tail = B._cap_named(segs, B.PACE_HEAD)
    else:
        head, tail = [], list(segs)
    out = []
    for mode, ss in (("b", head), ("y", tail)):
        if not ss:
            continue
        t = torch.tensor([v for p, tk, _n in ss for v in (p, tk)], dtype=torch.int64, device=device)
        S.keepalive.append(t)  # a captured graph keeps reading this table
        out.append((mode, t, len(ss), sum(tk for _p, tk, _n in ss)))
    return out


def rate() -> float:
    return GBPS_ATTN if R.active["rate"] else B.PACE_GBPS


# ----------------------------------------------------------------------------- execution -------------
def fork_part(part, where: str) -> None:
    """Launch a "y" part: the live pacer at the attn rate, on the own stream (joined by join()) or on base stream 3
    (joined by the base _join3), both at the end of the bracketed forward. Same fork protocol as the base:
    side.wait_stream(main) right after the last wq_b."""
    _m, t, nseg, _nb = part
    S = B.S
    main = torch.cuda.current_stream()
    own = R.active["stream"]
    side = _side_stream4() if own else B._side_stream3()
    side.wait_stream(main)
    period_ps = int(round(B.PACE_CHUNK * 1000.0 / rate()))
    rc = B._load_lib_g2().dsv41_l2pf_paced(t.data_ptr(), nseg, B.PACE_CHUNK, period_ps, B.PACE_DELAY_NS,
                                           B.PACE_MAX_NS, None, side.cuda_stream)
    if rc:
        raise RuntimeError(f"DSV41_L2_PREFETCH r4-xsretime: paced prefetch launch failed (cuda error {rc})")
    if own:
        R.pending4 = side
    else:
        S.pending3 = side
    S.g2_forks[f"{where}_y"] += 1


def join() -> None:
    """Base _bracketed finally hook: join the own attn pacer stream into the main stream (end of the forward)."""
    p = R.pending4
    if p is not None:
        torch.cuda.current_stream().wait_stream(p)
        R.pending4 = None
