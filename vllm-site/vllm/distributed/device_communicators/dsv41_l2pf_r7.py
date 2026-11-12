# SPDX-License-Identifier: Apache-2.0
"""DSv4.1 climb round 7 (r7-l2pf): two default-off L2-prefetch planner rules on top of dsv41_l2pf_r1.py.

Imported by dsv41_l2pf_r1.py (r1) only when one of the two master flags below is set; r1 itself is imported by the
base (dsv41_l2_prefetch.py) only when DSV41_L2_PREFETCH=1 and an r1 master flag is set (production 166d7aac: R1_KVSRC=1,
R1_SHW2=1). With both flags unset this file is never imported and r1 runs exactly as before: every r1 edit is an
`if _R7 is not None:` branch plus one module-level flag test. Both rules only add or move
`cp.async.bulk.prefetch.L2` hints issued by the live paced kernel (libdsv41_l2pf_g2.so) on the base's side streams,
joined exactly like the base's parts: no main-stream kernel, input, order, tile or reduction changes, so every output
bit is unchanged by construction.

Master flags (read once at import; "1"/"on"/"true" = on):
  DSV41_L2_PREFETCH_R7_HT10      HT10: the KV-source compressor weight pre-stage (r1 G11a, R1_KVSRC) re-planned.
        The compressor (attention.compressor.fused_wkv_wgate, a replicated bf16 [N, 5120] weight read by a bare
        torch.mm on aux stream 0: 10 MiB in layers 2/8/14, 5 MiB in layer 20) is DRAM-bound and delays the qkv_a
        skinny GEMM and all-gather by about its own duration (t2). R1_KVSRC stages its first 2.5 MiB in the nearest
        earlier ALL-REDUCE window. HT10 changes two things, and only for a compressor whose nearest collective
        window is the Engram kv all-gather issued right after the same layer's Engram wkv GEMM:
          window: that all-gather (kind 'ag', paced by DSV41_L2_PREFETCH_PACE_KINDS, with the layer's engram.wkv
                  weight record in the window that ends at it). In production that is layer 14 only (the Engram
                  host): its bytes move from the layer-13 post-MoE all-reduce ~140 us earlier, across the Engram wkv
                  GEMM (19.7 MB streamed), to the kv all-gather right after that GEMM.
                  Every other record keeps r1's rule unchanged (nearest all-reduce, R1_KVSRC_MB), byte-identical
                  to stock r1: layers 2/8/20 (their nearest window is an all-reduce), and layer 14 whenever the
                  measured shape is not there -- r6-step DSV41_STEP_ENGRAM_PACK=all (layer 14's Engram all-gathers
                  are gone, the nearest window is the layer-13 post-MoE all-reduce), r5-prio
                  DSV41_PRIO_ENGRAM_HOIST (the wkv GEMM runs on a side stream at layer 13 through
                  quant_method.apply: no wkv record before the all-gather), or unpaced all-gathers.
          bytes:  per layer from DSV41_L2_PREFETCH_R7_HT10_MB, a map "layer:MiB,...,*:MiB" (default "14:6.5"; '*'
                  = R1_KVSRC_MB unless given), applied ONLY to 'near' records, never more than the weight. 6.5 MiB
                  = 54.5 us at 125 GB/s: the t2 layer-14 idle pacer time before the compressor starts is
                  51.7-61.8 us (p10, 6 traces). To resize layers 2/8/20 use r1's own R1_KVSRC_MB.
        The part is exactly an R1_KVSRC part (paced at the base rate, side stream 3, appended after the window's
        base part, joined at the end of the forward); only its window and bytes change. Needs R1_KVSRC active.
  DSV41_L2_PREFETCH_R7_CERTPRE   CERTPRE: the CERT head tables of dsv41_r4_small (CertState: q int8 [V, K], e uint8
        [V, K/32], nv fp32 [V], rt fp32 [K/group, V]) for the first R rows, staged in L2 during the last collective
        of the target forward (the final layer's post-MoE all-reduce; t2: no prefetch in that window today, 28-38 us
        AR + ~14 us of tail kernels before the graph ends, then ~25-30 us to the screen). R = CERTPRE_MB /
        bytes-per-row, rounded down to 64 rows (one screen CTA). Issue order: nv, rt, then q/e in 256-row blocks
        (the screen's CTAs read rows in pid order, so the prefix is consumed by its first wave). One paced kernel
        on side stream CERTPRE_STREAM, forked right before that collective and joined at the end of the forward;
        its issue window is capped at CERTPRE_MAX_US after its start, so the forward-end join does not wait on it
        when the in-graph tail is at least that long (t2 p10: 31.6-42.9 us).
      DSV41_L2_PREFETCH_R7_CERTPRE_MB=3.5  _GBPS=0 (= base PACE_GBPS)  _MAX_US=32  _STREAM=1 (1/2/3)
        Needs the base pacer library (PACE_GBPS>0), and an active r1 target lever (KVSRC / SHW2 / IDXTAIL):
        augment_target_g2 only runs then. The CertState is taken from dsv41_r4_small._cert_state (wrapped at install;
        target_logits looks it up as a module global at call time), so a plan learned before the first eager
        target_logits carries no CERTPRE part (logged).

Fail closed: an invalid knob makes its lever INACTIVE at bind (logged; the r1 plan is untouched). A failure inside this
module's planning switches only the r7 lever off and returns the stock r1 result; it never raises into r1's planning
(which would switch the promoted r1 levers off).
"""
from __future__ import annotations

import json
import os
import re
import sys

import torch

FLAGS = ("DSV41_L2_PREFETCH_R7_HT10", "DSV41_L2_PREFETCH_R7_CERTPRE")
TAG = "r7-l2pf"
R4_MODULE = "vllm.models.deepseek_v41.nvidia.dsv41_r4_small"
CERT_ROW_ALIGN = 64  # rows of one screen CTA (dsv41_r4_small SBN; t2 grid 253 = ceil(16160 / 64))
CERT_BLOCK = 256  # q/e rows per issued block


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def _on(name: str) -> bool:
    return _env(name, "0").lower() in ("1", "on", "true")


def _num(name: str, default: str):
    """A numeric knob; None when it does not parse (bind rejects it: INACTIVE, never an import error)."""
    try:
        v = float(_env(name, default) or default)
    except (ValueError, OverflowError):
        return None
    return v if v == v and abs(v) != float("inf") else None


HT10 = _on("DSV41_L2_PREFETCH_R7_HT10")
HT10_MAP_RAW = _env("DSV41_L2_PREFETCH_R7_HT10_MB", "14:6.5")
CERTPRE = _on("DSV41_L2_PREFETCH_R7_CERTPRE")
CERT_MB = _num("DSV41_L2_PREFETCH_R7_CERTPRE_MB", "3.5")
CERT_GBPS = _num("DSV41_L2_PREFETCH_R7_CERTPRE_GBPS", "0")
CERT_MAX_US = _num("DSV41_L2_PREFETCH_R7_CERTPRE_MAX_US", "32")
CERT_STREAM = _num("DSV41_L2_PREFETCH_R7_CERTPRE_STREAM", "1")

B = None  # the base module (dsv41_l2_prefetch), set by bind()
R1 = None  # dsv41_l2pf_r1, set by bind()
logger = None


class _State:
    def __init__(self) -> None:
        self.ht10 = False  # validated at bind
        self.certpre = False
        self.ht10_map: dict = {}
        self.cert_bytes = 0
        self.cert = None  # the CertState seen by dsv41_r4_small._cert_state
        self.logged: set = set()
        self.installed: set = set()


ST = _State()


# ----------------------------------------------------------------------------- bind / install -----
def parse_map(spec: str, default_bytes: int) -> dict | None:
    """'14:6.5,2:3.5,*:2.5' -> {14: bytes, 2: bytes, '*': bytes} (MiB, 16-B rounded, 0..64 MiB). '*' defaults to
    default_bytes (R1_KVSRC_MB). None = invalid."""
    out = {"*": int(default_bytes) & ~15}
    for part in filter(None, (p.strip() for p in (spec or "").split(","))):
        k, sep, v = part.partition(":")
        if not sep:
            return None
        k = k.strip()
        try:
            mb = float(v)
        except ValueError:
            return None
        if not 0 <= mb <= 64 or mb != mb:
            return None
        if k == "*":
            key = "*"
        elif k.isdigit():
            key = int(k)
        else:
            return None
        out[key] = int(mb * (1 << 20)) & ~15
    return out


def bind(base, r1, why: dict) -> None:
    """Called from r1.bind() (after r1 decided its own levers). Never raises: a failure here leaves the r7 levers
    off and r1 untouched."""
    global B, R1, logger
    B, R1, logger = base, r1, base.logger
    try:
        _bind(why)
    except Exception as e:  # noqa: BLE001 - never break r1's bind
        ST.ht10 = ST.certpre = False
        why["r7"] = f"bind failed ({e!r})"


def _bind(why: dict) -> None:
    a = R1.R.active
    if HT10:
        m = parse_map(HT10_MAP_RAW, R1.KVSRC_BYTES)
        if not a.get("kvsrc"):
            why["r7_ht10"] = "needs the active R1_KVSRC lever (DSV41_L2_PREFETCH_R1_KVSRC=1 with paced AR windows)"
        elif m is None:
            why["r7_ht10"] = f"DSV41_L2_PREFETCH_R7_HT10_MB={HT10_MAP_RAW!r} is not 'layer:MiB,...' with 0<=MiB<=64"
        else:
            ST.ht10, ST.ht10_map = True, m
    if CERTPRE:
        bad = []
        if not (a.get("kvsrc") or a.get("shw2") or a.get("idxtail")):
            bad.append("needs an active r1 target lever (KVSRC / SHW2 / IDXTAIL): augment_target_g2 runs only then")
        if not base_paced():
            bad.append("needs the base pacer (G2 mode with DSV41_L2_PREFETCH_PACE_GBPS>0)")
        if CERT_MB is None or not 0 < CERT_MB <= 64:
            bad.append(f"CERTPRE_MB={_env('DSV41_L2_PREFETCH_R7_CERTPRE_MB', '')!r} not in (0, 64]")
        if CERT_GBPS is None or not 0 <= CERT_GBPS <= 400:
            bad.append(f"CERTPRE_GBPS={_env('DSV41_L2_PREFETCH_R7_CERTPRE_GBPS', '')!r} not in [0, 400]")
        if CERT_MAX_US is None or not 1 <= CERT_MAX_US <= 200:
            bad.append(f"CERTPRE_MAX_US={_env('DSV41_L2_PREFETCH_R7_CERTPRE_MAX_US', '')!r} not in [1, 200]")
        if CERT_STREAM not in (1.0, 2.0, 3.0):
            bad.append(f"CERTPRE_STREAM={_env('DSV41_L2_PREFETCH_R7_CERTPRE_STREAM', '')!r} not in 1/2/3")
        if bad:
            why["r7_certpre"] = "; ".join(bad)
        else:
            ST.certpre, ST.cert_bytes = True, int(CERT_MB * (1 << 20)) & ~15
    logger.info(
        "DSV41_L2_PREFETCH_R7 (%s) levers: %s; ht10 map %s; certpre(%.2f MiB gbps=%s max=%s us stream=%s)%s", TAG,
        json.dumps({"ht10": ST.ht10, "certpre": ST.certpre}, sort_keys=True),
        json.dumps({str(k): round(v / 2**20, 3) for k, v in ST.ht10_map.items()}, sort_keys=True) if ST.ht10 else "-",
        (CERT_MB or 0.0), (CERT_GBPS or "base"), CERT_MAX_US, CERT_STREAM,
        (" INACTIVE: " + json.dumps({k: v for k, v in why.items() if k.startswith("r7")}, sort_keys=True))
        if any(k.startswith("r7") for k in why) else "")


def base_paced() -> bool:
    return bool(B.G2) and B.PACE_GBPS > 0


def install(mod) -> None:
    """Called from r1.install() (first collective with the model module loaded). Never raises."""
    if not ST.certpre or "cert" in ST.installed:
        return
    ST.installed.add("cert")
    try:
        import importlib

        r4 = sys.modules.get(R4_MODULE) or importlib.import_module(R4_MODULE)
        fn = getattr(r4, "_cert_state", None)
        if fn is None:
            raise AttributeError("dsv41_r4_small has no _cert_state")
        if getattr(fn, "_dsv41_r7", False):
            return

        def _cert_state(model):
            st = fn(model)
            if st is not None and ST.cert is not st:
                ST.cert = st
                logger.info("DSV41_L2_PREFETCH_R7_CERTPRE: CertState seen (V=%s K=%s group=%s)",
                            getattr(st, "V", "?"), getattr(st, "K", "?"), getattr(st, "group", "?"))
            return st

        _cert_state._dsv41_r7 = True  # type: ignore[attr-defined]
        _cert_state.__wrapped__ = fn  # type: ignore[attr-defined]
        r4._cert_state = _cert_state
        logger.info("DSV41_L2_PREFETCH_R7_CERTPRE: dsv41_r4_small._cert_state wrapped")
    except Exception as e:  # noqa: BLE001
        ST.certpre = False
        logger.warning("DSV41_L2_PREFETCH_R7_CERTPRE: cannot wrap %s._cert_state (%r); lever off", R4_MODULE, e)


def _once(key, msg, *args, warn=False) -> None:
    if key in ST.logged:
        return
    ST.logged.add(key)
    (logger.warning if warn else logger.info)(msg, *args)


# ----------------------------------------------------------------------------- HT10 ---------------
_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def layer_of(name: str):
    m = _LAYER.search(name or "")
    return int(m.group(1)) if m else None


def paced_kinds() -> frozenset:
    """Collective-window kinds whose base parts are paced (PACE_GBPS>0 and kind in PACE_KINDS)."""
    if not base_paced():
        return frozenset()
    return frozenset(k for k in ("ar", "ag") if k in B.PACE_KINDS)


_WKV = re.compile(r"(?:^|\.)layers\.(\d+)\.engram\.wkv$")


def wkv_windows(events) -> dict:
    """{collective idx: {layers}}: the layers whose Engram wkv weight record lies in the window that ENDS at that
    collective (between the previous collective and it), i.e. collective idx is issued right after that wkv GEMM."""
    out: dict = {}
    pending: set = set()
    for ev in events:
        if ev[0] == "c":
            if pending:
                out[ev[1]] = pending
            pending = set()
        elif ev[0] == "w":
            m = _WKV.search(str(ev[1]))
            if m:
                pending.add(int(m.group(1)))
    return out


def ht10_window(order, kinds, idx, paced, wkv=None, layer=None):
    """(window idx, rule) for a compressor record at collective idx. 'near' = the nearest window at or before idx
    when it is the Engram kv ALL-GATHER issued right after this layer's own wkv GEMM (kind 'ag', paced, and the
    layer's engram.wkv record in the window that ends at it): the t2-measured layer-14 shape the HT10 bytes are sized
    for. Anything else (an all-reduce: r6-step ENGRAM_PACK=all removes layer 14's Engram all-gathers; an all-gather
    without its wkv GEMM before it: r5-prio ENGRAM_HOIST runs the GEMM on a side stream at layer 13 via
    quant_method.apply, so no wkv record; an unpaced all-gather) -> r1's own rule ('ar': nearest all-reduce,
    R1_KVSRC_MB), so the plan is byte-identical to stock r1. (None, None) if there is no window."""
    near = None
    for i in order:
        if i > idx:
            break
        near = i
    if (near is not None and kinds.get(near) == "ag" and "ag" in paced and layer is not None
            and layer in (wkv or {}).get(near, ())):
        return near, "near"
    tgt = R1._nearest_ar(order, kinds, idx)
    return (tgt, "ar") if tgt is not None else (None, None)


def ht10_plan(events, records, cap=None) -> list:
    """Pure planning: [(window idx, rule, layer, name, segs)] per compressor record. Only 'near' records take the
    per-layer map bytes; 'ar' records are exactly r1's stock rule (window, and cap or R1_KVSRC_MB as in
    r1.kvsrc_appends)."""
    stock = R1.KVSRC_BYTES if cap is None else cap
    order = [ev[1] for ev in events if ev[0] == "c"]
    kinds = {ev[1]: ev[2] for ev in events if ev[0] == "c"}
    paced = paced_kinds()
    wkv = wkv_windows(events)
    out = []
    for idx, name, w in records:
        layer = layer_of(name)
        tgt, rule = ht10_window(order, kinds, idx, paced, wkv, layer)
        if tgt is None:
            continue
        nb = ST.ht10_map.get(layer, ST.ht10_map["*"]) if rule == "near" else stock
        segs = B._take_named(R1._span_items([w], name), nb)
        if segs:
            out.append((tgt, rule, layer, name, segs))
    return out


def ht10_appends(events, records, cap=None):
    """r1.kvsrc_appends replacement (same contract: {window idx: segs}). None = fall back to the stock r1 rule."""
    if not ST.ht10:
        return None
    try:
        plan = ht10_plan(events, records, cap)
    except Exception as e:  # noqa: BLE001 - never raise into r1's planning
        ST.ht10 = False
        logger.warning("DSV41_L2_PREFETCH_R7_HT10: planning failed (%r); HT10 off, stock R1_KVSRC kept", e)
        return None
    out: dict = {}
    for tgt, _rule, _layer, _name, segs in plan:
        out.setdefault(tgt, []).extend(segs)
    desc = tuple((layer, tgt, rule, sum(t for _p, t, _n in segs)) for tgt, rule, layer, _n, segs in plan)
    _once(("ht10", desc), "DSV41_L2_PREFETCH_R7_HT10 plan: %s",
          " ".join(f"L{layer}:{rule}{tgt}:{nb / 2**20:.2f}MiB" for layer, tgt, rule, nb in desc))
    return out


# ----------------------------------------------------------------------------- CERTPRE ------------
def _span(t):
    return R1._span(t)


def cert_segments(st, budget: int) -> list:
    """[(ptr, bytes, name)] for the first R rows of the CERT tables, R = budget // bytes-per-row rounded down to
    CERT_ROW_ALIGN (and <= V). Order: nv, rt (one segment per group row), then q and e in CERT_BLOCK-row blocks.
    Every segment is 16-B aligned and lies inside its tensor; [] when a table is not a dense 16-B aligned tensor."""
    q, e, rt, nv = (getattr(st, k, None) for k in ("q", "e", "rt", "nv"))
    V, K = int(st.V), int(st.K)
    NG = K // int(st.group)
    spans = [_span(t) for t in (q, e, rt, nv)]
    if any(s is None for s in spans):
        return []
    (qp, qn), (ep, en), (rp, rn), (np_, nn) = spans
    if qn != V * K or en != V * (K // 32) or rn != NG * V * 4 or nn != V * 4:
        return []
    per_row = K + K // 32 + 4 * NG + 4
    rows = min(V, budget // per_row)
    rows -= rows % CERT_ROW_ALIGN
    if rows <= 0 or (K // 32) % 16 or K % 16 or (V * 4) % 16:
        return []
    segs = [(np_, rows * 4, "cert_nv")]
    segs += [(rp + g * V * 4, rows * 4, "cert_rt") for g in range(NG)]
    for r0 in range(0, rows, CERT_BLOCK):
        r1 = min(rows, r0 + CERT_BLOCK)
        segs.append((qp + r0 * K, (r1 - r0) * K, "cert_q"))
        segs.append((ep + r0 * (K // 32), (r1 - r0) * (K // 32), "cert_e"))
    return [s for s in segs if s[1] >= 16 and s[0] % 16 == 0 and s[1] % 16 == 0]


def cert_part(segs, device):
    """("x", table, nseg, bytes, opts) handled by fork() (r1.fork_part dispatches opts["r7"])."""
    t = R1._table(segs, device)
    nb = sum(tk for _p, tk, _n in segs)
    opt = {"stream": int(CERT_STREAM), "gbps": float(CERT_GBPS or 0.0), "delay_ns": 0,
           "max_ns": int(CERT_MAX_US * 1000), "r7": True}
    return ("x", t, len(segs), nb, opt)


def built_parts(events, device) -> list:
    """r1.augment_target_g2 hook: [(where, idx, part, segs, stat, key)] to attach to a target plan. Never raises."""
    if not ST.certpre:
        return []
    try:
        if not bool(B.S.pace_ok):
            _once("cert_nopace", "DSV41_L2_PREFETCH_R7_CERTPRE: pacer library not loaded; no part", warn=True)
            return []
        st = ST.cert
        if st is None:
            _once("cert_nost", "DSV41_L2_PREFETCH_R7_CERTPRE: no CertState at plan time (target_logits has not built "
                  "it yet); this plan has no CERTPRE part", warn=True)
            return []
        colls = [ev for ev in events if ev[0] == "c"]
        if not colls:
            return []
        last = colls[-1]
        if last[2] != "ar":
            _once(("cert_last", last[2]), "DSV41_L2_PREFETCH_R7_CERTPRE: the last collective is %r, not an "
                  "all-reduce; no part", last[2], warn=True)
            return []
        segs = cert_segments(st, ST.cert_bytes)
        if not segs:
            _once("cert_noseg", "DSV41_L2_PREFETCH_R7_CERTPRE: CertState tables not dense/aligned; no part", warn=True)
            return []
        part = cert_part(segs, device)
        rows = sum(t for _p, t, n in segs if n == "cert_q") // int(st.K)
        _once(("cert_plan", last[1], rows), "DSV41_L2_PREFETCH_R7_CERTPRE plan: window ar%d rows %d/%d %.2f MiB in "
              "%d segments, stream %d, %s GB/s, cap %.0f us", last[1], rows, int(st.V), part[3] / 2**20, len(segs),
              int(CERT_STREAM), CERT_GBPS or B.PACE_GBPS, CERT_MAX_US)
        return [("pre", last[1], part, segs, "certpre", "c")]
    except Exception as e:  # noqa: BLE001
        ST.certpre = False
        logger.warning("DSV41_L2_PREFETCH_R7_CERTPRE: planning failed (%r); CERTPRE off", e)
        return []


def fork(part, where: str) -> None:
    """Launch an r7 "x" part: the live paced kernel at its own rate and cap on base side stream 1/2/3, with the
    base's join bookkeeping (1: joined at the next collective or the end of the forward; 2, 3: end of the forward)."""
    _m, t, nseg, _nb, opt = part
    S = B.S
    if not S.pace_ok:
        return
    st = int(opt["stream"])
    main = torch.cuda.current_stream()
    side = {1: B._side_stream, 2: B._side_stream2, 3: B._side_stream3}[st]()
    side.wait_stream(main)
    gbps = float(opt["gbps"]) or float(B.PACE_GBPS)
    period_ps = int(round(B.PACE_CHUNK * 1000.0 / gbps))
    rc = B._load_lib_g2().dsv41_l2pf_paced(t.data_ptr(), nseg, B.PACE_CHUNK, period_ps, int(opt["delay_ns"]),
                                           int(opt["max_ns"]), None, side.cuda_stream)
    if rc:
        raise RuntimeError(f"DSV41_L2_PREFETCH_R7: prefetch launch failed (cuda error {rc})")
    if st == 1:
        S.pending = side
    elif st == 2:
        S.pending2 = side
    else:
        S.pending3 = side
    S.g2_forks[f"{where}_x"] += 1
