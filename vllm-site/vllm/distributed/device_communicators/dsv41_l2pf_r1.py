# SPDX-License-Identifier: Apache-2.0
"""DSv4.1 climb round 1 (r1-pf): four default-off L2-prefetch levers on top of dsv41_l2_prefetch.py.

Imported by dsv41_l2_prefetch.py (the "base") only when DSV41_L2_PREFETCH=1 AND at least one of the four
master flags below is set; with all of them unset this file is never imported and the base runs exactly
as before (every base edit is an `if _R1 is not None:` branch). Every lever only adds
`cp.async.bulk.prefetch.L2` hint kernels (the live libdsv41_l2pf.so burst kernel or the live
libdsv41_l2pf_g2.so pacer) on side streams that are joined exactly like the base's: no main-stream kernel,
input, order, tile or reduction changes, so every output bit is unchanged by construction.

Master flags (read once at import; "1"/"on"/"true" = on):
  DSV41_L2_PREFETCH_R1_DRAFT   G13  L2 prefetch for the DSpark drafter graph (both graph sets, incl.
                                    DSV41_DSPARK_GREEDY_SHARDED) and the eager target->drafter phase.
        drafter graph: a learn/capture bracket around DSparkSpeculator._generate_draft (registered at
          runtime, no drafter file is edited); every window paced (base pacer, PACE_GBPS); the q_b (wq_b)
          bytes of every qkv_a all-gather window move to the previous collective's window (qbprev); a
          drafter WOA window (wo_a after the last wq_b); the first DRAFT_HEAD_MB of the draft head
          (MXFP8 lm-head copy) in the last model all-reduce window; no mHC fn hints.
        E-phase: right after the CERT screen/refine of the target head (dsv41_r4_small.certified_logits)
          one paced kernel on its own stream walks main_proj (the R4 column-split shard) then the
          context-KV fold weight (DSV41_CTXKV_FOLD), EPH_MB in total, eager (never captured).
      DSV41_L2_PREFETCH_R1_DRAFT_GRAPH=1  0 = E-phase only
      DSV41_L2_PREFETCH_R1_DRAFT_EPH=1    0 = drafter graph only
      DSV41_L2_PREFETCH_R1_DRAFT_QBPREV=1 0 = leave wq_b in the drafter all-gather windows (paced there)
      DSV41_L2_PREFETCH_R1_DRAFT_HEAD_MB=5   draft-head bytes in the last model AR window (0 = none)
      DSV41_L2_PREFETCH_R1_DRAFT_WOA_BURST=0 1 = drafter WOA window as a burst on the WOA stream
      DSV41_L2_PREFETCH_R1_DRAFT_GBPS=<PACE_GBPS>  drafter pacing rate
      DSV41_L2_PREFETCH_R1_EPH_MB=18  DSV41_L2_PREFETCH_R1_EPH_GBPS=<PACE_GBPS>  DSV41_L2_PREFETCH_R1_EPH_MAX_US=200
  DSV41_L2_PREFETCH_R1_IDXTAIL G29  in-order tail on the target's indexer-layer qkv_a all-gather windows: the
                                    rest of the indexer wq_b (the recorded read list right after the window's
                                    budget cut) as a SECOND burst on side stream 1 after the window's burst head,
                                    joined at the post-attention all-reduce. SWA-layer windows get nothing.
                                    Built for BURST all-gather windows only (launcher fda6d05a, PACE_KINDS=ar):
                                    REFUSED (logged INACTIVE, no plan change) when "ag" is in PACE_KINDS with
                                    PACE_GBPS > 0 (launcher 08a249da / climb-w1): there the tail would queue on
                                    the pacer stream 3 between the paced all-gather window and the promoted
                                    paced wo_a+wo_b window and delay the latter by ~36 us in every indexer layer.
      DSV41_L2_PREFETCH_R1_IDXTAIL_MB=6  DSV41_L2_PREFETCH_R1_IDXTAIL_SPLIT=2 (CTAs = segments, 1..4)
  DSV41_L2_PREFETCH_R1_SHW2    G15  shared-expert w2 (ffn.shared_experts.down_proj, 1.52 MB/layer) as a burst
                                    in a "post" fork right after the post-attention all-reduce (needs
                                    PRUNE_SHARED=1, i.e. no shared hints in the AR windows).
      DSV41_L2_PREFETCH_R1_SHW2_MB=2
  DSV41_L2_PREFETCH_R1_KVSRC   G11a the kv-source compressor weight (compressor.fused_wkv_wgate, a bare torch.mm,
                                    never recorded by the base) appended, paced, to the preceding post-MoE
                                    all-reduce window (nearest earlier "ar" window; layers 2/8/14/20).
      DSV41_L2_PREFETCH_R1_KVSRC_MB=2.5
  (G01, the in-order wo_b fallback DSV41_L2_PREFETCH_R1_WOB, was DROPPED in the review fix: launcher 08a249da
   promoted the paced wo_a+wo_b window (WOB_MB=6), which superseded it. The flag is no longer a master flag, has
   no code behind it and is only named in the bind() log line as ignored.)

Target levers (IDXTAIL, SHW2, KVSRC) act through make_plan -> augment_target_g2 and need the base's G2 mode
(PACE_GBPS > 0 or a REPLAN; production: PACE_GBPS=125). Rank-local, no collective added or removed.
A failure inside this module's planning fails closed: the lever is switched off and the base plan is kept.
"""
from __future__ import annotations

import collections
import json
import os
import sys

import torch

FLAGS = ("DSV41_L2_PREFETCH_R1_DRAFT", "DSV41_L2_PREFETCH_R1_IDXTAIL", "DSV41_L2_PREFETCH_R1_SHW2",
         "DSV41_L2_PREFETCH_R1_KVSRC")
DROPPED = ("DSV41_L2_PREFETCH_R1_WOB",)  # G01: dropped (superseded by launcher 08a249da WOB_MB=6); ignored


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def _on(name: str, default: str = "0") -> bool:
    return _env(name, default).lower() in ("1", "on", "true")


def _mb(name: str, default: str) -> int:
    return int(float(_env(name, default) or "0") * (1 << 20)) & ~15


DRAFT = _on("DSV41_L2_PREFETCH_R1_DRAFT")
DRAFT_GRAPH = DRAFT and _on("DSV41_L2_PREFETCH_R1_DRAFT_GRAPH", "1")
DRAFT_EPH = DRAFT and _on("DSV41_L2_PREFETCH_R1_DRAFT_EPH", "1")
DRAFT_QBPREV = _on("DSV41_L2_PREFETCH_R1_DRAFT_QBPREV", "1")
DRAFT_HEAD = _mb("DSV41_L2_PREFETCH_R1_DRAFT_HEAD_MB", "5")
DRAFT_WOA_BURST = _on("DSV41_L2_PREFETCH_R1_DRAFT_WOA_BURST", "0")
DRAFT_GBPS = float(_env("DSV41_L2_PREFETCH_R1_DRAFT_GBPS", "0") or "0")  # 0 = the base PACE_GBPS
EPH_BYTES = _mb("DSV41_L2_PREFETCH_R1_EPH_MB", "18")
EPH_GBPS = float(_env("DSV41_L2_PREFETCH_R1_EPH_GBPS", "0") or "0")  # 0 = the base PACE_GBPS
EPH_MAX_NS = int(float(_env("DSV41_L2_PREFETCH_R1_EPH_MAX_US", "200") or "0") * 1000)
IDXTAIL = _on("DSV41_L2_PREFETCH_R1_IDXTAIL")
IDXTAIL_BYTES = _mb("DSV41_L2_PREFETCH_R1_IDXTAIL_MB", "6")
IDXTAIL_SPLIT = max(1, min(4, int(_env("DSV41_L2_PREFETCH_R1_IDXTAIL_SPLIT", "2") or "1")))
SHW2 = _on("DSV41_L2_PREFETCH_R1_SHW2")
SHW2_BYTES = _mb("DSV41_L2_PREFETCH_R1_SHW2_MB", "2")
def _r5_num(name: str, default: str, scale: float = 1.0, integer: bool = False):  # r5-ffnfront
    """r5-ffnfront: a numeric r5 knob. A value that does not parse gives -1, which _r5_bind rejects (INACTIVE,  # r5-ffnfront
    stock r1 levers kept) instead of raising at import (that would disable every promoted r1 lever)."""  # r5-ffnfront
    try:  # r5-ffnfront
        v = float(_env(name, default) or default) * scale  # r5-ffnfront
        return int(v) if integer else v  # r5-ffnfront
    except (ValueError, OverflowError):  # r5-ffnfront
        return -1  # r5-ffnfront
SHW2_GBPS = _r5_num("DSV41_L2_PREFETCH_R1_SHW2_GBPS", "0")  # 0 = burst (stock)  # r5-ffnfront
SHW2_DELAY_NS = _r5_num("DSV41_L2_PREFETCH_R1_SHW2_DELAY_US", "0", 1000.0, True)  # r5-ffnfront
SHW2_STREAM = _r5_num("DSV41_L2_PREFETCH_R1_SHW2_STREAM", "3", 1.0, True)  # r5-ffnfront
POST_EVENT = _env("DSV41_L2_PREFETCH_R1_POST_EVENT", "").lower()  # r5-ffnfront
R5X = False  # True once bind() validated an r5 SHW2 / post-event arm  # r5-ffnfront
KVSRC = _on("DSV41_L2_PREFETCH_R1_KVSRC")
KVSRC_BYTES = _mb("DSV41_L2_PREFETCH_R1_KVSRC_MB", "2.5")

SPEC_MODULE = "vllm.v1.worker.gpu.spec_decode.dspark.speculator"
R4_MODULE = "vllm.models.deepseek_v41.nvidia.dsv41_r4_small"
ATTN_MODULE = "vllm.models.deepseek_v41.attention"
DRAFT_TAG = "r1-dspark"
TAG = "r1-pf"

B = None  # the base module (dsv41_l2_prefetch), set by bind()
ALLOW_CPU = False  # CPU unit tests only: accept CPU tensors as prefetch targets
logger = None


class _R:
    def __init__(self) -> None:
        # side-channel records of the current learn forward: kind -> (events list object, [records])
        self.rec: dict[str, tuple] = {}
        self.spec = None  # the DSparkSpeculator (set by the drafter bracket)
        self.eph_stream: dict[int, torch.cuda.Stream] = {}
        self.eph_key = None
        self.eph_tab = None
        self.eph_off = False
        self.eph_fired = 0
        self.eph_calls = 0
        self.installed: set[str] = set()
        self.active = {"draft_graph": False, "draft_eph": False, "idxtail": False, "shw2": False,
                       "kvsrc": False}
        self.stats = collections.Counter()
        self.deferred: list = []  # [(fork-point event, part, where)] waiting for their flush point  # r5-ffnfront
        self.r5_exc = None  # a forward-end flush failure, re-raised after the bracket's cleanup  # r5-ffnfront


R = _R()
_R7 = None  # dsv41_l2pf_r7 (HT10 / CERTPRE), imported only when one of its master flags is set  # r7-l2pf
_R7_ERR = None  # r7-l2pf
if any(_on(_k7) for _k7 in ("DSV41_L2_PREFETCH_R7_HT10", "DSV41_L2_PREFETCH_R7_CERTPRE")):  # r7-l2pf
    try:  # r7-l2pf
        from vllm.distributed.device_communicators import dsv41_l2pf_r7 as _R7  # r7-l2pf
    except Exception as _e7:  # noqa: BLE001  # r7-l2pf
        _R7, _R7_ERR = None, _e7  # r7-l2pf


# ----------------------------------------------------------------------------- bind / install -----
def bind(base) -> None:
    """Called once by the base at the end of its import. Decides which levers can run with the base config."""
    global B, logger
    B = base
    logger = base.logger
    why = {}
    g2 = bool(base.G2)
    a = R.active
    if DRAFT_GRAPH:
        a["draft_graph"] = True
    if DRAFT_EPH:
        a["draft_eph"] = True
    if IDXTAIL:
        if not g2:
            why["idxtail"] = "needs the base G2 mode (PACE_GBPS>0 or REPLAN)"
        elif "ag" in base.PACE_KINDS and base.PACE_GBPS > 0:
            why["idxtail"] = ("built for burst all-gather windows; with 'ag' in PACE_KINDS (launcher 08a249da) the "
                              "tail would queue on pacer stream 3 ahead of the paced wo_a+wo_b window")
        elif base.PRUNE_SHARED_POST_BUDGET:
            why["idxtail"] = "not built for PRUNE_SHARED_POST_BUDGET"
        else:
            a["idxtail"] = True
    if SHW2:
        if not g2:
            why["shw2"] = "needs the base G2 mode (after_collective post forks)"
        elif not base.PRUNE_SHARED:
            why["shw2"] = "needs DSV41_L2_PREFETCH_PRUNE_SHARED=1 (shared hints would be prefetched twice)"
        else:
            a["shw2"] = True
    if KVSRC:
        if not g2:
            why["kvsrc"] = "needs the base G2 mode"
        elif "ar" not in base.PACE_KINDS or base.PACE_GBPS <= 0:
            why["kvsrc"] = "needs paced all-reduce windows (a burst would stretch the post-MoE AR)"
        else:
            a["kvsrc"] = True
    if SHW2_GBPS != 0 or SHW2_DELAY_NS != 0 or SHW2_STREAM != 3 or POST_EVENT:  # r5-ffnfront
        _r5_bind(base, why)  # r5-ffnfront
    if _R7_ERR is not None:  # r7-l2pf
        why["r7"] = f"dsv41_l2pf_r7 not importable ({_R7_ERR!r})"  # r7-l2pf
    elif _R7 is not None:  # r7-l2pf
        _R7.bind(base, sys.modules[__name__], why)  # r7-l2pf
    for k in DROPPED:
        if _on(k):
            why[k[len("DSV41_L2_PREFETCH_R1_"):].lower()] = "dropped lever (G01); flag ignored"
    logger.info(
        "DSV41_L2_PREFETCH_R1 (%s) levers: %s; draft(qbprev=%s head=%.1f MB woa_burst=%s gbps=%s) "
        "eph(%.1f MB gbps=%s max=%.0f us) idxtail(%.1f MB x%d) shw2(%.1f MB) kvsrc(%.1f MB)%s",
        TAG, json.dumps(a, sort_keys=True), DRAFT_QBPREV, DRAFT_HEAD / 2**20, DRAFT_WOA_BURST,
        DRAFT_GBPS or "base", EPH_BYTES / 2**20, EPH_GBPS or "base", EPH_MAX_NS / 1e3,
        IDXTAIL_BYTES / 2**20, IDXTAIL_SPLIT, SHW2_BYTES / 2**20, KVSRC_BYTES / 2**20,
        (" INACTIVE: " + json.dumps(why, sort_keys=True)) if why else "")


def install(mod) -> None:
    """Called from the base install_model_module (first collective with the model module loaded)."""
    if _R7 is not None:  # r7-l2pf
        _R7.install(mod)  # r7-l2pf
    if R.active["kvsrc"]:
        _wrap_kvsrc()
    if R.active["draft_graph"] or R.active["draft_eph"]:
        _wrap_dspark()
    if R.active["draft_eph"]:
        _wrap_cert()
    if R5X and POST_EVENT:  # r5-ffnfront
        _r5_install(mod)  # r5-ffnfront


def _import(name: str):
    m = sys.modules.get(name)
    if m is not None:
        return m
    import importlib

    return importlib.import_module(name)


def _wrap_kvsrc() -> None:
    """G11a: record the compressor weight read (a bare torch.mm) at the entry of the attention's parallel input
    projections, i.e. before this layer's qkv_a all-gather: S.idx then names the preceding collective."""
    if "kvsrc" in R.installed:
        return
    R.installed.add("kvsrc")
    try:
        cls = _import(ATTN_MODULE).DeepseekV4Attention
    except Exception as e:  # noqa: BLE001
        R.active["kvsrc"] = False
        logger.warning("DSV41_L2_PREFETCH_R1_KVSRC: %s not importable (%r); lever off", ATTN_MODULE, e)
        return
    fn = cls.__dict__.get("_run_parallel_input_projections")
    if fn is None or getattr(fn, "_dsv41_r1", False):
        return

    def _run_parallel_input_projections(self, hidden_states, *a, **k):
        S = B.S
        if S.mode == "learn" and S.events is not None and not _is_draft(S.sig):
            comp = getattr(self, "compressor", None)
            lin = getattr(comp, "fused_wkv_wgate", None) if comp is not None else None
            w = getattr(lin, "weight", None)
            if isinstance(w, torch.Tensor):
                name = str(getattr(lin, "prefix", "") or (str(getattr(self, "prefix", "attn")) + ".compressor"))
                _rec("cmp", S.events, (S.idx, name, w))
        return fn(self, hidden_states, *a, **k)

    _run_parallel_input_projections._dsv41_r1 = True  # type: ignore[attr-defined]
    _run_parallel_input_projections.__wrapped__ = fn  # type: ignore[attr-defined]
    cls._run_parallel_input_projections = _run_parallel_input_projections
    logger.info("DSV41_L2_PREFETCH_R1_KVSRC: compressor weight reads recorded")


def _wrap_dspark() -> None:
    """G13: bracket DSparkSpeculator._generate_draft (both graph sets go through it) and note the draft-head read
    at _sample_sequential entry. Runtime registration only; the drafter files are not edited."""
    if "dspark" in R.installed:
        return
    R.installed.add("dspark")
    try:
        cls = _import(SPEC_MODULE).DSparkSpeculator
    except Exception as e:  # noqa: BLE001
        R.active["draft_graph"] = R.active["draft_eph"] = False
        logger.warning("DSV41_L2_PREFETCH_R1_DRAFT: %s not importable (%r); lever off", SPEC_MODULE, e)
        return
    gen = cls.__dict__.get("_generate_draft")
    if gen is not None and not getattr(gen, "_dsv41_r1", False):

        def _generate_draft(self, *args, **kwargs):
            return _draft_bracketed(gen, self, args, kwargs)

        _generate_draft._dsv41_r1 = True  # type: ignore[attr-defined]
        _generate_draft.__wrapped__ = gen  # type: ignore[attr-defined]
        cls._generate_draft = _generate_draft
    ss = cls.__dict__.get("_sample_sequential")
    if ss is not None and not getattr(ss, "_dsv41_r1", False):

        def _sample_sequential(self, *args, **kwargs):
            S = B.S
            if S.mode == "learn" and S.events is not None and _is_draft(S.sig):
                _rec("head", S.events, (S.idx, self))
            return ss(self, *args, **kwargs)

        _sample_sequential._dsv41_r1 = True  # type: ignore[attr-defined]
        _sample_sequential.__wrapped__ = ss  # type: ignore[attr-defined]
        cls._sample_sequential = _sample_sequential
    logger.info("DSV41_L2_PREFETCH_R1_DRAFT: DSparkSpeculator._generate_draft bracketed (graph=%s eph=%s)",
                R.active["draft_graph"], R.active["draft_eph"])


def _wrap_cert() -> None:
    """G13 E-phase: fork the eager paced prefetch right after the CERT screen + refine are enqueued.
    dsv41_r4_small.target_logits looks certified_logits up as a module global at call time."""
    if "cert" in R.installed:
        return
    R.installed.add("cert")
    try:
        r4 = _import(R4_MODULE)
    except Exception as e:  # noqa: BLE001
        R.active["draft_eph"] = False
        logger.warning("DSV41_L2_PREFETCH_R1_DRAFT: %s not importable (%r); E-phase off", R4_MODULE, e)
        return
    fn = getattr(r4, "certified_logits", None)
    if fn is None or getattr(fn, "_dsv41_r1", False):
        return

    def certified_logits(*args, **kwargs):
        out = fn(*args, **kwargs)
        if R.active["draft_eph"] and not R.eph_off:
            try:
                eph_fire()
            except Exception as e:  # noqa: BLE001 - a hint must never break the step
                R.eph_off = True
                logger.warning("DSV41_L2_PREFETCH_R1_DRAFT E-phase disabled: %r", e)
        return out

    certified_logits._dsv41_r1 = True  # type: ignore[attr-defined]
    certified_logits.__wrapped__ = fn  # type: ignore[attr-defined]
    r4.certified_logits = certified_logits
    logger.info("DSV41_L2_PREFETCH_R1_DRAFT: E-phase fork after the CERT screen installed")


def _rec(kind: str, events, rec) -> None:
    """Side-channel record tied to one learn forward's event list (never written into the event list, so the
    base windows are unchanged)."""
    cur = R.rec.get(kind)
    if cur is None or cur[0] is not events:
        cur = R.rec[kind] = (events, [])
    cur[1].append(rec)


def _recs(kind: str, events) -> list:
    cur = R.rec.get(kind)
    if cur is None or cur[0] is not events:
        return []
    R.rec.pop(kind, None)
    return cur[1]


def _is_draft(sig) -> bool:
    return isinstance(sig, tuple) and len(sig) > 0 and sig[0] == DRAFT_TAG


# ----------------------------------------------------------------------------- pure planning ------
DRAFT_ITEM_MAX = 32 << 20  # drafter plans skip any single read larger than this (embedding tables)


def raw_windows(events, prune: bool = True, drop_mhc: bool = False, drafter: bool = False) -> list:
    """[(idx, kind, [(ptr, nbytes, name)])] per collective window in forward order. With prune=True this is the
    item list make_plan selects from (SKIP_N, PRUNE_ROUTER/SHARED, PRUNE_STALE_QKV applied). drafter=True also drops
    the Markov head (markov_w1 is an nn.Embedding: a lookup reads a few rows of a 63 MiB table) and any read
    larger than DRAFT_ITEM_MAX."""
    out, cur = [], None
    skip_n = B.SKIP_N
    drop = prune and (B.PRUNE_ROUTER or B.PRUNE_SHARED)
    for ev in events:
        k = ev[0]
        if k == "c":
            cur = (ev[1], ev[2], [])
            out.append(cur)
        elif k in ("L", "woa"):
            continue
        elif cur is not None:
            if prune and B.PRUNE_STALE_QKV and k == "prune_stale_qkv":
                continue
            name = ev[1]
            if drop and B._drop_weight(name):
                continue
            if drop_mhc and B._weight_category(name) == "mhc":
                continue
            if drafter and "markov_head" in name:
                continue
            for it in ev[2]:
                if skip_n and it[2] is not None and it[2] <= skip_n:
                    continue
                if drafter and it[1] > DRAFT_ITEM_MAX:
                    continue
                cur[2].append((it[0], it[1], name))
    return out


def after_budget(items, budget: int) -> list:
    """The read list left after make_plan's per-window selection of `budget` bytes (same rounding)."""
    left = budget
    for i, (ptr, nbytes, name) in enumerate(items):
        if left < 16:
            return list(items[i:])
        ran_out = nbytes > left  # the budget ends inside this item (a sub-16-B tail of a fully read item is not)
        take = min(nbytes, left) & ~15
        if take <= 0:
            continue
        left -= take
        if ran_out:
            rest = [(ptr + take, nbytes - take, name)] if nbytes - take >= 16 else []
            return rest + list(items[i + 1:])
    return []


def split_segs(segs, n: int) -> list:
    """Split every (ptr, take, name) into up to n 16-B aligned pieces (the burst kernel runs one CTA per segment)."""
    if n <= 1:
        return list(segs)
    out = []
    for ptr, take, name in segs:
        step = ((take + n - 1) // n + 15) & ~15
        off = 0
        while off < take:
            t = min(step, take - off) & ~15
            if t <= 0:
                break
            out.append((ptr + off, t, name))
            off += t
    return out


IDXTAIL_LEAD_MAX = 1 << 20  # at most this many bytes of other reads (indexer wk / k_norm ...) ahead of its wq_b


def idx_tails(events, budget: int | None = None, tail: int | None = None, split: int | None = None) -> dict:
    """G29: {ag window idx: tail segs}. The tail is the recorded read list right after the window's budget cut,
    through the end of the (first) indexer wq_b in it (the base selection already holds its first bytes), capped at
    `tail` bytes and split into `split` segments. Windows with no indexer wq_b after the cut (SWA layers: wo_b
    follows) get none; so do windows with more than IDXTAIL_LEAD_MAX bytes of other reads ahead of it."""
    budget = B.BUDGET if budget is None else budget
    tail = IDXTAIL_BYTES if tail is None else tail
    split = IDXTAIL_SPLIT if split is None else split
    out = {}
    for idx, kind, items in raw_windows(events):
        if kind != "ag":
            continue
        rest = after_budget(items, budget)
        first = next((i for i, it in enumerate(rest) if it[2].endswith("indexer.wq_b")), None)
        if first is None:
            continue
        end = first
        while end + 1 < len(rest) and rest[end + 1][2] == rest[first][2]:
            end += 1
        span = rest[:end + 1]
        if sum(nb for _p, nb, n in span[:first]) > IDXTAIL_LEAD_MAX:
            continue
        segs = B._take_named(span, tail)
        if segs:
            out[idx] = split_segs(segs, split)
    return out


def shw2_posts(events, cap: int | None = None) -> dict:
    """G15: {ar window idx: shared w2 segs} for every all-reduce window whose raw read list holds the
    shared-expert down_proj (scales first, then codes; the base hooks list a module's tensors small first)."""
    cap = SHW2_BYTES if cap is None else cap
    out = {}
    for idx, kind, items in raw_windows(events, prune=False):
        if kind != "ar":
            continue
        w2 = [it for it in items if it[2].endswith("shared_experts.down_proj")]
        segs = B._take_named(w2, cap)
        if segs:
            out[idx] = segs
    return out


def _nearest_ar(order, kinds, idx):
    best = None
    for i in order:
        if i > idx:
            break
        if kinds.get(i) == "ar":
            best = i
    return best


def kvsrc_appends(events, records, cap: int | None = None) -> dict:
    """G11a: {ar window idx: compressor segs}. records = [(idx at the projections' entry, name, weight)]; the bytes go
    to the nearest all-reduce window at or before that collective (layer 14: across its Engram all-gather)."""
    if _R7 is not None:  # r7-l2pf
        _r7 = _R7.ht10_appends(events, records, cap)  # r7-l2pf
        if _r7 is not None:  # r7-l2pf
            return _r7  # r7-l2pf
    cap = KVSRC_BYTES if cap is None else cap
    order = [ev[1] for ev in events if ev[0] == "c"]
    kinds = {ev[1]: ev[2] for ev in events if ev[0] == "c"}
    out: dict[int, list] = {}
    for idx, name, w in records:
        tgt = _nearest_ar(order, kinds, idx)
        if tgt is None:
            continue
        items = _span_items([w], name)
        segs = B._take_named(items, cap)
        if segs:
            out.setdefault(tgt, []).extend(segs)
    return out


def _span(t: torch.Tensor):
    """(ptr, nbytes) of a dense tensor or of the transpose view of one (w.t() GEMM operands)."""
    if not isinstance(t, torch.Tensor) or (not t.is_cuda and not ALLOW_CPU):
        return None
    if not t.is_contiguous():
        if t.dim() == 2 and t.t().is_contiguous():
            t = t.t()
        else:
            return None
    ptr = t.data_ptr()
    if ptr % 16:
        return None
    return ptr, t.numel() * t.element_size()


def _span_items(tensors, name: str) -> list:
    out = []
    for t in tensors:
        s = _span(t)
        if s is not None and s[1] >= 16:
            out.append((s[0], s[1], name))
    return out


def head_items(spec) -> list:
    """The draft head the DSpark logits read: the DSV41_FP8_LMHEAD MXFP8 copy (scales, codes) when built, else the
    bf16 lm_head shard."""
    lm_head = getattr(getattr(spec, "model", None), "lm_head", None)
    if lm_head is None:
        return []
    fh = getattr(lm_head, "_dsv41_fp8_head", None)
    if fh is not None and hasattr(fh, "w"):
        return _span_items([getattr(fh, "s_swz", None), fh.w], "draft_head")
    w = getattr(lm_head, "weight", None)
    return _span_items([w], "draft_head")


def draft_fork_lists(events, n_coll: int, head_recs, budget: int | None = None) -> dict:
    """G13 drafter re-plan (pure): base window selection without mHC fn hints, qbprev, the drafter WOA window and
    the draft-head bytes in the last model all-reduce window. Returns base g2_fork_lists' dict + "head"."""
    budget = B.BUDGET if budget is None else budget
    sel = [(idx, kind, B._take_named(items, budget))
           for idx, kind, items in raw_windows(events, drop_mhc=True, drafter=True)]
    fl = B.g2_fork_lists(events, sel, replan=frozenset({"qbprev"}) if DRAFT_QBPREV else frozenset(),
                         replan_budget=B.REPLAN_BUDGET, woa=B.WOA)
    order = [ev[1] for ev in events if ev[0] == "c"]
    kinds = {ev[1]: ev[2] for ev in events if ev[0] == "c"}
    hb = 0
    if DRAFT_HEAD > 0 and head_recs:
        idx, items = head_recs[0]
        tgt = _nearest_ar(order, kinds, idx)
        segs = B._take_named(items, DRAFT_HEAD)
        if tgt is not None and segs:
            fl["pre"].setdefault(tgt, []).extend(segs)
            hb = sum(t for _p, t, _n in segs)
    fl["head"] = hb
    return fl


# ----------------------------------------------------------------------------- parts / forks ------
def _table(segs, device) -> torch.Tensor:
    t = torch.tensor([v for p, tk, _n in segs for v in (p, tk)], dtype=torch.int64, device=device)
    B.S.keepalive.append(t)  # a captured graph keeps reading this table
    return t


def _part(mode: str, segs, device, stream: int = 0, gbps: float = 0.0):
    """("b"|"p", table, nseg, bytes) for the base _fork_part, or ("x", table, nseg, bytes, opts) for fork_part."""
    t = _table(segs, device)
    nb = sum(tk for _p, tk, _n in segs)
    if mode == "x":
        return ("x", t, len(segs), nb, {"stream": stream, "gbps": float(gbps)})
    return (mode, t, len(segs), nb)


def _paced_ok() -> bool:
    return B.PACE_GBPS > 0 and bool(B.S.pace_ok)


def fork_part(part, where: str) -> None:
    """Launch an "x" part: burst (gbps 0) or paced at its own rate, on base side stream 1/2/3, with the base's join
    bookkeeping (1: joined at the next collective; 2, 3: joined at the end of the bracketed forward)."""
    _m, t, nseg, _nb, opt = part
    if opt.get("r7"):  # r7-l2pf
        return _R7.fork(part, where)  # r7-l2pf
    if opt.get("r5"):  # r5-ffnfront
        return _r5_fork(part, where)  # r5-ffnfront
    S = B.S
    st, gbps = opt["stream"], opt["gbps"]
    main = torch.cuda.current_stream()
    side = {1: B._side_stream, 2: B._side_stream2, 3: B._side_stream3}[st]()
    side.wait_stream(main)
    if gbps > 0 and S.pace_ok:
        period_ps = int(round(B.PACE_CHUNK * 1000.0 / gbps))
        rc = B._load_lib_g2().dsv41_l2pf_paced(t.data_ptr(), nseg, B.PACE_CHUNK, period_ps, 0, B.PACE_MAX_NS,
                                               None, side.cuda_stream)
    else:
        rc = B._load_lib().dsv41_l2pf(t.data_ptr(), nseg, B.CHUNK, side.cuda_stream)
    if rc:
        raise RuntimeError(f"DSV41_L2_PREFETCH_R1: prefetch launch failed (cuda error {rc})")
    if st == 1:
        S.pending = side
    elif st == 2:
        S.pending2 = side
    else:
        S.pending3 = side
    S.g2_forks[f"{where}_x"] += 1


def _recount(g2: dict, extra: dict) -> None:
    nb, nf = collections.Counter(), collections.Counter()
    for where in ("pre", "post", "attn"):
        for parts in g2[where].values():
            for part in parts:
                nb[f"{where}_{part[0]}"] += part[3]
                nf[f"{where}_{part[0]}"] += 1
    g2["bytes"] = sum(nb.values())
    g2["nforks"] = json.dumps(dict(sorted(nf.items())))
    g2["summary"] = (g2.get("summary", "") + " r1=" +
                     json.dumps({k: round(v / 2**20, 2) if k.endswith("_mb") else v for k, v in sorted(extra.items())},
                                sort_keys=True))


def _target_adds(events) -> list:
    """(where, idx, mode, segs, stat, key) for every target part G29 / G15 / G11a add (pure planning)."""
    a = R.active
    adds = []
    if a["idxtail"]:
        # burst only: bind() refuses G29 when the all-gather windows are paced ("ag" in PACE_KINDS), so the tail
        # is the second burst on side stream 1 right after the window's burst head, joined at the next collective
        for idx, segs in sorted(idx_tails(events).items()):
            adds.append(("pre", idx, "b", segs, "idxtail", "t"))
    if a["shw2"]:
        for idx, segs in sorted(shw2_posts(events).items()):
            adds.append(("post", idx, "b", segs, "shw2", "s"))
    if a["kvsrc"]:
        recs = _recs("cmp", events)
        if _paced_ok():
            for idx, segs in sorted(kvsrc_appends(events, recs).items()):
                adds.append(("pre", idx, "p", segs, "kvsrc", "k"))
    return adds


def augment_target_g2(g2: dict, events, sel, device) -> dict:
    """Base make_plan hook (target plans, G2 mode): add the G29 / G15 / G11a parts. Only adds parts after the
    existing ones of a fork; never changes a byte of the base parts. Fails closed: if planning raises, the target
    levers are switched off and the base plan is returned untouched (all tables are built before any is attached)."""
    a = R.active
    if not (a["idxtail"] or a["shw2"] or a["kvsrc"]):
        return g2
    try:
        adds = _target_adds(events)
        built = [(where, idx, _part(mode, segs, device), segs, stat, key)
                 for where, idx, mode, segs, stat, key in adds]
        if _R7 is not None:  # r7-l2pf
            built += _R7.built_parts(events, device)  # r7-l2pf
        if R5X:  # r5-ffnfront
            built = [(w, i, _r5_part(p) if st == "shw2" else p, sg, st, k) for w, i, p, sg, st, k in built]  # r5-ffnfront
    except Exception as e:  # noqa: BLE001 - a hint plan must never break the launch
        a["idxtail"] = a["shw2"] = a["kvsrc"] = False
        logger.warning("DSV41_L2_PREFETCH_R1: target planning failed (%r); idxtail/shw2/kvsrc off, base plan kept", e)
        return g2
    extra = collections.Counter()
    keyp = []
    for where, idx, part, segs, stat, key in built:
        g2[where].setdefault(idx, []).append(part)
        extra[f"{stat}_mb"] += sum(t for _p, t, _n in segs)
        extra[f"{stat}_n"] += 1
        keyp.append((key, idx, tuple(s[:2] for s in segs)))
    if keyp:
        g2["gkey"] = tuple(g2["gkey"]) + ((TAG, tuple(keyp), IDXTAIL_SPLIT),)
        if R5X:  # r5-ffnfront
            g2["gkey"] = tuple(g2["gkey"]) + (("r5-ffnfront", SHW2_GBPS, SHW2_DELAY_NS, SHW2_STREAM, POST_EVENT),)  # r5-ffnfront
        _recount(g2, extra)
    R.stats.update(extra)
    return g2


def make_drafter_plan(events, n_coll: int, head_recs, device="cuda") -> dict:
    """G13: the drafter's plan in the base g2 format (executed by the base before_collective / wq_b post-hook)."""
    fl = draft_fork_lists(events, n_coll, head_recs)
    rate = DRAFT_GBPS if DRAFT_GBPS > 0 else B.PACE_GBPS
    custom = DRAFT_GBPS > 0 and abs(DRAFT_GBPS - B.PACE_GBPS) > 1e-9

    def parts(segs, attn=False):
        if not segs:
            return []
        if attn and DRAFT_WOA_BURST:
            return [_part("b", segs, device)]
        if custom:
            return [_part("x", segs, device, stream=3, gbps=rate)]
        return [_part("p", segs, device)]

    g2 = {"pre": {i: parts(s) for i, s in fl["pre"].items() if s},
          "post": {i: parts(s) for i, s in fl["post"].items() if s},
          "attn": {i: parts(s, attn=True) for i, s in fl["attn"].items() if s}}
    g2["gkey"] = (DRAFT_TAG,
                  tuple(sorted((i, tuple(s[:2] for s in v)) for i, v in fl["pre"].items())),
                  tuple(sorted((i, tuple(s[:2] for s in v)) for i, v in fl["attn"].items())),
                  rate, DRAFT_QBPREV, DRAFT_WOA_BURST)
    _recount(g2, {"moved_mb": sum(fl["moved"].values()), "head_mb": fl["head"]})
    desc = []
    for idx in sorted(fl["pre"])[:3]:
        desc.append(f"[{fl['kind'].get(idx, '?')}{idx} " +
                    " ".join(f"{n}:{t / 2**20:.2f}" for _p, t, n in fl["pre"][idx]) + "]")
    for lin in sorted(fl["attn"])[:1]:
        desc.append(f"[L{lin} " + " ".join(f"{n}:{t / 2**20:.2f}" for _p, t, n in fl["attn"][lin]) + "]")
    return {"n": n_coll, "nl": fl["nl"], "win": {}, "g2": g2, "bytes": g2["bytes"], "desc": " ".join(desc),
            "key": (n_coll, g2["gkey"]), "r1": DRAFT_TAG}


# ----------------------------------------------------------------------------- drafter bracket ----
def _draft_n(args, kwargs):
    nreq = args[0] if args else kwargs.get("num_reqs", 0)
    n = args[1] if len(args) > 1 else kwargs.get("num_tokens_padded", 0)
    try:
        return int(nreq), int(n)
    except (TypeError, ValueError):
        return 0, 0


def _draft_bracketed(orig, spec, args, kwargs):
    S = B.S
    if R.spec is None:
        R.spec = spec
    if not R.active["draft_graph"] or S.disabled or S.sig is not None:
        return orig(spec, *args, **kwargs)
    nreq, n = _draft_n(args, kwargs)
    model = getattr(spec, "model", None)
    if n <= 0 or n > B.MAX_TOKENS or model is None:
        return orig(spec, *args, **kwargs)
    # one plan per DSpark graph set: the greedy-sharded set runs with _sharded_markov set, the stock set without
    tag = 1 if getattr(spec, "_sharded_markov", None) is not None else 0
    sig = (DRAFT_TAG, tag, nreq, n)
    if torch.cuda.is_current_stream_capturing():
        mode = "capture" if sig in S.plans else None
    else:
        mode = "learn" if (sig not in S.frozen and S.learned[sig] < B.LEARN_MAX) else None
    if mode is None:
        return orig(spec, *args, **kwargs)
    B._hook_modules(model)
    S.sig, S.mode, S.idx, S.forks = sig, mode, 0, 0
    S.lin, S.forks2 = 0, 0
    S.g2_forks = collections.Counter()
    S.events = [] if mode == "learn" else None
    try:
        return orig(spec, *args, **kwargs)
    finally:
        B._join()
        B._join2()
        B._join3()
        try:
            _draft_finish(sig, mode)
        except Exception as e:  # noqa: BLE001 - fail closed: no (further) drafter plan, drafts unchanged
            R.active["draft_graph"] = False
            S.plans.pop(sig, None)
            logger.warning("DSV41_L2_PREFETCH_R1_DRAFT: drafter planning failed at %s (%r); drafter graph "
                           "prefetch off (graphs captured so far keep their joined hints)", mode, e)
        finally:
            S.sig = S.mode = S.events = None


def _draft_finish(sig, mode) -> None:
    S = B.S
    if mode == "learn":
        S.learned[sig] += 1
        ev = S.events or []
        hrecs = [(idx, head_items(spec)) for idx, spec in _recs("head", S.events)]
        if S.idx == 0:
            return
        plan = make_drafter_plan(ev, S.idx, hrecs, device=_device())
        old = S.plans.get(sig)
        S.plans[sig] = plan
        if old is None or old["key"] != plan["key"]:
            first = ("r1plan", sig[1], sig[3]) not in S.logged
            S.logged.add(("r1plan", sig[1], sig[3]))
            logger.info("DSV41_L2_PREFETCH_R1_DRAFT plan set=%s reqs=%d tokens=%d collectives=%d wq_b=%d %s%s%s",
                        "greedy" if sig[1] else "stock", sig[2], sig[3], plan["n"], plan["nl"],
                        plan["g2"]["summary"], " (relearned)" if old is not None else "",
                        ("; first: " + plan["desc"]) if first else "")
    else:
        S.frozen.add(sig)
        plan = S.plans.get(sig)
        ok = plan is not None and plan["n"] == S.idx and plan.get("nl", 0) == S.lin
        if ("r1cap", sig) not in S.logged or not ok:
            S.logged.add(("r1cap", sig))
            logger.info("DSV41_L2_PREFETCH_R1_DRAFT captured set=%s reqs=%d tokens=%d collectives=%d wq_b=%d "
                        "forks %s (planned %s, learned %d/%d)%s",
                        "greedy" if sig[1] else "stock", sig[2], sig[3], S.idx, S.lin,
                        json.dumps(dict(sorted(S.g2_forks.items()))), plan["g2"]["nforks"] if plan else "-",
                        plan["n"] if plan else -1, plan.get("nl", 0) if plan else -1,
                        "" if ok else " MISMATCH: windows past the mismatch were skipped")


def _device():
    try:
        return torch.device("cuda", torch.cuda.current_device())
    except Exception:  # noqa: BLE001 - CPU tests
        return "cpu"


# ----------------------------------------------------------------------------- E-phase ------------
def eph_items(spec, cap: int | None = None):
    """main_proj (R4 split shard when active, else the replicated weight) then the context-KV fold weight, first
    `cap` bytes in that order. None = not resolved yet (the split / fold states are built lazily)."""
    cap = EPH_BYTES if cap is None else cap
    m = getattr(getattr(spec, "model", None), "model", None)
    if m is None:
        return None
    items = []
    mp = getattr(m, "main_proj", None)
    r4 = sys.modules.get(R4_MODULE)
    split_on = bool(getattr(r4, "MAINPROJ_SPLIT", False))
    st = getattr(mp, "_dsv41_r4_split", None) if mp is not None else None
    if split_on and st is None:
        return None
    if st:
        items += _span_items([getattr(st, "s_swz", None), getattr(st, "w", None)], "main_proj_split")
    elif mp is not None:
        sc = getattr(mp, "weight_scale", None)
        if sc is None:
            sc = getattr(mp, "weight_scale_inv", None)
        items += _span_items([sc, getattr(mp, "weight", None)], "main_proj")
    f = getattr(m, "_dsv41_ctxkv", None)
    bnd = sys.modules.get("vllm.models.deepseek_v41.nvidia.dsv41_boundary")
    fold_on = str(getattr(bnd, "CTX_MODE", "0")) != "0" if bnd is not None else False
    if fold_on and f is None:
        return None
    if f:
        for w_t, s, _n in getattr(f, "parts", ()):
            items += _span_items([s, w_t], "ctxkv_fold")
    return B._take_named(items, cap)


def eph_fire() -> None:
    S = B.S
    if S.disabled or S.sig is not None or not S.pace_ok or R.spec is None:
        return
    if torch.cuda.is_current_stream_capturing():
        return
    R.eph_calls += 1
    if R.eph_tab is None or R.eph_calls % 4096 == 0:  # weights are static once resolved: re-check rarely
        segs = eph_items(R.spec)
        if not segs:
            return
        key = tuple(s[:2] for s in segs)
    else:
        segs, key = None, R.eph_key
    if key != R.eph_key:
        # built once (rebuilt only if a resolved weight moves); eager, like the base's learn-time tables
        dev = torch.tensor([v for p, tk, _n in segs for v in (p, tk)], dtype=torch.int64, device=_device())
        S.keepalive.append(dev)
        R.eph_key, R.eph_tab = key, (dev, len(segs), sum(s[1] for s in segs))
        logger.info("DSV41_L2_PREFETCH_R1_DRAFT E-phase table: %s",
                    " ".join(f"{n}:{t / 2**20:.2f}" for _p, t, n in segs))
    dev = torch.cuda.current_device()
    side = R.eph_stream.get(dev)
    if side is None:
        side = R.eph_stream[dev] = torch.cuda.Stream(device=dev)
    main = torch.cuda.current_stream()
    side.wait_stream(main)  # starts when the screen + refine have finished
    t, nseg, _nb = R.eph_tab
    gbps = EPH_GBPS if EPH_GBPS > 0 else B.PACE_GBPS
    period_ps = int(round(B.PACE_CHUNK * 1000.0 / gbps))
    rc = B._load_lib_g2().dsv41_l2pf_paced(t.data_ptr(), nseg, B.PACE_CHUNK, period_ps, 0, EPH_MAX_NS, None,
                                           side.cuda_stream)
    if rc:
        raise RuntimeError(f"paced prefetch launch failed (cuda error {rc})")
    R.eph_fired += 1
    if R.eph_fired in (1, 1000, 100000):
        logger.info("DSV41_L2_PREFETCH_R1_DRAFT E-phase forks: %d", R.eph_fired)
# ----------------------------------------------------------------------------- r5-ffnfront ----------  # r5-ffnfront
# DSV41_L2_PREFETCH_R1_SHW2_GBPS / _DELAY_US / _STREAM and DSV41_L2_PREFETCH_R1_POST_EVENT (climb round 5).  # r5-ffnfront
# The SHW2 post part becomes an "x" part (the same table, segments and bytes) carrying its own pace rate,  # r5-ffnfront
# pacer delay, side stream and an optional deferral: launched from an event recorded right after the  # r5-ffnfront
# all-reduce kernel (the point base after_collective forks today) but enqueued after the mHC boundary  # r5-ffnfront
# (mpn) or after the boundary's main successor (succ). Cache hints only.  # r5-ffnfront
_R5_STREAMS = (1, 2, 3)  # r5-ffnfront
def _r5_bind(base, why) -> None:  # r5-ffnfront
    """r5-ffnfront: validate the knobs once at bind time; all or nothing (an invalid arm runs stock + INACTIVE)."""  # r5-ffnfront
    global R5X  # r5-ffnfront
    bad = []  # r5-ffnfront
    if not R.active["shw2"]:  # r5-ffnfront
        bad.append("needs the active SHW2 lever (DSV41_L2_PREFETCH_R1_SHW2=1, G2 mode, PRUNE_SHARED=1)")  # r5-ffnfront
    if not 0 <= SHW2_GBPS <= 400:  # r5-ffnfront
        bad.append(f"SHW2_GBPS={_env('DSV41_L2_PREFETCH_R1_SHW2_GBPS', '')!r} not a number in [0, 400]")  # r5-ffnfront
    if SHW2_GBPS > 0 and base.PACE_GBPS <= 0:  # r5-ffnfront
        bad.append("SHW2_GBPS needs the base pacer library (DSV41_L2_PREFETCH_PACE_GBPS>0)")  # r5-ffnfront
    if not 0 <= SHW2_DELAY_NS <= 100000:  # r5-ffnfront
        bad.append(f"SHW2_DELAY_US={_env('DSV41_L2_PREFETCH_R1_SHW2_DELAY_US', '')!r} not a number in [0, 100]")  # r5-ffnfront
    if SHW2_DELAY_NS > 0 and SHW2_GBPS <= 0:  # r5-ffnfront
        bad.append("SHW2_DELAY_US needs SHW2_GBPS>0 (the delay is the pacer kernel's)")  # r5-ffnfront
    if SHW2_STREAM not in _R5_STREAMS:  # r5-ffnfront
        bad.append(f"SHW2_STREAM={_env('DSV41_L2_PREFETCH_R1_SHW2_STREAM', '')!r} not in 1/2/3")  # r5-ffnfront
    elif SHW2_STREAM != 3 and SHW2_GBPS <= 0:  # r5-ffnfront
        bad.append("SHW2_STREAM applies to the paced part only (set SHW2_GBPS)")  # r5-ffnfront
    if POST_EVENT not in ("", "mpn", "succ"):  # r5-ffnfront
        bad.append(f"POST_EVENT={POST_EVENT!r} not in mpn/succ")  # r5-ffnfront
    if bad:  # r5-ffnfront
        why["r5_shw2"] = "; ".join(bad)  # r5-ffnfront
        R5X = False  # r5-ffnfront
        return  # r5-ffnfront
    R5X = True  # r5-ffnfront
    R.active["r5_shw2"] = True  # r5-ffnfront
    logger.info("DSV41_L2_PREFETCH_R1 (r5-ffnfront): shw2 post part %s; post_event=%s",  # r5-ffnfront
                (f"paced {SHW2_GBPS:g} GB/s delay {SHW2_DELAY_NS / 1e3:g} us on side stream {SHW2_STREAM}"  # r5-ffnfront
                 if SHW2_GBPS > 0 else "burst on side stream 1 (stock)"), POST_EVENT or "off (fork at the AR)")  # r5-ffnfront
def _r5_part(p):  # r5-ffnfront
    """r5-ffnfront: a planned SHW2 burst part ("b", table, nseg, bytes) -> an "x" part with the r5 options."""  # r5-ffnfront
    if p[0] != "b":  # r5-ffnfront
        return p  # r5-ffnfront
    paced = SHW2_GBPS > 0  # r5-ffnfront
    opt = {"stream": SHW2_STREAM if paced else 1, "gbps": float(SHW2_GBPS) if paced else 0.0,  # r5-ffnfront
           "delay_ns": int(SHW2_DELAY_NS) if paced else 0, "defer": bool(POST_EVENT), "r5": True}  # r5-ffnfront
    return ("x", p[1], p[2], p[3], opt)  # r5-ffnfront
def _r5_launch_args(opt):  # r5-ffnfront
    """r5-ffnfront: (period_ps, delay_ns, max_ns) of a paced r5 part (the pacer's cap counts from the delay)."""  # r5-ffnfront
    return int(round(B.PACE_CHUNK * 1000.0 / opt["gbps"])), int(opt.get("delay_ns", 0)), int(B.PACE_MAX_NS)  # r5-ffnfront
def _r5_fork(part, where: str, ev=None) -> None:  # r5-ffnfront
    """r5-ffnfront: launch an r5 "x" part, or (defer, no event yet) record the fork-point event and keep it."""  # r5-ffnfront
    _m, t, nseg, _nb, opt = part  # r5-ffnfront
    S = B.S  # r5-ffnfront
    if opt.get("defer") and ev is None:  # r5-ffnfront
        R.deferred.append((torch.cuda.current_stream().record_event(), part, where))  # r5-ffnfront
        return  # r5-ffnfront
    st, gbps = opt["stream"], opt["gbps"]  # r5-ffnfront
    side = {1: B._side_stream, 2: B._side_stream2, 3: B._side_stream3}[st]()  # r5-ffnfront
    if ev is not None:  # r5-ffnfront
        side.wait_event(ev)  # r5-ffnfront
    else:  # r5-ffnfront
        side.wait_stream(torch.cuda.current_stream())  # r5-ffnfront
    if gbps > 0 and S.pace_ok:  # r5-ffnfront
        period_ps, delay_ns, max_ns = _r5_launch_args(opt)  # r5-ffnfront
        rc = B._load_lib_g2().dsv41_l2pf_paced(t.data_ptr(), nseg, B.PACE_CHUNK, period_ps, delay_ns, max_ns,  # r5-ffnfront
                                               None, side.cuda_stream)  # r5-ffnfront
    else:  # r5-ffnfront
        rc = B._load_lib().dsv41_l2pf(t.data_ptr(), nseg, B.CHUNK, side.cuda_stream)  # r5-ffnfront
    if rc:  # r5-ffnfront
        raise RuntimeError(f"DSV41_L2_PREFETCH_R1 (r5-ffnfront): prefetch launch failed (cuda error {rc})")  # r5-ffnfront
    if st == 1:  # r5-ffnfront
        S.pending = side  # r5-ffnfront
    elif st == 2:  # r5-ffnfront
        S.pending2 = side  # r5-ffnfront
    else:  # r5-ffnfront
        S.pending3 = side  # r5-ffnfront
    S.g2_forks[f"{where}_x"] += 1  # r5-ffnfront
def flush_deferred(why: str) -> None:  # r5-ffnfront
    """r5-ffnfront: enqueue every deferred part; each side stream waits on its own fork-point event."""  # r5-ffnfront
    items, R.deferred = R.deferred, []  # r5-ffnfront
    for ev, part, where in items:  # r5-ffnfront
        _r5_fork(part, where, ev)  # r5-ffnfront
    if items:  # r5-ffnfront
        R.stats[f"r5_flush_{why}"] += len(items)  # r5-ffnfront
        if ("r5flush", why) not in B.S.logged:  # r5-ffnfront
            B.S.logged.add(("r5flush", why))  # r5-ffnfront
            logger.info("DSV41_L2_PREFETCH_R1 (r5-ffnfront): deferred post parts flushed at %s (first time)", why)  # r5-ffnfront
def flush_deferred_end() -> None:  # r5-ffnfront
    """r5-ffnfront: the forward-end flush inside dsv41_l2_prefetch._bracketed's finally. An exception must not skip  # r5-ffnfront
    the joins / _finish / S.sig reset that follow it (that would switch prefetch off for the process), so it is  # r5-ffnfront
    kept in R.r5_exc and re-raised by raise_deferred_error() after the bracket's cleanup."""  # r5-ffnfront
    try:  # r5-ffnfront
        flush_deferred("end")  # r5-ffnfront
    except Exception as e:  # noqa: BLE001 - re-raised after the cleanup  # r5-ffnfront
        R.deferred = []  # r5-ffnfront
        R.r5_exc = e  # r5-ffnfront
def raise_deferred_error() -> None:  # r5-ffnfront
    """r5-ffnfront: re-raise a forward-end flush failure once the bracket has cleaned up (fail stop)."""  # r5-ffnfront
    e, R.r5_exc = R.r5_exc, None  # r5-ffnfront
    raise RuntimeError("DSV41_L2_PREFETCH_R1 (r5-ffnfront): deferred post-part flush failed at the forward end") from e  # r5-ffnfront
def _r5_install(mod) -> None:  # r5-ffnfront
    """r5-ffnfront POST_EVENT: install the flush point after the mHC boundary (mpn) or check the succ hook."""  # r5-ffnfront
    global POST_EVENT  # r5-ffnfront
    if POST_EVENT == "succ" and getattr(mod, "_DSV41_R5_SUCC", None) is None:  # r5-ffnfront
        logger.warning("DSV41_L2_PREFETCH_R1 (r5-ffnfront): model.py successor hook missing; POST_EVENT succ -> mpn")  # r5-ffnfront
        POST_EVENT = "mpn"  # r5-ffnfront
    if POST_EVENT != "mpn" or "r5mpn" in R.installed:  # r5-ffnfront
        return  # r5-ffnfront
    fn = getattr(mod, "mhc_shifted_post_pre", None)  # r5-ffnfront
    if fn is None:  # r5-ffnfront
        logger.warning("DSV41_L2_PREFETCH_R1 (r5-ffnfront): no mhc_shifted_post_pre; deferred parts flush at the "  # r5-ffnfront
                       "next collective")  # r5-ffnfront
        return  # r5-ffnfront
    R.installed.add("r5mpn")  # r5-ffnfront
    def mhc_shifted_post_pre(*args, **kwargs):  # r5-ffnfront
        out = fn(*args, **kwargs)  # r5-ffnfront
        if R.deferred:  # r5-ffnfront
            flush_deferred("mpn")  # r5-ffnfront
        return out  # r5-ffnfront
    mhc_shifted_post_pre._dsv41_r5 = True  # type: ignore[attr-defined]  # r5-ffnfront
    mhc_shifted_post_pre.__wrapped__ = fn  # type: ignore[attr-defined]  # r5-ffnfront
    setattr(mod, "mhc_shifted_post_pre", mhc_shifted_post_pre)  # r5-ffnfront
    logger.info("DSV41_L2_PREFETCH_R1 (r5-ffnfront): deferred post parts flush after the mHC boundary (mpn)")  # r5-ffnfront
