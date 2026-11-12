#!/usr/bin/env python3
"""g2 colltrim (b) CPU planner test: no GPU, no CUDA context (run with CUDA_VISIBLE_DEVICES="").

  cd ~/dsv41-stage/g2/colltrim && CUDA_VISIBLE_DEVICES= nice -n 19 ~/venvs/vllm032/bin/python tests/test_plan_g2_cpu.py

Loads the LIVE module (base/dsv41_l2_prefetch.py.998af3d7) and the staged one side by side and feeds both
synthetic learn-mode event streams shaped like the live decode forward (embedding AR, per layer: qkv_a
gather -> wq_b [-> indexer wq_b] -> wo_a -> wo_b -> post-attention AR -> mhc fn + router gate + shared
experts -> post-MoE AR -> next mhc fn + qkv_a shard; two Engram layers with their own AR/AG pairs).
Gates (exit 1 on any failure):
  1 flags off: the staged make_plan == the live make_plan (n, window tables, bytes, key, desc) for
    every combination of PRUNE_SHARED / PRUNE_ROUTER / budget / WOA on-off / SKIP_N, and no "g2" entry.
  2 every g2 arm conserves bytes (what leaves a window arrives in exactly one other fork), never grows
    a window beyond REPLAN_MB, moves only what it should (attn: post-attention AR windows; qb*: the q_b
    category of qkv_a gather windows), and leaves every other window byte-identical.
  3 pacing splits each paced fork into a burst head <= PACE_HEAD_KB and a paced tail, bytes conserved,
    only for kinds in PACE_KINDS; without the pace library everything stays burst.
  4 WOA + attn: the attn fork starts with exactly the WOA tables make_woa_plan builds.
"""
import collections
import importlib.machinery
import importlib.util
import itertools
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ["DSV41_L2_PREFETCH"] = "1"
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MB = 1 << 20


def load(path, name):
    loader = importlib.machinery.SourceFileLoader(name, path)  # the base copy has no .py suffix
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


OLD = load(os.path.join(ROOT, "base", "dsv41_l2_prefetch.py.998af3d7"), "l2pf_live_998af3d7")
NEW = load(os.path.join(ROOT, "site", "vllm", "distributed", "device_communicators", "dsv41_l2_prefetch.py"),
           "l2pf_g2_staged")
FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print("  FAIL " + msg, flush=True)
    return bool(cond)


class Ptr:
    def __init__(self):
        self.next = 1 << 32

    def __call__(self, nbytes):
        p = self.next
        self.next += (nbytes + 4095) // 4096 * 4096 + 4096
        return p


def events_for(n_layers=6, indexer_layers=(1, 3, 5), engram_layers=(1, 4), woa=False, lin=True):
    """Learn-mode events of one decode forward at M <= 8 (qkv_a shard on), shaped like the live 2-token plan."""
    P = Ptr()
    ev = []
    c = itertools.count(1)
    lc = itertools.count(1)

    def w(name, *sizes, kind="w"):
        ev.append((kind, name, [(P(int(s * MB)), int(s * MB) & ~15, None) for s in sizes]))

    def coll(kind):
        ev.append(("c", next(c), kind))

    coll("ar")  # embedding all-reduce
    w("engram_hash", 0.49)
    w("mhc_fn", 0.47)
    w("language_model.model.layers.0.attn.qkv_a_shard", 0.03, 1.09)
    for i in range(n_layers):
        pre = f"language_model.model.layers.{i}"
        coll("ag")  # qkv_a gather
        w(f"{pre}.attn.wq_b", 0.16, 5.00)
        if lin:
            ev.append(("L", next(lc)))
        if i in indexer_layers:
            w(f"{pre}.attn.indexer.wq_b", 0.06, 1.0)
            w(f"{pre}.attn.indexer.weights_proj", 0.06)
            if lin:
                ev.append(("L", next(lc)))
        if woa:
            w(f"{pre}.attn.wo_a", 0.13, 4.0, kind="woa")
        w(f"{pre}.attn.wo_b", 0.16, 2.5)
        coll("ar")  # post-attention all-reduce
        w("mhc_fn", 1.88)
        w(f"{pre}.ffn.gate", 3.75)
        w(f"{pre}.ffn.shared_experts.gate_up_proj", 0.1, 3.0)
        w(f"{pre}.ffn.shared_experts.down_proj", 0.05, 1.5)
        coll("ar")  # post-MoE all-reduce
        if i + 1 in engram_layers:
            w(f"language_model.model.layers.{i + 1}.engram", 0.04, 0.04)
            coll("ar")
            w(f"language_model.model.layers.{i + 1}.engram.wkv", 0.59, 5.41)
            coll("ag")
            w("engram_hash", 0.49)
        w("mhc_fn", 0.47)
        w(f"language_model.model.layers.{i + 1}.attn.qkv_a_shard", 0.03, 1.09)
    coll("ar")  # final
    w("final_norm", 0.04)
    n = sum(1 for e in ev if e[0] == "c")
    return ev, n


def set_flags(mod, **kw):
    for k, v in kw.items():
        setattr(mod, k, v)


def table(t):
    return [int(v) for v in t.tolist()]


def plan_sig(p):
    return (p["n"], p["bytes"], p["desc"], p["key"],
            {k: (table(v[0]), v[1], v[2]) for k, v in p["win"].items()},
            {k: (table(v[0]), v[1], v[2]) for k, v in p.get("lwin", {}).items()}, p.get("nl"))


# ---- 1. flags off: staged == live ----------------------------------------------------------------
print("[1] flags off: staged make_plan == live make_plan", flush=True)
n_cases = 0
for prune_shared, prune_router, budget, woa, skip_n in itertools.product(
        (False, True), (False, True), (6 * MB, 3 * MB), (False, True), (0, 64)):
    for mod in (OLD, NEW):
        set_flags(mod, PRUNE_SHARED=prune_shared, PRUNE_ROUTER=prune_router, WOA=woa)
    set_flags(NEW, G2=False, REPLAN=frozenset(), REPLAN_ATTN=False, PACE_GBPS=0.0)
    ev, n = events_for(woa=woa, lin=woa)
    a = OLD.make_plan(ev, n, budget=budget, skip_n=skip_n, device="cpu")
    b = NEW.make_plan(ev, n, budget=budget, skip_n=skip_n, device="cpu")
    n_cases += 1
    check(plan_sig(a) == plan_sig(b),
          f"flags-off plan differs (shared={prune_shared} router={prune_router} budget={budget} woa={woa} skip={skip_n})")
    check("g2" not in b, "flags-off plan carries a g2 entry")
for mod in (OLD, NEW):
    set_flags(mod, PRUNE_SHARED=False, PRUNE_ROUTER=False, WOA=False)
print(f"    {n_cases} combinations compared", flush=True)


# ---- helpers for the g2 arms ------------------------------------------------------------------------
def base_sel(ev, n, budget=6 * MB, woa=False):
    """The per-window selection make_plan uses (G2 on, no re-plan): [(idx, kind, segs)]."""
    set_flags(NEW, G2=True, REPLAN=frozenset(), REPLAN_ATTN=False, PACE_GBPS=0.0, WOA=woa)
    NEW.S.pace_ok = False
    captured = {}
    orig = NEW.make_g2_plan

    def spy(events, sel, device="cuda"):
        captured["sel"] = sel
        return orig(events, sel, device=device)

    NEW.make_g2_plan = spy
    try:
        NEW.make_plan(ev, n, budget=budget, device="cpu")
    finally:
        NEW.make_g2_plan = orig
    return captured["sel"]


def total(d):
    return sum(t for segs in d.values() for _p, t, _n in segs)


def bytes_by_ptr(d):
    out = collections.Counter()
    for segs in d.values():
        for p, t, _n in segs:
            out[p] += t
    return out


def cover(segs_dict):
    """set of (absolute byte ranges) as a sorted list of (start, end) merged"""
    iv = sorted((p, p + t) for segs in segs_dict.values() for p, t, _n in segs)
    out = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            check(s >= out[-1][1], f"overlapping prefetch ranges {hex(s)}")
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


ev, n = events_for()
sel = base_sel(ev, n)
by_idx = {idx: (kind, segs) for idx, kind, segs in sel}
base_pre = {idx: segs for idx, _k, segs in sel if segs}
base_total = total(base_pre)
order = [idx for idx, _k, _s in sel]

# ---- 2. re-plan arms ------------------------------------------------------------------------------
print("[2] re-plan arms: conservation and placement", flush=True)
for arm in ("attn", "qbprev", "qbpost", "attn,qbprev", "attn,qbpost"):
    rp = frozenset(arm.split(","))
    for cap in (16 * MB, 3 * MB):
        fl = NEW.g2_fork_lists(ev, sel, replan=rp, replan_budget=cap, woa=False)
        got = total(fl["pre"]) + total(fl["post"]) + total(fl["attn"])
        check(got == base_total, f"{arm} cap={cap / MB:.0f}MB: bytes {got} != base {base_total}")
        check(cover(base_pre) == cover({**{("pre", k): v for k, v in fl["pre"].items()},
                                       **{("post", k): v for k, v in fl["post"].items()},
                                       **{("attn", k): v for k, v in fl["attn"].items()}}),
              f"{arm}: prefetched byte ranges differ from the base plan")
        moved_attn = moved_qb = 0
        for pos, idx in enumerate(order):
            kind, segs = by_idx[idx]
            new = fl["pre"].get(idx, [])
            prev = order[pos - 1] if pos else None
            is_post_attn = (kind == "ar" and pos > 0 and by_idx[prev][0] == "ag"
                            and any("attn.wq_b" in s[2] for s in by_idx[prev][1]))
            if "attn" in rp and is_post_attn:
                moved = sum(t for _p, t, _n in segs) - sum(t for _p, t, _n in new)
                moved_attn += moved
                check(0 < moved <= cap and (not new or moved > cap - 16 * len(segs)),
                      f"{arm}: post-attn window {idx} moved {moved} (cap {cap})")
                if cap >= 6 * MB:
                    check(not new, f"{arm}: post-attn window {idx} not emptied")
            elif kind == "ag" and any(NEW._weight_category(s[2]) == "q_b" for s in segs) and (
                    "qbprev" in rp or "qbpost" in rp):
                qb_new = [s for s in new if NEW._weight_category(s[2]) == "q_b"]
                if cap >= 6 * MB:
                    check(not qb_new, f"{arm}: ag window {idx} still holds q_b")
                rest_old = [s for s in segs if NEW._weight_category(s[2]) != "q_b"]
                rest_new = [s for s in new if NEW._weight_category(s[2]) != "q_b"]
                check(rest_old == rest_new, f"{arm}: ag window {idx} non-q_b bytes changed")
                moved_qb += sum(t for _p, t, n_ in segs if NEW._weight_category(n_) == "q_b") - sum(
                    t for _p, t, _n in qb_new)
            elif not ("qbprev" in rp and pos + 1 < len(order) and by_idx[order[pos + 1]][0] == "ag"):
                check(new == segs, f"{arm}: untouched window {idx} ({kind}) changed")
        check(fl["moved"].get("attn", 0) == moved_attn, f"{arm}: moved attn bookkeeping")
        check(fl["moved"].get("qbprev", 0) + fl["moved"].get("qbpost", 0) == moved_qb, f"{arm}: moved qb bookkeeping")
        for lin, segs in fl["attn"].items():
            check(sum(t for _p, t, _n in segs) <= cap, f"{arm}: attn fork {lin} exceeds the cap")
        if "qbpost" in rp:
            for idx, segs in fl["post"].items():
                check(all(NEW._weight_category(s[2]) == "q_b" for s in segs), f"{arm}: post fork {idx} not q_b")
                nxt = order[order.index(idx) + 1] if order.index(idx) + 1 < len(order) else None
                check(nxt is not None and by_idx[nxt][0] == "ag"
                      and any(NEW._weight_category(s[2]) == "q_b" for s in by_idx[nxt][1]),
                      f"{arm}: post fork {idx} is not the collective right before a q_b gather window")
        print(f"    {arm:12s} cap {cap / MB:4.0f} MB: moved attn {moved_attn / MB:6.2f} MB, q_b {moved_qb / MB:6.2f} MB; "
              f"forks pre {len(fl['pre'])} post {len(fl['post'])} attn {len(fl['attn'])}", flush=True)

# attn fork lands at the LAST wq_b of the block (indexer layers have two)
fl = NEW.g2_fork_lists(ev, sel, replan=frozenset({"attn"}), replan_budget=16 * MB, woa=False)
lins = sorted(fl["attn"])
last_ls = []
cur = None
seen = {}
for e in ev:
    if e[0] == "c":
        cur = e[1]
    elif e[0] == "L":
        seen[cur] = e[1]
check(lins == sorted(seen.values()), f"attn forks at {lins}, expected the last wq_b per block {sorted(seen.values())}")

# ---- 3. pacing split --------------------------------------------------------------------------------
print("[3] pacing: head/tail split", flush=True)
for gbps, head_kb, kinds, ok in ((100.0, 0, "ar,ag,post,attn", True), (60.0, 2048, "ar", True),
                                  (150.0, 512, "ag,post", True), (100.0, 0, "ar,ag", False)):
    set_flags(NEW, PACE_GBPS=gbps, PACE_HEAD=head_kb * 1024, PACE_KINDS=frozenset(kinds.split(",")))
    NEW.S.pace_ok = ok
    for kind in ("ar", "ag", "post", "attn"):
        for idx, segs in list(base_pre.items())[:12]:
            parts = NEW._parts(segs, kind, "cpu")
            nb = sum(p[3] for p in parts)
            check(nb == sum(t for _p, t, _n in segs), f"pace {gbps} {kind}: bytes not conserved")
            paced = ok and kind in kinds.split(",")
            modes = [p[0] for p in parts]
            if not paced:
                check(modes == ["b"], f"pace {gbps} {kind}: unpaced fork has modes {modes}")
            else:
                hb = sum(p[3] for p in parts if p[0] == "b")
                check(hb <= head_kb * 1024, f"pace {gbps} {kind}: head {hb} > {head_kb} KB")
                check("p" in modes or nb <= head_kb * 1024, f"pace {gbps} {kind}: no paced tail")
                for mode, t, nseg, b in parts:
                    tab = table(t)
                    check(len(tab) == 2 * nseg and sum(tab[1::2]) == b, "table/nseg/bytes mismatch")
                    check(all(v % 16 == 0 for v in tab), "unaligned segment in a paced/burst table")
set_flags(NEW, PACE_GBPS=0.0, PACE_HEAD=0, PACE_KINDS=frozenset({"ar", "ag", "post", "attn"}))
NEW.S.pace_ok = False

# ---- 4. WOA + attn: WOA tables first ------------------------------------------------------------------
print("[4] WOA + attn re-plan", flush=True)
evw, nw = events_for(woa=True)
selw = base_sel(evw, nw, woa=True)
set_flags(NEW, WOA=True)
set_flags(OLD, WOA=True)
woa_plan = OLD.make_woa_plan(evw, device="cpu")
fl = NEW.g2_fork_lists(evw, selw, replan=frozenset({"attn"}), replan_budget=16 * MB, woa=True)
for lin, item in woa_plan["lwin"].items():
    tab = table(item[0])
    segs = fl["attn"].get(lin, [])
    head = [v for p, t, _n in segs[:item[1]] for v in (p, t)]
    check(head == tab, f"attn fork L{lin} does not start with the WOA table")
fl0 = NEW.g2_fork_lists(evw, selw, replan=frozenset(), replan_budget=16 * MB, woa=True)
for lin, item in woa_plan["lwin"].items():
    check([v for p, t, _n in fl0["attn"].get(lin, []) for v in (p, t)] == table(item[0]),
          f"WOA-only g2 fork L{lin} != make_woa_plan")
set_flags(NEW, WOA=False)
set_flags(OLD, WOA=False)

# ---- 5. full make_plan in G2 mode: plan structure -------------------------------------------------------
print("[5] make_plan G2 structure", flush=True)
set_flags(NEW, G2=True, REPLAN=frozenset({"attn", "qbpost"}), REPLAN_ATTN=True, PACE_GBPS=100.0,
          PACE_HEAD=1024 * 1024, PACE_KINDS=frozenset({"ar", "ag", "post", "attn"}), REPLAN_BUDGET=16 * MB)
NEW.S.pace_ok = True
p = NEW.make_plan(ev, n, device="cpu")
g2 = p.get("g2")
check(g2 is not None and set(g2) >= {"pre", "post", "attn", "summary", "gkey", "nl"}, "g2 plan entries missing")
check(p.get("nl") == max(e[1] for e in ev if e[0] == "L"), "plan nl (wq_b bound) wrong without WOA")
print("    " + g2["summary"], flush=True)
print("    " + g2["desc"][:300], flush=True)
set_flags(NEW, G2=False, REPLAN=frozenset(), REPLAN_ATTN=False, PACE_GBPS=0.0, PACE_HEAD=0)
NEW.S.pace_ok = False

# ---- 6. _cap_named edge cases ----------------------------------------------------------------------------
segs = [(0, 1000 * 16, "a"), (1 << 20, 50 * 16, "b"), (2 << 20, 7 * 16, "c")]
for cap in (0, 16, 1000 * 16, 1000 * 16 + 8, 1030 * 16, 10 ** 9):
    h, t = NEW._cap_named(segs, cap)
    check(sum(x[1] for x in h) == min(cap & ~15, 1057 * 16), f"_cap_named head bytes cap={cap}")
    check(sum(x[1] for x in h) + sum(x[1] for x in t) == 1057 * 16, f"_cap_named conservation cap={cap}")

print("RESULT", "PASS" if not FAILS else f"FAIL ({len(FAILS)})", flush=True)
sys.exit(1 if FAILS else 0)
