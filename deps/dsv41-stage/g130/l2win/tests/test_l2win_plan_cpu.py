#!/usr/bin/env python3
"""g130 l2win CPU test: no GPU, no CUDA context. Safe while serving under nice -n 19.

  cd ~/dsv41-stage/g130/l2win && CUDA_VISIBLE_DEVICES= nice -n 19 ~/venvs/vllm032/bin/python -I tests/test_l2win_plan_cpu.py

Loads base/dsv41_l2_prefetch.py.63737f9c (the live module) and site/dsv41_l2_prefetch.py side by side and feeds both learn-mode event
streams shaped like the live plan at <= 8 tokens with MHC=0 and PRUNE_SHARED (rank0.log, launcher 37757dde).
  1 apply_l2win.py(base) == site file; wrong base refused; every anchor found once
  2 flags off: patched make_plan == base make_plan over prune / budget / WOA / skip combinations, no g2 entry, G2 False
  3 env parsing: unset gives every new constant its off value; each arm's env gives G2 True and the expected values
  4 G2 forced on with no pacing and no delay: a captured forward issues the SAME launches as the live module, in the same order, on the
    same streams, with identical tables, WOA included. Fake library and streams record the calls.
  5 PACE_MAX_WIN: gate window stays burst, short window paced, gather and attn untouched, bytes and tables conserved
  6 PACE_BIG_GBPS: gate window paced at the big rate (period checked), short at PACE_GBPS, pacer stream used, burst side untouched
  7 BURST_DELAY: spin launched first on the same side stream with 1 CTA and the right ns; kinds respected; attn never delayed
  8 parse_mode and verdict decision functions on synthetic rows, including every KILL branch
"""
import collections
import hashlib
import importlib.machinery
import importlib.util
import itertools
import os
import subprocess
import sys
import types

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["DSV41_L2_PREFETCH"] = "1"
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = os.path.join(ROOT, "base", "dsv41_l2_prefetch.py.63737f9c")
SITE = os.path.join(ROOT, "site", "dsv41_l2_prefetch.py")
MB = 1 << 20
FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print("  FAIL " + msg, flush=True)
    return bool(cond)


def load(path, name):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def clean_env():
    return {k: v for k, v in os.environ.items() if not k.startswith("DSV41_L2_PREFETCH_")}


for k in [k for k in os.environ if k.startswith("DSV41_L2_PREFETCH_")]:
    del os.environ[k]
OLD = load(BASE, "l2pf_base")
NEW = load(SITE, "l2pf_new")


class Ptr:
    def __init__(self):
        self.next = 1 << 32

    def __call__(self, nbytes):
        p = self.next
        self.next += (nbytes + 4095) // 4096 * 4096 + 4096
        return p


def events_for(n_layers=6, indexer_layers=(1, 3, 5), engram_layers=(1, 4), woa=False, lin=True):
    P = Ptr()
    ev = []
    c = itertools.count(1)
    lc = itertools.count(1)

    def w(name, *sizes, kind="w"):
        ev.append((kind, name, [(P(int(s * MB)), int(s * MB) & ~15, None) for s in sizes]))

    def coll(kind):
        ev.append(("c", next(c), kind))

    coll("ar")
    w("engram_hash", 0.49)
    w("language_model.model.layers.0.attn.qkv_a_shard", 0.03, 1.09)
    for i in range(n_layers):
        pre = f"language_model.model.layers.{i}"
        coll("ag")
        w(f"{pre}.attn.wq_b", 0.16, 5.00)
        if lin:
            ev.append(("L", next(lc)))
        if i in indexer_layers:
            w(f"{pre}.attn.indexer.wq_b", 0.06, 1.0)
            if lin:
                ev.append(("L", next(lc)))
        if woa:
            w(f"{pre}.attn.wo_a", 0.13, 4.0, kind="woa")
        w(f"{pre}.attn.wo_b", 0.16, 0.69)
        coll("ar")  # post-attention
        w(f"{pre}.ffn.gate", 3.75)
        w(f"{pre}.ffn.shared_experts.gate_up_proj", 0.1, 3.0)
        coll("ar")  # post-MoE
        if i + 1 in engram_layers:  # plan log: [ar3 engram 0.04 + 0.04] [ag4 wkv 0.59 + 5.41]
            w(f"language_model.model.layers.{i + 1}.engram", 0.04, 0.04)
            coll("ag")
            w(f"language_model.model.layers.{i + 1}.engram.wkv", 0.59, 5.41)
            coll("ar")
        w(f"language_model.model.layers.{i + 1}.attn.qkv_a_shard", 0.03, 1.09)
    coll("ar")
    w("final_norm", 0.04)
    n = sum(1 for e in ev if e[0] == "c")
    return ev, n


def table(t):
    return [int(v) for v in t.tolist()]


def plan_sig(p):
    return (p["n"], p["bytes"], p["desc"], p["key"], {k: (table(v[0]), v[1], v[2]) for k, v in p["win"].items()},
            {k: (table(v[0]), v[1], v[2]) for k, v in p.get("lwin", {}).items()}, p.get("nl"))


def set_flags(mod, **kw):
    for k, v in kw.items():
        setattr(mod, k, v)


# ---- 1 ------------------------------------------------------------------------------------------------------------
print("[1] patcher", flush=True)
sys.path.insert(0, os.path.join(ROOT, "patch"))
sys.dont_write_bytecode = True
import apply_l2win as AP  # noqa: E402

base_text = open(BASE).read()
check(hashlib.sha256(base_text.encode()).hexdigest() == AP.BASE_SHA, "base copy is 63737f9c")
check(AP.apply(base_text) == open(SITE).read(), "site file == apply_l2win(base)")
for old, _new in AP.EDITS:
    check(base_text.count(old) == 1, f"anchor once: {old[:50]!r}")
try:
    AP.apply(base_text.replace(AP.EDITS[2][0], "G2 = False\n"))
    check(False, "a missing anchor must refuse")
except SystemExit:
    pass

# ---- 2 ------------------------------------------------------------------------------------------------------------
print("[2] flags off: patched make_plan == base make_plan", flush=True)
check(NEW.G2 is False and NEW.PACE_MAX_WIN == 0 and NEW.BURST_DELAY_NS == 0 and NEW.PACE_BIG_GBPS == 0.0, "off defaults")
n_cases = 0
for prune_shared, prune_router, budget, woa, skip_n in itertools.product(
        (False, True), (False, True), (6 * MB, 3 * MB), (False, True), (0, 64)):
    for mod in (OLD, NEW):
        set_flags(mod, PRUNE_SHARED=prune_shared, PRUNE_ROUTER=prune_router, WOA=woa)
    ev, n = events_for(woa=woa, lin=woa)
    a = OLD.make_plan(ev, n, budget=budget, skip_n=skip_n, device="cpu")
    b = NEW.make_plan(ev, n, budget=budget, skip_n=skip_n, device="cpu")
    n_cases += 1
    check(plan_sig(a) == plan_sig(b), f"plan differs shared={prune_shared} router={prune_router} budget={budget} woa={woa}")
    check("g2" not in b, "flags-off plan carries a g2 entry")
for mod in (OLD, NEW):
    set_flags(mod, PRUNE_SHARED=True, PRUNE_ROUTER=False, WOA=False)
print(f"    {n_cases} combinations", flush=True)

# ---- 3 ------------------------------------------------------------------------------------------------------------
print("[3] env parsing", flush=True)
PROBE = ("import importlib.machinery as m,importlib.util as u,json,sys;l=m.SourceFileLoader('x',sys.argv[1]);"
         "s=u.spec_from_loader('x',l);x=u.module_from_spec(s);l.exec_module(x);"
         "print(json.dumps([x.G2,x.PACE_GBPS,x.PACE_CHUNK,x.PACE_MAX_WIN,x.PACE_BIG_GBPS,x.BURST_DELAY_NS,sorted(x.BURST_DELAY_KINDS),x.PACE_MAX_NS]))")
import json  # noqa: E402

for env, want in (
        ({}, [False, 0.0, 16384, 0, 0.0, 0, ["ar"], 150000]),
        ({"PACE_GBPS": "125", "PACE_CHUNK": "65536", "PACE_KINDS": "ar"}, [True, 125.0, 65536, 0, 0.0, 0, ["ar"], 150000]),
        ({"PACE_GBPS": "85", "PACE_MAX_WIN_KB": "2048"}, [True, 85.0, 16384, 2097152, 0.0, 0, ["ar"], 150000]),
        ({"PACE_GBPS": "85", "PACE_MAX_WIN_KB": "2048", "PACE_BIG_GBPS": "150", "PACE_MAX_US": "26"},
         [True, 85.0, 16384, 2097152, 150.0, 0, ["ar"], 26000]),
        ({"BURST_DELAY_US": "1.5"}, [True, 0.0, 16384, 0, 0.0, 1500, ["ar"], 150000]),
        ({"BURST_DELAY_US": "3", "BURST_DELAY_KINDS": "ar,ag"}, [True, 0.0, 16384, 0, 0.0, 3000, ["ag", "ar"], 150000])):
    e = clean_env()
    e.update({"DSV41_L2_PREFETCH_" + k: v for k, v in env.items()})
    e["DSV41_L2_PREFETCH"] = "1"
    r = subprocess.run([sys.executable, "-c", PROBE, SITE], env=e, capture_output=True, text=True, timeout=120)
    got = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else r.stderr[-300:]
    check(got == want, f"env {env}: got {got} want {want}")


# ---- fakes for the launch path --------------------------------------------------------------------------------------
class FakeStream:
    def __init__(self, name):
        self.name = name
        self.cuda_stream = name
        self.waits = []

    def wait_stream(self, other):
        self.waits.append(other.name)


class FakeLib:
    def __init__(self, log):
        self.log = log

    def dsv41_l2pf(self, ptr, n, chunk, stream):
        self.log.append(("burst", stream, ptr, n, chunk))
        return 0

    def dsv41_spin(self, ns, ctas, stream):
        self.log.append(("spin", stream, ns, ctas))
        return 0

    def dsv41_l2pf_paced(self, ptr, n, chunk, period_ps, delay_ns, max_ns, stats, stream):
        self.log.append(("paced", stream, ptr, n, chunk, period_ps, delay_ns, max_ns, stats))
        return 0


MAIN = FakeStream("main")
torch.cuda.current_stream = lambda *a, **k: MAIN


def rig(mod, log):
    streams = {"side": FakeStream("side"), "side2": FakeStream("side2"), "side3": FakeStream("side3")}
    mod._side_stream = lambda: streams["side"]
    mod._side_stream2 = lambda: streams["side2"]
    mod._side_stream3 = lambda: streams["side3"]
    lib = FakeLib(log)
    mod._load_lib = lambda: lib
    mod._load_lib_g2 = lambda: lib
    mod.S.disabled = False
    mod.S.installed = True
    mod.S.mode = "capture"
    mod.S.idx = 0
    mod.S.lin = 0
    mod.S.forks = mod.S.forks2 = 0
    mod.S.pending = mod.S.pending2 = None
    if hasattr(mod.S, "pending3"):
        mod.S.pending3 = None
    return streams


def replay(mod, ev, n, plan):
    """Walk a captured forward: before_collective per collective, wq_b post-hook per L event."""
    log = []
    rig(mod, log)
    sig = (1, 2)
    mod.S.plans = {sig: plan}
    mod.S.sig = sig
    for e in ev:
        if e[0] == "c":
            mod.before_collective(e[2])
            log.append(("ar_or_ag", e[2], e[1]))
            mod.after_collective(e[2])
        elif e[0] == "L":
            mod._wq_b_post_hook(None, None, None)
    mod.S.sig = None
    return log


def norm(log, plan_tables):
    """Pointers -> table contents, so runs with distinct table allocations compare."""
    out = []
    for x in log:
        if x[0] in ("burst", "paced"):
            out.append((x[0], x[1], tuple(plan_tables[x[2]])) + tuple(x[3:]))
        else:
            out.append(x)
    return out


def tables_of(plan):
    d = {}

    def add(t):
        d[t.data_ptr()] = table(t)
    for v in plan.get("win", {}).values():
        add(v[0])
    for v in plan.get("lwin", {}).values():
        add(v[0])
    if "g2" in plan:
        for where in ("pre", "post", "attn"):
            for parts in plan["g2"][where].values():
                for p in parts:
                    add(p[1])
    return d


# ---- 4 ------------------------------------------------------------------------------------------------------------
print("[4] G2 forced on, nothing paced or delayed: same launches as the live module", flush=True)
for woa in (False, True):
    set_flags(OLD, WOA=woa, PRUNE_SHARED=True)
    set_flags(NEW, WOA=woa, PRUNE_SHARED=True, G2=False)
    ev, n = events_for(woa=woa, lin=woa)
    p_old = OLD.make_plan(ev, n, device="cpu")
    l_old = norm(replay(OLD, ev, n, p_old), tables_of(p_old))
    set_flags(NEW, G2=True, PACE_GBPS=0.0, REPLAN=frozenset(), REPLAN_ATTN=False, BURST_DELAY_NS=0, PACE_MAX_WIN=0)
    NEW.S.pace_ok = False
    p_new = NEW.make_plan(ev, n, device="cpu")
    check("g2" in p_new, "G2 plan present")
    l_new = norm(replay(NEW, ev, n, p_new), tables_of(p_new))
    check(l_old == l_new, f"woa={woa}: launch sequence differs (first diff "
          f"{next((i for i, (x, y) in enumerate(zip(l_old, l_new)) if x != y), None)}, lens {len(l_old)} {len(l_new)})")
    print(f"    woa={woa}: {sum(1 for x in l_old if x[0] == 'burst')} burst launches compared", flush=True)
set_flags(OLD, WOA=False)
set_flags(NEW, WOA=False, G2=False)


def arm(**kw):
    base = dict(G2=True, PACE_GBPS=0.0, PACE_KINDS=frozenset({"ar"}), PACE_HEAD=0, PACE_MAX_WIN=0, PACE_BIG_GBPS=0.0,
                PACE_CHUNK=16384, PACE_DELAY_NS=0, PACE_MAX_NS=150000, BURST_DELAY_NS=0,
                BURST_DELAY_KINDS=frozenset({"ar"}), REPLAN=frozenset(), REPLAN_ATTN=False, WOA=True)
    base.update(kw)
    set_flags(NEW, **base)
    NEW.S.pace_ok = base["PACE_GBPS"] > 0


ev, n = events_for(woa=True, lin=True)
set_flags(OLD, WOA=True, PRUNE_SHARED=True)
p_ref = OLD.make_plan(ev, n, device="cpu")
kinds = {e[1]: e[2] for e in ev if e[0] == "c"}
ref_bytes = {k: v[2] for k, v in p_ref["win"].items()}


def sum_parts(parts):
    return sum(p[3] for p in parts)


# ---- 5 ------------------------------------------------------------------------------------------------------------
print("[5] PACE_MAX_WIN: small AR forks paced, the rest burst", flush=True)
arm(PACE_GBPS=85.0, PACE_MAX_WIN=2048 * 1024)
p = NEW.make_plan(ev, n, device="cpu")
seen = collections.Counter()
for idx, parts in p["g2"]["pre"].items():
    modes = [x[0] for x in parts]
    if not parts:
        continue
    check(sum_parts(parts) == ref_bytes[idx], f"window {idx}: bytes not conserved")
    if kinds[idx] == "ar" and ref_bytes[idx] <= 2048 * 1024:
        check(modes == ["p"], f"small AR window {idx} modes {modes}")
        seen["paced"] += 1
    else:
        check(modes == ["b"], f"window {idx} kind {kinds[idx]} bytes {ref_bytes[idx]} modes {modes}")
        seen["burst"] += 1
for lin, parts in p["g2"]["attn"].items():
    check([x[0] for x in parts] == ["b"], f"attn fork {lin} not burst")
check(seen["paced"] > 0 and seen["burst"] > 0, f"both classes present {dict(seen)}")
log = replay(NEW, ev, n, p)
check(any(x[0] == "paced" for x in log) and all(x[1] == "side3" for x in log if x[0] == "paced"), "paced launches on the pacer stream")
check(all(x[5] == int(round(16384 * 1000.0 / 85.0)) for x in log if x[0] == "paced"), "period for 85 GB/s")
check(all(x[1] in ("side", "side2") for x in log if x[0] == "burst"), "burst launches on the usual sides")
check(not any(x[0] == "spin" for x in log), "no spin without BURST_DELAY")
print(f"    {dict(seen)}", flush=True)

# ---- 6 ------------------------------------------------------------------------------------------------------------
print("[6] PACE_BIG_GBPS: larger AR forks paced at their own rate", flush=True)
arm(PACE_GBPS=85.0, PACE_MAX_WIN=2048 * 1024, PACE_BIG_GBPS=150.0, PACE_CHUNK=65536)
p = NEW.make_plan(ev, n, device="cpu")
log = replay(NEW, ev, n, p)
per = collections.Counter(x[5] for x in log if x[0] == "paced")
check(set(per) == {int(round(65536 * 1000.0 / 85.0)), int(round(65536 * 1000.0 / 150.0))}, f"periods {per}")
for idx, parts in p["g2"]["pre"].items():
    if parts and kinds[idx] == "ar":
        want = "p" if ref_bytes[idx] <= 2048 * 1024 else "q"
        check([x[0] for x in parts] == [want], f"AR window {idx} modes {[x[0] for x in parts]} want {want}")
    elif parts:
        check([x[0] for x in parts] == ["b"], f"gather window {idx} not burst")
arm(PACE_GBPS=85.0, PACE_MAX_WIN=0, PACE_BIG_GBPS=150.0)
p = NEW.make_plan(ev, n, device="cpu")
check(all(x[0] != "q" for v in p["g2"]["pre"].values() for x in v), "BIG ignored when MAX_WIN is 0")
arm(PACE_GBPS=125.0, PACE_CHUNK=65536, PACE_MAX_NS=26000)
p = NEW.make_plan(ev, n, device="cpu")
log = replay(NEW, ev, n, p)
check(all(x[7] == 26000 and x[4] == 65536 for x in log if x[0] == "paced"), "env-only arm passes chunk and cap through")
check(sum(1 for x in log if x[0] == "paced") == sum(1 for k in kinds if kinds[k] == "ar" and k in ref_bytes),
      "env-only arm paces every AR window")

# ---- 7 ------------------------------------------------------------------------------------------------------------
print("[7] BURST_DELAY: spin chained before the burst", flush=True)
arm(BURST_DELAY_NS=1500)
p = NEW.make_plan(ev, n, device="cpu")
log = replay(NEW, ev, n, p)
n_spin = n_ar = 0
for i, x in enumerate(log):
    if x[0] == "spin":
        n_spin += 1
        check(x[1] == "side" and x[2] == 1500 and x[3] == 1, f"spin args {x}")
        check(log[i + 1][0] == "burst" and log[i + 1][1] == "side", "the burst follows its spin on the same stream")
        check(log[i + 2][0] == "ar_or_ag" and log[i + 2][1] == "ar", "the delayed fork precedes an all-reduce")
n_ar = sum(1 for idx, parts in p["g2"]["pre"].items() if parts and kinds[idx] == "ar")
check(n_spin == n_ar and n_ar > 0, f"one spin per AR window: {n_spin} vs {n_ar}")
check(not any(x[1] == "side2" and x[0] == "spin" for x in log), "attn forks never delayed")
arm(BURST_DELAY_NS=1000, BURST_DELAY_KINDS=frozenset({"ar", "ag"}))
p = NEW.make_plan(ev, n, device="cpu")
log = replay(NEW, ev, n, p)
check(sum(1 for x in log if x[0] == "spin") == sum(1 for v in p["g2"]["pre"].values() if v), "kinds ar,ag: every collective fork delayed")
arm(PACE_GBPS=85.0, PACE_MAX_WIN=2048 * 1024, BURST_DELAY_NS=1000)
p = NEW.make_plan(ev, n, device="cpu")
log = replay(NEW, ev, n, p)
check(any(x[0] == "paced" for x in log) and any(x[0] == "spin" for x in log), "paced small + delayed gate combine")
set_flags(NEW, G2=False, WOA=False, PACE_GBPS=0.0, BURST_DELAY_NS=0, PACE_MAX_WIN=0, PACE_BIG_GBPS=0.0)
NEW.S.pace_ok = False

# ---- 8 ------------------------------------------------------------------------------------------------------------
print("[8] parse_mode + verdict", flush=True)
for k in ("RANK", "WORLD_SIZE"):
    os.environ.pop(k, None)
sys.path.insert(0, os.path.join(ROOT, "bench"))
sys.path.insert(0, os.path.join(ROOT, "ops"))
import bench_l2win as B  # noqa: E402
import verdict_l2win as V  # noqa: E402

check(B.parse_mode("none") == {"kind": "none"}, "parse none")
check(B.parse_mode("d1.5")["delay_us"] == 1.5, "parse d")
c = B.parse_mode("p150_c64_d40_x26")
check((c["gbps"], c["chunk"], c["delay_us"], c["cap_us"]) == (150.0, 65536, 40.0, 26.0), "parse p")
c = B.parse_mode("s85_c64+b150")
check((c["gbps"], c["chunk"], c["big_gbps"], c["big_delay_us"]) == (85.0, 65536, 150.0, None), "parse s+b")
c = B.parse_mode("s85_c16+d3")
check((c["big_gbps"], c["big_delay_us"]) == (0.0, 3.0), "parse s+d")
try:
    B.parse_mode("x1")
    check(False, "bad mode must raise")
except ValueError:
    pass
check(abs(sum(V.WEIGHTS.values()) - 1.0) < 1e-9, "weights sum to 1")
for m in B.DEFAULT_MODES.split(",") + B.FORCED.split(","):
    B.parse_mode(m)
    V.mode_env(m, B.parse_mode)
check(V.mode_env("p125_c64", B.parse_mode) == ("DSV41_L2_PREFETCH_PACE_GBPS=125 DSV41_L2_PREFETCH_PACE_KINDS=ar "
                                               "DSV41_L2_PREFETCH_PACE_CHUNK=65536", False), "env-only mapping")
check(V.mode_env("s85_c16", B.parse_mode)[1] is True, "s modes need the kit module")
check(V.mode_env("d1", B.parse_mode) == ("DSV41_L2_PREFETCH_BURST_DELAY_US=1", True), "d mapping")


def row(tb, ck, r, ach, full=30, n=30):
    return dict(table=tb, chunk_kib=ck, req_gbps=r, achieved_gbps=ach, full=full, n=n)


ceil_ok = [row(t, 64, r, r) for t in ("short", "gate", "big") for r in (85, 125, 150)] + \
          [row(t, 16, 85, 85) for t in ("short", "gate", "big")]
good = {1: -1.0, 2: -5.0, 4: -5.0, 6: -5.0, 8: -5.0, 12: 0.0, 16: 0.0, 24: 0.0}
pm = B.parse_mode
d = V.decide("p125_c64", pm("p125_c64"), good, ceil_ok, 0)
check(d["verdict"] == "GO" and abs(d["saved_ms"] - 0.2) < 1e-6, f"GO path {d}")
check(V.decide("p125_c64", pm("p125_c64"), good, ceil_ok, 1)["verdict"] == "KILL", "mismatch kills")
d = V.decide("p125_c64", pm("p125_c64"), good, [row(t, 64, 125, 100) for t in ("short", "gate", "big")], 0)
check(d["verdict"] == "KILL" and "rate" in d["why"], f"rate kill {d}")
d = V.decide("p125_c64", pm("p125_c64"), good, [row(t, 64, 125, 125, full=29) for t in ("short", "gate", "big")], 0)
check(d["verdict"] == "KILL" and "issued" in d["why"], f"cap cut kill {d}")
d = V.decide("p125_c64", pm("p125_c64"), good, [], 0)
check(d["verdict"] == "KILL" and "no phase C" in d["why"], f"missing rate row {d}")
bad = dict(good)
bad[6] = 2.0
check("late" in V.decide("p125_c64", pm("p125_c64"), bad, ceil_ok, 0)["why"], "late M=6 kills")
bad = dict(good)
bad[24] = 4.0
check("late" in V.decide("p125_c64", pm("p125_c64"), bad, ceil_ok, 0)["why"], "late M=24 kills")
small = {m: -2.0 for m in good}
check(V.decide("p125_c64", pm("p125_c64"), small, ceil_ok, 0)["verdict"] == "HOLD", "0.06 .. 0.12 ms holds")
tiny = {m: -0.5 for m in good}
check(V.decide("p125_c64", pm("p125_c64"), tiny, ceil_ok, 0)["verdict"] == "KILL", "below 0.06 ms kills")
check(V.decide("d1", pm("d1"), good, [], 0)["verdict"] == "GO", "d modes need no phase C row")
check(V.decide("s85_c16", pm("s85_c16"), good, ceil_ok, 0)["verdict"] == "GO", "s mode only needs its small rate")
check(V.decide("p125_c64", pm("p125_c64"), {2: -1.0}, ceil_ok, 0)["verdict"] == "HOLD", "missing W rows hold")

print("RESULT", "PASS" if not FAILS else f"FAIL ({len(FAILS)})", flush=True)
sys.exit(1 if FAILS else 0)
