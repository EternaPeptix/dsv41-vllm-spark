#!/usr/bin/env python3
"""r1-pf CPU tests. Run: python3 tests/test_r1pf.py  (torch CPU build is enough; also collected by pytest)."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402
from harness import LOG, MI, PROD_ENV, PROD_FDA_ENV, Sim, launches  # noqa: E402

MASTERS = ("DSV41_L2_PREFETCH_R1_DRAFT", "DSV41_L2_PREFETCH_R1_IDXTAIL", "DSV41_L2_PREFETCH_R1_SHW2",
           "DSV41_L2_PREFETCH_R1_KVSRC")
ENVS = {"08a249da": PROD_ENV, "fda6d05a": PROD_FDA_ENV}

KIT = H.KIT
TAG = "# r1-pf"
REL_BASE = "distributed/device_communicators/dsv41_l2_prefetch.py"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def kit_json():
    return json.loads((KIT / "KIT.json").read_text())


def installer():
    import importlib.util

    spec = importlib.util.spec_from_file_location("r1pf_install", KIT / "install.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def logger_lines():
    import vllm.logger as lg

    return list(lg.LINES)


def run_sim(which, env, drafter=True, eph=True, target_sizes=(4, 6)):
    B = H.load(which, env)
    sim = Sim(B)
    sim.install()
    sim.capture_target(target_sizes)
    if drafter:
        sim.capture_drafter()
    if eph:
        sim.eager_step_logits()
    return B, sim, list(LOG), logger_lines()


def extras(off, on):
    """Entries of `on` that are not part of an in-order match of `off` (off must be a subsequence of on)."""
    i, extra = 0, []
    for j, e in enumerate(on):
        if i < len(off) and e == off[i]:
            i += 1
        else:
            extra.append((j, e))
    assert i == len(off), f"flag-off entry {i} ({off[i] if i < len(off) else None}) missing from the flag-on log"
    return extra


def section(log, tag, n):
    i = log.index(("CAPTURE", tag, n))
    return log[i + 1:log.index(("END", tag, n), i)]


def take_named(items, budget):
    out, left = [], budget
    for ptr, nb in items:
        if left < 16:
            break
        t = min(nb, left) & ~15
        if t <= 0:
            continue
        out.append((ptr, t))
        left -= t
    return out


# =============================================================================== anchors / installer
def test_base_sha_matches_kit():
    k = kit_json()
    entry = [f for f in k["files"] if f["path"] == "vllm/" + REL_BASE or f["path"] == REL_BASE][0]
    assert sha(H.BASE_LIVE) == entry["base_sha256"], (sha(H.BASE_LIVE), entry["base_sha256"])
    assert sha(H.PATCHED) == entry["patched_sha256"]
    m = [f for f in k["files"] if f["path"].endswith("dsv41_l2pf_r1.py")][0]
    assert sha(H.R1_FILE) == m["kit_sha256"]


def test_anchors_apply_to_live_base():
    ins = installer()
    src = H.BASE_LIVE.read_text()
    for old, _new, count in ins.EDITS:
        assert src.count(old) == count, (src.count(old), old[:60])
    out = ins.patched(src)
    assert out == H.PATCHED.read_text(), "kit patched file != installer(patch(live base))"
    assert out.count(TAG) == 16
    stripped = "".join(l for l in out.splitlines(True) if TAG not in l)
    assert stripped == src, "removing the tagged lines must give the live base byte for byte"
    compile(out, "patched", "exec")


def test_installer_cycle():
    ins_py = KIT / "install.py"
    with tempfile.TemporaryDirectory() as td:
        v = Path(td) / "vllm"
        dst = v / REL_BASE
        dst.parent.mkdir(parents=True)
        shutil.copy2(H.BASE_LIVE, dst)
        env = dict(os.environ, R1PF_VLLM=str(v))

        def run(cmd, ok=True):
            p = subprocess.run([sys.executable, str(ins_py), cmd], env=env, capture_output=True, text=True)
            assert (p.returncode == 0) == ok, (cmd, p.returncode, p.stdout, p.stderr)
            return p.stdout + p.stderr

        out = run("dry")
        assert out.count(TAG) == 16
        run("install")
        assert dst.read_text() == H.PATCHED.read_text()
        assert (dst.parent / "dsv41_l2pf_r1.py").read_bytes() == H.R1_FILE.read_bytes()
        assert "already patched" in run("install")  # idempotent
        assert dst.read_text() == H.PATCHED.read_text()
        st = run("status")
        assert "PATCHED" in st and "r1-pf status: INSTALLED" in st and "= kit_sha256" in st, st
        # uninstall refuses when the patched file was edited after install (foreign line) ...
        good = dst.read_bytes()
        dst.write_bytes(good + b"# foreign edit\n")
        assert "INCONSISTENT" in run("status", ok=False)
        out = run("uninstall", ok=False)
        assert "refusing" in out.lower() and dst.read_bytes() == good + b"# foreign edit\n", out
        assert Path(str(dst) + ".bak-r1-pf").exists() and (dst.parent / "dsv41_l2pf_r1.py").exists()
        # ... and when the module differs from kit_sha256
        dst.write_bytes(good)
        mod = dst.parent / "dsv41_l2pf_r1.py"
        mod.write_bytes(H.R1_FILE.read_bytes() + b"# x\n")
        assert "INCONSISTENT" in run("status", ok=False)
        out = run("uninstall", ok=False)
        assert "refusing" in out and sha(dst) == sha(H.PATCHED) and mod.exists(), out  # nothing touched
        p = subprocess.run([sys.executable, str(ins_py), "uninstall", "--force"], env=env, capture_output=True,
                           text=True)
        assert p.returncode == 0 and not mod.exists(), (p.stdout, p.stderr)
        assert "r1-pf status: CLEAN" in run("status")
        # --force restores over a foreign edit
        run("install")
        dst.write_bytes(dst.read_bytes() + b"# foreign edit\n")
        p = subprocess.run([sys.executable, str(ins_py), "uninstall", "--force"], env=env, capture_output=True,
                           text=True)
        assert p.returncode == 0 and "--force" in p.stdout, (p.stdout, p.stderr)
        # plain cycle
        run("install")
        run("uninstall")
        assert sha(dst) == sha(H.BASE_LIVE)
        assert not (dst.parent / "dsv41_l2pf_r1.py").exists()
        assert not Path(str(dst) + ".bak-r1-pf").exists()
        assert "r1-pf status: CLEAN" in run("status")
        dst.write_text(H.BASE_LIVE.read_text() + "\n# drift\n")
        out = run("install", ok=False)
        assert "refusing" in out
        assert TAG not in dst.read_text()


# =============================================================================== flag off
def test_flag_off_module_not_imported():
    B = H.load("patched", PROD_ENV)
    assert B._R1 is None
    assert H.R1_NAME not in sys.modules
    B = H.load("patched", dict(PROD_ENV, DSV41_L2_PREFETCH="0", DSV41_L2_PREFETCH_R1_DRAFT="1",
                                 DSV41_L2_PREFETCH_R1_IDXTAIL="1"))
    assert B._R1 is None, "main gate off must keep r1 off"
    assert H.R1_NAME not in sys.modules
    # G01 dropped: R1_WOB (and its old sub-knobs) alone is not a master flag any more
    B = H.load("patched", dict(PROD_ENV, DSV41_L2_PREFETCH_R1_WOB="1", DSV41_L2_PREFETCH_R1_WOB_GBPS="160"))
    assert B._R1 is None and H.R1_NAME not in sys.modules
    assert B._R1_FLAGS == MASTERS


FLAG_OFF_ENVS = {
    "prod 08a249da (paced ar,attn,ag + WOB_MB=6)": PROD_ENV,
    "prod fda6d05a (paced ar)": PROD_FDA_ENV,
    "fda6d05a+wob6+paced-attn (earlier env-only arm)": dict(PROD_FDA_ENV, DSV41_L2_PREFETCH_WOB_MB="6",
                                                            DSV41_L2_PREFETCH_PACE_KINDS="ar,attn"),
    "prod+replan-qbprev": dict(PROD_ENV, DSV41_L2_PREFETCH_REPLAN="qbprev"),
    "no-g2 (pre-g130 config)": {"DSV41_L2_PREFETCH": "1", "DSV41_L2_PREFETCH_WOA": "1"},
    "sub-knobs only, masters unset": dict(PROD_ENV, DSV41_L2_PREFETCH_R1_IDXTAIL_SPLIT="4",
                                          DSV41_L2_PREFETCH_R1_WOB_GBPS="160", DSV41_L2_PREFETCH_R1_DRAFT_EPH="1"),
    "dropped R1_WOB=1 only": dict(PROD_ENV, DSV41_L2_PREFETCH_R1_WOB="1", DSV41_L2_PREFETCH_R1_WOB_MB="3"),
}


def test_flag_off_identical_launch_sequence():
    for name, env in FLAG_OFF_ENVS.items():
        Bb, _s, log_b, lines_b = run_sim("base", env)
        keys_b = {k: v["key"] for k, v in Bb.S.plans.items() if not isinstance(k[0], str)}
        Bp, _s, log_p, lines_p = run_sim("patched", env)
        keys_p = {k: v["key"] for k, v in Bp.S.plans.items() if not isinstance(k[0], str)}
        assert Bp._R1 is None, name
        assert log_b == log_p, f"{name}: launch/wait sequence differs with r1 flags off"
        assert lines_b == lines_p, f"{name}: log lines differ"
        # plan keys contain id(model); compare structure
        assert [v for _k, v in sorted(keys_b.items(), key=lambda x: x[0][1])] == \
               [v for _k, v in sorted(keys_p.items(), key=lambda x: x[0][1])], name
        assert len(launches(log_b)) > 0, name


# =============================================================================== shared planner
def _learn_events(B, sim, n=4):
    S = B.S
    S.sig, S.mode, S.idx, S.lin, S.events = ("t", n), "learn", 0, 0, []
    try:
        sim.target_forward(n)
        return S.events, S.idx
    finally:
        S.sig = S.mode = S.events = None


def test_raw_window_selection_equals_make_plan():
    combos = [{}, {"DSV41_L2_PREFETCH_PRUNE_SHARED": "0"}, {"DSV41_L2_PREFETCH_PRUNE_ROUTER": "1"},
              {"DSV41_L2_PREFETCH_SKIP_N": "100000000"}, {"DSV41_L2_PREFETCH_PRUNE_STALE_QKV": "1"},
              {"DSV41_L2_PREFETCH_MB": "12"}]
    for c in combos:
        B = H.load("patched", dict(PROD_ENV, DSV41_L2_PREFETCH_R1_IDXTAIL="1", **c))
        R1 = sys.modules[H.R1_NAME]
        sim = Sim(B)
        sim.install()
        ev, n = _learn_events(B, sim)
        if c.get("DSV41_L2_PREFETCH_PRUNE_STALE_QKV"):
            ev = [("prune_stale_qkv",) + e[1:] if e[0] == "w" and "layers.3.attn.qkv_a_shard" in e[1] else e
                  for e in ev]
        plan = B.make_plan(ev, n, device="cpu")
        for idx, _kind, items in R1.raw_windows(ev):
            segs = [(p, t) for p, t, _n in B._take_named(items, B.BUDGET)]
            if not segs:
                assert idx not in plan["win"], (c, idx)
                continue
            t, nseg, nb = plan["win"][idx]
            got = [tuple(t.tolist()[2 * i:2 * i + 2]) for i in range(nseg)]
            assert got == segs, (c, idx)


# =============================================================================== G29 idxtail
def _tail_expect(sim, L, budget=6 * MI, cap=6 * MI, split=2):
    a = int(0.16 * MI) & ~15
    wk = 128 * 1024 if "kvsrc" in Sim.TARGET_LAYERS[L] else 0
    left = budget - a - 5 * MI - wk - (int(0.16 * MI) & ~15)
    take = left & ~15
    ptr = sim._ptrs[(f"layers.{L}.attn.indexer.wq_b", 1)]
    rest = min(5 * MI - take, cap) & ~15
    step = ((rest + split - 1) // split + 15) & ~15
    out, off = [], 0
    while off < rest:
        t = min(step, rest - off) & ~15
        out.append((ptr + take + off, t))
        off += t
    return tuple(out)


def test_g29_refused_under_paced_ag():
    """08a249da paces the all-gather windows: G29 must refuse itself (INACTIVE) and change nothing."""
    for env0 in (PROD_ENV, dict(PROD_FDA_ENV, DSV41_L2_PREFETCH_PACE_KINDS="ar,ag")):
        _b, _s, off, _l = run_sim("patched", env0)
        B, sim, on, lines = run_sim("patched", dict(env0, DSV41_L2_PREFETCH_R1_IDXTAIL="1",
                                                      DSV41_L2_PREFETCH_R1_IDXTAIL_SPLIT="4"))
        R1 = sys.modules[H.R1_NAME]
        assert R1.R.active["idxtail"] is False
        lv = [ln for ln in lines if "levers" in ln and "r1-pf" in ln]
        assert lv and "INACTIVE" in lv[0] and "idxtail" in lv[0].split("INACTIVE")[1], lv
        assert off == on, "refused G29 must leave every launch/wait unchanged"
        assert not any("idxtail_n" in ln for ln in lines)


def test_g29_idxtail():
    """Burst all-gather windows (launcher fda6d05a, PACE_KINDS=ar): the designed second burst on side 1."""
    _b, _s, off, _l = run_sim("patched", PROD_FDA_ENV)
    for split in (2, 4, 1):
        env = dict(PROD_FDA_ENV, DSV41_L2_PREFETCH_R1_IDXTAIL="1", DSV41_L2_PREFETCH_R1_IDXTAIL_SPLIT=str(split))
        B, sim, on, lines = run_sim("patched", env)
        assert any("idxtail" in ln and "INACTIVE" not in ln for ln in lines if "levers" in ln)
        for n in (4, 6):
            so, sn = section(off, "target", n), section(on, "target", n)
            ex = extras(so, sn)
            # one (wait side1<-main, burst side1 tail) pair per indexer layer, right after that window's head burst
            idx_layers = [L for L, k in enumerate(Sim.TARGET_LAYERS) if "idx" in k]
            assert len(ex) == 2 * len(idx_layers), ex
            for k, L in enumerate(idx_layers):
                (j0, w), (j1, b) = ex[2 * k], ex[2 * k + 1]
                assert w == ("wait", "side1", "main") and j1 == j0 + 1
                assert b[0] == "burst" and b[1] == "side1"
                assert b[2] == _tail_expect(sim, L, split=split), (L, b[2])
                head = sn[j0 - 1]
                assert head[0] == "burst" and head[1] == "side1", "tail must follow the head burst on side 1"
                assert sn[j1 + 1] == ("COLL", "ag"), "tail is forked before the qkv_a all-gather launch"
                # joined at the next collective (post-attention AR), as the head
                nxt = sn.index(("COLL", "ar"), j1)
                assert ("wait", "main", "side1") in sn[j1:nxt]


def test_g29_pure_rules():
    B = H.load("patched", dict(PROD_FDA_ENV, DSV41_L2_PREFETCH_R1_IDXTAIL="1"))
    R1 = sys.modules[H.R1_NAME]
    sc = int(0.16 * MI) & ~15

    def ev(*reads):
        out, p = [("c", 1, "ag")], 0x10000
        for name, nb in reads:
            out.append(("w", name, [(p, nb, None)]))
            p += (nb + 4095) & ~4095
        out.append(("c", 2, "ar"))
        return out

    # cut inside a leading indexer wk: the tail keeps read order (wk rest, then the wq_b)
    e = ev(("l.attn.wq_b", sc), ("l.attn.wq_b", 5 * MI), ("l.attn.indexer.wk", MI), ("l.attn.indexer.wq_b", sc),
           ("l.attn.indexer.wq_b", 5 * MI), ("l.attn.wo_b", 5 * MI))
    t = R1.idx_tails(e, split=1)[1]
    assert [n for _p, _t, n in t] == ["l.attn.indexer.wk", "l.attn.indexer.wq_b", "l.attn.indexer.wq_b"], t
    assert t[0][0] == e[3][2][0][0] + ((MI - sc) & ~15), "tail starts where the budget cut the wk read"
    assert sum(x for _p, x, _n in t) == ((MI - ((MI - sc) & ~15)) + sc + 5 * MI) and len(t) == 3
    # more than IDXTAIL_LEAD_MAX of other reads ahead of the indexer wq_b: no tail
    e = ev(("l.attn.wq_b", sc), ("l.attn.wq_b", 5 * MI), ("l.attn.compressor.x", 3 * MI),
           ("l.attn.indexer.wq_b", sc), ("l.attn.indexer.wq_b", 5 * MI))
    assert R1.idx_tails(e) == {}
    # SWA layer (wo_b follows): no tail; AR windows never get one
    e = ev(("l.attn.wq_b", sc), ("l.attn.wq_b", 5 * MI), ("l.attn.wo_b", sc), ("l.attn.wo_b", 5 * MI))
    assert R1.idx_tails(e) == {}
    # cap and split
    e = ev(("l.attn.wq_b", sc), ("l.attn.wq_b", 5 * MI), ("l.attn.indexer.wq_b", sc), ("l.attn.indexer.wq_b", 5 * MI))
    t = R1.idx_tails(e, tail=2 * MI, split=4)[1]
    assert len(t) == 4 and sum(x for _p, x, _n in t) == 2 * MI and all(p % 16 == 0 and x % 16 == 0 for p, x, _n in t)
    assert [p for p, _x, _n in t] == sorted(p for p, _x, _n in t)


# =============================================================================== G15 shw2
def test_g15_shared_w2_post_fork():
    for env0 in ENVS.values():
        _g15_case(env0)
    _g15_refused()


def _g15_case(env0):
    _b, _s, off, _l = run_sim("patched", env0)
    B, sim, on, lines = run_sim("patched", dict(env0, DSV41_L2_PREFETCH_R1_SHW2="1"))
    for n in (4, 6):
        so, sn = section(off, "target", n), section(on, "target", n)
        ex = extras(so, sn)
        assert len(ex) == 3 * len(Sim.TARGET_LAYERS), len(ex)
        for L in range(len(Sim.TARGET_LAYERS)):
            (j0, w), (j1, b), (j2, jn) = ex[3 * L:3 * L + 3]
            assert w == ("wait", "side1", "main") and b[:2] == ("burst", "side1")
            assert sn[j0 - 1] == ("COLL", "ar"), "post fork right after the post-attention all-reduce launch"
            s0 = sim._ptrs[(f"layers.{L}.ffn.shared_experts.down_proj", 0)]
            s1 = sim._ptrs[(f"layers.{L}.ffn.shared_experts.down_proj", 1)]
            assert b[2] == ((s0, int(0.05 * MI) & ~15), (s1, int(1.47 * MI) & ~15)), b[2]
            nxt = [e for e in sn[j2 + 1:] if e[0] == "COLL"][0]
            assert jn == ("wait", "main", "side1") and nxt == ("COLL", "ar"), "joined at the post-MoE AR"
            assert [e for e in sn[j0:j2] if e[0] == "COLL"] == [], "no collective between fork and join"


def _g15_refused():
    # refused without PRUNE_SHARED (the AR windows would already hold shared hints)
    _b, _s, off2, _l = run_sim("patched", dict(PROD_ENV, DSV41_L2_PREFETCH_PRUNE_SHARED="0"))
    B, sim, on2, lines = run_sim("patched", dict(PROD_ENV, DSV41_L2_PREFETCH_PRUNE_SHARED="0",
                                                   DSV41_L2_PREFETCH_R1_SHW2="1"))
    assert off2 == on2 and any("INACTIVE" in ln and "shw2" in ln for ln in lines)


# =============================================================================== G11a kvsrc
def test_g11_kvsrc_compressor_in_post_moe_window():
    for env0 in ENVS.values():
        _g11_case(env0)


def _g11_case(env0):
    _b, _s, off, _l = run_sim("patched", env0)
    B, sim, on, lines = run_sim("patched", dict(env0, DSV41_L2_PREFETCH_R1_KVSRC="1"))
    kv_layers = [L for L, k in enumerate(Sim.TARGET_LAYERS) if "kvsrc" in k]
    for n in (4, 6):
        so, sn = section(off, "target", n), section(on, "target", n)
        ex = extras(so, sn)
        assert len(ex) == 2 * len(kv_layers), ex
        for k, L in enumerate(kv_layers):
            (j0, w), (j1, p) = ex[2 * k:2 * k + 2]
            assert w == ("wait", "side3", "main") and p[:2] == ("paced", "side3")
            wt = sim.attn[L].compressor.fused_wkv_wgate.weight
            assert p[2] == ((wt.data_ptr(), int(2.5 * MI) & ~15),), p[2]
            assert p[4] == round(65536 * 1000 / 125) and p[6] == 150000
            assert sn[j0 - 1][:2] == ("paced", "side3"), "after the window's own paced part"
            assert sn[j1 + 1] == ("COLL", "ar"), "in a post-MoE all-reduce window"
            # the engram layer: the nearest AR before the Engram all-gather, not the burst Engram AG window
            nxt = [e for e in sn[j1 + 2:] if e[0] == "COLL"][:2]
            if "engram" in Sim.TARGET_LAYERS[L]:  # skipped the burst Engram AG window: AR, Engram AG, qkv_a AG
                assert nxt == [("COLL", "ag"), ("COLL", "ag")], nxt
            else:  # AR, qkv_a AG, post-attention AR
                assert nxt == [("COLL", "ag"), ("COLL", "ar")], nxt
    # refused when the AR windows are not paced
    B, sim, on2, lines = run_sim("patched", dict(env0, DSV41_L2_PREFETCH_PACE_GBPS="0",
                                                   DSV41_L2_PREFETCH_REPLAN="qbpost", DSV41_L2_PREFETCH_R1_KVSRC="1"))
    assert any("INACTIVE" in ln and "kvsrc" in ln for ln in lines)


# =============================================================================== G01 dropped
def test_g01_dropped():
    """G01 (R1_WOB) was dropped: its flag must have no effect, alone or next to a live lever."""
    _b, _s, off, _l = run_sim("patched", dict(PROD_ENV, DSV41_L2_PREFETCH_R1_SHW2="1"))
    B, sim, on, lines = run_sim("patched", dict(PROD_ENV, DSV41_L2_PREFETCH_R1_SHW2="1", DSV41_L2_PREFETCH_R1_WOB="1",
                                                  DSV41_L2_PREFETCH_R1_WOB_GBPS="160"))
    R1 = sys.modules[H.R1_NAME]
    assert off == on
    assert "wob" not in R1.R.active and not hasattr(R1, "wob_tails")
    lv = [ln for ln in lines if "levers" in ln and "r1-pf" in ln][0]
    assert "dropped lever (G01)" in lv, lv
    src = H.R1_FILE.read_text()
    assert "WOB_GBPS" not in src and "WOB_BUDGET" not in src


# =============================================================================== G13 drafter + E-phase
def _drafter_expect(sim):
    P = sim._ptrs
    sc = int(0.16 * MI) & ~15
    win = {}
    win[1] = ((P[("model.layers.40.attn.qkv_a_shard", 0)], 32 * 1024),
              (P[("model.layers.40.attn.qkv_a_shard", 1)], int(1.1 * MI) & ~15),
              (P[("model.layers.40.attn.wq_b", 0)], sc), (P[("model.layers.40.attn.wq_b", 1)], 5 * MI))
    first_wob = (6 * MI - sc - 5 * MI - sc) & ~15
    win[2] = ((P[("model.layers.40.attn.wo_b", 0)], sc), (P[("model.layers.40.attn.wo_b", 1)], first_wob))
    return win


def test_g13_drafter_graph_and_ephase():
    for env0 in ENVS.values():
        _g13_case(env0)


def _g13_case(env0):
    _b, _s, off, _l = run_sim("patched", env0)
    B, sim, on, lines = run_sim("patched", dict(env0, DSV41_L2_PREFETCH_R1_DRAFT="1"))
    R1 = sys.modules[H.R1_NAME]
    # the target graphs are untouched
    for n in (4, 6):
        assert section(off, "target", n) == section(on, "target", n)
    # one plan per DSpark graph set and shape, learned at that set's own warm-up, no mismatch
    dplans = {k: v for k, v in B.S.plans.items() if isinstance(k[0], str) and k[0] == R1.DRAFT_TAG}
    assert sorted(dplans) == sorted([(R1.DRAFT_TAG, t, r, n) for t in (0, 1) for r, n in ((1, 5), (2, 10))])
    assert {v["n"] for k, v in dplans.items() if k[1] == 0} == {16} and \
           {v["n"] for k, v in dplans.items() if k[1] == 1} == {15}
    cap = [ln for ln in lines if "R1_DRAFT captured" in ln]
    assert len(cap) == 4 and not any("MISMATCH" in ln for ln in cap), cap
    for ln in cap:
        m = re.search(r"forks (\{.*?\}) \(planned (\{.*?\})", ln)
        assert json.loads(m.group(1)) == json.loads(m.group(2)), ln
    mhc = {sim._ptrs[("mhc_fn", 0)], sim._ptrs[("model.model.markov_head.markov_w1", 0)]}
    exp = _drafter_expect(sim)
    head = take_named(sim.head_expect, 5 * MI)
    for tag in ("stock", "greedy"):
        for n in (5, 10):
            sec = section(on, tag, n)
            ls = launches(sec)
            assert ls and all(e[0] == "paced" and e[1] == "side3" for e in ls), "every drafter window paced"
            assert all(p not in mhc for e in ls for p, _t in e[2]), "no mHC fn / Markov embedding hints"
            assert len(ls) == 10 + 3, len(ls)  # 9 layer collectives + embed AR window, 3 WOA windows
            colls = [i for i, e in enumerate(sec) if e[0] == "COLL"]
            # first window (embedding AR): layer-40 qkv_a shard + the wq_b bytes moved from its all-gather (qbprev)
            assert launches(sec[:colls[0]])[0][2] == exp[1]
            assert launches(sec[colls[0]:colls[1]])[0][2] == exp[2], "AG window keeps only wo_b"
            # draft head in the last model all-reduce window (10th collective)
            assert launches(sec[colls[8]:colls[9]])[-1][2] == tuple(head)
            assert sec[-1] == ("wait", "main", "side3"), "paced side stream joined at the end of the draft forward"
            assert all(e[4] == round(65536 * 1000 / 125) for e in ls)
    # E-phase: one eager paced fork after the CERT screen, on its own stream, main_proj split then ctxkv fold
    i = on.index(("EPHASE",))
    eph = on[i + 1:]
    st = R1.R.eph_stream[0].name
    assert eph[0] == ("wait", st, "main")
    assert eph[1][:2] == ("paced", st) and eph[1][2] == tuple(take_named(sim.eph_expect, 18 * MI))
    assert eph[1][6] == 200000
    assert ("EPHASE",) in off and off[off.index(("EPHASE",)) + 1:] == []
    # the next eager step reuses the table: same launch, no rebuild log
    LOG.clear()
    sim.eager_step_logits()
    assert LOG[1:] == eph[:2], LOG
    # never inside a capture
    sim.cuda.capturing = True
    LOG.clear()
    sim.eager_step_logits()
    assert LOG == [("EPHASE",)]


def test_g13_arms():
    B, sim, on, lines = run_sim("patched", dict(PROD_ENV, DSV41_L2_PREFETCH_R1_DRAFT="1",
                                                  DSV41_L2_PREFETCH_R1_DRAFT_GRAPH="0"))
    for tag in ("stock", "greedy"):
        assert launches(section(on, tag, 5)) == []
    assert launches(on[on.index(("EPHASE",)):])
    B, sim, on, lines = run_sim("patched", dict(PROD_ENV, DSV41_L2_PREFETCH_R1_DRAFT="1",
                                                  DSV41_L2_PREFETCH_R1_DRAFT_EPH="0"))
    assert launches(section(on, "greedy", 5)) and on[on.index(("EPHASE",)) + 1:] == []
    B, sim, on, lines = run_sim("patched", dict(PROD_ENV, DSV41_L2_PREFETCH_R1_DRAFT="1",
                                                  DSV41_L2_PREFETCH_R1_DRAFT_WOA_BURST="1",
                                                  DSV41_L2_PREFETCH_R1_DRAFT_QBPREV="0"))
    ls = launches(section(on, "greedy", 5))
    assert [e[:2] for e in ls].count(("burst", "side2")) == 3
    sec = section(on, "greedy", 5)
    colls = [i for i, e in enumerate(sec) if e[0] == "COLL"]
    exp = _drafter_expect(sim)
    assert launches(sec[:colls[0]])[0][2] == exp[1][:2], "no qbprev: first window keeps only the qkv_a shard"


# =============================================================================== everything on
def test_all_levers_together():
    for name, env0 in ENVS.items():
        _b, _s, off, _l = run_sim("patched", env0)
        B, sim, on, lines = run_sim("patched", dict(env0, **{k: "1" for k in MASTERS}))
        R1 = sys.modules[H.R1_NAME]
        idx_on = R1.R.active["idxtail"]
        assert idx_on == (name == "fda6d05a"), (name, R1.R.active)
        for n in (4, 6):
            ex = extras(section(off, "target", n), section(on, "target", n))
            idx_l = sum("idx" in k for k in Sim.TARGET_LAYERS) if idx_on else 0
            kv_l = sum("kvsrc" in k for k in Sim.TARGET_LAYERS)
            nl = len(Sim.TARGET_LAYERS)
            assert len(ex) == 2 * idx_l + 3 * nl + 2 * kv_l, (name, len(ex))
        tline = [ln for ln in lines if "g2 plan tokens=4" in ln][-1]
        assert "r1=" in tline and "shw2_n" in tline and "kvsrc_n" in tline
        assert ("idxtail_n" in tline) == idx_on, tline
        # the captured fork counts equal the planned ones (the base capture log line)
        for ln in lines:
            if "g2 captured tokens=" in ln:
                m = re.search(r"forks (\{.*?\}) \(planned (\{.*?\})\)", ln)
                assert json.loads(m.group(1)) == json.loads(m.group(2)), ln


def _joins_bad(sec):
    """Side-stream launches not joined back to main by the end of the section, and side-1 launches that live across
    more than one collective (the base contract: a side-1 window forked in before_collective(k), or a post fork
    after collective k-1, is joined in before_collective(k+1) / (k) respectively)."""
    bad, pend, s1, last_coll = [], {}, [], -1
    for i, e in enumerate(sec):
        if e[0] in ("burst", "paced") and e[1] != "main":
            pend[e[1]] = i
            if e[1] == "side1":
                s1.append(i)
        elif e[0] == "wait" and e[1] == "main":
            pend.pop(e[2], None)
            if e[2] == "side1":
                s1.clear()
        elif e[0] == "COLL":
            old = [j for j in s1 if j < last_coll]
            if old:
                bad.append(("side1 launch not joined at the next collective", old, i))
            last_coll = i
    if pend:
        bad.append(("unjoined at END", dict(pend)))
    return bad


def _fork_without_wait(sec):
    """launches on a side stream not preceded (since its previous launch) by side.wait_stream(main)."""
    bad, waited = [], set()
    for i, e in enumerate(sec):
        if e[0] == "wait" and e[2] == "main":
            waited.add(e[1])
        elif e[0] in ("burst", "paced"):
            if e[1] not in waited:
                bad.append(i)
            waited.discard(e[1])
    return bad


def test_stream_protocol_every_env_and_lever():
    """Reviewer's adv.py checks, as a test: in every captured section (target 4/6, both drafter sets 5/10) each side
    launch follows a wait_stream(main), every side stream is joined back before the end, side 1 before every
    collective, and the base launches are an in-order subsequence of the patched ones (r1 only adds)."""
    # the checkers themselves catch the faults they are for
    t = (("p", 1),)
    assert _joins_bad([("wait", "side1", "main"), ("burst", "side1", t, 0), ("COLL", "ar"), ("COLL", "ar"),
                       ("wait", "main", "side1")])
    assert _joins_bad([("wait", "side3", "main"), ("paced", "side3", t, 0)])
    assert _fork_without_wait([("burst", "side2", t, 0)])
    for ename, env0 in ENVS.items():
        _b, _s, base_log, _l = run_sim("base", env0)
        arms = [(k, {k: "1"}) for k in MASTERS] + [("ALL", {k: "1" for k in MASTERS})]
        for aname, extra in arms:
            _B, _sim, log, _lines = run_sim("patched", dict(env0, **extra))
            for tag, n in (("target", 4), ("target", 6), ("stock", 5), ("stock", 10), ("greedy", 5), ("greedy", 10)):
                sec = section(log, tag, n)
                assert not _joins_bad(sec), (ename, aname, tag, n, _joins_bad(sec))
                assert not _fork_without_wait(sec), (ename, aname, tag, n)
            for n in (4, 6):
                a = [e for e in section(base_log, "target", n) if e[0] in ("burst", "paced", "COLL")]
                b = iter([e for e in section(log, "target", n) if e[0] in ("burst", "paced", "COLL")])
                assert all(x in b for x in a), (ename, aname, n)


def test_fail_closed():
    """A planning exception inside r1 must switch the lever off and keep the base behaviour (no crash)."""
    # target levers: planning raises -> base plan kept, launches identical to flag-off
    _b, _s, off, _l = run_sim("patched", PROD_ENV, drafter=False, eph=False)
    B = H.load("patched", dict(PROD_ENV, DSV41_L2_PREFETCH_R1_SHW2="1", DSV41_L2_PREFETCH_R1_KVSRC="1"))
    R1 = sys.modules[H.R1_NAME]

    def boom(_events):
        raise ValueError("planner test failure")

    R1._target_adds = boom
    sim = Sim(B)
    sim.install()
    sim.capture_target((4, 6))
    lines = logger_lines()
    assert list(LOG) == off, "target planning failure must leave the base plan untouched"
    assert not R1.R.active["shw2"] and not R1.R.active["kvsrc"]
    assert any("target planning failed" in ln for ln in lines)
    # G13: make_drafter_plan raises at the first drafter warm-up -> no drafter plan, captures run unbracketed
    _b, _s, off, _l = run_sim("patched", PROD_ENV, eph=False)
    B = H.load("patched", dict(PROD_ENV, DSV41_L2_PREFETCH_R1_DRAFT="1", DSV41_L2_PREFETCH_R1_DRAFT_EPH="0"))
    R1 = sys.modules[H.R1_NAME]

    def boom2(*a, **k):
        raise ValueError("drafter planner test failure")

    R1.make_drafter_plan = boom2
    sim = Sim(B)
    sim.install()
    sim.capture_target((4, 6))
    sim.capture_drafter()
    lines = logger_lines()
    assert list(LOG) == off, "drafter planning failure must give the flag-off launch sequence"
    assert R1.R.active["draft_graph"] is False
    assert not [k for k in B.S.plans if isinstance(k[0], str)]
    assert B.S.sig is None and B.S.events is None and B.S.pending is None
    assert sum("drafter planning failed" in ln for ln in lines) == 1, lines


# =============================================================================== runner
def main():
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)]
    bad = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:  # noqa: BLE001
            bad += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"{len(tests) - bad}/{len(tests)} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
