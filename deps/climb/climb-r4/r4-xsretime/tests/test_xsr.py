#!/usr/bin/env python3
"""r4-xsretime CPU tests. Run: python3 tests/test_xsr.py  (torch CPU build is enough; also collected by pytest).

Drives the live base (8bc9a1b5 = r1-pf on 63737f9c) and the kit's patched copy through the harness's fake CUDA
(streams, waits, capture flag) and fake native libraries, with the launcher b22e4aa9 l2pf environment, and compares
the full launch logs (op, stream, segment table, chunk, period, delay, cap; waits; collectives) of every capture."""
from __future__ import annotations

import ast
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402
from harness import LOG, MI, PROD_ENV, Sim  # noqa: E402

KIT = H.KIT
TAG = "# r4-xsretime"
REL_BASE = "distributed/device_communicators/dsv41_l2_prefetch.py"
REL_MOD = "distributed/device_communicators/dsv41_l2pf_xsr.py"
MASTERS = ("DSV41_L2_PREFETCH_PACE_GBPS_ATTN", "DSV41_L2_PREFETCH_PACE_ATTN_STREAM", "DSV41_L2_PREFETCH_ATTN_DEDUP")
CHUNK = 65536
P125 = int(round(CHUNK * 1000.0 / 125))
SIZES = (1, 4, 6)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def kit_json():
    return json.loads((KIT / "KIT.json").read_text())


def installer():
    import importlib.util

    spec = importlib.util.spec_from_file_location("xsr_install", KIT / "install.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def logger_lines():
    import vllm.logger as lg

    return list(lg.LINES)


def normalize(log, sim):
    """Replace every device address in launch tables by (allocation key, offset): the harness's named fake
    allocations are deterministic, but r1-pf KVSRC reads real CPU tensors whose addresses differ per run."""
    ranges = [((name, i), p, p + 64 * MI) for (name, i), p in sim._ptrs.items()]
    ranges += [(("keep", j), t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()) for j, t in enumerate(sim.keep)]
    ranges.sort(key=lambda r: r[1])

    def key(ptr):
        best = None
        for k, a, b in ranges:
            if a <= ptr < b and (best is None or a > best[1]):
                best = (k, a)
        assert best is not None, hex(ptr)
        return best[0], ptr - best[1]

    out = []
    for e in log:
        if e[0] in ("burst", "paced"):
            e = (e[0], e[1], tuple(key(p) + (t,) for p, t in e[2])) + tuple(e[3:])
        out.append(e)
    return out


def run_sim(which, env, sizes=SIZES):
    B = H.load(which, env)
    sim = Sim(B)
    sim.install()
    sim.capture_target(sizes)
    return B, sim, normalize(list(LOG), sim), logger_lines()


def section(log, n):
    i = log.index(("CAPTURE", "target", n))
    return log[i + 1:log.index(("END", "target", n), i)]


def attn_launches(sec):
    """Launches right after a wq_b fork: the base logs no wq_b marker, so attn launches are the ones whose table
    starts with a wo_a pointer; we identify them by being issued between two COLL entries after the 'ag' COLL
    and carrying wo_a bytes. Simpler and exact: compare against the flag-off log position by position."""
    return [e for e in sec if e[0] in ("burst", "paced")]


def split_attn(B, sec_off, sec_on):
    """Pair the flag-off and flag-on launch lists position by position (the kit never adds or removes a fork)."""
    a = [e for e in sec_off if e[0] in ("burst", "paced")]
    b = [e for e in sec_on if e[0] in ("burst", "paced")]
    assert len(a) == len(b), (len(a), len(b))
    return list(zip(a, b))


def is_attn(entry, sim=None):
    """An attn window launch: its first segment is a wo_a allocation (normalized key (name, i))."""
    k = entry[2][0][0]
    return isinstance(k[0], str) and k[0].endswith(".attn.wo_a")


def bytes_of(entry):
    return sum(t for _k, _o, t in entry[2])


def byteset(entry):
    out = set()
    for k, o, t in entry[2]:
        out.update((k, x) for x in range(o, o + t, 16))
    return out


# =============================================================================== anchors / installer
def test_base_sha_matches_kit():
    k = kit_json()
    fb = [f for f in k["files"] if f["path"] == REL_BASE][0]
    fm = [f for f in k["files"] if f["path"] == REL_MOD][0]
    assert sha(H.BASE_LIVE) == fb["base_sha256"] == "8bc9a1b5b73c808580a350fe8618fd6abb34a6f629de11c2053f640a7cd15596"
    assert sha(H.PATCHED) == fb["patched_sha256"]
    assert sha(H.MODDIR / "dsv41_l2pf_xsr.py") == fm["kit_sha256"]
    assert sha(H.R1_FILE) == "5af558fe96edfcd630caa3e673b71788587649a4e0bf9b45b4e8963b39e10548"


def test_anchors_apply_to_live_base():
    m = installer()
    src = H.BASE_LIVE.read_text()
    for old, _new, count in m.EDITS:
        assert src.count(old) == count, old[:60]
    out = m.patched(src, "live")
    assert out == H.PATCHED.read_text()
    assert out.count(TAG) == 24


def test_flag_off_text_and_ast_identical():
    """The patched file minus its tagged lines is the live base byte for byte; and the AST of the patched file with
    every `if _XSR is not None...` statement, the `_xp` branch and the gate block removed equals the base AST."""
    base = H.BASE_LIVE.read_text()
    pat = H.PATCHED.read_text()
    stripped = "".join(l for l in pat.splitlines(keepends=True) if TAG not in l)
    assert stripped == base

    class Strip(ast.NodeTransformer):
        def _keep(self, n):
            src = ast.unparse(n)
            return not ("_XSR" in src and (isinstance(n, (ast.If, ast.Assign)) or "dsv41_l2pf_xsr" in src))

        def generic_visit(self, node):
            super().generic_visit(node)
            for field in ("body", "orelse", "finalbody"):
                v = getattr(node, field, None)
                if isinstance(v, list):
                    setattr(node, field, [n for n in v if not isinstance(n, ast.stmt) or self._keep(n)])
            return node

    a = ast.dump(Strip().visit(ast.parse(pat)))
    b = ast.dump(ast.parse(base))
    assert a == b


def test_installer_cycle():
    """install / idempotent install / status / foreign edit refusal / uninstall / drifted base refusal on a copy."""
    py = sys.executable
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "vllm" / "distributed" / "device_communicators"
        d.mkdir(parents=True)
        shutil.copy2(H.BASE_LIVE, d / "dsv41_l2_prefetch.py")
        env = dict(os.environ, XSR_VLLM=str(Path(td) / "vllm"))

        def run(*a):
            p = subprocess.run([py, str(KIT / "install.py"), *a], env=env, capture_output=True, text=True)
            return p.returncode, p.stdout + p.stderr

        rc, out = run("status"); assert rc == 0 and "CLEAN" in out, out
        rc, out = run("dry"); assert rc == 0 and out.count(TAG) == 24, out
        rc, out = run("install"); assert rc == 0 and "24 tagged lines" in out, out
        rc, out = run("install"); assert rc == 0 and "already patched" in out, out
        rc, out = run("status"); assert rc == 0 and "INSTALLED" in out, out
        assert sha(d / "dsv41_l2_prefetch.py") == sha(H.PATCHED)
        assert sha(d / "dsv41_l2pf_xsr.py") == sha(H.MODDIR / "dsv41_l2pf_xsr.py")
        with open(d / "dsv41_l2_prefetch.py", "a") as f:
            f.write("# foreign\n")
        rc, out = run("status"); assert rc == 2 and "INCONSISTENT" in out, out
        rc, out = run("uninstall"); assert rc != 0 and "refusing" in out, out
        p = d / "dsv41_l2_prefetch.py"
        p.write_text(p.read_text()[:-len("# foreign\n")])
        rc, out = run("uninstall"); assert rc == 0 and "restored" in out, out
        rc, out = run("status"); assert rc == 0 and "CLEAN" in out, out
        assert sha(p) == sha(H.BASE_LIVE)
        with open(p, "a") as f:
            f.write("# drift\n")
        rc, out = run("install"); assert rc != 0 and "refusing" in out, out
        assert TAG not in p.read_text()


# =============================================================================== flag-off / inactive
def test_flag_off_module_not_imported():
    B = H.load("patched", PROD_ENV)
    assert B._XSR is None and H.XSR_NAME not in sys.modules
    B = H.load("patched", dict(PROD_ENV, DSV41_L2_PREFETCH="0", DSV41_L2_PREFETCH_PACE_GBPS_ATTN="200"))
    assert B._XSR is None and H.XSR_NAME not in sys.modules  # main gate off: never imported
    for v in ("0", "", "off", "false"):
        B = H.load("patched", dict(PROD_ENV, **{k: v for k in MASTERS}))
        assert B._XSR is None and H.XSR_NAME not in sys.modules, v


INACTIVE = {
    "GBPS_ATTN=0.0": dict(PROD_ENV, DSV41_L2_PREFETCH_PACE_GBPS_ATTN="0.0"),
    "GBPS_ATTN=abc (invalid)": dict(PROD_ENV, DSV41_L2_PREFETCH_PACE_GBPS_ATTN="abc"),
    "GBPS_ATTN=2000 (> 400 bound)": dict(PROD_ENV, DSV41_L2_PREFETCH_PACE_GBPS_ATTN="2000"),
    "rate+stream, attn not paced": dict(PROD_ENV, DSV41_L2_PREFETCH_PACE_KINDS="ar,ag",
                                        DSV41_L2_PREFETCH_PACE_GBPS_ATTN="200", DSV41_L2_PREFETCH_PACE_ATTN_STREAM="1"),
    "rate+stream, no pacing": dict(PROD_ENV, DSV41_L2_PREFETCH_PACE_GBPS="0", DSV41_L2_PREFETCH_PACE_GBPS_ATTN="200",
                                   DSV41_L2_PREFETCH_PACE_ATTN_STREAM="1"),
}


def test_inactive_settings_identical_to_live():
    """Master flag set but nothing to do: the module logs INACTIVE, the base drops it (_XSR None), and the launch log
    of every captured size equals the live base's."""
    for name, env in INACTIVE.items():
        base_env = {k: v for k, v in env.items() if k not in MASTERS}
        _b, _s, off, _l = run_sim("base", base_env)
        B, _s, on, lines = run_sim("patched", env)
        assert B._XSR is None, name
        assert on == off, name
        assert any("r4-xsretime" in l and "INACTIVE" in l for l in lines), (name, lines[-5:])


def test_flag_off_identical_launch_sequence():
    for env in (PROD_ENV, dict(PROD_ENV, DSV41_L2_PREFETCH_PACE_HEAD_KB="512"),
                dict(PROD_ENV, DSV41_L2_PREFETCH_REPLAN="attn"), dict(PROD_ENV, DSV41_L2_PREFETCH_WOB_MB="3")):
        _b, _s, off, _l = run_sim("base", env)
        _b, _s, on, _l = run_sim("patched", env)
        assert on == off
        assert sum(1 for e in off if e[0] == "paced") > 0


# =============================================================================== levers
def _pairs(env_on, env_off=PROD_ENV):
    _b, sim0, off, _ = run_sim("base", env_off)
    B, sim, on, lines = run_sim("patched", env_on)
    assert B._XSR is not None
    return B, sim0, sim, off, on, lines


def test_rate_only_attn_windows_change_period():
    B, sim0, sim, off, on, lines = _pairs(dict(PROD_ENV, DSV41_L2_PREFETCH_PACE_GBPS_ATTN="200"))
    p200 = int(round(CHUNK * 1000.0 / 200))
    n_attn = 0
    for n in SIZES:
        so, sn = section(off, n), section(on, n)
        assert [e for e in so if e[0] not in ("burst", "paced")] == [e for e in sn if e[0] not in ("burst", "paced")]
        for a, b in split_attn(B, so, sn):
            if is_attn(a, sim0):
                n_attn += 1
                assert a[0] == b[0] == "paced" and a[1] == b[1] == "side3"
                assert a[2] == b[2]  # same bytes, same order
                assert a[4] == P125 and b[4] == p200
                assert a[3] == b[3] and a[5:] == b[5:]  # chunk, delay, cap unchanged
            else:
                assert a == b
    assert n_attn == len(SIZES) * len(Sim.TARGET_LAYERS)
    assert any("g2 captured" in l and "attn_y" in l for l in lines)
    assert not any("MISMATCH" in l for l in lines)


def test_own_stream_moves_attn_pacer_and_joins_at_end():
    B, sim0, sim, off, on, lines = _pairs(dict(PROD_ENV, DSV41_L2_PREFETCH_PACE_ATTN_STREAM="1"))
    for n in SIZES:
        so, sn = section(off, n), section(on, n)
        pairs = split_attn(B, so, sn)
        for a, b in pairs:
            if is_attn(a, sim0):
                assert a[1] == "side3" and b[1] == "side4"
                assert a[2:] == b[2:]  # same table, chunk, period (base 125), delay, cap
            else:
                assert a == b
        # every side4 launch is preceded by its fork wait on main, and the forward ends with main waiting on side4
        for i, e in enumerate(sn):
            if e[0] == "paced" and e[1] == "side4":
                assert sn[i - 1] == ("wait", "side4", "main")
        tail = sn[-4:]
        assert ("wait", "main", "side4") in tail, tail
        assert sn.index(("wait", "main", "side3")) < sn.index(("wait", "main", "side4"))
        # side3 no longer carries attention windows
        assert not any(e[0] == "paced" and e[1] == "side3" and is_attn(e, sim) for e in sn)
    # the own stream is created at install time (outside capture), never inside a capture
    assert 0 in sim.XSR.R.side4 and sim.XSR.R.side4[0].name == "side4"


def test_dedup_removes_owner_window_bytes_only():
    B, sim0, sim, off, on, lines = _pairs(dict(PROD_ENV, DSV41_L2_PREFETCH_ATTN_DEDUP="1"))
    removed_total = 0
    for n in SIZES:
        so, sn = section(off, n), section(on, n)
        # owner window of an attn fork = the launches of the last collective before it: its pre parts (logged right
        # before its COLL entry, after the previous attn launch / COLL) and its post parts (logged after the COLL)
        pairs = split_attn(B, so, sn)
        it = iter(pairs)
        cur, since = [], []
        for e in sn:
            if e[0] == "COLL":
                cur, since = list(since), []
            elif e[0] in ("burst", "paced"):
                a, b = next(it)
                assert b is e
                if not is_attn(a):
                    since.append(e)
                    cur.append(e)
                else:
                    since = []
                if is_attn(a, sim0):
                    owner = set().union(*[byteset(x) for x in cur]) if cur else set()
                    ba, bb = byteset(a), byteset(b)
                    assert bb <= ba  # only shrinks
                    assert not (bb & owner)  # nothing the owner window already issued
                    assert ba - bb <= owner  # everything removed was in the owner window
                    removed_total += bytes_of(a) - bytes_of(b)
                    assert a[1] == b[1] and a[3:] == b[3:]
                else:
                    assert a == b
    # sim: SWA layers' all-gather window = wq_b 5.16 MiB + wo_b 0.84 MiB (as the live tokens=4 plan); indexer layers
    # carry the indexer wq_b there instead, so only the 2 plain SWA layers (0, 4) dedup; swa_engram's AG window is the
    # Engram gather, whose window holds no wo_b
    per = 6 * MI - ((int(0.16 * MI) & ~15) + 5 * MI)  # the AG window's wo_b part (scales + head of the codes)
    # 3 qualifying layers (0 swa, 1 swa_engram: its qkv_a gather follows the Engram gather, 4 swa) x 3 captures
    assert removed_total == len(SIZES) * 3 * per, (removed_total, per)
    assert any("xsr_dedup" in l for l in lines)
    assert not any("MISMATCH" in l for l in lines)
    print(f"    dedup removed {removed_total / MI:.2f} MiB over {len(SIZES)} captures "
          f"(expected ~{per / MI:.2f} MiB x qualifying layers)")


def test_all_three_combined():
    env = dict(PROD_ENV, DSV41_L2_PREFETCH_PACE_GBPS_ATTN="160", DSV41_L2_PREFETCH_PACE_ATTN_STREAM="1",
               DSV41_L2_PREFETCH_ATTN_DEDUP="1")
    B, sim0, sim, off, on, lines = _pairs(env)
    p160 = int(round(CHUNK * 1000.0 / 160))
    for n in SIZES:
        for a, b in split_attn(B, section(off, n), section(on, n)):
            if is_attn(a, sim0):
                assert b[1] == "side4" and b[4] == p160 and byteset(b) <= byteset(a)
            else:
                assert a == b


def test_pace_head_kept_as_base_burst():
    env0 = dict(PROD_ENV, DSV41_L2_PREFETCH_PACE_HEAD_KB="256")
    B, sim0, sim, off, on, lines = _pairs(dict(env0, DSV41_L2_PREFETCH_PACE_GBPS_ATTN="200"), env0)
    p200 = int(round(CHUNK * 1000.0 / 200))
    seen = 0
    for n in SIZES:
        for a, b in split_attn(B, section(off, n), section(on, n)):
            if a[0] == "burst" and a[1] == "side2":
                assert a == b  # the burst head of every attn window is unchanged
                seen += 1
            elif is_attn(a, sim0) or (a[0] == "paced" and a[1] == "side3" and a[4] == P125 and b[4] == p200):
                assert a[2] == b[2]
            else:
                assert a == b
    assert seen > 0


def test_pacer_library_missing_falls_back_to_base():
    """S.pace_ok False (libdsv41_l2pf_g2.so missing): attn_parts returns None and the base burst fallback is used,
    exactly as without the kit."""
    env = dict(PROD_ENV, DSV41_L2_PREFETCH_PACE_GBPS_ATTN="200", DSV41_L2_PREFETCH_PACE_ATTN_STREAM="1")
    logs = []
    for which, e in (("base", PROD_ENV), ("patched", env)):
        B = H.load(which, e)
        sim = Sim(B)
        B.S.pace_ok = False
        sim.install()
        sim.capture_target(SIZES)
        logs.append(normalize(list(LOG), sim))
    assert logs[0] == logs[1]


def test_dedup_fails_closed():
    env = dict(PROD_ENV, DSV41_L2_PREFETCH_ATTN_DEDUP="1", DSV41_L2_PREFETCH_PACE_GBPS_ATTN="200")
    B = H.load("patched", env)
    X = sys.modules[H.XSR_NAME]

    def boom(*a, **k):
        raise RuntimeError("planted")

    X.dedup_lists = boom
    sim = Sim(B)
    sim.install()
    sim.capture_target(SIZES)
    on = normalize(list(LOG), sim)
    lines = logger_lines()
    assert X.R.active["dedup"] is False and X.R.active["rate"] is True
    _b, sim0, off, _ = run_sim("base", PROD_ENV)
    for n in SIZES:  # bytes identical to the base (dedup off), only the attn period differs
        for a, b in split_attn(B, section(off, n), section(on, n)):
            assert a[2] == b[2]
    assert any("dedup planning failed" in l for l in lines)


# =============================================================================== pure planning
def test_subtract_matches_bruteforce():
    import importlib.util

    spec = importlib.util.spec_from_file_location("xsr_pure", H.MODDIR / "dsv41_l2pf_xsr.py")
    X = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(X)
    rng = random.Random(1234)
    for _ in range(3000):
        base = 0x10000 * rng.randint(1, 50)
        segs = []
        for j in range(rng.randint(1, 4)):
            p = base + 16 * rng.randint(0, 400)
            segs.append((p, 16 * rng.randint(1, 300), f"s{j}"))
        cov = [(base + 16 * rng.randint(0, 500), 16 * rng.randint(0, 200)) for _ in range(rng.randint(0, 5))]
        out, removed = X.subtract(segs, cov)
        cset = set()
        for p, t in cov:
            cset.update(range(p, p + t, 16))
        exp_bytes = 0
        for (p, t, nm) in segs:
            keep = [a for a in range(p, p + t, 16) if a not in cset]
            exp_bytes += 16 * len(keep)
        assert sum(t for _p, t, _n in out) == exp_bytes
        assert removed == sum(t for _p, t, _n in segs) - exp_bytes
        for p, t, nm in out:
            assert p % 16 == 0 and t % 16 == 0 and t >= 16
            assert not any(a in cset for a in range(p, p + t, 16))
        # order kept: pieces of segment i precede pieces of segment i+1
        names = [nm for _p, _t, nm in out]
        assert names == sorted(names, key=lambda s: int(s[1:]))


def test_dedup_lists_owner_mapping():
    import importlib.util

    spec = importlib.util.spec_from_file_location("xsr_pure2", H.MODDIR / "dsv41_l2pf_xsr.py")
    X = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(X)
    events = [("c", 1, "ar"), ("c", 2, "ag"), ("L", 1), ("c", 3, "ar"), ("c", 4, "ag"), ("L", 2), ("L", 3)]
    fl = {"pre": {2: [(0x1000, 0x100, "wq_b"), (0x8000, 0x40, "wo_b")], 4: [(0x9000, 0x30, "wo_b")]},
          "post": {}, "attn": {1: [(0x7000, 0x100, "wo_a"), (0x8000, 0x80, "wo_b")],
                                3: [(0x9000, 0x40, "wo_b")]}, "kind": {}, "moved": {}, "nl": 3}
    new, removed = X.dedup_lists(fl, events)
    assert new["attn"][1] == [(0x7000, 0x100, "wo_a"), (0x8040, 0x40, "wo_b")]
    assert new["attn"][3] == [(0x9030, 0x10, "wo_b")]
    assert removed == 0x40 + 0x30 and new["moved"] == {"xsr_dedup": 0x70}
    assert fl["attn"][1] == [(0x7000, 0x100, "wo_a"), (0x8000, 0x80, "wo_b")]  # input untouched
    # a window fully covered disappears (no fork), never grows
    fl2 = dict(fl, attn={1: [(0x8000, 0x40, "wo_b")]})
    new2, _ = X.dedup_lists(fl2, events)
    assert 1 not in new2["attn"]


# =============================================================================== runner
def main():
    tests = [(k, v) for k, v in sorted(globals().items(), key=lambda kv: kv[1].__code__.co_firstlineno
                                       if callable(kv[1]) and hasattr(kv[1], "__code__") else 0)
             if k.startswith("test_") and callable(v)]
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
