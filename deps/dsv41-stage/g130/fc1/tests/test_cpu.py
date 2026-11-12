#!/usr/bin/env python3
"""g130 fc1 CPU test. No GPU, no torch. Run on the build node: cd ~/dsv41-stage/g130/fc1 && nice -n 19 python3 -I tests/test_cpu.py
Checks, stopping at the first failure:
  1 host C++ walker test of the exact header (every tile once, walkers agree, find_G, q) and the Python model's n per G matches it
  2 each arm's anchored header patch on the pinned all-base header; deal arms have no live-walk call left and nomma keeps it
  3 site patch on the pinned base: unset flags take the production path (root, jit, manifest, gemm dispatch); bad values refuse
  4 recorder dump decoder: a synthetic ring round-trips, wrap and prefix violations are detected
  5 verdict tool on synthetic probes: GO, HOLD, KILL
  6 layout contract: the ep_scatter_1 / mfront rule on random routes always gives a live-m-block prefix
  7 every GPU script byte-compiles
  8 window fixes: sanitizer PATH and G coverage in run-gpu.sh, supervisor fetched before any stop, launcher pin,
    CLUSTER_LOCK read, taken, logged and foreign traffic asked to stop before any stop, released only after section 7's exact check
"""
import array
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
import types

KIT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KIT / "tools"))
sys.path.insert(0, str(KIT / "site"))
sys.path.insert(0, str(KIT / "tests"))
TMP = Path(tempfile.mkdtemp(dir=str(KIT)))
HDR = Path.home() / "dsv41-stage/g1/m32-bw/trees/all-base/include/deep_gemm/impls/sm120_fp8_fp4_gemm_1d1d.cuh"


def t1():
    exe = TMP / "host"
    subprocess.check_call(["g++", "-std=c++20", "-O2", "-I.", "tests/deal_host_test.cpp", "-o", str(exe)], cwd=KIT)
    out = subprocess.run([str(exe)], capture_output=True, text=True, check=True)
    assert "HOST TEST PASS" in out.stderr
    rows = [ln.split("\t") for ln in out.stdout.splitlines()[1:]]
    assert len(rows) == 421
    ns = {}
    for r in rows:
        ns[int(r[0])] = int(r[3])

    def active(K, tail=24):
        if K <= 48:
            return K
        q = (K + 47) // 48
        n = min(48, (K - tail) // (q - 1))
        return max(n, (K + q - 1) // q)
    for G, n in ns.items():
        assert active(9 * G) == n, (G, n)
    assert ns[6] == 30 and ns[9] == 48 and ns[11] == 37 and ns[17] == 43, ns
    print("1 host walker OK, python model agrees over G 0..420")


def t2():
    import make_deal_tree as mk
    src = HDR.read_text()
    assert hashlib.sha256(src.encode()).hexdigest() == mk.BASE_HDR_SHA
    for arm, v in mk.ARMS.items():
        p = mk.patch(src, arm)
        if v["deal"]:
            assert "scheduler.get_next_block" not in p and "pf_sched.get_next_block" not in p
            assert p.count("g1_deal_init();") == 3 and p.count("g1_next_block(") == 4 and "auto g1_next_block = [&]" in p  # 4 walkers
            assert p.index("cudaGridDependencySynchronize();") < p.index("g1_deal_init();"), "layout read before the PDL wait"
            assert "kG1DealOrder = %d" % v["order"] in p
        else:
            assert p.count("scheduler.get_next_block") == 3 and "not kG1NoMMA" in p
    print("2 header patches OK")


def load_site():
    import make_site
    man = json.loads((KIT / "MANIFESTS.json").read_text())
    base = (KIT / "site/base_sm121_target_m32.py").read_bytes()
    assert hashlib.sha256(base).hexdigest() == make_site.BASE_SHA
    new = make_site.patch(base.decode(), man)
    p = TMP / "patched_site.py"
    p.write_text(new)
    g = {"__name__": "patched_site", "__file__": str(p)}
    exec(compile(new, str(p), "exec"), g)
    return g, man, new, base.decode()


def t3():
    g, man, new, base = load_site()
    env0 = dict(os.environ)
    try:
        for k in ("DSV41_MOE_FC1_DEAL", "DSV41_MOE_FC1_ROUTE_REC", "DSV41_MOE_M32_G1", "DSV41_MOE_FC2_D05"):
            os.environ.pop(k, None)
        assert g["fc1_deal_arm"]() == "" and g["fc1_route_rec"]() is None
        for v in ("", "0", "off", "OFF"):
            os.environ["DSV41_MOE_FC1_DEAL"] = v
            assert g["fc1_deal_arm"]() == ""
        os.environ["DSV41_MOE_FC1_DEAL"] = "nomma"
        try:
            g["fc1_deal_arm"]()
            raise SystemExit("nomma must refuse")
        except AssertionError:
            pass
        os.environ["DSV41_MOE_FC1_DEAL"] = "deal"
        try:
            g["fc1_deal_arm"]()     # G1 unset: refuse
            raise SystemExit("deal without M32_G1=all must refuse")
        except AssertionError:
            pass
        os.environ.update(DSV41_MOE_M32_G1="all", DSV41_MOE_FC2_D05="1")
        assert g["fc1_deal_arm"]() == "deal"
        assert g["_FC1DEAL_MANIFESTS"]["deal"] == man["deal-base"] and g["_FC1DEAL_MANIFESTS"]["dealx"] == man["dealx-base"]
        assert "nomma" not in g["_FC1DEAL_MANIFESTS"]
    finally:
        os.environ.clear()
        os.environ.update(env0)
    # gemm dispatch with flags off: same native call, and fc1_rec is None
    calls = []

    class Native:
        def m_grouped_fp8_fp4_gemm_nt_contiguous(self, *a, **k):
            calls.append((a, k))
            return "r"
    k = g["TargetM32Kernel"](Native())
    assert k.fc1_rec is None
    assert k.gemm(("aq", "s"), ("bw", "sb"), "d", "lay", recipe_a=(1, 128)) == "r"
    assert calls == [((("aq", "s"), ("bw", "sb"), "d", "lay"), dict(ensure_zero_padding=False, recipe_a=(1, 128)))]
    # text: the patch only adds lines to the base, except the two it anchors on
    bl, nl = base.splitlines(), new.splitlines()
    missing = [x for x in bl if x not in nl]
    assert not missing, missing[:3]
    print("3 site patch OK, %d lines added, 0 removed" % (len(nl) - len(bl)))


def t4():
    import routes_to_hist as rh
    REC, CAP = 128, 1000
    recs = []
    for i in range(50):
        G = 9 + i % 5
        r = [0] * REC
        r[0], r[1] = 12, G
        for m in range(12):
            r[2 + 2 * m] = m if m < G else -1
            r[3 + 2 * m] = 3 if m < G else 0
        recs.append(r)
    p = TMP / "d.bin"

    def dump(rs, total):
        a = array.array("i", [REC, CAP, total, 0])
        for r in rs:
            a.extend(r)
        a.extend([0] * (REC * (CAP - len(rs))))
        p.write_bytes(a.tobytes())
    dump(recs, 50)
    h = rh.hist(str(p))
    assert h["records"] == 50 and h["prefix_violations"] == 0 and not h["wrapped"]
    assert h["by_tokens"][2][9] == 10 and sum(h["by_tokens"][2].values()) == 50, h
    h2 = rh.hist(str(p), 10, 20)
    assert sum(h2["by_tokens"][2].values()) == 10
    bad = [list(r) for r in recs]
    bad[3][2 + 2 * 1] = -1
    dump(bad, 50)
    assert rh.hist(str(p))["prefix_violations"] == 1
    dump(recs, 5000)
    assert rh.hist(str(p))["wrapped"]
    print("4 recorder decoder OK")


def t5():
    d = TMP / "ev"
    d.mkdir()
    (d / "g_prior.json").write_text(json.dumps(dict(prose={"2": {"10": 100}}, code={"4": {"17": 100}})))

    def run(sp, sc):
        rows = []
        for t, G, s in ((2, 10, sp), (4, 17, sc)):
            rows.append(dict(event="p", section="A5", tokens=t, G=G, arm="all", us=100.0))
            rows.append(dict(event="p", section="A5", tokens=t, G=G, arm="deal", us=100.0 - s, saving_us=s))
        p = d / "probe.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        o = subprocess.run([sys.executable, "-I", str(KIT / "tools/eval_probe.py"), str(p)], capture_output=True, text=True, check=True).stdout
        return [ln for ln in o.splitlines() if ln.startswith("VERDICT")][0]
    assert " GO " in run(4.0, 3.0), run(4.0, 3.0)
    assert " HOLD " in run(3.0, 3.0)
    assert " KILL " in run(1.0, 1.0)
    assert " KILL " in run(5.0, -3.0)

    def stair(step, drop=None):
        rows = [dict(event="p", section="A5", tokens=6, nmb=36, G=G, arm="all", us=50.0 + (step if G >= 9 else 0.0)) for G in range(1, 19) if G != drop]
        rows.append(dict(event="p", section="A5", tokens=6, nmb=36, G=10, arm="deal", us=50.0, saving_us=1.0))
        p = d / "stair.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        o = subprocess.run([sys.executable, "-I", str(KIT / "tools/eval_probe.py"), str(p)], capture_output=True, text=True, check=True).stdout
        return [ln for ln in o.splitlines() if ln.startswith("STAIRCASE")][0]
    assert stair(18.0).startswith("STAIRCASE holds"), stair(18.0)
    assert stair(1.0).startswith("STAIRCASE model wrong"), stair(1.0)
    assert stair(18.0, drop=5).startswith("STAIRCASE not tested"), stair(18.0, drop=5)
    print("5 verdict tool OK, staircase line OK")


def t6():
    rnd = random.Random(20261102)
    for _ in range(4000):
        E = 384
        M = rnd.choice([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96])
        hot = rnd.random() < 0.3
        ids = []
        for _t in range(M):
            ex = rnd.sample(range(6 if hot else E), 6) if hot else rnd.sample(range(E), 6)
            ids.append([e if rnd.random() > 0.1 else -1 for e in ex])
        cnt = [0] * E
        for row in ids:
            for e in row:
                if e >= 0:
                    cnt[e] += 1
        # plan: same rounding / cumsum as ep_scatter_1, full -1 over M_sum rows
        routes = 6 * M
        rows = routes + min(routes, E) * 31
        M_sum = -(-rows // 32) * 32
        lay = [-1] * M_sum
        cur = 0
        for e in range(E):
            if cnt[e]:
                for r in range(cnt[e]):
                    lay[cur + r] = e
                cur += -(-cnt[e] // 32) * 32
        assert cur <= M_sum
        live = [lay[m * 32] >= 0 for m in range(M_sum // 32)]
        G = sum(live)
        assert all(live[:G]) and not any(live[G:]), "not a prefix"
    print("6 layout prefix contract OK")


def t7():
    for f in ("tests/test_deal_gpu.py", "tests/deallib.py", "tests/test_route_rec_gpu.py", "bench/fc1_probe.py", "site/fc1_route_rec.py", "tools/rec_g1.py", "tools/reprice.py"):
        compile((KIT / f).read_text(), f, "exec")
    print("7 scripts compile")


def t8():
    import gsched
    sh = (KIT / "ops/run-gpu.sh").read_text()
    san = [ln for ln in sh.splitlines() if "compute-sanitizer --tool" in ln]
    assert len(san) == 2 and all("--replays $SAN_R --gs $SAN_G --sanitizer" in ln for ln in san), san
    assert sh.index("bin/activate") < sh.index("export PATH=/usr/local/cuda/bin:$PATH") < sh.index("compute-sanitizer --tool")
    gs = gsched.parse_gs(re.search(r"^SAN_G=(\S+?);", sh, re.M).group(1))
    replays = int(re.search(r"SAN_R=(\d+)$", sh, re.M).group(1))
    top = 36   # nmb at tokens 6
    gsched.check(gs, replays, top)
    run = [gsched.g_for(r, top, gs, None) for r in range(replays)]
    assert set(range(0, 19)) <= set(run), run
    assert gsched.parse_gs("0-2,8,9") == [0, 1, 2, 8, 9] and gsched.parse_gs("") == []
    for bad in ((list(range(19)), 18, 36), ([37], 1, 36)):
        try:
            gsched.check(*bad)
        except AssertionError:
            continue
        raise SystemExit("gsched.check must refuse %r" % (bad,))
    for top in (6, 12, 18, 24, 36, 48, 72, 96):     # the exact run's default: 240 replays reach every G
        assert {gsched.g_for(r, top, [], lambda n: 0) for r in range(240)} == set(range(top + 1)), top
    win = (KIT / "WINDOW.md").read_text()
    assert win.index("> /tmp/rel-fc1.sh") < win.index("test -s /tmp/rel-fc1.sh") < win.index("systemctl --user stop"), "supervisor must be fetched before any stop"
    assert win.count("> /tmp/rel-fc1.sh") == 1
    for text in (win, (KIT / "ops/pair.sh").read_text()):
        assert "37757dde" not in text and "--expect-launcher b3976c4b" in text
    stop = win.index("systemctl --user stop")
    sec0 = win[win.index("## 0. Setup"):win.index("## 1. Stage")]
    order = ["> /tmp/rel-fc1.sh", "|| echo NO-LOCK", "set -C; printf", "> /tmp/fc1-lock.txt", ">> $D/AGENTS_LOG.md", "cmp - /tmp/fc1-lock.txt"]
    at = [sec0.index(x) for x in order]
    assert at == sorted(at), "section 0: supervisor, read, take, copy, log, check"
    assert "CLUSTER_LOCK && echo LOCK-TAKEN" in sec0 and "action=g130 fc1 window start" in sec0 and "REQUEST: Codex and survey, stop traffic" in sec0
    assert win.index("## 0. Setup") < win.index("set -C; printf") < stop and win.count("set -C") == 1, "lock must be taken before any stop"
    sec2 = win[win.index("## 2. Stop and probe"):win.index("## 3. Exactness")]
    assert sec2.index("cmp - /tmp/fc1-lock.txt") < sec2.index("systemctl --user stop"), "lock check before the section 2 stop"
    assert "lock check" in win[win.index("## 3. Exactness"):win.index("## 4. Record")]
    assert "lock check" in win[win.index("## 4. Record"):win.index("## 5. Pinned")]
    assert "lock check" in win[win.index("## 5. Pinned"):win.index("## 6. Promotion")]
    sec7 = win[win.index("## 7. Back to production"):win.index("## Reasoning on exactness")]
    assert "lock check first" in sec7.lower() and sec7.lower().index("lock check first") < sec7.index("/tmp/rel-fc1.sh"), "section 7 must check the lock before it relaunches"
    release = sec7.index("rm $D/CLUSTER_LOCK")
    last_rel = sec7.rindex("bash /tmp/rel-fc1.sh")
    last_stop = sec7.rindex("systemctl --user stop")
    assert max(last_rel, last_stop) < sec7.index("python3 exact.py") < sec7.index('startswith(\\"a8c3394af9d28249') < release, "release only after the exact check"
    assert sec7.rindex("cmp - /tmp/fc1-lock.txt", 0, release) > sec7.index("python3 exact.py"), "release must compare the lock first"
    assert "action=g130 fc1 window done" in sec7[release:] and "keep the lock" in sec7[release:]
    assert win.count("rm $D/CLUSTER_LOCK") == 1 and "rm ~/dsv41-opt-20261021/CLUSTER_LOCK" not in win and ">| " not in win, "the only lock removal is the guarded one"
    print("8 window fixes OK")


try:
    for t in (t2, t3, t4, t5, t6, t7, t8, t1) if HDR.exists() else (t3, t4, t5, t6, t7, t8):
        t()
    print("CPU TEST PASS" + ("" if HDR.exists() else " (no node tree here: 1 and 2 skipped)"))
finally:
    subprocess.call(["rm", "-rf", str(TMP)])
