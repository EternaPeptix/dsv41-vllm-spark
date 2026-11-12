#!/usr/bin/env python3
"""r5-glue CPU test of install-all.sh (review N3) on fake homes (R5ALL_TRANSPORT=local; nothing remote).

Three fake nodes carry copies of the live r2 modules (base/). Checks: status CLEAN on 3/3 (exit 0); install ->
INSTALLED on 3/3 with byte-identical files on every node; status INSTALLED 3/3; a node whose r2 module was re-copied
(HOOKS-LOST) or locally edited (FOREIGN) makes status exit 2 and preflight / install / uninstall refuse, writing
nothing; a stale .tmp-r5-glue file is refused; an install that fails on node 3 rolls nodes 1-3 back to CLEAN;
uninstall -> CLEAN 3/3, files byte-identical to base.

    python3 tests/test_install_all.py
"""
import hashlib
import os
import shutil
import stat
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VREL = "venvs/vllm032/lib/python3.12/site-packages/vllm"
AO = "models/deepseek_v4/nvidia/ops/dsv41_attnout.py"
RP = "model_executor/layers/fused_moe/dsv41_routeprep.py"
HOSTS = ["n1", "n2", "n3"]
FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name}{(': ' + detail[-2000:]) if detail and not ok else ''}", flush=True)
    if not ok:
        FAILS.append(name)


def tree(root):
    out = {}
    for d, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(d, f)
            out[os.path.relpath(p, root)] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    return out


def mk():
    fake = tempfile.mkdtemp(prefix="r5all-")
    for h in HOSTS:
        v = os.path.join(fake, h, VREL)
        for rel, src in ((AO, "base/dsv41_attnout.py"), (RP, "base/dsv41_routeprep.py")):
            os.makedirs(os.path.dirname(os.path.join(v, rel)), exist_ok=True)
            shutil.copy2(os.path.join(HERE, src), os.path.join(v, rel))
    return fake


def run(fake, cmd):
    env = dict(os.environ, R5ALL_TRANSPORT="local", R5ALL_FAKE=fake, R5ALL_HOSTS=" ".join(HOSTS),
               R5ALL_PY=sys.executable, R5ALL_LPY=sys.executable)
    r = subprocess.run(["bash", os.path.join(HERE, "install-all.sh"), cmd], env=env, capture_output=True, text=True,
                       timeout=300)
    return r.returncode, r.stdout + r.stderr


def vtree(fake, h):
    return tree(os.path.join(fake, h, VREL))


def main():
    fake = mk()
    base = vtree(fake, "n1")
    rc, out = run(fake, "status")
    check("status on base nodes: CLEAN, exit 0", rc == 0 and "OVERALL CLEAN, 3 nodes agree" in out, out)
    rc, out = run(fake, "install")
    check("install exit 0, OVERALL INSTALLED", rc == 0 and "OVERALL INSTALLED, 3 nodes agree" in out, out)
    t = [vtree(fake, h) for h in HOSTS]
    check("installed trees byte-identical on every node", t[0] == t[1] == t[2] and t[0] != base)
    check("installed tree has the two new modules and both .bak files",
          sum(k.endswith(("dsv41_r5g_rq.py", "dsv41_r5g_rp.py")) for k in t[0]) == 2
          and sum(k.endswith(".bak-r5-glue") for k in t[0]) == 2, str(sorted(t[0])))
    rc, out = run(fake, "status")
    check("status after install: INSTALLED, exit 0", rc == 0 and "OVERALL INSTALLED" in out, out)
    rc, out = run(fake, "install")
    check("install again is idempotent (exit 0, trees unchanged)",
          rc == 0 and [vtree(fake, h) for h in HOSTS] == t, out)

    # HOOKS-LOST on n2 (an r2 kit re-copied its module)
    shutil.copy2(os.path.join(HERE, "base/dsv41_routeprep.py"), os.path.join(fake, "n2", VREL, RP))
    snap = [vtree(fake, h) for h in HOSTS]
    rc, out = run(fake, "status")
    check("HOOKS-LOST node: status exit 2, n2 HOOKS-LOST", rc == 2 and "HOOKS-LOST" in out and "DISAGREE" in out, out)
    for c in ("preflight", "install", "uninstall"):
        rc, out = run(fake, c)
        check(f"HOOKS-LOST node: {c} refuses and writes nothing",
              rc != 0 and "preflight: n2 is HOOKS-LOST" in out and [vtree(fake, h) for h in HOSTS] == snap, out)
    shutil.copy2(os.path.join(fake, "n1", VREL, RP), os.path.join(fake, "n2", VREL, RP))

    # FOREIGN on n3
    p = os.path.join(fake, "n3", VREL, AO)
    keep = open(p).read()
    open(p, "a").write("# local edit\n")
    snap = [vtree(fake, h) for h in HOSTS]
    rc, out = run(fake, "uninstall")
    check("FOREIGN node: uninstall refuses and writes nothing",
          rc != 0 and "n3 is FOREIGN" in out and [vtree(fake, h) for h in HOSTS] == snap, out)
    open(p, "w").write(keep)

    # stale tmp file on n1
    q = os.path.join(fake, "n1", VREL, "models/deepseek_v4/nvidia/ops/dsv41_attnout.py.tmp-r5-glue")
    open(q, "w").write("x")
    rc, out = run(fake, "preflight")
    check("stale .tmp-r5-glue: preflight refuses", rc != 0 and "n1 is STALE-TMP" in out, out)
    os.remove(q)

    rc, out = run(fake, "uninstall")
    check("uninstall exit 0, OVERALL CLEAN", rc == 0 and "OVERALL CLEAN, 3 nodes agree" in out, out)
    check("uninstalled trees == base on every node", all(vtree(fake, h) == base for h in HOSTS),
          str([sorted(set(vtree(fake, h)) ^ set(base)) for h in HOSTS]))

    # install that fails on n3 rolls back n1..n3
    d = os.path.join(fake, "n3", VREL, "model_executor/layers/fused_moe")
    os.chmod(d, stat.S_IRUSR | stat.S_IXUSR)
    try:
        rc, out = run(fake, "install")
    finally:
        os.chmod(d, stat.S_IRWXU)
    check("failing install on n3: exit 1, rollback message", rc == 1 and "rolling back: n1 n2 n3" in out, out)
    check("after rollback every node is CLEAN (trees == base)", all(vtree(fake, h) == base for h in HOSTS),
          str([sorted(set(vtree(fake, h)) ^ set(base)) for h in HOSTS]))
    rc, out = run(fake, "status")
    check("status after rollback: CLEAN, exit 0", rc == 0 and "OVERALL CLEAN" in out, out)
    shutil.rmtree(fake, ignore_errors=True)
    print("ALL PASS" if not FAILS else f"FAILED ({len(FAILS)}): {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
