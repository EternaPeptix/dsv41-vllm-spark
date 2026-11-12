#!/usr/bin/env python3
"""CPU test: the installer's anchors apply to the live base files (and to the files with round-1 r1-pdl applied),
install is idempotent, refuses unknown bases, uninstall restores the exact bytes, the patched files compile.

Run: python3 tests/test_install.py   (stdlib only; the base copies in base/ were pulled from spark-one and their
sha256 equal KIT.json's recorded bases, which this test also asserts)."""
import hashlib
import importlib.util
import os
import py_compile
import shutil
import subprocess
import sys
import tempfile

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R1PDL = os.path.expanduser("~/Documents/dsv41-offline/climb-r1/r1-pdl/install.py")
FILES = ["models/deepseek_v4/nvidia/ops/o_proj.py", "models/deepseek_v4/nvidia/ops/sonar_bf16_bmm.py"]
NEWMOD = "models/deepseek_v4/nvidia/ops/dsv41_attnout.py"
TAG = "# r2-attnout"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def run(tree, cmd, ok=True):
    r = subprocess.run([sys.executable, os.path.join(KIT, "install.py"), cmd], env=dict(os.environ, R2AO_VLLM=tree),
                       capture_output=True, text=True)
    if ok and r.returncode != 0:
        raise AssertionError(f"install.py {cmd} failed: {r.stdout}{r.stderr}")
    return r


def make_tree(root):
    for rel in FILES:
        dst = os.path.join(root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(os.path.join(KIT, "base", rel), dst)
    return root


def load_r1pdl():
    spec = importlib.util.spec_from_file_location("r1pdl_install", R1PDL)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def check_cycle(tree, label):
    before = {rel: sha(os.path.join(tree, rel)) for rel in FILES}
    r = run(tree, "dry")
    assert r.stdout.count(TAG) == 15, f"{label}: dry shows {r.stdout.count(TAG)} tagged lines, expected 15 (11 o_proj + 4 sonar)"
    run(tree, "install")
    for rel in FILES:
        p = os.path.join(tree, rel)
        src = open(p).read()
        assert TAG in src, f"{label}: {rel} not patched"
        assert sha(p + ".bak-r2-attnout") == before[rel], f"{label}: backup of {rel} differs from base"
        py_compile.compile(p, doraise=True)
    newp = os.path.join(tree, NEWMOD)
    assert sha(newp) == sha(os.path.join(KIT, "files", NEWMOD)), f"{label}: new module not copied"
    py_compile.compile(newp, doraise=True)
    after = {rel: sha(os.path.join(tree, rel)) for rel in FILES}
    r = run(tree, "install")  # idempotent
    assert "already patched" in r.stdout, f"{label}: second install not idempotent: {r.stdout}"
    assert {rel: sha(os.path.join(tree, rel)) for rel in FILES} == after, f"{label}: second install changed files"
    st = run(tree, "status").stdout
    assert st.count("PATCHED") == 2 and "(kit copy)" in st, f"{label}: status wrong:\n{st}"
    run(tree, "uninstall")
    assert {rel: sha(os.path.join(tree, rel)) for rel in FILES} == before, f"{label}: uninstall did not restore bytes"
    assert not os.path.exists(newp), f"{label}: new module left behind"
    print(f"PASS {label}: dry/install/idempotent/status/uninstall; patched files compile; "
          f"patched shas o_proj={after[FILES[0]][:8]} sonar={after[FILES[1]][:8]}")
    return after


def main():
    import json

    kit = json.load(open(os.path.join(KIT, "KIT.json")))
    bases = {f["path"]: f for f in kit["files"]}
    for rel in FILES:
        assert sha(os.path.join(KIT, "base", rel)) == bases[f"vllm/{rel}"]["base_sha256"], f"base/{rel} != KIT.json"
    print("PASS base copies match KIT.json base_sha256 (= live spark-one files)")

    with tempfile.TemporaryDirectory() as t:
        check_cycle(make_tree(os.path.join(t, "live")), "live base")

    # unknown base -> refuse, nothing written
    with tempfile.TemporaryDirectory() as t:
        tree = make_tree(os.path.join(t, "mut"))
        p = os.path.join(tree, FILES[0])
        open(p, "a").write("# local edit\n")
        h = sha(p)
        r = run(tree, "install", ok=False)
        assert r.returncode != 0 and "refusing" in (r.stdout + r.stderr), "install did not refuse an unknown base"
        assert sha(p) == h and TAG not in open(os.path.join(tree, FILES[1])).read(), "refused install wrote files"
        print("PASS unknown base sha -> refused, nothing written")

    # review fix: uninstall refuses (nothing restored) when a patched file changed after install
    with tempfile.TemporaryDirectory() as t:
        tree = make_tree(os.path.join(t, "late"))
        run(tree, "install")
        p = os.path.join(tree, FILES[1])
        open(p, "a").write("# r1-pdl-like edit installed after r2-attnout\n")
        snap = {rel: sha(os.path.join(tree, rel)) for rel in FILES}
        r = run(tree, "uninstall", ok=False)
        assert r.returncode != 0 and "Refusing" in (r.stdout + r.stderr), r.stdout + r.stderr
        assert {rel: sha(os.path.join(tree, rel)) for rel in FILES} == snap, "refused uninstall changed files"
        assert all(os.path.exists(os.path.join(tree, rel) + ".bak-r2-attnout") for rel in FILES)
        assert os.path.exists(os.path.join(tree, NEWMOD)), "refused uninstall removed the module"
        print("PASS uninstall refuses when a patched file changed after install (nothing restored or removed)")

    # composition with round 1: r1-pdl patches sonar_bf16_bmm.py
    if os.path.exists(R1PDL):
        m = load_r1pdl()
        key = ("vllm", FILES[1])
        with tempfile.TemporaryDirectory() as t:
            tree = make_tree(os.path.join(t, "r1"))
            p = os.path.join(tree, FILES[1])
            txt = m.patched(key, open(p).read())
            open(p, "w").write(txt)
            rec = bases[f"vllm/{FILES[1]}"]["base_sha256_with_round1"]["r1-pdl"]
            assert sha(p) == rec, f"r1-pdl-patched sonar sha {sha(p)[:16]} != KIT.json {rec[:16]} (r1-pdl changed?)"
            check_cycle(tree, "live + r1-pdl")
            # and r1-pdl's own anchors still apply after ours (reverse order)
        with tempfile.TemporaryDirectory() as t:
            tree = make_tree(os.path.join(t, "r2first"))
            run(tree, "install")
            p = os.path.join(tree, FILES[1])
            both = m.patched(key, open(p).read())
            assert both.count(TAG) == 4 and both.count("# r1-pdl") == 7
            print("PASS r1-pdl anchors still apply on top of r2-attnout (supported order: install r1-pdl first, uninstall r2-attnout first)")
    else:
        print(f"SKIP r1-pdl composition: {R1PDL} not found")
    print("ALL PASS test_install")


if __name__ == "__main__":
    main()
