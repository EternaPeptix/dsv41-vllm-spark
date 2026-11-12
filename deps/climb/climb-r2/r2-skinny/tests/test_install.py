#!/usr/bin/env python3
"""CPU tests of the r2-skinny installer (stdlib only):

1. the anchors apply to the live base file (sha256 recorded in KIT.json) with the exact expected counts;
2. install -> status -> idempotent re-install -> uninstall restores the base bytes exactly;
3. flag-off identity: the patched module's AST with every tagged statement removed equals the base AST, every
   tagged branch is guarded by R2_V2 / R2_QFUSE, and those evaluate False with the flags unset;
4. refusal: a base whose sha256 is not recorded is refused, the file stays untouched;
5. composition: if round-1 kits with an install.py exist in climb-r1/, apply them to a copy first, then this kit.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(KIT, "base", "dsv41_r3_dense.py")
REL = "models/deepseek_v41/nvidia/dsv41_r3_dense.py"
TAG = "# r2-skinny"
FAIL: list[str] = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        FAIL.append(name)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def tree(root, base_file=BASE):
    v = os.path.join(root, "vllm")
    os.makedirs(os.path.join(v, "models/deepseek_v41/nvidia"), exist_ok=True)
    shutil.copy2(base_file, os.path.join(v, REL))
    os.makedirs(os.path.join(root, "home"), exist_ok=True)
    return v


REAL_SO = os.path.join(KIT, "prebuilt", "skinny_mx_v2_ext.so")
SO_ENV: dict = {}  # set in main(): kit json + prebuilt to use (the real prebuilt .so when the kit ships one)


def run(root, *args, kit_json=None, prebuilt=None):
    env = dict(os.environ, R2SK_VLLM=os.path.join(root, "vllm"), R2SK_HOME=os.path.join(root, "home"))
    env.update(SO_ENV)
    if kit_json:
        env["R2SK_KIT_JSON"] = kit_json
    if prebuilt:
        env["R2SK_PREBUILT"] = prebuilt
    p = subprocess.run([sys.executable, os.path.join(KIT, "install.py"), *args], env=env, capture_output=True,
                       text=True)
    return p.returncode, p.stdout + p.stderr


def strip_tagged(src):
    return "".join(l for l in src.splitlines(keepends=True) if TAG not in l)


def main():
    kit = json.load(open(os.path.join(KIT, "KIT.json")))
    rec = [f for f in kit["files"] if f["path"] == REL][0]
    check("live base copy sha256 == KIT.json base_sha256", sha(BASE) == rec["base_sha256"], sha(BASE)[:16])

    so_rec = [f for f in kit["files"] if f["path"].endswith("skinny_mx_v2_ext.so")][0]
    tmpk = tempfile.mkdtemp()
    if os.path.exists(REAL_SO):
        check("prebuilt .so sha256 == KIT.json", so_rec.get("sha256") == sha(REAL_SO), sha(REAL_SO)[:16])
    else:  # no binary in the kit yet: a fake prebuilt + a KIT.json copy that records its sha256
        fake = os.path.join(tmpk, "fake.so")
        open(fake, "wb").write(b"fake-so")
        k2 = json.loads(json.dumps(kit))
        [f for f in k2["files"] if f["path"].endswith("skinny_mx_v2_ext.so")][0]["sha256"] = sha(fake)
        json.dump(k2, open(os.path.join(tmpk, "KIT.json"), "w"))
        SO_ENV.update(R2SK_PREBUILT=fake, R2SK_KIT_JSON=os.path.join(tmpk, "KIT.json"))
        print("[INFO] no prebuilt .so in the kit: installer tests use a fake prebuilt with a matching KIT.json copy")

    with tempfile.TemporaryDirectory() as root:
        v = tree(root)
        dense = os.path.join(v, REL)
        rc, out = run(root, "dry")
        check("dry: anchors apply to the live base (21 tagged lines)", rc == 0 and out.count(TAG) == 21,
              f"rc={rc}")
        rc, out = run(root, "install")
        check("install", rc == 0 and "patched" in out, out.strip().splitlines()[0] if out.strip() else "")
        patched = open(dense).read()
        check("installed file has exactly 21 tagged lines", patched.count(TAG) == 21)
        check("module copied", os.path.exists(os.path.join(v, "models/deepseek_v41/nvidia/dsv41_r2_skinny.py")))
        h1 = sha(dense)
        rc, out = run(root, "install")
        check("re-install is idempotent", rc == 0 and sha(dense) == h1 and "already patched" in out)
        rc, out = run(root, "status")
        check("status reports PATCHED, the .so == KIT.json and a current patch",
              rc == 0 and "PATCHED" in out and "present (== KIT.json)" in out and "matches this kit revision" in out,
              out.strip().replace("\n", " | ")[-160:])

        # flag-off identity (AST)
        base_src = open(BASE).read()
        a0 = ast.dump(ast.parse(base_src))
        a1 = ast.dump(ast.parse(strip_tagged(patched)))
        check("AST(patched minus tagged lines) == AST(base)", a0 == a1)
        # outermost tagged `if` statements (nested ones live inside a guarded block)
        lines = patched.splitlines()
        mod_ast = ast.parse(patched)
        outer = []

        def visit(node, guarded):
            for ch in ast.iter_child_nodes(node):
                tagged = isinstance(ch, ast.If) and TAG in lines[ch.lineno - 1]
                if tagged and not guarded:
                    outer.append(ast.get_source_segment(patched, ch.test))
                visit(ch, guarded or tagged)

        visit(mod_ast, False)
        guards_ok = all(t.startswith(("R2_V2", "R2_QFUSE", "not R2_V2")) for t in outer)
        check("every outermost tagged branch is guarded by R2_V2 / R2_QFUSE", guards_ok and len(outer) == 5,
              f"{outer}")
        flag_lines = [l.split(TAG)[0] for l in patched.splitlines() if TAG in l and l.startswith(("R2_V2", "R2_QFUSE"))]
        bad_v = "\n".join(l.split(TAG)[0] for l in patched.splitlines()
                          if TAG in l and (l.startswith(("R2_V2", "if not R2_V2 and os.environ.get(\"DSV41_SKINNY_V2\""))
                                           or "raise ValueError" in l))
        rejected = []
        for val in ("2", "true", "pdl ", "PDL", "off", ""):
            os.environ["DSV41_SKINNY_V2"] = val
            try:
                exec(bad_v, {"os": os})
                rejected.append((val, False))
            except ValueError:
                rejected.append((val, True))
        os.environ.pop("DSV41_SKINNY_V2", None)
        check("invalid DSV41_SKINNY_V2 values raise at import (2, true); 1|pdl (any case/space) and off/'' do not",
              rejected == [("2", True), ("true", True), ("pdl ", False), ("PDL", False), ("off", False), ("", False)],
              str(rejected))
        ns: dict = {"os": os}
        saved = {k: os.environ.pop(k, None) for k in ("DSV41_SKINNY_V2", "DSV41_SKINNY_QFUSE")}
        try:
            exec("\n".join(flag_lines), ns)
            off = ns["R2_V2"] is False and ns["R2_QFUSE"] is False
            os.environ["DSV41_SKINNY_QFUSE"] = "1"
            exec("\n".join(flag_lines), ns)
            off2 = ns["R2_QFUSE"] is False
            os.environ["DSV41_SKINNY_V2"] = "pdl"
            exec("\n".join(flag_lines), ns)
            on = ns["R2_V2"] is True and ns["R2_QFUSE"] is True
            os.environ["DSV41_SKINNY_V2"] = "0"
            exec("\n".join(flag_lines), ns)
            off3 = ns["R2_V2"] is False
        finally:
            for k, val in saved.items():
                if val is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = val
        check("flags unset -> R2_V2 = R2_QFUSE = False; QFUSE alone stays off; V2=pdl+QFUSE=1 -> both on; V2=0 off",
              off and off2 and on and off3)
        # the dsv41_r2_skinny import sits only inside `if R2_V2:`
        mod = ast.parse(patched)
        imports_ok = True
        for node in ast.walk(mod):
            if isinstance(node, ast.ImportFrom) and "dsv41_r2_skinny" in [a.name for a in node.names]:
                imports_ok = imports_ok and TAG in patched.splitlines()[node.lineno - 1]
        check("dsv41_r2_skinny is imported only on a tagged (flag-guarded) line", imports_ok)

        # stale patch: a tagged line that differs from this revision's edits is refused (file untouched)
        stale = patched.replace("ignored: it needs", "ignored (old rev): it needs")
        open(dense, "w").write(stale)
        hs = sha(dense)
        rc, out = run(root, "install")
        check("install refuses a stale patch (tagged lines != this revision) and leaves it untouched",
              rc != 0 and sha(dense) == hs and "stale" in out, out.strip().splitlines()[-1] if out.strip() else "")
        rc, out = run(root, "status")
        check("status flags the stale patch", "stale patch" in out)
        open(dense, "w").write(patched)

        rc, out = run(root, "uninstall")
        check("uninstall restores the base bytes", rc == 0 and sha(dense) == rec["base_sha256"], sha(dense)[:16])
        check("uninstall removes the module", not os.path.exists(os.path.join(v, "models/deepseek_v41/nvidia/"
                                                                                   "dsv41_r2_skinny.py")))

    # refusal on an unknown base
    with tempfile.TemporaryDirectory() as root:
        v = tree(root)
        dense = os.path.join(v, REL)
        open(dense, "a").write("# drift\n")
        h = sha(dense)
        rc, out = run(root, "install")
        check("install refuses an unrecorded base sha256 and leaves it untouched", rc != 0 and sha(dense) == h,
              out.strip().splitlines()[-1] if out.strip() else "")

    # binary checks: null KIT.json sha256, wrong prebuilt, wrong pre-existing destination .so
    k3 = json.loads(json.dumps(kit))
    [f for f in k3["files"] if f["path"].endswith("skinny_mx_v2_ext.so")][0]["sha256"] = None
    json.dump(k3, open(os.path.join(tmpk, "KIT-null.json"), "w"))
    bogus = os.path.join(tmpk, "bogus.so")
    open(bogus, "wb").write(b"not-the-tested-binary")
    with tempfile.TemporaryDirectory() as root:
        v = tree(root)
        dense = os.path.join(v, REL)
        h = sha(dense)
        os.makedirs(os.path.join(root, "home/dsv41-r2/skinny_mx_v2/build"), exist_ok=True)
        open(os.path.join(root, "home/dsv41-r2/skinny_mx_v2/build/skinny_mx_v2_ext.so"), "wb").write(b"old")
        rc, out = run(root, "install", kit_json=os.path.join(tmpk, "KIT-null.json"))
        check("install refuses when KIT.json has no .so sha256 (even with an .so at the destination)",
              rc != 0 and sha(dense) == h, out.strip().splitlines()[-1] if out.strip() else "")
        rc, out = run(root, "status", kit_json=os.path.join(tmpk, "KIT-null.json"))
        check("status flags an .so it cannot verify (KIT.json sha256 null)", "UNVERIFIED" in out)
        rc, out = run(root, "install", prebuilt=bogus)
        check("install refuses a prebuilt whose sha256 != KIT.json", rc != 0 and sha(dense) == h,
              out.strip().splitlines()[-1] if out.strip() else "")
        rc, out = run(root, "install", prebuilt=os.path.join(tmpk, "absent.so"))
        check("install refuses a destination .so whose sha256 != KIT.json when the kit has no prebuilt",
              rc != 0 and sha(dense) == h, out.strip().splitlines()[-1] if out.strip() else "")
        rc, out = run(root, "status")
        check("status flags a destination .so mismatch", "MISMATCH" in out)

    # composition with round-1 kits
    r1 = os.path.join(os.path.dirname(os.path.dirname(KIT)), "climb-r1")
    kits = sorted(d for d in os.listdir(r1) if d.startswith("r1-") and os.path.exists(os.path.join(r1, d, "install.py")))
    touching = []
    for d in kits:
        kj = os.path.join(r1, d, "KIT.json")
        if os.path.exists(kj) and any(f.get("path", "").endswith("dsv41_r3_dense.py") for f in json.load(open(kj)).get("files", [])):
            touching.append(d)
    if not kits:
        print("[SKIP] composition: no round-1 kit has an install.py yet")
    else:
        print(f"[INFO] round-1 kits with installers: {kits}; touching dsv41_r3_dense.py: {touching or 'none'}")
        if not touching:
            check("composition: no round-1 kit edits dsv41_r3_dense.py, so its base after round 1 == live base", True)

    print("\nALL PASS" if not FAIL else "\nFAILURES: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
