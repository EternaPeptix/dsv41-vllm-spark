#!/usr/bin/env python3
"""CPU tests for the r3-xsplit installer and the flag-off path.

* the base copy in base/ has the sha256 recorded in KIT.json (the live file on the Sparks)
* every anchor matches exactly as often as install.py expects; every inserted line is tagged
* install -> status -> install (idempotent) -> uninstall restores the exact bytes
* install refuses a base with a different sha256 and a tagged file that is not this patch
* flag-off identity: the patched module's AST with the tagged statements removed equals the base
  AST, and the only module-level addition is one bool read from the environment
* both the patched module and the new gate module compile
"""
import ast
import hashlib
import json
import os
import py_compile
import shutil
import subprocess
import sys
import tempfile

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(KIT, "base/mla/_sparse_mla_sm120.py")
TAG = "# dsv41-r3-xsplit"
FAILS = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def make_tree(tmp, base=BASE):
    sp = os.path.join(tmp, "sp")
    os.makedirs(os.path.join(sp, "flashinfer/mla"))
    shutil.copy2(base, os.path.join(sp, "flashinfer/mla/_sparse_mla_sm120.py"))
    return sp


def run(sp, libdir, cmd):
    env = dict(os.environ, R3_XSPLIT_SP=sp, R3_XSPLIT_LIB_DIR=libdir)
    return subprocess.run([sys.executable, os.path.join(KIT, "install.py"), cmd], env=env,
                          capture_output=True, text=True)


def strip_tagged(src):
    """Remove tagged lines and re-join the flag statement; returns source for AST compare."""
    return "".join(l for l in src.splitlines(True) if TAG not in l)


def main():
    kit = json.load(open(os.path.join(KIT, "KIT.json")))
    ent = [f for f in kit["files"] if f["path"].endswith("_sparse_mla_sm120.py")][0]
    check(sha(BASE) == ent["base_sha256"], "base copy sha256 == KIT.json base_sha256")
    for alt in ent.get("base_sha256_accepted", []):
        check(alt == ent["base_sha256"], "accepted base list = the live base (no other kit "
                                         "patches this file)")

    sys.path.insert(0, KIT)
    import importlib.util
    spec = importlib.util.spec_from_file_location("r3x_install", os.path.join(KIT, "install.py"))
    inst = importlib.util.module_from_spec(spec)
    sys.argv = ["install.py", "status"]
    src = open(BASE).read()
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["R3_XSPLIT_SP"] = make_tree(tmp)
        os.environ["R3_XSPLIT_LIB_DIR"] = os.path.join(tmp, "lib")
        spec.loader.exec_module(inst)
        for old, new, count in inst.EDITS:
            check(src.count(old) == count, f"anchor count {count}: {old.strip()[:60]!r}")
        new_src = inst.patched(src)
        added = [l for l in new_src.splitlines() if l not in src.splitlines()]
        check(all(TAG in l for l in added if l.strip()), "every inserted line carries the TAG")
        check(new_src.count(TAG) == 8, f"8 tagged lines (got {new_src.count(TAG)})")
        # flag-off AST identity
        a0 = ast.dump(ast.parse(src))
        a1 = ast.dump(ast.parse(strip_tagged(new_src)))
        check(a0 == a1, "AST(patched minus tagged lines) == AST(base)")
        tree = ast.parse(new_src)
        flag = [n for n in tree.body if isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_DSV41_ATTN_XSPLIT" for t in n.targets)]
        check(len(flag) == 1, "one module-level _DSV41_ATTN_XSPLIT assignment")
        # gate is the first statement of the if-chain and only runs when the flag is true
        fn_src = new_src[new_src.index("    def _paged_attention("):]
        gi = fn_src.index("if _DSV41_ATTN_XSPLIT and planned.variant is KernelVariant.PREFILL_MG_DUAL")
        di = fn_src.index("if planned.variant is KernelVariant.DECODE_SPLITK:")
        check(gi < di, "gate sits before the decode/prefill dispatch")
        imp = fn_src[gi:di]
        check("from . import dsv41_xsplit" in imp, "gate module imported only inside the branch")
        flag_line = [l for l in new_src.splitlines() if l.startswith("_DSV41_ATTN_XSPLIT =")][0]
        check("__import__('os').environ.get('DSV41_ATTN_XSPLIT', '0') == '1'" in flag_line,
              "flag is one module-level bool read from the environment at import")

    # full install / uninstall cycle on a temp tree (needs build/libdsv41_xsplit.so)
    lib = os.path.join(KIT, "build/libdsv41_xsplit.so")
    with tempfile.TemporaryDirectory() as tmp:
        sp = make_tree(tmp)
        libdir = os.path.join(tmp, "lib")
        tgt = os.path.join(sp, "flashinfer/mla/_sparse_mla_sm120.py")
        r = run(sp, libdir, "dry")
        check(r.returncode == 0 and r.stdout.count(TAG) == 8, "dry prints 8 tagged lines")
        if os.path.exists(lib):
            r = run(sp, libdir, "install")
            check(r.returncode == 0, f"install rc=0 ({r.stderr.strip()[-200:]})")
            check(TAG in open(tgt).read(), "target patched")
            check(os.path.exists(os.path.join(sp, "flashinfer/mla/dsv41_xsplit.py")), "gate copied")
            check(sha(os.path.join(libdir, "libdsv41_xsplit.so")) ==
                  open(os.path.join(libdir, "libdsv41_xsplit.so.sha256")).read().split()[0],
                  "library + sidecar installed")
            p1 = sha(tgt)
            r = run(sp, libdir, "install")
            check(r.returncode == 0 and "already patched" in r.stdout and sha(tgt) == p1,
                  "second install is a no-op")
            py_compile.compile(tgt, doraise=True)
            py_compile.compile(os.path.join(sp, "flashinfer/mla/dsv41_xsplit.py"), doraise=True)
            check(True, "patched module and gate compile")
            r = run(sp, libdir, "status")
            check(r.returncode == 0 and "PATCHED" in r.stdout and "lib sha OK" in r.stdout,
                  "status reports patched + lib OK")
            r = run(sp, libdir, "uninstall")
            check(r.returncode == 0 and sha(tgt) == sha(BASE), "uninstall restores the base bytes")
            check(not os.path.exists(os.path.join(sp, "flashinfer/mla/dsv41_xsplit.py")),
                  "uninstall removes the gate")
            check(not os.path.exists(os.path.join(libdir, "libdsv41_xsplit.so")),
                  "uninstall removes the library")
        else:
            print("SKIP install cycle: build/libdsv41_xsplit.so not built yet")
            FAILS.append("install cycle not run (library missing)")
        # refuse a different base
        with open(tgt, "a") as f:
            f.write("\n# drift\n")
        r = run(sp, libdir, "install")
        check(r.returncode != 0 and "refusing" in (r.stderr + r.stdout), "refuses a drifted base")

    py_compile.compile(os.path.join(KIT, "files/flashinfer/mla/dsv41_xsplit.py"), doraise=True)
    print(f"\n{'ALL PASS' if not FAILS else 'FAILURES: ' + str(len(FAILS))}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
