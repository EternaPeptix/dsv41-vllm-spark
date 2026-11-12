#!/usr/bin/env python3
"""r5-glue CPU test: installer on copies of the live files (base/ = byte copies of the site files, sha-checked).

Checks: dry writes nothing; install patches both files and copies both modules; idempotent; strip(tagged) == base;
every inserted line is tagged and every inserted statement sits under a gate that is False for unset/0/off/false/'';
the hooks are the last statements of the file and only rebind _launch_rq / _launch; uninstall restores exact bytes,
removes the modules and the redundant .bak; refusals (foreign base, absent parent module, file changed after install)
write nothing; HOOKS-LOST status after a parent-kit re-copy; foreign-tagged (r1-pdl) lines underneath are refused; the hook lines
execute (flag off: nothing imported; flag on: wrap called) in a sandbox namespace.
"""
import ast
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.environ.pop("R5G_VLLM", None)
import install as I  # noqa: E402

FAILS = []
BASE = {I.AO: os.path.join(HERE, "base/dsv41_attnout.py"), I.RP: os.path.join(HERE, "base/dsv41_routeprep.py")}


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name}{(': ' + detail) if detail and not ok else ''}")
    if not ok:
        FAILS.append(name)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else "-"


def tree(v):
    out = {}
    for root, _, files in os.walk(v):
        for f in files:
            p = os.path.join(root, f)
            out[os.path.relpath(p, v)] = sha(p)
    return out


def mk():
    v = tempfile.mkdtemp(prefix="r5g-")
    for rel, src in BASE.items():
        os.makedirs(os.path.dirname(os.path.join(v, rel)), exist_ok=True)
        shutil.copy2(src, os.path.join(v, rel))
    return v


def run(v, cmd):
    r = subprocess.run([sys.executable, os.path.join(HERE, "install.py"), cmd], env=dict(os.environ, R5G_VLLM=v),
                       capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def gate_false(expr_src):
    keys = ("DSV41_GLUE_RQPAD", "DSV41_GLUE_RPSPLIT")
    for val, want in [(None, False), ("0", False), ("off", False), ("false", False), ("", False), ("OFF", False),
                      (" 0 ", False), ("1", True), ("hoist", True), ("split", True), ("on", True)]:
        class E:
            environ = {} if val is None else {k: val for k in keys}
        if bool(eval(expr_src, {"os": E})) != want:
            return False
    return True


def main():
    for rel, p in BASE.items():
        check(f"base copy {rel} has the KIT.json live sha", sha(p) == I.bases()[rel])
    for f in I.kit()["files"]:
        if f["action"] == "new":
            rel = f["path"].split("/", 1)[1]
            check(f"KIT.json new_sha256 of {rel} == kit file", f["new_sha256"] == sha(I.NEW[rel]))

    v = mk()
    t0 = tree(v)
    rc, out = run(v, "dry")
    check("dry exit 0", rc == 0, out)
    check("dry writes nothing", tree(v) == t0)
    rc, out = run(v, "status")
    check("status before install: base x2, modules absent, NOT-INSTALLED", rc == 0 and out.count(" base ") == 2
          and "OVERALL NOT-INSTALLED" in out, out)
    rc, out = run(v, "install")
    check("install exit 0", rc == 0, out)
    check("status after install: INSTALLED", "OVERALL INSTALLED" in out, out)
    t1 = tree(v)
    rc, out = run(v, "install")
    check("install idempotent (no byte changes)", rc == 0 and tree(v) == t1, out)
    for rel, bp in BASE.items():
        p = os.path.join(v, rel)
        src = open(p).read()
        base = open(bp).read()
        check(f"{rel}: strip(tagged) == base", I.strip_tag(src, I.TAG) == base)
        check(f"{rel}: backup == base", sha(p + I.BAK) == sha(bp))
        added = [ln for ln in src.splitlines() if ln not in base.splitlines()]
        check(f"{rel}: every inserted line carries the tag ({len(added)} lines)",
              len(added) == 4 and all(ln.endswith(I.TAG) for ln in added))
        check(f"{rel}: insertion is at the end of the file", src.startswith(base) and src[len(base):].count("\n") == 4)
        mod = ast.parse(src)
        tail = mod.body[-2:]
        check(f"{rel}: hook = one flag assignment + one `if <flag>:` block",
              isinstance(tail[0], ast.Assign) and isinstance(tail[1], ast.If)
              and ast.unparse(tail[1].test) == tail[0].targets[0].id)
        check(f"{rel}: flag expression False for unset/0/off/false/'' and True for 1/hoist/split/on",
              gate_false(ast.unparse(tail[0].value)))
        body = [ast.unparse(s) for s in tail[1].body]
        want_name = "_launch_rq" if rel == I.AO else "_launch"
        check(f"{rel}: gated body imports the kit module and rebinds only {want_name}",
              len(body) == 2 and body[0].startswith("from vllm.") and body[1].startswith(f"{want_name} = _r5g_")
              and body[1].endswith(f".wrap({want_name})"), str(body))
        # exec the hook lines with a fake module namespace: flag off imports nothing, flag on calls wrap
        hook = "\n".join(ln for ln in src.splitlines() if ln.endswith(I.TAG))
        for val, expect in ((None, False), ("0", False), ("1" if rel == I.AO else "split", True)):
            called = []

            class FakeMod:
                @staticmethod
                def wrap(f):
                    called.append(f)
                    return "wrapped"

            env = {} if val is None else {("DSV41_GLUE_RQPAD" if rel == I.AO else "DSV41_GLUE_RPSPLIT"): val}

            class Os:
                environ = env
            imported = []

            def fake_import(name, g=None, lc=None, fromlist=(), level=0):
                imported.append(name)
                m = type(sys)("x")
                for a in fromlist or ():
                    setattr(m, a, FakeMod)
                return m
            ns = {"os": Os, want_name: "orig", "__builtins__": dict(__builtins__.__dict__, __import__=fake_import)}
            exec(hook, ns)
            ok = (ns[want_name] == "wrapped") == expect and bool(imported) == expect and (called == ["orig"]) == expect
            check(f"{rel}: hook executes as designed with flag={val!r} (imported={imported})", ok)
    for rel, srcp in I.NEW.items():
        check(f"module {rel} copied", sha(os.path.join(v, rel)) == sha(srcp))

    # HOOKS-LOST: a parent kit re-copies its module
    shutil.copy2(BASE[I.RP], os.path.join(v, I.RP))
    rc, out = run(v, "status")
    check("status reports HOOKS-LOST after a parent re-copy", "HOOKS-LOST" in out and "PARTIAL" in out, out)
    rc, out = run(v, "install")
    check("re-install after HOOKS-LOST restores INSTALLED", rc == 0 and "OVERALL INSTALLED" in out, out)

    # uninstall
    rc, out = run(v, "uninstall")
    check("uninstall exit 0", rc == 0, out)
    check("uninstall restores exact bytes, removes modules and redundant backups", tree(v) == t0,
          f"{set(tree(v).items()) ^ set(t0.items())}")
    rc, out = run(v, "uninstall")
    check("uninstall idempotent", rc == 0 and tree(v) == t0, out)

    # refusals write nothing
    p = os.path.join(v, I.AO)
    open(p, "a").write("# local edit\n")
    t2 = tree(v)
    rc, out = run(v, "install")
    check("install refuses a foreign base and writes nothing", rc != 0 and tree(v) == t2, out)
    shutil.copy2(BASE[I.AO], p)
    os.remove(os.path.join(v, I.RP))
    t3 = tree(v)
    rc, out = run(v, "install")
    check("install refuses when a parent module is absent and writes nothing", rc != 0 and tree(v) == t3, out)
    shutil.copy2(BASE[I.RP], os.path.join(v, I.RP))
    rc, out = run(v, "install")
    q = os.path.join(v, I.RP)
    s = open(q).read().replace("_FILL_ROWS = 64\n", "_FILL_ROWS = 128\n")
    open(q, "w").write(s)
    t4 = tree(v)
    rc, out = run(v, "uninstall")
    check("uninstall refuses when a patched file changed after install, and writes nothing", rc != 0 and tree(v) == t4,
          out)
    shutil.rmtree(v)

    # review N4: lines tagged by another kit (e.g. a future r1-pdl rebase) are REFUSED in both modules, nothing written
    check("FOREIGN_TAGS is empty", I.FOREIGN_TAGS == ())
    for rel in (I.AO, I.RP):
        v = mk()
        p = os.path.join(v, rel)
        s = open(BASE[rel]).read().replace("import os\n", "import os\n_X_R1 = 1  # r1-pdl\n", 1)
        assert "# r1-pdl" in s
        open(p, "w").write(s)
        before = tree(v)
        rc, out = run(v, "install")
        check(f"{rel}: install refuses a base carrying r1-pdl tagged lines and writes nothing",
              rc != 0 and "not an accepted base" in out and tree(v) == before, out)
        shutil.rmtree(v)
    print("ALL PASS" if not FAILS else f"FAILED ({len(FAILS)}): {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
