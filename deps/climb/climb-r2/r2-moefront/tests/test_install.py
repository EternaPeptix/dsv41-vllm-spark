"""CPU tests for install.py and the flag-off contract (stdlib only).

  1. the live base files (base/, copied from spark-one; 8/8 identical) have the KIT.json sha256
  2. install on a copy of the live tree: anchors apply once, status 'patched', idempotent, uninstall restores the exact
     bytes; dry works; a foreign base is refused
  3. rule 5: same on a copy with the r1-pdl edits applied first (its install.py EDITS), and uninstall leaves r1-pdl's
     lines intact; r1-pdl's anchors are untouched by ours
  4. flag off: stripping our tagged lines gives the base byte for byte, and every tagged statement is the flag
     assignment, the gated import, or an `if _DSV41_RP ...:` whose whole body is tagged (so with the flag unset no
     inserted statement executes); the gate expression is False for unset / 0 / off / false / ''
  5. the Athena FC2 runtime transform (getsource of DeepGemmFP4Experts.apply + 3 single-anchor replacements) still
     applies to the patched apply and compiles
  6. composition with r2-draftmoe (its real install.py on a copy of its 4 live base files): moefront -> draftmoe and
     draftmoe -> moefront both install; uninstalling either kit first leaves exactly the other kit's install; both
     orders end at the live bytes; also with r1-pdl underneath
  7. uninstall is all-or-nothing: when the second file is not ours-of-an-accepted-base, nothing is written
  8. the installer works under LC_ALL=C with PYTHONUTF8=0 (explicit UTF-8 I/O; the base files hold non-ASCII bytes)
Run: python3 tests/test_install.py
"""
import ast
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R1PDL = os.path.join(os.path.dirname(os.path.dirname(KIT)), "climb-r1", "r1-pdl", "install.py")
DRAFTMOE = os.path.join(os.path.dirname(KIT), "r2-draftmoe")
TAG = "# r2-moefront"
FAIL = []


def check(name, ok, detail=""):
    detail = detail if (detail and (not ok or len(detail) < 100)) else ""
    print(f"{'PASS' if ok else 'FAIL'} {name}{(': ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAIL.append(name)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def make_tree(tmp, r1pdl=False, draftmoe_files=False):
    v = os.path.join(tmp, "vllm")
    if draftmoe_files:  # the other two files r2-draftmoe patches (live copies in its kit)
        for rel in ("model_executor/layers/quantization/mxfp4.py",
                    "model_executor/layers/fused_moe/experts/dsv41_draft_m32.py"):
            os.makedirs(os.path.dirname(os.path.join(v, rel)), exist_ok=True)
            shutil.copy2(os.path.join(DRAFTMOE, "base", rel), os.path.join(v, rel))
    for rel in ("model_executor/layers/fused_moe/deep_gemm_utils.py",
                "model_executor/layers/fused_moe/experts/deep_gemm_moe.py"):
        os.makedirs(os.path.dirname(os.path.join(v, rel)), exist_ok=True)
        shutil.copy2(os.path.join(KIT, "base", rel), os.path.join(v, rel))
    if r1pdl:
        m = load(R1PDL, "r1pdl_install")
        for (root, rel), edits in m.EDITS.items():
            p = os.path.join(v, rel)
            if root != "vllm" or not os.path.exists(p):
                continue
            src = open(p).read()
            for old, new, count in edits:
                assert src.count(old) == count, (rel, old[:60])
                src = src.replace(old, new)
            open(p, "w").write(src)
    return v


def run(v, *args, env_extra=None):
    env = dict(os.environ, R2MF_VLLM=v, **(env_extra or {}))
    r = subprocess.run([sys.executable, os.path.join(KIT, "install.py"), *args], env=env, capture_output=True,
                       text=True)
    return r.returncode, r.stdout + r.stderr


def run_dm(v, home, *args):
    env = dict(os.environ, R2DM_VLLM=v, R2DM_HOME=home)
    r = subprocess.run([sys.executable, os.path.join(DRAFTMOE, "install.py"), *args], env=env, capture_output=True,
                       text=True)
    return r.returncode, r.stdout + r.stderr


def ours_only(snap):
    return {k: h for k, h in snap.items() if ".bak-" not in k and "__pycache__" not in k}


def snapshot(v):
    out = {}
    for d, _, fs in os.walk(v):
        for f in fs:
            p = os.path.join(d, f)
            out[os.path.relpath(p, v)] = sha(p)
    return out


def test_bases():
    k = json.load(open(os.path.join(KIT, "KIT.json")))
    for f in k["files"]:
        if f["action"] != "patch":
            continue
        rel = f["path"].split("/", 1)[1]
        check(f"1 live base sha {rel}", sha(os.path.join(KIT, "base", rel)) == f["base_sha256"])


def test_install_cycle(r1pdl):
    lab = "r1-pdl+" if r1pdl else ""
    with tempfile.TemporaryDirectory() as tmp:
        v = make_tree(tmp, r1pdl)
        before = snapshot(v)
        rc, out = run(v, "dry")
        check(f"2 {lab}dry", rc == 0 and "+" in out, out.strip().splitlines()[-1] if out.strip() else "")
        check(f"2 {lab}dry writes nothing", snapshot(v) == before)
        rc, out = run(v, "install")
        check(f"2 {lab}install", rc == 0, out[-300:])
        after = snapshot(v)
        rc, out = run(v, "status")
        check(f"2 {lab}status patched x2 + module present",
              out.count(" patched ") == 2 and "dsv41_routeprep.py present" in out, out)
        rc, out = run(v, "install")
        check(f"2 {lab}install idempotent", rc == 0 and snapshot(v) == after)
        for rel in ("model_executor/layers/fused_moe/deep_gemm_utils.py",
                    "model_executor/layers/fused_moe/experts/deep_gemm_moe.py"):
            p = os.path.join(v, rel)
            src = open(p).read()
            ast.parse(src)
            check(f"2 {lab}backup is the pre-install file {os.path.basename(rel)}",
                  sha(p + ".bak-r2-moefront") == before[rel])
            if r1pdl and "deep_gemm_utils" in rel:
                check(f"3 r1-pdl lines kept in {os.path.basename(rel)}", "# r1-pdl" in src)
        rc, out = run(v, "uninstall")
        restored = snapshot(v)
        same = all(restored.get(k) == before[k] for k in before)
        check(f"2 {lab}uninstall restores exact bytes", rc == 0 and same, out[-300:])
        check(f"2 {lab}uninstall removes the module",
              not os.path.exists(os.path.join(v, "model_executor/layers/fused_moe/dsv41_routeprep.py")))


def test_refuse_foreign():
    with tempfile.TemporaryDirectory() as tmp:
        v = make_tree(tmp)
        p = os.path.join(v, "model_executor/layers/fused_moe/deep_gemm_utils.py")
        open(p, "a").write("# foreign edit\n")
        before = snapshot(v)
        rc, out = run(v, "install")
        check("2 refuses a foreign base and writes nothing", rc != 0 and snapshot(v) == before, out.strip()[-120:])


def tagged_stmt_check(src, rel):
    """Every statement touching a tagged line is wholly tagged and is a flag assignment/gated block."""
    lines = src.splitlines()
    tagged = {i + 1 for i, ln in enumerate(lines) if ln.rstrip().endswith(TAG)}
    tree = ast.parse(src)
    ok, why = True, []
    seen = set()

    def stmt_lines(n):
        return set(range(n.lineno, n.end_lineno + 1))

    for node in ast.walk(tree):
        if not isinstance(node, ast.stmt):
            continue
        sl = stmt_lines(node)
        if not (sl & tagged):
            continue
        if sl <= tagged:
            # a fully tagged statement must be the flag assignment or `if _DSV41_RP...` (or inside one)
            if isinstance(node, ast.Assign) and [t.id for t in node.targets if isinstance(t, ast.Name)] == ["_DSV41_RP"]:
                seen |= sl
            elif isinstance(node, ast.If):
                t = node.test
                first = t.values[0] if isinstance(t, ast.BoolOp) and isinstance(t.op, ast.And) else t
                if isinstance(first, ast.Name) and first.id == "_DSV41_RP" and not node.orelse:
                    seen |= sl
                elif not (sl <= seen):
                    ok = False
                    why.append(f"line {node.lineno}: tagged if not gated by _DSV41_RP")
            elif not (sl <= seen):
                ok = False
                why.append(f"line {node.lineno}: tagged {type(node).__name__} outside a gated block")
        elif not isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module, ast.If, ast.With, ast.For, ast.Try)):
            ok = False
            why.append(f"line {node.lineno}: statement mixes tagged and untagged lines")
    check(f"4 every inserted statement is gated ({rel})", ok and tagged <= seen,
          "; ".join(why) or f"{len(tagged)} tagged lines")


def test_flag_off():
    inst = load(os.path.join(KIT, "install.py"), "r2mf_install")
    for rel in inst.EDITS:
        base = open(os.path.join(KIT, "base", rel)).read()
        patched = inst.patched(rel, base)
        check(f"4 strip(tagged) == base ({os.path.basename(rel)})", inst.strip_tag(patched, TAG) == base)
        tagged_stmt_check(patched, os.path.basename(rel))
        # the module-level gate, evaluated as written
        m = re.search(r"^(_DSV41_RP = .*?)  " + re.escape(TAG) + "$", patched, re.M)
        expr = m.group(1).split("=", 1)[1]
        for val, want in ((None, False), ("0", False), ("off", False), ("false", False), ("", False),
                          (" OFF ", False), ("1", True), ("on", True)):
            env = {} if val is None else {"DSV41_MOE_ROUTEPREP": val}

            class _OS:
                environ = env
            got = eval(expr, {"os": _OS})
            if got != want:
                check(f"4 gate value for {val!r}", False)
                break
        else:
            check(f"4 gate expression False for unset/0/off/false/'' ({os.path.basename(rel)})", True)


def test_athena_transform():
    txt = open(os.path.join(KIT, "ref", "athena_overlay.txt")).read().split("====\n", 1)[1]
    tree = ast.parse(txt)
    ns = {"textwrap": textwrap, "ast": ast}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ("_once", "transform_apply"):
            exec(compile(ast.Module([node], []), "athena", "exec"), ns)
    inst = load(os.path.join(KIT, "install.py"), "r2mf_install2")
    rel = "model_executor/layers/fused_moe/experts/deep_gemm_moe.py"
    for lab, src in (("base", open(os.path.join(KIT, "base", rel)).read()),
                     ("patched", inst.patched(rel, open(os.path.join(KIT, "base", rel)).read()))):
        t = ast.parse(src)
        cls = next(n for n in t.body if isinstance(n, ast.ClassDef) and n.name == "DeepGemmFP4Experts")
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "apply")
        seg = "\n".join(src.splitlines()[fn.lineno - 1:fn.end_lineno]) + "\n"
        try:
            outsrc = ns["transform_apply"](seg)
            ok = "athena_fc2_begin(self)" in outsrc and (lab == "base" or "_dsv41_rp.front" in outsrc)
            if lab == "patched":
                i_front, i_begin = outsrc.index("_dsv41_rp.front"), outsrc.index("athena_fc2_begin(self)")
                ok = ok and i_front < i_begin
            check(f"5 Athena transform_apply on the {lab} DeepGemmFP4Experts.apply", ok)
        except Exception as exc:  # noqa: BLE001
            check(f"5 Athena transform_apply on the {lab} DeepGemmFP4Experts.apply", False, repr(exc))


def test_r1pdl_anchors_untouched():
    m = load(R1PDL, "r1pdl_install2")
    inst = load(os.path.join(KIT, "install.py"), "r2mf_install3")
    rel = "model_executor/layers/fused_moe/deep_gemm_utils.py"
    mine = inst.patched(rel, open(os.path.join(KIT, "base", rel)).read())
    ok = all(mine.count(old) == count for old, _, count in m.EDITS[("vllm", rel)])
    check("3 r1-pdl's deep_gemm_utils anchors still match once after our edits (either install order)", ok)


def test_draftmoe_composition(r1pdl):
    lab = "r1-pdl+" if r1pdl else ""
    if not os.path.exists(os.path.join(DRAFTMOE, "install.py")):
        check(f"6 {lab}r2-draftmoe kit present", False, DRAFTMOE)
        return
    with tempfile.TemporaryDirectory() as tmp:
        home = os.path.join(tmp, "home")
        os.makedirs(home)
        v = make_tree(tmp, r1pdl, draftmoe_files=True)
        base = ours_only(snapshot(v))
        # A: moefront then draftmoe
        rc1, o1 = run(v, "install")
        mf_only = ours_only(snapshot(v))
        rc2, o2 = run_dm(v, home, "install")
        both_a = ours_only(snapshot(v))
        check(f"6 {lab}moefront -> draftmoe installs", rc1 == 0 and rc2 == 0, (o1 + o2)[-300:])
        rc, o = run(v, "status")
        check(f"6 {lab}status 'patched' x2 with draftmoe on top", rc == 0 and o.count(" patched ") == 2, o[-300:])
        for rel in ("model_executor/layers/fused_moe/deep_gemm_utils.py",
                    "model_executor/layers/fused_moe/experts/deep_gemm_moe.py"):
            ast.parse(open(os.path.join(v, rel), encoding="utf-8").read())
        rc, o = run(v, "uninstall")  # ours first (non-LIFO): must leave exactly the draftmoe-only install
        after_mf_un = ours_only(snapshot(v))
        rc2, o2 = run_dm(v, home, "uninstall")
        rc3, o3 = run_dm(v, home, "install")
        dm_only = ours_only(snapshot(v))
        check(f"6 {lab}uninstall moefront under draftmoe works and leaves the draftmoe-only bytes",
              rc == 0 and rc2 == 0 and rc3 == 0 and after_mf_un == dm_only, (o + o2 + o3)[-300:])
        # B: draftmoe then moefront (draftmoe already installed now)
        rc, o = run(v, "install")
        both_b = ours_only(snapshot(v))
        check(f"6 {lab}draftmoe -> moefront installs", rc == 0, o[-300:])
        rc, o = run(v, "install")
        check(f"6 {lab}idempotent over draftmoe", rc == 0 and ours_only(snapshot(v)) == both_b, o[-200:])
        for rel in both_b:
            if rel.endswith(".py"):
                ast.parse(open(os.path.join(v, rel), encoding="utf-8").read())
        same_set = {k for k in both_a} == {k for k in both_b}
        strip_eq = True
        inst = load(os.path.join(KIT, "install.py"), "r2mf_inst_c")
        for rel in ("model_executor/layers/fused_moe/deep_gemm_utils.py",
                    "model_executor/layers/fused_moe/experts/deep_gemm_moe.py"):
            src = open(os.path.join(v, rel), encoding="utf-8", newline="").read()
            # either order: removing the draftmoe lines gives exactly our install
            strip_eq &= hashlib.sha256(inst.strip_tag(src, "# r2-draftmoe").encode()).hexdigest() == mf_only[rel]
        check(f"6 {lab}both orders: same file set, draftmoe-stripped files == moefront-only install",
              same_set and strip_eq)
        rc, o = run_dm(v, home, "uninstall")
        check(f"6 {lab}uninstall draftmoe (above us) leaves the moefront-only bytes",
              rc == 0 and ours_only(snapshot(v)) == mf_only, o[-300:])
        rc, o = run(v, "uninstall")
        check(f"6 {lab}uninstall moefront last restores the starting bytes",
              rc == 0 and ours_only(snapshot(v)) == base, o[-300:])


def test_uninstall_all_or_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        v = make_tree(tmp)
        rc, o = run(v, "install")
        p = os.path.join(v, "model_executor/layers/fused_moe/experts/deep_gemm_moe.py")
        with open(p, "a", encoding="utf-8") as f:
            f.write("# an untagged foreign edit\n")
        before = snapshot(v)
        rc, o = run(v, "uninstall")
        check("7 uninstall refuses when the 2nd file is foreign and writes nothing (1st file untouched)",
              rc != 0 and snapshot(v) == before and "nothing was changed" in o, o.strip()[-160:])


def test_c_locale():
    with tempfile.TemporaryDirectory() as tmp:
        v = make_tree(tmp)
        before = snapshot(v)
        env = {"LC_ALL": "C", "LANG": "C", "PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0"}
        rc1, o1 = run(v, "install", env_extra=env)
        rc2, o2 = run(v, "status", env_extra=env)
        rc3, o3 = run(v, "uninstall", env_extra=env)
        after = snapshot(v)
        check("8 install/status/uninstall under LC_ALL=C PYTHONUTF8=0 restore exact bytes",
              rc1 == 0 and rc2 == 0 and o2.count(" patched ") == 2 and rc3 == 0
              and all(after.get(k) == h for k, h in before.items()), (o1 + o2 + o3)[-300:])


if __name__ == "__main__":
    test_bases()
    test_install_cycle(False)
    test_install_cycle(True)
    test_refuse_foreign()
    test_r1pdl_anchors_untouched()
    test_flag_off()
    test_athena_transform()
    test_draftmoe_composition(False)
    test_draftmoe_composition(True)
    test_uninstall_all_or_nothing()
    test_c_locale()
    print("ALL PASS" if not FAIL else f"FAILED: {FAIL}")
    sys.exit(1 if FAIL else 0)
