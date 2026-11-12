#!/usr/bin/env python3
"""r2-draftmoe CPU tests: installer, flag-off inertness, Athena anchor composition, round-1 composition, flag parse.

Run: python3 tests/test_install.py   (stdlib only; uses ../base = byte copies of the live files, sha-checked)
"""
import ast
import copy
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(KIT, "base")
META = json.load(open(os.path.join(KIT, "KIT.json")))
PATCHED = [f["path"] for f in META["files"] if f["action"] == "patch"]
NEWMOD = "model_executor/layers/fused_moe/experts/dsv41_r2_draftmoe.py"
FLAG_NAMES = {"_R2DM_FC1": False, "_R2DM_PERM": "", "_r2dm_fp": False}
TAG = "# r2-draftmoe"
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""), flush=True)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def run_installer(vdir, *args, home=None):
    env = dict(os.environ, R2DM_VLLM=vdir, R2DM_HOME=home or vdir)
    env.pop("DSV41_DRAFT_MOE_G17", None)
    return subprocess.run([sys.executable, os.path.join(KIT, "install.py"), *args], env=env, capture_output=True,
                          text=True)


def fresh_tree(src_root=BASE):
    d = tempfile.mkdtemp(prefix="r2dm-v-")
    for rel in PATCHED:
        os.makedirs(os.path.dirname(os.path.join(d, rel)), exist_ok=True)
        shutil.copy2(os.path.join(src_root, rel), os.path.join(d, rel))
    return d


# ------------------------------------------------------------------ 1. base copies are the recorded live bases
for f in META["files"]:
    if f["action"] == "patch":
        check(f"base sha256 {f['path']}", sha(os.path.join(BASE, f["path"])) == f["base_sha256"], f["base_sha256"][:8])

# ------------------------------------------------------------------ 2. install / idempotent / uninstall round trip
d = fresh_tree()
r = run_installer(d, "dry")
check("dry runs and reports all bases OK", r.returncode == 0 and r.stdout.count(": base ") == 4
      and "NOT A RECORDED BASE" not in r.stdout, r.stderr.strip()[-200:])
r = run_installer(d, "install")
check("install exit 0", r.returncode == 0, (r.stdout + r.stderr).strip()[-300:])
after = {rel: sha(os.path.join(d, rel)) for rel in PATCHED}
check("install created the new module", os.path.exists(os.path.join(d, NEWMOD))
      and sha(os.path.join(d, NEWMOD)) == sha(os.path.join(KIT, "new", NEWMOD)))
for rel in PATCHED:
    src = open(os.path.join(d, rel)).read()
    base = open(os.path.join(BASE, rel)).read()
    ok = True
    try:
        compile(src, rel, "exec")
    except SyntaxError as exc:
        ok = False
    check(f"patched {rel} compiles", ok)
    # pure insertion: removing the lines that END with the TAG gives back the base byte for byte (so any kit that
    # strips '# r2-draftmoe' lines, and our own uninstall, recovers the exact pre-install file); no other line
    # mentions the TAG
    stripped = "".join(l for l in src.splitlines(keepends=True) if not l.rstrip("\n").endswith(TAG))
    tagged = [l for l in src.splitlines() if TAG in l]
    check(f"pure tagged insertion in {rel}: strip('{TAG}') == base bytes, every tagged line ends with the tag",
          stripped == base and all(l.endswith(TAG) for l in tagged), f"{len(tagged)} tagged lines")
r = run_installer(d, "install")
check("second install is idempotent", r.returncode == 0 and "already patched" in r.stdout
      and all(sha(os.path.join(d, rel)) == after[rel] for rel in PATCHED))
r = run_installer(d, "status")
check("status reports PATCHED x4 + new module", r.returncode == 0 and r.stdout.count("PATCHED") == 5
      and "FOREIGN" not in r.stdout, r.stdout[-300:])
r = run_installer(d, "uninstall")
check("uninstall restores byte-identical bases", r.returncode == 0
      and all(sha(os.path.join(d, rel)) == sha(os.path.join(BASE, rel)) for rel in PATCHED)
      and not os.path.exists(os.path.join(d, NEWMOD)), r.stderr[-200:])
shutil.rmtree(d)

# ------------------------------------------------------------------ 3. refuses a foreign base, changes nothing
d = fresh_tree()
p = os.path.join(d, PATCHED[1])
open(p, "a").write("# drift\n")
before = {rel: sha(os.path.join(d, rel)) for rel in PATCHED}
r = run_installer(d, "install")
check("install refuses a base whose sha256 is not recorded", r.returncode != 0 and "REFUSED" in (r.stdout + r.stderr)
      and all(sha(os.path.join(d, rel)) == before[rel] for rel in PATCHED)
      and not os.path.exists(os.path.join(d, NEWMOD)))
shutil.rmtree(d)

# ------------------------------------------------------------------ 3b. uninstall refuses a file whose tagged lines were altered
d = fresh_tree()
run_installer(d, "install")
p = os.path.join(d, PATCHED[1])
txt = open(p).read()
open(p, "w").write(txt.replace("compact_fc1 = True  " + TAG, "compact_fc1 = False  " + TAG))
before = {rel: sha(os.path.join(d, rel)) for rel in PATCHED}
r = run_installer(d, "uninstall")
check("uninstall refuses (changes nothing) when our tagged lines are not our patch", r.returncode != 0
      and "REFUSED" in (r.stdout + r.stderr) and all(sha(os.path.join(d, rel)) == before[rel] for rel in PATCHED)
      and os.path.exists(os.path.join(d, NEWMOD)))
r = run_installer(d, "install")
check("install refuses a tagged file that is not our patch", r.returncode != 0 and "REFUSED" in (r.stdout + r.stderr))
shutil.rmtree(d)

# ------------------------------------------------------------------ 4. round-1 composition
r1_root = os.environ.get("R2DM_R1_ROOT") or os.path.expanduser("~/Documents/dsv41-offline/climb-r1")
check("round-1 kit directory present (composition is really tested)", os.path.isdir(r1_root), r1_root)
touch = {}
for kit in sorted(glob.glob(os.path.join(r1_root, "r1-*"))):
    files = set()
    for kj in glob.glob(os.path.join(kit, "KIT.json")):
        try:
            for f in json.load(open(kj)).get("files", []):
                files.add(f.get("path", "").replace("$HOME/", ""))
        except Exception:
            pass
    for b in glob.glob(os.path.join(kit, "base", "**", "*.py"), recursive=True):
        files.add(os.path.relpath(b, os.path.join(kit, "base")))
    for inst in glob.glob(os.path.join(kit, "install*.py")):
        txt = open(inst).read()
        for rel in PATCHED:
            if rel in txt or os.path.basename(rel) in txt:
                files.add(rel)
    touch[os.path.basename(kit)] = files
overlap = {k: sorted(set(PATCHED) & v) for k, v in touch.items() if set(PATCHED) & v}
check("round-1 kits scanned; those touching our files are composed below", True,
      f"r1 kits scanned: {sorted(touch)}; overlap {overlap or 'none'}")
# Compose with any round-1 installer whose EDITS touch our files: apply ITS edit list (imported, never run) to our
# base copy, then ours on top; the composed base must be recorded in KIT.json and our anchors must still match.
import importlib.util  # noqa: E402

composed = {}
for kit in sorted(glob.glob(os.path.join(r1_root, "r1-*"))):
    inst = os.path.join(kit, "install.py")
    if not os.path.exists(inst) or not any(rel in open(inst).read() for rel in PATCHED):
        continue
    spec = importlib.util.spec_from_file_location("r1inst_" + os.path.basename(kit).replace("-", "_"), inst)
    mod = importlib.util.module_from_spec(spec)
    saved_env = dict(os.environ)
    for k in list(os.environ):
        if k.endswith("_VLLM") or k.endswith("_HOME") and k.startswith("R1"):
            os.environ.pop(k)
    os.environ.update({f"R1{x}_VLLM": "/nonexistent-r2dm" for x in ("PDL", "ATTN", "D", "EAGER", "PF")})
    try:
        spec.loader.exec_module(mod)
    finally:
        os.environ.clear()
        os.environ.update(saved_env)
    for key, edits in mod.EDITS.items():
        rel = key[1] if isinstance(key, tuple) else key
        rel = rel[len("vllm/"):] if rel.startswith("vllm/") else rel
        if rel not in PATCHED:
            continue
        text = composed.get(rel, open(os.path.join(BASE, rel)).read())
        for old, new_, count in edits:
            assert text.count(old) == count, (kit, rel, old[:60])
            text = text.replace(old, new_)
        composed[rel] = text
        print(f"      composed {os.path.basename(kit)} edits onto {rel}")
sys.path.insert(0, KIT)
if composed:
    spec = importlib.util.spec_from_file_location("r2dm_install", os.path.join(KIT, "install.py"))
    r2 = importlib.util.module_from_spec(spec)
    saved_argv = sys.argv
    sys.argv = ["install.py", "--noop"]
    try:
        spec.loader.exec_module(r2)
    except SystemExit:
        pass
    sys.argv = saved_argv
    for rel, text in composed.items():
        csha = hashlib.sha256(text.encode()).hexdigest()
        rec = next(f for f in META["files"] if f["path"] == rel)
        check(f"r1-composed base of {rel} ({csha[:8]}) is recorded in KIT.json base_sha256_with_r1",
              csha in rec.get("base_sha256_with_r1", []), csha)
        try:
            on_top = r2.patched(rel, text)
            ok = True
        except SystemExit as exc:
            ok, on_top = False, str(exc)
        check(f"our anchors apply exactly once on the r1-composed {rel}", ok)
        if ok:
            compile(on_top, rel, "exec")
            COMPOSED_PATCHED = on_top
            globals().setdefault("COMPOSED", {})[rel] = (text, on_top)
else:
    check("no round-1 installer edits our files", True)

# ------------------------------------------------------------------ 5. Athena FC2 transform_apply still matches
ATHENA = os.path.join(KIT, "live-ref/fc2/athena_fc2_forty_layer_trial.py")
ath_src = open(ATHENA).read()
ath_tree = ast.parse(ath_src)
ns = {"ast": ast, "textwrap": textwrap}
for node in ath_tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in ("_once", "transform_apply"):
        exec(compile(ast.Module([node], []), ATHENA, "exec"), ns)


def method_source(path, cls, meth):
    src = open(path).read()
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == meth:
                    start = (sub.decorator_list[0].lineno if sub.decorator_list else sub.lineno) - 1
                    return "".join(lines[start:sub.end_lineno])
    raise KeyError(meth)


d = fresh_tree()
run_installer(d, "install")
for label, root in (("base", BASE), ("patched", d)):
    try:
        out = ns["transform_apply"](method_source(os.path.join(root, PATCHED[1]), "DeepGemmFP4Experts", "apply"))
        ok = "athena_fc2_try_launch(" in out and "compact_fc1, align_used" in out
        err = ""
    except Exception as exc:
        ok, err = False, repr(exc)
    check(f"Athena transform_apply anchors match exactly once on the {label} apply()", ok, err)

# ------------------------------------------------------------------ 6. flag-off AST equivalence (constant folding)


class Fold(ast.NodeTransformer):
    """Fold the kit's flags to their flag-off values and simplify; flag-off code must equal the base AST."""

    def visit_Module(self, node):
        node.body = self._stmts(node.body)
        return self.generic_visit(node)

    def _stmts(self, body):
        out = []
        for s in body:
            if isinstance(s, ast.Assign):
                names = {n.id for t in s.targets for n in ast.walk(t) if isinstance(n, ast.Name)}
                if names and names <= set(FLAG_NAMES):
                    continue  # the flag definitions themselves
            if isinstance(s, ast.If):
                t = self.visit(copy.deepcopy(s.test))
                if isinstance(t, ast.Constant) and not t.value:
                    out.extend(self._stmts(s.orelse))
                    continue
                if (isinstance(s.test, ast.Compare) and isinstance(s.test.left, ast.Call)
                        and "DSV41_DRAFT_MOE_G17" in ast.unparse(s.test)):
                    continue  # the gated import block: env unset -> never executed (checked separately below)
            out.append(s)
        return out

    def generic_visit(self, node):
        for field in ("body", "orelse", "finalbody"):
            if isinstance(getattr(node, field, None), list) and not isinstance(node, ast.Module):
                setattr(node, field, self._stmts(getattr(node, field)))
        return super().generic_visit(node)

    def visit_Name(self, node):
        if node.id in FLAG_NAMES and isinstance(node.ctx, ast.Load):
            return ast.copy_location(ast.Constant(FLAG_NAMES[node.id]), node)
        return node

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.And):
            if isinstance(node.values[0], ast.Constant) and not node.values[0].value:
                return node.values[0]
            return node
        vals = [v for v in node.values if not (isinstance(v, ast.Constant) and not v.value)]
        if len(vals) == 1:
            return vals[0]
        node.values = vals
        return node

    def visit_Compare(self, node):
        self.generic_visit(node)
        if isinstance(node.left, ast.Constant) and not node.left.value and all(isinstance(c, ast.Constant)
                                                                              for c in node.comparators):
            try:
                return ast.Constant(eval(compile(ast.Expression(node), "<f>", "eval")))
            except Exception:
                return node
        return node


for rel in PATCHED:
    base_t = ast.parse(open(os.path.join(BASE, rel)).read())
    pat_t = Fold().visit(ast.parse(open(os.path.join(d, rel)).read()))
    same = ast.dump(base_t) == ast.dump(pat_t)
    detail = ""
    if not same:
        a, b = ast.dump(base_t, indent=1).splitlines(), ast.dump(pat_t, indent=1).splitlines()
        i = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        detail = f"first diff at dump line {i}: {a[i:i+2]} vs {b[i:i+2]}"
    check(f"flag-off AST of {rel} == base AST (flags folded to off)", same, detail)
for rel, (cbase, cpatched) in globals().get("COMPOSED", {}).items():
    same = ast.dump(ast.parse(cbase)) == ast.dump(Fold().visit(ast.parse(cpatched)))
    check(f"flag-off AST of r1-composed {rel} + ours == r1-composed base AST", same)

# ------------------------------------------------------------------ 7. the gated import never runs with the env unset
prelude = [l for l in open(os.path.join(d, PATCHED[0])).read().splitlines() if l.startswith(("_R2DM_", "if os.environ",
                                                                                              "    from vllm.model_executor.layers.fused_moe.experts import dsv41_r2_draftmoe",
                                                                                              "    _R2DM_FC1, _R2DM_PERM ="))]
prelude_src = "import os\n" + "\n".join(prelude) + "\n"
for env_val, want in ((None, (False, "")), ("0", (False, "")), ("", (False, "")), ("off", (False, ""))):
    g = {}
    saved = os.environ.pop("DSV41_DRAFT_MOE_G17", None)
    if env_val is not None:
        os.environ["DSV41_DRAFT_MOE_G17"] = env_val
    exec(prelude_src, g)
    if saved is not None:
        os.environ["DSV41_DRAFT_MOE_G17"] = saved
    else:
        os.environ.pop("DSV41_DRAFT_MOE_G17", None)
    check(f"prelude with DSV41_DRAFT_MOE_G17={env_val!r}: flags off, module not imported",
          (g["_R2DM_FC1"], g["_R2DM_PERM"]) == want and "_r2dm" not in g)
shutil.rmtree(d)

# ------------------------------------------------------------------ 8. flag parse
sys.path.insert(0, os.path.join(KIT, "new/model_executor/layers/fused_moe/experts"))
os.environ.pop("DSV41_DRAFT_MOE_G17", None)
import dsv41_r2_draftmoe as m  # noqa: E402

cases = {"0": (False, ""), "": (False, ""), "1": (True, "1"), "fc1": (True, ""), "perm": (False, "1"),
         "fc1,perm": (True, "1"), " FC1 , Perm ": (True, "1"), "permverify": (False, "verify"),
         "fc1,permverify": (True, "verify"), "on": (True, "1")}
check("parse() accepted values", all(m.parse(k) == v for k, v in cases.items()),
      str({k: m.parse(k) for k in cases}))
bad_ok = True
for bad in ("2", "verify", "fc2", "perm,permverify", "fc1,fc2", ","):
    try:
        m.parse(bad)
        bad_ok = False
    except ValueError:
        pass
check("parse() rejects 2, verify, fc2, perm+permverify, unknown tokens", bad_ok)
check("module-level flags off with env unset", (m.FC1, m.PERM) == (False, ""))

n_fail = sum(1 for _, ok in results if not ok)
print(f"\n{len(results) - n_fail}/{len(results)} passed")
sys.exit(1 if n_fail else 0)
