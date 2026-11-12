#!/usr/bin/env python3
"""Composition with the round-1 kits (stdlib only).

1. Scan every round-1 kit (~/Documents/dsv41-offline/climb-r1/r1-*) and every other round-2 kit's KIT.json
   files[].path (install.py text only when a kit has no parseable KIT.json) for the three files this kit patches. Expected: only r1-pdl touches one of them (the gather).
2. r1-pdl's own anchor edits (imported from its install.py, not copied) applied to the live gather, then ours:
   applies, compiles, our tagged lines present, r1-pdl's lines intact; the r1-pdl+ours sha is in KIT.json accepted
   bases (printed so KIT.json can be filled). Also ours first then r1-pdl (anchors still match).
3. Text-level removal in both orders (our exact-inverse revert; the inverse of r1-pdl's EDITS, NOT its real
   uninstall, which restores its .bak and refuses a file it did not write; see 5): each removal leaves the other
   kit's edits byte-identical. Real installers must be uninstalled LIFO (section 5).
5. (review fix) REAL installers, not inverse replaces: on a temp tree holding r1-pdl's six base files plus our
   three, run r1-pdl's own install.py and ours as subprocesses: (A) r1-pdl install -> ours install -> ours
   uninstall -> r1-pdl uninstall (LIFO): every intermediate file is what it should be and the end state is the
   base, byte for byte; (B) ours first, then r1-pdl install: r1-pdl refuses and writes nothing; (C) wrong-order
   uninstall (r1-pdl's uninstall while ours is installed on top): its outcome is recorded and must leave no
   half-state (either it refuses with every file untouched, or it restores the base and our gather hook is gone
   while moe_runner keeps the add, i.e. G25 inert).
4. Flag-off safety with both installed: r1-pdl's DSV41_GLUE_PDL hook and our DSV41_FC2_GATHER_ADD hook are both
   inside `if <module-level flag>:` blocks; with both unset the executed statements of gated_gather are the base ones.
Usage: python3 test_compose_r1.py <kit_dir> [<climb_r1_dir>]
"""
import ast
import hashlib
import importlib.util
import json
import os
import sys

KITDIR = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "..")
R1 = os.path.abspath(sys.argv[2] if len(sys.argv) > 2 else os.path.join(KITDIR, "../../climb-r1"))
R2 = os.path.dirname(KITDIR)
os.environ.setdefault("R2MB_VLLM", "/nonexistent")
os.environ.setdefault("R2MB_FC2", "/nonexistent")
FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        FAILS.append(name)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


ours = load("r2mb_install", os.path.join(KITDIR, "install.py"))
base_gather = open(os.path.join(KITDIR, "base/fc2_status_gated_gather_trial.py")).read()
targets = {"moe_runner.py": "fused_moe/runner/moe_runner.py",
           "fc2_status_gated_gather_trial.py": "fc2_status_gated_gather_trial.py",
           "sitecustomize.py": "sitecustomize.py"}

# 1. scan
touch = {}
for root in (R1, R2):
    for k in sorted(os.listdir(root)):
        d = os.path.join(root, k)
        if not os.path.isdir(d) or k == "r2-moeback":
            continue
        # KIT.json files[].path is authoritative (brief); free-text notes that merely mention a file do not count.
        # Without a parseable KIT.json fall back to a text search of install.py.
        kj = os.path.join(d, "KIT.json")
        paths = None
        try:
            paths = [f.get("path", "") for f in json.load(open(kj)).get("files", [])]
        except (OSError, ValueError, AttributeError):
            pass
        if paths is None:
            ip = os.path.join(d, "install.py")
            paths = [open(ip, errors="replace").read()] if os.path.exists(ip) else []
        for t, needle in targets.items():
            if any(needle in x for x in paths):
                touch.setdefault(t, set()).add(k)
print("other kits touching our targets:", {k: sorted(v) for k, v in touch.items()})
check("only r1-pdl touches one of our targets (the gather)",
      touch.get("fc2_status_gated_gather_trial.py", set()) <= {"r1-pdl"} and "moe_runner.py" not in touch
      and "sitecustomize.py" not in touch, str(touch))

# 2. r1-pdl edits then ours, and ours then r1-pdl
r1pdl_dir = os.path.join(R1, "r1-pdl")
r1 = load("r1pdl_install", os.path.join(r1pdl_dir, "install.py"))
key = ("home", "dsv41-fc2/fc2_status_gated_gather_trial.py")
r1_kit = json.load(open(os.path.join(r1pdl_dir, "KIT.json"))) if os.path.exists(os.path.join(r1pdl_dir, "KIT.json")) else None
if r1_kit:
    r1_base = [f["base_sha256"] for f in r1_kit["files"] if f["path"].endswith("fc2_status_gated_gather_trial.py")]
    check("r1-pdl records the same live gather base", r1_base and r1_base[0] == sha(base_gather), str(r1_base))
r1_src = r1.patched(key, base_gather)
compile(r1_src, "r1", "exec")
both_a = ours.apply_edits(r1_src, ours.GATHER_EDITS, "ours-after-r1")
compile(both_a, "both_a", "exec")
check("ours applies on top of r1-pdl", ours.T25 in both_a and both_a.count(r1.TAG) == r1_src.count(r1.TAG))
our_src = ours.apply_edits(base_gather, ours.GATHER_EDITS, "ours")
both_b = r1.patched(key, our_src)
compile(both_b, "both_b", "exec")
check("r1-pdl applies on top of ours", both_b.count(ours.T25) == our_src.count(ours.T25) and r1.TAG in both_b)
check("both orders give the same file", both_a == both_b)
accepted = ours.accepted_bases.__wrapped__("gather") if hasattr(ours.accepted_bases, "__wrapped__") else None
kitj = json.load(open(os.path.join(KITDIR, "KIT.json")))
acc = [set(f["base_sha256"]) for f in kitj["files"] if f.get("key") == "gather"][0]
print("live gather sha", sha(base_gather), "r1-pdl gather sha", sha(r1_src))
check("KIT.json accepts the live and the r1-pdl-installed gather", {sha(base_gather), sha(r1_src)} <= acc)

# 3. uninstall in both orders
undo_ours = ours.revert_edits(both_a, ours.GATHER_EDITS, "undo-ours")
check("removing ours leaves exactly r1-pdl's file", undo_ours == r1_src)


def r1_revert(src):
    for old, new, count in reversed(r1.EDITS[key]):
        assert src.count(new) == count
        src = src.replace(new, old)
    return src


undo_r1 = r1_revert(both_a)
check("removing r1-pdl (inverse of its edits) leaves exactly our file", undo_r1 == our_src)
check("removing both restores the live base", ours.revert_edits(undo_r1, ours.GATHER_EDITS, "x") == base_gather)


# 4. flag-off: both hooks are under module-level flag tests
def gated_body(src):
    tree = ast.parse(src)
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "gated_gather"][0]
    lines = src.splitlines()
    kept = []
    for st in fn.body:
        if isinstance(st, ast.If) and isinstance(st.test, ast.Name) and st.test.id in ("_DSV41_GLUE_PDL",
                                                                                      "_DSV41_FC2_GATHER_ADD"):
            continue
        kept.append(ast.dump(st))
    return kept


check("gated_gather with both flags off executes exactly the base statements",
      gated_body(both_a) == gated_body(base_gather))


# 5. real installers
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402


def fsha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else "-"


def tree(root):
    shutil.copytree(os.path.join(r1pdl_dir, "base/vllm"), os.path.join(root, "vllm"))
    shutil.copytree(os.path.join(r1pdl_dir, "base/home"), os.path.join(root, "home"))
    os.makedirs(os.path.join(root, "vllm/triton_utils"), exist_ok=True)
    os.makedirs(os.path.join(root, "vllm/model_executor/layers/fused_moe/runner"), exist_ok=True)
    os.makedirs(os.path.join(root, "home/dsv41-fc2/site"), exist_ok=True)
    shutil.copy2(os.path.join(KITDIR, "base/model_executor_layers_fused_moe_runner_moe_runner.py"),
                 os.path.join(root, "vllm/model_executor/layers/fused_moe/runner/moe_runner.py"))
    shutil.copy2(os.path.join(KITDIR, "base/sitecustomize.py"), os.path.join(root, "home/dsv41-fc2/site/sitecustomize.py"))
    files = {}
    for dp, _, fns in os.walk(root):
        for fn in fns:
            files[os.path.relpath(os.path.join(dp, fn), root)] = fsha(os.path.join(dp, fn))
    return files


def snap(root):
    out = {}
    for dp, _, fns in os.walk(root):
        for fn in fns:
            out[os.path.relpath(os.path.join(dp, fn), root)] = fsha(os.path.join(dp, fn))
    return out


def run(root, who, cmd):
    env = dict(os.environ, R1PDL_VLLM=os.path.join(root, "vllm"), R1PDL_HOME=os.path.join(root, "home"),
               R2MB_VLLM=os.path.join(root, "vllm"), R2MB_FC2=os.path.join(root, "home/dsv41-fc2"))
    inst = os.path.join(r1pdl_dir if who == "r1" else KITDIR, "install.py")
    return subprocess.run([sys.executable, inst, cmd], env=env, capture_output=True, text=True)


G = "home/dsv41-fc2/fc2_status_gated_gather_trial.py"
MR = "vllm/model_executor/layers/fused_moe/runner/moe_runner.py"
if os.path.exists(os.path.join(KITDIR, "build/athena_tp8_fc2_packed_graph_pdl_sm121.so")) and not \
        json.load(open(os.path.join(KITDIR, "KIT.json"))).get("pdl_binary_sha256"):
    print("NOTE: kit build/ holds an unvetted .so; section 5 expects our install to refuse")
with tempfile.TemporaryDirectory() as root:
    base = tree(root)
    r = run(root, "r1", "install")
    check("A: r1-pdl real install on the base tree", r.returncode == 0, r.stderr[-200:])
    after_r1 = snap(root)
    check("A: r1-pdl's gather == its patch of the base (sha in our KIT.json)", after_r1[G] == sha(r1_src) and sha(r1_src) in acc)
    r = run(root, "r2", "install")
    check("A: ours installs on top of r1-pdl", r.returncode == 0, r.stderr[-300:])
    check("A: gather now carries both kits' tags",
          ours.T25 in open(os.path.join(root, G)).read() and r1.TAG in open(os.path.join(root, G)).read())
    r = run(root, "r1", "status")
    check("A: r1-pdl status flags the gather as not its exact patch while ours is on top",
          "NOT OUR EXACT PATCH" in r.stdout, r.stdout[-300:])
    r = run(root, "r2", "uninstall")
    s_mid = snap(root)
    check("A: ours uninstall (LIFO) restores exactly r1-pdl's state (backups aside)",
          r.returncode == 0 and all(s_mid.get(k) == v for k, v in after_r1.items()), r.stderr[-200:])
    r = run(root, "r1", "status")
    check("A: r1-pdl status PATCHED again for the gather", "PATCHED" in [l for l in r.stdout.splitlines() if G.split("/")[-1] in l][0], r.stdout)
    r = run(root, "r1", "uninstall")
    end = snap(root)
    check("A: r1-pdl real uninstall -> every original file byte-identical to base",
          r.returncode == 0 and all(end.get(k) == v for k, v in base.items()), r.stderr[-200:])
with tempfile.TemporaryDirectory() as root:
    base = tree(root)
    run(root, "r2", "install")
    s0 = snap(root)
    r = run(root, "r1", "install")
    check("B: r1-pdl after ours: refused, nothing written", r.returncode != 0 and snap(root) == s0,
          (r.stdout + r.stderr)[-200:])
with tempfile.TemporaryDirectory() as root:
    base = tree(root)
    run(root, "r1", "install")
    run(root, "r2", "install")
    s0 = snap(root)
    r = run(root, "r1", "uninstall")
    s1 = snap(root)
    gtxt = open(os.path.join(root, G)).read()
    mtxt = open(os.path.join(root, MR)).read()
    refused = r.returncode != 0 and s1 == s0
    wiped = (r.returncode == 0 and ours.T25 not in gtxt and r1.TAG not in gtxt and ours.T25 in mtxt
             and "result = shared_output + fused_output" in mtxt)
    print(f"   C: r1-pdl uninstall with ours on top -> rc {r.returncode}; "
          f"{'REFUSED, nothing changed' if refused else 'restored its backup (our gather hook gone)' if wiped else 'OTHER'}: "
          f"{(r.stdout + r.stderr).strip()[-220:]}")
    check("C: wrong-order uninstall leaves no half-state (refuses untouched, or G25 inert)", refused or wiped)
print("ALL PASS" if not FAILS else f"FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
