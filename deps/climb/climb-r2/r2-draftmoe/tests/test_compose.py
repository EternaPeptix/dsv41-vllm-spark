#!/usr/bin/env python3
"""r2-draftmoe composition tests with the REAL installers of r1-pdl and r2-moefront, on scratch copies.

Run: python3 tests/test_compose.py   (stdlib only; needs ~/Documents/dsv41-offline/climb-r1/r1-pdl and
     ~/Documents/dsv41-offline/climb-r2/r2-moefront; override with R2DM_R1PDL / R2DM_R2MF)

Scenarios (every file of every kit is checked byte for byte against snapshots):
  S1  LIFO, documented order: r1-pdl -> r2-moefront -> ours; uninstall ours -> moefront -> r1-pdl = live bases.
  S2  FIFO attempts: with our lines present, r1-pdl's and r2-moefront's uninstalls refuse and change nothing; ours
      then uninstalls to the exact pre-install state, after which theirs work.
  S3  the review's blind-restore case: r1-pdl -> ours, then r1-pdl's .bak copied over the file (what a blind restore
      does); our uninstall must leave the live base 58e264c8 (never resurrect c3a05cca) and exit 0.
  S4  ours first, r2-moefront on top: the moefront installer as it is today refuses (and changes nothing); a copy of
      it that also strips '# r2-draftmoe' installs on top; our status says PATCHED; uninstall in BOTH orders gives the
      exact other-kit-only file and then the live bases.
  S5  r1-pdl -> ours -> moefront(+our tag): our uninstall gives exactly r1-pdl -> moefront.
  Flag-off AST: in S1 the composed files with our flags folded off == the r1+moefront files.
"""
import ast
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(KIT, "base")
META = json.load(open(os.path.join(KIT, "KIT.json")))
OURS = [f["path"] for f in META["files"] if f["action"] == "patch"]
NEWMOD = "model_executor/layers/fused_moe/experts/dsv41_r2_draftmoe.py"
R1PDL = os.environ.get("R2DM_R1PDL") or os.path.expanduser("~/Documents/dsv41-offline/climb-r1/r1-pdl")
R2MF = os.environ.get("R2DM_R2MF") or os.path.expanduser("~/Documents/dsv41-offline/climb-r2/r2-moefront")
TAG = "# r2-draftmoe"
LIVE_DGUTL = "58e264c8b870196fd006e80dc6484247820d75ec1561705b2fe3f1f74bba8cd6"
DGUTL = "model_executor/layers/fused_moe/deep_gemm_utils.py"
SCR = tempfile.mkdtemp(prefix="r2dm-compose-")
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail and not ok else ""), flush=True)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else "-"


def tree():
    """V (site vllm) + H (home) with every file any of the three kits touches, at its recorded live base."""
    root = tempfile.mkdtemp(dir=SCR)
    V, H = os.path.join(root, "vllm"), os.path.join(root, "home")
    for rel in OURS:
        os.makedirs(os.path.dirname(os.path.join(V, rel)), exist_ok=True)
        shutil.copy2(os.path.join(BASE, rel), os.path.join(V, rel))
    for src in glob.glob(os.path.join(R1PDL, "base", "**", "*.py"), recursive=True):
        rel = os.path.relpath(src, os.path.join(R1PDL, "base"))
        top, rest = rel.split("/", 1)
        dst = os.path.join(V if top == "vllm" else H, rest)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
    os.makedirs(os.path.join(V, "triton_utils"), exist_ok=True)
    return V, H


def snap(V, H):
    out = {}
    for d in (V, H):
        for p in glob.glob(os.path.join(d, "**", "*"), recursive=True):
            if os.path.isfile(p):
                out[os.path.relpath(p, os.path.dirname(V))] = sha(p)
    return out


def code_snap(V, H):
    """Snapshot without our informational .bak copies (they are removed by our uninstall)."""
    return {k: v for k, v in snap(V, H).items() if ".bak-" not in k and ".tmp-" not in k}


def run(kind, V, H, *args, mf_dir=None):
    env = dict(os.environ)
    for k in ("DSV41_DRAFT_MOE_G17", "DSV41_GLUE_PDL", "DSV41_MOE_ROUTEPREP"):
        env.pop(k, None)
    if kind == "ours":
        env.update(R2DM_VLLM=V, R2DM_HOME=H)
        script = os.path.join(KIT, "install.py")
    elif kind == "r1":
        env.update(R1PDL_VLLM=V, R1PDL_HOME=H)
        script = os.path.join(R1PDL, "install.py")
    else:
        env.update(R2MF_VLLM=V)
        script = os.path.join(mf_dir or R2MF, "install.py")
    r = subprocess.run([sys.executable, script, *args], env=env, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def mf_with_our_tag():
    """A copy of the r2-moefront kit whose installer also strips '# r2-draftmoe' (the change it needs to accept us)."""
    d = os.path.join(SCR, "r2-moefront-plus")
    if not os.path.isdir(d):
        shutil.copytree(R2MF, d, ignore=shutil.ignore_patterns("__pycache__", "results"))
        p = os.path.join(d, "install.py")
        s = open(p).read()
        old = 'ROUND1_TAGS = ("# r1-pdl",)'
        assert s.count(old) == 1, "r2-moefront ROUND1_TAGS anchor moved"
        open(p, "w").write(s.replace(old, 'ROUND1_TAGS = ("# r1-pdl", "# r2-draftmoe")'))
    return d


def ours_status_ok(V, H):
    rc, out = run("ours", V, H, "status")
    return rc == 0 and out.count("PATCHED") == 5 and "FOREIGN" not in out and "UNKNOWN" not in out, out


# ---------------------------------------------------------------- flag-off AST folding (shared with test_install)
_ti = open(os.path.join(KIT, "tests", "test_install.py")).read()
_ns = {"ast": ast, "copy": __import__("copy"), "FLAG_NAMES": {"_R2DM_FC1": False, "_R2DM_PERM": "", "_r2dm_fp": False}}
for node in ast.parse(_ti).body:
    if isinstance(node, ast.ClassDef) and node.name == "Fold":
        exec(compile(ast.Module([node], []), "test_install.py", "exec"), _ns)
Fold = _ns["Fold"]

check("r1-pdl and r2-moefront kits present", os.path.isfile(os.path.join(R1PDL, "install.py"))
      and os.path.isfile(os.path.join(R2MF, "install.py")), f"{R1PDL} {R2MF}")

# ---------------------------------------------------------------- S1 LIFO, documented order
V, H = tree()
s0 = code_snap(V, H)
rc, out = run("r1", V, H, "install")
check("S1 r1-pdl install on live bases", rc == 0, out[-300:])
s1 = code_snap(V, H)
check("S1 r1-pdl composed deep_gemm_utils == c3a05cca (recorded)",
      sha(os.path.join(V, DGUTL)) == META["files"][3]["base_sha256_with_other_kits"]["r1-pdl"])
rc, out = run("mf", V, H, "install")
check("S1 r2-moefront install on top of r1-pdl", rc == 0, out[-300:])
s2 = code_snap(V, H)
rec = {f["path"]: f.get("base_sha256_with_other_kits", {}) for f in META["files"]}
check("S1 r1-pdl + r2-moefront composed shas recorded in KIT.json",
      sha(os.path.join(V, DGUTL)) == rec[DGUTL]["r1-pdl + r2-moefront"]
      and sha(os.path.join(V, OURS[1])) in rec[OURS[1]].values())
pre = {rel: open(os.path.join(V, rel)).read() for rel in OURS}
rc, out = run("ours", V, H, "install")
check("S1 r2-draftmoe install on top of r1-pdl + r2-moefront", rc == 0, out[-400:])
ok, out = ours_status_ok(V, H)
check("S1 our status: 4 PATCHED + module", ok, out)
rc, out = run("mf", V, H, "status")
print("INFO  S1 today's r2-moefront `status` with our lines on top (its strip list lacks '# r2-draftmoe'): "
      + " | ".join(l.split()[1] for l in out.splitlines()[:2]))
rc, out = run("mf", V, H, "status", mf_dir=mf_with_our_tag())
check("S1 r2-moefront(+our tag) status: 'patched' for both files with our lines on top",
      rc == 0 and out.count(" patched ") == 2 and "FOREIGN" not in out, out)
for rel in OURS:
    src = open(os.path.join(V, rel)).read()
    try:
        compile(src, rel, "exec")
        comp = True
    except SyntaxError:
        comp = False
    same = ast.dump(ast.parse(pre[rel])) == ast.dump(Fold().visit(ast.parse(src)))
    check(f"S1 composed {os.path.basename(rel)} compiles; flag-off AST == r1+moefront AST", comp and same)
rc, out = run("ours", V, H, "install")
check("S1 our second install is idempotent on the composed files", rc == 0 and out.count("already patched") == 4, out)
rc, out = run("ours", V, H, "uninstall")
check("S1 our uninstall -> exact r1-pdl + r2-moefront state (all files, bytes)", rc == 0 and code_snap(V, H) == s2,
      out[-300:])
rc, out = run("mf", V, H, "uninstall")
check("S1 r2-moefront uninstall -> exact r1-pdl state", rc == 0 and code_snap(V, H) == s1, out[-300:])
rc, out = run("r1", V, H, "uninstall")
check("S1 r1-pdl uninstall -> live bases (all files, bytes)", rc == 0 and code_snap(V, H) == s0, out[-300:])

# ---------------------------------------------------------------- S2 FIFO attempts
V, H = tree()
s0 = code_snap(V, H)
run("r1", V, H, "install")
s1 = code_snap(V, H)
run("mf", V, H, "install")
s2 = code_snap(V, H)
rc, out = run("ours", V, H, "install")
s3 = snap(V, H)
rc1, out1 = run("r1", V, H, "uninstall")
check("S2 r1-pdl uninstall refuses while our lines are present, changes nothing", rc1 != 0 and snap(V, H) == s3, out1)
rc2, out2 = run("mf", V, H, "uninstall")
check("S2 r2-moefront uninstall (today's) refuses while our lines are present, changes nothing",
      rc2 != 0 and snap(V, H) == s3, out2[-300:])
rc, out = run("ours", V, H, "uninstall")
check("S2 our uninstall still gives the exact r1+moefront state", rc == 0 and code_snap(V, H) == s2, out[-300:])
run("mf", V, H, "uninstall")
run("r1", V, H, "uninstall")
check("S2 then moefront + r1-pdl uninstall -> live bases", code_snap(V, H) == s0)

# ---------------------------------------------------------------- S3 blind restore by another kit
V, H = tree()
s0 = code_snap(V, H)
run("r1", V, H, "install")
rc, out = run("ours", V, H, "install")
check("S3 ours on top of r1-pdl", rc == 0, out[-300:])
p = os.path.join(V, DGUTL)
shutil.copy2(p + ".bak-r1-pdl", p)  # what a blind (FIFO) r1-pdl restore did in the review's simulation
rc, out = run("ours", V, H, "uninstall")
check("S3 after a blind restore our uninstall exits 0 and leaves deep_gemm_utils at the live 58e264c8 (no c3a05cca)",
      rc == 0 and sha(p) == LIVE_DGUTL and not os.path.exists(p + ".bak-r2-draftmoe"), out[-300:])
check("S3 our other 3 files back at live bases, module removed",
      all(sha(os.path.join(V, rel)) == sha(os.path.join(BASE, rel)) for rel in OURS)
      and not os.path.exists(os.path.join(V, NEWMOD)))

# ---------------------------------------------------------------- S4 ours first, moefront on top
V, H = tree()
s0 = code_snap(V, H)
run("ours", V, H, "install")
sA = snap(V, H)
rc, out = run("mf", V, H, "install")
check("S4 today's r2-moefront installer refuses a file with our lines (documented: install it first) and changes "
      "nothing", rc != 0 and snap(V, H) == sA, out[-200:])
mfp = mf_with_our_tag()
rc, out = run("mf", V, H, "install", mf_dir=mfp)
check("S4 r2-moefront that strips '# r2-draftmoe' installs on top of ours", rc == 0, out[-300:])
ok, out = ours_status_ok(V, H)
check("S4 our status with moefront on top: 4 PATCHED + module (not FOREIGN)", ok, out)
rc, out = run("ours", V, H, "install")
check("S4 our install idempotent with moefront on top", rc == 0 and out.count("already patched") == 4, out)
sB = code_snap(V, H)
# reference: moefront alone on a clean tree
V2, H2 = tree()
run("mf", V2, H2, "install", mf_dir=mfp)
ref_mf = {rel: sha(os.path.join(V2, rel)) for rel in OURS}
V3, H3 = tree()
run("ours", V3, H3, "install")
ref_ours = {rel: sha(os.path.join(V3, rel)) for rel in OURS}
rc, out = run("ours", V, H, "uninstall")
check("S4 FIFO: our uninstall first -> exactly moefront-alone files", rc == 0
      and {rel: sha(os.path.join(V, rel)) for rel in OURS} == ref_mf and not os.path.exists(os.path.join(V, NEWMOD)),
      out[-300:])
rc, out = run("mf", V, H, "uninstall", mf_dir=mfp)
check("S4 then moefront uninstall -> live bases", rc == 0 and code_snap(V, H) == s0, out[-300:])
V, H = tree()
run("ours", V, H, "install")
run("mf", V, H, "install", mf_dir=mfp)
rc, out = run("mf", V, H, "uninstall", mf_dir=mfp)
check("S4 LIFO: moefront uninstall first -> exactly ours-alone files", rc == 0
      and {rel: sha(os.path.join(V, rel)) for rel in OURS} == ref_ours, out[-300:])
rc, out = run("ours", V, H, "uninstall")
check("S4 then our uninstall -> live bases", rc == 0 and code_snap(V, H) == s0, out[-300:])

# ---------------------------------------------------------------- S5 r1-pdl -> ours -> moefront(+tag)
V, H = tree()
run("r1", V, H, "install")
rc, out = run("ours", V, H, "install")
rc2, out2 = run("mf", V, H, "install", mf_dir=mfp)
check("S5 r1-pdl -> ours -> moefront(+tag) all install", rc == 0 and rc2 == 0, (out + out2)[-300:])
V4, H4 = tree()
run("r1", V4, H4, "install")
run("mf", V4, H4, "install", mf_dir=mfp)
ref = {rel: sha(os.path.join(V4, rel)) for rel in OURS}
rc, out = run("ours", V, H, "uninstall")
check("S5 our uninstall -> exactly r1-pdl -> moefront files", rc == 0
      and {rel: sha(os.path.join(V, rel)) for rel in OURS} == ref, out[-300:])

shutil.rmtree(SCR, ignore_errors=True)
n_fail = results.count(False)
print(f"\n{len(results) - n_fail}/{len(results)} passed")
sys.exit(1 if n_fail else 0)
