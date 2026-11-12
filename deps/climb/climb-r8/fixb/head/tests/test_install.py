#!/usr/bin/env python3
"""r6-head installer + flag-off tests (stdlib only; runs on any CPU host).

  1. install / status / idempotent re-install / uninstall on a copy of the live base: byte-exact restore, backups.
  2. refusals with nothing written: drifted base, foreign-tagged file, stale tmp, half-edited file on uninstall.
  3. flag-off identity: every inserted line carries the tag; stripping them gives the base byte for byte; the AST of
     the patched module differs from the base only by the three tagged statements; the flag expression is False for
     unset / 0 / off / false / no / '' and True for row / tile / 1; certified_logits' first statement is the gated
     dispatch and the rest of its body is the base body verbatim.
  4. composition: (a) climb-r1 integrate/composed (= live) dsv41_r4_small.py == the base; (b) no r5 kit lists
     dsv41_r4_small.py; (c) the r5-composed tree's callers of dsv41_r4_small (model.py `_r4.` lines, dsv41_l2pf_r1
     _wrap_cert) are identical to the live ones, so the r6 dispatch sees the same calls with and without r5; (d) the
     installer applied on the r5-composed tree (with the live r4_small inserted) gives the same patched bytes.
Usage: python3 tests/test_install.py   (exit 0 = all pass)
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

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFF = os.path.normpath(os.path.join(HERE, "..", ".."))  # ~/Documents/dsv41-offline
REL = "models/deepseek_v41/nvidia/dsv41_r4_small.py"
BASE = os.path.join(HERE, "base/vllm", REL)
FAILS = []
N = [0]


def check(name, ok, detail=""):
    N[0] += 1
    print(f"{'PASS' if ok else 'FAIL'} {name}{(': ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def run(tree, *args):
    env = dict(os.environ, R6H_VLLM=tree)
    p = subprocess.run([sys.executable, os.path.join(HERE, "install.py"), *args], env=env, capture_output=True,
                       text=True)
    return p.returncode, p.stdout + p.stderr


def mktree(src_r4=BASE):
    d = tempfile.mkdtemp(prefix="r6h-")
    os.makedirs(os.path.join(d, os.path.dirname(REL)))
    shutil.copy2(src_r4, os.path.join(d, REL))
    return d


def snapshot(tree):
    out = {}
    for root, _, files in os.walk(tree):
        for f in files:
            p = os.path.join(root, f)
            out[os.path.relpath(p, tree)] = sha(p)
    return out


spec = importlib.util.spec_from_file_location("r6inst", os.path.join(HERE, "install.py"))
inst = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inst)
kitj = json.load(open(os.path.join(HERE, "KIT.json")))
base_sha = [f for f in kitj["files"] if f["action"] == "patch"][0]["base_sha256"]
check("base copy == KIT.json base sha", sha(BASE) == base_sha, sha(BASE)[:16])
live_copy = os.path.join(HERE, "live/dsv41_r4_small.py")
if os.path.exists(live_copy):
    check("base copy == live copy fetched from spark-one", sha(live_copy) == base_sha)

# ---- 1. install / uninstall round trip
t = mktree()
before = snapshot(t)
rc, out = run(t, "status")
check("status on base: NOT-INSTALLED (exit 0)", rc == 0 and "NOT-INSTALLED" in out, out.strip().splitlines()[-1])
rc, out = run(t, "dry")
check("dry: +7 tagged lines", rc == 0 and "+7 tagged lines" in out, out.strip().replace("\n", " | "))
check("dry writes nothing", snapshot(t) == before)
rc, out = run(t, "install")
check("install exit 0 + INSTALLED", rc == 0 and "OVERALL INSTALLED" in out, out.strip().splitlines()[-1])
pfile = os.path.join(t, REL)
patched_sha = sha(pfile)
check("backup .bak-r6-head == base", sha(pfile + ".bak-r6-head") == base_sha)
check("new module copied verbatim",
      sha(os.path.join(t, "models/deepseek_v41/nvidia/dsv41_head_r6.py")) ==
      sha(os.path.join(HERE, "files/vllm/models/deepseek_v41/nvidia/dsv41_head_r6.py")))
kit_patched = [f.get("patched_sha256") for f in kitj["files"] if f["action"] == "patch"][0]
if kit_patched:
    check("patched sha == KIT.json patched_sha256", patched_sha == kit_patched, patched_sha[:16])
snap1 = snapshot(t)
rc, out = run(t, "install")
check("re-install idempotent (no byte changes)", rc == 0 and snapshot(t) == snap1)
rc, out = run(t, "status")
check("status INSTALLED exit 0", rc == 0 and "OVERALL INSTALLED" in out)
src = open(pfile).read()
base = open(BASE).read()
tagged = [ln for ln in src.splitlines() if ln.endswith("# dsv41-r6-head")]
check("exactly 7 tagged lines", len(tagged) == 7, str(len(tagged)))
check("strip tagged lines == base (byte-exact)", inst.strip_tag(src) == base)
check("every non-base line is tagged",
      [ln for ln in src.splitlines() if ln not in set(base.splitlines())] == tagged)
rc, out = run(t, "uninstall")
check("uninstall exit 0", rc == 0, out.strip().splitlines()[-1] if out.strip() else "")
check("uninstall byte-exact (tree == before install)", snapshot(t) == before,
      str(set(snapshot(t).items()) ^ set(before.items()))[:200])
shutil.rmtree(t)

# ---- 2. refusals
t = mktree()
p = os.path.join(t, REL)
with open(p, "a") as f:
    f.write("\n")
b = snapshot(t)
rc, out = run(t, "install")
check("refuses drifted base, nothing written", rc != 0 and "not an accepted base" in out and snapshot(t) == b)
shutil.rmtree(t)
t = mktree()
p = os.path.join(t, REL)
s = open(p).read().replace("CERT_GROUP = 128\n", "CERT_GROUP = 128  # dsv41-r6-head\n")
open(p, "w").write(s)
b = snapshot(t)
rc, out = run(t, "install")
check("refuses foreign/odd tagged file, nothing written", rc != 0 and snapshot(t) == b)
rc, out = run(t, "uninstall")
check("uninstall refuses a file that is not exactly our patch", rc != 0 and snapshot(t) == b)
shutil.rmtree(t)
t = mktree()
open(os.path.join(t, REL) + ".tmp-r6-head", "w").write("x")
b = snapshot(t)
rc, out = run(t, "install")
check("refuses with a stale tmp file, nothing written", rc != 0 and snapshot(t) == b)
shutil.rmtree(t)
t = mktree()
run(t, "install")
p = os.path.join(t, REL)
s = open(p).read()
s2 = s.replace("    if _DSV41_R6_HEAD and not full:  # dsv41-r6-head\n", "", 1)
open(p, "w").write(s2)
b = snapshot(t)
rc, out = run(t, "status")
check("status flags a half-removed patch (exit 2, FOREIGN-PATCHED)", rc == 2 and "FOREIGN-PATCHED" in out)
rc, out = run(t, "install")
check("install refuses a half-removed patch", rc != 0 and snapshot(t) == b)
shutil.rmtree(t)

# ---- 2b. the new module: foreign bytes refused on install and uninstall, a known earlier kit version upgraded
MODREL = "models/deepseek_v41/nvidia/dsv41_head_r6.py"
PRIOR = os.path.join(HERE, "tests/fixtures/dsv41_head_r6.prior-9bd17b03.py")
modent = [f for f in kitj["files"] if f["action"] == "new"][0]
check("KIT.json module sha256 == the kit file", modent["sha256"] == sha(os.path.join(HERE, "files/vllm", MODREL)))
check("prior fixture sha is listed in KIT.json prior_sha256", os.path.exists(PRIOR)
      and sha(PRIOR) in (modent.get("prior_sha256") or []))
t = mktree()
open(os.path.join(t, MODREL), "w").write("# someone else's module\n")
b = snapshot(t)
rc, out = run(t, "install")
check("install refuses a FOREIGN dsv41_head_r6.py, nothing written", rc != 0 and "refusing" in out and snapshot(t) == b,
      out.strip().splitlines()[-1] if out.strip() else "")
shutil.rmtree(t)
t = mktree()
run(t, "install")
open(os.path.join(t, MODREL), "w").write("# replaced after install\n")
b = snapshot(t)
rc, out = run(t, "uninstall")
check("uninstall refuses when the module is FOREIGN, nothing changed", rc != 0 and snapshot(t) == b)
rc, out = run(t, "status")
check("status reports FOREIGN module and PARTIAL (exit 2)", rc == 2 and "FOREIGN" in out and "PARTIAL" in out)
shutil.rmtree(t)
if os.path.exists(PRIOR):
    t = mktree()
    run(t, "install")
    shutil.copy2(PRIOR, os.path.join(t, MODREL))  # an earlier kit version installed
    rc, out = run(t, "status")
    check("status: earlier kit module -> STALE-PRIOR, PARTIAL", rc == 2 and "STALE-PRIOR" in out)
    rc, out = run(t, "install")
    check("install upgrades a known earlier module version -> INSTALLED", rc == 0 and "upgraded" in out
          and "OVERALL INSTALLED" in out and sha(os.path.join(t, MODREL)) == modent["sha256"])
    shutil.copy2(PRIOR, os.path.join(t, MODREL))
    rc, out = run(t, "uninstall")
    check("uninstall with a known earlier module version: byte-exact base, module removed",
          rc == 0 and sha(os.path.join(t, REL)) == base_sha and not os.path.exists(os.path.join(t, MODREL)))
    shutil.rmtree(t)

# ---- 3. flag-off identity
t = mktree()
run(t, "install")
psrc = open(os.path.join(t, REL)).read()
shutil.rmtree(t)
pa, ba = ast.parse(psrc), ast.parse(base)


def top_dump(tree):
    return [ast.dump(n) for n in tree.body]


extra = [d for d in top_dump(pa) if d not in top_dump(ba)]
check("module level: exactly 3 changed top-level statements (flag assign, gated import, certified_logits)",
      len(extra) == 3 and sum(d.startswith("FunctionDef(name='certified_logits'") for d in extra) == 1, str(len(extra)))
flagassign = [n for n in pa.body if isinstance(n, ast.Assign) and any(getattr(t_, "id", "") == "_DSV41_R6_HEAD"
                                                                       for t_ in n.targets)]
gate = [n for n in pa.body if isinstance(n, ast.If) and getattr(n.test, "id", "") == "_DSV41_R6_HEAD"]
check("gated import is the only statement under `if _DSV41_R6_HEAD:` and imports dsv41_head_r6",
      len(gate) == 1 and len(gate[0].body) == 1 and isinstance(gate[0].body[0], ast.ImportFrom)
      and gate[0].body[0].names[0].name == "dsv41_head_r6" and not gate[0].orelse)


def fn(tree, name):
    return [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name][0]


pf, bf = fn(pa, "certified_logits"), fn(ba, "certified_logits")
check("certified_logits signature unchanged", ast.dump(pf.args) == ast.dump(bf.args))
check("certified_logits body = docstring + gated dispatch + base body verbatim",
      [ast.dump(n) for n in pf.body[:1] + pf.body[2:]] == [ast.dump(n) for n in bf.body])
disp = pf.body[1]
check("dispatch test is `_DSV41_R6_HEAD and not full`",
      isinstance(disp, ast.If) and ast.unparse(disp.test) == "_DSV41_R6_HEAD and (not full)" and not disp.orelse
      and len(disp.body) == 1 and isinstance(disp.body[0], ast.Return))
others = [n.name for n in pa.body if isinstance(n, ast.FunctionDef) and n.name != "certified_logits"]
check("every other function byte-identical (AST)",
      all(ast.dump(fn(pa, n_)) == ast.dump(fn(ba, n_)) for n_ in others), f"{len(others)} functions")
expr = ast.unparse(flagassign[0].value)
_OFF = ("", "0", "off", "false", "no")
for env, want in [({}, False), ({"DSV41_R6_HEAD_REFINE": "0"}, False), ({"DSV41_R6_HEAD_REFINE": "off"}, False),
                  ({"DSV41_R6_HEAD_REFINE": " False "}, False), ({"DSV41_R6_HEAD_SCREEN": "no"}, False),
                  ({"DSV41_R6_HEAD_SCREEN": ""}, False), ({"DSV41_R6_HEAD_REFINE": "row"}, True),
                  ({"DSV41_R6_HEAD_REFINE": "tile"}, True), ({"DSV41_R6_HEAD_SCREEN": "1"}, True),
                  ({"DSV41_R6_HEAD_REFINE_CFG": "256,4,4,0,1"}, False)]:
    class _Env(dict):
        pass
    fake_os = type("o", (), {"environ": env})
    got = eval(expr, {"os": fake_os, "_OFF": _OFF})
    check(f"flag expr {env or 'unset'} -> {want}", got == want)

# ---- 4. composition
r1c = os.path.join(OFF, "climb-r1/integrate/composed/venvs/vllm032/lib/python3.12/site-packages/vllm", REL)
if os.path.exists(r1c):
    check("climb-r1 composed (live) dsv41_r4_small.py == base", sha(r1c) == base_sha)
r5 = os.path.join(OFF, "climb-r5")
listed = []
for k in ("glue", "ffnfront", "prio", "harness"):
    kj = os.path.join(r5, k, "KIT.json")
    if os.path.exists(kj):
        listed += [f["path"] for f in json.load(open(kj)).get("files", []) if "dsv41_r4_small" in f["path"]]
check("no r5 kit lists dsv41_r4_small.py", not listed, str(listed))
r5live = os.path.join(r5, "integrate/live/vllm")
r5comp = os.path.join(r5, "integrate/composed/vllm")
for rel, pat in [("models/deepseek_v41/nvidia/model.py", r"_r4\.|dsv41_r4_small"),
                 ("distributed/device_communicators/dsv41_l2pf_r1.py", r"certified_logits|R4_MODULE|_wrap_cert")]:
    a, b2 = os.path.join(r5live, rel), os.path.join(r5comp, rel)
    if os.path.exists(a) and os.path.exists(b2):
        la = [ln.strip() for ln in open(a) if re.search(pat, ln)]
        lb = [ln.strip() for ln in open(b2) if re.search(pat, ln)]
        check(f"r5-composed {os.path.basename(rel)}: dsv41_r4_small call/wrap lines == live ({len(la)} lines)",
              la == lb and len(la) > 0)
    else:
        check(f"r5 live/composed copies of {rel} present", False, "missing")
# the r5 composed tree does not carry r4_small; drop the live one in and install there
if os.path.isdir(r5comp):
    t = tempfile.mkdtemp(prefix="r6h-r5-")
    shutil.copytree(r5comp, os.path.join(t, "vllm"))
    tv = os.path.join(t, "vllm")
    os.makedirs(os.path.join(tv, os.path.dirname(REL)), exist_ok=True)
    shutil.copy2(BASE, os.path.join(tv, REL))
    b = snapshot(tv)
    rc, out = run(tv, "install")
    check("install on the r5-composed tree: INSTALLED, same patched bytes", rc == 0 and "OVERALL INSTALLED" in out
          and sha(os.path.join(tv, REL)) == patched_sha)
    rc, out = run(tv, "uninstall")
    check("uninstall on the r5-composed tree: byte-exact", rc == 0 and snapshot(tv) == b)
    shutil.rmtree(t)

print(f"\n{N[0] - len(FAILS)}/{N[0]} PASS" + (f"; FAILED: {FAILS}" if FAILS else ""))
print(f"patched sha256 {patched_sha}")
sys.exit(1 if FAILS else 0)
