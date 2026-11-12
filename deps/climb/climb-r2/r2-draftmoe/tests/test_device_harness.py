#!/usr/bin/env python3
"""r2-draftmoe: CPU checks of device_test.py that need no GPU: it parses, and its launcher-env reader picks up the
production MoE defaults from the live launcher copy (live-ref/dsv41-tp8.sh) and never sets DSV41_DRAFT_MOE_G17."""
import ast
import os
import re
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = open(os.path.join(HERE, "device_test.py")).read()
tree = ast.parse(src)
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "apply_launcher_env")
home = tempfile.mkdtemp()
shutil.copy(os.path.join(HERE, "live-ref", "dsv41-tp8.sh"), os.path.join(home, "dsv41-tp8.sh"))
saved = dict(os.environ)
for k in list(os.environ):
    if k.startswith("DSV41_") or k == "VLLM_USE_DEEP_GEMM":
        os.environ.pop(k)
os.environ["DSV41_DRAFT_MOE_G17"] = "1"
ns = {"os": os, "re": re, "HOME": home}
exec(compile(ast.Module(body=[fn], type_ignores=[]), "device_test.py", "exec"), ns)
applied = ns["apply_launcher_env"]()
env = dict(os.environ)
os.environ.clear()
os.environ.update(saved)
shutil.rmtree(home)
want = {"DSV41_SM121_TARGET_M32": "1", "DSV41_DRAFT_MOE_M32": "1", "DSV41_MOE_COMPACT": "1", "DSV41_MOE_LIVE_ACT": "1",
        "DSV41_MOE_LIVE_ACT_MIN_ROWS": "192", "DSV41_MOE_M32_G1": "all", "DSV41_MOE_FC1_DEAL": "deal"}
fails = 0
for k, v in want.items():
    ok = applied.get(k) == v
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'} launcher default {k}={applied.get(k)!r} (want {v!r})")
ok = "DSV41_DRAFT_MOE_G17" not in env
fails += not ok
print(f"{'PASS' if ok else 'FAIL'} DSV41_DRAFT_MOE_G17 removed from the test process environment")
ok = all(v and "$" not in v for v in applied.values())
fails += not ok
print(f"{'PASS' if ok else 'FAIL'} {len(applied)} launcher defaults applied, all expanded ($HOME -> home)")
print(f"device_harness: {'ALL PASS' if not fails else f'{fails} FAIL'}")
sys.exit(1 if fails else 0)
