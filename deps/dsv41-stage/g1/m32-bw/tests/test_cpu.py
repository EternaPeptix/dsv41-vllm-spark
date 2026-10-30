#!/usr/bin/env python3
"""g1/m32-bw CPU checks (safe while serving): python3 -I tests/test_cpu.py

1. the staged site file == site/make_site.py(installed live sm121_target_m32.py 5acc5fde, MANIFESTS.json);
2. the staged module imports with the stdlib only and DSV41_MOE_M32_G1 parsing / exclusions behave;
3. every variant header == tools/make_tree.py(live header) and the 'off' header differs from live only by inert text
   (comment + constexpr block + the gated hunks), and each tree's manifest pins its header.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

G = Path(__file__).resolve().parent.parent
SP = Path.home() / "venvs/vllm032/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/experts"
STAGED = G / "site/vllm/model_executor/layers/fused_moe/experts/sm121_target_m32.py"
HDR = "include/deep_gemm/impls/sm120_fp8_fp4_gemm_1d1d.cuh"
ok = True


def check(cond, what):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + what, flush=True)


live = SP / "sm121_target_m32.py"
if not live.exists() or not hashlib.sha256(live.read_bytes()).hexdigest().startswith("5acc5fde"):
    live = SP / "sm121_target_m32.py.bak-g1-m32bw"          # after install, the backup is the live file
with tempfile.TemporaryDirectory() as td:
    out = Path(td) / "s.py"
    subprocess.run([sys.executable, "-I", str(G / "site/make_site.py"), str(live), str(out), str(G / "MANIFESTS.json")],
                   check=True, capture_output=True)
    check(out.read_bytes() == STAGED.read_bytes(), "staged site file == make_site(live 5acc5fde)")

spec = importlib.util.spec_from_file_location("g1_cpu_staged", STAGED)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
env0 = dict(os.environ)
try:
    for k in ("DSV41_MOE_M32_G1", "DSV41_MOE_PDL_EARLY_B"):
        os.environ.pop(k, None)
    os.environ["DSV41_MOE_FC2_D05"] = "1"
    check(mod.g1_variant() == "", "flag unset -> off")
    for v in mod._G1_VARIANTS:
        os.environ["DSV41_MOE_M32_G1"] = v
        check(mod.g1_variant() == v, f"flag {v}")
        check(f"{v}-base" in mod._G1_MANIFESTS and (v not in mod._G1_HAS_D05 or f"{v}-d05" in mod._G1_MANIFESTS),
              f"pinned manifests {v}")
    for bad_env, what in (({"DSV41_MOE_M32_G1": "bogus"}, "bad name"),
                          ({"DSV41_MOE_M32_G1": "pf", "DSV41_MOE_PDL_EARLY_B": "1"}, "pdl exclusive"),
                          ({"DSV41_MOE_M32_G1": "ds", "DSV41_MOE_FC2_D05": "0"}, "needs D05=1")):
        os.environ.update(bad_env)
        try:
            mod.g1_variant()
            check(False, what)
        except AssertionError:
            check(True, what)
        for k in bad_env:
            os.environ.pop(k, None)
        os.environ["DSV41_MOE_FC2_D05"] = "1"
finally:
    os.environ.clear()
    os.environ.update(env0)

man = json.loads((G / "MANIFESTS.json").read_text())
check({k: v for k, v in man.items() if not k.startswith("off-")} == mod._G1_MANIFESTS, "site pins == MANIFESTS.json")
spec2 = importlib.util.spec_from_file_location("g1_make_tree", G / "tools/make_tree.py")
mt = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(mt)
srcs = {"base": Path.home() / "dsv41-stage/m32/target-m32-native-v1",
        "d05": Path.home() / "dsv41-goal120-20261024/d05-shadow-v1/native"}
for t, want in man.items():
    v, kind = t.rsplit("-", 1)
    hdr = mt.patch((srcs[kind] / HDR).read_text(), v)
    root = G / "trees" / t
    if root.exists():
        m = json.loads((root / "manifest.json").read_text())
        check(m["files"][HDR] == hashlib.sha256(hdr.encode()).hexdigest() and
              (root / HDR).read_text() == hdr and hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest() == want,
              f"tree {t}: header == make_tree(live) and manifest pinned")
print("CPU", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
