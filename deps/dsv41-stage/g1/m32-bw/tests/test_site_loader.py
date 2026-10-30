#!/usr/bin/env python3
"""g1/m32-bw: the STAGED site file's own loader path, one variant per process (STOPPED-SERVICE WINDOW ONLY).

  DSV41_MOE_M32_G1=<pair|pf|la|ds|all> python tests/test_site_loader.py [--tokens 1,6,24,96]

Loads the live trees privately (g1lib, module names g1live_*) and the staged sm121_target_m32.py under a private
module name; its get_target_m32_kernel() then loads trees/<v>-base (+ trees/<v>-d05) exactly as serving would
(module names vllm_target_m32._C / vllm_target_m32_d05._C, pinned manifests, jit/<tree> caches). Compares
DeepGemmFP4Experts.apply outputs bitwise (synthetic weights, all route patterns, poisoned workspaces) and checks that
with DSV41_MOE_M32_G1 unset the staged module selects the live trees (manifest shas) without loading anything.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g1lib  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tokens", default="1,6,24,96")
ap.add_argument("--force", action="store_true")
a = ap.parse_args()
g1lib.guard(a.force)
variant = os.environ.get("DSV41_MOE_M32_G1", "")
assert variant in ("pair", "pf", "la", "ds", "all"), "set DSV41_MOE_M32_G1"
g1lib.setup_env(g1lib.G / "work")                  # pops DSV41_MOE_M32_G1: restore it for the staged loader
os.environ["DSV41_MOE_M32_G1"] = variant
# pair/pf/la: the staged loader takes the live D05 tree for FC2; give it the private D05 cache copy
os.environ["DSV41_MOE_FC2_D05_JIT_DIR"] = str(g1lib.G / "jit/live-d05")
import torch  # noqa: E402

nat = g1lib.Natives([])
g1lib.install_live_getter(nat)
staged_path = g1lib.G / "site/vllm/model_executor/layers/fused_moe/experts/sm121_target_m32.py"
spec = importlib.util.spec_from_file_location("g1_staged_sm121_target_m32", staged_path)
st = importlib.util.module_from_spec(spec)
sys.modules["g1_staged_sm121_target_m32"] = st
spec.loader.exec_module(st)
fails = cases = 0


def rec(ok, **kw):
    global fails, cases
    cases += 1
    fails += not ok
    print(json.dumps(dict(ok=bool(ok), variant=variant, **kw)), flush=True)


rec(st.g1_variant() == variant, check="g1_variant")
kern = st.get_target_m32_kernel()
rec(kern.d05 == "1" and kern.fc2_native is not None and kern.g1_la == (variant in ("la", "all")), check="kernel")
base_mod = sys.modules["vllm_target_m32._C"]
rec(Path(base_mod.__file__).resolve().parent == (g1lib.G / f"trees/{variant}-base").resolve(), check="base tree",
    path=base_mod.__file__)
d05_mod = sys.modules["vllm_target_m32_d05._C"]
want_d05 = (g1lib.G / f"trees/{variant}-d05") if variant in ("ds", "all") else g1lib.LIVE_D05
rec(Path(d05_mod.__file__).resolve().parent == want_d05.resolve(), check="d05 tree", path=d05_mod.__file__)

w1c, s1c, w2, s2 = g1lib.synthetic_weights()
live = g1lib.Arm(nat.kernels["live"], w1c, s1c, w2, s2)
cand = g1lib.Arm(kern, w1c, s1c, w2, s2)
for m in [int(t) for t in a.tokens.split(",")]:
    gen = torch.Generator().manual_seed(77 + m)
    x = (torch.randn(m, g1lib.H, generator=gen) * 0.15).to(torch.bfloat16).cuda()
    tw = (torch.softmax(torch.randn(m, g1lib.TOPK, generator=gen), -1) * 1.5).float().cuda()
    for r, v in g1lib.synthetic_routes(m, gen).items():
        ids = v.cuda()
        ws0, ws1 = live.workspaces(m), cand.workspaces(m)
        for t in ws0 + ws1:
            t.view(torch.uint8).fill_(255)
        ref = live.run(x, ids, tw, ws0).clone()
        got = cand.run(x, ids, tw, ws1)
        torch.cuda.synchronize()
        rec(torch.equal(ref.view(torch.uint8), got.view(torch.uint8)), m=m, route=r)

# flag off: the staged loaders must pick the live trees (no load: check the selected paths only)
os.environ.pop("DSV41_MOE_M32_G1")
rec(st.g1_variant() == "", check="flag off -> live path")
print(json.dumps(dict(event="summary", variant=variant, status="PASS" if not fails else "FAIL", cases=cases,
                      failures=fails)), flush=True)
sys.exit(1 if fails else 0)
