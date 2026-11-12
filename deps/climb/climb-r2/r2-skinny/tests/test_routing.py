#!/usr/bin/env python3
"""CPU routing test (torch CPU, no GPU, no vllm install): the PATCHED dsv41_r3_dense.py + dsv41_r2_skinny.py run
against stub vllm modules and stub extensions that record every call. Each scenario runs in a fresh subprocess
(module-level flags are read at import).

  off        flags unset: v1 skinny + separate quant, dsv41_r2_skinny never imported, no _r2 attribute
  qfuse-only DSV41_SKINNY_QFUSE=1 alone: identical to off (+ one warning)
  v2         DSV41_SKINNY_V2=1: quant kernel + v2 A8 (pdl False); M outside the self-checked set -> v1
  v2pdl      DSV41_SKINNY_V2=pdl + QFUSE=1: fused v2 (pdl True), no quant call; non-contiguous x -> quant + v2 A8
  check      self-check on: a v2 stub that is wrong for M=3 (A8) and M=5 (fused) keeps those M on the live path
  ftz        self-check picks the FTZ mode whose fused quant matches the live quant bytes; none -> QFUSE off
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KIT)

STUBS = {
    "vllm/__init__.py": "",
    "vllm/logger.py": "import logging\nLOG=[]\nclass _L:\n    def __init__(s,n): s.n=n\n"
                      "    def info(s,m,*a): LOG.append(('info', m % a if a else m))\n"
                      "    def warning(s,m,*a): LOG.append(('warning', m % a if a else m))\n"
                      "def init_logger(n): return _L(n)\n",
    "vllm/distributed/__init__.py": "import torch\nCALLS=[]\n"
                                    "def get_tensor_model_parallel_rank(): return 0\n"
                                    "def get_tensor_model_parallel_world_size(): return 8\n"
                                    "def tensor_model_parallel_all_gather(y, dim=-1):\n"
                                    "    CALLS.append(('ag', tuple(y.shape)))\n"
                                    "    return torch.cat([y]*8, dim=dim)\n",
    "vllm/models/__init__.py": "", "vllm/models/deepseek_v41/__init__.py": "",
    "vllm/models/deepseek_v41/nvidia/__init__.py": "",
    "vllm/model_executor/__init__.py": "", "vllm/model_executor/layers/__init__.py": "",
    "vllm/model_executor/layers/fusion/__init__.py": "",
    "vllm/model_executor/layers/fusion/quant_activation.py": "def as_quantized_activation(x, k): return None\n",
    "vllm/model_executor/layers/quantization/__init__.py": "",
    "vllm/model_executor/layers/quantization/utils/__init__.py": "",
    "vllm/model_executor/layers/quantization/utils/quant_utils.py": "kMxfp8Dynamic = 'mxfp8'\n",
    "vllm/model_executor/layers/quantization/utils/mxfp8_utils.py":
        "import torch\nCALLS=[]\n"
        "def mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=False, alignment=0):\n"
        "    CALLS.append(('quant', tuple(x.shape)))\n"
        "    M, K = x.shape\n"
        "    q = (x.float().abs().clamp(max=448)).to(torch.float8_e4m3fn)\n"
        "    s = torch.full((((M+127)//128)*128*((K//32+3)//4)*4,), 120, dtype=torch.uint8)\n"
        "    return q, s\n",
}

DRIVER = r'''
import os, sys, json, torch
import vllm.distributed as D
import vllm.model_executor.layers.quantization.utils.mxfp8_utils as Q
from vllm.models.deepseek_v41.nvidia import dsv41_r3_dense as R3
import vllm.logger as L

CALLS = []
BAD_A8 = set(json.loads(os.environ.get("T_BAD_A8", "[]")))
BAD_Q = set(json.loads(os.environ.get("T_BAD_Q", "[]")))
FTZ_MATCH = json.loads(os.environ.get("T_FTZ_MATCH", "[0, 1]"))

def ref_out(xq_or_x, N):
    M = xq_or_x.shape[0]
    return (torch.arange(M * N, dtype=torch.float32).view(M, N) % 7).to(torch.bfloat16)

class V1:
    def fits(self, m, k): return 1 <= m <= 8
    def skinny_mx_a8(self, xq, xs, w, ws, split=0, grid=0):
        CALLS.append(("v1", xq.shape[0])); return ref_out(xq, w.shape[0])

class V2:
    def prepare(self): CALLS.append(("prepare",))
    def supports(self, M, N, K, grid=0): return 1 <= M <= 8 and K == 5120
    def v1_split(self, N, K, grid=0): return 8
    def skinny_v2_a8(self, xq, xs, w, ws, grid, pdl):
        M = xq.shape[0]; CALLS.append(("v2a8", M, bool(pdl)))
        y = ref_out(xq, w.shape[0])
        return y + 1 if M in BAD_A8 else y
    def skinny_v2_q(self, x, w, ws, grid, pdl, ftz):
        M = x.shape[0]; CALLS.append(("v2q", M, bool(pdl), int(ftz), bool(x.is_contiguous())))
        y = ref_out(x, w.shape[0])
        return y + 1 if M in BAD_Q else y
    def quant_debug(self, x, ftz):
        q, s = Q.mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
        sr = R3.unswizzle_scale(s, x.shape[0], x.shape[1])
        if ftz not in FTZ_MATCH: sr = sr + 1
        return q.view(torch.uint8), sr, x

R3._ext = V1()
if R3.R2_V2:
    try:
        from vllm.models.deepseek_v41.nvidia import dsv41_r2_skinny as S
        S._ext = V2()
    except ImportError:
        pass

class Lin(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1792, 5120, dtype=torch.float8_e4m3fn), requires_grad=False)
        self.weight_scale = torch.nn.Parameter(torch.full((1792, 160), 120, dtype=torch.uint8), requires_grad=False)

class Attn(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.fused_wqa_wkv = Lin(); self.prefix = "layers.0.attn"

attn = Attn()
out = {"r2_imported": "vllm.models.deepseek_v41.nvidia.dsv41_r2_skinny" in sys.modules,
       "R2_V2": R3.R2_V2, "R2_QFUSE": R3.R2_QFUSE, "runs": []}
for M in json.loads(os.environ.get("T_MS", "[1, 3, 5, 8]")):
    for contig in (True, False):
        CALLS.clear(); Q.CALLS.clear(); D.CALLS.clear()
        base = torch.randn(M, 5120 * (1 if contig else 2)).to(torch.bfloat16)
        x = base if contig else base[:, ::2]
        y = R3.qkv_a_sharded(attn, x)
        out["runs"].append({"M": M, "contig": contig, "shape": list(y.shape), "calls": CALLS[:] ,
                            "quant": len([c for c in Q.CALLS if c[1][0] == M]), "ag": D.CALLS[:]})
st = getattr(attn, "_dsv41_qkv_shard", None)
out["has_r2_attr"] = hasattr(st, "_r2")
out["r2_state"] = None
if getattr(st, "_r2", None) is not None:
    out["r2_state"] = {"ok_a8": sorted(st._r2.ok_a8), "ok_q": sorted(st._r2.ok_q), "ftz": st._r2.ftz}
out["log"] = [m for _, m in L.LOG if "SKINNY" in m]
print("RESULT" + json.dumps(out))
'''

FAIL = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        FAIL.append(name)


def scenario(root, env):
    e = {k: v for k, v in os.environ.items() if not k.startswith("DSV41_")}
    e.update(env)
    e["PYTHONPATH"] = root
    p = subprocess.run([sys.executable, "-c", DRIVER], env=e, capture_output=True, text=True, cwd=root)
    line = [l for l in p.stdout.splitlines() if l.startswith("RESULT")]
    if p.returncode or not line:
        print(p.stdout[-3000:], p.stderr[-3000:])
        return None
    return json.loads(line[0][6:])


def main():
    import install  # noqa: E402  (the kit's installer: patched() applies the anchors)

    root = tempfile.mkdtemp()
    try:
        for rel, txt in STUBS.items():
            os.makedirs(os.path.join(root, os.path.dirname(rel)), exist_ok=True)
            open(os.path.join(root, rel), "w").write(txt)
        nv = os.path.join(root, "vllm/models/deepseek_v41/nvidia")
        src = open(os.path.join(KIT, "base", "dsv41_r3_dense.py")).read()
        open(os.path.join(nv, "dsv41_r3_dense.py"), "w").write(install.patched(src, "dsv41_r3_dense.py"))
        shutil.copy2(os.path.join(KIT, "new", "dsv41_r2_skinny.py"), nv)

        r = scenario(root, {})
        ok = r and not r["r2_imported"] and not r["has_r2_attr"] and all(
            [c[0] for c in run["calls"]] == ["v1"] and run["quant"] == 1 and run["shape"] == [run["M"], 1792]
            for run in r["runs"])
        check("off: v1 + separate quant on every call, r2 module never imported, no _r2 attribute", bool(ok))

        r = scenario(root, {"DSV41_SKINNY_QFUSE": "1"})
        ok = r and not r["r2_imported"] and all([c[0] for c in run["calls"]] == ["v1"] and run["quant"] == 1
                                                for run in r["runs"])
        check("QFUSE=1 without V2: identical to off", bool(ok))

        r = scenario(root, {"DSV41_SKINNY_V2": "1", "DSV41_SKINNY_V2_CHECK": "0"})
        ok = r and r["r2_imported"] and all(run["calls"] == [["v2a8", run["M"], False]] and run["quant"] == 1
                                            for run in r["runs"])
        check("V2=1: quant kernel + v2 A8 (pdl off) on every call", bool(ok))

        r = scenario(root, {"DSV41_SKINNY_V2": "pdl", "DSV41_SKINNY_QFUSE": "1", "DSV41_SKINNY_V2_CHECK": "0"})
        ok = r is not None
        for run in (r or {}).get("runs", []):
            if run["contig"]:
                ok &= run["calls"] == [["v2q", run["M"], True, 0, True]] and run["quant"] == 0
            else:
                ok &= run["calls"] == [["v2a8", run["M"], True]] and run["quant"] == 1
            ok &= run["shape"] == [run["M"], 1792] and run["ag"] == [["ag", [run["M"], 224]]]
        check("V2=pdl+QFUSE: fused v2 (pdl), no quant call; non-contiguous x -> quant + v2 A8; AG of [M,224]",
              bool(ok))

        r = scenario(root, {"DSV41_SKINNY_V2": "1", "DSV41_SKINNY_QFUSE": "1", "T_BAD_A8": "[3]",
                            "T_BAD_Q": "[5]", "T_FTZ_MATCH": "[1]"})
        st = r and r["r2_state"]
        ok = bool(st) and st["ok_a8"] == [1, 2, 4, 5, 6, 7, 8] and st["ok_q"] == [1, 2, 3, 4, 6, 7, 8] \
            and st["ftz"] == 1
        for run in (r or {}).get("runs", []):
            M, c = run["M"], [x[0] for x in run["calls"] if x[0] in ("v1", "v2a8", "v2q")]
            if run["contig"]:
                want = ["v2q"] if M != 5 else ["v2a8"]
            else:
                want = ["v1"] if M == 3 else ["v2a8"]
            ok &= c[-1:] == want
        check("self-check: bad A8 M=3 and bad fused M=5 stay live, FTZ mode 1 picked when only it matches",
              bool(ok), json.dumps(st))

        r = scenario(root, {"DSV41_SKINNY_V2": "1", "DSV41_SKINNY_QFUSE": "1", "T_FTZ_MATCH": "[]"})
        st = r and r["r2_state"]
        ok = bool(st) and st["ok_q"] == [] and st["ok_a8"] == list(range(1, 9)) and any(
            "QFUSE kept off" in m for m in r["log"])
        check("self-check: fused quant matching in neither FTZ mode -> QFUSE off, v2 A8 still on", bool(ok))

        bad = subprocess.run([sys.executable, "-c", "import vllm.models.deepseek_v41.nvidia.dsv41_r3_dense"],
                             env=dict(os.environ, PYTHONPATH=root, DSV41_SKINNY_V2="2"), capture_output=True,
                             text=True)
        check("DSV41_SKINNY_V2=2 is rejected at import of dsv41_r3_dense (no silent fallback)",
              bad.returncode != 0 and "must be 1|pdl" in bad.stderr)
        # a broken r2 module must not disable the live shard path (stock replicated GEMM is NOT bitwise == skinny)
        os.rename(os.path.join(nv, "dsv41_r2_skinny.py"), os.path.join(nv, "dsv41_r2_skinny.py.off"))
        r = scenario(root, {"DSV41_SKINNY_V2": "1", "DSV41_SKINNY_V2_CHECK": "0"})
        os.rename(os.path.join(nv, "dsv41_r2_skinny.py.off"), os.path.join(nv, "dsv41_r2_skinny.py"))
        ok = r and all([c[0] for c in run["calls"]] == ["v1"] and run["quant"] == 1 and run["ag"]
                       for run in r["runs"]) and any("disabled (live skinny kept)" in m for m in r["log"])
        check("V2=1 with the r2 module missing: live v1 shard path kept (not the stock GEMM), warning logged",
              bool(ok))
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("\nALL PASS" if not FAIL else "\nFAILURES: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
