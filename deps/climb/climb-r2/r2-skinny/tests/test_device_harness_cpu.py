#!/usr/bin/env python3
"""CPU smoke test of device_test.py's input builders and quant section (no GPU): the live quant and the v2
quant_debug are both replaced by the bit-exact numpy emulation (tests/emu.py), so this checks the harness code
paths (exhaustive row construction, unswizzle, v1 deq formula, FTZ pick) run and agree, not the kernels."""
import os
import sys
import types

import numpy as np
import torch

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(KIT, "tests"))
sys.path.insert(0, KIT)
import emu  # noqa: E402

# stub vllm.logger for the kit module import
vl = types.ModuleType("vllm"); vlog = types.ModuleType("vllm.logger")
vlog.init_logger = lambda n: types.SimpleNamespace(info=lambda *a: None, warning=lambda *a: None)
sys.modules["vllm"], sys.modules["vllm.logger"] = vl, vlog
os.environ["DSV41_SKINNY_V2"] = "1"
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("r2", os.path.join(KIT, "new", "dsv41_r2_skinny.py"))
r2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r2)
import device_test as DT  # noqa: E402


def emu_quant(x, ftz):
    b = x.contiguous().view(torch.int16).numpy().astype(np.uint16)
    R, K = b.shape
    codes, ue = emu.fi_quant_blocks(b.reshape(R, K // 32, 32), ftz=ftz)
    return codes.reshape(R, K).astype(np.uint8), ue.reshape(R, K // 32).astype(np.uint8)


SEEN_ROWS = []


def live_quant(x, is_sf_swizzled_layout=True):
    SEEN_ROWS.append(x.shape[0])
    q, s = emu_quant(x, False)
    R, K = q.shape
    nm, nk = (R + 127) // 128, (K // 32 + 3) // 4
    pad = torch.zeros((nm * 128, nk * 4), dtype=torch.uint8)
    pad[:R, : K // 32] = torch.from_numpy(s)
    sw = pad.view(nm, 4, 32, nk, 4).transpose(1, 3).contiguous().view(-1)
    return torch.from_numpy(q).view(torch.float8_e4m3fn), sw


class V2:
    def quant_debug(self, x, ftz):
        q, s = emu_quant(x, bool(ftz))
        d = emu.v1_deq(q.reshape(*q.shape[:-1], -1, 32), s.astype(np.int64)).reshape(q.shape)
        dt = torch.from_numpy(emu.f_to_bf16_bits(d).astype(np.int32)).to(torch.int16).view(torch.bfloat16)
        return torch.from_numpy(q), torch.from_numpy(s), dt


torch.manual_seed(0)
pick = DT.quant_section(V2(), live_quant, r2, "cpu", quick=True)
print("pick", pick, DT.RESULTS["quant"])
# Q must call the live quant only with <= 409 rows (4T/SF variant: rows * 160 < 65536)
print("Q live-quant chunk rows", sorted(set(SEEN_ROWS)))
chunk_ok = max(SEEN_ROWS) <= 409
# ckpt-derived activations from a tiny fake checkpoint (safetensors, same key names as official-fp8)
import json, tempfile  # noqa: E401,E402
from safetensors.torch import save_file  # noqa: E402
md = tempfile.mkdtemp()
emb = (torch.randn(1000, 5120) * 0.02).to(torch.bfloat16)
nw = (torch.rand(5120) * 2).to(torch.bfloat16)
save_file({"embed.weight": emb, "layers.0.attn_norm.weight": nw}, os.path.join(md, "a.safetensors"))
json.dump({"weight_map": {"embed.weight": "a.safetensors", "layers.0.attn_norm.weight": "a.safetensors"}},
          open(os.path.join(md, "model.safetensors.index.json"), "w"))
ca = DT.ckpt_acts(md, "0")
h = emb[100].float()
want0 = (h * torch.rsqrt(h.pow(2).mean() + 1e-6) * nw.float()).to(torch.bfloat16)
ck_ok = ca.shape == (64, 5120) and ca.dtype == torch.bfloat16 and torch.equal(ca[0], want0)
print("ckpt acts", tuple(ca.shape), str(ca.dtype), "row0 ok" if ck_ok else "row0 WRONG")
sets = DT.act_sets(5, 3, "cpu", r2, [torch.randn(8, 5120).to(torch.bfloat16), ("ckpt-L0", ca)])
print("act sets", [(n, tuple(t.shape), str(t.dtype)) for n, t in sets])
w, s = DT.rand_weights(2, True, "cpu")
print("hostile weights", tuple(w.shape), int(((w & 0x7F) == 0x7F).sum()), "NaN codes")
ok = (not DT.FAILS and pick == 0 and len(sets) == 6 and [n for n, _ in sets][-2:] == ["recorded0", "ckpt-L0"]
      and chunk_ok and ck_ok)
print("main run:", "ok" if ok else f"FAILURES {DT.FAILS}")

# ---- r4 additions -------------------------------------------------------------------------------------------
# (a) pad_mask marks exactly the swizzled padding (rows >= M) of the layout live_quant (stub = FlashInfer layout) emits
for M in (1, 3, 8, 130):
    xs_rows = torch.arange(((M + 127) // 128) * 128, dtype=torch.int32).view(-1, 1).expand(-1, 160)
    nm = xs_rows.shape[0] // 128
    sw = xs_rows.contiguous().view(nm, 4, 32, 40, 4).transpose(1, 3).contiguous().view(-1)  # row id per 1-D slot
    pm = DT.pad_mask(M, 5120, "cpu")
    good = pm.numel() == sw.numel() and torch.equal(pm, sw >= M)
    print(f"pad_mask M={M}: {'exact' if good else 'WRONG'} ({int(pm.sum())} padding slots)")
    ok &= good

# (b) negative control: the 2026-11-08 situation. Reference = vLLM's torch fallback (_mxfp8_e4m3_quantize_torch,
#     copied from the live mxfp8_utils.py) -> Q must FAIL in both modes with the logged counts (2 / 8458 scale bytes
#     in the full run; quick here), and the first difference must be reported with (row, block).
import ast  # noqa: E402
live_mxu = os.environ.get("LIVE_MXFP8_UTILS", "")
if live_mxu and os.path.exists(live_mxu):
    tree = ast.parse(open(live_mxu).read())
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("_mxfp8_e4m3_quantize_torch",
                                                                                   "swizzle_mxfp8_scale")]
    ns = {"torch": torch, "MXFP8_BLOCK_SIZE": 32, "MXFP8_VALUE_DTYPE": torch.float8_e4m3fn,
          "MXFP8_SCALE_DTYPE": torch.uint8}
    exec(compile(ast.Module(body=fns, type_ignores=[]), live_mxu, "exec"), ns)
    DT.FAILS.clear(); DT.RESULTS["quant"] = {}
    p2 = DT.quant_section(V2(), ns["_mxfp8_e4m3_quantize_torch"], r2, "cpu", quick=True)
    q = DT.RESULTS["quant"]
    neg = (p2 == -1 and len(DT.FAILS) == 2 and q["exhaustive_match_ftz"] == [] and q["ftz0"]["scales"]["n"] > 0
           and "row" in q["ftz0"]["scales"])
    print(f"negative control (torch-fallback reference): Q {'FAILS as it must' if neg else 'DID NOT FAIL'}; "
          f"ftz0 scales {q['ftz0']['scales']}, ftz1 scales n={q['ftz1']['scales']['n']}")
    ok &= neg
else:
    print("negative control skipped: set LIVE_MXFP8_UTILS=<copy of the live mxfp8_utils.py>")
    ok = False

# (c) negative control: a fused quant that is wrong in ONE scale byte must fail Q with that (row, block) reported
class V2Bad(V2):
    def quant_debug(self, x, ftz):
        q, s, d = super().quant_debug(x, ftz)
        s = s.clone(); s[7, 3] ^= 1
        return q, s, d
DT.FAILS.clear(); DT.RESULTS["quant"] = {}
p3 = DT.quant_section(V2Bad(), live_quant, r2, "cpu", quick=True)
f0 = DT.RESULTS["quant"]["ftz0"]["scales"]
neg2 = p3 == -1 and f0.get("n") == 1 and f0.get("row") == 7 and f0.get("block") == 3
print(f"negative control (one flipped scale byte): {'caught at row 7 block 3' if neg2 else 'NOT CAUGHT'} {f0}")
ok &= neg2
DT.FAILS.clear()
print("ALL PASS" if ok else "FAILURES")
sys.exit(0 if ok else 1)
