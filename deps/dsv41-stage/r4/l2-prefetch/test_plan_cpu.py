#!/usr/bin/env python3
"""CPU-only check of the l2-prefetch window planner (no GPU touched).
  CUDA_VISIBLE_DEVICES= python3 test_plan_cpu.py"""
import os
import sys
import types

os.environ["DSV41_L2_PREFETCH"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import vllm.logger  # noqa: F401
except Exception:  # noqa: BLE001
    import logging
    vl = types.ModuleType("vllm.logger")
    vl.init_logger = logging.getLogger
    sys.modules.setdefault("vllm", types.ModuleType("vllm"))
    sys.modules["vllm.logger"] = vl
import dsv41_l2_prefetch as L  # noqa: E402

MB = 1 << 20
ev = [
    ("c", 1, "ar"),
    ("w", "hc_ffn_fn", [(0x1000000, int(1.97 * MB) & ~15, 24)]),
    ("w", "gate", [(0x2000000, int(3.93 * MB) & ~15, 384)]),
    ("w", "w13", [(0x3000000, 3 * MB, 576)]),
    ("c", 2, "ar"),
    ("w", "hc_attn_fn", [(0x4000000, int(1.97 * MB) & ~15, 24)]),
    ("w", "qkv_a", [(0x5000000, 4096, None), (0x6000000, int(9.46 * MB) & ~15, 1792)]),
    ("c", 3, "ag"),
    ("c", 4, "ar"),
]
p = L.make_plan(ev, 4, budget=6 * MB, device="cpu")
assert p["n"] == 4
assert set(p["win"]) == {1, 2}, p["win"].keys()
t1, n1, b1 = p["win"][1]
assert n1 == 3 and b1 <= 6 * MB and b1 >= 6 * MB - 16, (n1, b1)
segs = t1.view(-1, 2).tolist()
assert segs[0][0] == 0x1000000 and segs[1][0] == 0x2000000 and segs[2][0] == 0x3000000
assert all(s[1] % 16 == 0 for s in segs)
t2, n2, b2 = p["win"][2]
assert n2 == 3 and t2.view(-1, 2).tolist()[1] == [0x5000000, 4096]
# SKIP_N drops small-N weights (hc fn rows=24, gate 384)
p2 = L.make_plan(ev, 4, budget=6 * MB, skip_n=512, device="cpu")
assert p2["win"][1][1] == 1 and p2["win"][1][0].view(-1, 2).tolist()[0][0] == 0x3000000
# identical events -> identical key (no relearn log spam)
assert L.make_plan(ev, 4, budget=6 * MB, device="cpu")["key"] == p["key"]
print("plan:", p["desc"])
print("PASS")
