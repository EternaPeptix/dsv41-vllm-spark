#!/usr/bin/env python3
"""r6-head compile check WITHOUT a GPU (spark-eight, CUDA_VISIBLE_DEVICES=): Triton -> PTX -> cubin for sm_121 in
production's launch specialization, live kernels and r6 kernels side by side, then SASS census.

    cd ~/climb-scratch/r6-head && CUDA_VISIBLE_DEVICES= TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
        TRITON_CACHE_DIR=$PWD/tcache nice -n 19 ~/venvs/vllm032/bin/python tools/compile_check.py

Specialization (Triton 3.x rules, as in climb-r5/glue/tools/compile_check.py): ints == 1 -> constexpr, ints and
pointers divisible by 16 -> tt.divisibility 16. Production values: x [M, 5120] bf16 (sx 5120), V = VREAL = 16160,
out stride 16160, M = 7 (BM = BMX = 16; M not divisible by 16) and M = 32 (BM = BMX = 32, M divisible by 16).

Kernels: live _xnorm/_screen/_refine (base copy of dsv41_r4_small.py, sha 3364d8c8) vs r6 _r6_xnorm, _r6_screen DQ=1 / DQ=2,
_r6_cand (tile, row), _r6_refine (live cfg BK128/s3, and the BK/stages variants the device test sweeps).

Checks (exit 0 when all pass):
  every kernel compiles; no local-memory spills (STL/LDL);
  refine: the r6 refine's MMA opcode (HMMA.16816.F32.BF16) == the live refine's, and HMMAs per K-loop iteration
          per thread are equal at the live cfg (same instruction, same count per BK);
  screen: the r6 screen's MMA opcode and per-iteration HMMA count == live; the DQ variants have no I2F in the
          K loop and fewer LDG per iteration than live;
INFO: per kernel regs / shared / static opcode census of the HMMA loop body, instructions per weight element, and an
issue-time estimate (warp instructions / (48 SM x 4 issue/clk x clock)) next to the DRAM-time estimate.
"""
import collections
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
import types

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import triton  # noqa: E402
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.compiler import ASTSource  # noqa: E402

TARGET = GPUTarget("cuda", 121, 32)
FAILS = []
OUT = os.path.join(HERE, "results", "sass")
os.makedirs(OUT, exist_ok=True)


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name}{(': ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAILS.append(name)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


r4 = load("vllm.models.deepseek_v41.nvidia.dsv41_r4_small",
          os.path.join(HERE, "base/vllm/models/deepseek_v41/nvidia/dsv41_r4_small.py"))
r6 = load("vllm.models.deepseek_v41.nvidia.dsv41_head_r6",
          os.path.join(HERE, "files/vllm/models/deepseek_v41/nvidia/dsv41_head_r6.py"))
K4 = r4._kernels()
K6 = r6._kernels6()

_DT = {torch.bfloat16: "bf16", torch.float32: "fp32", torch.int32: "i32", torch.int64: "i64", torch.uint8: "u8",
       torch.int8: "i8"}


class P:  # pointer stub: dtype + 16-byte aligned address
    def __init__(self, dtype):
        self.dtype = dtype

    def data_ptr(self):
        return 1 << 20


def spec(fn, vals):
    names = [p.name for p in fn.params]
    assert set(vals) == set(names), (fn.__name__, set(names) ^ set(vals))
    sig, const, attrs = {}, {}, {}
    for i, p in enumerate(fn.params):
        v = vals[p.name]
        if p.is_constexpr:
            sig[p.name], const[(i,)] = "constexpr", v
        elif isinstance(v, bool):
            sig[p.name] = "u1"
        elif isinstance(v, int):
            if not p.do_not_specialize and v == 1:
                sig[p.name], const[(i,)] = "constexpr", 1
                continue
            sig[p.name] = "i32"
            if not p.do_not_specialize and v % 16 == 0:
                attrs[(i,)] = [["tt.divisibility", 16]]
        elif isinstance(v, float):
            sig[p.name] = "fp32"
        else:
            sig[p.name] = "*" + _DT[v.dtype]
            attrs[(i,)] = [["tt.divisibility", 16]]
    return sig, const, attrs


def cuobjdump(cubin, flag):
    d = tempfile.mkdtemp()
    p = os.path.join(d, "k.cubin")
    open(p, "wb").write(cubin)
    try:
        r = subprocess.run(["/usr/local/cuda/bin/cuobjdump", flag, p], capture_output=True, text=True, timeout=120)
        return r.stdout + r.stderr
    finally:
        shutil.rmtree(d, ignore_errors=True)


LINE = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)([^;]*);")


def parse_sass(s):
    ins = []
    for m in LINE.finditer(s):
        ins.append((int(m.group(1), 16), m.group(3), m.group(4)))
    return ins


def loops(ins):
    """[(start, end)] for every backward branch (BRA to a lower address)."""
    out = []
    for addr, op, rest in ins:
        if op.startswith("BRA"):
            t = re.search(r"0x([0-9a-f]+)", rest)
            if t and int(t.group(1), 16) < addr:
                out.append((int(t.group(1), 16), addr))
    return out


def hmma_loop(ins):
    """The innermost backward-branch loop that contains HMMA (the K loop)."""
    best = None
    for a, b in loops(ins):
        body = [x for x in ins if a <= x[0] <= b]
        if any(x[1].startswith("HMMA") for x in body):
            if best is None or (b - a) < (best[1] - best[0]):
                best = (a, b)
    if best is None:
        return []
    return [x for x in ins if best[0] <= x[0] <= best[1]]


def census(body):
    c = collections.Counter()
    for _, op, _ in body:
        c[op.split(".")[0]] += 1
    return c


def res_usage(txt):
    m = re.search(r"REG:(\d+).*?SHARED:(\d+).*?LOCAL:(\d+)", txt, re.S)
    return dict(reg=int(m.group(1)), shared=int(m.group(2)), local=int(m.group(3))) if m else None


def compile_one(name, fn, vals, num_warps, num_stages):
    sig, const, attrs = spec(fn, vals)
    k = triton.compile(ASTSource(fn=fn, signature=sig, constexprs=const, attrs=attrs), target=TARGET,
                       options={"num_warps": num_warps, "num_stages": num_stages})
    cub = k.asm.get("cubin", b"")
    sass = cuobjdump(cub, "-sass")
    open(os.path.join(OUT, f"{name}.sass"), "w").write(sass)
    open(os.path.join(OUT, f"{name}.ptx"), "w").write(k.asm["ptx"])
    ru = res_usage(cuobjdump(cub, "-res-usage"))
    ins = parse_sass(sass)
    body = hmma_loop(ins)
    spills = sum(1 for _, op, _ in ins if op.startswith("STL") or op.startswith("LDL"))
    return dict(k=k, ins=ins, body=body, ru=ru, shared=k.metadata.shared, spills=spills,
                hmma_ops=sorted({op for _, op, _ in ins if op.startswith("HMMA")}))


M_CASES = (7, 32)
V, K = 16160, 5120
res = {}
for M in M_CASES:
    BM = max(16, r4.next_pow2(M))
    x = dict(X=P(torch.bfloat16), sx=K)
    common_screen = dict(**x, Q=P(torch.int8), E=P(torch.uint8), RT=P(torch.float32), NV=P(torch.float32),
                         XG=P(torch.float32), XN=P(torch.float32), HI=P(torch.bfloat16), THR=P(torch.float32),
                         M=M, V=V, VREAL=V, K=K, G=128, BM=BM, BN=r4.SBN, BK=r4.SBK, CACC2=2.0 * r4.c_acc(K),
                         INFL_Q=r4.INFL_Q, INFL_B=r4.INFL_B, SLOP_REL=r4.SLOP_REL, SLOP_ABS=r4.SLOP_ABS)
    sw = r4.SWARPS if BM <= 32 else max(r4.SWARPS, 8)
    jobs = {
        f"live_screen_M{M}": (K4["screen"], common_screen, sw, r4.SSTAGES),
        f"r6_screen_dq1_M{M}": (K6["screen"], dict(common_screen, DQ=1), sw, r4.SSTAGES),
        f"r6_screen_dq2_M{M}": (K6["screen"], dict(common_screen, DQ=2), sw, r4.SSTAGES),
        f"live_refine_M{M}": (K4["refine"], dict(**x, W=P(torch.bfloat16), HI=P(torch.bfloat16),
                                                 THR=P(torch.float32), BAD=P(torch.int32), OUT=P(torch.bfloat16),
                                                 so=V, NEED=P(torch.int32), M=M, V=V, K=K, BV=r4.XBV, BMX=BM,
                                                 BK=r4.XBK, FULL=False), r4.XWARPS, r4.XSTAGES),
        f"r6_cand_tile_M{M}": (K6["cand"], dict(HI=P(torch.bfloat16), THR=P(torch.float32), BAD=P(torch.int32),
                                                OUT=P(torch.bfloat16), so=V, IDX=P(torch.int32), CNT=P(torch.int32),
                                                NEED=P(torch.int32), STATS=P(torch.int64), M=M, V=V, BMX=BM,
                                                CB=r6.cand_rows(BM, 16), TB=16, TILE=True), 4, 3),
        f"r6_cand_row_M{M}": (K6["cand"], dict(HI=P(torch.bfloat16), THR=P(torch.float32), BAD=P(torch.int32),
                                               OUT=P(torch.bfloat16), so=V, IDX=P(torch.int32), CNT=P(torch.int32),
                                               NEED=P(torch.int32), STATS=P(torch.int64), M=M, V=V, BMX=BM,
                                               CB=r6.cand_rows(BM, 16), TB=16, TILE=False), 4, 3),
    }
    for bk, stg in ((128, 3), (128, 4), (256, 3), (256, 4), (512, 2)) if BM == 16 else ((128, 3),):
        jobs[f"r6_refine_bk{bk}_s{stg}_M{M}"] = (
            K6["refine"], dict(**x, W=P(torch.bfloat16), OUT=P(torch.bfloat16), so=V, IDX=P(torch.int32),
                               CNT=P(torch.int32), M=M, NPROG=192, K=K, BV=16, BMX=BM, BK=bk, SPREAD=True),
            r4.XWARPS, stg)
    if M == M_CASES[0]:
        jobs["live_xnorm"] = (K4["xnorm"], dict(**x, XG=P(torch.float32), XN=P(torch.float32), BAD=P(torch.int32),
                                                THR=P(torch.float32), K=K, G=128, NGP=64, INFL=r4.INFL_X,
                                                XABS=r4.X_ABS), 4, 3)
        jobs["r6_xnorm"] = (K6["xnorm"], dict(**x, XG=P(torch.float32), XN=P(torch.float32), BAD=P(torch.int32),
                                              THR=P(torch.float32), CNT=P(torch.int32), K=K, G=128, NGP=64,
                                              INFL=r4.INFL_X, XABS=r4.X_ABS), 4, 3)
    for name, (fn, vals, nw, ns) in jobs.items():
        try:
            r = compile_one(name, fn, vals, nw, ns)
        except Exception as e:  # noqa: BLE001
            check(f"compile {name}", False, repr(e)[:600])
            continue
        res[name] = r
        c = census(r["body"])
        check(f"compile {name}", True, f"regs {r['ru']}, shared {r['shared']} B, warps {nw}, stages {ns}, "
              f"spills {r['spills']}, HMMA ops {r['hmma_ops']}, K-loop body {len(r['body'])} instr")
        check(f"no spills {name}", r["spills"] == 0, str(r["spills"]))
        if r["body"]:
            top = ", ".join(f"{op} {n}" for op, n in c.most_common(16))
            print(f"INFO {name} K-loop census: {top}")

# ---- device_test.py mutants must compile (a mutant that fails to build would be "caught" for the wrong reason)
sys.path.insert(0, HERE)
import device_test as _dt  # noqa: E402  (module level is stdlib only)

R6SRC = os.path.join(HERE, "files/vllm/models/deepseek_v41/nvidia/dsv41_head_r6.py")
for tag, (old, new) in _dt.MUTANTS.items():
    src = open(R6SRC).read()
    check(f"mutant {tag}: anchor applies exactly once", src.count(old) == 1)
    d = tempfile.mkdtemp(prefix="r6mut-")
    mp = os.path.join(d, f"r6mut_{tag}.py")
    open(mp, "w").write(src.replace(old, new))
    mm = load(f"r6mut_{tag}", mp)
    KM = mm._kernels6()
    M = 7
    BM = 16
    if tag in ("kreverse", "ksplit"):
        fn, vals, nw, ns = KM["refine"], dict(X=P(torch.bfloat16), sx=K, W=P(torch.bfloat16), OUT=P(torch.bfloat16),
                                              so=V, IDX=P(torch.int32), CNT=P(torch.int32), M=M, NPROG=192, K=K, BV=16,
                                              BMX=BM, BK=128, SPREAD=True), 4, 3
    elif tag == "egroup":
        fn, vals, nw, ns = KM["screen"], dict(X=P(torch.bfloat16), sx=K, Q=P(torch.int8), E=P(torch.uint8),
                                              RT=P(torch.float32), NV=P(torch.float32), XG=P(torch.float32),
                                              XN=P(torch.float32), HI=P(torch.bfloat16), THR=P(torch.float32), M=M,
                                              V=V, VREAL=V, K=K, G=128, BM=BM, BN=64, BK=128,
                                              CACC2=2.0 * r4.c_acc(K), INFL_Q=r4.INFL_Q, INFL_B=r4.INFL_B,
                                              SLOP_REL=r4.SLOP_REL, SLOP_ABS=r4.SLOP_ABS, DQ=1), 4, 3
    else:
        fn, vals, nw, ns = KM["cand"], dict(HI=P(torch.bfloat16), THR=P(torch.float32), BAD=P(torch.int32),
                                            OUT=P(torch.bfloat16), so=V, IDX=P(torch.int32), CNT=P(torch.int32),
                                            NEED=P(torch.int32), STATS=P(torch.int64), M=M, V=V, BMX=BM, CB=256,
                                            TB=16, TILE=(tag == "tile8")), 4, 3
    try:
        r = compile_one(f"mutant_{tag}", fn, vals, nw, ns)
        check(f"mutant {tag} compiles", True, f"HMMA ops {r['hmma_ops']}")
    except Exception as e:  # noqa: BLE001
        check(f"mutant {tag} compiles", False, repr(e)[:400])
    shutil.rmtree(d, ignore_errors=True)

# ---- comparisons
for M in M_CASES:
    BM = max(16, r4.next_pow2(M))
    lv = res.get(f"live_refine_M{M}")
    for name in [n for n in res if n.startswith("r6_refine") and n.endswith(f"_M{M}")]:
        r = res[name]
        if lv is None:
            continue
        check(f"{name}: MMA opcode == live refine", r["hmma_ops"] == lv["hmma_ops"], f"{r['hmma_ops']} vs "
              f"{lv['hmma_ops']}")
        bk = int(re.search(r"bk(\d+)", name).group(1))
        a = census(r["body"])["HMMA"] / (bk / r4.XBK)
        b = census(lv["body"])["HMMA"]
        if "_s3_" in name and bk == r4.XBK:
            check(f"{name}: HMMA per K-loop iteration == live ({a:.0f} vs {b})", a == b)
        else:
            print(f"INFO {name}: HMMA per {r4.XBK}-wide K step {a:.1f} (live {b}); body may be unrolled/pipelined")
    ls = res.get(f"live_screen_M{M}")
    for dq in (1, 2):
        r = res.get(f"r6_screen_dq{dq}_M{M}")
        if r is None or ls is None:
            continue
        cl, cr = census(ls["body"]), census(r["body"])
        check(f"r6_screen_dq{dq}_M{M}: MMA opcode == live", r["hmma_ops"] == ls["hmma_ops"])
        check(f"r6_screen_dq{dq}_M{M}: HMMA per iteration == live ({cr['HMMA']} vs {cl['HMMA']})",
              cr["HMMA"] == cl["HMMA"])
        check(f"r6_screen_dq{dq}_M{M}: no I2F in the K loop (live has {cl['I2F']})", cr["I2F"] == 0)
        check(f"r6_screen_dq{dq}_M{M}: fewer LDG per iteration than live ({cr['LDG']} vs {cl['LDG']})",
              cr["LDG"] < cl["LDG"])
        print(f"INFO r6_screen_dq{dq}_M{M}: K-loop instructions {len(r['body'])} vs live {len(ls['body'])}")

# ---- issue-time model for the screen (INFO)
for name in [n for n in res if "screen" in n]:
    body = res[name]["body"]
    if not body:
        continue
    M = int(name.split("_M")[1])
    BM = max(16, r4.next_pow2(M))
    warps = r4.SWARPS if BM <= 32 else max(r4.SWARPS, 8)
    ctas = (V + r4.SBN - 1) // r4.SBN
    iters = K // r4.SBK
    winstr = len(body) * warps * iters * ctas
    elem = r4.SBN * r4.SBK / (warps * 32)
    for clk in (1.9e9, 2.4e9):
        us = winstr / (48 * 4 * clk) * 1e6
        print(f"INFO {name}: {len(body)} instr/iter/thread = {len(body) / elem:.2f} per weight element; "
              f"issue-only floor {us:.0f} us at {clk / 1e9:.1f} GHz (48 SM x 4 issue/clk); DRAM floor for 88 MB: "
              f"{88e6 / 230e9 * 1e6:.0f} us @230 GB/s, {88e6 / 200e9 * 1e6:.0f} us @200 GB/s")

print(f"\n{'ALL PASS' if not FAILS else 'FAILED: ' + str(FAILS)}")
sys.exit(1 if FAILS else 0)
