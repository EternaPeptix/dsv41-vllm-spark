#!/usr/bin/env python3
"""r8-fixb head: every device-test mutant must FIT (shared memory) at every config the device test runs it at,
compiled for sm_121 WITHOUT a GPU (spark-eight, CUDA_VISIBLE_DEVICES=, nice 19, no CLUSTER_LOCK):

    cd ~/climb-scratch/r8-fixb/head && CUDA_VISIBLE_DEVICES= TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
        TRITON_CACHE_DIR=$PWD/tcache nice -n 19 ~/venvs/vllm032/bin/python tools/mutfit.py

For every (mutant, config, BMX/BM bucket) the device test can run, compile the UNMUTATED r6 kernel and the mutant with
the same specialization and launch options and compare metadata.shared to the GB10 limit (101376 B, the value Triton
reported in r6-dwA). Gates (exit 0 when all hold):
  - every mutant compiles;
  - wherever the unmutated kernel fits, the mutant fits;
  - ksplit uses EXACTLY the unmutated refine's shared memory (the point of the two-sequential-loops rewrite).
Negative control (must reproduce the r6-dwA failure): the round-6 ksplit text (two interleaved tile loads per iteration)
at the live refine cfg, BMX 128, needs more than 101376 B.
Buckets: refine BMX 16 (every BK x stages of the sweep), 32 / 64 / 128 (live BK128 / s3: refine_cfg uses the live tile
above BMX 16); screen BM 16 / 32 / 64 / 128 x every (BN, BK, warps, stages) of the sweep (warps = max(w, 8) at BM > 32,
as the module launches it); cand mutants at every BMX with CB = cand_rows(BMX, 16).
"""
import importlib.util
import os
import sys
import tempfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import torch  # noqa: E402
import triton  # noqa: E402
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.compiler import ASTSource  # noqa: E402

TARGET = GPUTarget("cuda", 121, 32)
LIMIT = 101376
V, K = 16160, 5120
FAILS = []


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


class P:
    def __init__(self, dtype):
        self.dtype = dtype

    def data_ptr(self):
        return 1 << 20


_DT = {torch.bfloat16: "bf16", torch.float32: "fp32", torch.int32: "i32", torch.int64: "i64", torch.uint8: "u8",
       torch.int8: "i8"}


def spec(fn, vals):
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


_CACHE = {}


def shared(fn, vals, nw, ns):
    key = (id(fn), tuple(sorted((k, str(v.dtype) if isinstance(v, P) else v) for k, v in vals.items())), nw, ns)
    if key not in _CACHE:
        sig, const, attrs = spec(fn, vals)
        k = triton.compile(ASTSource(fn=fn, signature=sig, constexprs=const, attrs=attrs), target=TARGET,
                           options={"num_warps": nw, "num_stages": ns})
        _CACHE[key] = int(k.metadata.shared)
    return _CACHE[key]


r4 = load("vllm.models.deepseek_v41.nvidia.dsv41_r4_small",
          os.path.join(HERE, "base/vllm/models/deepseek_v41/nvidia/dsv41_r4_small.py"))
R6SRC = os.path.join(HERE, "files/vllm/models/deepseek_v41/nvidia/dsv41_head_r6.py")
r6 = load("vllm.models.deepseek_v41.nvidia.dsv41_head_r6", R6SRC)
import device_test as DT  # noqa: E402  (module level is stdlib only)


def mutant_kernels(tag, old_new=None):
    old, new = old_new or DT.MUTANTS[tag]
    src = open(R6SRC).read()
    assert src.count(old) == 1, tag
    d = tempfile.mkdtemp(prefix="r8mut-")
    p = os.path.join(d, f"r8mut_{tag}.py")
    open(p, "w").write(src.replace(old, new))
    return load(f"r8mut_{tag}_{id(old_new)}", p)._kernels6()


K6 = r6._kernels6()
BUCKETS = (16, 32, 64, 128)  # M 2/7 -> 16, 24 -> 32, 48 -> 64, 96 -> 128


def refine_vals(BMX, bk, M):
    return dict(X=P(torch.bfloat16), sx=K, W=P(torch.bfloat16), OUT=P(torch.bfloat16), so=V, IDX=P(torch.int32),
                CNT=P(torch.int32), M=M, NPROG=192, K=K, BV=16, BMX=BMX, BK=bk, SPREAD=True)


def screen_vals(BM, bn, bk, M, dq):
    return dict(X=P(torch.bfloat16), sx=K, Q=P(torch.int8), E=P(torch.uint8), RT=P(torch.float32),
                NV=P(torch.float32), XG=P(torch.float32), XN=P(torch.float32), HI=P(torch.bfloat16),
                THR=P(torch.float32), M=M, V=V, VREAL=V, K=K, G=128, BM=BM, BN=bn, BK=bk, CACC2=2.0 * r4.c_acc(K),
                INFL_Q=r4.INFL_Q, INFL_B=r4.INFL_B, SLOP_REL=r4.SLOP_REL, SLOP_ABS=r4.SLOP_ABS, DQ=dq)


def cand_vals(BMX, M, tile):
    return dict(HI=P(torch.bfloat16), THR=P(torch.float32), BAD=P(torch.int32), OUT=P(torch.bfloat16), so=V,
                IDX=P(torch.int32), CNT=P(torch.int32), NEED=P(torch.int32), STATS=P(torch.int64), M=M, V=V, BMX=BMX,
                CB=r6.cand_rows(BMX, 16), TB=16, TILE=tile)


M_OF = {16: 7, 32: 24, 64: 48, 128: 96}
rows = []
# ---- refine mutants
for tag in ("kreverse", "ksplit"):
    KM = mutant_kernels(tag)
    for BMX in BUCKETS:
        cfgs = [(bk, st) for bk in (128, 256, 512) for st in (2, 3, 4)] if BMX == 16 else [(r4.XBK, r4.XSTAGES)]
        for bk, st in cfgs:
            v = refine_vals(BMX, bk, M_OF[BMX])
            try:
                base = shared(K6["refine"], v, r4.XWARPS, st)
            except Exception as e:  # noqa: BLE001
                check(f"r6 refine compiles BMX{BMX} BK{bk} s{st}", False, repr(e)[:200])
                continue
            try:
                mut = shared(KM["refine"], v, r4.XWARPS, st)
            except Exception as e:  # noqa: BLE001
                check(f"mutant {tag} compiles BMX{BMX} BK{bk} s{st}", False, repr(e)[:200])
                continue
            rows.append((tag, f"BMX{BMX} BK{bk} s{st}", base, mut))
# ---- screen mutant
KM = mutant_kernels("egroup")
for BM in BUCKETS:
    for bn in (32, 64, 128):
        for bk in (128, 256):
            for w in (4, 8):
                for st in (2, 3, 4):
                    nw = w if BM <= 32 else max(w, 8)
                    v = screen_vals(BM, bn, bk, M_OF[BM], 2)
                    base = shared(K6["screen"], v, nw, st)
                    mut = shared(KM["screen"], v, nw, st)
                    rows.append(("egroup", f"BM{BM} BN{bn} BK{bk} w{nw} s{st}", base, mut))
    # live screen shape, DQ1 (the mutant part runs egroup with DQ1 at the live cfg)
    v = screen_vals(BM, r4.SBN, r4.SBK, M_OF[BM], 1)
    nw = r4.SWARPS if BM <= 32 else max(r4.SWARPS, 8)
    rows.append(("egroup", f"BM{BM} live DQ1", shared(K6["screen"], v, nw, r4.SSTAGES), shared(KM["screen"], v, nw,
                                                                                                r4.SSTAGES)))
# ---- cand mutants
for tag in ("strict", "dropfirst", "tile8"):
    KM = mutant_kernels(tag)
    for BMX in BUCKETS:
        for tile in ((True,) if tag == "tile8" else (False, True)):
            v = cand_vals(BMX, M_OF[BMX], tile)
            rows.append((tag, f"BMX{BMX} {'tile' if tile else 'row'}", shared(K6["cand"], v, 4, 3),
                         shared(KM["cand"], v, 4, 3)))

nofit, grow = [], []
for tag, cfg, base, mut in rows:
    fits_base, fits_mut = base <= LIMIT, mut <= LIMIT
    print(f"INFO {tag:9s} {cfg:28s} r6 {base:6d} B  mutant {mut:6d} B  {'' if fits_base else '(r6 itself > limit)'}")
    if fits_base and not fits_mut:
        nofit.append((tag, cfg, base, mut))
    if tag == "ksplit" and mut != base:
        grow.append((cfg, base, mut))
check(f"every mutant compiles and FITS wherever its unmutated kernel fits ({len(rows)} (mutant, cfg) pairs)",
      not nofit, str(nofit[:6]))
check("ksplit (two sequential K-half loops) uses exactly the unmutated refine's shared memory at every cfg",
      not grow and any(r[0] == "ksplit" for r in rows), str(grow[:6]))
# ---- negative control: the round-6 ksplit text
r6_ksplit_new = DT.MUTANTS["ksplit"][1]
old_r6 = ("            acc = tl.zeros((BV, BMX), dtype=tl.float32)\n            acc2 = tl.zeros((BV, BMX), "
          "dtype=tl.float32)\n            for k0 in range(0, K, 2 * BK):\n                k = k0 + tl.arange(0, BK)\n"
          "                w = tl.load(W + j64[:, None] * K + k[None, :], mask=rok[:, None], other=0.0)\n"
          "                xt = tl.load(X + mm[None, :] * sx + k[:, None], mask=mok[None, :], other=0.0)\n"
          "                acc = tl.dot(w, xt, acc)\n                k = k + BK\n"
          "                w = tl.load(W + j64[:, None] * K + k[None, :], mask=rok[:, None], other=0.0)\n"
          "                xt = tl.load(X + mm[None, :] * sx + k[:, None], mask=mok[None, :], other=0.0)\n"
          "                acc2 = tl.dot(w, xt, acc2)\n            acc = acc + acc2\n")
KO = mutant_kernels("ksplit_r6", (DT.MUTANTS["ksplit"][0], old_r6))
s_old = shared(KO["refine"], refine_vals(128, r4.XBK, 96), r4.XWARPS, r4.XSTAGES)
s_base = shared(K6["refine"], refine_vals(128, r4.XBK, 96), r4.XWARPS, r4.XSTAGES)
check(f"negative control: the round-6 ksplit at the live cfg BMX 128 does NOT fit ({s_old} B > {LIMIT}; r6 refine "
      f"{s_base} B; r6-dwA reported 147456)", s_old > LIMIT)
print("ALL PASS" if not FAILS else f"FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
