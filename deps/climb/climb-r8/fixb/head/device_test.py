#!/usr/bin/env python3
"""r6-head DEVICE test: live certified-head kernels vs the r6 kernels, side by side on one GB10, bitwise.

Run ONLY on a stopped node (serving down, no GPU process) inside a device window, in the mandatory environment:

    source ~/venvs/vllm032/bin/activate; export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
    cd ~/climb-scratch/r6-head && python device_test.py --part all [--weights real --rank 4] [--source kit|installed]

Parts (default all): env, anchor, bitwise, ties, mutants, graph, dispatch, time, l2.  Runtime ~25-35 min (Triton
compiles ~200 variants, most of them in the config sweeps).
Exit 0 only when: the environment assertions hold, every anchor / bitwise / tie / graph / dispatch check passes at every
gated M, every must-fail mutant is caught at every gated M, and the forced-set bitwise checks of the time part pass.
The timings and the L2 micro-bench themselves never gate (INFO + a JSON summary).

env      asserts the venv (sys.prefix == ~/venvs/vllm032), CUDA_HOME=/usr/local/cuda, ptxas from /usr/local/cuda/bin,
         a GB10 (sm_121), Triton 3.7.1, no other compute process on the GPU (unless --allow-busy), and that the module
         under test is the SERVING one: vllm.models.deepseek_v41.nvidia.dsv41_r4_small imported from the venv with
         sha256 3364d8c8 (live) or c77844fb (r6-head installed), whose _kernels() returns the Triton JIT kernels
         _xnorm_kernel / _screen_kernel / _refine_kernel (not a torch fallback), and DSV41_R4_CERT_HEAD semantics:
         CERT_GROUP 128, screen tile 64,128,4,3, exact tile 16,128,4,3 (the launcher's defaults).
anchor   ties the reference to the serving entry point: with the r6 flags off, r4.certified_logits(st, x) == the
         r6._run(st, x, False, "0") copy of the live launch sequence, bitwise (out, HI, THR, BAD), at every M in 2..96
         x 4 cases. Must-fail: the same compare against full=True must report a difference at every M.
bitwise  every M in 2..96 (DSV41_R4_CERT_HEAD_M=2-96), x cases {flat, peaked, near-tie cluster, hostile}, on the head
         shard (synthetic N(0, 0.02) or --weights real: rows [rank*16160, (rank+1)*16160) of head.weight):
           screen DQ=1 and DQ=2: HI and THR bit-identical to the live screen;
           refine tile: whole output bit-identical to the live refine (-inf included);
           refine row:  output == where(row predicate, live, -inf) bit for bit, list length == predicate count;
           per-logit stock check (INFO): computed logits vs F.linear (cuBLAS) bits, live and r6 side by side;
         plus config sweeps at one M per served bucket, M in {2, 7, 24, 48, 96} (BM / BMX 16, 16, 32, 64, 128: every
         bucket up to the module's SWEPT_MAX_B; BM 64 is where the screen's num_warps becomes max(w, 8)):
           screen BN/BK/warps/stages with DQ2 (the module refuses a non-default SCREEN_CFG without DQ2);
           refine BK/stages (act at BMX 16 only) x grid x spread (act at every BMX), tile and row, warps 0 (the
           module refuses non-zero refine warps).
         A config is ELIGIBLE only if bit-identical at every swept M and case; each prints as a ready-to-paste
         `ELIGIBLE-ENV` line.
ties     forced ties: duplicated head rows (a < b) across tiles, inside one tile, across a cand-CTA boundary
         (CB=256), across the sf_local_argmax 4096 block boundary, at the shard end, all rows identical; x aligned
         with row a. The r6 chain (xnorm, screen, cand, refine, live _sf_local_argmax_kernel) must return a == the
         stock leftmost argmax of F.linear, at every gated M, tile and row mode.
mutants  textual mutants of the kit module (each asserted to apply once), each must be caught at EVERY gated M:
           kreverse  refine K loop in reverse order            (per-logit bitwise)
           ksplit    refine as two sequential K-half loops + a final add (per-logit bitwise; same smem as r6)
           strict    cand predicate h > T instead of h >= T     (exact-T construction: T := max hi per row)
           dropfirst first list entry of each cand CTA replaced by its left neighbour
           egroup    screen exponent of the next 32-group       (HI bitwise)
           tile8     tile mode with 8-row tiles                  (tile output bitwise)
graph    the r6 pass captured in a CUDA graph, replayed on 3 different inputs written into the static x: each replay
         == the eager r6 pass on that input, bitwise (list counter re-zeroed per replay; no host sync in capture).
dispatch the INSTALLED path on the GPU, one child process per mode (row, tile) with DSV41_R6_HEAD_REFINE=<mode>
         DSV41_R6_HEAD_SCREEN=1 SELFCHECK=1 STATS_EVERY=3. The patched dsv41_r4_small (the venv's with --source
         installed; otherwise the live file patched in a temp dir by install.py's own edit, sha-checked == c77844fb)
         must dispatch certified_logits to dsv41_head_r6.certified_logits. At M in {2, 7, 24, 33, 48, 96} x 4 cases:
         result == live (tile) or where(pred, live, -inf) (row) bitwise, sf_local_argmax pairs == live, the
         self-check passes and logs at the first call per M, after_screen runs once, debug is filled, full=True stays
         live, the stats counters and log line appear. Must-fail mutants (a flipped r6 output bit, a flipped HI bit)
         must each be caught by the serving self-check at every M: live result served, r6 off, one FAILED line.
time     median of 200 runs, warm and cold L2 (a 256 MB scrub between runs): live vs r6 screen (default and every
         eligible sweep config), live vs r6 refine at forced candidate sets (n rows in {1, 16, 64, 256, 1024, 4096,
         16160}; scattered = one row per tile, clustered = whole tiles), and the full chain on the x cases. The times
         never gate; the forced-set bitwise checks run before timing (tile == live, row == where(sel, live, -inf))
         DO gate (stricter than INFO, on purpose).
l2       cert-head L2 pre-stage micro-bench (lever 3, not built): screen time cold vs after an evict_last pre-read of
         the first P MB of the MXINT8 table (P in 0, 2, 4, 6, 8, 12), and the pre-read's own time.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time

KIT = os.path.dirname(os.path.abspath(__file__))
LIVE_SHA = "3364d8c877db7e532923fe03d6da5fa2763867d06d18f575c4c8835556faae8e"
# r8-fixb: the serving dsv41_r4_small.py may also be a COMPOSED state (climb-r6 install-all.sh / climb-r7 integrate):
#   c77844fb = r6-head on live; ebe73cdb = r6-head -> r6-step (7 + 13 tagged lines; identical with and without r5, and
#   no r7 kit touches this file). Both carry the head dispatch lines and are used as installed.
HEAD_SHA = "c77844fb3188f261236b2f9f1cea4688933355b190878c9ccadfc2a61df8bb09"
COMPOSED_HEAD_STEP_SHA = "ebe73cdb57f06670370fb539ce77a836eadce74f51da29c1a15767086990247c"
STEP_ONLY_SHA = "d9bf5d596c261771379d1ce5c8fb505cd3b5edff19540f7d8ccf1973fa84"  # prefix (r6-step alone): no head lines
INSTALLED_WITH_HEAD = {HEAD_SHA: "r6-head", COMPOSED_HEAD_STEP_SHA: "r6-head -> r6-step (composed)"}
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
R4NAME = "vllm.models.deepseek_v41.nvidia.dsv41_r4_small"
R6NAME = "vllm.models.deepseek_v41.nvidia.dsv41_head_r6"
R6SRC = os.path.join(KIT, "files/vllm/models/deepseek_v41/nvidia/dsv41_head_r6.py")
V_SHARD, K_HID = 16160, 5120
DEV = "cuda"
SMOKE = False  # --smoke: CPU + Triton interpreter, small shapes, harness code paths only (NOT a gate)
FAILS, INFO = [], {}
GATED_M = frozenset(range(2, 97))
# config sweeps: one M per served bucket. BM / BMX = max(16, next_pow2(M)): 2, 7 -> 16; 24 -> 32; 48 -> 64; 96 -> 128.
SWEEP_M = [2, 7, 24, 48, 96]
REFINE_GRIDS = (48, 96, 192, 384)


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name}{(': ' + str(detail)) if detail else ''}", flush=True)
    if not ok:
        FAILS.append(name)
    return ok


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


# ---------------------------------------------------------------------------------------------------- environment
def env_checks(args):
    import prodenv

    venv = os.path.expanduser("~/venvs/vllm032")
    ok = True
    for name, good, det in prodenv.env_rule():
        ok &= check(name, good, det)
    _, mch = prodenv.mirror_launcher(args.launcher, args.expect)
    for name, good, det in mch:
        ok &= check(name, good, det)
    if os.environ.get("R6_DT_PARENT") is None:  # parent only: the must-fail controls of the rule
        for name, caught, det in prodenv.selftest_mutants(args.launcher, args.expect):
            ok &= check(f"MUTANT {name}", caught, det)
    import shutil

    w = shutil.which("ptxas") or ""
    ok &= check("env: ptxas on PATH is /usr/local/cuda/bin/ptxas", os.path.realpath(w).startswith(
        os.path.realpath("/usr/local/cuda")), w or "none")
    import torch
    import triton

    ok &= check("env: CUDA available", torch.cuda.is_available())
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name()
    ok &= check("env: GB10 sm_121", cap == (12, 1) and "GB10" in name, f"{name} {cap}")
    ok &= check("env: Triton 3.7.1 (production)", triton.__version__ == "3.7.1", triton.__version__)
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()
    mine = {str(os.getpid()), os.environ.get("R6_DT_PARENT", "") or str(os.getpid())}  # dispatch child: parent too
    others = [ln for ln in apps.splitlines() if ln.strip() and ln.split(",")[0].strip() not in mine]
    ok &= check("env: no other compute process on the GPU (stopped node)", not others or args.allow_busy,
                "; ".join(others)[:200] or "none")
    import importlib

    r4 = importlib.import_module(R4NAME)
    f = os.path.realpath(r4.__file__)
    ok &= check("serving op: dsv41_r4_small imported from the venv", f.startswith(os.path.realpath(venv)), f)
    kit = json.load(open(os.path.join(KIT, "KIT.json")))
    patched = [x.get("patched_sha256") for x in kit["files"] if x["action"] == "patch"][0]
    s = sha(f)
    accepted = {LIVE_SHA: "live", patched: "r6-head", **INSTALLED_WITH_HEAD}
    ok &= check("serving op: dsv41_r4_small is live 3364d8c8, r6-head c77844fb or composed head->step ebe73cdb",
                s in accepted, f"{s[:16]} ({accepted.get(s, 'UNKNOWN: step-only d9bf5d59 has no head lines; install r6-head first' if s.startswith(STEP_ONLY_SHA) else 'UNKNOWN')})")
    k = r4._kernels()
    names = {n: getattr(k[n], "fn", None).__name__ if hasattr(k[n], "fn") else None for n in ("xnorm", "screen",
                                                                                             "refine")}
    ok &= check("serving op: live kernels are Triton JIT _xnorm/_screen/_refine",
                names == {"xnorm": "_xnorm_kernel", "screen": "_screen_kernel", "refine": "_refine_kernel"}, names)
    ok &= check("serving op: launcher tile defaults (64,128,4,3 / 16,128,4,3, group 128)",
                (r4.SBN, r4.SBK, r4.SWARPS, r4.SSTAGES, r4.XBV, r4.XBK, r4.XWARPS, r4.XSTAGES, r4.CERT_GROUP)
                == (64, 128, 4, 3, 16, 128, 4, 3, 128))
    INFO["env"] = dict(name=name, triton=triton.__version__, torch=torch.__version__, r4=s[:16],
                       l2=torch.cuda.get_device_properties(0).L2_cache_size)
    return ok


# ---------------------------------------------------------------------------------------------------- modules
def load_r6(args, path=R6SRC, name=R6NAME):
    if args.source == "installed" and path == R6SRC:
        import importlib as _il

        return _il.import_module(R6NAME)
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


MUTANTS = {
    "kreverse": ("            for k0 in range(0, K, BK):\n                k = k0 + tl.arange(0, BK)\n"
                 "                w = tl.load(W + j64",
                 "            for k1 in range(0, K, BK):\n                k0 = K - BK - k1\n"
                 "                k = k0 + tl.arange(0, BK)\n                w = tl.load(W + j64"),
    # r8-fixb: ksplit = the SAME reduction-order change (two fp32 partial sums added at the end) written as two
    # SEQUENTIAL K-half loops, each with ONE tile load per iteration like the live loop, so its shared memory equals the
    # unmutated kernel's at every (BK, stages, BMX). The round-6 form interleaved two tile loads per iteration and needed
    # 147456 B (> 101376) at the live cfg, BMX 128. Needs (K // 2) % BK == 0 (K 5120: BK 128 / 256 / 512 all divide 2560;
    # asserted by the harness before use).
    "ksplit": ("            acc = tl.zeros((BV, BMX), dtype=tl.float32)\n            for k0 in range(0, K, BK):\n"
               "                k = k0 + tl.arange(0, BK)\n                w = tl.load(W + j64[:, None] * K + k[None, :], "
               "mask=rok[:, None], other=0.0)\n                xt = tl.load(X + mm[None, :] * sx + k[:, None], "
               "mask=mok[None, :], other=0.0)\n                acc = tl.dot(w, xt, acc)\n",
               "            acc = tl.zeros((BV, BMX), dtype=tl.float32)\n            for k0 in range(0, K // 2, BK):\n"
               "                k = k0 + tl.arange(0, BK)\n                w = tl.load(W + j64[:, None] * K + k[None, :], "
               "mask=rok[:, None], other=0.0)\n                xt = tl.load(X + mm[None, :] * sx + k[:, None], "
               "mask=mok[None, :], other=0.0)\n                acc = tl.dot(w, xt, acc)\n"
               "            acc2 = tl.zeros((BV, BMX), dtype=tl.float32)\n            for k0 in range(K // 2, K, BK):\n"
               "                k = k0 + tl.arange(0, BK)\n                w = tl.load(W + j64[:, None] * K + k[None, :], "
               "mask=rok[:, None], other=0.0)\n                xt = tl.load(X + mm[None, :] * sx + k[:, None], "
               "mask=mok[None, :], other=0.0)\n                acc2 = tl.dot(w, xt, acc2)\n            acc = acc + acc2\n"),
    "strict": ("hrow = tl.max((h >= T[:, None])", "hrow = tl.max((h > T[:, None])"),
    "dropfirst": ("tl.store(IDX + pos, j, mask=sel)",
                  "tl.store(IDX + pos, tl.where((pos == base) & (j > 0), j - 1, j), mask=sel)"),
    "egroup": ("(k0 // 32 + tl.arange(0, G32))[:, None]", "((k0 // 32 + 1 + tl.arange(0, G32)) % (K // 32))[:, None]"),
    "tile8": ("sel = tl.reshape(tl.broadcast_to(tneed[:, None], (CB // TB, TB)), (CB,))",
              "sel = tl.reshape(tl.broadcast_to(((tl.max(tl.reshape(hrow, (CB // 8, 8)), axis=1) > 0) | (bad > 0))"
              "[:, None], (CB // 8, 8)), (CB,))"),
}


def mutant_module(args, tag):
    old, new = MUTANTS[tag]
    src = open(R6SRC if args.source == "kit" else sys.modules[R6NAME].__file__).read()
    assert src.count(old) == 1, (tag, src.count(old))
    d = tempfile.mkdtemp(prefix="r6mut-")
    p = os.path.join(d, f"r6mut_{tag}.py")
    open(p, "w").write(src.replace(old, new))
    return load_r6(args, p, f"r6mut_{tag}")


# ---------------------------------------------------------------------------------------------------- data
def head_shard(args, torch):
    if args.weights == "real":
        from safetensors import safe_open

        md = os.path.expanduser(args.model)
        idx = json.load(open(os.path.join(md, "model.safetensors.index.json")))["weight_map"]
        with safe_open(os.path.join(md, idx["head.weight"]), framework="pt", device="cpu") as f:
            sl = f.get_slice("head.weight")
            W = sl[args.rank * V_SHARD:(args.rank + 1) * V_SHARD]
        print(f"INFO real head.weight rows [{args.rank * V_SHARD}, {(args.rank + 1) * V_SHARD}) {tuple(W.shape)} "
              f"{W.dtype}")
        return W.to(torch.bfloat16).cuda().contiguous()
    g = torch.Generator(device=DEV).manual_seed(1234)
    return (torch.randn(V_SHARD, K_HID, device=DEV, generator=g) * 0.02).to(torch.bfloat16)


def x_cases(W, M, torch, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed * 1000 + M)
    out = {}
    out["flat"] = torch.randn(M, K_HID, device=DEV, generator=g).to(torch.bfloat16)
    t = torch.randint(0, W.shape[0], (M,), device=DEV, generator=g)
    wn = W[t].float()
    out["peaked"] = (wn / wn.norm(dim=1, keepdim=True) * 60.0 + torch.randn(M, K_HID, device=DEV, generator=g)
                     * 0.3).to(torch.bfloat16)
    # near-tie cluster: x close to the mean of a contiguous block of rows (many candidates in few tiles)
    c0 = min(4000, W.shape[0] // 4)
    blk = W[c0:c0 + 64].float().mean(0)
    out["cluster"] = (blk / blk.norm() * 80.0 + torch.randn(M, K_HID, device=DEV, generator=g) * 0.05).to(
        torch.bfloat16)
    h = torch.randn(M, K_HID, device=DEV, generator=g)
    h[0] = 0.0
    if M > 2:
        h[1] *= 1e4
        h[2] *= 1e-30
    if M > 3:
        h[3, 7] = float("inf")
    out["hostile"] = h.to(torch.bfloat16)
    return out


def run_pair(r6, st, x, screen=False, refine="0", dq=1):
    r6.SCREEN_DQ = dq
    return r6._run(st, x, screen, refine)


def bits(t):
    import torch
    return t.contiguous().view(torch.int16) if t.element_size() == 2 else t.contiguous().view(torch.int32)


def first_diff(a, b):
    d = (bits(a) != bits(b)).nonzero()
    return d[0].tolist() if len(d) else None


# ---------------------------------------------------------------------------------------------------- parts
def part_bitwise(args, r4, r6, st, torch):
    global GATED_M
    GATED_M = r4.parse_m(os.environ.get("DSV41_R4_CERT_HEAD_M") or "2-96")
    Ms = list(range(2, 97)) if not args.quick else [2, 3, 7, 8, 16, 17, 31, 33, 64, 96]
    if SMOKE:
        Ms = [2, 7, 17]
    nbad = 0
    stock_mm = {"live": 0, "r6row": 0, "n": 0}
    for M in Ms:
        for cname, x in x_cases(st.W, M, torch).items():
            live = run_pair(r6, st, x)
            for dq in (1, 2):
                new = run_pair(r6, st, x, screen=True, dq=dq)
                ok = torch.equal(bits(live["hi"]), bits(new["hi"])) and torch.equal(bits(live["thr"]),
                                                                                     bits(new["thr"]))
                if not ok:
                    nbad += 1
                    check(f"bitwise screen DQ{dq} M={M} {cname}", False, f"first HI diff {first_diff(live['hi'], new['hi'])}")
            for mode in ("tile", "row"):
                new = run_pair(r6, st, x, refine=mode)
                err = r6.compare(live, new, mode, False)
                if err:
                    nbad += 1
                    check(f"bitwise refine {mode} M={M} {cname}", False, err)
            if cname != "hostile" and M in GATED_M:
                stock = torch.nn.functional.linear(x, st.W)
                comp = live["out"] != float("-inf")
                stock_mm["live"] += int(((bits(stock) != bits(live["out"])) & comp).sum())
                newr = run_pair(r6, st, x, refine="row")
                compr = newr["out"] != float("-inf")
                stock_mm["r6row"] += int(((bits(stock) != bits(newr["out"])) & compr).sum())
                stock_mm["n"] += int(comp.sum())
    check(f"bitwise: screen DQ1/DQ2 + refine tile/row at every M in {Ms[0]}..{Ms[-1]} x 4 cases", nbad == 0,
          f"{nbad} failing (M, case, variant)")
    print(f"INFO per-logit vs F.linear (cuBLAS) on computed logits: live mismatches {stock_mm['live']}, r6-row "
          f"mismatches {stock_mm['r6row']} of {stock_mm['n']} live-computed (r6 must equal live; live vs cuBLAS was "
          f"qualified by gpu_test_r4 for CERT_M)")
    check("per-logit: r6-row mismatches vs cuBLAS == live's (r6 adds none)", stock_mm["r6row"] <= stock_mm["live"])
    INFO["stock_mismatch"] = stock_mm
    # ---- config sweeps: one M per served BM / BMX bucket (16, 32, 64, 128; M=7 = production's typical width). A
    # non-default screen shape applies at every BM <= r6.SWEPT_MAX_B, and refine grid/spread at every BMX <= it, so each
    # bucket is run here (BM 64 is where the screen's num_warps becomes max(w, 8)). The live result per (M, case) is
    # computed once and shared by every config.
    swM = SWEEP_M if not SMOKE else [7, 24]
    assert sorted({max(16, r4.next_pow2(m)) for m in swM}) == ([16, 32, 64, 128] if not SMOKE else [16, 32]), swM
    assert max(max(16, r4.next_pow2(m)) for m in swM) == (r6.SWEPT_MAX_B if not SMOKE else 32)
    live_s = {(M, c): run_pair(r6, st, x) for M in swM for c, x in x_cases(st.W, M, torch, seed=1).items()}
    xs_s = {(M, c): x for M in swM for c, x in x_cases(st.W, M, torch, seed=1).items()}
    elig_s, elig_r = [], []
    for bn in ((32, 64, 128) if not SMOKE else (64,)):
        for bk in ((128, 256) if not SMOKE else (128,)):
            for warps in ((4, 8) if not SMOKE else (4,)):
                for stages in ((2, 3, 4) if not SMOKE else (3,)):
                    cfg = (bn, bk, warps, stages)
                    r6.S_BN, r6.S_BK, r6.S_WARPS, r6.S_STAGES = cfg
                    good = True
                    try:
                        # DQ2 only: the module refuses a non-zero SCREEN_CFG unless SCREEN_DQ=2
                        for key, x in xs_s.items():
                            live = live_s[key]
                            new = run_pair(r6, st, x, screen=True, dq=2)
                            if not (torch.equal(bits(live["hi"]), bits(new["hi"]))
                                    and torch.equal(bits(live["thr"]), bits(new["thr"]))):
                                good = False
                                print(f"INFO screen cfg {cfg} NOT bit-identical at M={key[0]} {key[1]}")
                                break
                    except Exception as e:  # noqa: BLE001 (out of resources etc.)
                        good = False
                        print(f"INFO screen cfg {cfg} does not run: {repr(e)[:160]}")
                    if good:
                        elig_s.append(cfg)
    r6.S_BN = r6.S_BK = r6.S_WARPS = r6.S_STAGES = 0
    print(f"INFO screen configs bit-identical (DQ2) at M {swM} (BM 16/32/64/128) x 4 cases (ELIGIBLE): {elig_s}")
    for c in elig_s:
        print(f"ELIGIBLE-ENV screen: DSV41_R6_HEAD_SCREEN=1 DSV41_R6_HEAD_SCREEN_DQ=2 "
              f"DSV41_R6_HEAD_SCREEN_CFG={c[0]},{c[1]},{c[2]},{c[3]}")
    print("ELIGIBLE-ENV screen (live shape, DQ 1 or 2; gated by the bitwise part at every M): "
          "DSV41_R6_HEAD_SCREEN=1 DSV41_R6_HEAD_SCREEN_DQ=<1|2> DSV41_R6_HEAD_SCREEN_CFG=0,0,0,0")
    check("sweep: the live screen cfg (64,128,4,3) with DQ2 is eligible", (64, 128, 4, 3) in elig_s)
    live_r = {(M, c): run_pair(r6, st, x) for M in swM for c, x in x_cases(st.W, M, torch, seed=2).items()}
    xs_r = {(M, c): x for M in swM for c, x in x_cases(st.W, M, torch, seed=2).items()}
    grids = REFINE_GRIDS if not SMOKE else (7,)
    for bk in ((128, 256, 512) if not SMOKE else (128,)):
        for stages in ((2, 3, 4) if not SMOKE else (3,)):
            for grid in grids:
                for spread in (0, 1):
                    # warps stays 0 (live XWARPS): the module refuses any other value
                    r6.R_BK, r6.R_STAGES, r6.R_GRID, r6.R_SPREAD = bk, stages, grid, spread
                    good = True
                    try:
                        for key, x in xs_r.items():
                            live = live_r[key]
                            for mode in ("tile", "row"):
                                if r6.compare(live, run_pair(r6, st, x, refine=mode), mode, False):
                                    good = False
                                    print(f"INFO refine cfg BK{bk} s{stages} g{grid} sp{spread} NOT bitwise at "
                                          f"M={key[0]} {key[1]} {mode}")
                                    break
                            if not good:
                                break
                    except Exception as e:  # noqa: BLE001
                        good = False
                        print(f"INFO refine cfg BK{bk} s{stages} g{grid} sp{spread} does not run: {repr(e)[:160]}")
                    if good:
                        elig_r.append((bk, stages, grid, spread))
    r6.R_BK = r6.R_STAGES = r6.R_GRID = 0
    r6.R_SPREAD = 1
    print(f"INFO refine configs bit-identical at M {swM} (BMX 16/32/64/128) x 4 cases, tile and row (ELIGIBLE; "
          f"BK/stages act at BMX 16 only, grid/spread at every BMX): {elig_r}")
    for bk, stages, grid, spread in elig_r:
        print(f"ELIGIBLE-ENV refine: DSV41_R6_HEAD_REFINE_CFG={bk},0,{stages},{grid},{spread}")
    check("sweep: the live refine cfg (BK128, stages 3) is eligible at every grid/spread",
          all((128, 3, g, s_) in elig_r for g in grids for s_ in (0, 1)))
    INFO["eligible_screen"], INFO["eligible_refine"] = elig_s, elig_r


# ---------------------------------------------------------------------------------------------------- serving path
def part_anchor(args, r4, r6, st, torch):
    """The reference every other part uses is r6._run(st, x, False, "0"), a copy of the live launch sequence. Anchor it
    to the serving entry point: with the r6 flags off, r4.certified_logits(st, x) (the module the env part asserted is
    the serving one) must equal it bit for bit (out, HI, THR, BAD) at every M and case. Must-fail: the same compare
    against r4.certified_logits(..., full=True) (every tile computed) must report a difference at every M."""
    flags_off = all(os.environ.get(n_, "0").strip().lower() in ("", "0", "off", "false", "no")
                    for n_ in ("DSV41_R6_HEAD_REFINE", "DSV41_R6_HEAD_SCREEN"))
    if not check("anchor: r6 flags off in this process (r4.certified_logits runs the live body)",
                 flags_off and not getattr(r4, "_DSV41_R6_HEAD", False)):
        return
    Ms = sorted(GATED_M) if not args.quick else [2, 3, 7, 8, 16, 17, 31, 33, 64, 96]
    if SMOKE:
        Ms = [2, 7, 17]
    nbad, missed = 0, []
    for M in Ms:
        for cname, x in x_cases(st.W, M, torch, seed=40).items():
            dbg = {}
            a = r4.certified_logits(st, x, debug=dbg)
            b = r6._run(st, x, False, "0")
            same = all(torch.equal(bits(u), bits(v)) for u, v in
                       ((a, b["out"]), (dbg["hi"], b["hi"]), (dbg["thr"], b["thr"]), (dbg["bad"], b["bad"])))
            if not same:
                nbad += 1
                check(f"anchor M={M} {cname}: r4.certified_logits == r6._run(live)", False,
                      f"first out diff {first_diff(a, b['out'])}")
        x = x_cases(st.W, M, torch, seed=40)["peaked"]
        if torch.equal(bits(r4.certified_logits(st, x, full=True)), bits(r6._run(st, x, False, "0")["out"])):
            missed.append(M)
    check(f"anchor: r4.certified_logits (flags off) == r6._run(st, x, False, '0') bitwise at every M in "
          f"{Ms[0]}..{Ms[-1]} x 4 cases (out, HI, THR, BAD)", nbad == 0, f"{nbad} failing (M, case)")
    check(f"anchor MUTANT full=True caught at every M ({len(Ms)} values)", not missed, f"missed at M={missed[:12]}")


DISPATCH_M = [2, 7, 24, 33, 48, 96]


def part_dispatch(args, r4, r6, st, torch):
    """Run the INSTALLED dispatch path on the GPU: a child process with the r6 flags set imports the patched
    dsv41_r4_small (the venv's when --source installed; otherwise the live file patched in a temp dir by install.py's
    own edit, sha-checked against KIT.json) and calls r4.certified_logits, which must dispatch to
    dsv41_head_r6.certified_logits (self-check, _done, stats, fallback). One child per mode (row, tile)."""
    for mode in ("row", "tile"):
        cmd = [sys.executable, os.path.abspath(__file__), "--part", "dispatch-child", "--source", args.source,
               "--weights", args.weights, "--model", args.model, "--rank", str(args.rank), "--json", os.devnull,
               "--launcher", args.launcher, "--expect", args.expect]
        cmd += (["--smoke"] if SMOKE else []) + (["--allow-busy"] if args.allow_busy else [])
        env = dict(os.environ, DSV41_R6_HEAD_REFINE=mode, DSV41_R6_HEAD_SCREEN="1", DSV41_R6_HEAD_SELFCHECK="1",
                   DSV41_R6_HEAD_STATS_EVERY="3", R6_DT_PARENT=str(os.getpid()))
        for n_ in ("DSV41_R6_HEAD_REFINE_CFG", "DSV41_R6_HEAD_SCREEN_CFG", "DSV41_R6_HEAD_SCREEN_DQ",
                   "DSV41_R6_HEAD_CAND_ROWS"):
            env.pop(n_, None)
        p = subprocess.run(cmd, env=env, capture_output=True, text=True)
        for ln in (p.stdout + p.stderr).splitlines():
            print(f"  [dispatch {mode}] {ln}")
        check(f"dispatch ({mode}): installed path r4.certified_logits -> r6 on the device, child exit 0",
              p.returncode == 0, f"exit {p.returncode}")


def part_dispatch_child(args, r4, r6, st, torch):
    import logging

    kitj = json.load(open(os.path.join(KIT, "KIT.json")))
    patched_sha = [f.get("patched_sha256") for f in kitj["files"] if f["action"] == "patch"][0]
    mode = r6.REFINE
    check("child: r6 flags parsed (REFINE row|tile, SCREEN 1, SELFCHECK 1)",
          mode in ("row", "tile") and r6.SCREEN and r6.SELFCHECK == 1, f"{mode} {r6.SCREEN} {r6.SELFCHECK}")
    src = open(r4.__file__, encoding="utf-8", newline="").read()
    s0 = hashlib.sha256(src.encode("utf-8")).hexdigest()
    if s0 == patched_sha or s0 in INSTALLED_WITH_HEAD:
        pr4 = r4
        print(f"INFO child: serving dsv41_r4_small is {INSTALLED_WITH_HEAD.get(s0, 'r6-head')} ({s0[:16]}), used as is")
    else:
        check("child: serving dsv41_r4_small is live 3364d8c8 (to be patched in a temp dir)", s0 == LIVE_SHA, s0[:16])
        spec = importlib.util.spec_from_file_location("r6inst", os.path.join(KIT, "install.py"))
        inst = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(inst)
        ptxt = inst.patched(inst.R4, src)
        check("child: install.py's patch of the serving file == KIT.json patched_sha256",
              hashlib.sha256(ptxt.encode("utf-8")).hexdigest() == patched_sha)
        d = tempfile.mkdtemp(prefix="r6disp-")
        pp = os.path.join(d, "dsv41_r4_small.py")
        with open(pp, "w", encoding="utf-8", newline="") as f:
            f.write(ptxt)
        sys.modules[R6NAME] = r6  # the gated `from ...nvidia import dsv41_head_r6` resolves to the module under test
        try:
            import importlib as _il

            setattr(_il.import_module(R6NAME.rsplit(".", 1)[0]), R6NAME.rsplit(".", 1)[1], r6)
        except Exception as e:  # noqa: BLE001
            print(f"INFO child: package attribute not set ({repr(e)[:100]}); sys.modules entry used")
        spec = importlib.util.spec_from_file_location(R4NAME, pp)
        pr4 = importlib.util.module_from_spec(spec)
        sys.modules[R4NAME] = pr4
        spec.loader.exec_module(pr4)
    if not check("child: patched module armed: _DSV41_R6_HEAD True and _r6_head is the module under test",
                 getattr(pr4, "_DSV41_R6_HEAD", False) is True and getattr(pr4, "_r6_head", None) is r6
                 and r6._r4() is pr4):
        return
    stt = pr4.build_cert_tables(st.W)
    logs = []

    class _H(logging.Handler):
        def emit(self, rec):
            logs.append(rec.getMessage())
    r6.logger.addHandler(_H())
    try:
        r6.logger.setLevel(logging.INFO)
    except Exception:  # noqa: BLE001
        pass

    def live(x, dbg=None, full=False):
        pr4._DSV41_R6_HEAD = False
        try:
            return pr4.certified_logits(stt, x, full=full, debug=dbg)
        finally:
            pr4._DSV41_R6_HEAD = True

    Ms = DISPATCH_M if not SMOKE else [2, 7, 17]
    nbad = 0
    for M in Ms:
        if SMOKE and mode == "row":
            # interpreter only: its bf16 dot runs on raw bit patterns, so the certificate (l <= hi) need not hold and a
            # row-mode argmax may legitimately differ; such an M is skipped here (never on the device, where a
            # certificate miss is a real failure)
            inval = []
            for cname, x in x_cases(stt.W, M, torch, seed=30).items():
                dbg = {}
                ref = live(x, dbg)
                exp = r6.expected_row_out(ref, r6.row_select(dbg["hi"], dbg["thr"], dbg["bad"]))
                if not torch.equal(ref.float().argmax(dim=-1), exp.float().argmax(dim=-1)):
                    inval.append(cname)
            if inval:
                print(f"INFO smoke: M={M} skipped for row mode: interpreter numerics INVALID (certificate does not hold "
                      f"for {inval})")
                continue
        for i, (cname, x) in enumerate(x_cases(stt.W, M, torch, seed=30).items()):
            dbg = {}
            ref = live(x, dbg)
            exp = ref if mode == "tile" else r6.expected_row_out(ref, r6.row_select(dbg["hi"], dbg["thr"], dbg["bad"]))
            hits = []
            d2 = {}
            out = pr4.certified_logits(stt, x, debug=d2, after_screen=lambda: hits.append(1))
            if i == 0:
                check(f"child M={M}: first call ran the self-check and passed", M in r6._S6["checked"]
                      and not r6._S6["dead"] and any(f"self-check passed at M={M} " in m_ for m_ in logs))
            ok = torch.equal(bits(out), bits(exp)) and hits == [1] and "cnt" in d2 and "hi" in d2
            pa = pr4.sf_local_pairs(out, stt.vreal, 0)[:, :2]
            pb = pr4.sf_local_pairs(ref, stt.vreal, 0)[:, :2]
            ok &= torch.equal(bits(pa), bits(pb))
            if not ok:
                nbad += 1
                check(f"child M={M} {cname}: dispatched output", False,
                      f"out==exp {torch.equal(bits(out), bits(exp))} after_screen {hits} debug {sorted(d2)}")
        x = x_cases(stt.W, M, torch, seed=31)["peaked"]
        if not torch.equal(bits(pr4.certified_logits(stt, x, full=True)), bits(live(x, full=True))):
            nbad += 1
            check(f"child M={M}: full=True stays live under the flag", False)
    check(f"child: dispatched r4.certified_logits ({mode}) == {'live' if mode == 'tile' else 'where(pred, live, -inf)'}"
          f" bitwise, sf_local_argmax pairs == live, after_screen once, debug filled, full=True live, at M {Ms} x 4 "
          f"cases", nbad == 0, f"{nbad} failing")
    c = [int(v) for v in stt._r6["stats"].tolist()]
    check("child: stats counters (rows computed > 0, rows <= 16 x live-equivalent tiles) and the stats log line",
          c[0] > 0 and c[0] <= 16 * c[1] and any("r6 refine: calls" in m_ for m_ in logs), f"{c}")
    bad_lines = [m_ for m_ in logs if "self-check FAILED" in m_]
    check("child: no self-check FAILED line on the clean pass", not bad_lines, " | ".join(bad_lines)[:400])
    # must-fail: a corrupted r6 result must be caught by the serving self-check at every M; the live result is served
    # and r6 is off for the rest of the process
    orig = r6._run

    def flip_out(st_, x_, scr, ref_, after_screen=None):
        res = orig(st_, x_, scr, ref_, after_screen)
        if ref_ != "0":
            o = res["out"]
            nz = (o != float("-inf")).nonzero()
            if len(nz):
                ob = o.view(torch.int16)
                ob[nz[0, 0], nz[0, 1]] ^= 1
        return res

    def flip_hi(st_, x_, scr, ref_, after_screen=None):
        res = orig(st_, x_, scr, ref_, after_screen)
        if scr:
            res["hi"].view(torch.int16)[0, 0] ^= 1
        return res

    for tag, fn_ in (("out-flip", flip_out), ("hi-flip", flip_hi)):
        missed = []
        for M in Ms:
            x = x_cases(stt.W, M, torch, seed=32)["peaked"]
            ref = live(x)
            r6._S6["checked"], r6._S6["dead"] = set(), False
            n0 = sum("self-check FAILED" in m_ for m_ in logs)
            r6._run = fn_
            try:
                out = pr4.certified_logits(stt, x)
            finally:
                r6._run = orig
            out2 = pr4.certified_logits(stt, x)  # dead: the live kernels serve
            caught = (r6._S6["dead"] and sum("self-check FAILED" in m_ for m_ in logs) == n0 + 1
                      and torch.equal(bits(out), bits(ref)) and torch.equal(bits(out2), bits(ref)))
            if not caught:
                missed.append(M)
        check(f"child MUTANT {tag}: self-check catches it, serves live, r6 off, at every M ({len(Ms)} values)",
              not missed, f"missed at M={missed}")
    r6._S6["checked"], r6._S6["dead"] = set(), False


def tie_weights(W0, torch):
    W = W0.clone()
    V = W0.shape[0]
    pairs = {"across_tiles": (37, 37 + 16 * 20 + 3), "in_tile": (32, 41), "cand_cta": (250, 260),
             "shard_end": (V - 2, V - 1), "far": (5, V - 160)}
    if V > 4096:
        pairs["sf_block"] = (4095, 4096)
    return W, pairs


def part_ties(args, r4, r6, st, torch):
    ok_all = True
    Ms = list(range(2, 97)) if not args.quick else [2, 7, 16, 33, 96]
    if SMOKE:
        Ms = [2, 7]
    for pname, (a, b) in tie_weights(st.W, torch)[1].items():
        W = st.W.clone()
        W[b] = W[a]
        stt = r4.build_cert_tables(W)
        r6._state6(stt, W.device)
        for M in Ms:
            x = (W[a].float() / W[a].float().norm() * 60.0).repeat(M, 1).to(torch.bfloat16)
            stock = torch.nn.functional.linear(x, W).float().argmax(dim=-1)
            if not bool((stock == a).all()):
                print(f"INFO tie {pname} M={M}: stock argmax {stock[:3].tolist()} is not row {a} (x not aligned "
                      f"enough); skipped")
                continue
            for mode in ("tile", "row"):
                res = r6._run(stt, x, True, mode)
                tok = r4.sf_local_pairs(res["out"], stt.vreal, 0)[:, 1].long()
                if not bool((tok == a).all()):
                    ok_all = False
                    check(f"tie {pname} M={M} {mode}", False, f"got {tok[:4].tolist()} want {a}")
    # all rows identical: every row is a candidate, n = V
    W = st.W[:1].repeat(st.W.shape[0], 1).contiguous()
    stt = r4.build_cert_tables(W)
    r6._state6(stt, W.device)
    for M in (2, 7, 33):
        x = torch.randn(M, K_HID, device=DEV).to(torch.bfloat16)
        for mode in ("tile", "row"):
            res = r6._run(stt, x, True, mode)
            tok = r4.sf_local_pairs(res["out"], stt.vreal, 0)[:, 1].long()
            ok = bool((tok == 0).all()) and int(res["cnt"][0]) == W.shape[0]
            ok_all &= ok
            if not ok:
                check(f"tie all-identical M={M} {mode}", False, f"tok {tok[:3].tolist()} n {int(res['cnt'][0])}")
    check("forced ties: r6 chain + live sf_local_argmax == stock leftmost argmax at every M, tile and row", ok_all)


def exact_T(r6, st, x, torch):
    live = r6._run(st, x, False, "0")
    thr = live["hi"].float().max(dim=1).values.contiguous()
    return live, thr


def cand_refine(mod, st, x, hi, thr, bad, mode, torch, r4, bufs=None):
    """cand + refine of `mod` on given HI / THR / BAD. With bufs (timing): preallocated out/idx/need/stats and ONE
    cnt.zero_() memset (production folds that store into _r6_xnorm_kernel, so the r6 times here carry ~2 us extra)."""
    k6 = mod._kernels6()
    M = x.shape[0]
    V = st.V
    BMX = max(16, r4.next_pow2(M))
    if bufs is None:
        out = torch.empty((M, V), dtype=torch.bfloat16, device=DEV)
        idx = torch.empty((V,), dtype=torch.int32, device=DEV)
        cnt = torch.zeros((1,), dtype=torch.int32, device=DEV)
        need = torch.zeros((1,), dtype=torch.int32, device=DEV)
        stats = torch.zeros((2,), dtype=torch.int64, device=DEV)
    else:
        out, idx, cnt, need, stats = bufs
        cnt.zero_()
    cb = mod.cand_rows(BMX, 16)
    k6["cand"][((V + cb - 1) // cb,)](hi, thr, bad, out, out.stride(0), idx, cnt, need, stats, M, V,
                                      BMX=BMX, CB=cb, TB=16, TILE=(mode == "tile"), num_warps=4)
    bk, warps, stages = mod.refine_cfg(BMX, r4)
    gr = 192 if not SMOKE else 7
    k6["refine"][(gr,)](x, x.stride(0), st.W, out, out.stride(0), idx, cnt, M, gr, K=x.shape[1], BV=16, BMX=BMX,
                         BK=bk, SPREAD=True, num_warps=warps, num_stages=stages)
    return out, (cnt if bufs is not None else int(cnt[0]))  # no host sync inside a timed call


def part_mutants(args, r4, r6, st, torch):
    Ms = list(range(2, 97)) if not args.full_mutants else list(range(2, 97))
    if args.quick:
        Ms = [2, 7, 16, 33, 96]
    if SMOKE:
        Ms = [2, 7]
    for tag in MUTANTS:
        if SMOKE and tag in ("kreverse", "ksplit"):
            print(f"INFO smoke: numeric mutant {tag} skipped (interpreter dot numerics are not the GPU's)")
            continue
        m = mutant_module(args, tag)
        m._S6["nsm"] = r6._S6.get("nsm") or (2 if SMOKE else None)
        missed = []
        for M in Ms:
            caught = False
            for seed in range(3):
                xs = x_cases(st.W, M, torch, seed=seed + 10)
                x = xs["flat"] if seed != 2 else xs["cluster"]
                if tag in ("kreverse", "ksplit"):
                    # numeric mutants: every row selected (T := -inf) so all V x M logits are compared; a reduction-
                    # order change flips the bf16 rounding of ~2^-11 of them (>= ~16 flips at M=2)
                    live = r6._run(st, x, False, "0")
                    allT = torch.full_like(live["thr"], float("-inf"))
                    a, _ = cand_refine(r6, st, x, live["hi"], allT, live["bad"], "row", torch, r4)
                    b, _ = cand_refine(m, st, x, live["hi"], allT, live["bad"], "row", torch, r4)
                    caught |= not torch.equal(bits(a), bits(b))
                elif tag == "dropfirst":
                    live = r6._run(st, x, False, "0")
                    m._state6(st, x.device)
                    caught |= r6.compare(live, m._run(st, x, False, "row"), "row", False) != ""
                elif tag == "tile8":
                    live = r6._run(st, x, False, "0")
                    caught |= r6.compare(live, m._run(st, x, False, "tile"), "tile", False) != ""
                elif tag == "egroup":
                    live = r6._run(st, x, False, "0")
                    m._state6(st, x.device)
                    m.SCREEN_DQ = 1
                    new = m._run(st, x, True, "0")
                    caught |= not torch.equal(bits(live["hi"]), bits(new["hi"]))
                elif tag == "strict":
                    live, thr = exact_T(r6, st, x, torch)
                    a, na = cand_refine(r6, st, x, live["hi"], thr, live["bad"], "row", torch, r4)
                    b, nb = cand_refine(m, st, x, live["hi"], thr, live["bad"], "row", torch, r4)
                    caught |= (na != nb) or not torch.equal(bits(a), bits(b))
                if caught:
                    break
            if not caught:
                missed.append(M)
        check(f"MUTANT {tag} caught at every gated M ({len(Ms)} values)", not missed, f"missed at M={missed[:12]}")
    part_mutants_sweep(args, r4, r6, st, torch)


def _oor(e):
    return type(e).__name__ == "OutOfResources" or "out of resource" in repr(e).lower()


def part_mutants_sweep(args, r4, r6, st, torch):
    """r8-fixb: every mutant must FIT and be CAUGHT at every swept config, not only at the live one.
    refine mutants (kreverse, ksplit): every (BK, stages) of the refine sweep x grid 192 x spread 1, at every sweep M
      (BMX 16/16/32/64/128; BK/stages act at BMX 16, the live XBK/XSTAGES above it, so BMX 32-128 re-check the live cfg);
    screen mutant (egroup): every screen (BN, BK, warps, stages) of the screen sweep with DQ2, at every sweep M;
    cand mutants (strict, dropfirst, tile8): no swept cand config exists (CB is DSV41_R6_HEAD_CAND_ROWS, one value); they
      run at every gated M above.
    A config where the UNMUTATED r6 kernel does not run (OutOfResources) is out of the sweep (it is never eligible) and
    is skipped for the mutant too; anywhere the unmutated kernel runs, the mutant must run (FIT) and differ (CAUGHT)."""
    swM = SWEEP_M if not SMOKE else [7]
    nofit, missed, ran, skipped = [], [], 0, 0
    K = st.W.shape[1]
    for tag in ("kreverse", "ksplit"):
        if SMOKE:
            break
        m = mutant_module(args, tag)
        m._S6["nsm"] = r6._S6.get("nsm")
        for bk in (128, 256, 512):
            if tag == "ksplit":
                assert (K // 2) % bk == 0, (K, bk)
            for stages in (2, 3, 4):
                r6.R_BK, r6.R_STAGES, r6.R_GRID, r6.R_SPREAD = bk, stages, 0, 1
                m.R_BK, m.R_STAGES, m.R_GRID, m.R_SPREAD = bk, stages, 0, 1
                for M in swM:
                    x = x_cases(st.W, M, torch, seed=31)["flat"]
                    live = r6._run(st, x, False, "0")
                    allT = torch.full_like(live["thr"], float("-inf"))
                    try:
                        a, _ = cand_refine(r6, st, x, live["hi"], allT, live["bad"], "row", torch, r4)
                    except Exception as e:  # noqa: BLE001
                        if _oor(e):
                            skipped += 1
                            continue
                        raise
                    try:
                        b, _ = cand_refine(m, st, x, live["hi"], allT, live["bad"], "row", torch, r4)
                    except Exception as e:  # noqa: BLE001
                        if _oor(e):
                            nofit.append((tag, bk, stages, M, repr(e)[:90]))
                            continue
                        raise
                    ran += 1
                    if torch.equal(bits(a), bits(b)):
                        missed.append((tag, bk, stages, M))
        r6.R_BK = r6.R_STAGES = r6.R_GRID = 0
        r6.R_SPREAD = 1
    if not SMOKE:
        m = mutant_module(args, "egroup")
        m._S6["nsm"] = r6._S6.get("nsm")
        for bn in (32, 64, 128):
            for bk in (128, 256):
                for warps in (4, 8):
                    for stages in (2, 3, 4):
                        cfg = (bn, bk, warps, stages)
                        r6.S_BN, r6.S_BK, r6.S_WARPS, r6.S_STAGES = cfg
                        m.S_BN, m.S_BK, m.S_WARPS, m.S_STAGES = cfg
                        for M in swM:
                            x = x_cases(st.W, M, torch, seed=32)["flat"]
                            try:
                                ref = run_pair(r6, st, x, screen=True, dq=2)
                            except Exception as e:  # noqa: BLE001
                                if _oor(e):
                                    skipped += 1
                                    continue
                                raise
                            m._state6(st, x.device)
                            m.SCREEN_DQ = 2
                            try:
                                new = m._run(st, x, True, "0")
                            except Exception as e:  # noqa: BLE001
                                if _oor(e):
                                    nofit.append(("egroup",) + cfg + (M, repr(e)[:90]))
                                    continue
                                raise
                            ran += 1
                            if torch.equal(bits(ref["hi"]), bits(new["hi"])):
                                missed.append(("egroup",) + cfg + (M,))
        r6.S_BN = r6.S_BK = r6.S_WARPS = r6.S_STAGES = 0
    print(f"INFO mutant sweep: {ran} (mutant, cfg, M) runs; {skipped} skipped where the unmutated r6 kernel itself does "
          f"not fit (never eligible)")
    if SMOKE:
        print("INFO smoke: the per-config mutant sweep is numeric (GPU only); tools/mutfit.py checks the fit without a GPU")
        return
    check("MUTANT sweep: every mutant FITS (shared memory) wherever its unmutated kernel runs, every swept cfg x M",
          not nofit, str(nofit[:6]))
    check("MUTANT sweep: every mutant CAUGHT at every swept cfg x M (refine BK x stages; screen BN x BK x warps x stages)",
          not missed and ran > 0, str(missed[:8]) if missed else f"{ran} runs")


def part_graph(args, r4, r6, st, torch):
    ok = True
    for M in (2, 7, 33):
        for mode in ("tile", "row"):
            xs = list(x_cases(st.W, M, torch, seed=20).values())[:3]
            xstat = xs[0].clone()
            r6._run(st, xstat, True, mode)  # warm / JIT outside capture
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                with torch.cuda.graph(g):
                    res = r6._run(st, xstat, True, mode)
            torch.cuda.current_stream().wait_stream(s)
            for xi in xs:
                xstat.copy_(xi)
                g.replay()
                torch.cuda.synchronize()
                eager = r6._run(st, xi, True, mode)
                same = torch.equal(bits(res["out"]), bits(eager["out"])) and torch.equal(bits(res["hi"]),
                                                                                          bits(eager["hi"]))
                ok &= same
                if not same:
                    check(f"graph M={M} {mode}", False, "replay != eager")
    check("graph: captured r6 pass replays == eager on 3 inputs (M 2/7/33, tile and row)", ok)


def timeit(fn, torch, n=200, cold=False, scrub=None):
    ts = []
    for i in range(n + 10):
        if cold:
            scrub.add_(1)
        torch.cuda.synchronize()
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        if i >= 10:
            ts.append(a.elapsed_time(b) * 1000)
    ts.sort()
    return ts[len(ts) // 2]


def part_time(args, r4, r6, st, torch):
    scrub = torch.zeros(64 << 20, dtype=torch.int32, device=DEV)  # 256 MB
    k = r4._kernels()
    k6 = r6._kernels6()
    T = {}
    for M in (2, 7, 16, 32):
        x = x_cases(st.W, M, torch)["peaked"]
        live = r6._run(st, x, False, "0")
        BM = max(16, r4.next_pow2(M))
        hi = torch.empty_like(live["hi"])
        thr = torch.empty_like(live["thr"])

        def scr_live():
            k["screen"][((st.V + 63) // 64,)](x, x.stride(0), st.q, st.e, st.rt, st.nv, live["xg"], live["xn"], hi,
                                             thr, M, st.V, st.vreal, K=K_HID, G=128, BM=BM, BN=64, BK=128,
                                             CACC2=2.0 * r4.c_acc(K_HID), INFL_Q=r4.INFL_Q, INFL_B=r4.INFL_B,
                                             SLOP_REL=r4.SLOP_REL, SLOP_ABS=r4.SLOP_ABS, num_warps=4 if BM <= 32
                                             else 8, num_stages=3)

        def scr_r6(cfg, dq):
            bn, bk, warps, stages = cfg

            def f():
                k6["screen"][((st.V + bn - 1) // bn,)](x, x.stride(0), st.q, st.e, st.rt, st.nv, live["xg"],
                                                       live["xn"], hi, thr, M, st.V, st.vreal, K=K_HID, G=128, BM=BM,
                                                       BN=bn, BK=bk, CACC2=2.0 * r4.c_acc(K_HID), INFL_Q=r4.INFL_Q,
                                                       INFL_B=r4.INFL_B, SLOP_REL=r4.SLOP_REL, SLOP_ABS=r4.SLOP_ABS,
                                                       DQ=dq, num_warps=warps if BM <= 32 else max(warps, 8),
                                                       num_stages=stages)
            return f
        for cold in (False, True):
            tag = "cold" if cold else "warm"
            T[f"screen live M{M} {tag}"] = timeit(scr_live, torch, cold=cold, scrub=scrub)
            for cfg in sorted(set([(64, 128, 4, 3)] + [tuple(c) for c in INFO.get("eligible_screen", [])])):
                for dq in ((1, 2) if cfg == (64, 128, 4, 3) else (2,)):  # non-default shapes serve with DQ2 only
                    try:
                        T[f"screen r6 DQ{dq} {cfg} M{M} {tag}"] = timeit(scr_r6(cfg, dq), torch, cold=cold,
                                                                         scrub=scrub)
                    except Exception as e:  # noqa: BLE001
                        print(f"INFO screen r6 {cfg} failed to run: {repr(e)[:120]}")
    # refine at forced candidate sets (same HI/THR/BAD both sides)
    for M in (2, 7, 16):
        x = x_cases(st.W, M, torch)["flat"]
        BMX = max(16, r4.next_pow2(M))
        for n in (1, 16, 64, 256, 1024, 4096, 16160):
            for pat in ("scattered", "clustered"):
                if pat == "scattered" and n > st.V // 16:
                    continue
                rows = (torch.arange(n, device=DEV) * 16 + 5) if pat == "scattered" else torch.arange(n,
                                                                                                         device=DEV)
                hi = torch.full((M, st.V), float("-inf"), dtype=torch.bfloat16, device=DEV)
                hi[:, rows] = 1.0
                thr = torch.zeros((M,), dtype=torch.float32, device=DEV)
                bad = torch.zeros((M,), dtype=torch.int32, device=DEV)
                out_l = torch.empty((M, st.V), dtype=torch.bfloat16, device=DEV)

                def ref_live():
                    k["refine"][((st.V + 15) // 16,)](x, x.stride(0), st.W, hi, thr, bad, out_l, out_l.stride(0),
                                                     st.need, M, st.V, K=K_HID, BV=16, BMX=BMX, BK=128, FULL=False,
                                                     num_warps=4, num_stages=3)
                ref_live()
                for mode in ("tile", "row"):
                    o, nn = cand_refine(r6, st, x, hi, thr, bad, mode, torch, r4)
                    if mode == "tile":
                        check(f"forced set n={n} {pat} M={M} tile == live (bitwise)", torch.equal(bits(o),
                                                                                                    bits(out_l)))
                    else:
                        sel = torch.zeros(st.V, dtype=torch.bool, device=DEV)
                        sel[rows] = True
                        check(f"forced set n={n} {pat} M={M} row == where(sel, live, -inf)", torch.equal(
                            bits(o), bits(r6.expected_row_out(out_l, sel))) and nn == n)
                for cold in (False, True):
                    tag = "cold" if cold else "warm"
                    T[f"refine live n{n} {pat} M{M} {tag}"] = timeit(ref_live, torch, cold=cold, scrub=scrub)
                    for mode in ("tile", "row"):
                        bufs = (torch.empty((M, st.V), dtype=torch.bfloat16, device=DEV),
                                torch.empty((st.V,), dtype=torch.int32, device=DEV),
                                torch.zeros((1,), dtype=torch.int32, device=DEV),
                                torch.zeros((1,), dtype=torch.int32, device=DEV),
                                torch.zeros((2,), dtype=torch.int64, device=DEV))
                        T[f"refine r6 {mode} n{n} {pat} M{M} {tag}"] = timeit(
                            lambda: cand_refine(r6, st, x, hi, thr, bad, mode, torch, r4, bufs), torch, cold=cold,
                            scrub=scrub)
    # full chain on the realistic cases
    for M in (2, 7, 16):
        for cname, x in x_cases(st.W, M, torch).items():
            if cname == "hostile":
                continue
            for cold in (False, True):
                tag = "cold" if cold else "warm"
                T[f"chain live {cname} M{M} {tag}"] = timeit(lambda: r6._run(st, x, False, "0"), torch, cold=cold,
                                                            scrub=scrub)
                for mode in ("tile", "row"):
                    for scr in (False, True):
                        T[f"chain r6 {mode} screen{int(scr)} {cname} M{M} {tag}"] = timeit(
                            lambda: r6._run(st, x, scr, mode), torch, cold=cold, scrub=scrub)
            res = r6._run(st, x, False, "row")
            live = r6._run(st, x, False, "0")
            print(f"INFO candidates {cname} M={M}: rows {int(res['cnt'][0])}, live tiles "
                  f"{int((live['out'] != float('-inf')).any(dim=0).view(-1, 16).any(dim=1).sum())}")
    for kk, v in T.items():
        print(f"TIME {kk}: {v:.1f} us")
    INFO["time_us"] = T


def part_l2(args, r4, r6, st, torch):
    import triton
    import triton.language as tl

    @triton.jit
    def _pre(P, OUT, NW, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        acc = tl.zeros((BLOCK,), dtype=tl.int32)
        for i in range(pid * BLOCK, NW, tl.num_programs(0) * BLOCK):
            o = i + tl.arange(0, BLOCK)
            acc ^= tl.load(P + o, mask=o < NW, other=0, eviction_policy="evict_last")
        tl.store(OUT + pid * BLOCK + tl.arange(0, BLOCK), acc)

    scrub = torch.zeros(64 << 20, dtype=torch.int32, device=DEV)
    k = r4._kernels()
    q32 = st.q.view(-1).view(torch.int32)
    sink = torch.empty((96 * 1024,), dtype=torch.int32, device=DEV)
    res = {}
    for M in (2, 7):
        x = x_cases(st.W, M, torch)["peaked"]
        live = r6._run(st, x, False, "0")
        hi, thr = torch.empty_like(live["hi"]), torch.empty_like(live["thr"])
        BM = 16

        def scr():
            k["screen"][((st.V + 63) // 64,)](x, x.stride(0), st.q, st.e, st.rt, st.nv, live["xg"], live["xn"], hi,
                                             thr, M, st.V, st.vreal, K=K_HID, G=128, BM=BM, BN=64, BK=128,
                                             CACC2=2.0 * r4.c_acc(K_HID), INFL_Q=r4.INFL_Q, INFL_B=r4.INFL_B,
                                             SLOP_REL=r4.SLOP_REL, SLOP_ABS=r4.SLOP_ABS, num_warps=4, num_stages=3)
        for mb in (0, 2, 4, 6, 8, 12):
            nw = mb * (1 << 20) // 4

            def pre():
                if nw:
                    _pre[(96,)](q32, sink, nw, BLOCK=1024, num_warps=4)
            ts, tp = [], []
            for i in range(60):
                scrub.add_(1)
                torch.cuda.synchronize()
                a, b, c = (torch.cuda.Event(enable_timing=True) for _ in range(3))
                a.record()
                pre()
                b.record()
                scr()
                c.record()
                torch.cuda.synchronize()
                if i >= 10:
                    tp.append(a.elapsed_time(b) * 1000)
                    ts.append(b.elapsed_time(c) * 1000)
            ts.sort()
            tp.sort()
            res[f"M{M} P{mb}MB"] = (ts[len(ts) // 2], tp[len(tp) // 2])
            print(f"L2 M={M} pre-read {mb:2d} MB: pre-read {tp[len(tp) // 2]:7.1f} us, screen after it "
                  f"{ts[len(ts) // 2]:7.1f} us")
    INFO["l2"] = res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all")
    ap.add_argument("--weights", default="synthetic", choices=("synthetic", "real"))
    ap.add_argument("--model", default="~/models/dsv41/official-fp8")
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--source", default="kit", choices=("kit", "installed"))
    ap.add_argument("--allow-busy", action="store_true")
    ap.add_argument("--launcher", default=os.path.expanduser("~/dsv41-tp8.sh"),
                    help="production launcher whose env is mirrored (r8 rule)")
    ap.add_argument("--expect", default=os.environ.get("FIXB_EXPECT", "4f3cc1c6"),
                    help="production launcher sha8 (pass the new one after any promotion)")
    ap.add_argument("--quick", action="store_true", help="fewer M values (development only; not a gate run)")
    ap.add_argument("--full-mutants", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="CPU + Triton interpreter smoke run of the harness (NOT a gate)")
    ap.add_argument("--json", default=os.path.join(KIT, "results", "device_test.json"))
    args = ap.parse_args()
    global DEV, SMOKE, V_SHARD, K_HID
    if args.smoke:
        DEV, SMOKE, V_SHARD, K_HID = "cpu", True, 1000, 512
        os.environ["TRITON_INTERPRET"] = "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        if args.part == "all":
            args.part = "anchor,bitwise,ties,mutants,dispatch"
        print("SMOKE MODE: CPU + Triton interpreter, V 1000 K 512; the interpreter evaluates bf16 dots on raw bit "
              "patterns, so numeric checks are not meaningful; this exercises the harness only. NOT A GATE.")
    parts = ("env", "anchor", "bitwise", "ties", "mutants", "graph", "dispatch", "time", "l2") \
        if args.part == "all" else tuple(args.part.split(","))
    t0 = time.time()
    if SMOKE:
        spec = importlib.util.spec_from_file_location(R4NAME, os.path.join(KIT, "base/vllm/models/deepseek_v41/nvidia/dsv41_r4_small.py"))
        m4 = importlib.util.module_from_spec(spec)
        sys.modules[R4NAME] = m4
        spec.loader.exec_module(m4)
    elif not env_checks(args):
        print("environment / serving-op assertions failed: refusing to run (exit 3)")
        return 3
    import torch

    torch.manual_seed(0)
    r4 = sys.modules[R4NAME]
    global GATED_M
    GATED_M = r4.parse_m(os.environ.get("DSV41_R4_CERT_HEAD_M") or "2-96")
    r6 = load_r6(args)
    if SMOKE:
        r6._S6["nsm"] = 2
    W = head_shard(args, torch)
    st = r4.build_cert_tables(W)
    r6._state6(st, W.device)
    print(f"INFO CERT table: V {st.V} K {st.K} eps {st.eps:.5f} unsafe rows {st.n_unsafe}; r6 dq_ok "
          f"{st._r6['dq_ok'] if r6.SCREEN else '(SCREEN flag off: forcing the table check)'}")
    st._r6["dq_ok"] = int(st.e.min()) >= 1 and int(st.e.max()) <= 254
    check("CERT table exponent bytes in 1..254 (DQ precondition)", st._r6["dq_ok"])
    for p in parts:
        if p == "env":
            continue
        t = time.time()
        {"anchor": part_anchor, "bitwise": part_bitwise, "ties": part_ties, "mutants": part_mutants,
         "graph": part_graph, "dispatch": part_dispatch, "dispatch-child": part_dispatch_child,
         "time": part_time, "l2": part_l2}[p](args, r4, r6, st, torch)
        print(f"INFO part {p} done in {time.time() - t:.0f} s", flush=True)
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    INFO["fails"] = FAILS
    INFO["args"] = vars(args)
    json.dump(INFO, open(args.json, "w"), indent=1, default=str)
    print(f"\n{'ALL PASS' if not FAILS else 'FAILED: ' + str(FAILS[:20])} ({time.time() - t0:.0f} s)")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
