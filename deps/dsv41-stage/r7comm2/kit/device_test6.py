#!/usr/bin/env python3
"""device_test.py: r6-comm device test (r7-comm2 copy: paths point at ~/dsv41-stage/r7comm2; the r7 .so must pass every r6 part unchanged) (C1 selftest8 + C2 PDL edge/poison test + must-fail mutants). One GPU, no network.

STOPPED NODE ONLY (the service must not run on this Spark). Mandatory environment (BRIEF "Device-test environment"):

    source ~/venvs/vllm032/bin/activate; export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
    cd ~/dsv41-stage/r7comm2/kit && python -I device_test.py --part all

The test refuses (exit 2, before any GPU work) unless: python is the vllm032 venv, nvcc / cuobjdump / ninja are on PATH,
CUDA_HOME is set, no serving process runs, the live .so is 14d049dd.., the candidate .so and every mutant .so have the
sha256 KIT.json records, and the launcher's RDMA defaults give the production kernel mask 7 (PUB_V1 + PUB_V2 + HYG_POLL)
with 512 vectors per CTA. It then ASSERTS THE SERVING OP: a CUDA profiler trace of one reference call must show the
production kernel ar_kernel_pl<mode, f32, 8, 5|7> from the live .so, and of one candidate call the r3/r6 kernel with
the expected mask (C2: mask & 128), else exit 2.

Reference = the LIVE .so (~/dsv41-stage/g130/rdma/build/rdma_ar_ext.so, the serving binary), loaded beside the candidate
(module names live.rdma_ar_ext / cand.rdma_ar_ext). Each uses its own SelfTest arena (no HCA, no proxy: flags and
posted_seq are preset far in the future, so no kernel waits on a network; the local term, the receive slots and the
two-shot phase-2 slots are images written by the test). Comparison: raw bytes (bitwise), first mismatch index printed.

SHAPES (review B1): derived from the BYTE BOUNDS of launcher 166d7aac, not from decode (shapes.py). The launcher sets
RDMA_AR_MAX_REDUCE_BYTES=2097152 and leaves RDMA_AR_MAX_BYTES at 2 MiB, so rdma_ar (and ar_kernel_r3 under TRIM=1) serves
bf16 [M, 5120] all-reduces up to M = 204, fp32 up to M = 102 and gathers up to 2 MiB of input (conc17-32 decode and
prefill chunks <= 204 tokens). AR_MS = 1..24 + {25, 32, 38, 39, 40, 48, 51, 52, 64, 96, 102, 103, 128, 160, 204} + both
sides of every kernel threshold (S1, two-shot routing, P2, 16 CTAs, 4x-unrolled staging entered once / twice, multi-
iteration two-shot consumer and phase 2), capped at 204 (bf16) / 102 (fp32); gathers: 1..24 + both sides of every
threshold + 1 MiB + the 2 MiB maximum, per segment size. shapes.check() (run by --cpu-selfcheck) proves that every list
below reaches every code-path signature any served shape has, and that the old 1..24 list did not.
--part c1 (selftest8, the C1 gate; run on all 8 nodes):
  AR bf16 and fp32 [M, 5120], M in AR_MS (41 bf16 values up to 204, 37 fp32 values up to 102), one-shot AND two-shot for
  every M (the router picks two-shot at >= 32 KiB; both are tested), S1 thresholds 24576 (production) / 0 / 1 GiB,
  candidate kernels with PDL bits 0 / 128 / 384; dim-0 gathers [8, M, seg / 2] and column gathers [M, 8 * seg / 2]
  (seg 448 / 16 / 32 / 1280 B), M in shapes.gather_ms(seg) (up to the 2 MiB input bound). Inputs per case, two fresh
  calls each:
    rand      wide dynamic range (2^-14..2^14 scale), distinct per rank and call
    hostile   random + planted per-element patterns in EVERY two-shot chunk (so every rank's reduced chunk holds them):
              all -0.0 (seed), [2^24, 1, 1, -2^24] at 5 rank rotations (order: serial 0, reversed 2, tree 1), [max, max,
              -max, -max] (overflow order: +Inf / -Inf / NaN), +Inf & -Inf, NaN payloads, bf16/fp32 denormals (FTZ),
              bf16 rounding ties (RN-even); gathers: -0.0 and NaN payloads must be copied bit for bit
    recorded  production data: rows of the served checkpoint's embed.weight split exactly as the vocab-parallel
              embedding AR sees them (rank r holds the row of a token in its 16160-row shard, +0.0 elsewhere), and
              --recorded DIR (*.pt dicts {"inputs": [8, ...] tensor}) when the lead has dumps; without either the part
              is INCOMPLETE (exit 3), never PASS (embed rows: every bf16 M of AR_MS; --recorded: any M <= 204)
  Per call: candidate bytes == live bytes; send slot == input bytes (S1 / PDL staging); two-shot send2 slot == this
  rank's reduced chunk; st->seq += 1 and every arrival counter back at 0. Plus a captured mixed graph (one-shot,
  two-shot incl. M = 39 / 103 / 204, gather incl. a 2 MiB dim-0 gather, column gather; trim and PDL kernels) replayed 20x
  with fresh inputs, == eager live. An independent
  numpy emulation of the spec (+0.0 seed, FADD r = 0..7, one RN, canonical NaN) is compared too (reported, and a
  disagreement with the LIVE kernel is a harness error).
--part mutants (must-fail, C1): the same AR cases on prebuilt/mutants/{m_rev, m_seed0, m_negzero, m_tree}.so: each must
  mismatch the live kernel in EVERY gated case (every M of AR_MS up to 204 / 102, one- and two-shot, bf16 and fp32). m_ffma is run and reported
  as gate-only: x*1 is exact and FFMA rounds once, so its results are bitwise identical; only gate/fpflow.py sees it.
--part c2 (PDL, C2):
  edge   a captured producer -> AR graph: the edge into ar_kernel_r3 must be PROGRAMMATIC (cuGraphGetEdges_v2) with PDL
         bits 128 / 384 and DEFAULT with bits 0 (negative control); AR -> AR back to back: PROGRAMMATIC; trig: AR ->
         a PDL consumer: PROGRAMMATIC.
  poison graph [poison(x) -> slow producer (griddepcontrol.launch_dependents at entry, spins 20 us, then writes x) ->
         AR (PDL)] replayed 8x per case with new data: the staged send slot (what the NIC sends) must equal the final x,
         the output must equal the live kernel on the final x, counters exact. Cases (shapes.c2_*): AR bf16 M = 1..24 +
         {38, 39, 89, 90, 102, 103, 204}, AR fp32 M in a cover of every served fp32 signature (up to 102), dim-0 and
         column gathers [M, 224] in a cover of every served gather signature (up to 2 MiB); one- and two-shot; bits 0
         (edge control), 128 and 384; ranks 0 and 5. Must-fail mutant prebuilt/mutants/m_waitlate.so (griddepcontrol.wait after the stage +
         publish): must be caught at EVERY case. Control: the same graph without the producer must be flagged (the
         comparator sees poison).
  trig   [slow producer -> AR (bits 384) -> PDL consumer (waits, copies the AR output)]: consumer copy == output ==
         live reference over the replays.
         AR -> AR back to back and AR -> PDL consumer: M = 1, 4, 6, 24, 103, 204.
--time: median kernel time (200 runs) live vs candidate per size (no network wait).
Exit: 0 PASS; 1 FAIL (a mismatch, an uncaught mutant, a failed control); 2 refused (environment / serving-op assertion);
3 INCOMPLETE (a part could not run: e.g. no recorded inputs). Last line: "R6_DEVICE_TEST PASS|FAIL|INCOMPLETE".
Expected runtime (estimate, not measured on a GPU): c1 ~15-30 min per node (2 ranks), mutants ~10-20 min, c2 ~15-30 min;
ops/run-device8-r6.sh allows 7200 s per part.
--cpu-selfcheck: no GPU, numpy only: shapes.check() (the lists reach every served signature) and, for every gated AR
case (both dtypes x every M of AR_MS x one/two-shot x 8 ranks), checks with the spec emulation that each device mutant is
detectable by the planted patterns (tests/run_tests.py T9 runs it)."""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import shapes as S  # noqa: E402  (the served shape lists, derived from the launcher's byte bounds: review B1)

W = 8
HIDDEN = 5120
QKV = 224
MAX_BYTES = 2 << 20                 # RDMA_AR_MAX_BYTES (default, unset in 166d7aac): slot size and gather bound
MAX_REDUCE_BYTES = S.MAX_REDUCE_BYTES  # RDMA_AR_MAX_REDUCE_BYTES=2097152 in 166d7aac and 4f3cc1c6 (same RDMA env)
assert S.W == W and S.HIDDEN == HIDDEN and S.QKV == QKV and S.MAX_GATHER_BYTES == MAX_BYTES
AR_MS = {False: S.ar_ms(False), True: S.ar_ms(True)}  # bf16 up to 204 rows, fp32 up to 102
QUICK_AR_MS = {False: [1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 39, 103, 204], True: [1, 4, 20, 24, 52, 102]}
LIVE_SO = os.path.expanduser("~/dsv41-stage/g130/rdma/build/rdma_ar_ext.so")
LIVE_SHA = "14d049dda41a2a8ff9e619bf6667c4107dc16fbd25ede4d69616ab1507570ef7"
CAND_SO = os.path.expanduser("~/dsv41-stage/r7comm2/build/rdma_ar_ext.so")
LAUNCHER = os.path.expanduser("~/dsv41-tp8.sh")
MODEL = os.path.expanduser("~/models/dsv41/official-fp8")
# r8-comm2fix (review B2): the production launcher 4f3cc1c6 exports the promoted r5 glue flags; every device test must
# run with them (BRIEF "Device-test environment"). ENV_PREFIX sets them and preflight() refuses unless the process env
# holds exactly these values AND the launcher on the node still defaults to them (a stale kit refuses instead of testing
# a different env than the one that serves).
PROD_GLUE_ENV = (("DSV41_GLUE_RQPAD", "1"), ("DSV41_GLUE_RPSPLIT", "split"))
ENV_PREFIX = ("source ~/venvs/vllm032/bin/activate; export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda "
              + " ".join(f"{k}={v}" for k, v in PROD_GLUE_ENV))
DEVICE_MUTANTS = ["m_rev", "m_seed0", "m_negzero", "m_tree"]
GATE_ONLY_MUTANTS = ["m_ffma"]
PDL_MUTANT = "m_waitlate"
VOCAB_SHARD = 129280 // W


# ================================================================================================ numpy emulation ===
def f32_of_bf16(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def bf16_rn(f):
    """fp32 -> bf16 bits, round to nearest even; NaN -> 0x7FFF (GPU F2FP canonical NaN)."""
    b = f.astype(np.float32).view(np.uint32).astype(np.uint64)
    nan = np.isnan(f)
    r = ((b + 0x7FFF + ((b >> 16) & 1)) >> 16).astype(np.uint16)
    r[nan] = 0x7FFF
    return r


def canon(f):
    """GPU FADD returns the canonical NaN 0x7FFFFFFF for any NaN result."""
    f = f.astype(np.float32).copy()
    f.view(np.uint32)[np.isnan(f)] = 0x7FFFFFFF
    return f


def reduce_terms(terms, order="serial"):
    """terms: [8, n] fp32 (exact widenings). The spec (serial, +0.0 seed) and the mutant orders."""
    with np.errstate(over="ignore", invalid="ignore"):
        if order == "serial" or order == "ffma":  # FFMA(x, 1, acc) == FADD(acc, x) bitwise: x * 1 exact, one rounding
            acc = np.zeros(terms.shape[1], np.float32)
            for r in range(W):
                acc = canon(acc + terms[r])
        elif order == "rev":
            acc = np.zeros(terms.shape[1], np.float32)
            for r in reversed(range(W)):
                acc = canon(acc + terms[r])
        elif order in ("seed0", "negzero"):  # -0.0 + x == x exactly (ptxas folds it): both start from rank 0's term
            acc = terms[0].astype(np.float32).copy()
            for r in range(1, W):
                acc = canon(acc + terms[r])
        elif order == "tree":
            t = [canon(np.float32(0) + terms[0])]
            t01 = canon(t[0] + terms[1])
            t23 = canon(terms[2] + terms[3])
            t45 = canon(terms[4] + terms[5])
            t67 = canon(terms[6] + terms[7])
            acc = canon(canon(t01 + t23) + canon(t45 + t67))
        else:
            raise ValueError(order)
    return acc


def emulate_ar(vals_bits, rank, two, f32, recv2_bits=None, order="serial"):
    """vals_bits: [8, n] uint16 (bf16) or uint32 (fp32) = every rank's input. Output bits of this rank's all-reduce."""
    n = vals_bits.shape[1]
    epv = 4 if f32 else 8
    terms = vals_bits.view(np.float32) if f32 else f32_of_bf16(vals_bits)
    if not two:
        s = reduce_terms(terms, order)
        return s.view(np.uint32) if f32 else bf16_rn(s)
    nvec = n // epv
    chunk = (nvec + W - 1) // W
    out = np.zeros(n, vals_bits.dtype)
    b, ln = rank * chunk, max(0, min(chunk, nvec - rank * chunk))
    if ln:
        sl = slice(b * epv, (b + ln) * epv)
        s = reduce_terms(terms[:, sl], order)
        out[sl] = s.view(np.uint32) if f32 else bf16_rn(s)
    for r in range(W):
        if r == rank:
            continue
        lr = max(0, min(chunk, nvec - r * chunk))
        if lr:
            out[r * chunk * epv:(r * chunk + lr) * epv] = recv2_bits[r][:lr * epv]
    return out


# ========================================================================================== inputs and patterns ===
BF = {"one": 0x3F80, "big": 0x4B80, "nbig": 0xCB80, "max": 0x7F7F, "nmax": 0xFF7F, "inf": 0x7F80, "ninf": 0xFF80,
      "nz": 0x8000, "z": 0x0000, "den": 0x0001, "nden": 0x8001, "half_ulp": 0x3B80, "one_p": 0x3F81, "nan1": 0x7FC1,
      "nan2": 0xFFA5, "minn": 0x0080}
FP = {"one": 0x3F800000, "big": 0x4B800000, "nbig": 0xCB800000, "max": 0x7F7FFFFF, "nmax": 0xFF7FFFFF, "inf": 0x7F800000,
      "ninf": 0xFF800000, "nz": 0x80000000, "z": 0x0, "den": 0x1, "nden": 0x80000001, "half_ulp": 0x33800000,
      "one_p": 0x3F800001, "nan1": 0x7FC00001, "nan2": 0xFFA00005, "minn": 0x00800000}


def patterns(f32):
    """per-element rank vectors (8 bit patterns). Each mutant of mk_mutants.py changes the result of at least one."""
    c = FP if f32 else BF
    z = c["z"]
    pats = [[c["nz"]] * W]                                            # seed: +0 -> +0.0, rank-0 seed -> -0.0
    core = [c["big"], c["one"], c["one"], c["nbig"]]
    for s in range(5):                                                 # order: serial 0 / reversed 2 / tree 1
        v = [z] * W
        v[s:s + 4] = core
        pats.append(v)
    pats.append([c["max"], c["max"], c["nmax"], c["nmax"], z, z, z, z])  # +Inf / -Inf / NaN by order
    pats.append([z, z, z, z, c["max"], c["max"], c["nmax"], c["nmax"]])
    pats.append([c["inf"], z, z, z, z, z, z, c["ninf"]])              # NaN
    pats.append([z, z, z, c["inf"], z, z, z, z])                      # +Inf
    pats.append([z, z, z, z, z, c["nan1"], z, z])                     # NaN payload -> canonical
    pats.append([c["nan2"], c["one"], z, z, z, z, z, z])
    pats.append([c["den"]] * W)                                       # denormal sum (FTZ would give 0)
    pats.append([c["den"], c["nden"], c["den"], c["den"], c["nden"], c["den"], c["den"], c["minn"]])
    pats.append([c["one"], c["half_ulp"], z, z, z, z, z, z])          # bf16 tie -> even
    pats.append([c["one_p"], c["half_ulp"], z, z, z, z, z, z])
    pats.append([c["z"], c["nz"], c["nz"], c["z"], c["nz"], c["nz"], c["nz"], c["nz"]])
    return pats


def plant_positions(n, f32):
    """element offsets: the first elements of every two-shot chunk (so every rank's own chunk holds every pattern) and
    the last elements of the tensor."""
    epv = 4 if f32 else 8
    nvec = n // epv
    chunk = (nvec + W - 1) // W
    k = len(patterns(f32))
    starts = sorted({r * chunk * epv for r in range(W) if r * chunk < nvec} | {max(0, n - k)})
    pos = []
    for s in starts:
        pos.extend(range(s, min(n, s + k)))
    return starts, pos


class Gen:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def wide(self, n, f32):
        x = (self.rng.standard_normal(n) * np.exp2(self.rng.integers(-14, 14, n))).astype(np.float32)
        return x.view(np.uint32) if f32 else bf16_rn(x)

    def row(self, n, f32, family, rank):
        """one rank's input (the same planting as ranks(), for that rank only)"""
        x = self.wide(n, f32)
        if family == "hostile":
            pats = np.array(patterns(f32), dtype=np.uint32 if f32 else np.uint16)[:, rank]
            starts, _ = plant_positions(n, f32)
            for s in starts:
                k = min(len(pats), n - s)
                x[s:s + k] = pats[:k]
        return x

    def ranks(self, n, f32, family):
        v = np.stack([self.wide(n, f32) for _ in range(W)])
        if family == "hostile":
            pats = np.array(patterns(f32), dtype=np.uint32 if f32 else np.uint16)  # [P, 8]
            starts, _ = plant_positions(n, f32)
            for s in starts:
                k = min(len(pats), n - s)
                v[:, s:s + k] = pats[:k].T
        return v


def recorded_embed(model, Ms, seed, rows_expected=VOCAB_SHARD * W, hidden=HIDDEN):
    """{M: [8, M*5120] uint16}: the production embedding all-reduce inputs for M random token ids (bf16 rows of
    embed.weight, +0.0 on every rank whose vocab shard does not hold the token), or None."""
    idx = os.path.join(model, "model.safetensors.index.json")
    if not os.path.exists(idx):
        return None
    try:
        from safetensors import safe_open
    except Exception:  # noqa: BLE001
        return None
    fn = json.load(open(idx))["weight_map"].get("embed.weight")
    if not fn:
        return None
    import torch
    rng = np.random.default_rng(seed)
    out = {}
    with safe_open(os.path.join(model, fn), "pt") as f:
        sl = f.get_slice("embed.weight")
        rows, cols = sl.get_shape()
        if cols != hidden or rows != rows_expected or str(sl.get_dtype()).upper() not in ("BF16", "BFLOAT16"):
            return None
        for M in Ms:
            ids = rng.integers(0, rows, M)
            mat = np.stack([sl[int(i):int(i) + 1].contiguous().view(torch.int16).numpy().view(np.uint16).reshape(-1)
                            for i in ids])
            v = np.zeros((W, M, hidden), np.uint16)  # +0.0 on every rank whose shard does not hold the token
            for j, i in enumerate(ids):
                v[int(i) // (rows_expected // W), j] = mat[j]
            out[M] = v.reshape(W, -1)
    return out


def recorded_dir(d):
    """{name: [8, n] uint16/uint32} from --recorded DIR/*.pt ({"inputs": tensor [8, ...] bf16 or fp32})."""
    import torch
    out = {}
    for p in sorted(os.listdir(d)):
        if p.endswith(".pt"):
            t = torch.load(os.path.join(d, p), map_location="cpu")["inputs"]
            if t.dtype == torch.bfloat16:
                out[p] = t.reshape(W, -1).view(torch.int16).numpy().view(np.uint16)
            elif t.dtype == torch.float32:
                out[p] = t.reshape(W, -1).view(torch.int32).numpy().view(np.uint32)
    return out


# ================================================================================================== CPU selfcheck ===
def out_bits(acc, f32):
    return acc.view(np.uint32) if f32 else bf16_rn(acc)


def cpu_selfcheck():
    """(1) shapes.check(): AR_MS, the gather lists and the C2 lists reach every code-path signature of every served shape
    (byte bounds of launcher 166d7aac), and the old 1..24 list does NOT (negative control: the check can fail);
    (2) every device mutant is visible (spec emulation, STORED bits: bf16 after RN) in every gated AR case: bf16 and fp32
    x every M of AR_MS x one/two-shot x every rank (two-shot: inside the rank's own reduced chunk, the only part the
    kernel computes); m_ffma must be invisible (gate-only); emulate_ar (the device test's reference) agrees with the
    chunk-wise construction used here."""
    import time
    t0 = time.time()
    sbad, srep = S.check()
    for line in srep:
        print("SHAPES", line)
    bad = 0
    for b in sbad:
        print("SELFCHECK FAIL: shapes:", b)
        bad += 1
    for f32 in (False, True):  # negative control: the pre-fix list (M = 1..24) must miss served signatures
        if not S.served_ar_sigs(f32) - S.tested_ar_sigs(range(1, 25), f32):
            print(f"SELFCHECK FAIL: shapes check cannot fail (M = 1..24 covers every served {'fp32' if f32 else 'bf16'} "
                  "signature)")
            bad += 1
    g = Gen(7)
    ncase = 0
    orders = ("serial", "rev", "seed0", "negzero", "tree", "ffma")
    for f32 in (False, True):
        epv = 4 if f32 else 8
        for M in AR_MS[f32]:
            n = M * HIDDEN
            v = g.ranks(n, f32, "hostile")
            terms = v.view(np.float32) if f32 else f32_of_bf16(v)
            ob = {o: out_bits(reduce_terms(terms, o), f32) for o in orders}
            nvec = n // epv
            chunk = (nvec + W - 1) // W
            for two in (False, True):
                for rank in range(W):
                    if two:
                        b0, ln = rank * chunk, max(0, min(chunk, nvec - rank * chunk))
                        sl = slice(b0 * epv, (b0 + ln) * epv)
                    else:
                        sl = slice(0, n)
                    ref = ob["serial"][sl]
                    for m in ("rev", "seed0", "negzero", "tree"):
                        if np.array_equal(ob[m][sl], ref):
                            print(f"SELFCHECK FAIL: mutant {m} invisible at f32={f32} M={M} two={two} rank={rank}")
                            bad += 1
                    if not np.array_equal(ob["ffma"][sl], ref):
                        print(f"SELFCHECK FAIL: ffma emulation differs at f32={f32} M={M} two={two} rank={rank}")
                        bad += 1
                    ncase += 1
            # tie-in: the device test's reference (emulate_ar) == the chunk-wise construction above, two ranks
            r2 = np.stack([g.wide(chunk * epv, f32) for _ in range(W)])
            for rank in (0, W - 1):
                for two in (False, True):
                    e = emulate_ar(v, rank, two, f32, r2)
                    want = ob["serial"].copy()
                    if two:
                        for r in range(W):
                            if r != rank:
                                lr = max(0, min(chunk, nvec - r * chunk))
                                want[r * chunk * epv:(r * chunk + lr) * epv] = r2[r][:lr * epv]
                    if not np.array_equal(e, want):
                        print(f"SELFCHECK FAIL: emulate_ar != chunk-wise reference at f32={f32} M={M} two={two} rank={rank}")
                        bad += 1
    nm = f"bf16 {len(AR_MS[False])} M (max {max(AR_MS[False])}), fp32 {len(AR_MS[True])} M (max {max(AR_MS[True])})"
    print(f"CPU_SELFCHECK {'PASS' if not bad else 'FAIL'}: shapes cover every served signature; 4 device mutants visible "
          f"in every case ({ncase} cases: {nm} x 2 modes x 8 ranks), ffma invisible ({bad} problems, "
          f"{time.time() - t0:.0f} s)")
    return 0 if not bad else 1


# ======================================================================================================= preflight ===
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def kit_shas():
    k = json.load(open(os.path.join(HERE, "KIT.json")))
    so = {f["path"]: f.get("sha256") for f in k["files"]}
    return so.get("$HOME/dsv41-stage/r7comm2/build/rdma_ar_ext.so"), k.get("mutant_so_sha256", {})


def launcher_default(var, txt):
    """the launcher's `export VAR=${VAR:-default}` default, or None"""
    m = re.search(r"export\s+[^\n]*\b" + var + r"=\$\{" + var + r":-([^}]*)\}", txt)
    return m.group(1) if m else None


def check_glue_env(env, launcher_txt):
    """review B2: problems (empty = OK) with the production glue env. env = the process environment; launcher_txt =
    the node's ~/dsv41-tp8.sh text (None if unreadable: then the launcher half of the check fails too)."""
    bad = []
    for k, v in PROD_GLUE_ENV:
        if env.get(k) != v:
            bad.append(f"{k}={env.get(k)!r} in the environment, production runs {k}={v} (launcher 4f3cc1c6): "
                       "the device test must mirror the production env")
        if launcher_txt is None:
            bad.append(f"{LAUNCHER} unreadable: cannot confirm the launcher still defaults {k}={v}")
        elif launcher_default(k, launcher_txt) != v:
            bad.append(f"{LAUNCHER} defaults {k} to {launcher_default(k, launcher_txt)!r}, not {v!r}: the kit's "
                       "production env is stale (update PROD_GLUE_ENV after re-reading the launcher)")
    return bad


def launcher_rdma_defaults():
    """(pub mask, vec_per_cta) the launcher's defaults give, or (None, None)."""
    try:
        txt = open(LAUNCHER).read()
    except OSError:
        return None, None
    def dflt(var):
        return launcher_default(var, txt)
    pub = 0
    for var, bit in (("RDMA_AR_PUB_V1", 1), ("RDMA_AR_PUB_V2", 2), ("RDMA_AR_HYG_POLL", 4)):
        v = dflt(var)  # the launcher default (a window arm never overrides these)
        if v is not None and v.strip() not in ("", "0"):
            pub |= bit
    vpc = dflt("RDMA_AR_VEC_PER_CTA")
    return pub, int(vpc) if vpc and vpc.isdigit() else 512


def preflight(a):
    bad = []
    if "venvs/vllm032" not in sys.prefix:
        bad.append(f"python is not the vllm032 venv (sys.prefix={sys.prefix})")
    for t in ("nvcc", "cuobjdump", "ninja"):
        if not shutil.which(t):
            bad.append(f"{t} not on PATH")
    ch = os.environ.get("CUDA_HOME", "")
    if not ch or not os.path.isdir(ch):
        bad.append(f"CUDA_HOME not set to an existing dir (got {ch!r})")
    try:
        procs = subprocess.run(["pgrep", "-af", "vllm.entrypoints|EngineCore|VLLM::|dsv41-tp8"], capture_output=True,
                               text=True, timeout=10).stdout.splitlines()
        procs = [p for p in procs if "pgrep" not in p and not p.startswith(str(os.getpid()) + " ")]
        if procs:
            bad.append("serving processes are running (STOPPED node only): " + "; ".join(procs[:3]))
    except Exception as e:  # noqa: BLE001
        bad.append(f"pgrep failed: {e}")
    if not os.path.exists(a.live) or sha256(a.live) != LIVE_SHA:
        bad.append(f"{a.live}: not the live .so {LIVE_SHA[:16]}")
    want_so, want_mut = kit_shas()
    if not want_so:
        bad.append("KIT.json records no r6 .so sha256 (build it first: ops/build8-r6.sh r6)")
    elif not os.path.exists(a.cand) or sha256(a.cand) != want_so:
        bad.append(f"{a.cand}: sha256 != KIT.json {str(want_so)[:16]}")
    if a.part in ("mutants", "c2", "all"):
        need = (DEVICE_MUTANTS + GATE_ONLY_MUTANTS if a.part in ("mutants", "all") else []) + \
            ([PDL_MUTANT] if a.part in ("c2", "all") else [])
        for m in need:
            p = os.path.join(a.mutant_dir, f"{m}.so")
            if not os.path.exists(p) or not want_mut.get(m) or sha256(p) != want_mut[m]:
                bad.append(f"{p}: missing or sha256 != KIT.json mutant_so_sha256[{m}]")
    pub, vpc = launcher_rdma_defaults()
    if pub != 7:
        bad.append(f"launcher RDMA defaults give pub mask {pub}, not 7 (V1+V2+HYG_POLL): the reference would not be "
                   "the serving kernel")
    a.vpc = vpc or 512
    try:
        ltxt = open(LAUNCHER).read()
    except OSError:
        ltxt = None
    bad += check_glue_env(os.environ, ltxt)
    if bad:
        bad.append(f"run every device command as:  {ENV_PREFIX}; <cmd>")
    return bad


# ============================================================================================== GPU side helpers ===
def load_ext(path, tag):
    import importlib.machinery
    import importlib.util
    name = f"{tag}.rdma_ar_ext"  # the init symbol comes from the last component (PyInit_rdma_ar_ext)
    loader = importlib.machinery.ExtensionFileLoader(name, path)
    spec = importlib.util.spec_from_file_location(name, path, loader=loader)
    m = importlib.util.module_from_spec(spec)
    loader.exec_module(m)
    return m


KNAME = re.compile(r"(ar_kernel_pl|ar_kernel_r3|ar_kernel)<(\d), (false|true), 8(?:, (\d+))?>")


def kernels_of(fn):
    """demangled kernel names (mode, f32, mask) launched by fn(), from the CUDA profiler (fail closed if none)."""
    import torch
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    out = []
    for e in prof.events():
        m = KNAME.search(e.name or "")
        if m:
            out.append((m.group(1), int(m.group(2)), m.group(3) == "true", int(m.group(4) or -1)))
    return out


class Rig:
    """one SelfTest per module, same rank; helpers that run one op with identical images"""

    def __init__(self, mod, rank, tag):
        self.t = mod.SelfTest(rank, W, MAX_BYTES, MAX_BYTES, 0)
        self.rank, self.tag, self.mod = rank, tag, mod
        self.mc = self.t.max_chunk()

    def mode(self, r3, s1=24576, r6=0):
        if hasattr(self.t, "set_r3"):  # the live g130 SelfTest has set_pub only
            self.t.set_r3(0, 24576)
        if hasattr(self.t, "set_r6"):
            self.t.set_r6(0)
        self.t.set_pub(7)
        if r3:
            self.t.set_r3(1, s1)
            if r6:
                self.t.set_r6(r6)


def tensor_of(bits, f32, dev):
    import torch
    t = torch.from_numpy(bits.view(np.int32 if f32 else np.int16).copy())
    return t.view(torch.float32 if f32 else torch.bfloat16).to(dev)


def bytes_np(t):
    import torch
    return t.contiguous().view(torch.uint8).reshape(-1).cpu().numpy()


# ================================================================================================= C1 selftest8 ===
class Tally:
    def __init__(self):
        self.checks = self.fails = 0
        self.notes = []

    def check(self, ok, what):
        self.checks += 1
        if not ok:
            self.fails += 1
            print("FAIL", what, flush=True)
        return ok


def first_diff(a, b):
    d = np.nonzero(a != b)[0]
    return int(d[0]) if len(d) else -1


def ar_images(v, rank, two, f32, g):
    """recv image ([8, w] uint8: one-shot = every rank's whole input, two-shot = every peer's chunk `rank` at offset 0),
    recv2 image ([8, max_chunk] uint8) and the recv2 bits for the emulator"""
    import torch  # noqa: F401
    epv = 4 if f32 else 8
    n = v.shape[1]
    nvec = n // epv
    chunk = (nvec + W - 1) // W
    if two:
        b, ln = rank * chunk, max(0, min(chunk, nvec - rank * chunk))
        img = np.zeros((W, max(1, chunk) * 16), np.uint8)
        for r in range(W):
            seg = v[r, b * epv:(b + ln) * epv]
            img[r, :seg.nbytes] = seg.view(np.uint8)
    else:
        img = np.ascontiguousarray(v).view(np.uint8).reshape(W, -1)
    r2bits = np.stack([g.wide(chunk * epv, f32) for _ in range(W)])
    return img, r2bits


def run_c1(a, live, cand, recorded):
    import torch
    dev = torch.device("cuda", 0)
    T = Tally()
    ranks = [int(x) for x in a.ranks.split(",")]
    s1s = [24576] if a.quick else [24576, 0, 1 << 30]
    r6s = [0, 128, 384] if hasattr(cand.SelfTest, "set_r6") else [0]
    ar_ms = QUICK_AR_MS if a.quick else AR_MS
    emu_bad = 0
    for rank in ranks:
        L, C = Rig(live, rank, "live"), Rig(cand, rank, "cand")
        g = Gen(a.seed * 31 + rank)
        for f32 in (False, True):
            es = 4 if f32 else 2
            cases = [(M, fam) for M in ar_ms[f32] for fam in ("rand", "hostile")]
            if recorded and not f32:  # every recorded M (embed rows: AR_MS; --recorded: any M <= 204)
                cases += [(M, "recorded") for M in sorted(recorded) if not a.quick or M in ar_ms[f32]]
            for M, fam in cases:
                n = M * HIDDEN
                for two in (False, True):
                    for call in range(2):
                        if fam == "recorded":
                            v = recorded.get(M)
                            if v is None:
                                continue
                        else:
                            v = g.ranks(n, f32, fam)
                        img, r2 = ar_images(v, rank, two, f32, g)
                        for R in (L, C):
                            R.t.set_recv(torch.from_numpy(img))
                            R.t.set_recv2(torch.from_numpy(np.ascontiguousarray(r2).view(np.uint8).reshape(W, -1)))
                        x = tensor_of(v[rank], f32, dev)
                        L.mode(False)
                        ref = torch.empty_like(x)
                        L.t.all_reduce_into(ref, x, two, True, a.vpc, False)
                        rb = bytes_np(ref)
                        emu = emulate_ar(v, rank, two, f32, r2)
                        if not np.array_equal(emu.view(np.uint8), rb):
                            emu_bad += 1
                            if emu_bad <= 5:
                                print(f"NOTE spec emulation != live kernel: rank {rank} f32 {f32} M {M} {fam} two {two}"
                                      f" first byte {first_diff(emu.view(np.uint8), rb)}", flush=True)
                        for s1 in s1s:
                            for r6 in r6s:
                                C.mode(True, s1, r6)
                                seq0 = C.t.counters()[0]
                                out = torch.empty_like(x)
                                C.t.all_reduce_into(out, x, two, True, a.vpc, call == 1 and s1 == 24576 and r6 == 0)
                                ob = bytes_np(out)
                                c = C.t.counters()
                                tag = (f"rank {rank} {'f32' if f32 else 'bf16'} M {M} {fam} two {two} s1 {s1} r6 {r6} "
                                       f"call {call}")
                                T.check(np.array_equal(ob, rb), f"{tag}: candidate != live (first byte {first_diff(ob, rb)})")
                                T.check(c[0] == seq0 + 1 and list(c[1:]) == [0, 0, 0, 0], f"{tag}: counters {c} (seq before {seq0})")
                                slot = (seq0 + 1) % 4
                                T.check(np.array_equal(C.t.peek_send(0, slot, n * es).numpy(), bytes_np(x)),
                                        f"{tag}: send slot != input")
                                if two:
                                    epv = 16 // es
                                    nvec = n // epv
                                    chunk = (nvec + W - 1) // W
                                    b, ln = rank * chunk, max(0, min(chunk, nvec - rank * chunk))
                                    T.check(np.array_equal(C.t.peek_send(1, slot, ln * 16).numpy(),
                                                           ob[b * 16:(b + ln) * 16]), f"{tag}: send2 slot != my chunk")
        # gathers: copies, incl. -0.0 and NaN payloads; rows per seg from the 2 MiB gather bound (shapes.gather_ms)
        for segb in S.SEGS:
            gms = S.gather_ms(segb)
            if a.quick:
                gms = sorted({1, 6, 24} | set(S.cover(gms, lambda m: S.gather_sig(m * segb, "ag"))))
            for M in gms:
                cols = segb // 2
                for call in range(2):
                    v = np.stack([g.wide(M * cols, False) for _ in range(W)])
                    v[:, 0] = 0x8000
                    v[:, 1 % (M * cols)] = 0x7FC3
                    v[:, -1] = 0xFFA9
                    img = np.ascontiguousarray(v).view(np.uint8).reshape(W, -1)
                    x = tensor_of(v[rank], False, dev).view(M, cols)
                    outs = {}
                    for R, r3, r6 in ((L, False, 0), (C, True, 0), (C, True, 128), (C, True, 384)):
                        if r6 and not hasattr(C.t, "set_r6"):
                            continue
                        R.t.set_recv(torch.from_numpy(img))
                        R.mode(r3, 24576, r6)
                        o1 = torch.empty(W, M, cols, dtype=torch.bfloat16, device=dev)
                        o2 = torch.empty(M, W * cols, dtype=torch.bfloat16, device=dev)
                        # counters() is a test-only SelfTest method absent from the pinned live
                        # g130 .so (14d049dd); its reads are consumed only under `if R is C:` below,
                        # so guard them to the candidate rig (matches the set_r6 hasattr guard at :670).
                        seq0 = R.t.counters()[0] if R is C else 0
                        R.t.all_gather_into(o1, x, True, a.vpc, False)
                        R.t.all_gather_cols_into(o2, x, segb, True, a.vpc, False)
                        c = R.t.counters() if R is C else None
                        if R is C:
                            T.check(c[0] == seq0 + 2 and list(c[1:]) == [0, 0, 0, 0],
                                    f"gather rank {rank} M {M} seg {segb} r6 {r6}: counters {c}")
                        outs[(R.tag, r6)] = (bytes_np(o1), bytes_np(o2))
                    want1 = np.ascontiguousarray(v.reshape(W, M, cols)).view(np.uint8).reshape(-1)
                    want2 = np.ascontiguousarray(v.reshape(W, M, cols).transpose(1, 0, 2)).view(np.uint8).reshape(-1)
                    T.check(np.array_equal(outs[("live", 0)][0], want1) and np.array_equal(outs[("live", 0)][1], want2),
                            f"gather rank {rank} M {M} seg {segb}: LIVE != concatenation (harness error)")
                    for k_, (b1, b2) in outs.items():
                        if k_[0] == "cand":
                            T.check(np.array_equal(b1, outs[("live", 0)][0]) and np.array_equal(b2, outs[("live", 0)][1]),
                                    f"gather rank {rank} M {M} seg {segb} r6 {k_[1]}: candidate != live")
        # captured mixed graph, candidate kernels, replayed with fresh inputs; == eager live
        for r6 in r6s:
            graph_check(T, L, C, rank, r6, g, dev, a.vpc)
        del L, C
    T.notes.append(f"spec emulation disagreed with the live kernel in {emu_bad} AR cases")
    if emu_bad:
        T.check(False, f"spec emulation != live kernel in {emu_bad} cases (harness error: fix the emulator first)")
    return T


def graph_check(T, L, C, rank, r6, g, dev, vpc):
    import torch
    # decode shapes plus the large served ones (review B1): M = 39 (4x-unrolled staging), 103 and 204 (multi-iteration
    # two-shot consumer + phase 2), and a 2 MiB dim-0 gather (the RDMA_AR_MAX_BYTES bound)
    seqs = [("ar", 2 * HIDDEN, False), ("ar", 6 * HIDDEN, True), ("ag", 6 * QKV, None), ("agc", 6 * QKV, None),
            ("ar", 4 * HIDDEN, True), ("ar", 1 * HIDDEN, False), ("ar", 24 * HIDDEN, True), ("ar", 5 * HIDDEN, True),
            ("ar", 39 * HIDDEN, True), ("ar", 103 * HIDDEN, True), ("ag", MAX_BYTES // 2, None),
            ("ar", 204 * HIDDEN, True)]
    img = np.stack([g.wide(MAX_BYTES // 2, False) for _ in range(W)]).view(np.uint8)
    r2 = np.stack([g.wide(L.mc // 2, False) for _ in range(W)]).view(np.uint8)
    for R in (L, C):
        R.t.set_recv(torch.from_numpy(img))
        R.t.set_recv2(torch.from_numpy(r2))
    xs = [torch.zeros(n, dtype=torch.bfloat16, device=dev) for _, n, _ in seqs]
    outs = [torch.empty(n * (W if k != "ar" else 1), dtype=torch.bfloat16, device=dev) for k, n, _ in seqs]

    def run_all(R):
        for (k, n, two), x, o in zip(seqs, xs, outs):
            if k == "ar":
                R.t.all_reduce_into(o, x, two, True, vpc, False)
            elif k == "ag":
                R.t.all_gather_into(o.view(W, n), x, True, vpc, False)
            else:
                R.t.all_gather_cols_into(o.view(6, W * QKV), x.view(6, QKV), 2 * QKV, True, vpc, False)

    C.mode(True, 24576, r6)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        run_all(C)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        run_all(C)
    torch.cuda.synchronize()
    for rep in range(20):
        new = [tensor_of(g.ranks(x.numel(), False, "hostile" if rep % 2 else "rand")[rank], False, dev) for x in xs]
        for x, nx in zip(xs, new):
            x.copy_(nx)
        seq0 = C.t.counters()[0]
        gr.replay()
        torch.cuda.synchronize()
        c = C.t.counters()
        got = [bytes_np(o).copy() for o in outs]
        L.mode(False)
        run_all(L)
        torch.cuda.synchronize()
        for i, o in enumerate(outs):
            T.check(np.array_equal(got[i], bytes_np(o)), f"graph rank {rank} r6 {r6} replay {rep} op {i}: != live")
        T.check(c[0] == seq0 + len(seqs) and list(c[1:]) == [0, 0, 0, 0], f"graph rank {rank} r6 {r6} replay {rep}: counters {c}")
        C.mode(True, 24576, r6)
    del gr


# ============================================================================================ mutants (device) ===
def run_mutants(a, live):
    import torch
    dev = torch.device("cuda", 0)
    res = {}
    ar_ms = QUICK_AR_MS if a.quick else AR_MS  # every M up to 204 (bf16) / 102 (fp32): review B1
    for m in DEVICE_MUTANTS + GATE_ONLY_MUTANTS:
        mod = load_ext(os.path.join(a.mutant_dir, f"{m}.so"), f"mut_{m}")
        if getattr(mod, "R6_MUTANT", None) != m:
            raise SystemExit(f"PREFLIGHT FAIL: {m}.so does not carry R6_MUTANT={m}")
        cases = caught = 0
        missed = []
        for rank in (0, 5):
            L, X = Rig(live, rank, "live"), Rig(mod, rank, m)
            g = Gen(a.seed * 77 + rank)
            for f32 in (False, True):
                for M in ar_ms[f32]:
                    n = M * HIDDEN
                    v = g.ranks(n, f32, "hostile")
                    for two in (False, True):
                        img, r2 = ar_images(v, rank, two, f32, g)
                        for R in (L, X):
                            R.t.set_recv(torch.from_numpy(img))
                            R.t.set_recv2(torch.from_numpy(np.ascontiguousarray(r2).view(np.uint8).reshape(W, -1)))
                        x = tensor_of(v[rank], f32, dev)
                        L.mode(False)
                        ref = torch.empty_like(x)
                        L.t.all_reduce_into(ref, x, two, True, a.vpc, False)
                        for r6 in (0, 128):
                            X.mode(True, 24576, r6)
                            out = torch.empty_like(x)
                            X.t.all_reduce_into(out, x, two, True, a.vpc, False)
                            cases += 1
                            if not np.array_equal(bytes_np(out), bytes_np(ref)):
                                caught += 1
                            else:
                                missed.append(f"rank {rank} f32 {f32} M {M} two {two} r6 {r6}")
            del L, X
        res[m] = (caught, cases, missed)
        exp = "every case" if m in DEVICE_MUTANTS else "none (gate-only)"
        print(f"MUTANT {m}: caught {caught}/{cases} (expected: {exp})" +
              (f"; first missed: {missed[0]}" if missed and m in DEVICE_MUTANTS else ""), flush=True)
    ok = all(res[m][0] == res[m][1] and res[m][1] > 0 for m in DEVICE_MUTANTS)
    return ok, res


# ===================================================================================================== C2 (PDL) ===
HELPERS_CU = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>
__device__ __forceinline__ unsigned long long gt() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__global__ void poison_k(uint32_t* x, long long n, uint32_t pat) {
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) x[i] = pat;
}
__global__ void slow_producer_k(uint32_t* x, const uint32_t* src, long long n, long long spin_ns) {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // dependents may launch NOW (before x is written)
  const unsigned long long t0 = gt();
  while (gt() - t0 < (unsigned long long)spin_ns) {}
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) x[i] = src[i];
}
__global__ void pdl_consumer_k(uint32_t* dst, const uint32_t* src, long long n, int wait) {
  if (wait) asm volatile("griddepcontrol.wait;" ::: "memory");
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) dst[i] = src[i];
}
void poison(torch::Tensor x, int64_t pat) {
  poison_k<<<8, 256, 0, at::cuda::getCurrentCUDAStream()>>>((uint32_t*)x.data_ptr(), x.numel() * x.element_size() / 4, (uint32_t)pat);
}
void slow_producer(torch::Tensor x, torch::Tensor src, int64_t spin_ns) {
  slow_producer_k<<<4, 256, 0, at::cuda::getCurrentCUDAStream()>>>((uint32_t*)x.data_ptr(), (const uint32_t*)src.data_ptr(),
      x.numel() * x.element_size() / 4, spin_ns);
}
void pdl_consumer(torch::Tensor dst, torch::Tensor src, bool wait) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(8); cfg.blockDim = dim3(256); cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at_[1];
  at_[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at_[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at_; cfg.numAttrs = 1;
  TORCH_CHECK(cudaLaunchKernelEx(&cfg, pdl_consumer_k, (uint32_t*)dst.data_ptr(), (const uint32_t*)src.data_ptr(),
      (long long)(dst.numel() * dst.element_size() / 4), wait ? 1 : 0) == cudaSuccess, "pdl_consumer launch");
}
"""


def helpers():
    from torch.utils.cpp_extension import load_inline
    return load_inline(name="r6pdl_helpers", cpp_sources="void poison(torch::Tensor x, int64_t pat);\n"
                       "void slow_producer(torch::Tensor x, torch::Tensor src, int64_t spin_ns);\n"
                       "void pdl_consumer(torch::Tensor dst, torch::Tensor src, bool wait);",
                       cuda_sources=HELPERS_CU, functions=["poison", "slow_producer", "pdl_consumer"],
                       extra_cuda_cflags=["-gencode=arch=compute_121,code=sm_121", "-O2"], verbose=False)


def c2_cases(quick=False):
    """(kind, f32, M, seg bytes, two) of the C2 poison / m_waitlate replay (review B1 + N5): AR bf16 shapes.c2_ar_ms
    (1..24 + {38, 39, 89, 90, 102, 103, 204} + a cover of every served signature), AR fp32 a cover up to 102, and dim-0 /
    column gathers [M, 224] covering every served gather signature up to 2 MiB; ARs one- and two-shot."""
    cs = []
    for f32 in (False, True):
        ms = S.c2_ar_ms(f32)
        if quick:
            ms = [m for m in ms if m in (1, 6, 24, 39, 103, 204)] if not f32 else [1, 52, 102]
        cs += [("ar", f32, M, 0, two) for M in ms for two in (False, True)]
    for kind in ("ag", "agc"):
        for segb in S.SEGS_C2:
            gm = S.c2_gather_ms(segb)
            cs += [(kind, False, M, segb, None) for M in ((min(gm), max(gm)) if quick else gm)]
    return cs


def ar_edge_types(gr):
    sys.path.insert(0, os.path.join(HERE, "tests", "device"))
    import graph_edges  # noqa: E402
    el = graph_edges.edges(gr.raw_cuda_graph())
    into_ar = [e for e in graph_edges.incoming(el, lambda n: "ar_kernel_r3" in n) if e[1]]  # kernel -> AR edges
    out_ar = [e for e in el if e[1] and "ar_kernel_r3" in e[0] and e[3]]
    return into_ar, out_ar


def run_c2(a, live, cand, mut):
    import torch
    dev = torch.device("cuda", 0)
    T = Tally()
    H = helpers()
    POISON = 0x7FA17FA1  # bf16 NaN pair / fp32 NaN with a payload: never a real input value
    mres = {"cases": 0, "caught": 0, "missed": []}
    for rank in (0, 5):
        L, C = Rig(live, rank, "live"), Rig(cand, rank, "cand")
        X = Rig(mut, rank, PDL_MUTANT) if mut is not None else None
        g = Gen(a.seed * 13 + rank)
        for kind, f32, M, segb, two in c2_cases(a.quick):
            if kind == "ar":
                n, es = M * HIDDEN, (4 if f32 else 2)
                shape, ishape = (n,), (n,)
            else:  # dim-0 / column gather of a bf16 [M, segb / 2] input (review N5)
                cols = segb // 2
                n, es = M * cols, 2
                shape = (W, M, cols) if kind == "ag" else (M, W * cols)
                ishape = (M, cols)
            dt = torch.float32 if f32 else torch.bfloat16
            for bits in (0, 128, 384):
                if kind == "ar":
                    v = g.ranks(n, f32, "hostile")
                    img, r2 = ar_images(v, rank, two, f32, g)
                else:
                    v = np.stack([g.wide(n, False) for _ in range(W)])
                    img, r2 = np.ascontiguousarray(v).view(np.uint8).reshape(W, -1), None
                for R in [L, C] + ([X] if X else []):
                    R.t.set_recv(torch.from_numpy(img))
                    if r2 is not None:
                        R.t.set_recv2(torch.from_numpy(np.ascontiguousarray(r2).view(np.uint8).reshape(W, -1)))

                def op(R_, out_, x_):
                    if kind == "ar":
                        R_.t.all_reduce_into(out_, x_, two, True, a.vpc, False)
                    elif kind == "ag":
                        R_.t.all_gather_into(out_, x_, True, a.vpc, False)
                    else:
                        R_.t.all_gather_cols_into(out_, x_, segb, True, a.vpc, False)

                for who, R in (("cand", C), ("mutant", X), ("ctl-noproducer", C)):
                    if R is None or (who != "cand" and bits == 0):
                        continue
                    x = torch.empty(ishape, dtype=dt, device=dev)
                    src = torch.empty_like(x)
                    out = torch.empty(shape, dtype=dt, device=dev)
                    R.mode(True, 24576, bits)

                    def body():
                        H.poison(x, POISON)
                        if who != "ctl-noproducer":
                            H.slow_producer(x, src, a.spin_ns)
                        op(R, out, x)

                    src.copy_(tensor_of(v[rank], f32, dev).view(ishape))
                    s = torch.cuda.Stream()
                    s.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(s):
                        body()
                    torch.cuda.current_stream().wait_stream(s)
                    torch.cuda.synchronize()
                    gr = torch.cuda.CUDAGraph(keep_graph=True)
                    with torch.cuda.graph(gr):
                        body()
                    into, _ = ar_edge_types(gr)
                    tag = (f"{who} rank {rank} {kind} {'f32' if f32 else 'bf16'} M {M}" +
                           (f" seg {segb}" if kind != "ar" else f" two {two}") + f" bits {bits}")
                    want_t = 1 if bits else 0
                    if who != "ctl-noproducer":
                        T.check(len(into) == 1 and into[0][4] == want_t,
                                f"{tag}: edge into the AR {[(e[0][:30], e[4]) for e in into]} (want type {want_t})")
                    bad_reps = 0
                    for rep in range(a.replays):
                        xr = g.row(n, f32, "hostile" if (rep % 2 and kind == "ar") else "rand", rank)
                        xf = tensor_of(xr, f32, dev).view(ishape)
                        src.copy_(xf)
                        seq0 = R.t.counters()[0]
                        gr.replay()
                        torch.cuda.synchronize()
                        c = R.t.counters()
                        slot = (seq0 + 1) % 4
                        staged = R.t.peek_send(0, slot, n * es).numpy()
                        ob = bytes_np(out).copy()
                        L.mode(False)
                        ref = torch.empty(shape, dtype=dt, device=dev)
                        op(L, ref, xf)
                        ok = (np.array_equal(staged, bytes_np(xf)) and np.array_equal(ob, bytes_np(ref)) and
                              c[0] == seq0 + 1 and list(c[1:]) == [0, 0, 0, 0])
                        bad_reps += not ok
                        R.mode(True, 24576, bits)
                    if who == "cand":
                        T.check(bad_reps == 0, f"{tag}: {bad_reps}/{a.replays} replays wrong (staged slot / output / counters)")
                    elif who == "ctl-noproducer":
                        T.check(bad_reps == a.replays, f"{tag}: control not flagged in {a.replays - bad_reps} replays "
                                "(the comparator must see the poison)")
                    else:
                        mres["cases"] += 1
                        if bad_reps:
                            mres["caught"] += 1
                        else:
                            mres["missed"].append(tag)
                    del gr
        # back-to-back PDL ARs and the trig consumer
        for bits in (128, 384):
            for M in ((1, 6, 24, 103) if a.quick else (1, 4, 6, 24, 103, 204)):  # incl. the large served shapes (B1)
                n = M * HIDDEN
                two = n * 2 >= 32768
                v = g.ranks(n, False, "rand")
                img, r2 = ar_images(v, rank, two, False, g)
                for R in (L, C):
                    R.t.set_recv(torch.from_numpy(img))
                    R.t.set_recv2(torch.from_numpy(np.ascontiguousarray(r2).view(np.uint8).reshape(W, -1)))
                x1, x2 = (torch.empty(n, dtype=torch.bfloat16, device=dev) for _ in range(2))
                src = torch.empty_like(x1)
                o1, o2, cp = (torch.empty_like(x1) for _ in range(3))
                C.mode(True, 24576, bits)

                def body2():
                    H.slow_producer(x1, src, a.spin_ns)
                    C.t.all_reduce_into(o1, x1, two, True, a.vpc, False)
                    C.t.all_reduce_into(o2, o1, two, True, a.vpc, False)  # AR -> AR, the second reads the first's output
                    H.pdl_consumer(cp, o2, True)

                src.copy_(tensor_of(v[rank], False, dev))
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    body2()
                torch.cuda.current_stream().wait_stream(s)
                torch.cuda.synchronize()
                gr = torch.cuda.CUDAGraph(keep_graph=True)
                with torch.cuda.graph(gr):
                    body2()
                into, outof = ar_edge_types(gr)
                tag = f"b2b+consumer rank {rank} M {M} bits {bits}"
                T.check(len(into) == 2 and all(e[4] == 1 for e in into), f"{tag}: edges into ARs {[e[4] for e in into]}")
                cons = [e for e in outof if "pdl_consumer" in e[2]]
                T.check(len(cons) == 1 and cons[0][4] == 1, f"{tag}: AR -> PDL consumer edge {[e[4] for e in cons]}")
                for rep in range(a.replays):
                    vv = g.ranks(n, False, "rand")
                    src.copy_(tensor_of(vv[rank], False, dev))
                    seq0 = C.t.counters()[0]
                    gr.replay()
                    torch.cuda.synchronize()
                    c = C.t.counters()
                    got1, got2, gotc = bytes_np(o1).copy(), bytes_np(o2).copy(), bytes_np(cp).copy()
                    L.mode(False)
                    r1, r2_ = torch.empty_like(x1), torch.empty_like(x1)
                    xf = tensor_of(vv[rank], False, dev)
                    L.t.all_reduce_into(r1, xf, two, True, a.vpc, False)
                    L.t.all_reduce_into(r2_, r1, two, True, a.vpc, False)
                    T.check(np.array_equal(got1, bytes_np(r1)) and np.array_equal(got2, bytes_np(r2_)) and
                            np.array_equal(gotc, got2) and c[0] == seq0 + 2 and list(c[1:]) == [0, 0, 0, 0],
                            f"{tag} replay {rep}: outputs / consumer copy / counters")
                    C.mode(True, 24576, bits)
                del gr
        del L, C, X
    mok = mut is None or (mres["cases"] > 0 and mres["caught"] == mres["cases"])
    print(f"MUTANT {PDL_MUTANT}: caught {mres['caught']}/{mres['cases']} cases (expected: every case)" +
          (f"; first missed: {mres['missed'][0]}" if mres["missed"] else ""), flush=True)
    return T, mok, mres


# ============================================================================================================ main ===
def assert_serving_op(live, cand):
    import torch
    dev = torch.device("cuda", 0)
    L, C = Rig(live, 0, "live"), Rig(cand, 0, "cand")
    x1 = torch.randn(2 * HIDDEN, device=dev).to(torch.bfloat16)
    x2 = torch.randn(6 * HIDDEN, device=dev).to(torch.bfloat16)
    probs = []
    L.mode(False)
    ks = kernels_of(lambda: (L.t.all_reduce_into(torch.empty_like(x1), x1, False, True, 512, False),
                             L.t.all_reduce_into(torch.empty_like(x2), x2, True, True, 512, False)))
    if ("ar_kernel_pl", 0, False, 5) not in ks or ("ar_kernel_pl", 2, False, 7) not in ks or \
            any(k[0] != "ar_kernel_pl" for k in ks):
        probs.append(f"live reference launched {ks}, expected ar_kernel_pl<0, false, 8, 5> and <2, false, 8, 7>")
    for bits in ((0, 128, 384) if hasattr(C.t, "set_r6") else (0,)):
        C.mode(True, 24576, bits)
        ks = kernels_of(lambda: C.t.all_reduce_into(torch.empty_like(x2), x2, True, True, 512, False))
        if len(ks) != 1 or ks[0][0] != "ar_kernel_r3" or (ks[0][3] & 384) != bits or not ks[0][3] & 8:
            probs.append(f"candidate (r6 bits {bits}) launched {ks}")
    if not hasattr(C.t, "set_r6"):
        probs.append("candidate .so has no SelfTest.set_r6 (not the r6 build)")
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["c1", "mutants", "c2", "all"])
    ap.add_argument("--cand", default=CAND_SO)
    ap.add_argument("--live", default=LIVE_SO)
    ap.add_argument("--mutant-dir", default=os.path.join(HERE, "prebuilt", "mutants"))
    ap.add_argument("--ranks", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--recorded", default=None)
    ap.add_argument("--replays", type=int, default=8)
    ap.add_argument("--spin-ns", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20261109)
    ap.add_argument("--quick", action="store_true", help="smoke: fewer M / S1 values (never prints PASS)")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--cpu-selfcheck", action="store_true")
    a = ap.parse_args()
    if a.cpu_selfcheck:
        return cpu_selfcheck()
    bad = preflight(a)
    if bad:
        for b in bad:
            print("PREFLIGHT FAIL:", b)
        print("R6_DEVICE_TEST REFUSED")
        return 2
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1):
        print("PREFLIGHT FAIL: no GB10 (sm_121) device")
        print("R6_DEVICE_TEST REFUSED")
        return 2
    live = load_ext(a.live, "live")
    cand = load_ext(a.cand, "cand")
    if getattr(live, "R3_VARIANT", None) is not None or getattr(cand, "R6_VARIANT", None) is None:
        print("PREFLIGHT FAIL: live must be the g130 .so and the candidate the r6 .so")
        print("R6_DEVICE_TEST REFUSED")
        return 2
    probs = assert_serving_op(live, cand)
    if probs:
        for p in probs:
            print("SERVING-OP FAIL:", p)
        print("R6_DEVICE_TEST REFUSED")
        return 2
    print("SERVING-OP OK: reference = live ar_kernel_pl<0,false,8,5> / <2,false,8,7> (14d049dd, launcher mask 7, "
          f"{a.vpc} vec/CTA); candidate = ar_kernel_r3 masks with PDL bits 0/128/384", flush=True)
    verdict, incomplete = True, []
    summary = {}
    if a.part in ("c1", "all"):
        rec = None
        if a.recorded:
            rec_d = recorded_dir(a.recorded)
            rec = {}
            for k, v in rec_d.items():
                if v.dtype == np.uint16 and v.shape[1] % HIDDEN == 0 and 1 <= v.shape[1] // HIDDEN <= max(AR_MS[False]):
                    rec[v.shape[1] // HIDDEN] = v
        if not rec:
            rec = recorded_embed(a.model, AR_MS[False], a.seed)
        if not rec:
            incomplete.append("c1: no recorded inputs (no --recorded dir and no embed.weight under --model)")
        T = run_c1(a, live, cand, rec)
        print(f"C1 selftest8: {T.checks} checks, {T.fails} failures; " + "; ".join(T.notes), flush=True)
        summary["c1"] = (T.checks, T.fails)
        verdict &= T.fails == 0
    if a.part in ("mutants", "all"):
        ok, res = run_mutants(a, live)
        summary["mutants"] = {k: v[:2] for k, v in res.items()}
        verdict &= ok
    if a.part in ("c2", "all"):
        mut = load_ext(os.path.join(a.mutant_dir, f"{PDL_MUTANT}.so"), "mut_waitlate")
        if getattr(mut, "R6_MUTANT", None) != PDL_MUTANT:
            print(f"PREFLIGHT FAIL: {PDL_MUTANT}.so does not carry R6_MUTANT")
            return 2
        T2, mok, mres = run_c2(a, live, cand, mut)
        print(f"C2 PDL: {T2.checks} checks, {T2.fails} failures; m_waitlate caught {mres['caught']}/{mres['cases']}", flush=True)
        summary["c2"] = (T2.checks, T2.fails, mres["caught"], mres["cases"])
        verdict &= T2.fails == 0 and mok
    if a.time:
        timing(a, live, cand)
    print("SUMMARY", json.dumps(summary), flush=True)
    if not verdict:
        print("R6_DEVICE_TEST FAIL")
        return 1
    if incomplete or a.quick:
        for i in incomplete:
            print("INCOMPLETE:", i)
        print("R6_DEVICE_TEST INCOMPLETE" + (" (quick smoke)" if a.quick else ""))
        return 3
    print("R6_DEVICE_TEST PASS")
    return 0


def timing(a, live, cand):
    import statistics
    import torch
    dev = torch.device("cuda", 0)
    L, C = Rig(live, 0, "live"), Rig(cand, 0, "cand")
    print("kernel time, median of 200 (us), no network wait: bytes | live | r6 trim | r6 trim+pdl", flush=True)
    for M in (1, 2, 4, 6, 24, 39, 103, 204):
        x = torch.randn(M * HIDDEN, device=dev).to(torch.bfloat16)
        out = torch.empty_like(x)
        two = M * HIDDEN * 2 >= 32768
        res = []
        for R, r3, bits in ((L, False, 0), (C, True, 0), (C, True, 128)):
            R.mode(r3, 24576, bits)
            ts = []
            for _ in range(220):
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                R.t.all_reduce_into(out, x, two, True, a.vpc, False)
                e1.record()
                torch.cuda.synchronize()
                ts.append(e0.elapsed_time(e1) * 1000)
            res.append(statistics.median(ts[20:]))
        print(f"  {M * HIDDEN * 2:7d} | {res[0]:7.2f} | {res[1]:7.2f} | {res[2]:7.2f}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
