#!/usr/bin/env python3
"""device_test7.py: r7-comm2 device test. One GPU, no network. STOPPED NODE ONLY.

Mandatory environment (BRIEF "Device-test environment"), every invocation:

    source ~/venvs/vllm032/bin/activate; export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda \
        DSV41_GLUE_RQPAD=1 DSV41_GLUE_RPSPLIT=split
    cd ~/dsv41-stage/r7comm2/kit && python -I device_test7.py --part all

(r8-comm2fix, review B2: the two glue flags are the promoted r5 glue that production launcher 4f3cc1c6 exports; the
preflight and the start of main() refuse unless the env holds them and the node's launcher still defaults to them.)

The r7 .so is r6-comm's .so (C1 DSV41_RDMA_TRIM, C2 DSV41_RDMA_PDL, unchanged) plus HT13 (kR7Proj = 512: the
dsv41_oblit `_proj_rows` projection computed in the collective's tail). Parts:

  r6       the whole r6 device test (device_test6.py: c1 selftest8 on the served shapes, the r6 must-fail mutants rebuilt
           from the r7 source, c2 PDL edges / poison / m_waitlate) against THE r7 .so: flag-off and TRIM / PDL
           behaviour must be unchanged by the r7 edits (it refuses unless the .so and mutant .so shas match KIT.json);
  stage    must-fail m_stage (prebuilt/mutants/m_stage.so: the 4x-unrolled staging stores a2 / a3 swapped): the staged
           send slot (what the NIC sends) must differ from the input in EVERY case where the unrolled body runs
           (bf16 M >= 39, fp32 M >= 20, every AR_MS value above, one- and two-shot, S1 24576, PDL bits 0 / 128); the
           output is not affected (the reduction reads the receive images), so only the send-slot check sees it;
           the unedited r7 .so must give send == input in the same cases (control);
  proj     HT13 bitwise: reference = the LIVE .so all-reduce (ar_kernel_pl, mask 7) followed by THE SERVED Triton
           kernel (vllm.models.deepseek_v41.nvidia.dsv41_oblit._proj_rows, BLOCK 8192, 8 warps; its cubin sha256 must
           be one of the three served cubins in ref/triton, else REFUSED); candidate = the r7 .so all_reduce_proj_into
           (one kernel). Cases: rows M = 1..6 (DSV41_RDMA_PROJ_MAX_ROWS = 6, the only rows the fold takes), one-shot
           and two-shot, S1 24576 / 0 / 1 GiB, PDL bits 0 / 128 / 384, ranks 0..7 (--ranks), directions = the served
           npz (~/models/dsv41/oblit/obliteratus_dirs.npz, layers 2..39 cycled) + a random unit vector + the planted
           direction; rows = random wide-range, the served embed.weight rows (production data), and the planted row
           (proj_emu.planted_row: every HT13 mutant differs from the served kernel on it, proven on CPU by
           --cpu-selfcheck) scaled by 2^r per row. Checks per call: out == live out (bytes), y == served Triton y
           (bytes), y == proj_emu spec (reported; a disagreement with the SERVED kernel is a harness error), the
           send slot == input, st->seq + 1, every arrival counter (arrive, arrive2, arrive3) back at 0. Plus a captured
           graph (one-shot M = 2 and two-shot M = 4 / 6 projections, PDL bits 0 / 128, mixed with plain ARs) replayed
           20x with fresh inputs == eager reference;
  projmut  must-fail HT13 mutants prebuilt/mutants/{mp_first, mp_warp, mp_l2, mp_unfused, mp_chunk}.so: y must differ
           from the served kernel in EVERY gated proj case (planted rows; M 1..6 x one/two-shot x PDL 0/128 x ranks);
  projtime (INFO ONLY: it can print R7_DEVICE_TEST PASS, but that PASS proves nothing about exactness; it only means
           the timing ran) eager-free timing: graph [live AR -> served _proj_rows] vs graph [r7 AR+proj], 200 replays each, median,
           M = 1, 2, 4, 6, bits 0 and 128; and [r7 TRIM AR -> served _proj_rows] (the C1 baseline the fold is armed on).
           This is the measured fold gain per call that the estimate needs (README §5); INFO, never a gate.

Exit 0 PASS / 1 FAIL / 2 REFUSED / 3 INCOMPLETE. Last line: "R7_DEVICE_TEST PASS|FAIL|REFUSED|INCOMPLETE".
--cpu-selfcheck (no GPU): proj_emu selftest; planted row: every mutant visible in every row scale used; the emulation
of the served op order on random rows reproduces itself (determinism); shapes.check() from r6."""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import device_test6 as D  # noqa: E402  (the r6 device test, repointed at ~/dsv41-stage/r7comm2)
import proj_emu as PE  # noqa: E402

H = 5120
PROJ_ROWS = [1, 2, 3, 4, 5, 6]
PROJ_MUTANTS = ["mp_first", "mp_warp", "mp_l2", "mp_unfused", "mp_chunk"]
STAGE_MUTANT = "m_stage"
OBLIT_NPZ = os.path.expanduser("~/models/dsv41/oblit/obliteratus_dirs.npz")


def served_cubins():
    out = set()
    tdir = os.path.join(HERE, "ref", "triton")
    for d in sorted(os.listdir(tdir)) if os.path.isdir(tdir) else []:
        p = os.path.join(tdir, d, "_proj_rows.cubin")
        if os.path.exists(p):
            out.add(D.sha256(p))
    return out


# r8-comm2fix fix B (2026-11-11): the served-cubin guard must identify the kernel by its SASS, not by
# raw cubin bytes. cuobjdump of two nodes compiling the SAME _proj_rows SASS yields different cubin
# containers (load address, ELF metadata) but the identical instruction stream -- proven: the ref dirs
# 2ITY and X7VL ship different .cubin bytes yet one instruction-only fingerprint. Comparing cubin shas
# (the old served_cubins membership) therefore rejects a legitimate node-local recompile (spark-two/
# three/seven at dw 2026-11-11). We instead disassemble on the node and compare the instruction stream,
# offsets and encodings stripped, so ref and candidate go through the identical cuobjdump pipeline.
_CUOBJ = os.environ.get("R7C_CUOBJ") or "/usr/local/cuda/bin/cuobjdump"
_SASS_LINE = re.compile(r"\s*/\*[0-9a-fA-F]+\*/\s*(.*?)\s*;")


def _proj_sass_fp_from_text(txt):
    """ordered _proj_rows instruction mnemonics+operands (drop the /*addr*/ offset and the trailing
    /* encoding */), sha256 + instruction count. Container/offset/encoding independent, node-portable."""
    insns = []
    for ln in txt.splitlines():
        m = _SASS_LINE.match(ln)
        if m:
            insns.append(re.sub(r"\s+", " ", m.group(1)).strip())
    return hashlib.sha256("\n".join(insns).encode()).hexdigest(), len(insns)


def _proj_sass_fp_of_cubin(path):
    txt = subprocess.check_output([_CUOBJ, "-sass", path], stderr=subprocess.DEVNULL, text=True)
    return _proj_sass_fp_from_text(txt)


def served_proj_fingerprints():
    """{instruction-fingerprint: (ref_dir, n_insns)} for every ref/triton/*/_proj_rows.cubin, disassembled
    on this node with the same cuobjdump the candidate uses (the .sass files are not staged; the cubins are)."""
    out = {}
    tdir = os.path.join(HERE, "ref", "triton")
    for d in sorted(os.listdir(tdir)) if os.path.isdir(tdir) else []:
        p = os.path.join(tdir, d, "_proj_rows.cubin")
        if os.path.exists(p):
            fp, n = _proj_sass_fp_of_cubin(p)
            out[fp] = (d, n)
    return out


# ------------------------------------------------------------------------------------------------ CPU selfcheck ----
def cpu_selfcheck():
    ok, nd = PE.selftest()
    print(f"proj_emu selftest {'PASS' if ok else 'FAIL'} (fma != mul+add on {nd} random triples)")
    good = ok
    x, d = PE.planted_row(*PLANTED)
    for r in range(6):
        xs = [scale_bf16(v, r) for v in x]
        y = PE.proj_row(xs, d)
        for m in PE.VARIANTS[1:]:
            ym = PE.proj_row(xs, d, m)
            n = sum(a != b for a, b in zip(y, ym))
            good &= n > 0
            print(f"planted row scale 2^{r}: {m:10s} differs on {n} elements {'OK' if n else 'NOT VISIBLE'}")
    try:
        import shapes as S
        rc = S.check()
        good &= bool(rc) if rc is not None else True
    except Exception as e:  # noqa: BLE001
        print("shapes.check failed:", e)
        good = False
    good &= glue_env_selfcheck()
    print(f"R7_CPU_SELFCHECK {'PASS' if good else 'FAIL'}")
    return 0 if good else 1


def glue_env_selfcheck():
    """review B2: the glue-env check passes the production env against a 4f3cc1c6-style launcher and fails each
    must-fail variant (a check that cannot fail is not a check); run-device8-r7.sh's prefix equals D.ENV_PREFIX."""
    lt = ("export DSV41_GLUE_RPSPLIT=${DSV41_GLUE_RPSPLIT:-split}  # climb-r5w2\n"
          "export DSV41_GLUE_RQPAD=${DSV41_GLUE_RQPAD:-1}  # climb-r5w2\n")
    prod = {"DSV41_GLUE_RQPAD": "1", "DSV41_GLUE_RPSPLIT": "split"}
    cases = [("production env, 4f3cc1c6 launcher", prod, lt, True),
             ("no glue flags (the r7 prefix)", {}, lt, False),
             ("RQPAD only", {"DSV41_GLUE_RQPAD": "1"}, lt, False),
             ("RPSPLIT=fused", dict(prod, DSV41_GLUE_RPSPLIT="fused"), lt, False),
             ("RQPAD=0", dict(prod, DSV41_GLUE_RQPAD="0"), lt, False),
             ("166d7aac launcher (no glue defaults)", prod, "export DSV41_RDMA_AR=${DSV41_RDMA_AR:-1}\n", False),
             ("launcher unreadable", prod, None, False)]
    good = True
    for name, env, txt, want in cases:
        probs = D.check_glue_env(env, txt)
        ok = (not probs) == want
        good &= ok
        print(f"glue-env check: {name:38s} {'accepted' if not probs else 'refused '} "
              f"{'OK' if ok else 'WRONG (expected ' + ('accept' if want else 'refuse') + ')'}")
    envp = None
    try:
        for line in open(os.path.join(HERE, "ops", "run-device8-r7.sh")):
            if line.startswith("ENVP="):
                envp = line.strip()[len("ENVP="):].strip("'")
    except OSError:
        pass
    same = envp == D.ENV_PREFIX
    good &= same and not D.check_glue_env(dict(kv.split("=", 1) for kv in D.ENV_PREFIX.split() if kv.startswith("DSV41_")),
                                          lt)
    print(f"glue-env prefix: run-device8-r7.sh ENVP {'== D.ENV_PREFIX' if same else '!= D.ENV_PREFIX: ' + repr(envp)}; "
          f"D.ENV_PREFIX exports {[kv for kv in D.ENV_PREFIX.split() if kv.startswith('DSV41_')]}")
    return good


PLANTED = (0.5, 1.5)  # proj_emu.find_planted() result (re-proved by --cpu-selfcheck)


def scale_bf16(b, k):
    """bf16 bits times 2^k (exact; the planted values are far from the exponent limits)"""
    if b & 0x7FFF == 0:
        return b
    e = (b >> 7) & 0xFF
    return (b & 0x807F) | ((e + k) << 7)


# ----------------------------------------------------------------------------------------------------- GPU side ----
def served_proj():
    """the served Triton kernel object and a launcher identical to dsv41_oblit._project's.
    Review B1: under DSV41_GLUE_PDL with its 'proj' family the op that serves is dsv41_glue_pdl.proj_rows (gdc_wait,
    launch_pdl), NOT dsv41_oblit._proj_rows, and its SASS record is unproven; HT13 refuses that state at import
    (dsv41_oblit), and this test refuses it too, so the serving-op assertion below always names the op that serves.
    DSV41_OBLIT_FOLD is refused for the same reason (the folded layers never call _project)."""
    from vllm.models.deepseek_v41.nvidia import dsv41_oblit as O
    import torch
    if getattr(O, "_DSV41_GLUE_PDL", False):
        print("SERVING-OP proj: DSV41_GLUE_PDL enables the 'proj' family, so the served projection is "
              "dsv41_glue_pdl.proj_rows, not dsv41_oblit._proj_rows: REFUSED (unset DSV41_GLUE_PDL)", flush=True)
        raise SystemExit(2)
    if os.environ.get("DSV41_OBLIT_FOLD", "0").strip().lower() not in ("", "0", "off", "false", "no"):
        print("SERVING-OP proj: DSV41_OBLIT_FOLD is set: REFUSED (HT13 refuses it at import)", flush=True)
        raise SystemExit(2)

    def run(x2, dvec):
        y = torch.empty_like(x2)
        k = O._proj_rows[(x2.shape[0],)](x2, y, dvec, x2.stride(0), y.stride(0), H, BLOCK=8192, num_warps=8)
        return y, k
    return run


def directions(dev, rng):
    import torch
    out = []
    try:
        z = np.load(OBLIT_NPZ)
        for k in sorted(z.files, key=lambda s: (len(s), s)):
            if k.startswith("L"):
                v = np.asarray(z[k], np.float64).reshape(-1)
                out.append(("npz:" + k, (v / np.linalg.norm(v)).astype(np.float32)))
    except OSError:
        pass
    r = rng.standard_normal(H)
    out.append(("rand-unit", (r / np.linalg.norm(r)).astype(np.float32)))
    _, dp = PE.planted_row(*PLANTED)
    out.append(("planted", np.array(dp, np.uint32).view(np.float32)))
    return [(n, torch.from_numpy(v.copy()).to(dev)) for n, v in out]


def rows_for(M, fam, rng, embed):
    """[M, H] uint16 rows of the projection input (the AR output)"""
    if fam == "planted":
        x, _ = PE.planted_row(*PLANTED)
        return np.stack([np.array([scale_bf16(v, r) for v in x], np.uint16) for r in range(M)])
    if fam == "embed" and embed is not None:
        idx = rng.integers(0, embed.shape[0], M)
        return embed[idx]
    f = rng.standard_normal((M, H)) * np.exp2(rng.integers(-10, 10, (M, 1)))
    b = f.astype(np.float32).view(np.uint32)
    return ((b + 0x7FFF + ((b >> 16) & 1)) >> 16).astype(np.uint16)


def split_to_ranks(rows, rank_of_data):
    """[8, M*H] inputs whose AR is exactly `rows`: the data on one rank, +0.0 elsewhere"""
    v = np.zeros((D.W, rows.size), np.uint16)
    v[rank_of_data] = rows.reshape(-1)
    v[rank_of_data][v[rank_of_data] == 0x8000] = 0  # -0.0 + 0.0 = +0.0 in the AR; keep the emulation in step
    return v


def run_proj(a, live, cand, muts=None):
    import torch
    dev = torch.device("cuda", 0)
    T = D.Tally()
    rng = np.random.default_rng(a.seed)
    served = served_proj()
    dirs = directions(dev, rng)
    embed = None
    try:
        rec = D.recorded_embed(a.model, [6], a.seed)  # {6: [8, 6 * H]}: each token's row on one rank, +0.0 elsewhere
        embed = np.bitwise_or.reduce(rec[6].reshape(D.W, 6, H), axis=0) if rec else None
    except Exception as e:  # noqa: BLE001
        print("NOTE no embed rows:", e)
    if embed is None:
        T.notes.append("no embed.weight rows (production data family skipped: INCOMPLETE)")
    ranks = [int(x) for x in a.ranks.split(",")]
    s1s = [24576] if a.quick else [24576, 0, 1 << 30]
    bits = [0, 128, 384]
    caught = {m: [0, 0] for m in (muts or {})}
    emu_bad = 0
    cub_ok = None
    for rank in ranks:
        L, C = D.Rig(live, rank, "live"), D.Rig(cand, rank, "cand")
        Ms = {m: D.Rig(mod, rank, m) for m, mod in (muts or {}).items()}
        for M in PROJ_ROWS:
            fams = ["planted"] if muts else ["rand", "planted"] + (["embed"] if embed is not None else [])
            for fam in fams:
                for di, (dname, dvec) in enumerate(dirs):
                    if muts and dname != "planted":
                        continue
                    if not muts and fam == "planted" and dname != "planted":
                        continue
                    if not muts and fam != "planted" and di % 7 != (M + rank) % 7 and dname != "rand-unit":
                        continue  # every direction is used, not every combination (runtime)
                    rows = rows_for(M, fam, rng, embed)
                    v = split_to_ranks(rows, (rank + M) % D.W)
                    for two in (False, True):
                        img, r2 = ar_images7(v, rank, two)
                        for R in [L, C] + list(Ms.values()):
                            R.t.set_recv(torch.from_numpy(img))
                            R.t.set_recv2(torch.from_numpy(np.ascontiguousarray(r2).view(np.uint8).reshape(D.W, -1)))
                        x = D.tensor_of(v[rank], False, dev)
                        L.mode(False)
                        ref = torch.empty_like(x)
                        L.t.all_reduce_into(ref, x, two, True, a.vpc, False)
                        rb = D.bytes_np(ref)
                        yref, k = served(ref.view(M, H), dvec)
                        if cub_ok is None:
                            cub = hashlib.sha256(k.asm["cubin"]).hexdigest()
                            with tempfile.NamedTemporaryFile(suffix=".cubin", delete=True) as _tf:
                                _tf.write(k.asm["cubin"])
                                _tf.flush()
                                cand_fp, cand_n = _proj_sass_fp_of_cubin(_tf.name)
                            served_fps = served_proj_fingerprints()
                            cub_ok = cand_fp in served_fps
                            print(f"SERVING-OP proj: _proj_rows SASS-fp {cand_fp[:16]} ({cand_n} insns) "
                                  f"{'matches a served kernel' if cub_ok else 'IS NOT a served kernel'} "
                                  f"(cubin {cub[:16]}; served fps "
                                  f"{sorted(f[:12] + '/' + str(served_fps[f][1]) for f in served_fps)})", flush=True)
                            if not cub_ok:
                                raise SystemExit(2)
                        yb = D.bytes_np(yref)
                        if fam == "planted" or (rank == ranks[0] and M <= 2):
                            emu = np.array([PE.proj_row(list(np.frombuffer(rb, np.uint16)[r * H:(r + 1) * H]),
                                                        list(dvec.cpu().numpy().view(np.uint32)))
                                            for r in range(M)], np.uint16)
                            if not np.array_equal(emu.view(np.uint8).reshape(-1), yb):
                                emu_bad += 1
                                if emu_bad <= 5:
                                    print(f"NOTE proj spec emulation != served Triton: rank {rank} M {M} {fam} {dname}",
                                          flush=True)
                        targets = [("cand", C)] if not muts else list(Ms.items())
                        for s1 in s1s if not muts else [24576]:
                            for b6 in bits if not muts else [0, 128]:
                                for tname, R in targets:
                                    R.mode(True, s1, b6)
                                    seq0 = R.t.counters()[0]
                                    out = torch.empty_like(x)
                                    y = torch.empty((M, H), dtype=torch.bfloat16, device=dev)
                                    R.t.all_reduce_proj_into(out, x, two, dvec, y, True, a.vpc, False)
                                    ob, yb2 = D.bytes_np(out), D.bytes_np(y)
                                    tag = f"rank {rank} M {M} {fam} {dname} two {two} s1 {s1} bits {b6}"
                                    if muts:
                                        caught[tname][1] += 1
                                        caught[tname][0] += int(not np.array_equal(yb2, yb))
                                        continue
                                    T.check(np.array_equal(ob, rb), f"{tag}: out != live (first byte {D.first_diff(ob, rb)})")
                                    T.check(np.array_equal(yb2, yb), f"{tag}: y != served _proj_rows (first byte "
                                                                      f"{D.first_diff(yb2, yb)})")
                                    c = R.t.counters()
                                    T.check(c[0] == seq0 + 1 and list(c[1:]) == [0, 0, 0, 0] and R.t.arrive3() == 0,
                                            f"{tag}: counters {c} arrive3 {R.t.arrive3()}")
                                    slot = (seq0 + 1) % 4
                                    T.check(np.array_equal(R.t.peek_send(0, slot, x.numel() * 2).numpy(), D.bytes_np(x)),
                                            f"{tag}: send slot != input")
        if not muts:
            proj_graph_check(T, L, C, rank, served, dirs, rng, dev, a.vpc)
        del L, C
    if emu_bad:
        T.check(False, f"proj spec emulation != served Triton kernel in {emu_bad} cases (harness error)")
    return T, caught


def ar_images7(v, rank, two):
    """receive images whose AR is exactly the data row: recv as device_test6.ar_images; recv2 (two-shot phase 2) holds,
    for each peer r, chunk r of the REDUCED message (here the data rank's input, peers are +0.0), so the projection
    sees the same full row on every path (device_test6's recv2 images are random, which the projection must not see)"""
    g = D.Gen(7 + rank)
    img, _ = D.ar_images(v, rank, two, False, g)
    red = np.bitwise_or.reduce(v, axis=0)  # exactly one rank holds data at each element
    n = red.size
    nvec = n // 8
    chunk = (nvec + D.W - 1) // D.W
    r2 = np.zeros((D.W, max(1, chunk) * 8), np.uint16)
    for r in range(D.W):
        lo, hi = r * chunk * 8, min(n, (r + 1) * chunk * 8)
        if hi > lo:
            r2[r, :hi - lo] = red[lo:hi]
    return img, r2


def proj_graph_check(T, L, C, rank, served, dirs, rng, dev, vpc):
    """captured graph of r7 projection ARs (and plain ARs) replayed 20x with fresh inputs == eager reference"""
    import torch
    dvec = dirs[0][1]
    shapes = [(2, False), (4, True), (6, True)]
    for b6 in (0, 128):
        C.mode(True, 24576, b6)
        xs = [torch.zeros(M * H, dtype=torch.bfloat16, device=dev) for M, _ in shapes]
        outs = [torch.empty_like(t) for t in xs]
        ys = [torch.empty((M, H), dtype=torch.bfloat16, device=dev) for M, _ in shapes]
        plain_x = torch.zeros(4 * H, dtype=torch.bfloat16, device=dev)
        plain_o = torch.empty_like(plain_x)
        imgs = []
        for (M, two) in shapes:
            v = split_to_ranks(rows_for(M, "rand", rng, None), (rank + 1) % D.W)
            imgs.append((v, two))
        # images: the graph replays read the same receive images; only the local input changes per replay, so the
        # images hold zeros for every peer and the data is this rank's own input (AR == local input exactly)
        zero = np.zeros((D.W, D.MAX_BYTES), np.uint8)
        for R in (L, C):
            R.t.set_recv(torch.from_numpy(zero))
            R.t.set_recv2(torch.from_numpy(np.zeros((D.W, R.mc), np.uint8)))
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.stream(s):
            for i, (M, two) in enumerate(shapes):  # warm
                C.t.all_reduce_proj_into(outs[i], xs[i], two, dvec, ys[i], True, vpc, False)
            C.t.all_reduce_into(plain_o, plain_x, True, True, vpc, False)
            with torch.cuda.graph(gr, stream=s):
                for i, (M, two) in enumerate(shapes):
                    C.t.all_reduce_proj_into(outs[i], xs[i], two, dvec, ys[i], True, vpc, False)
                C.t.all_reduce_into(plain_o, plain_x, True, True, vpc, False)
        torch.cuda.current_stream().wait_stream(s)
        for rep in range(20):
            for i, (M, two) in enumerate(shapes):
                xs[i].copy_(D.tensor_of(rows_for(M, "rand", rng, None).reshape(-1), False, dev))
            plain_x.copy_(D.tensor_of(rows_for(4, "rand", rng, None).reshape(-1), False, dev))
            gr.replay()
            torch.cuda.synchronize()
            L.mode(False)
            for i, (M, two) in enumerate(shapes):
                ref = torch.empty_like(xs[i])
                L.t.all_reduce_into(ref, xs[i], two, True, vpc, False)  # eager live on the same (zero) images
                yr, _ = served(ref.view(M, H), dvec)
                T.check(np.array_equal(D.bytes_np(outs[i]), D.bytes_np(ref)),
                        f"graph rank {rank} bits {b6} rep {rep} M {M}: out != eager live")
                T.check(np.array_equal(D.bytes_np(ys[i]), D.bytes_np(yr)),
                        f"graph rank {rank} bits {b6} rep {rep} M {M}: y != served _proj_rows(eager live)")
            pref = torch.empty_like(plain_x)
            L.t.all_reduce_into(pref, plain_x, True, True, vpc, False)
            T.check(np.array_equal(D.bytes_np(plain_o), D.bytes_np(pref)), f"graph rank {rank} bits {b6} rep {rep}: plain AR")


def run_stage(a, cand, mut):
    """m_stage: the staged send slot must differ from the input wherever the unrolled body runs; control: r7 .so equal"""
    import torch
    import shapes as S
    dev = torch.device("cuda", 0)
    T = D.Tally()
    res = {"cases": 0, "caught": 0}
    ranks = [int(x) for x in a.ranks.split(",")][:2]
    g = D.Gen(a.seed)
    for rank in ranks:
        C, Mu = D.Rig(cand, rank, "cand"), D.Rig(mut, rank, "m_stage")
        for f32 in (False, True):
            ms = [M for M in D.AR_MS[f32] if M >= (39 if not f32 else 20)]
            for M in ms:
                n = M * H
                v = g.ranks(n, f32, "rand")
                for two in (False, True):
                    img, r2 = D.ar_images(v, rank, two, f32, g)
                    for b6 in (0, 128):
                        for R in (C, Mu):
                            R.t.set_recv(torch.from_numpy(img))
                            R.t.set_recv2(torch.from_numpy(np.ascontiguousarray(r2).view(np.uint8).reshape(D.W, -1)))
                            R.mode(True, 24576, b6)
                        x = D.tensor_of(v[rank], f32, dev)
                        es = 4 if f32 else 2
                        sends = []
                        for R in (C, Mu):
                            seq0 = R.t.counters()[0]
                            out = torch.empty_like(x)
                            R.t.all_reduce_into(out, x, two, True, a.vpc, False)
                            sends.append(R.t.peek_send(0, (seq0 + 1) % 4, n * es).numpy())
                        xb = D.bytes_np(x)
                        T.check(np.array_equal(sends[0], xb), f"control r7 rank {rank} f32 {f32} M {M} two {two} bits {b6}: send != input")
                        res["cases"] += 1
                        res["caught"] += int(not np.array_equal(sends[1], xb))
    return T, res["caught"] == res["cases"] and res["cases"] > 0, res


def projtime(a, live, cand, trimmed=True):
    import statistics
    import torch
    dev = torch.device("cuda", 0)
    served = served_proj()
    rng = np.random.default_rng(1)
    d = directions(dev, rng)[0][1]
    L, C = D.Rig(live, 0, "live"), D.Rig(cand, 0, "cand")
    zero = np.zeros((D.W, D.MAX_BYTES), np.uint8)
    for R in (L, C):
        R.t.set_recv(torch.from_numpy(zero))
        R.t.set_recv2(torch.from_numpy(np.zeros((D.W, R.mc), np.uint8)))
    print("PROJTIME (graph replay medians, 200 runs; no network wait; INFO)")
    for M in (1, 2, 4, 6):
        two = M * H * 2 >= 32768
        x = D.tensor_of(rows_for(M, "rand", rng, None).reshape(-1), False, dev)
        o = torch.empty_like(x)
        y = torch.empty((M, H), dtype=torch.bfloat16, device=dev)
        arms = {}
        for name in ("live+triton", "trim+triton", "r7fold", "r7fold-pdl"):
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            gr = torch.cuda.CUDAGraph()
            def body():
                if name == "live+triton":
                    L.mode(False)
                    L.t.all_reduce_into(o, x, two, True, a.vpc, False)
                    served(o.view(M, H), d)
                elif name == "trim+triton":
                    C.mode(True, 24576, 0)
                    C.t.all_reduce_into(o, x, two, True, a.vpc, False)
                    served(o.view(M, H), d)
                else:
                    C.mode(True, 24576, 128 if name.endswith("pdl") else 0)
                    C.t.all_reduce_proj_into(o, x, two, d, y, True, a.vpc, False)
            with torch.cuda.stream(s):
                body()
                with torch.cuda.graph(gr, stream=s):
                    body()
            torch.cuda.current_stream().wait_stream(s)
            ts = []
            for _ in range(220):
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                gr.replay()
                e1.record()
                torch.cuda.synchronize()
                ts.append(e0.elapsed_time(e1) * 1000)
            arms[name] = statistics.median(ts[20:])
        print(f"PROJTIME M {M} two {two}: " + " ".join(f"{k} {v:.2f} us" for k, v in arms.items()) +
              f" | fold gain vs trim+triton {arms['trim+triton'] - arms['r7fold']:+.2f} us", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["r6", "stage", "proj", "projmut", "projtime", "all"])
    ap.add_argument("--cand", default=D.CAND_SO)
    ap.add_argument("--live", default=D.LIVE_SO)
    ap.add_argument("--mutant-dir", default=os.path.join(HERE, "prebuilt", "mutants"))
    ap.add_argument("--ranks", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--model", default=D.MODEL)
    ap.add_argument("--seed", type=int, default=20261109)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--r6-part", default="c1", choices=["c1", "mutants", "c2", "all"],
                    help="--part r6/all: which part of the r6 device test (run-device8-r7.sh: c1 on 8/8, mutants on "
                         "spark-one, c2 on spark-five, as in r6)")
    ap.add_argument("--cpu-selfcheck", action="store_true")
    a = ap.parse_args()
    if a.cpu_selfcheck:
        return cpu_selfcheck()
    # the r6 preflight (venv, nvcc/cuobjdump/ninja on PATH, CUDA_HOME, no serving process, live .so 14d049dd, the r7
    # .so and mutant shas from KIT.json, launcher mask 7): the mandatory environment is enforced before any GPU work
    a.part6 = a.part
    pa = argparse.Namespace(**vars(a))
    pa.part = a.r6_part if a.part in ("r6", "all") else "c1"
    bad = D.preflight(pa)
    a.vpc = pa.vpc
    if a.part in ("stage", "projmut", "all"):
        _, want = D.kit_shas()
        for m in ([STAGE_MUTANT] if a.part in ("stage", "all") else []) + (PROJ_MUTANTS if a.part in ("projmut", "all") else []):
            p = os.path.join(a.mutant_dir, f"{m}.so")
            if not os.path.exists(p) or not want.get(m) or D.sha256(p) != want[m]:
                bad.append(f"{p}: missing or sha256 != KIT.json mutant_so_sha256[{m}]")
    if not served_cubins():
        bad.append("ref/triton holds no served _proj_rows cubin")
    if bad:
        for b in bad:
            print("PREFLIGHT FAIL:", b)
        print(f"run every device command as:  {D.ENV_PREFIX}; <cmd>")
        print("R7_DEVICE_TEST REFUSED")
        return 2
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1):
        print("PREFLIGHT FAIL: no GB10 (sm_121) device\nR7_DEVICE_TEST REFUSED")
        return 2
    live = D.load_ext(a.live, "live")
    cand = D.load_ext(a.cand, "cand")
    if getattr(cand, "R7_VARIANT", None) is None or getattr(live, "R3_VARIANT", None) is not None:
        print("PREFLIGHT FAIL: live must be the g130 .so and the candidate the r7 .so\nR7_DEVICE_TEST REFUSED")
        return 2
    probs = D.assert_serving_op(live, cand)
    if probs:
        for p in probs:
            print("SERVING-OP FAIL:", p)
        print("R7_DEVICE_TEST REFUSED")
        return 2
    # the candidate proj kernel: mask & 512 in the profiler trace
    T0 = D.Tally()
    ks = D.kernels_of(lambda: _one_proj(cand, a.vpc))
    if not any(k[0] == "ar_kernel_r3" and k[3] >= 0 and (k[3] & 512) for k in ks):
        print(f"SERVING-OP FAIL: the r7 projection call launched {ks}, not ar_kernel_r3 with mask & 512")
        print("R7_DEVICE_TEST REFUSED")
        return 2
    print(f"SERVING-OP OK: candidate projection kernel {[k for k in ks if k[0] == 'ar_kernel_r3']}", flush=True)
    # review B2: the production glue env, asserted next to the serving-op assertion (preflight checked it too)
    try:
        ltxt = open(D.LAUNCHER).read()
    except OSError:
        ltxt = None
    gprobs = D.check_glue_env(os.environ, ltxt)
    if gprobs:
        for p in gprobs:
            print("GLUE-ENV FAIL:", p)
        print("R7_DEVICE_TEST REFUSED")
        return 2
    print("GLUE-ENV OK: " + " ".join(f"{k}={os.environ[k]}" for k, _ in D.PROD_GLUE_ENV) +
          f" (= the defaults of {D.LAUNCHER})", flush=True)
    verdict, incomplete, summary = True, [], {}
    if a.part in ("r6", "all"):
        sys.argv = [sys.argv[0], "--part", a.r6_part, "--cand", a.cand, "--live", a.live, "--mutant-dir", a.mutant_dir,
                    "--ranks", a.ranks, "--model", a.model, "--seed", str(a.seed)] + (["--quick"] if a.quick else [])
        rc = D.main()
        summary["r6"] = rc
        verdict &= rc in (0, 3)
        if rc == 3:
            incomplete.append("r6 part INCOMPLETE")
    if a.part in ("stage", "all"):
        mut = D.load_ext(os.path.join(a.mutant_dir, f"{STAGE_MUTANT}.so"), "mut_stage")
        T, ok, res = run_stage(a, cand, mut)
        print(f"STAGE: control {T.checks} checks {T.fails} failures; m_stage caught {res['caught']}/{res['cases']}", flush=True)
        summary["stage"] = res
        verdict &= ok and T.fails == 0
    if a.part in ("proj", "all"):
        T, _ = run_proj(a, live, cand)
        print(f"PROJ: {T.checks} checks, {T.fails} failures; " + "; ".join(T.notes), flush=True)
        summary["proj"] = (T.checks, T.fails)
        verdict &= T.fails == 0 and T.checks > 0
        if T.notes:
            incomplete += T.notes
    if a.part in ("projmut", "all"):
        muts = {m: D.load_ext(os.path.join(a.mutant_dir, f"{m}.so"), f"mut_{m}") for m in PROJ_MUTANTS}
        for m, mod in muts.items():
            if getattr(mod, "R7_MUTANT", None) != m:
                print(f"PREFLIGHT FAIL: {m}.so does not carry R7_MUTANT={m}\nR7_DEVICE_TEST REFUSED")
                return 2
        _, caught = run_proj(a, live, cand, muts)
        for m, (c, n) in caught.items():
            print(f"PROJMUT {m}: caught {c}/{n} {'OK' if n and c == n else 'NOT EVERY CASE'}", flush=True)
            verdict &= n > 0 and c == n
        summary["projmut"] = caught
    if a.part in ("projtime", "all"):
        projtime(a, live, cand)
    print("SUMMARY", json.dumps(summary, default=str), flush=True)
    if not verdict:
        print("R7_DEVICE_TEST FAIL")
        return 1
    if incomplete or a.quick:
        for i in incomplete:
            print("INCOMPLETE:", i)
        print("R7_DEVICE_TEST INCOMPLETE")
        return 3
    print("R7_DEVICE_TEST PASS")
    return 0


def _one_proj(cand, vpc):
    import torch
    dev = torch.device("cuda", 0)
    R = D.Rig(cand, 0, "cand")
    R.t.set_recv(torch.from_numpy(np.zeros((D.W, D.MAX_BYTES), np.uint8)))
    R.mode(True, 24576, 0)
    x = torch.zeros(4 * H, dtype=torch.bfloat16, device=dev)
    d = torch.zeros(H, dtype=torch.float32, device=dev)
    R.t.all_reduce_proj_into(torch.empty_like(x), x, True, d, torch.empty((4, H), dtype=torch.bfloat16, device=dev),
                             True, vpc, False)


if __name__ == "__main__":
    raise SystemExit(main())
