#!/usr/bin/env python3
"""r2-draftmoe device test (G17, DSV41_DRAFT_MOE_G17): live drafter MoE kernels vs the compact / fused-permute path,
side by side, bitwise, plus timings. Self-contained; run by the lead on a STOPPED node (dsv41-rank inactive).

    cd <kit> && ~/venvs/vllm032/bin/python device_test.py [--real-weights] [--layers 0,1,2] [--rank R]
                                                          [--quick] [--iters 200] [--json out.json] [--force]

Production environment: every `export DSV41_...=${DSV41_...:-value}` default of ~/dsv41-tp8.sh for the MoE stack
(DSV41_SM121_TARGET_M32, DSV41_MOE_*, DSV41_TARGET_M32_*, DSV41_DRAFT_*, VLLM_USE_DEEP_GEMM) is applied unless already
set; DSV41_DRAFT_MOE_G17 is forced unset (the test drives both paths explicitly).

Arms, each against OLD = padded FC1 (w13 [128, 768, 2560]) + stock permute, on identical inputs:
  fc1    compact FC1 (w13 [128, 576, 2560], built from the same rows) + compact SiLU writer (padded_output_size=384)
  perm   padded FC1 + dsv41_fused_permute (E 128, top-3)          (M32 path only)
  both   compact FC1 + fused permute                              (M32 path; 128-row path: = fc1)
  e2e    if the kit is installed: the PATCHED DeepGemmFP4Experts.apply with the module flags switched on, on compact
         weights, vs OLD (the patched apply with flags off on padded weights)
Compared bitwise per routed slot: FC1 output (compact columns vs padded live columns; padded pad columns must be +0),
FC2 input FP8 bytes and packed UE8M0 scale words, FC2 output rows, and the MoE output [M, 5120]; first mismatch
reported. M: 1..10, 12, 15, 16, 20, 24, 25, 30, 32, 40, 48, 50, 60, 64, 80, 96, 100, 120, 128, 150, 160 (M32 path;
DSpark block 5 x 1..32 seqs) and 161, 200, 256, 512, 1024 (128-row path, prefill-sized drafter calls).
Inputs: random, hostile (zero rows, max-magnitude, tiny scales, ties, all tokens on 3 experts, all 128 experts
touched, int64 ids) and optionally the real mtp.0-2 expert weights of this TP rank. Eager and CUDA-graph replay
(poisoned workspaces, new inputs copied into the static buffers before every replay).
Timing: median of --iters runs, warm and cold-L2 (64 MiB write between runs), for permute, FC1, full MoE.
Expected runtime: 3-6 min (+ about 1-2 min per drafter FC1 instance DeepGEMM has to JIT when install.py prejit was
not run; the JIT goes into the production cache dirs exactly as a boot would). Exit 0 only if every case is bitwise
identical (and, with --real-weights, the real weights loaded).
"""
import argparse
import json
import os
import re
import socket
import statistics
import subprocess
import sys
import time

AP = argparse.ArgumentParser()
AP.add_argument("--real-weights", action="store_true")
AP.add_argument("--layers", default="0,1,2")
AP.add_argument("--rank", type=int, default=None, help="TP rank for --real-weights (default: from hostname spark-N)")
AP.add_argument("--quick", action="store_true", help="fewer M values, 50 timing iterations")
AP.add_argument("--iters", type=int, default=200)
AP.add_argument("--json", default=None)
AP.add_argument("--force", action="store_true", help="run even if dsv41-rank is active (NOT for a serving node)")
ARGS = AP.parse_args()
HOME = os.path.expanduser("~")


def apply_launcher_env():
    applied = {}
    path = os.path.join(HOME, "dsv41-tp8.sh")
    pat = re.compile(r"^export ((?:DSV41_SM121_TARGET_M32|DSV41_MOE_\w+|DSV41_TARGET_M32\w*|DSV41_DRAFT_\w+|"
                     r"VLLM_USE_DEEP_GEMM))=\$\{\1:-([^}]*)\}")
    for line in open(path):
        m = pat.match(line.strip())
        if m and m.group(1) not in os.environ:
            val = m.group(2).replace("$HOME", HOME)
            os.environ[m.group(1)] = val
            applied[m.group(1)] = val
    os.environ.pop("DSV41_DRAFT_MOE_G17", None)
    return applied


ENV_APPLIED = apply_launcher_env()
if not ARGS.force:
    st = subprocess.run(["systemctl", "--user", "is-active", "dsv41-rank"], capture_output=True, text=True).stdout
    if st.strip() == "active":
        sys.exit("REFUSED: dsv41-rank is active on this node (stop it in a window, or --force on a non-serving node)")

import torch  # noqa: E402

from vllm.model_executor.layers.fused_moe import deep_gemm_utils as dgu  # noqa: E402
from vllm.model_executor.layers.fused_moe.experts import deep_gemm_moe as dgm  # noqa: E402
from vllm.model_executor.layers.fused_moe.experts.sm121_target_m32 import get_target_m32_kernel  # noqa: E402
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import _pack_deepgemm_mxfp4_scales  # noqa: E402
from vllm.model_executor.layers.quantization.utils.fp8_utils import (  # noqa: E402
    silu_mul_quant_fp8_packed_triton as fused_silu)
from vllm.utils.deep_gemm import m_grouped_fp8_fp4_gemm_nt_contiguous, mk_alignment_scope  # noqa: E402

DEV = torch.device("cuda")
E, TOPK, H = 128, 3, 5120
LIVE = torch.cat([torch.arange(0, 288), torch.arange(384, 672)]).to(DEV)
PADC = torch.cat([torch.arange(288, 384), torch.arange(672, 768)]).to(DEV)
CLAMP = 10.0  # config.json text_config.swiglu_limit (the drafter's gemm1_clamp_limit)
LIVE_ACT_MIN = int(os.environ.get("DSV41_MOE_LIVE_ACT_MIN_ROWS", "1024")) if os.environ.get("DSV41_MOE_LIVE_ACT") == "1" else None
FAIL = []
REPORT = {"host": socket.gethostname(), "env_applied": ENV_APPLIED, "cases": [], "timing": [], "jit": {}}


def log(msg):
    print(msg, flush=True)


def jit_listing():
    dirs = {"m32": os.path.join(HOME, "dsv41-stage/g130/fc1/jit/deal-base/cache"),
            "dg128": os.path.join(HOME, ".cache/vllm/deep_gemm/cache")}
    return {k: sorted(x for x in os.listdir(v) if "fp8_fp4" in x) if os.path.isdir(v) else [] for k, v in dirs.items()}


REPORT["jit"]["before"] = jit_listing()
K32 = get_target_m32_kernel()


# ------------------------------------------------------------------------------------------------------ inputs
def rand_fp8_bytes(shape, gen):
    b = torch.randint(0, 256, shape, dtype=torch.uint8, device=DEV, generator=gen)
    b[(b & 0x7F) == 0x7F] = 0x7E
    return b


def quant_ue8m0(x):
    """FP8 e4m3 per-128 with power-of-two (UE8M0-valued) fp32 scales, as the MoE input quant produces."""
    M = x.shape[0]
    g = x.float().view(M, H // 128, 128)
    amax = g.abs().amax(-1).clamp(min=1e-10)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    q = (g / scale[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).view(M, H)
    return q.contiguous(), scale.contiguous()


def make_hidden(M, kind, gen):
    x = torch.randn(M, H, device=DEV, generator=gen) * torch.exp(torch.randn(M, 1, device=DEV, generator=gen))
    if kind == "zeros":
        x[: max(1, M // 2)] = 0
    elif kind == "max":
        x = torch.where(torch.rand(M, H, device=DEV, generator=gen) < 0.5, 1.0, -1.0) * 448.0 * 8
    elif kind == "tiny":
        x = x * 1e-30
    elif kind == "ties":
        x = torch.where(torch.rand(M, H, device=DEV, generator=gen) < 0.5, 3.0, -3.0)
    return quant_ue8m0(x)


def make_ids(M, kind, gen, dtype=torch.int32):
    if kind == "same3":
        ids = torch.tensor([[0, 1, 2]], device=DEV).repeat(M, 1)
    elif kind == "all":
        ids = (torch.arange(M * TOPK, device=DEV) % E).view(M, TOPK)
        ids = ids if M * TOPK >= E else torch.stack([torch.randperm(E, device=DEV, generator=gen)[:TOPK]
                                                     for _ in range(M)])
    else:
        ids = torch.stack([torch.randperm(E, device=DEV, generator=gen)[:TOPK] for _ in range(M)])
    w = torch.rand(M, TOPK, device=DEV, generator=gen)
    return ids.to(dtype).contiguous(), (w / w.sum(-1, keepdim=True)).float().contiguous()


def weights_random(gen, kind="random"):
    w13 = torch.randint(0, 256, (E, 768, H // 2), dtype=torch.uint8, device=DEV, generator=gen)
    s13 = torch.randint(118, 128, (E, 768, H // 32), dtype=torch.uint8, device=DEV, generator=gen)
    w2 = torch.randint(0, 256, (E, H, 192), dtype=torch.uint8, device=DEV, generator=gen)
    s2 = torch.randint(118, 128, (E, H, 12), dtype=torch.uint8, device=DEV, generator=gen)
    if kind == "hostile":
        w13[:8] = 0x77            # +6,+6 everywhere
        w13[8:16] = 0xFF          # -6,-6
        w13[16:24] = 0x88         # -0 codes
        w13[24:32] = 0            # zero experts
        s13[:16] = 127 + 2
        s13[32:40] = 1            # denormal-range weight scales
    return pad_layout(w13, s13, w2, s2)


def pad_layout(w13, s13, w2, s2):
    """Exactly what the zero-initialised padded parameters hold after the loader: pad rows / K-tail zero."""
    w13[:, 288:384] = 0
    w13[:, 672:768] = 0
    s13[:, 288:384] = 0
    s13[:, 672:768] = 0
    w2[:, :, 144:] = 0
    s2[:, :, 9:] = 0
    return w13.contiguous(), s13.contiguous(), w2.contiguous(), s2.contiguous()


def weights_real(layer, rank):
    from safetensors import safe_open

    root = os.path.join(HOME, "models/dsv41/official-fp8")
    wm = json.load(open(os.path.join(root, "model.safetensors.index.json")))["weight_map"]
    w13 = torch.zeros(E, 768, H // 2, dtype=torch.uint8)
    s13 = torch.zeros(E, 768, H // 32, dtype=torch.uint8)
    w2 = torch.zeros(E, H, 192, dtype=torch.uint8)
    s2 = torch.zeros(E, H, 12, dtype=torch.uint8)
    handles = {}

    def get(name):
        f = wm[name]
        if f not in handles:
            handles[f] = safe_open(os.path.join(root, f), "pt")
        t = handles[f].get_tensor(name)
        return t.view(torch.uint8) if t.dtype in (torch.int8, torch.float8_e8m0fnu) else t

    r = slice(rank * 288, (rank + 1) * 288)
    for e in range(E):
        p = f"mtp.{layer}.ffn.experts.{e}"
        w1, w3 = get(p + ".w1.weight"), get(p + ".w3.weight")
        assert w1.shape == (2304, 2560) and w3.shape == (2304, 2560)
        w13[e, 0:288], w13[e, 384:672] = w1[r], w3[r]
        a, b = get(p + ".w1.scale"), get(p + ".w3.scale")
        s13[e, 0:288], s13[e, 384:672] = a[r], b[r]
        wd, sd = get(p + ".w2.weight"), get(p + ".w2.scale")
        assert wd.shape == (5120, 1152) and sd.shape == (5120, 72)
        w2[e, :, :144] = wd[:, rank * 144:(rank + 1) * 144]
        s2[e, :, :9] = sd[:, rank * 9:(rank + 1) * 9]
    return pad_layout(w13.to(DEV), s13.to(DEV), w2.to(DEV), s2.to(DEV))


def finish_weights(w13p, s13p, w2, s2):
    """Padded and compact parameter sets through the live post-load scale pack (_pack_deepgemm_mxfp4_scales)."""
    w13c = w13p[:, LIVE].contiguous()
    s13c = s13p[:, LIVE].contiguous()
    p13, p2 = _pack_deepgemm_mxfp4_scales(w13p, w2, s13p, s2)
    c13, c2 = _pack_deepgemm_mxfp4_scales(w13c, w2, s13c, s2)
    assert torch.equal(p2.view(torch.int32) if p2.dtype != torch.int32 else p2,
                       c2.view(torch.int32) if c2.dtype != torch.int32 else c2), "w2 scales must not depend on w13"
    return dict(padded=(w13p, p13, w2, p2), compact=(w13c, c13, w2, c2))


# ------------------------------------------------------------------------------------------------------ pipeline
class WS:
    """apply()'s workspaces: ws13 holds a1q_perm then (reused) the FC2 input; ws2 holds FC1 out then FC2 out."""

    def __init__(self, max_rows):
        self.ws13 = torch.empty(max_rows * H // 2 + 8, dtype=torch.bfloat16, device=DEV)
        self.ws2 = torch.empty(max_rows * H, dtype=torch.bfloat16, device=DEV)

    def poison(self, pattern):
        self.ws13.view(torch.int16).fill_(pattern)
        self.ws2.view(torch.int16).fill_(pattern)


def resize(x, shape):
    n = 1
    for d in shape:
        n *= d
    return x.flatten()[:n].view(*shape)


def moe(W, ws, aq, sc, ids, tw, *, compact, perm, block_m):
    """DeepGemmFP4Experts.apply for the drafter, step by step with the live kernels; returns intermediates."""
    w1, w1s, w2, w2s = W
    N = w1.shape[1]
    M = ids.shape[0]
    M_sum, _ = dgu.compute_aligned_M_and_alignment(M, TOPK, E, block_m, None)
    a1q_perm = resize(ws.ws13.view(torch.float8_e4m3fn), (M_sum, H))
    live_act = LIVE_ACT_MIN is not None and M_sum >= LIVE_ACT_MIN
    if perm == "fused":
        a1q, a1s, ids_l, inv, ends = dgu.dsv41_fused_permute(aq, sc, ids, E, a1q_perm, M_sum, 32)
        ends = ends if live_act else None  # as apply(): expert_ends only under LIVE_ACT
        align_used = 32
        assert block_m == 32
    else:
        res = dgu.deepgemm_moe_permute(aq=aq, aq_scale=sc, topk_ids=ids, local_num_experts=E, expert_map=None,
                                       expert_tokens_meta=None, aq_out=a1q_perm, alignment_m=block_m,
                                       **({"return_expert_ends": True} if live_act else {}))
        a1q, a1s, ids_l, inv, align_used = res[:5]
        ends = res[5] if live_act else None
    gemm = K32.gemm if align_used == 32 else m_grouped_fp8_fp4_gemm_nt_contiguous
    scope = K32.alignment_scope() if align_used == 32 else mk_alignment_scope(align_used)
    with scope:
        mm1 = resize(ws.ws2, (M_sum, N))
        gemm((a1q, a1s), (w1.view(torch.int8), w1s), mm1, ids_l, recipe_a=(1, 128), recipe_b=(1, 32))
        mm1_copy = mm1.clone()  # ws2 is reused by FC2
        qout = resize(ws.ws13.view(torch.float8_e4m3fn), (M_sum, 384))
        kw = dict(input=mm1.view(-1, N), output_q=qout, group_size=128, clamp_limit=CLAMP,
                  expert_ends=ends, expert_alignment=align_used if live_act else 0)
        if compact:
            kw["padded_output_size"] = 384
        a2q, a2s = fused_silu(**kw)
        a2q_copy, a2s_copy = a2q.clone(), a2s.clone()
        mm2 = resize(ws.ws2, (M_sum, H))
        gemm((a2q, a2s), (w2.view(torch.int8), w2s), mm2, ids_l, recipe_a=(1, 128), recipe_b=(1, 32))
    out = torch.empty(M, H, dtype=torch.bfloat16, device=DEV)
    dgu.deepgemm_unpermute_and_reduce(a=mm2, topk_ids=ids, topk_weights=tw, inv_perm=inv, expert_map=None, output=out)
    return dict(mm1=mm1_copy, a2q=a2q_copy, a2s=a2s_copy, mm2=mm2.clone(), out=out, inv=inv.clone(),
                layout=ids_l.clone(), M_sum=M_sum, align=align_used)


def bits(t):
    return t.contiguous().view({1: torch.uint8, 2: torch.int16, 4: torch.int32}[t.element_size()])


def first_mismatch(a, b):
    ne = (bits(a) != bits(b))
    if not bool(ne.any()):
        return None
    idx = ne.nonzero()[0].tolist()
    return f"{int(ne.sum())} words differ, first at {idx}: {bits(a)[tuple(idx)].item()} vs {bits(b)[tuple(idx)].item()}"


def compare(old, new, ids, new_compact, label):
    """Per routed slot (rows found through each pipeline's own inv_perm)."""
    valid = (ids >= 0).flatten()
    so = old["inv"].flatten()[valid].long()
    sn = new["inv"].flatten()[valid].long()
    errs = []
    m1o = old["mm1"][so]
    m1n = new["mm1"][sn]
    if new_compact:
        m1o_live = m1o[:, LIVE]
        pad = m1o[:, PADC]
        if bool((bits(pad) != 0).any()):
            errs.append(f"padded FC1 pad columns not +0: {int((bits(pad) != 0).sum())} words")
        r = first_mismatch(m1o_live, m1n)
    else:
        r = first_mismatch(m1o, m1n)
    if r:
        errs.append("FC1 " + r)
    for key, nm in (("a2q", "FC2 input bytes"), ("mm2", "FC2 output")):
        r = first_mismatch(old[key][so], new[key][sn])
        if r:
            errs.append(f"{nm} {r}")
    r = first_mismatch(old["a2s"][so], new["a2s"][sn])
    if r:
        errs.append("FC2 input scale words " + r)
    r = first_mismatch(old["out"], new["out"])
    if r:
        errs.append("MoE output " + r)
    if old["M_sum"] != new["M_sum"] or old["align"] != new["align"]:
        errs.append(f"M_sum/align {old['M_sum']}/{old['align']} vs {new['M_sum']}/{new['align']}")
    ok = not errs
    REPORT["cases"].append(dict(case=label, ok=ok, errors=errs))
    if not ok:
        FAIL.append(label)
        log(f"FAIL {label}: {'; '.join(errs)}")
    return ok


# ------------------------------------------------------------------------------------------------------ e2e (kit)
def e2e_runner(Wset):
    """The installed, patched DeepGemmFP4Experts.apply driven with the kit's module flags (if installed)."""
    if not hasattr(dgm, "_R2DM_FC1") or not hasattr(dgu, "_R2DM_PERM"):
        return None
    import importlib
    import types

    r2 = importlib.import_module("vllm.model_executor.layers.fused_moe.experts.dsv41_r2_draftmoe")
    exp = object.__new__(dgm.DeepGemmFP4Experts)
    exp.gemm1_clamp_limit = CLAMP
    exp._dsv41_target_m32 = K32
    exp._dsv41_force_m32 = False
    exp._dsv41_draft_verify = None
    live = os.environ.get("DSV41_MOE_LIVE_ACT") == "1"
    exp._dsv41_live_act_min_rows = int(os.environ.get("DSV41_MOE_LIVE_ACT_MIN_ROWS", "1024")) if live else None
    exp._dsv41_live_act_max_rows = int(os.environ.get("DSV41_MOE_LIVE_ACT_MAX_ROWS", "0")) if live else 0

    def run(which, ws, aq, sc, ids, tw, on):
        w1, w1s, w2, w2s = Wset[which]
        exp.quant_config = types.SimpleNamespace(w1_scale=w1s, w2_scale=w2s)
        saved = (dgm._R2DM_FC1, dgu._R2DM_PERM, getattr(r2, "PERM", ""))
        dgm._R2DM_FC1, dgu._R2DM_PERM, r2.PERM = (True, "1", "1") if on else (False, "", "")
        dgm._r2dm = dgu._r2dm = r2
        try:
            M = ids.shape[0]
            M_sum, _ = dgu.compute_aligned_M_and_alignment(M, TOPK, E, 128, None)
            out = torch.empty(M, H, dtype=torch.bfloat16, device=DEV)
            exp.apply(output=out, hidden_states=aq, w1=w1, w2=w2, topk_weights=tw, topk_ids=ids,
                      activation=dgm.MoEActivation.SILU, global_num_experts=-1, expert_map=None, a1q_scale=sc,
                      a2_scale=None, workspace13=ws.ws13, workspace2=ws.ws2, expert_tokens_meta=None,
                      apply_router_weight_on_input=False)
            return out
        finally:
            dgm._R2DM_FC1, dgu._R2DM_PERM, r2.PERM = saved

    return run


# ------------------------------------------------------------------------------------------------------ timing
FLUSH = torch.empty(64 << 20, dtype=torch.uint8, device=DEV)


def timeit(fn, iters, cold):
    ts = []
    for i in range(iters + 5):
        if cold:
            FLUSH.fill_(i & 0xFF)
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        if i >= 5:
            ts.append(a.elapsed_time(b) * 1000.0)
    return statistics.median(ts)


# ------------------------------------------------------------------------------------------------------ main
def main():
    t0 = time.time()
    gen = torch.Generator(device=DEV)
    gen.manual_seed(20261108)
    m32_ms = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 20, 24, 25, 30, 32, 40, 48, 50, 60, 64, 80, 96, 100, 120,
              128, 150, 160]
    big_ms = [161, 200, 256, 512, 1024]
    if ARGS.quick:
        m32_ms, big_ms = [1, 2, 5, 10, 16, 32, 64, 96, 160], [161, 512]
    weight_sets = [("random", lambda: weights_random(gen)), ("hostile-w", lambda: weights_random(gen, "hostile"))]
    if ARGS.real_weights:
        rank = ARGS.rank
        if rank is None:
            names = ["one", "two", "three", "four", "five", "six", "seven", "eight"]
            host = socket.gethostname().split(".")[0]
            rank = names.index(host.split("-")[1]) if host.startswith("spark-") else 0
        for L in [int(x) for x in ARGS.layers.split(",")]:
            weight_sets.append((f"real-mtp{L}-r{rank}", lambda L=L, rank=rank: weights_real(L, rank)))
    ws = WS(max_rows=1024 * TOPK + E * 128 + 256)
    ws_new = WS(max_rows=1024 * TOPK + E * 128 + 256)
    n_cases = 0
    for wname, wfn in weight_sets:
        Wset = finish_weights(*wfn())
        e2e = e2e_runner(Wset)
        for M in m32_ms + big_ms:
            block_m = 32 if M <= int(os.environ.get("DSV41_DRAFT_M32_MAX_M", "160")) else 128
            kinds = [("random", "random", torch.int32)]
            if M in (1, 5, 32, 160, 161, 512) or ARGS.quick:
                kinds += [("zeros", "same3", torch.int32), ("max", "all", torch.int32), ("tiny", "random", torch.int64),
                          ("ties", "all", torch.int32)]
            for hk, ik, idt in kinds:
                aq, sc = make_hidden(M, hk, gen)
                ids, tw = make_ids(M, ik, gen, idt)
                ws.poison(0x7FC1)   # NaN pattern in the OLD workspaces
                old = moe(Wset["padded"], ws, aq, sc, ids, tw, compact=False, perm="stock", block_m=block_m)
                arms = [("fc1", True, "stock")]
                if block_m == 32:
                    arms += [("perm", False, "fused"), ("both", True, "fused")]
                for arm, cmp_, prm in arms:
                    ws_new.poison(0x5A5A)
                    new = moe(Wset["compact" if cmp_ else "padded"], ws_new, aq, sc, ids, tw, compact=cmp_, perm=prm,
                              block_m=block_m)
                    compare(old, new, ids, cmp_, f"{wname} M={M} x={hk} ids={ik}/{str(idt)[6:]} arm={arm} "
                                                 f"path={'m32' if block_m == 32 else 'dg128'}")
                    n_cases += 1
                if e2e is not None:
                    ws.poison(0x7FC1)
                    o_off = e2e("padded", ws, aq, sc, ids, tw, False)
                    ws_new.poison(0x5A5A)
                    o_on = e2e("compact", ws_new, aq, sc, ids, tw, True)
                    lbl = f"{wname} M={M} x={hk} ids={ik} arm=e2e-patched-apply"
                    r1, r2_ = first_mismatch(old["out"], o_off), first_mismatch(old["out"], o_on)
                    ok = r1 is None and r2_ is None
                    REPORT["cases"].append(dict(case=lbl, ok=ok, errors=[x for x in (r1, r2_) if x]))
                    n_cases += 1
                    if not ok:
                        FAIL.append(lbl)
                        log(f"FAIL {lbl}: off {r1} / on {r2_}")
        log(f"weights {wname}: {n_cases} cases so far, {len(FAIL)} failures, {time.time() - t0:.0f} s")

    # ---------------------------------------------------------------- CUDA graphs: capture old and new, replay
    Wset = finish_weights(*weights_random(gen))
    for M in ([1, 5, 32, 160] if not ARGS.quick else [5, 160]):
        aq0, sc0 = make_hidden(M, "random", gen)
        ids0, tw0 = make_ids(M, "random", gen)
        st = dict(aq=aq0.clone(), sc=sc0.clone(), ids=ids0.clone(), tw=tw0.clone())
        wsg_o, wsg_n = WS(M * TOPK + E * 32 + 64), WS(M * TOPK + E * 32 + 64)
        res = {}
        for name, W, cmp_, prm, wsg in (("old", Wset["padded"], False, "stock", wsg_o),
                                        ("new", Wset["compact"], True, "fused", wsg_n)):
            moe(W, wsg, st["aq"], st["sc"], st["ids"], st["tw"], compact=cmp_, perm=prm, block_m=32)  # warm/JIT
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s), torch.cuda.graph(g):
                res[name] = moe(W, wsg, st["aq"], st["sc"], st["ids"], st["tw"], compact=cmp_, perm=prm, block_m=32)
            torch.cuda.current_stream().wait_stream(s)
            res[name + "_g"] = g
        for rep in range(4):
            aq, sc = make_hidden(M, ["random", "zeros", "max", "ties"][rep], gen)
            ids, tw = make_ids(M, ["random", "same3", "all", "random"][rep], gen)
            st["aq"].copy_(aq), st["sc"].copy_(sc), st["ids"].copy_(ids), st["tw"].copy_(tw)
            wsg_o.poison(0x7FC1), wsg_n.poison(0x5A5A)
            res["old_g"].replay()
            res["new_g"].replay()
            torch.cuda.synchronize()
            eager = moe(Wset["padded"], ws, aq, sc, ids, tw, compact=False, perm="stock", block_m=32)
            compare(res["old"], res["new"], ids, True, f"graph M={M} replay={rep} old-graph vs new-graph")
            r = first_mismatch(eager["out"], res["new"]["out"])
            lbl = f"graph M={M} replay={rep} eager-old vs new-graph"
            REPORT["cases"].append(dict(case=lbl, ok=r is None, errors=[r] if r else []))
            if r:
                FAIL.append(lbl)
                log(f"FAIL {lbl}: {r}")
            n_cases += 2

    # ---------------------------------------------------------------- timing
    REPORT["jit"]["after"] = jit_listing()
    iters = 50 if ARGS.quick else ARGS.iters
    for M in [1, 5, 10, 20, 40, 80, 160]:
        aq, sc = make_hidden(M, "random", gen)
        ids, tw = make_ids(M, "random", gen)
        M_sum, _ = dgu.compute_aligned_M_and_alignment(M, TOPK, E, 32, None)
        a1q_perm = resize(ws.ws13.view(torch.float8_e4m3fn), (M_sum, H))
        live_act = LIVE_ACT_MIN is not None and M_sum >= LIVE_ACT_MIN
        res = dgu.deepgemm_moe_permute(aq=aq, aq_scale=sc, topk_ids=ids, local_num_experts=E, expert_map=None,
                                       expert_tokens_meta=None, aq_out=a1q_perm, alignment_m=32)
        f_res = dgu.dsv41_fused_permute(aq, sc, ids, E, resize(ws_new.ws13.view(torch.float8_e4m3fn), (M_sum, H)),
                                        M_sum, 32)
        mm1 = resize(ws.ws2, (M_sum, 768))
        mm1c = resize(ws_new.ws2, (M_sum, 576))

        def fc1(W, out, r):
            with K32.alignment_scope():
                K32.gemm((r[0], r[1]), (W[0].view(torch.int8), W[1]), out, r[2], recipe_a=(1, 128),
                         recipe_b=(1, 32))

        fns = {
            "permute_stock": lambda: dgu.deepgemm_moe_permute(aq=aq, aq_scale=sc, topk_ids=ids, local_num_experts=E,
                                                              expert_map=None, expert_tokens_meta=None,
                                                              aq_out=a1q_perm, alignment_m=32),
            "permute_fused": lambda: dgu.dsv41_fused_permute(aq, sc, ids, E, f_res[0], M_sum, 32),
            "fc1_n768_stock_scales": lambda: fc1(Wset["padded"], mm1, res),
            "fc1_n576_stock_scales": lambda: fc1(Wset["compact"], mm1c, res),
            "fc1_n576_packed_scales": lambda: fc1(Wset["compact"], mm1c, f_res),
            "moe_old": lambda: moe(Wset["padded"], ws, aq, sc, ids, tw, compact=False, perm="stock", block_m=32),
            "moe_new": lambda: moe(Wset["compact"], ws_new, aq, sc, ids, tw, compact=True, perm="fused", block_m=32),
        }
        row = {"M": M, "M_sum": M_sum, "live_act": live_act}
        for k, fn in fns.items():
            row[k + "_warm_us"] = round(timeit(fn, iters, False), 2)
            row[k + "_cold_us"] = round(timeit(fn, iters, True), 2)
        REPORT["timing"].append(row)
        log("timing " + json.dumps(row))
    REPORT["n_cases"] = n_cases
    REPORT["failures"] = FAIL
    REPORT["seconds"] = round(time.time() - t0, 1)
    new_jit = {k: sorted(set(REPORT["jit"]["after"][k]) - set(REPORT["jit"]["before"][k])) for k in ("m32", "dg128")}
    REPORT["jit"]["new_entries"] = new_jit
    log(f"JIT entries created by this run: {new_jit}")
    if ARGS.json:
        json.dump(REPORT, open(ARGS.json, "w"), indent=1)
    log(f"{'PASS' if not FAIL else 'FAIL'}: {n_cases} bitwise cases, {len(FAIL)} failures, {REPORT['seconds']} s")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
