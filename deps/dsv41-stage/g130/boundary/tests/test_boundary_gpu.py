#!/usr/bin/env python3
"""g130 boundary kit, GPU test. STOPPED-SERVICE WINDOW ONLY, one GPU, about 2 GB. NOT RUN YET: syntax-checked only, expect a
first-run fix or two. Prints JSON lines and a last VERDICT line. Exit 0 = GO on every part asked for, 1 = KILL, 2 = HOLD.

  source ~/venvs/vllm032/bin/activate
  CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH python3 tests/test_boundary_gpu.py --tp-rank R

Part a, context KV, real mtp.{0,1,2} wq_a + wkv weights:
  int16 view of stock qr_kv[..., 1280:] (all six tactics, N=1792) against the N=1536 fold and three N=512 views (all six tactics),
  also after RMSNorm with the real kv_norm weights, M = 1,2,4,6,8,12,16,24 and the graph rows 6 x reqs, >200 replays per M, plus
  CUDA-graph replays, plus CUDA-graph timing of the whole block with L2 made cold between replays.
Part b, Markov glue, real markov_head.embed / head weights and real lm_head shards, TP=8 emulated on one device:
  int32 view of the 16-byte pair, draft ids and int16 view of the next embed, stock ops against the fold kernels, tiles 1 and 16,
  num_reqs 1..32 and >200 replays per M in the list, injected ties (inside a tile, across tiles, across ranks, -0 vs +0), -inf
  rows, +inf, NaN and subnormal rows; CUDA-graph replays; CUDA-graph timing of the 5-position chain, stock against fold.
Audit plumbing: the device counters work inside a captured graph; a forced difference is counted.

Thresholds (GPU, one node, medians; a pinned serving pair still decides promotion):
  a  GO:   0 differing words anywhere AND fold block at M=6 saves >= 90 us. Fold GEMM alone <= 50 us.
     HOLD: 0 differing words, saves 70-90 us (or split saves >= 70 us while merge does not: serve DSV41_CTXKV_FOLD=split).
     KILL: any differing word on the pinned tactic, or saving < 70 us.
  b  GO:   0 differing words in every normal class AND chain at num_reqs=1 saves >= 45 us. Use the faster of tiles 1 / 16.
     HOLD: saves 25-45 us, or a difference ONLY in the NaN / subnormal classes (never served until the user reviews it).
     KILL: any difference in a normal class, in real-weight rows or in a graph replay, or saving < 25 us.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
KIT = HERE.parent
HOME = Path.home()
MODEL = HOME / "models/dsv41/uncensored-fp8"
GRAPH_REQS = (1, 2, 3, 4, 5, 7, 8, 10, 12, 13, 15, 16, 18, 20, 21, 23, 24, 26, 28, 29, 31, 32)
BASIC_M = (1, 2, 4, 6, 8, 12, 16, 24)

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--tp-rank", type=int, default=0)
ap.add_argument("--parts", default="a,b")
ap.add_argument("--replays", type=int, default=256, help="replays per M in BASIC_M, a quarter on the other row counts")
ap.add_argument("--time-replays", type=int, default=300)
ap.add_argument("--force", action="store_true", help="skip the stopped-service check (never on a serving node)")
a = ap.parse_args()
PARTS = a.parts.split(",")
if not PARTS or not set(PARTS) <= {"a", "b"}:
    ap.error(f"--parts takes a and/or b, got {a.parts!r}")


def emit(**kw):
    print(json.dumps(kw), flush=True)


if not a.force:
    st = subprocess.run(["systemctl", "--user", "is-active", "dsv41-rank"], capture_output=True, text=True)
    assert st.stdout.strip() in ("inactive", "failed"), f"dsv41-rank is {st.stdout.strip()!r}: stopped-service window only"
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, check=True)
    assert not apps.stdout.strip(), "GPU busy: " + apps.stdout.strip()

os.environ.update(DSV41_CTXKV_FOLD="1", DSV41_MK_FOLD="1", DSV41_BOUNDARY_AUDIT_DIR=str(KIT / "results"))
os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda/bin/ptxas")   # as the launcher

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

assert torch.cuda.get_device_capability() == (12, 1)
DEV = torch.device("cuda:0")
torch.set_grad_enabled(False)


def load_side(tiles: str = "16"):
    os.environ["DSV41_MK_FOLD_TILES"] = tiles
    p = KIT / "site/vllm/models/deepseek_v41/nvidia/dsv41_boundary.py"
    spec = importlib.util.spec_from_file_location("dsv41_boundary_gpu", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


B = load_side()
INDEX = json.loads((MODEL / "model.safetensors.index.json").read_text())["weight_map"]
T0 = time.time()


def tensor(name: str, rows=None) -> torch.Tensor:
    with safe_open(str(MODEL / INDEX[name]), framework="pt") as f:
        if rows is None:
            t = f.get_tensor(name)
        else:
            t = f.get_slice(name)[rows[0]:rows[1]]
    return t.to(DEV)


def graph_of(fn):
    """CUDA graph of fn(); fn must only touch persistent tensors and may allocate from the graph pool."""
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    return g, out


def timed(graphs: dict, n: int, cold) -> dict:
    """Interleaved replays, median us per graph. cold() runs before each replay and is not timed."""
    res = {k: [] for k in graphs}
    for _ in range(n):
        for k, g in graphs.items():
            if cold is not None:
                cold()
            torch.cuda.synchronize()
            t = time.perf_counter()
            g.replay()
            torch.cuda.synchronize()
            res[k].append((time.perf_counter() - t) * 1e6)
    return {k: round(statistics.median(v), 2) for k, v in res.items()}


COLD = torch.zeros((64 << 20) // 4, dtype=torch.int32, device=DEV)


def make_cold():
    COLD.add_(1)


verdicts: dict[str, str] = {}


def serving_table() -> dict[int, int]:
    """Serving autotune cache for (M, 5120) x (5120, 1792): M bucket -> tactic. Empty if no cache is found on this node."""
    import glob
    import re

    cands = sorted(glob.glob(str(HOME / f".cache/vllm/flashinfer_autotune_cache/*/121a/*/autotune_configs_dp0_rank{a.tp_rank}.json")),
                   key=os.path.getmtime)
    tab: dict[int, int] = {}
    for path in reversed(cands):
        d = json.loads(Path(path).read_text())
        for k, v in d.items():
            m = re.match(r"\('mxfp8_gemm', 'CutlassMxfp8GemmRunner', \(\((\d+), 5120\), \(5120, 1792\)", k)
            if m:
                tab[int(m.group(1))] = int(v[1])
        if tab:
            break
    return tab


# ===================================================================================================== part a
def part_a():
    from flashinfer.gemm.gemm_base import DEFAULT_WORKSPACE_SIZE, get_cutlass_mxfp8_gemm_module
    from flashinfer.utils import _get_cache_buf
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize, swizzle_mxfp8_scale
    from vllm.utils import flashinfer as vfi

    major, _ = torch.cuda.get_device_capability(DEV)
    runner = get_cutlass_mxfp8_gemm_module(major).cutlass_mxfp8_gemm_runner()
    ws = _get_cache_buf("mm_mxfp8_workspace", DEFAULT_WORKSPACE_SIZE, DEV)
    tactics = list(runner.get_valid_tactics(None, None))
    emit(part="a", tactics=tactics)

    class Lin:
        def __init__(self, i):
            q_w, kv_w = tensor(f"mtp.{i}.attn.wq_a.weight"), tensor(f"mtp.{i}.attn.wkv.weight")
            # Checkpoint scales are e8m0 in 32 x 32 blocks, [40, 160] and [16, 160]. Same as modelopt KMxfp8Static.scaled_loader:
            # raw bytes, one row per weight row. Same as the CUTLASS kernel's post-load: [:N, :K // 32], then swizzle.
            q_s = tensor(f"mtp.{i}.attn.wq_a.scale").view(torch.uint8).repeat_interleave(32, dim=0)
            kv_s = tensor(f"mtp.{i}.attn.wkv.scale").view(torch.uint8).repeat_interleave(32, dim=0)
            self.weight = torch.cat([q_w, kv_w], dim=0).contiguous()
            N, K = self.weight.shape
            s2 = torch.cat([q_s, kv_s], dim=0)
            assert s2.shape == (N, K // 32), s2.shape
            self.weight_scale = swizzle_mxfp8_scale(s2[:N, : K // 32].contiguous(), M=N, K=K).contiguous()
            assert self.weight_scale.numel() == 286720, self.weight_scale.shape
            self.bias = None
            self.gather_output = False
            self.tp_size = 1

        def __call__(self, x):
            xq, xs = mxfp8_e4m3_quantize(x.view(-1, self.weight.shape[1]), is_sf_swizzled_layout=True)
            out = vfi.mm_mxfp8(xq, self.weight.t(), xs, self.weight_scale, out_dtype=x.dtype, backend="cutlass")
            return out, None

        def tactic(self, xq, xs, t):
            out = torch.empty((xq.shape[0], self.weight.shape[0]), dtype=torch.bfloat16, device=DEV)
            runner(inputs=[xq, self.weight.t(), xs, self.weight_scale, torch.bfloat16, out, ws], tactic=t)
            return out

    def kv_norm_of(i):
        # A CustomOp needs a current vLLM config to build. eps is the checkpoint's rms_norm_eps, as attention.py passes it.
        with set_current_vllm_config(VllmConfig()):
            norm = RMSNorm(512, 1e-20).to(DEV)
        norm.weight.data.copy_(tensor(f"mtp.{i}.attn.kv_norm.weight").to(norm.weight.dtype))
        return norm.to(torch.bfloat16)

    layers = []
    for i in range(3):
        layers.append(SimpleNamespace(attn=SimpleNamespace(fused_wqa_wkv=Lin(i), q_lora_rank=1280, kv_norm=kv_norm_of(i))))
    N, K = layers[0].attn.fused_wqa_wkv.weight.shape
    assert (N, K) == (1792, 5120), (N, K)
    tm = B.parse_map(B.CTX_MAP_RAW)
    merge = B.CtxKvFold(layers, "merge", runner=runner, workspace=ws)
    split = B.CtxKvFold(layers, "split", runner=runner, workspace=ws)
    assert merge.parts[0][1].numel() == 3 * 4 * 128 * 160, merge.parts[0][1].shape

    def quant(x):
        return mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)

    def inputs(kind, M, g):
        x = torch.randn(M, K, generator=g)
        if kind == "spread":
            x = x * torch.exp2(torch.randint(-6, 7, (M, K // 32), generator=g).float()).repeat_interleave(32, dim=1)
            x[:, ::97] = 0.0
            x[torch.arange(M), torch.randint(0, K, (M,), generator=g)] = 300.0
        elif kind == "rms":
            x = x / x.pow(2).mean(dim=1, keepdim=True).sqrt()
        elif kind == "zeros":
            x[M // 2 :] = 0.0
        elif kind == "onehot":
            x = torch.zeros(M, K)
            x[torch.arange(M), torch.randint(0, K, (M,), generator=g)] = 1.0
        elif kind == "tiny":
            x = x * 1e-30
        return x.to(torch.bfloat16).to(DEV)

    ms = sorted(set(BASIC_M) | {6 * r for r in GRAPH_REQS} | {64, 128})
    kinds = ("spread", "rms", "zeros", "onehot", "tiny")
    bad = {"merge": 0, "split": 0, "post_norm": 0}
    table = serving_table()
    emit(part="a", serving_tactics_n1792=sorted(table.items()))

    def serving(M):
        ups = [b for b in table if b >= M]
        return table[min(ups)] if ups else tm.get(M, 1)

    merge.self_check(layers, quant, tm, DEV)
    emit(part="a", section="self_check", pinned=sorted(merge.ok), rejected=merge.rejected)
    bad["merge"] += len(merge.rejected)
    words = 0
    t_a = time.time()
    for M in ms:
        n = a.replays if M in BASIC_M else max(a.replays // 4, 16)
        t = tm.get(M, 1)
        cases = 0
        for rep in range(n):
            g = torch.Generator(device="cpu").manual_seed(M * 100003 + rep)
            x = inputs(kinds[rep % len(kinds)], M, g)
            xq, xs = quant(x)
            # first replays of each kind: every tactic, same tactic on both shapes. The rest: the pinned fold against stock N=1792 at
            # the tactic the serving autotune cache gives this M, and against the stock vLLM call (tactic 0 here, no cache loaded).
            tset = tactics if rep < 2 * len(kinds) else [t]
            for tac in tset:
                folded = merge.project(xq, xs, tac)
                spl = split.project(xq, xs, tac)
                for i, ly in enumerate(layers):
                    lin = ly.attn.fused_wqa_wkv
                    ref = lin.tactic(xq, xs, tac if rep < 2 * len(kinds) else serving(M))[..., 1280:]
                    if rep == 2 * len(kinds):
                        bad["merge"] += int((lin(x)[0][..., 1280:].view(torch.int16) != folded[i].view(torch.int16)).sum())
                    r16 = ref.view(torch.int16)
                    bad["merge"] += int((folded[i].view(torch.int16) != r16).sum())
                    bad["split"] += int((spl[i].view(torch.int16) != r16).sum())
                    if rep % 16 == 0:
                        n0, n1, n2 = ly.attn.kv_norm(ref), ly.attn.kv_norm(folded[i]), ly.attn.kv_norm(spl[i])
                        bad["post_norm"] += int((n0.view(torch.int16) != n1.view(torch.int16)).sum())
                        bad["post_norm"] += int((n0.view(torch.int16) != n2.view(torch.int16)).sum())
                    words += r16.numel()
            cases += 1
        if bad["merge"] or bad["split"]:
            emit(part="a", M=M, replays=cases, differing=dict(bad))
    emit(part="a", section="bitwise", ms=ms, words=words, differing=bad, seconds=round(time.time() - t_a))

    # graph replays: the fold recorded, new inputs copied into a persistent buffer, compared with the eager stock output
    graph_bad = 0
    for M in (6, 12, 24, 48):
        xbuf = torch.zeros((M, K), dtype=torch.bfloat16, device=DEV)
        t = tm[M]

        def rec():
            xq, xs = quant(xbuf)
            return merge.project(xq, xs, t)

        gr, outs = graph_of(rec)
        for rep in range(max(64, a.replays // 2)):
            xbuf.copy_(inputs(kinds[rep % len(kinds)], M, torch.Generator().manual_seed(rep + 7 * M)))
            gr.replay()
            for i, ly in enumerate(layers):
                ref = ly.attn.fused_wqa_wkv(xbuf)[0][..., 1280:]
                graph_bad += int((outs[i].view(torch.int16) != ref.view(torch.int16)).sum())
    emit(part="a", section="graph", differing=graph_bad)

    # timing, whole block as served: quant + GEMM(s) + rms_norm, without the insert (same in all arms)
    M = 6
    xbuf = inputs("rms", M, torch.Generator().manual_seed(5))

    def stock_block():
        res = []
        for ly in layers:
            xq, xs = quant(xbuf)
            out = ly.attn.fused_wqa_wkv.tactic(xq, xs, serving(M))
            res.append(ly.attn.kv_norm(out[..., 1280:]))
        return res

    def fold_block(f, tac):
        def run():
            xq, xs = quant(xbuf)
            return [ly.attn.kv_norm(k) for ly, k in zip(layers, f.project(xq, xs, tac))]
        return run

    def gemm_only(f, tac):
        xq, xs = quant(xbuf)
        return lambda: f.project(xq, xs, tac)

    graphs = {"stock": graph_of(stock_block)[0]}
    for tac in tactics:
        graphs[f"merge_t{tac}"] = graph_of(fold_block(merge, tac))[0]
        graphs[f"split_t{tac}"] = graph_of(fold_block(split, tac))[0]
    graphs["merge_gemm_t1"] = graph_of(gemm_only(merge, 1))[0]
    graphs["noop"] = graph_of(lambda: torch.zeros((1,), device=DEV).add_(1))[0]
    tm_us = timed(graphs, a.time_replays, make_cold)
    emit(part="a", section="timing", M=M, us=tm_us)
    base = tm_us["stock"]
    pinned = base - tm_us[f"merge_t{tm[6]}"]
    best_merge = base - min(tm_us[f"merge_t{t}"] for t in tactics)
    best_split = base - min(tm_us[f"split_t{t}"] for t in tactics)
    gemm = tm_us["merge_gemm_t1"] - tm_us["noop"]
    any_bad = bad["merge"] or bad["post_norm"] or graph_bad
    if any_bad:
        v = "KILL merge differs" if not (bad["split"] == 0 and graph_bad == 0 and best_split >= 70) else "HOLD merge differs, split clean"
    elif pinned >= 90 and gemm <= 50:
        v = "GO"
    elif pinned >= 70 or (bad["split"] == 0 and best_split >= 70):
        v = "HOLD"
    else:
        v = "KILL saving too small"
    if v == "GO" and not table:
        v = "HOLD no serving autotune table on this node, stock tactic not compared"
    emit(part="a", saving_pinned_us=round(pinned, 1), saving_best_merge_us=round(best_merge, 1),
         saving_best_split_us=round(best_split, 1), split_differing=bad["split"], gemm_us=round(gemm, 1), verdict=v)
    verdicts["a"] = v


# ===================================================================================================== part b
def stock_winner(gathered, tp, M, w1):
    g4 = gathered.view(tp, M, 4)
    best = g4[:, :, 0].argmax(dim=0, keepdim=True)
    tok = g4[:, :, 1].gather(0, best).squeeze(0).to(torch.int64)
    return tok, w1[tok]


NORMAL = ("real", "tie", "tie_far", "tie_rank", "zero", "zero_rev", "neginf", "inf")
ODD = ("nan", "subnormal")


def inject(x: torch.Tensor, kind: str, g, peers: list[torch.Tensor] | None):
    M, V = x.shape
    for r in range(M):
        a_, b_ = torch.randint(0, V, (2,), generator=g).tolist()
        top = float(x[r].float().max()) + 1.0
        if kind == "tie":
            x[r, a_ // 1024 * 1024] = top
            x[r, a_ // 1024 * 1024 + 7] = top
        elif kind == "tie_far":
            x[r, 0] = top
            x[r, V - 1] = top
            x[r, V // 2] = top
        elif kind in ("zero", "zero_rev"):
            lo, hi = min(a_, b_), max(a_, b_)
            x[r] = -3.0
            x[r, lo] = -0.0 if kind == "zero" else 0.0
            x[r, hi] = 0.0 if kind == "zero" else -0.0
        elif kind == "neginf":
            x[r] = float("-inf")
        elif kind == "inf":
            x[r, a_] = float("inf")
            x[r, b_] = float("inf")
        elif kind == "nan":
            x[r, a_] = float("inf")
            x[r, b_] = float("nan")
        elif kind == "subnormal":
            x[r] = -3.0
            x[r, min(a_, b_)] = 2.0 ** -130
            x[r, max(a_, b_)] = 2.0 ** -129
    return x


def part_b():
    w1 = tensor("mtp.2.markov_head.embed.weight")
    w2 = tensor("mtp.2.markov_head.head.weight")
    assert w1.dtype == torch.bfloat16 and w1.shape[1] == 256, (w1.dtype, w1.shape)
    TP, Vl = 8, 16160
    head = [tensor("head.weight", (r * Vl, (r + 1) * Vl)).to(torch.bfloat16) for r in range(TP)]
    assert head[0].shape == (Vl, 5120), head[0].shape
    emit(part="b", w1=str(tuple(w1.shape)), w2=str(tuple(w2.shape)), seconds=round(time.time() - T0))
    mods = {16: B, 1: load_side("1")}
    bad = {k: 0 for k in NORMAL + ODD}
    cases = 0
    t_b = time.time()
    ms = sorted(set(BASIC_M) | set(range(1, 33)))
    for M in ms:
        n = a.replays if M in BASIC_M else max(a.replays // 8, 8)
        for rep in range(n):
            g = torch.Generator(device="cpu").manual_seed(M * 1000003 + rep)
            kind = (NORMAL + ODD)[rep % len(NORMAL + ODD)]
            hid = torch.randn(M, 5120, generator=g).to(torch.bfloat16).to(DEV)
            tok_prev = torch.randint(0, w1.shape[0], (M,), generator=g).to(DEV)
            emb = w1[tok_prev]
            pairs, shared = [], None
            for r in range(TP):
                base = hid @ head[r].t()
                bias = emb @ w2[r * Vl : (r + 1) * Vl].t()
                x = (base + bias).cpu().clone()   # the torch bf16 add stays in the fold, so it is common to both arms
                if kind != "real":
                    x = inject(x, kind, g, None)
                if kind == "tie_rank":
                    x[:] = -5.0
                    c = torch.randint(0, Vl, (M,), generator=g)
                    x[torch.arange(M), c] = 9.0
                x = x.to(DEV)
                ref = B._stock_pair(x, r * Vl)
                for tiles, mod in mods.items():
                    p = mod.mk_local_pair(x, r * Vl, tiles)
                    d = int((p.view(torch.int32) != ref.view(torch.int32)).sum())
                    if d:
                        bad[kind] += d
                        emit(part="b", diff="pair", M=M, rep=rep, kind=kind, tiles=tiles, rank=r, words=d)
                pairs.append(ref)
            G = torch.cat(pairs, dim=0)
            tok, emb_next = stock_winner(G, TP, M, w1)
            dt = torch.zeros((M, 5), dtype=torch.int64, device=DEV)
            col = rep % 5
            e = B.mk_winner(G, TP, M, dt, col, w1, True)
            d = int((dt[:, col] != tok).sum()) + int((e.view(torch.int16) != emb_next.view(torch.int16)).sum())
            d2 = dt.clone()
            d2[:, col] = 0
            d += int(d2.ne(0).sum())   # nothing else in draft_tokens is touched
            if d:
                bad[kind] += d
                emit(part="b", diff="winner", M=M, rep=rep, kind=kind, words=d)
            cases += 1
    emit(part="b", section="bitwise", cases=cases, differing=bad, seconds=round(time.time() - t_b))

    # graph replays + timing of the chain after the add. The all-gather is a local repeat in both arms.
    gbad = 0
    timing = {}
    for M in (1, 2, 4, 8, 16, 32):
        ibuf = [torch.zeros((M, Vl), dtype=torch.bfloat16, device=DEV) for _ in range(5)]
        dt = torch.zeros((M, 5), dtype=torch.int64, device=DEV)
        dt_s = torch.zeros((M, 5), dtype=torch.int64, device=DEV)

        def stock_chain():
            out = None
            for i in range(5):
                pair = B._stock_pair(ibuf[i], 0)
                gathered = pair.repeat(TP, 1)
                tok, e = stock_winner(gathered, TP, M, w1)
                dt_s[:, i] = tok
                out = e
            return out

        def fold_chain(mod, tiles):
            def run():
                e = None
                for i in range(5):
                    pair = mod.mk_local_pair(ibuf[i], 0, tiles)
                    gathered = pair.repeat(TP, 1)
                    e = mod.mk_winner(gathered, TP, M, dt, i, w1, i < 4)
                return e
            return run

        gs = {"stock": graph_of(stock_chain)[0]}
        for tiles, mod in mods.items():
            gs[f"fold{tiles}"] = graph_of(fold_chain(mod, tiles))[0]
        for rep in range(max(64, a.replays // 2)):
            g = torch.Generator().manual_seed(rep * 31 + M)
            for i in range(5):
                x = (torch.randint(-20, 21, (M, Vl), generator=g).float() / 2).to(torch.bfloat16)
                ibuf[i].copy_(x.to(DEV))
            gs["stock"].replay()
            ref = dt_s.clone()
            for tiles in mods:
                dt.zero_()
                gs[f"fold{tiles}"].replay()
                gbad += int((dt != ref).sum())
        timing[M] = timed(gs, a.time_replays if M in (1, 4) else a.time_replays // 3, None)
        emit(part="b", section="timing", M=M, us=timing[M])
    emit(part="b", section="graph", differing=gbad)

    normal_bad = sum(bad[k] for k in NORMAL) + gbad
    odd_bad = sum(bad[k] for k in ODD)
    t1 = timing[1]
    save = t1["stock"] - min(t1["fold1"], t1["fold16"])
    best = 1 if t1["fold1"] <= t1["fold16"] else 16
    if normal_bad:
        v = "KILL differs"
    elif save < 25:
        v = "KILL saving too small"
    elif odd_bad:
        v = "HOLD differs in NaN/subnormal class only, user review"
    elif save >= 45:
        v = "GO"
    else:
        v = "HOLD saving 25-45 us"
    emit(part="b", saving_us_num_reqs_1=round(save, 1), best_tiles=best, verdict=v)
    verdicts["b"] = v


# ===================================================================================================== audit plumbing
def part_audit():
    ctr = torch.zeros((8,), dtype=torch.int64, device=DEV)
    x = torch.zeros((4, 64), dtype=torch.int16, device=DEV)
    y = torch.zeros((4, 64), dtype=torch.int16, device=DEV)

    def rec():
        B._bump(ctr, 3, 1, 4, (x != y).sum())

    g, _ = graph_of(rec)
    for _ in range(10):
        g.replay()
    c0 = ctr.tolist()
    y[1, 7] = 1
    g.replay()
    c1 = ctr.tolist()
    ok = c0[3] == 10 and c0[4] == 40 and c0[5] == 0 and c1[5] == 1
    emit(part="audit", counters_clean=c0, counters_forced=c1, ok=ok)
    verdicts["audit"] = "GO" if ok else "KILL audit counters do not work in a graph"


if "a" in PARTS:
    part_a()
if "b" in PARTS:
    part_b()
part_audit()
# The replay count is part of the gate, so a smoke run can never report GO.
if a.replays < 200 or a.time_replays < 200:
    verdicts = {k: f"HOLD smoke run, replays {a.replays} / {a.time_replays} under 200: {v}" if v == "GO" else v for k, v in verdicts.items()}
print("VERDICT " + json.dumps(verdicts), flush=True)
vals = list(verdicts.values())
sys.exit(1 if any(v.startswith("KILL") for v in vals) else 2 if any(v.startswith("HOLD") for v in vals) else 0)
