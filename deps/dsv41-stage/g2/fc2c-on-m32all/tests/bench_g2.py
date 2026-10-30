#!/usr/bin/env python3
"""g2 fc2c-on-m32all microbench: FC2-compact tail on the m32-bw 'all' FC2 tree vs the LIVE 'all' FC2 (all-d05),
recorded decode routes, L2-cold, CUDA graphs.

STOPPED-SERVICE WINDOW ONLY (same refusal checks and private caches as test_g2_unit.py; run that first: it proves
the arms bitwise before anything is timed, and ops/prep-node.sh pre must have made jit-test/).

  ~/venvs/vllm032/bin/python ~/dsv41-stage/g2/fc2c-on-m32all/tests/bench_g2.py                 # rank 0 slice
  ~/venvs/vllm032/bin/python .../bench_g2.py --tp-rank 7 --Ms 1,6,12,24,48 --trials 25

Method (= g1 fc2c-d05 bench_fc2c.py, whose numbers on the plain D05 kernel this extends): real checkpoint weights
of one TP rank slice (vLLM RoutedExperts loader), recorded 10-21 decode routes (rank-0 recorder, all 40 layers;
"u6xk" = k real 6-row C1 verify windows, M=1 = first row of a window, "c4rec" = real 4-user batches, M=12 and 24 =
the 4-user verify widths). One CUDA graph per arm runs 40 sequential layer calls cycling the K loaded layers with
each layer's own routes (weights stay L2-cold); arms interleaved, order alternating per trial; median of --trials.
  moe  : full DeepGemmFP4Experts.apply (FC1, SiLU live rows, FC2, gather) = the serving MoE call
  fc2  : FC2 grouped GEMM only, on that apply's own FC2 inputs (scatter32 layout, LIVE_ACT activation)
Arms: live (serving today: launcher 395bf5b2, G1=all FC2 = all-d05), fc2c (DSV41_MOE_FC2C=1 serving gate: M>=2
tokens compact, M=1 live), fc2c_all (compact at every M; = DSV41_MOE_FC2C_MIN_TOKENS=1).
Projection: delta ms/step = (live - arm) per-layer us x 40 target MoE layers / 1000 (drafter not included), and
x0.96 (1 user) / x0.90 (4 users) bench-to-serving scale (m32-bw bench/project.py) as the "serving" figure.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent
HOME = Path.home()
EXP = HOME / "venvs/vllm032/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/experts"
LIVE_M32_SHA = "db659d809cbcc3949f94dd810d9f00359f1770af082aadd8f7037816371f02da"
BAK = ".bak-g2-fc2c-on-m32all"
p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--tp-rank", type=int, default=0)
p.add_argument("--layers", default="0,13,26,39")
p.add_argument("--Ms", default="1,6,12,24,48")
p.add_argument("--sets", type=int, default=2, help="u6xk route sets per M (plus c4rec where recorded)")
p.add_argument("--trials", type=int, default=25)
p.add_argument("--modes", default="moe,fc2")
p.add_argument("--arms", default="live,fc2c,fc2c_all")
p.add_argument("--routes", default=str(STAGE / "bench/routes"))
p.add_argument("--uniform", action="store_true", help="also time uniform random routes (high expert diversity)")
p.add_argument("--profile", default="6,24", help="M values whose moe graphs get a per-kernel profile")
p.add_argument("--step-1u-code-ms", type=float, default=29.94, help="live matched C1 coding ms/step (gw1-U-2)")
p.add_argument("--step-1u-prose-ms", type=float, default=25.84, help="live matched C1 prose ms/step (gw1-U-2)")
p.add_argument("--step-4u-ms", type=float, default=45.0, help="4-user decode step estimate (50.4 x m32/clock gains)")
p.add_argument("--force", action="store_true")
a = p.parse_args()


def emit(**kw):
    print(json.dumps(kw), flush=True)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if not a.force:
    st = subprocess.run(["systemctl", "--user", "is-active", "dsv41-rank"], capture_output=True, text=True)
    assert st.stdout.strip() in ("inactive", "failed", "unknown", ""), "dsv41-rank is active: " + st.stdout
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, check=True)
    assert not apps.stdout.strip(), "GPU occupied: " + apps.stdout.strip()

JT = STAGE / "jit-test"
for sub in ("g1root/jit/all-base", "g1root/jit/all-d05", "fc2c-all", "dg", "triton"):
    assert (JT / sub).is_dir(), f"missing {JT / sub}: run ops/prep-node.sh pre first"
os.environ.update(
    DSV41_SM121_TARGET_M32="1", DSV41_TARGET_M32_MAX_M="96", DSV41_SM121_TARGET_M64="1",
    DSV41_MOE_COMPACT="1", DSV41_MOE_LIVE_ACT="1", VLLM_USE_DEEP_GEMM="1",
    CUDA_HOME="/usr/local/cuda", TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas",
    DSV41_MOE_FC2_D05="1", DSV41_MOE_M32_G1="all", DSV41_MOE_FC2C="1",
    DSV41_MOE_FC2C_JIT_DIR=str(JT / "fc2c-all"), DG_JIT_CACHE_DIR=str(JT / "dg"), TRITON_CACHE_DIR=str(JT / "triton"))
for k in ("DSV41_MOE_PDL_EARLY_B", "DSV41_MOE_FC2C_MIN_TOKENS", "DSV41_MOE_FC2C_NATIVE_DIR", "VLLM_MARLIN_INPUT_DTYPE",
          "DSV41_MOE_FC2_D05_NATIVE_DIR", "DSV41_MOE_FC2_D05_JIT_DIR"):
    os.environ.pop(k, None)
os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")
os.environ["DSV41_TARGET_M32_NATIVE_DIR"] = str(HOME / "dsv41-stage/m32/target-m32-native-v1")

import torch  # noqa: E402

assert torch.cuda.get_device_capability() == (12, 1)
DEV = torch.device("cuda")
import vllm.model_executor.layers.fused_moe.experts as _experts_pkg  # noqa: E402,F401


def load_as(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


PKG = "vllm.model_executor.layers.fused_moe.experts"
m32 = load_as(f"{PKG}.sm121_target_m32", STAGE / "site/vllm/model_executor/layers/fused_moe/experts/sm121_target_m32.py")
INSTALLED = EXP / "sm121_target_m32.py"
LIVE_SRC = INSTALLED if sha(INSTALLED) == LIVE_M32_SHA else INSTALLED.with_name(INSTALLED.name + BAK)
assert sha(LIVE_SRC) == LIVE_M32_SHA, f"no db659d80 live site file ({LIVE_SRC})"
live_m32 = load_as("g2live_sm121_target_m32", LIVE_SRC)
m32._G1_ROOT = JT / "g1root"

from vllm.model_executor.layers.fused_moe.activation import MoEActivation  # noqa: E402
from vllm.model_executor.layers.fused_moe.config import (  # noqa: E402
    FusedMoEConfig, FusedMoEParallelConfig, RoutingMethodType)
from vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe import DeepGemmFP4Experts  # noqa: E402
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (  # noqa: E402
    Mxfp4MoeBackend, make_mxfp4_moe_quant_config)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (  # noqa: E402
    deepgemm_post_process_weight_scale_block, per_token_group_quant_fp8)
from vllm.model_executor.layers.fused_moe import deep_gemm_utils as utils  # noqa: E402

E, H, I, P, TOPK = 384, 5120, 288, 384, 6
SILU = MoEActivation.SILU
MODEL = HOME / "models/dsv41/uncensored-fp8"
INDEX = json.loads((MODEL / "model.safetensors.index.json").read_text())["weight_map"]
FC2_BYTES = {"live": H * (P // 2 + P // 32), "fc2c": H * (128 + 32 + P // 32)}   # weights + scales per expert

k_serv = m32.get_target_m32_kernel()                   # the serving construction (G1=all, D05=1, FC2C=1)
assert k_serv.fc2c is not None and k_serv.g1_la and k_serv.fc2c.min_rows == m32.fc2c_min_rows(2)
fc_serv = k_serv.fc2c
fc_all = m32.FC2Compact(fc_serv.native, "1", m32.fc2c_min_rows(1))
fc_all.entries = fc_serv.entries                       # one tail per weight, shared by both gates


def with_la(k):
    k.g1_la = True
    return k


KERN = {"live": with_la(live_m32.TargetM32Kernel(k_serv.native, k_serv.fc2_native, "1")),
        "fc2c": k_serv,
        "fc2c_all": with_la(m32.TargetM32Kernel(k_serv.native, k_serv.fc2_native, "1", None, fc_all))}
ARMS = [x for x in a.arms.split(",") if x]
assert ARMS[0] == "live" and set(ARMS) <= set(KERN), ARMS

parallel = FusedMoEParallelConfig(tp_size=8, pcp_size=1, dp_size=1, ep_size=1, tp_rank=0, pcp_rank=0, dp_rank=0,
                                  ep_rank=0, sp_size=1, use_ep=False, all2all_backend="naive", enable_eplb=False)
cfg = FusedMoEConfig(num_experts=E, experts_per_token=TOPK, hidden_dim=H, intermediate_size=I * 8,
                     num_local_experts=E, num_logical_experts=E, activation=SILU, device="cuda",
                     routing_method=list(RoutingMethodType)[0], moe_parallel_config=parallel,
                     in_dtype=torch.bfloat16, swiglu_limit=10.)
cfg.intermediate_size_per_partition = P


def sf(raw, n, k):
    return deepgemm_post_process_weight_scale_block(ws=raw, mn=n, k=k, quant_block_shape=(1, 32), num_groups=E)


def load_layer(layer):
    from safetensors import safe_open
    from types import SimpleNamespace
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    loader = object.__new__(RoutedExperts)
    torch.nn.Module.__init__(loader)
    loader.moe_config = SimpleNamespace(is_act_and_mul=True, tp_shard_with_padding=False,
                                        moe_parallel_config=SimpleNamespace(tp_size=8))
    w1 = torch.zeros(E, 2 * P, H // 2, dtype=torch.uint8)
    s1 = torch.zeros(E, 2 * P, H // 32, dtype=torch.uint8)
    w2 = torch.zeros(E, H, P // 2, dtype=torch.uint8)
    s2 = torch.zeros(E, H, P // 32, dtype=torch.uint8)
    readers = {}
    for e in range(E):
        for proj in ("w1", "w3", "w2"):
            for kind, dst in (("weight", w2 if proj == "w2" else w1), ("scale", s2 if proj == "w2" else s1)):
                key = f"layers.{layer}.ffn.experts.{e}.{proj}.{kind}"
                f = INDEX[key]
                if f not in readers:
                    readers[f] = safe_open(str(MODEL / f), framework="pt", device="cpu")
                src = readers[f].get_tensor(key).view(torch.uint8)
                loader._load_model_weight_or_group_weight_scale(
                    1 if proj == "w2" else 0, dst[e], proj, src, a.tp_rank, is_scale=kind == "scale")
    w1, s1, w2, s2 = (x.to(DEV) for x in (w1, s1, w2, s2))
    w1[:, I:P] = 0
    w1[:, P + I:] = 0
    s1[:, I:P] = 0
    s1[:, P + I:] = 0
    w2[:, :, I // 2:] = 0
    s2[:, :, I // 32:] = 0
    w1c = torch.cat((w1[:, :I], w1[:, P:P + I]), 1).contiguous()
    s1c = torch.cat((s1[:, :I], s1[:, P:P + I]), 1).contiguous()
    del w1, s1
    qc = make_mxfp4_moe_quant_config(Mxfp4MoeBackend.DEEPGEMM_MXFP4, sf(s1c, 2 * I, H), sf(s2, H, P),
                                     swiglu_limit=10.)
    ex = {}
    for n in ARMS:
        ex[n] = DeepGemmFP4Experts(cfg, qc)
        ex[n]._dsv41_target_m32 = KERN[n]
    return dict(layer=layer, w1=w1c, w2=w2, qc=qc, ex=ex)


def load_routes():
    wins, c4 = [], {}
    for fn in sorted(Path(a.routes).glob("*.jsonl")):
        for line in open(fn):
            d = json.loads(line)
            if d.get("event") != "routes" or d.get("validation_errors") not in ([], "[]", None):
                continue
            if d.get("has_prefill") in (True, "True"):
                continue
            ids = d["ids_by_layer"]
            ids = json.loads(ids) if isinstance(ids, str) else ids
            t = torch.tensor(ids, dtype=torch.int32)
            rows, nreq = int(d["physical_rows"]), int(d["num_requests"])
            assert t.shape == (40, rows, TOPK), t.shape
            if rows == 6 and nreq == 1:
                wins.append(t)
            elif nreq == 4 and rows in (12, 24):
                c4.setdefault(rows, []).append(t)
    return wins, c4


def route_sets(Ms):
    wins, c4 = load_routes()
    emit(event="routes", windows6=len(wins), c4={k: len(v) for k, v in c4.items()})
    g = torch.Generator().manual_seed(20261026)
    sets = []
    for m in Ms:
        for s in range(a.sets):
            perm = torch.randperm(len(wins), generator=g).tolist()
            if m == 1:
                ids = wins[perm[0]][:, :1]
            else:
                assert m % 6 == 0
                ids = torch.cat([wins[perm[i % len(perm)]] for i in range(m // 6)], 1)
            sets.append((m, f"u6x{max(1, m // 6)}-s{s}", ids))
        if m in c4:
            sets.append((m, "c4rec", c4[m][0]))
        if a.uniform:
            ids = torch.stack([torch.stack([torch.randperm(E, generator=g)[:TOPK] for _ in range(m)]) for _ in range(40)])
            sets.append((m, "uniform", ids.int()))
    return sets


def pooled(sizes):
    nbytes = [int(torch.Size(s).numel()) * 2 for s in sizes]
    offs, tot = [], 0
    for n in nbytes:
        offs.append(tot)
        tot += (n + 255) // 256 * 256
    pool = torch.empty(tot, dtype=torch.uint8, device=DEV)
    return [pool[o:o + n].view(torch.bfloat16).view(s) for o, n, s in zip(offs, nbytes, sizes)]


def capture(body):
    body()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        body()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        body()
    return g


def time_graphs(graphs):
    order = list(graphs)
    for _ in range(3):
        for n in order:
            graphs[n].replay()
    torch.cuda.synchronize()
    ts = {n: [] for n in order}
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for t in range(a.trials):
        for n in (order if t % 2 == 0 else order[::-1]):
            e0.record()
            graphs[n].replay()
            e1.record()
            e1.synchronize()
            ts[n].append(e0.elapsed_time(e1))
    return ts


t0 = time.time()
layers = [load_layer(int(x)) for x in a.layers.split(",")]
Kn = len(layers)
for L in layers:                                    # tails built eagerly up front, as at serving warm-up
    fc_serv.descs(L["w2"].view(torch.int8))
emit(event="setup", layers=[L["layer"] for L in layers], tp_rank=a.tp_rank, arms=ARMS, s=round(time.time() - t0, 1),
     tail_bytes_total=fc_serv.tail_bytes, mem_alloc_gb=round(torch.cuda.memory_allocated() / 2 ** 30, 2),
     natives={n: sha(mod.__file__)[:16] for n, mod in (("base", k_serv.native), ("d05", k_serv.fc2_native),
                                                       ("fc2c", fc_serv.native))},
     tree_manifest=sha(STAGE / "tree/manifest.json")[:16], host=os.uname().nodename)
Ms = [int(x) for x in a.Ms.split(",")]
gen = torch.Generator().manual_seed(11)
summary = []
for (m, setname, ids40c) in route_sets(Ms):
    ids40 = ids40c.to(DEV).contiguous()
    tw40 = torch.stack([(torch.softmax(torch.randn(m, TOPK, generator=gen), -1) * 1.5) for _ in range(40)]).to(DEV)
    x = (torch.randn(m, H, generator=gen) * 0.15).to(torch.bfloat16).to(DEV)
    aq, sfa = per_token_group_quant_fp8(x, 128, use_ue8m0=True)
    distinct = [int(torch.unique(ids40c[i]).numel()) for i in range(40)]
    sizes = layers[0]["ex"]["live"].workspace_shapes(m, 576, H, TOPK, E, E, None, SILU)
    res = {"m": m, "set": setname, "distinct_mean": round(statistics.mean(distinct), 1)}
    for mode in a.modes.split(","):
        graphs, outs = {}, {}
        if mode == "moe":
            for n in ARMS:
                ws = pooled(sizes)

                def body(n=n, ws=ws):
                    for i in range(40):
                        L = layers[i % Kn]
                        x13, y2, out = ws
                        L["ex"][n].apply(output=out, hidden_states=aq, w1=L["w1"], w2=L["w2"], topk_weights=tw40[i],
                                         topk_ids=ids40[i], activation=SILU, global_num_experts=E, expert_map=None,
                                         a1q_scale=sfa, a2_scale=None, workspace13=x13, workspace2=y2,
                                         expert_tokens_meta=None, apply_router_weight_on_input=False)
                graphs[n] = capture(body)
                outs[n] = ws[2]
        else:
            # FC2 inputs per layer slot: this set's real scatter32 layout + LIVE_ACT activation (as served)
            fin = []
            for i in range(40):
                L = layers[i % Kn]
                ex = L["ex"]["live"]
                if ex._workspace_m_alignment(m, 576, H, TOPK, E) != 32:
                    fin = None
                    break
                perm = utils.deepgemm_moe_permute(aq=aq, aq_scale=sfa, topk_ids=ids40[i], local_num_experts=E,
                                                  expert_map=None, expert_tokens_meta=None, alignment_m=32,
                                                  return_expert_ends=True)
                layout = perm[2]
                msum = layout.numel()
                mm1 = (torch.randn(msum, 576, device=DEV) * 0.5).to(torch.bfloat16)
                q = torch.empty(msum, 384, dtype=torch.float8_e4m3fn, device=DEV)
                live_act = ex._dsv41_live_act(msum, SILU)
                a2q, a2s = ex._act_mul_quant(input=mm1, output=q, activation=SILU,
                                             expert_ends=perm[5] if live_act else None,
                                             expert_alignment=32 if live_act else 0)
                fin.append((a2q, a2s, layout, msum))
            if fin is None:
                continue
            for n in ARMS:
                outs_n = [torch.empty(f[3], H, dtype=torch.bfloat16, device=DEV) for f in fin[:Kn]]

                def body(n=n, outs_n=outs_n):
                    kern = KERN[n]
                    with kern.alignment_scope():
                        for i in range(40):
                            a2q, a2s, layout, _ = fin[i]
                            kern.gemm((a2q, a2s), (layers[i % Kn]["w2"].view(torch.int8), layers[i % Kn]["ex"]["live"].w2_scale),
                                      outs_n[i % Kn], layout, recipe_a=(1, 128), recipe_b=(1, 32))
                graphs[n] = capture(body)
                outs[n] = outs_n
        ts = time_graphs(graphs)
        med = {n: statistics.median(v) for n, v in ts.items()}
        r = {}
        for n in ARMS:
            us = med[n] * 1000 / 40
            r[n] = dict(layer_us=round(us, 2), step40_ms=round(med[n], 4), min_ms=round(min(ts[n]), 4),
                        p90_ms=round(sorted(ts[n])[int(0.9 * (len(ts[n]) - 1))], 4),
                        speedup=round(med["live"] / med[n], 4), delta_ms_step=round(med["live"] - med[n], 4))
            if mode == "fc2":
                byt = FC2_BYTES["live" if (n == "live" or (n == "fc2c" and m == 1)) else "fc2c"]
                r[n]["weight_gbps"] = round(sum(distinct) * byt / (med[n] / 1000) / 1e9, 1)
        # bitwise: every arm's final MoE output equals live (moe)
        if mode == "moe":
            for n in ARMS[1:]:
                r[n]["bitwise_vs_live"] = bool(torch.equal(outs[n].view(torch.uint8), outs["live"].view(torch.uint8)))
        res[mode] = r
        if mode == "moe" and str(m) in a.profile.split(",") and setname.endswith("s0"):
            try:
                from torch.profiler import profile, ProfilerActivity
                prof_out = {}
                for n in ARMS:
                    with profile(activities=[ProfilerActivity.CUDA]) as prof:
                        graphs[n].replay()
                        torch.cuda.synchronize()
                    agg = {}
                    for ev in prof.events():
                        if ev.device_type.name == "CUDA" and ev.device_time > 0:
                            k = ev.name[:100]
                            c, tt = agg.get(k, (0, 0.0))
                            agg[k] = (c + 1, tt + ev.device_time)
                    prof_out[n] = [dict(k=k, calls=c, us_per_layer=round(tt / 40, 2))
                                   for k, (c, tt) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:6]]
                res["profile"] = prof_out
            except Exception as ex_:  # profiling is optional
                res["profile_error"] = repr(ex_)[:300]
        del graphs
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    emit(event="timing", **res)
    summary.append(res)

# projection rows: 1 user = C1 full-width (M=6) and M=1; 4 users = c4rec (M=12/24) else u6x2/u6x4
proj = []
for row in summary:
    if "moe" not in row:
        continue
    d = {n: row["moe"][n]["delta_ms_step"] for n in ARMS[1:]}
    f2 = {n: row.get("fc2", {}).get(n, {}).get("speedup") for n in ARMS[1:]}
    users = 4 if row["m"] >= 12 else 1
    scale = 0.90 if users == 4 else 0.96
    step = a.step_4u_ms if users == 4 else a.step_1u_code_ms
    pct = {n: round(100 * (step / (step - v * scale) - 1), 2) for n, v in d.items()}
    proj.append(dict(m=row["m"], set=row["set"], live_moe_us=row["moe"]["live"]["layer_us"], moe_delta_ms_step=d,
                     serving_delta_ms_step={n: round(v * scale, 3) for n, v in d.items()},
                     tok_s_pct_at_step={n: pct[n] for n in pct}, step_ms=step, fc2_speedup=f2))
emit(event="projection", rows=proj, note="delta_ms_step = 40 target MoE layers (bench); serving = x0.96 (1u) / x0.90 "
     "(4u); tok_s_pct at the given step (1u: C1 coding; M>=12: 4-user step estimate); positive = faster than live")
emit(event="done", s=round(time.time() - t0, 1))
