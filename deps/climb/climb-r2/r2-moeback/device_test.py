#!/usr/bin/env python3
"""r2-moeback device test: live kernels vs kit kernels side by side, bitwise, plus timing.

Run by the lead on a STOPPED node (no vLLM on the GPU), from the kit folder copied to the node:
    CUDA_VISIBLE_DEVICES=0 ~/venvs/vllm032/bin/python device_test.py --kit . [--part g25|g18|all] [--reps 200]
    (--quick: fewer cases/reps, ~1 min; --plan: build every case on the CPU only, no GPU, no kernels)
Expected runtime: g25 ~2-4 min (Triton JIT of two kernels + 9 kinds x 23 M x eager/graph + timing),
g18 ~3-6 min (283 MB packed weights, SiLU JIT, 9 M x 3 checks + 200 graph replays per M + timing).
Optional race tools (slow, run with --quick):
    compute-sanitizer --tool racecheck  ~/venvs/vllm032/bin/python device_test.py --kit . --part g18 --quick
    compute-sanitizer --tool synccheck  ~/venvs/vllm032/bin/python device_test.py --kit . --part g18 --quick
Exit code 0 only when every case is bit-identical (raw bytes; for G25 also NaN positions identical).

G25 (DSV41_FC2_GATHER_ADD): live ~/dsv41-fc2/fc2_status_gated_gather_trial.py gated_gather() followed by
  torch `shared + routed` (the runner's add)  vs  kit dsv41_moe_fold.launch_gather_add() (one kernel).
  M (tokens) 1..24, 32, 40, 48, 64, 80, 96 (all target-M32 verify sizes) and 1500, 2048 (grid-stride path);
  kinds: normal, wide, subnormal, zeros, maxmag, ties, cancel, neg_ids, status_failed; eager and CUDA-graph
  replay with new inputs + poisoned outputs between replays.
G18 (DSV41_FC2_PDL): live athena_tp8_fc2_packed_graph_sm121.so  vs  kit athena_tp8_fc2_packed_graph_pdl_sm121.so
  (both export launch(); loaded side by side under the module name the overlay uses).
  (a) eager, identical inputs, poisoned outputs: every byte of out and status equal;
  (b) the real primary: vLLM's silu_mul_quant_fp8_packed_triton with the early PDL trigger
      (fp8_utils._DSV41_PDL_TRIGGER = 1, both the dense and the expert_ends/live-rows variant), then the PDL
      FC2 on the same stream, with FC1-output (mm1) and FC2-output (mm2) ALIASING one workspace exactly as
      deep_gemm_moe.py does (workspace2), captured in a CUDA graph and replayed >= reps times with new mm1 and
      new layouts; each replay is checked against an eager reference (SiLU then live FC2 on separate buffers).
  M (tokens) 1, 2, 3, 4, 5, 6, 8, 16, 48, 96; recorded-like routes (distinct experts), hostile routes
  (all tokens on one expert, experts 0/383, -1 ids are not produced by the router so not used).
Timing: median of >= reps, warm (20 back-to-back iterations inside one graph replay, per-iteration) and
cold-L2 (64 MiB scratch write before each eager iteration), old vs new.
"""
import argparse
import importlib.util
import math
import os
import statistics
import sys
import time

import torch

AR = argparse.ArgumentParser()
AR.add_argument("--kit", default=os.path.dirname(os.path.abspath(__file__)))
AR.add_argument("--part", default="all", choices=["g25", "g18", "all"])
AR.add_argument("--reps", type=int, default=200)
AR.add_argument("--quick", action="store_true")
AR.add_argument("--plan", action="store_true", help="CPU only: build all inputs, launch nothing")
AR.add_argument("--fc2-dir", default=os.path.expanduser("~/dsv41-fc2"))
ARGS = AR.parse_args()
DEV = "cpu" if ARGS.plan else "cuda"
FAIL = []
RESULTS = []


def report(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print(f"{'OK  ' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        FAIL.append(name)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bits(t):
    return t.contiguous().view(torch.int16) if t.dtype == torch.bfloat16 else t.contiguous().view(torch.uint8)


def first_mismatch(a, b):
    d = (bits(a) != bits(b))
    if not bool(d.any()):
        return None
    idx = torch.nonzero(d)[0].tolist()
    return idx, int(bits(a)[tuple(idx)]), int(bits(b)[tuple(idx)])


def sync():
    if DEV == "cuda":
        torch.cuda.synchronize()


def median_us(fn, reps, cold=False, scratch=None):
    if DEV != "cuda":
        return float("nan")
    ts = []
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(10):
        fn()
    for _ in range(reps):
        if cold:
            scratch.add_(1)
        s.record()
        fn()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)
    return statistics.median(ts)


# ============================================================================ G25
def g25_case(g, m, kind, d=5120, topk=6):
    m_sum = ((6 * m + min(6 * m, 384) * 31 + 31) // 32) * 32
    ids = torch.randint(0, 384, (m, topk), generator=g, dtype=torch.int32)
    inv = torch.randperm(m_sum, generator=g)[: m * topk].view(m, topk).to(torch.int32)
    w = torch.distributions.Dirichlet(torch.ones(topk)).sample((m,)).float() * 2.5
    x = torch.randn(m_sum, d, generator=g) * 0.05
    sh = torch.randn(m, d, generator=g) * 0.3
    failed = False
    if kind == "wide":
        x = x * torch.exp2(torch.randint(-60, 60, (m_sum, d), generator=g).float())
        sh = sh * torch.exp2(torch.randint(-60, 60, (m, d), generator=g).float())
    elif kind == "subnormal":
        x = torch.randn(m_sum, d, generator=g) * 2.0 ** -128
        sh = torch.randn(m, d, generator=g) * 2.0 ** -130
        w = torch.rand(m, topk, generator=g) * 2
    elif kind == "zeros":
        x[:, ::3] = 0.0
        x[:, 1::3] = -0.0
        sh[:, ::2] = -0.0
        sh[:, 1::4] = 0.0
    elif kind == "maxmag":
        big = torch.tensor(3.3895313892515355e38)
        x[:, ::5] = big * torch.sign(x[:, ::5] + 1e-30)
        w = w / 8
        sh[:, ::7] = big * 0.9 * torch.sign(sh[:, ::7] + 1e-30)
    elif kind == "ties":
        x = 1.0 + torch.randint(0, 128, (m_sum, d), generator=g).float() / 128.0
        ids[:, 1:] = -1
        w[:, 0] = 1.0
        sh = (torch.randint(0, 64, (m, d), generator=g) * 2 + 1).float() * 2.0 ** -8
    elif kind == "cancel":
        r = (x.to(torch.bfloat16).float()[inv[:, 0].long()] * w[:, :1]).to(torch.bfloat16).float()
        sh = -r
        ids[:, 1:] = -1
    elif kind == "neg_ids":
        ids[torch.rand(m, topk, generator=g) < 0.3] = -1
    elif kind == "status_failed":
        failed = True
    status = torch.tensor([1 if failed else 0], dtype=torch.int32)
    to = lambda t: t.to(DEV)
    return dict(mm2=to(x.to(torch.bfloat16)), ids=to(ids), w=to(w.float()), inv=to(inv),
                shared=to(sh.to(torch.bfloat16)), status=to(status), m=m)


def run_g25():
    print("== G25: fold of the routed+shared add into the Athena gather", flush=True)
    os.environ.pop("DSV41_FC2_GATHER_ADD", None)  # the live module's own flag stays off: plain gather path
    live_path = os.path.join(ARGS.fc2_dir, "fc2_status_gated_gather_trial.py")
    fold_path = os.path.join(ARGS.kit, "files/vllm/model_executor/layers/fused_moe/runner/dsv41_moe_fold.py")
    if ARGS.plan:
        live = fold = None
    else:
        live = load("live_athena_gather", live_path)
        fold = load("kit_dsv41_moe_fold", fold_path)
    g = torch.Generator().manual_seed(20261108)
    ms = list(range(1, 25)) + [32, 40, 48, 64, 80, 96, 1500, 2048]
    kinds = ["normal", "wide", "subnormal", "zeros", "maxmag", "ties", "cancel", "neg_ids", "status_failed"]
    if ARGS.quick:
        ms = [1, 2, 6, 24, 96, 2048]

    def old(c, out):
        live.gated_gather(c["mm2"], c["ids"], c["w"], c["inv"], None, out, c["status"], experimental=True)
        return c["shared"] + out                     # the runner's add (moe_runner.py:789)

    def new(c, out):
        n = c["m"]
        block_d = math.gcd(c["mm2"].shape[1], 1024)
        grid = (triton_cdiv(c["mm2"].shape[1], block_d), min(n, 1024))
        return fold.launch_gather_add(grid, n, c["mm2"], c["ids"], c["w"], c["inv"], out, c["status"],
                                      c["shared"], block_d)

    ncase = 0
    for kind in kinds:
        bad = []
        for m in ms:
            c = g25_case(g, m, kind)
            ncase += 1
            if ARGS.plan:
                continue
            o_old = torch.full((m, 5120), float("nan"), dtype=torch.bfloat16, device=DEV)
            o_new = torch.full((m, 5120), -7.0, dtype=torch.bfloat16, device=DEV)
            ref = old(c, o_old)
            got = new(c, o_new)
            sync()
            nan_r, nan_g = torch.isnan(ref), torch.isnan(got)
            if not torch.equal(nan_r, nan_g):
                bad.append((m, "NaN positions differ"))
                continue
            r = torch.where(nan_r, torch.zeros_like(ref), ref)
            q = torch.where(nan_g, torch.zeros_like(got), got)
            fm = first_mismatch(r, q)
            if fm:
                bad.append((m, fm))
        report(f"g25 eager kind={kind} M={ms[0]}..{ms[-1]} ({len(ms)} sizes)", not bad, str(bad[:3]))
    if ARGS.plan:
        report(f"g25 plan: {ncase} cases built on CPU", True)
        return
    # CUDA graph replay, poisoned outputs, new inputs each replay
    for m in ([1, 6, 48, 96] if not ARGS.quick else [6]):
        c = g25_case(g, m, "normal")
        o_old = torch.empty((m, 5120), dtype=torch.bfloat16, device=DEV)
        o_new = torch.empty((m, 5120), dtype=torch.bfloat16, device=DEV)
        old(c, o_old), new(c, o_new)
        sync()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        g_old, g_new = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        with torch.cuda.stream(s):
            with torch.cuda.graph(g_old):
                ref_static = old(c, o_old)
            with torch.cuda.graph(g_new):
                got_static = new(c, o_new)
        torch.cuda.current_stream().wait_stream(s)
        bad = 0
        for rep in range(ARGS.reps if not ARGS.quick else 20):
            c2 = g25_case(g, m, ["normal", "wide", "zeros", "ties", "subnormal"][rep % 5])
            for k in ("mm2", "ids", "w", "inv", "shared", "status"):
                c[k].copy_(c2[k])
            o_old.fill_(float("nan")), o_new.fill_(-7.0), ref_static.fill_(5.0)
            g_old.replay()
            g_new.replay()
            sync()
            if not torch.equal(bits(ref_static), bits(got_static)):
                bad += 1
        report(f"g25 graph replay M={m} ({ARGS.reps if not ARGS.quick else 20} replays, new inputs, poisoned)", bad == 0, f"bad={bad}")
    # timing
    scratch = torch.empty(64 << 20, dtype=torch.uint8, device=DEV)
    for m in [1, 6, 24, 96]:
        c = g25_case(g, m, "normal")
        o1 = torch.empty((m, 5120), dtype=torch.bfloat16, device=DEV)
        o2 = torch.empty_like(o1)
        go, gn = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        old(c, o1), new(c, o2)
        sync()
        with torch.cuda.graph(go):
            for _ in range(20):
                old(c, o1)
        with torch.cuda.graph(gn):
            for _ in range(20):
                new(c, o2)
        w_old = median_us(go.replay, ARGS.reps) / 20
        w_new = median_us(gn.replay, ARGS.reps) / 20
        c_old = median_us(lambda: old(c, o1), ARGS.reps, cold=True, scratch=scratch)
        c_new = median_us(lambda: new(c, o2), ARGS.reps, cold=True, scratch=scratch)
        print(f"TIME g25 M={m:3d}: warm graph gather+add {w_old:7.2f} us  gather_add {w_new:7.2f} us  "
              f"(-{w_old - w_new:5.2f});  cold-L2 eager {c_old:7.2f} -> {c_new:7.2f} us", flush=True)


def triton_cdiv(a, b):
    return -(-a // b)


# ============================================================================ G18
def g18_inputs(g, m, route):
    """Athena FC2 operands for M tokens (topk 6) in the production layout (align 32, K=384 with 288 live)."""
    topk = 6
    if route == "one_expert":
        ids = torch.full((m, topk), 7, dtype=torch.int64)
        ids[:, 1:] = torch.arange(1, topk) + 200
    elif route == "edges":
        ids = torch.tensor([[0, 383, 1, 382, 191, 192]] * m)
    else:
        ids = torch.stack([torch.randperm(384, generator=g)[:topk] for _ in range(m)])
    m_sum = ((6 * m + min(6 * m, 384) * 31 + 31) // 32) * 32
    counts = torch.bincount(ids.flatten(), minlength=384)
    layout = torch.full((m_sum,), -1, dtype=torch.int32)
    ends = torch.zeros(384, dtype=torch.int32)
    end = 0
    for e in range(384):
        start = ((end + 31) // 32) * 32
        cnt = int(counts[e])
        layout[start:start + cnt] = e
        end = start + cnt
        ends[e] = end
    assert end <= m_sum
    return m_sum, layout, ends


def rand_e4m3(g, shape):
    b = torch.randint(0, 256, shape, generator=g, dtype=torch.uint8)
    b = torch.where((b & 0x7F) == 0x7F, b - 1, b)  # no NaN encodings
    return b


def run_g18():
    print("== G18: Athena FC2 PDL early-B vs live FC2", flush=True)
    live_so = os.path.join(ARGS.fc2_dir, "athena_tp8_fc2_packed_graph_sm121.so")
    pdl_so = os.path.join(ARGS.kit, "build/athena_tp8_fc2_packed_graph_pdl_sm121.so")
    if not ARGS.plan:
        if not os.path.exists(pdl_so):
            report("g18 pdl .so present in kit build/", False, pdl_so)
            return
        name = "athena_tp8_fc2_packed_graph_sm121"   # PyInit name of both binaries (the overlay's load name)
        try:
            live = load(name, live_so)
            pdl = load(name, pdl_so)
        except Exception as exc:  # side-by-side load refused: run the two halves in separate processes
            report("g18 side-by-side load of live + pdl .so", False, repr(exc))
            return
        assert live is not pdl and live.launch is not pdl.launch
        os.environ.setdefault("DSV41_MOE_PDL_TRIGGER", "0")
        from vllm.model_executor.layers.quantization.utils import fp8_utils
    g = torch.Generator().manual_seed(18)
    gw = torch.Generator().manual_seed(1818)
    weight = torch.randint(0, 256, (384, 80, 9, 1024), generator=gw, dtype=torch.uint8).to(DEV)
    sbytes = torch.randint(118, 137, (384, 3, 5120, 4), generator=gw, dtype=torch.int32)
    swords = sbytes[..., 0] | (sbytes[..., 1] << 8) | (sbytes[..., 2] << 16) | (sbytes[..., 3] << 24)
    w_scale = swords.to(DEV).permute(0, 2, 1)  # [384,5120,3], strides (15360,1,5120)
    assert w_scale.stride() == (15360, 1, 5120)
    status = torch.zeros(1, dtype=torch.int32, device=DEV)
    ms = [1, 2, 3, 4, 5, 6, 8, 16, 48, 96] if not ARGS.quick else [1, 6, 96]
    for m in ms:
        for route in (["distinct", "one_expert", "edges"] if not ARGS.quick else ["distinct"]):
            m_sum, layout, ends = g18_inputs(g, m, route)
            mn = ((m_sum + 3) // 4) * 4
            aq = rand_e4m3(g, (m_sum, 384))
            aq[:, 288:] = 0
            aq = aq.to(DEV).view(torch.float8_e4m3fn)
            sc = torch.randint(120, 135, (m_sum, 3), generator=g, dtype=torch.int32)
            sw = (sc[:, 0] | (sc[:, 1] << 8) | (sc[:, 2] << 16) | (127 << 24)).to(torch.int32)
            a_scale = torch.empty_strided((m_sum, 1), (1, mn), dtype=torch.int32, device=DEV)
            a_scale.copy_(sw.view(m_sum, 1).to(DEV))
            lay = layout.to(DEV)
            if ARGS.plan:
                continue
            o1 = torch.full((m_sum, 5120), float("nan"), dtype=torch.bfloat16, device=DEV)
            o2 = o1.clone()
            s1, s2 = status.clone(), status.clone()
            live.launch(aq, a_scale, lay, weight, w_scale, o1, s1)
            pdl.launch(aq, a_scale, lay, weight, w_scale, o2, s2)
            sync()
            fm = first_mismatch(o1, o2)
            report(f"g18 eager M={m} route={route} M_sum={m_sum}", fm is None and torch.equal(s1, s2),
                   f"first={fm} status={int(s1)},{int(s2)}")
    if ARGS.plan:
        report("g18 plan: inputs built on CPU", True)
        return
    # (b) real SiLU primary with the early trigger, aliased workspace, CUDA graph replays
    clamp = 10.0
    for m in ([1, 6, 16, 96] if not ARGS.quick else [6]):
        for live_rows in (False, True):
            m_sum, layout, ends = g18_inputs(g, m, "distinct")
            lay = layout.to(DEV)
            ends_d = ends.to(DEV)
            ws2 = torch.empty(m_sum * 5120 * 2, dtype=torch.uint8, device=DEV)      # workspace2
            mm1 = ws2[: m_sum * 576 * 2].view(torch.bfloat16).view(m_sum, 576)     # FC1 output (SiLU input)
            mm2 = ws2.view(torch.bfloat16).view(m_sum, 5120)                        # FC2 output, same storage
            ws13 = torch.empty(m_sum * 384, dtype=torch.uint8, device=DEV)
            q_out = ws13.view(torch.float8_e4m3fn).view(m_sum, 384)
            src = torch.empty(m_sum, 576, dtype=torch.bfloat16, device=DEV)         # staged next mm1

            def silu():
                kw = dict(expert_ends=ends_d, expert_alignment=32) if live_rows else {}
                return fp8_utils.silu_mul_quant_fp8_packed_triton(
                    input=mm1, output_q=q_out, group_size=128, clamp_limit=clamp, padded_output_size=384, **kw)

            def step(fc2):
                mm1.copy_(src)                                   # "FC1" writes workspace2 (a kernel)
                a2q, a2s = silu()                                # primary (trigger at start when enabled)
                fc2.launch(a2q, a2s, lay, weight, w_scale, mm2, status)   # writes workspace2 (aliases mm1)
                return a2q, a2s

            def reference(mm1_vals):
                r_mm1 = mm1_vals.clone()
                r_q = torch.empty(m_sum, 384, dtype=torch.float8_e4m3fn, device=DEV)
                fp8_utils._DSV41_PDL_TRIGGER = 0
                kw = dict(expert_ends=ends_d, expert_alignment=32) if live_rows else {}
                ra, rs = fp8_utils.silu_mul_quant_fp8_packed_triton(
                    input=r_mm1, output_q=r_q, group_size=128, clamp_limit=clamp, padded_output_size=384, **kw)
                r_out = torch.full((m_sum, 5120), float("nan"), dtype=torch.bfloat16, device=DEV)
                live.launch(ra, rs, lay, weight, w_scale, r_out, torch.zeros_like(status))
                return ra, r_out

            src.copy_((torch.randn(m_sum, 576, generator=g) * 3).to(torch.bfloat16))
            fp8_utils._DSV41_PDL_TRIGGER = 1
            step(pdl)
            step(pdl)
            sync()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(s), torch.cuda.graph(graph):
                a2q_static, _ = step(pdl)
            torch.cuda.current_stream().wait_stream(s)
            bad_q = bad_o = 0
            n_rep = ARGS.reps if not ARGS.quick else 20
            live_mask = (layout >= 0).to(DEV)
            for rep in range(n_rep):
                vals = (torch.randn(m_sum, 576, generator=g) * (0.01 + 4 * (rep % 7))).to(torch.bfloat16).to(DEV)
                src.copy_(vals)
                mm2.fill_(float("nan"))
                graph.replay()
                sync()
                ra, r_out = reference(vals)
                fp8_utils._DSV41_PDL_TRIGGER = 1
                if not torch.equal(bits(a2q_static[live_mask]), bits(ra[live_mask])):
                    bad_q += 1
                if not torch.equal(bits(mm2[live_mask]), bits(r_out[live_mask])):
                    bad_o += 1
            report(f"g18 graph SiLU(trigger)->PDL FC2, aliased workspace2, M={m} live_rows={live_rows} "
                   f"({n_rep} replays)", bad_q == 0 and bad_o == 0, f"bad a2q={bad_q} bad out={bad_o}")
    # timing: SiLU -> FC2 pair, live (no trigger) vs PDL (trigger)
    scratch = torch.empty(64 << 20, dtype=torch.uint8, device=DEV)
    for m in [1, 6, 16, 96]:
        m_sum, layout, ends = g18_inputs(g, m, "distinct")
        lay, ends_d = layout.to(DEV), ends.to(DEV)
        mm1 = (torch.randn(m_sum, 576, generator=g)).to(torch.bfloat16).to(DEV)
        q = torch.empty(m_sum, 384, dtype=torch.float8_e4m3fn, device=DEV)
        out = torch.empty(m_sum, 5120, dtype=torch.bfloat16, device=DEV)

        def pair(fc2, trig):
            fp8_utils._DSV41_PDL_TRIGGER = trig
            a, s_ = fp8_utils.silu_mul_quant_fp8_packed_triton(input=mm1, output_q=q, group_size=128, clamp_limit=clamp,
                                                               padded_output_size=384, expert_ends=ends_d,
                                                               expert_alignment=32)
            fc2.launch(a, s_, lay, weight, w_scale, out, status)

        res = {}
        for label, fc2, trig in (("live", live, 0), ("pdl", pdl, 1)):
            pair(fc2, trig), pair(fc2, trig)
            sync()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                for _ in range(20):
                    pair(fc2, trig)
            res[label] = (median_us(gr.replay, ARGS.reps) / 20,
                          median_us(lambda: pair(fc2, trig), ARGS.reps, cold=True, scratch=scratch))
        print(f"TIME g18 M={m:3d} M_sum={m_sum:5d}: warm graph SiLU+FC2 live {res['live'][0]:7.2f} us  pdl "
              f"{res['pdl'][0]:7.2f} us (-{res['live'][0] - res['pdl'][0]:5.2f});  cold-L2 eager "
              f"{res['live'][1]:7.2f} -> {res['pdl'][1]:7.2f} us", flush=True)
    fp8_utils._DSV41_PDL_TRIGGER = 0


def main():
    t0 = time.time()
    torch.manual_seed(0)
    if DEV == "cuda":
        print(f"device {torch.cuda.get_device_name()} torch {torch.__version__}", flush=True)
    if ARGS.part in ("g25", "all"):
        run_g25()
    if ARGS.part in ("g18", "all"):
        run_g18()
    print(f"elapsed {time.time() - t0:.1f}s; {len(RESULTS) - len(FAIL)}/{len(RESULTS)} OK", flush=True)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
