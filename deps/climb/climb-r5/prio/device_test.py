#!/usr/bin/env python3
"""r5-prio device test: graph stream-priority gate (T1) + bitwise/race checks of the three levers (T2-T4).

RUN ONLY ON A STOPPED NODE (spark-eight), never while serving, never while ~/dsv41-opt-20261021/CLUSTER_LOCK is
held by someone else's window. The mandatory environment (the test refuses to run without it):

    ssh spark-eight 'source ~/venvs/vllm032/bin/activate; export PATH=/usr/local/cuda/bin:$PATH \
        CUDA_HOME=/usr/local/cuda; cd ~/climb-scratch/r5-prio && python3 device_test.py --json t.json'

  --source kit        (default) load the patched copies in ./patched/vllm and ./files/vllm (kit not installed)
  --source installed  use the files installed by install.py (status must say INSTALLED)
  --quick             fewer replays (smoke run, not a gate)
  --cpu-selftest      only the verdict logic, no GPU (used by tests/test_prio.py)

Expected runtime ~2-4 min (Triton JIT for the spin kernels, skinny_mx .so load, FlashInfer MXFP8 GEMMs).

Exit codes: 0 = every gated check passed AND T1 says the graph honours priority (all arms usable);
            4 = T2-T4 all pass but T1 GATE FAIL (only DSV41_PRIO_CMP=serial and DSV41_PRIO_ENGRAM_HOIST are worth
                a window; prio and shared arms are inert in graphs);
            3 = T2-T4 pass, T1 INCONCLUSIVE (its controls did not discriminate; see the printed P values);
            1 = environment wrong, a bitwise mismatch / race, or a must-fail mutant NOT caught (harness too weak).

T1  priority microtest. A spin kernel records per-CTA %globaltimer start/end and %smid. A backlog "flood" (many
    waves of 512-thread CTAs, 3 resident per SM) runs on one stream and a 48-CTA, 256-thread "probe" on another.
    P = fraction of flood CTAs that start after the probe's median CTA start. Priority honoured -> the probe is
    scheduled at the first flood-CTA retirement (P ~ 1 - 1/waves); not honoured -> after the backlog (P ~ 0).
    Arms: eager hi/same/swap, graph (torch.cuda.graph, launched by g.replay()) hi/same/swap with the flood
    captured first, plus diagnostics (graph with the probe captured first, node priority attributes read through
    cuda.bindings, the same graph re-instantiated with cudaGraphInstantiateFlagUseNodePriority, and a one-wave
    persistent flood showing that priority never preempts a running CTA). same/swap are the must-fail mutants.
T2  CMPPRIO: the real r3 QkvAShard (skinny_mx / r2 v2 GEMM, FlashInfer MXFP8 activation quant) next to a BF16
    torch.mm compressor flood and the indexer GEMM, through dsv41_prio.cmp_parallel_input_projections (prio and
    serial) vs the live execute_in_parallel; eager and CUDA-graph replays with new inputs every replay; M = 1..8;
    bitwise (raw bytes). Mutants: prio without the join (+ a spin before the GEMM), serial/prio with the compressor
    done-event wait dropped (+ a spin in the compressor) -- each must mismatch at every M.
T3  SHARED: vLLM's SharedExperts (patched) with the stream from dsv41_prio.shared_stream, an MXFP8 shared MLP
    (FlashInfer cutlass GEMMs) against a routed flood that also runs MXFP8 GEMMs on main (covers the process-wide
    FlashInfer workspace); vs the live aux_stream; mutant without wait().
T4  ENGRAM_HOIST: dsv41_prio.hoist/engram_take with an MXFP8 wkv (K 6144 -> N 3200 local, FlashInfer cutlass),
    main-stream "layers" (MXFP8 + BF16 GEMMs) between hoist and take; vs the live statement; mutant without the
    join (+ a spin on the side stream).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import statistics as st
import subprocess
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS: dict = {"checks": [], "t1": {}, "timing": {}}
FAILS: list[str] = []


def log(msg):
    print(msg, flush=True)


def check(name, ok, detail=""):
    RESULTS["checks"].append({"name": name, "ok": bool(ok), "detail": detail})
    log(f"  [{'PASS' if ok else 'FAIL'}] {name}{(': ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)
    return ok


# ------------------------------------------------------------------------------------- T1 verdict logic
HONOURED, NOT, AMBIG = "honoured", "not-honoured", "ambiguous"


def classify(ps, hi=0.5, lo=0.2):
    if not ps:
        return AMBIG
    m = st.median(ps)
    return HONOURED if m >= hi else NOT if m <= lo else AMBIG


def gate_verdict(arms):
    """arms: name -> classification. Returns (verdict, reason)."""
    e_ok = arms.get("eager_hi") == HONOURED
    e_mut = arms.get("eager_same") == NOT and arms.get("eager_swap") == NOT
    g_mut = arms.get("graph_same") == NOT and arms.get("graph_swap") == NOT
    if not e_mut:
        return "INCONCLUSIVE", "eager controls (same/swap priority) did not come out not-honoured"
    if not e_ok:
        return "FAIL", "even eager launches ignore stream priority on this device/driver"
    if not g_mut:
        return "INCONCLUSIVE", "graph controls did not come out not-honoured (capture order dominates)"
    if arms.get("graph_hi") == HONOURED:
        return "PASS", "a torch-captured graph launched with replay() honours the captured stream priorities"
    if arms.get("graph_hi") == NOT:
        return "FAIL", "the captured graph runs every node at the launch stream's priority"
    return "INCONCLUSIVE", "graph_hi ambiguous"


def cpu_selftest():
    assert classify([0.8, 0.83, 0.79]) == HONOURED
    assert classify([0.0, 0.01, 0.02]) == NOT
    assert classify([0.3, 0.4]) == AMBIG and classify([]) == AMBIG
    good = dict(eager_hi=HONOURED, eager_same=NOT, eager_swap=NOT, graph_hi=HONOURED, graph_same=NOT, graph_swap=NOT)
    assert gate_verdict(good)[0] == "PASS"
    assert gate_verdict(dict(good, graph_hi=NOT))[0] == "FAIL"
    assert gate_verdict(dict(good, graph_same=HONOURED))[0] == "INCONCLUSIVE"
    assert gate_verdict(dict(good, eager_hi=NOT))[0] == "FAIL"
    assert gate_verdict(dict(good, eager_swap=HONOURED))[0] == "INCONCLUSIVE"
    assert gate_verdict(dict(good, graph_hi=AMBIG))[0] == "INCONCLUSIVE"
    assert exit_code("PASS", []) == 0 and exit_code("FAIL", []) == 4 and exit_code("INCONCLUSIVE", []) == 3
    assert exit_code("PASS", ["x"]) == 1
    # B1 serving-op verdict: v2 on and built at every served M passes; every degraded case must fail
    v2 = types.SimpleNamespace(ok_a8=set(range(1, 9)))
    allok = lambda r: all(ok for _, ok, _ in r)  # noqa: E731
    assert allok(skinny_serving_checks(True, "skinny", True, v2, 8))
    assert not allok(skinny_serving_checks(True, "skinny", True, None, 8))  # build returned None / raised
    assert not allok(skinny_serving_checks(True, "skinny", True, None, 8, "RuntimeError('x')"))
    assert not allok(skinny_serving_checks(True, "skinny", True, types.SimpleNamespace(ok_a8={1, 2, 3, 4}), 8))
    assert not allok(skinny_serving_checks(True, "skinny", True, types.SimpleNamespace(ok_a8=set()), 8))
    assert not allok(skinny_serving_checks(False, "skinny", True, None, 8))  # v1 is not the serving op
    assert not allok(skinny_serving_checks(True, "cutlass", True, v2, 8))
    assert not allok(skinny_serving_checks(True, "skinny", False, v2, 8))
    print("cpu-selftest: OK")
    return 0


def skinny_serving_checks(r2_v2, qkv_gemm, ext_ok, r2, max_m, err=""):
    """B1: the qkv_a GEMM under test must be the serving one. With DSV41_SKINNY_V2 on (production), a v2 build that
    raised or returned None (the serving code then silently keeps v1) or that does not cover every served M is a FAIL,
    never a note. Returns [(name, ok, detail)]."""
    out = [("serving qkv_a GEMM is skinny (r3 QKV_GEMM)", qkv_gemm == "skinny" and bool(ext_ok),
            f"QKV_GEMM={qkv_gemm} ext={'ok' if ext_ok else 'missing'}"),
           ("DSV41_SKINNY_V2 on, as in launcher 166d7aac", bool(r2_v2), f"R2_V2={r2_v2}")]
    if r2_v2:
        ms = sorted(getattr(r2, "ok_a8", None) or ()) if r2 is not None else []
        want = list(range(1, int(max_m) + 1))
        out.append(("serving qkv_a GEMM is skinny v2 (r2 build OK, a8 at every served M)",
                    r2 is not None and set(want) <= set(ms),
                    f"r2={'v2' if r2 is not None else 'None'} a8 M={ms} served M={want}"
                    + (f" build error {err}" if err else "")))
    return out


def exit_code(verdict, fails):
    if fails:
        return 1
    return {"PASS": 0, "FAIL": 4}.get(verdict, 3)


# ------------------------------------------------------------------------------------------------ T0 env
def env_check(args):
    log("== T0 environment (mandatory: venv vllm032, CUDA_HOME=/usr/local/cuda, cuda/bin on PATH)")
    ok = True
    ok &= check("CUDA_HOME", os.environ.get("CUDA_HOME") == "/usr/local/cuda", os.environ.get("CUDA_HOME", "unset"))
    ok &= check("PATH has /usr/local/cuda/bin", "/usr/local/cuda/bin" in os.environ.get("PATH", "").split(":"))
    nvcc = shutil.which("nvcc") or ""
    ok &= check("nvcc from /usr/local/cuda", nvcc.startswith("/usr/local/cuda"), nvcc)
    ok &= check("venv vllm032", sys.prefix.rstrip("/").endswith("venvs/vllm032"), sys.prefix)
    if not ok:
        return False
    import torch

    ok &= check("cuda available", torch.cuda.is_available())
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    ok &= check("device GB10 sm_121", "GB10" in name and cap == (12, 1), f"{name} {cap}")
    least, greatest = torch.cuda.Stream.priority_range()
    RESULTS["priority_range"] = [least, greatest]
    ok &= check("stream priority range has >1 level", greatest < least, f"least {least} greatest {greatest}")
    from vllm.utils.flashinfer import has_flashinfer

    ok &= check("FlashInfer kernels enabled (not the torch MXFP8 fallback)", bool(has_flashinfer()))
    RESULTS["torch"] = torch.__version__
    try:
        import triton

        RESULTS["triton"] = triton.__version__
    except Exception as e:  # noqa: BLE001
        ok &= check("triton importable", False, repr(e))
    if os.path.exists(os.path.expanduser("~/dsv41-opt-20261021/CLUSTER_LOCK")):
        log("  note: CLUSTER_LOCK exists on this node's home (the lead's window lock): make sure this node is stopped")
    return ok


# ------------------------------------------------------------------------------------------- module setup
def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load_as(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    parent, _, leaf = name.rpartition(".")
    spec.loader.exec_module(m)
    if parent in sys.modules:
        setattr(sys.modules[parent], leaf, m)
    return m


def setup_modules(args):
    """Env as in the launcher, then the r3 / shared_experts / dsv41_prio modules under test."""
    os.environ.setdefault("DSV41_QKV_A_SHARD", "1")
    os.environ.setdefault("DSV41_SKINNY_V2", "1")
    for k in ("DSV41_PRIO_CMP", "DSV41_PRIO_SHARED", "DSV41_PRIO_ENGRAM_HOIST"):
        os.environ[k] = {"DSV41_PRIO_CMP": "prio"}.get(k, "1")  # file flags on; modes toggled at runtime
    import vllm.models.deepseek_v41.nvidia  # noqa: F401  (package)
    import vllm.model_executor.layers.fused_moe.runner  # noqa: F401

    kit = json.load(open(os.path.join(HERE, "KIT.json")))
    rels = {
        "r3": ("vllm.models.deepseek_v41.nvidia.dsv41_r3_dense", "models/deepseek_v41/nvidia/dsv41_r3_dense.py"),
        "se": ("vllm.model_executor.layers.fused_moe.runner.shared_experts",
               "model_executor/layers/fused_moe/runner/shared_experts.py"),
        "prio": ("vllm.models.deepseek_v41.nvidia.dsv41_prio", "models/deepseek_v41/nvidia/dsv41_prio.py"),
    }
    vroot = os.path.dirname(sys.modules["vllm"].__file__)
    mods = {}
    for key, (name, rel) in rels.items():
        if args.source == "installed":
            path = os.path.join(vroot, rel)
        elif key == "prio":
            path = os.path.join(HERE, "files", "vllm", rel)
        else:
            path = os.path.join(HERE, "patched", "vllm", rel)
        want = [f.get("patched_sha256") or f.get("kit_sha256") for f in kit["files"] if f["path"] == rel][0]
        check(f"module under test {rel}", sha(path) == want, f"{path} sha {sha(path)[:16]}")
        mods[key] = load_as(name, path)
    prio = mods["prio"]
    check("dsv41_prio flags at import", prio.CMP_MODE == "prio" and prio.SHARED and prio.ENGRAM, prio.active_line())
    return mods


# ------------------------------------------------------------------------------------------- T1 kernels
def spin_kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def _gt(d):
        return tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l,r", [d], dtype=tl.int64,
                                         is_pure=False, pack=1)

    @triton.jit
    def _sm(d):
        return tl.inline_asm_elementwise("mov.u32 $0, %smid;", "=r,r", [d], dtype=tl.int32, is_pure=True, pack=1)

    @triton.jit
    def spin(start_ptr, end_ptr, sm_ptr, spin_ns):
        pid = tl.program_id(0)
        t0 = _gt(pid)
        t = t0
        while t - t0 < spin_ns:
            t = _gt(pid)
        tl.store(start_ptr + pid, t0)
        tl.store(end_ptr + pid, t)
        tl.store(sm_ptr + pid, _sm(pid))

    @triton.jit
    def stamp(buf_ptr, idx):
        pid = tl.program_id(0)
        tl.store(buf_ptr + idx, _gt(pid))

    return spin, stamp


class Spin:
    def __init__(self, n, warps, spin_us, dev="cuda"):
        import torch

        self.n, self.warps, self.ns = n, warps, int(spin_us * 1000)
        self.s = torch.zeros(n, dtype=torch.int64, device=dev)
        self.e = torch.zeros(n, dtype=torch.int64, device=dev)
        self.sm = torch.zeros(n, dtype=torch.int32, device=dev)

    def launch(self, k):
        k[(self.n,)](self.s, self.e, self.sm, self.ns, num_warps=self.warps)

    def host(self):
        return self.s.cpu().tolist(), self.e.cpu().tolist(), self.sm.cpu().tolist()


def occupancy(s, e):
    ev = sorted([(a, 1) for a in s] + [(b, -1) for b in e])
    cur = best = 0
    for _, d in ev:
        cur += d
        best = max(best, cur)
    return best


def p_metric(flood, probe):
    fs, fe, _ = flood.host()
    ps, _, _ = probe.host()
    pm = st.median(ps)
    return sum(1 for a in fs if a > pm) / len(fs), occupancy(fs, fe), (pm - min(fs)) / 1e3, (max(fe) - min(fs)) / 1e3


def t1(args):
    import torch

    log("== T1 graph stream-priority microtest (gate)")
    spin, _ = spin_kernels()
    least, greatest = torch.cuda.Stream.priority_range()
    reps = 5 if args.quick else 20
    # size the flood: 512-thread CTAs, ~3 per SM; at least 6 waves
    probe_cal = Spin(48, 8, 2.0)
    flood = Spin(48 * 3 * 8, 16, 15.0)
    for _ in range(3):  # JIT + warm
        flood.launch(spin)
        probe_cal.launch(spin)
    torch.cuda.synchronize()
    fs, fe, _ = flood.host()
    occ = occupancy(fs, fe)
    waves = flood.n / max(1, occ)
    RESULTS["t1"]["flood"] = {"ctas": flood.n, "resident": occ, "waves": waves}
    log(f"  flood {flood.n} CTAs x 512 thr, {occ} resident -> {waves:.1f} waves; expected P(honoured) ~ {1 - 1 / waves:.2f}")
    if waves < 3:
        flood = Spin(int(occ * 8), 16, 15.0)
        RESULTS["t1"]["flood_resized"] = flood.n
        flood.launch(spin)
        torch.cuda.synchronize()

    def streams(fp, pp):
        return torch.cuda.Stream(priority=fp), torch.cuda.Stream(priority=pp)

    arms_def = {"same": (least, least), "hi": (least, greatest), "swap": (greatest, least)}
    arms, raw = {}, {}
    # eager
    for arm, (fp, pp) in arms_def.items():
        sf, sp = streams(fp, pp)
        probe = Spin(48, 8, 2.0)
        ps = []
        for _ in range(reps):
            torch.cuda.synchronize()
            with torch.cuda.stream(sf):
                flood.launch(spin)
            with torch.cuda.stream(sp):
                probe.launch(spin)
            torch.cuda.synchronize()
            ps.append(p_metric(flood, probe))
        raw[f"eager_{arm}"] = ps
        arms[f"eager_{arm}"] = classify([p[0] for p in ps])
    # graph, flood captured first (gated) and probe captured first (diagnostic)
    for order in ("flood_first", "probe_first"):
        for arm, (fp, pp) in arms_def.items():
            sf, sp = streams(fp, pp)
            probe = Spin(48, 8, 2.0)
            g = torch.cuda.CUDAGraph()
            cap = torch.cuda.Stream()
            with torch.cuda.graph(g, stream=cap):
                cur = torch.cuda.current_stream()
                sf.wait_stream(cur)
                sp.wait_stream(cur)
                seq = [(sf, flood), (sp, probe)] if order == "flood_first" else [(sp, probe), (sf, flood)]
                for s, k in seq:
                    with torch.cuda.stream(s):
                        k.launch(spin)
                cur.wait_stream(sf)
                cur.wait_stream(sp)
            ps = []
            for _ in range(reps):
                torch.cuda.synchronize()
                g.replay()
                torch.cuda.synchronize()
                ps.append(p_metric(flood, probe))
            key = f"graph_{arm}" if order == "flood_first" else f"graphrev_{arm}"
            raw[key] = ps
            arms[key] = classify([p[0] for p in ps])
            del g
    for k in sorted(raw):
        ps = raw[k]
        log(f"  {k:16s} P median {st.median([p[0] for p in ps]):.3f}  probe start +{st.median([p[2] for p in ps]):7.1f} us"
            f"  flood span {st.median([p[3] for p in ps]):7.1f} us  -> {arms[k]}")
    RESULTS["t1"]["arms"] = arms
    RESULTS["t1"]["raw"] = {k: [list(p) for p in v] for k, v in raw.items()}
    t1_diagnostics(spin, flood, arms_def, reps)
    verdict, reason = gate_verdict(arms)
    RESULTS["t1"]["verdict"] = verdict
    log(f"  GATE graph-priority: {verdict} ({reason})")
    for mut in ("eager_same", "eager_swap", "graph_same", "graph_swap"):
        log(f"  mutant {mut}: {'caught (not-honoured)' if arms.get(mut) == NOT else 'NOT caught -> ' + str(arms.get(mut))}")
    return verdict


def t1_diagnostics(spin, flood, arms_def, reps):
    import torch

    d = RESULTS["t1"].setdefault("diag", {})
    # (a) captured node priorities + re-instantiation with cudaGraphInstantiateFlagUseNodePriority
    try:
        from cuda.bindings import runtime as cudart

        least, greatest = arms_def["hi"]
        sf, sp = torch.cuda.Stream(priority=least), torch.cuda.Stream(priority=greatest)
        probe = Spin(48, 8, 2.0)
        g = torch.cuda.CUDAGraph(keep_graph=True)
        cap = torch.cuda.Stream()
        with torch.cuda.graph(g, stream=cap):
            cur = torch.cuda.current_stream()
            sf.wait_stream(cur)
            sp.wait_stream(cur)
            with torch.cuda.stream(sf):
                flood.launch(spin)
            with torch.cuda.stream(sp):
                probe.launch(spin)
            cur.wait_stream(sf)
            cur.wait_stream(sp)
        raw_graph = cudart.cudaGraph_t(init_value=g.raw_cuda_graph())
        err, _, n = cudart.cudaGraphGetNodes(raw_graph, numNodes=0)
        err, nodes, n = cudart.cudaGraphGetNodes(raw_graph, numNodes=n)
        prios = []
        for nd in nodes:
            err, t = cudart.cudaGraphNodeGetType(nd)
            if t == cudart.cudaGraphNodeType.cudaGraphNodeTypeKernel:
                err, v = cudart.cudaGraphKernelNodeGetAttribute(
                    nd, cudart.cudaLaunchAttributeID.cudaLaunchAttributePriority)
                prios.append(int(v.priority) if err == cudart.cudaError_t.cudaSuccess else f"err {err}")
        d["captured_kernel_node_priorities"] = prios
        log(f"  diag: captured kernel-node priority attributes {prios} (flood stream {least}, probe stream {greatest})")
        flag = cudart.cudaGraphInstantiateFlags.cudaGraphInstantiateFlagUseNodePriority
        err, gexec = cudart.cudaGraphInstantiateWithFlags(raw_graph, int(flag))
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"instantiate: {err}")
        ps = []
        launch_s = torch.cuda.Stream()
        for _ in range(reps):
            torch.cuda.synchronize()
            cudart.cudaGraphLaunch(gexec, launch_s.cuda_stream)
            torch.cuda.synchronize()
            ps.append(p_metric(flood, probe)[0])
        d["use_node_priority_P"] = ps
        log(f"  diag: same graph instantiated with UseNodePriority: P median {st.median(ps):.3f} -> {classify(ps)}")
        cudart.cudaGraphExecDestroy(gexec)
    except Exception as e:  # noqa: BLE001 - diagnostics only
        d["node_priority_error"] = repr(e)
        log(f"  diag: node-priority read / re-instantiate unavailable: {e!r}")
    # (b) one-wave persistent flood: priority cannot preempt a running CTA
    try:
        least, greatest = arms_def["hi"]
        occ = RESULTS["t1"]["flood"]["resident"]
        pers = Spin(int(occ), 16, 200.0)
        probe = Spin(48, 8, 2.0)
        sf, sp = torch.cuda.Stream(priority=least), torch.cuda.Stream(priority=greatest)
        torch.cuda.synchronize()
        with torch.cuda.stream(sf):
            pers.launch(spin)
        with torch.cuda.stream(sp):
            probe.launch(spin)
        torch.cuda.synchronize()
        fs, fe, _ = pers.host()
        ps, _, _ = probe.host()
        wait = (st.median(ps) - min(fs)) / 1e3
        d["persistent_probe_wait_us"] = wait
        log(f"  diag: one-wave 200 us persistent flood: hi-priority probe started after {wait:.1f} us "
            f"(~200 = no preemption; the FC1 starvation of act_and_mul is of this kind)")
    except Exception as e:  # noqa: BLE001
        d["persistent_error"] = repr(e)


# -------------------------------------------------------------------------------------------- helpers
def bits_equal(a, b):
    import torch

    if a.shape != b.shape or a.dtype != b.dtype:
        return False, f"shape/dtype {tuple(a.shape)}/{a.dtype} vs {tuple(b.shape)}/{b.dtype}"
    av = a.contiguous().view(torch.uint8).flatten()
    bv = b.contiguous().view(torch.uint8).flatten()
    if torch.equal(av, bv):
        return True, ""
    i = int((av != bv).nonzero()[0])
    return False, f"first differing byte {i} of {av.numel()}"


def hostile_inputs(M, K, seed, dev):
    import torch

    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(M, K, generator=g) * 0.5
    if seed % 3 == 1:  # large dynamic range per 32-block, exact zeros, ties
        x = x * torch.exp2(torch.randint(-8, 8, (M, K // 32), generator=g).float()).repeat_interleave(32, 1)
        x[:, ::97] = 0.0
        x[:, 1::131] = 1.0
    if seed % 3 == 2:
        x[:, :64] = 448.0
    return x.to(torch.bfloat16).to(dev)


class AG:
    """Stand-in for tensor_model_parallel_all_gather (rank-local, exact copy): concat 8 rank slices."""

    def __call__(self, y, dim=-1):
        import torch

        return torch.cat([y] * 8, dim=dim)


def patch_ag():
    import vllm.distributed as vd

    vd.tensor_model_parallel_all_gather = AG()


def sleep_on(stream, cycles=100_000):
    import torch

    with torch.cuda.stream(stream):
        torch.cuda._sleep(cycles)


def capture(fn, static_in):
    """Capture fn(static_in) -> tuple of tensors; return (graph, outputs)."""
    import torch

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn(static_in)  # warm (first-use stream/event creation happens outside capture)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        outs = fn(static_in)
    return g, outs


def run_modes(name, make_fn, refs_fn, Ms, n_in, mutants=()):
    """make_fn(mode, M) -> fn(x) -> tuple; refs_fn(M, x) -> reference tuple (live path, eager). Bitwise checks in
    eager and graph replays for every mode; mutants must mismatch at every M."""
    import torch

    for mode in [m for m in ("off", "serial", "prio") if m in make_fn.modes]:
        for M in Ms:
            fn = make_fn(mode, M)
            bad = []
            ins = [make_fn.input(M, i) for i in range(n_in)]
            refs = [tuple(t.clone() for t in refs_fn(M, x)) for x in ins]
            for x, ref in zip(ins, refs):  # eager
                out = fn(x)
                torch.cuda.synchronize()
                for a, b in zip(out, ref):
                    ok, why = bits_equal(a, b)
                    if not ok:
                        bad.append(f"eager {why}")
            static = ins[0].clone()
            g, outs = capture(fn, static)
            for x, ref in zip(ins[1:] + ins[:1], refs[1:] + refs[:1]):  # graph, new input every replay
                static.copy_(x)
                g.replay()
                torch.cuda.synchronize()
                for a, b in zip(outs, ref):
                    ok, why = bits_equal(a, b)
                    if not ok:
                        bad.append(f"graph {why}")
            check(f"{name} {mode} M={M} bitwise (eager + {n_in} graph replays)", not bad, "; ".join(bad[:3]))
            del g
    for mname, mk in mutants:
        caught = []
        for M in Ms:
            fn = mk(M)
            ins = [make_fn.input(M, i) for i in range(n_in)]
            refs = [tuple(t.clone() for t in refs_fn(M, x)) for x in ins]
            static = ins[0].clone()
            g, outs = capture(fn, static)
            mism = False
            for x, ref in zip(ins[1:] + ins[:1], refs[1:] + refs[:1]):
                static.copy_(x)
                g.replay()
                torch.cuda.synchronize()
                mism |= any(not bits_equal(a, b)[0] for a, b in zip(outs, ref))
            caught.append(mism)
            del g
        check(f"{name} mutant '{mname}' caught at every M", all(caught),
              f"caught at M={[M for M, c in zip(Ms, caught) if c]} of {list(Ms)}")


# --------------------------------------------------------------------------------------------- T2 CMPPRIO
def t2(mods, args):
    import torch

    log("== T2 CMPPRIO (real qkv_a shard GEMM vs compressor flood)")
    r3, prio = mods["r3"], mods["prio"]
    from vllm.utils.multi_stream_utils import execute_in_parallel

    patch_ag()
    dev = "cuda"
    K, N, tp = 5120, 1792, 8
    n = N // tp
    g = torch.Generator(device="cpu").manual_seed(7)
    wq, ws = r3.quantize_rows_mxfp8((torch.randn(n, K, generator=g) * 0.02).to(torch.bfloat16).to(dev))
    st_ = object.__new__(r3.QkvAShard)
    st_.N, st_.n, st_.K, st_.tp = N, n, K, tp
    st_.w, st_.s_rm, st_.s_swz = wq, ws, r3.swizzle_scale(ws, n, K)
    st_._r2 = None
    v2_err = ""
    if r3.R2_V2:  # production (launcher 166d7aac) serves qkv_a with skinny_v2_kernel (t2: g48x1x1 b256)
        try:
            from vllm.models.deepseek_v41.nvidia import dsv41_r2_skinny as _r2s

            st_._r2 = _r2s.build(st_)  # None = the live path would silently keep v1: that is a FAIL here
        except Exception as e:  # noqa: BLE001
            v2_err = repr(e)
    ext_ok = r3.QKV_GEMM == "skinny" and r3.skinny_ext() is not None
    for name, ok, detail in skinny_serving_checks(r3.R2_V2, r3.QKV_GEMM, ext_ok, st_._r2, r3.QKV_MAX_M, v2_err):
        check(name, ok, detail)
    if r3.R2_V2 and st_._r2 is not None:
        # dynamic: the shard path really launches the v2 kernel at every served M (count the ext entry calls)
        calls = {"n": 0}
        real_e = st_._r2.e

        class _Spy:
            def __getattr__(self, k):
                return getattr(real_e, k)

            def skinny_v2_a8(self, *a, **kw):
                calls["n"] += 1
                return real_e.skinny_v2_a8(*a, **kw)

        st_._r2.e = _Spy()
        r3._DSV41_PRIO_CMP = False
        per_m = {}
        gs = torch.Generator(device="cpu").manual_seed(11)  # own generator: the T2 weights below stay as before
        try:
            for M in range(1, r3.QKV_MAX_M + 1):
                calls["n"] = 0
                h = torch.randn(M, K, generator=gs).to(torch.bfloat16).to(dev)
                st_(h)
                per_m[M] = calls["n"]
            torch.cuda.synchronize()
        finally:
            st_._r2.e = real_e
        check("serving qkv_a GEMM launches skinny_v2_a8 once per call at every served M",
              all(v == 1 for v in per_m.values()) and len(per_m) == r3.QKV_MAX_M, f"calls per M {per_m}")
    NC = int(os.environ.get("PRIO_T2_NC", "2048"))
    Wc = (torch.randn(NC, K, generator=g) * 0.02).to(torch.bfloat16).to(dev)
    Wi = (torch.randn(32, K, generator=g) * 0.02).to(torch.bfloat16).to(dev)
    aux = [torch.cuda.Stream(), torch.cuda.Stream()]

    class Attn:
        prefix = "dev.attn"

        def __init__(self):
            self._dsv41_qkv_shard = st_
            self.ln_events = [torch.cuda.Event() for _ in range(4)]

        def _fused_wqa_wkv_gemm(self, h):
            y = r3.qkv_a_sharded(self, h)
            assert y is not None, "shard path not taken"
            return y

    spin_cmp = {"on": False}

    def aux_fns(h):
        def cmp_():
            if spin_cmp["on"]:
                torch.cuda._sleep(200_000)
            return torch.mm(h, Wc.T, out_dtype=torch.float32)

        return [cmp_, lambda: torch.mm(h, Wi.T)]

    def make(mode, M, mutant=None):
        attn = Attn()
        if mutant == "drop_cmp_wait":
            class NoWait(torch.cuda.Event):
                def wait(self, stream=None):
                    return None
            attn.ln_events[1] = NoWait()

        def fn(h):
            r3._DSV41_PRIO_CMP = mode != "off"
            if mode == "off":
                qr, (kvs, iw) = execute_in_parallel(lambda: attn._fused_wqa_wkv_gemm(h), aux_fns(h), attn.ln_events[0],
                                                    attn.ln_events[1:3], aux, enable=True)
            else:
                prio.CMP_MODE = mode
                qr, kvs, iw = prio.cmp_parallel_input_projections(attn, h, aux_fns(h), aux, True)
            if mutant is not None:  # keep the capture legal: join the side streams after the consumers
                cur = torch.cuda.current_stream()
                for s in aux + [prio._HI_STREAMS.get((torch.cuda.current_device(), "cmp"))]:
                    if s is not None:
                        cur.wait_stream(s)
            return qr.clone(), kvs.clone(), iw.clone()

        return fn

    def refs(M, x):
        r3._DSV41_PRIO_CMP = False
        attn = Attn()
        qr, (kvs, iw) = execute_in_parallel(lambda: attn._fused_wqa_wkv_gemm(x), aux_fns(x), attn.ln_events[0],
                                            attn.ln_events[1:3], aux, enable=True)
        torch.cuda.synchronize()
        return qr, kvs, iw

    make.modes = ("off", "serial", "prio")
    make.input = lambda M, i: hostile_inputs(M, K, 1000 * M + i, dev)
    Ms = [1, 2, 4, 8] if args.quick else list(range(1, 9))
    n_in = 4 if args.quick else 12

    orig_leave = prio.cmp_leave
    orig_gemm = st_.gemm

    def mut_nojoin(M):
        def bad_leave(tok, y):
            if tok is None:
                return y
            tok.done.record(tok.hi)
            tok.ctx.__exit__(None, None, None)
            tok.ctx = None
            tok.used = True
            return y

        def slow_gemm(xq, xs):
            torch.cuda._sleep(200_000)
            return orig_gemm(xq, xs)

        prio.cmp_leave = bad_leave
        st_.gemm = slow_gemm
        st_._r2_saved, st_._r2 = st_._r2, None
        return make("prio", M, mutant="nojoin")

    def mut_dropcmp(mode):
        def mk(M):
            spin_cmp["on"] = True
            return make(mode, M, mutant="drop_cmp_wait")
        return mk

    try:
        run_modes("T2", make, refs, Ms, n_in, mutants=[("prio: no join after the shard GEMM", mut_nojoin)])
    finally:
        prio.cmp_leave = orig_leave
        st_.gemm = orig_gemm
        if hasattr(st_, "_r2_saved"):
            st_._r2 = st_._r2_saved
    try:
        run_modes("T2", types.SimpleNamespace(modes=(), input=make.input), refs, Ms, n_in,
                  mutants=[("serial: compressor done-event wait dropped", mut_dropcmp("serial")),
                           ("prio: compressor done-event wait dropped", mut_dropcmp("prio"))])
    finally:
        spin_cmp["on"] = False
    t2_timing(mods, make, make.input, args)


def t2_timing(mods, make, inp, args):
    """Informational: main-chain latency (start -> qkv_a all-gather done) per mode in graph replay, M=4."""
    import torch

    _, stamp = spin_kernels()
    buf = torch.zeros(4, dtype=torch.int64, device="cuda")
    res = {}
    for mode in ("off", "serial", "prio"):
        fn0 = make(mode, 4)

        def fn(h, fn0=fn0):
            stamp[(1,)](buf, 0)
            out = fn0(h)
            stamp[(1,)](buf, 1)
            return out

        x = inp(4, 0)
        static = x.clone()
        g, _ = capture(fn, static)
        ts = []
        for _ in range(20 if args.quick else 100):
            g.replay()
            torch.cuda.synchronize()
            b = buf.cpu().tolist()
            ts.append((b[1] - b[0]) / 1e3)
        res[mode] = st.median(ts)
        del g
    RESULTS["timing"]["t2_main_chain_us"] = res
    log(f"  timing (info): start -> outputs joined, median us: {res}")


# ---------------------------------------------------------------------------------------------- T3 SHARED
def mx_linear(N, K, seed, dev="cuda"):
    import torch

    from vllm.models.deepseek_v41.nvidia import dsv41_r3_dense as r3

    g = torch.Generator(device="cpu").manual_seed(seed)
    w, s = r3.quantize_rows_mxfp8((torch.randn(N, K, generator=g) * 0.02).to(torch.bfloat16).to(dev))
    return w, r3.swizzle_scale(s, N, K), K


def mx_mm(x, lin):
    import torch

    from vllm.models.deepseek_v41.nvidia import dsv41_r3_dense as r3
    from vllm.utils import flashinfer as vfi

    w, s_swz, K = lin
    xq, xs, _ = r3._quantize_act(x.contiguous(), K)
    return vfi.mm_mxfp8(xq, w.t(), xs, s_swz, out_dtype=torch.bfloat16, backend="cutlass")


def t3(mods, args):
    import torch

    log("== T3 SHARED (SharedExperts stream swap, MXFP8 shared MLP vs MXFP8 routed flood)")
    se_mod, prio = mods["se"], mods["prio"]
    K = 5120
    gate_up, down = mx_linear(768, K, 11), mx_linear(K, 384, 12)  # K=384: 128-aligned for the cutlass tile
    routed = [mx_linear(K, K, 13 + i) for i in range(3)]

    class MLP(torch.nn.Module):
        def forward(self, x):
            h = mx_mm(x, gate_up)
            a = (torch.nn.functional.silu(h[:, :384]) * h[:, 384:]).contiguous()
            return mx_mm(a, down)

    cfg = types.SimpleNamespace(moe_parallel_config=types.SimpleNamespace(
        enable_eplb=False, all2all_backend="x", use_fi_nvl_two_sided_kernels=False))

    def make_se(on):
        se_mod._DSV41_PRIO_SHARED = on
        prio.SHARED = on
        return se_mod.SharedExperts(MLP(), cfg, False, lambda: False)

    se_off, se_on = make_se(False), make_se(True)
    least, greatest = torch.cuda.Stream.priority_range()
    check("shared stream swapped to a high-priority stream", se_on._stream.priority == greatest
          and se_off._stream.priority == least and se_on._stream is not se_off._stream,
          f"on {se_on._stream.priority} off {se_off._stream.priority}")
    spin = {"on": False}

    def make(mode, M, mutant=None):
        se = se_on if mode == "prio" or mutant else se_off

        def fn(x):
            assert se.maybe_forward_async(x)
            r = x
            for lin in routed:
                r = mx_mm(r, lin)
            if mutant is None:
                se.wait()
            out = se.output
            res = r + out
            if mutant is not None:
                torch.cuda.current_stream().wait_stream(se._stream)
            return (res.clone(),)

        return fn

    orig_fwd = MLP.forward

    def refs(M, x):
        return make("off", M)(x)

    make.modes = ("off", "prio")
    make.input = lambda M, i: hostile_inputs(M, K, 3000 * M + i, "cuda")
    Ms = [1, 4, 8] if args.quick else [1, 2, 3, 4, 6, 8]

    def mut(M):
        def slow(self, x):
            torch.cuda._sleep(200_000)
            return orig_fwd(self, x)

        MLP.forward = slow
        return make("prio", M, mutant="nowait")

    try:
        run_modes("T3", make, refs, Ms, 4 if args.quick else 10, mutants=[("shared: no wait()", mut)])
    finally:
        MLP.forward = orig_fwd


# ------------------------------------------------------------------------------------------ T4 ENGRAM
def t4(mods, args):
    import torch

    log("== T4 ENGRAM_HOIST (hoisted MXFP8 wkv GEMM on the side stream)")
    prio = mods["prio"]
    patch_ag()
    dev = "cuda"
    KW, NW = 6144, 3200
    wkv_lin = mx_linear(NW, KW, 21)
    layer_lins = [mx_linear(5120, 5120, 30 + i) for i in range(2)]
    table = (torch.randn(4096, 3, 256, generator=torch.Generator().manual_seed(5)) * 0.1).to(torch.bfloat16).to(dev)

    class QM:
        slow = False

        def apply(self, layer, x, bias=None):
            if QM.slow:
                torch.cuda._sleep(300_000)
            return mx_mm(x, wkv_lin)

    class ColumnParallelLinear:
        def __init__(self):
            self.quant_method, self.bias, self.skip_bias_add = QM(), None, False
            self.gather_output, self.tp_size, self.return_bias = True, 8, False

        def __call__(self, x):  # == ColumnParallelLinear.forward
            out = self.quant_method.apply(self, x, None)
            import vllm.distributed as vd

            return vd.tensor_model_parallel_all_gather(out)

    class Engram:
        _batch = None
        use_sequence_parallel = False
        layer_hash_index = 0

        def __init__(self):
            self.wkv = ColumnParallelLinear()
            self.embed_tokens = types.SimpleNamespace(tp_size=8)

        def embed(self, hash_ids):  # lookup + rows all-gather stand-in (dim 1: 3 local heads -> 24)
            rows = table[hash_ids[:, 0] % table.shape[0]]
            import vllm.distributed as vd

            return vd.tensor_model_parallel_all_gather(rows, dim=1)

    eng = Engram()
    prio.EH.side = torch.cuda.Stream(priority=torch.cuda.Stream.priority_range()[0])
    prio.EH.events[id(eng)] = torch.cuda.Event()
    prio.EH.on = True

    def layers(h):
        for lin in layer_lins:
            h = mx_mm(h, lin)
        return h

    def make(mode, M, mutant=None):
        def fn(x):
            hash_ids = (x[:, :2].float().abs() * 1000).to(torch.int64)
            h = x
            if mode == "prio" or mutant:
                prio.hoist(eng, hash_ids, host=0)
                h = layers(h)
                if mutant is None:
                    kv = prio.engram_take(eng, hash_ids)
                else:
                    rec = prio.EH.pending.pop(id(eng))
                    import vllm.distributed as vd

                    kv = vd.tensor_model_parallel_all_gather(rec.out)  # mutant: no join
                    torch.cuda.current_stream().wait_stream(prio.EH.side)
            else:
                h = layers(h)
                kv = eng.wkv(eng.embed(hash_ids).flatten(-2))  # the live statement
            return h.clone(), kv.clone()

        return fn

    def refs(M, x):
        return make("off", M)(x)

    make.modes = ("off", "prio")
    make.input = lambda M, i: hostile_inputs(M, 5120, 5000 * M + i, dev)
    Ms = [1, 4, 8] if args.quick else [1, 2, 3, 4, 6, 8]

    def mut(M):
        QM.slow = True
        return make("prio", M, mutant="nojoin")

    try:
        run_modes("T4", make, refs, Ms, 4 if args.quick else 10, mutants=[("engram: take without join", mut)])
    finally:
        QM.slow = False
    check("T4 no stale hoist left", not prio.EH.pending)


# ------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("kit", "installed"), default="kit")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json", default="")
    ap.add_argument("--cpu-selftest", action="store_true")
    ap.add_argument("--only", default="", help="comma list of t1,t2,t3,t4")
    args = ap.parse_args()
    if args.cpu_selftest:
        return cpu_selftest()
    t0 = time.time()
    if not env_check(args):
        log("ENVIRONMENT CHECK FAILED: refusing to run (a plain ssh shell tests the wrong reference)")
        return 1
    mods = setup_modules(args)
    only = set(args.only.split(",")) if args.only else {"t1", "t2", "t3", "t4"}
    verdict = "INCONCLUSIVE"
    if "t1" in only:
        verdict = t1(args)
    for name, fn in (("t2", t2), ("t3", t3), ("t4", t4)):
        if name in only:
            try:
                fn(mods, args)
            except Exception as e:  # noqa: BLE001
                import traceback

                traceback.print_exc()
                check(f"{name} ran", False, repr(e))
    RESULTS["elapsed_s"] = time.time() - t0
    RESULTS["fails"] = FAILS
    code = exit_code(verdict, FAILS)
    RESULTS["exit"] = code
    log(f"== SUMMARY: T1 gate {verdict}; {len(RESULTS['checks']) - len(FAILS)}/{len(RESULTS['checks'])} checks pass; "
        f"exit {code}; {RESULTS['elapsed_s']:.0f} s")
    if args.json:
        json.dump(RESULTS, open(args.json, "w"), indent=1, default=str)
    return code


if __name__ == "__main__":
    sys.exit(main())
