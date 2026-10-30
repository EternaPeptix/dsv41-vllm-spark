#!/usr/bin/env python3
"""bench_rdma_ar.py: 8-rank offline correctness + latency test of the DSv4.1 rdma_ar integration
against vLLM's PyNccl, one process per Spark (run-microbench.sh starts it on all 8).

Needs the dsv41 service STOPPED on all ranks (GPU memory + fabric must be idle).
Env: RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT (+ the NCCL_* / RDMA_AR_* knobs).

Phases (all ranks run all of them in lock-step):
  1 correctness, eager: rdma one-shot / two-shot vs an exact reference (NCCL all-gather of the
    inputs, fp32 sum in rank order, rounded once) -> must be BITWISE equal; also vs NCCL
    all-reduce (reported max abs / rel diff); output hash identical on every rank.
  2 all-gather correctness (dim 0, 1, -1) vs PyNccl all-gather: bitwise.
  3 CUDA-graph correctness: one graph mixing rdma ARs (1/6/48 rows, one-shot + two-shot), an
    rdma all-gather and a PyNccl AR (8-byte, as the argmax gather fallback); 50 replays with
    fresh inputs each, every replay checked.
  4 latency: graph of N back-to-back ARs (default 88, one decode step's worth) per size and
    backend, ABBA ordered (pynccl, rdma, rdma, pynccl) x trials; per-op us = replay time / N.
  5 optional skew (--skew-us): each rank sleeps rank/(world-1)*skew before every AR inside the
    graph, to see how both backends absorb peer wait.
  6 soak: --soak replays of the 88-op 6-row graph, then a final correctness replay.
Writes <out>.rank<R>.json; rank 0 prints the summary table.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist

HIDDEN = 5120


def log0(*a):
    if dist.get_rank() == 0:
        print(*a, flush=True)


def load_wrapper(path):
    spec = importlib.util.spec_from_file_location("dsv41_rdma_all_reduce", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_pynccl(group, device):
    try:
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

        c = PyNcclCommunicator(group=group, device=device)
        if not c.disabled:
            return "pynccl", c
    except Exception as e:  # noqa: BLE001
        print(f"rank {dist.get_rank()}: PyNccl unavailable ({e}); using torch NCCL", flush=True)
    return "torch_nccl", dist.new_group(backend="nccl")


class Nccl:
    def __init__(self, group, device):
        self.kind, self.c = make_pynccl(group, device)

    def all_reduce(self, x):
        if self.kind == "pynccl":
            return self.c.all_reduce(x, stream=torch.cuda.current_stream())
        out = x.clone()
        dist.all_reduce(out, group=self.c)
        return out

    def all_gather_stacked(self, x, world):
        out = torch.empty((world,) + tuple(x.shape), dtype=x.dtype, device=x.device)
        if self.kind == "pynccl":
            self.c.all_gather(out.view(-1), x.contiguous().view(-1), stream=torch.cuda.current_stream())
        else:
            dist.all_gather_into_tensor(out.view(-1), x.contiguous().view(-1), group=self.c)
        return out


def sha(t):
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]


def rand_input(rows, dtype, rank, salt, device, cols=HIDDEN):
    g = torch.Generator(device="cpu").manual_seed(1000003 * rank + 7919 * rows + salt)
    return (torch.randn(rows, cols, generator=g) * 0.5).to(dtype).to(device)


def exact_ref(nccl, x, world):
    """fp32 sum of all ranks' inputs in rank order, rounded once (the rdma rule)."""
    stacked = nccl.all_gather_stacked(x, world).float()
    acc = torch.zeros_like(stacked[0])
    for r in range(world):
        acc += stacked[r]
    return acc.to(x.dtype)


def same_on_all_ranks(h, cpu_group):
    hs = [None] * dist.get_world_size()
    dist.all_gather_object(hs, h, group=cpu_group)
    return all(x == hs[0] for x in hs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="1,2,4,6,12,24,48")
    ap.add_argument("--fp32-rows", default="1,6,24")
    ap.add_argument("--n-ops", type=int, default=88)
    ap.add_argument("--replays", type=int, default=50)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--skew-us", type=float, default=0.0)
    ap.add_argument("--soak", type=int, default=2000)
    ap.add_argument("--wrapper", default=os.path.expanduser("~/rdma_ar_dsv41/rdma_all_reduce.py"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    cpu_group = dist.group.WORLD
    res = {"rank": rank, "world": world, "env": {k: v for k, v in os.environ.items()
                                                 if k.startswith(("RDMA_AR", "DSV41_", "NCCL_"))}}

    nccl = Nccl(cpu_group, device)
    res["nccl_kind"] = nccl.kind
    W = load_wrapper(a.wrapper)
    t0 = time.time()
    rd = W.RdmaAllReduce(cpu_group, rank, world, device)
    res["rdma_init_s"] = time.time() - t0
    assert not rd.disabled, "rdma_ar failed to initialise (see log)"
    res["arena_info"] = list(rd.h.arena_info())
    res["pingpong_ns"] = rd.h.pingpong(2000)
    log0(f"init ok: nccl={nccl.kind} arena={res['arena_info']} pingpong={res['pingpong_ns']:.0f} ns")

    rows_list = [int(r) for r in a.rows.split(",")]
    fp32_rows = [int(r) for r in a.fp32_rows.split(",") if r]

    # warm NCCL (eager) before any graph capture
    for _ in range(3):
        nccl.all_reduce(torch.ones(16, HIDDEN, dtype=torch.bfloat16, device=device))
    torch.cuda.synchronize()

    # ---- 1 correctness eager ----------------------------------------------------------------
    corr = []
    fails = 0
    cases = [(r, torch.bfloat16) for r in rows_list] + [(r, torch.float32) for r in fp32_rows]
    for rows, dtype in cases:
        x = rand_input(rows, dtype, rank, 1, device)
        ref = exact_ref(nccl, x, world)
        nc = nccl.all_reduce(x)
        nbytes = x.numel() * x.element_size()
        for two in (False, True):
            if nbytes > rd.max_reduce_bytes:
                continue
            y = rd.h.all_reduce(x, two)
            torch.cuda.synchronize()
            bit = bool(torch.equal(y, ref))
            allsame = same_on_all_ranks(sha(y), cpu_group)
            d = (y.float() - nc.float()).abs()
            rel = (d / nc.float().abs().clamp_min(1e-3)).max().item()
            ok = bit and allsame
            fails += 0 if ok else 1
            corr.append(dict(rows=rows, dtype=str(dtype), bytes=nbytes, two_shot=two, bitwise_ref=bit,
                             same_all_ranks=allsame, max_abs_vs_nccl=d.max().item(), max_rel_vs_nccl=rel))
    res["correctness"] = corr
    log0(f"[1] eager AR correctness: {len(corr) - fails}/{len(corr)} cases bitwise==ref and identical on all ranks")

    # ---- 2 all-gather -----------------------------------------------------------------------
    agc = []
    for shape in [(6, 2), (48, 2), (6, 4, 512), (48, 4, 512), (6, 16160)]:
        for dtype in (torch.float32, torch.bfloat16):
            x = rand_input(1, dtype, rank, 2, device, cols=int(torch.tensor(shape).prod())).view(shape)
            nbytes = x.numel() * x.element_size()
            if not rd.should_gather(x):
                agc.append(dict(shape=shape, dtype=str(dtype), bytes=nbytes, routed=False))
                continue
            y = rd.all_gather_stacked(x)
            ref = nccl.all_gather_stacked(x, world)
            torch.cuda.synchronize()
            ok = bool(torch.equal(y, ref))
            fails += 0 if ok else 1
            agc.append(dict(shape=shape, dtype=str(dtype), bytes=nbytes, routed=True, bitwise=ok))
    res["all_gather"] = agc
    log0(f"[2] all-gather: {sum(1 for c in agc if c.get('bitwise'))}/{sum(1 for c in agc if c['routed'])} routed cases bitwise "
         f"({sum(1 for c in agc if not c['routed'])} not routed: size %16 != 0)")

    # ---- 3 graph correctness ----------------------------------------------------------------
    specs = [(1, False), (6, False), (6, True), (48, True), (48, False)]
    ins = [torch.zeros(r, HIDDEN, dtype=torch.bfloat16, device=device) for r, _ in specs]
    g_in = torch.zeros(6, 4, 512, dtype=torch.bfloat16, device=device)
    tiny = torch.zeros(1, 2, dtype=torch.float32, device=device)  # 8 B -> NCCL as in serving
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):  # warm-up allocations on the capture stream
        outs = [rd.h.all_reduce(t, two) for t, (_, two) in zip(ins, specs)]
        torch.cuda.synchronize()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        outs = []
        for i, (t, (_, two)) in enumerate(zip(ins, specs)):
            outs.append(rd.h.all_reduce(t, two))
            if i == 1:
                tiny_out = nccl.all_reduce(tiny)
                g_out = rd.all_gather_stacked(g_in)
    gfail = 0
    for rep in range(a.replays):
        for t, (r, _) in zip(ins, specs):
            t.copy_(rand_input(r, torch.bfloat16, rank, 100 + rep, device))
        g_in.copy_(rand_input(1, torch.bfloat16, rank, 200 + rep, device, cols=6 * 4 * 512).view(6, 4, 512))
        tiny.fill_(float(rank + rep))
        g.replay()
        torch.cuda.synchronize()
        for t, o in zip(ins, outs):
            if not torch.equal(o, exact_ref(nccl, t, world)):
                gfail += 1
        if not torch.equal(g_out, nccl.all_gather_stacked(g_in, world)):
            gfail += 1
        if not bool((tiny_out == float(sum(range(world)) + world * rep)).all().item()):
            gfail += 1
    fails += gfail
    res["graph_correctness"] = dict(replays=a.replays, failures=gfail)
    log0(f"[3] graph replay correctness: {a.replays} replays, {gfail} failures on rank 0")

    # ---- 4/5 latency -------------------------------------------------------------------------
    sleep_cycles = 0
    if a.skew_us > 0:
        # GB10 SM clock ~2.4 GHz at load; this is an approximate per-rank delay, reported as such
        sleep_cycles = int(a.skew_us * 2400 * rank / max(1, world - 1))

    def graph_for(fn, x):
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(2):
                fn(x)
        torch.cuda.current_stream().wait_stream(st)
        torch.cuda.synchronize()
        gg = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gg):
            for _ in range(a.n_ops):
                if sleep_cycles:
                    torch.cuda._sleep(sleep_cycles)
                fn(x)
        return gg

    def time_graph(gg):
        for _ in range(5):
            gg.replay()
        torch.cuda.synchronize()
        dist.barrier(group=cpu_group)
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(a.replays):
            gg.replay()
        e1.record()
        torch.cuda.synchronize()
        return e0.elapsed_time(e1) * 1000.0 / (a.replays * a.n_ops)

    lat = []
    backends = {
        "pynccl": lambda x: nccl.all_reduce(x),
        "rdma_1shot": lambda x: rd.h.all_reduce(x, False),
        "rdma_2shot": lambda x: rd.h.all_reduce(x, True),
        "rdma_auto": lambda x: rd.all_reduce(x),
    }
    for rows in rows_list:
        x = rand_input(rows, torch.bfloat16, rank, 3, device)
        graphs = {k: graph_for(f, x) for k, f in backends.items()}
        per = {k: [] for k in backends}
        order = ["pynccl", "rdma_1shot", "rdma_2shot", "rdma_auto"]
        for t in range(a.trials):
            seq = order if t % 2 == 0 else list(reversed(order))  # ABBA
            for k in seq:
                per[k].append(time_graph(graphs[k]))
        row = dict(rows=rows, bytes=x.numel() * 2, skew_us=a.skew_us,
                   **{f"{k}_us": statistics.median(v) for k, v in per.items()},
                   **{f"{k}_all": v for k, v in per.items()})
        lat.append(row)
        log0(f"[4] rows={rows:3d} bytes={row['bytes']:7d}  pynccl {row['pynccl_us']:7.1f} us | rdma 1shot "
             f"{row['rdma_1shot_us']:6.1f}  2shot {row['rdma_2shot_us']:6.1f}  auto {row['rdma_auto_us']:6.1f}  "
             f"speedup(auto) {row['pynccl_us'] / row['rdma_auto_us']:.2f}x")
        del graphs
    res["latency"] = lat
    res["rdma_stats"] = rd.stats(reset=True)

    # ---- 6 soak --------------------------------------------------------------------------------
    if a.soak > 0:
        x = rand_input(6, torch.bfloat16, rank, 4, device)
        gg = graph_for(lambda t: rd.all_reduce(t), x)
        t0 = time.time()
        for _ in range(a.soak):
            gg.replay()
        torch.cuda.synchronize()
        y = rd.all_reduce(x)
        ok = bool(torch.equal(y, exact_ref(nccl, x, world)))
        fails += 0 if ok else 1
        res["soak"] = dict(replays=a.soak, ops=a.soak * a.n_ops, wall_s=time.time() - t0, final_ok=ok)
        log0(f"[6] soak: {a.soak * a.n_ops} ops in {res['soak']['wall_s']:.1f} s, final check {'OK' if ok else 'FAIL'}")

    res["failures"] = fails
    allf = [None] * world
    dist.all_gather_object(allf, fails, group=cpu_group)
    res["failures_all_ranks"] = allf
    with open(f"{a.out}.rank{rank}.json", "w") as f:
        json.dump(res, f, indent=1, default=str)
    log0(f"DONE failures per rank: {allf}  -> {'PASS' if not any(allf) else 'FAIL'}")
    dist.barrier(group=cpu_group)
    os._exit(0)  # proxy thread / QPs are left to process teardown


if __name__ == "__main__":
    sys.exit(main())
