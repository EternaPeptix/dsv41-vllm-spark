#!/usr/bin/env python3
"""bench_rdma_ext.py (r2 rdma-ext): 8-rank OFFLINE sweep of the sizes the round-1 bench did not
cover, to set RDMA_AR_MAX_REDUCE_BYTES / RDMA_AR_MAX_BYTES / RDMA_AR_TWOSHOT_MIN_BYTES.

Lives next to bench_rdma_ar.py in ~/rdma_ar_dsv41 and reuses its helpers. Started on all 8 Sparks
by run-ext-bench.sh. Needs the dsv41 service STOPPED (unloaded window).

Phases (all ranks in lock-step):
  X1 AR correctness, bf16, rows (x5120) from --rows: one-shot (if <= RDMA_AR_MAX_BYTES) and
     two-shot (if <= RDMA_AR_MAX_REDUCE_BYTES) must be BITWISE equal to the fixed-order fp32
     reference and identical on all ranks.
  X2 AR latency, CUDA graph of --n-ops back-to-back ARs per size (81 = one target verify step),
     ABBA over pynccl / rdma_1shot / rdma_2shot; per-op us = replay time / n_ops. For bytes > 1 MiB
     n_ops drops to --n-ops-big to bound runtime.
  X3 all-gather correctness (bitwise vs PyNccl) + latency for the decode all-gathers:
     logits shard (rows, 16160) bf16 and engram (rows, 3, 256) bf16, graph of --n-ag-ops.
  X4 (--big) prefill-medium ARs, rows from --big-rows (205/410/819 = 2/4/8 MiB; needs
     RDMA_AR_MAX_BYTES and RDMA_AR_MAX_REDUCE_BYTES >= 8 MiB) plus 4096 rows (40 MiB) for PyNccl
     only, as the prefill reference.
Writes <out>.rank<R>.json; rank 0 prints the tables and the crossover summary.
"""
import argparse
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_rdma_ar as B  # noqa: E402  (same directory on every Spark)

HIDDEN = B.HIDDEN
VOCAB_SHARD = 16160  # 129280 / 8
log0 = B.log0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="48,52,56,64,72,80,96,128,160,192")
    ap.add_argument("--n-ops", type=int, default=81)
    ap.add_argument("--n-ops-big", type=int, default=16)
    ap.add_argument("--replays", type=int, default=30)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--ag-logit-rows", default="5,6,30,48,60,64,80,96")
    ap.add_argument("--ag-engram-rows", default="6,48,96,192")
    ap.add_argument("--n-ag-ops", type=int, default=8)
    ap.add_argument("--big", action="store_true")
    ap.add_argument("--big-rows", default="205,410,819")
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
    nccl = B.Nccl(cpu_group, device)
    W = B.load_wrapper(a.wrapper)
    rd = W.RdmaAllReduce(cpu_group, rank, world, device)
    assert not rd.disabled, "rdma_ar failed to initialise (see log)"
    res["arena_info"] = list(rd.h.arena_info())
    res["limits"] = dict(max_bytes=rd.max_bytes, max_reduce_bytes=rd.max_reduce_bytes,
                         twoshot_min_bytes=rd.twoshot_min_bytes)
    log0(f"init ok: nccl={nccl.kind} arena={res['arena_info']} limits={res['limits']}")
    for _ in range(3):
        nccl.all_reduce(torch.ones(16, HIDDEN, dtype=torch.bfloat16, device=device))
    torch.cuda.synchronize()
    fails = 0

    def fits_1shot(nb):
        return nb <= rd.max_bytes and nb <= rd.max_reduce_bytes

    def fits_2shot(nb):
        return nb <= rd.max_reduce_bytes

    def graph_for(fn, x, n_ops):
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(2):
                fn(x)
        torch.cuda.current_stream().wait_stream(st)
        torch.cuda.synchronize()
        gg = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gg):
            for _ in range(n_ops):
                fn(x)
        return gg

    def time_graph(gg, n_ops):
        for _ in range(3):
            gg.replay()
        torch.cuda.synchronize()
        dist.barrier(group=cpu_group)
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(a.replays):
            gg.replay()
        e1.record()
        torch.cuda.synchronize()
        return e0.elapsed_time(e1) * 1000.0 / (a.replays * n_ops)

    def sweep_ar(rows_list, tag, pynccl_only_rows=()):
        out = []
        for rows in list(rows_list) + list(pynccl_only_rows):
            x = B.rand_input(rows, torch.bfloat16, rank, 11, device)
            nb = x.numel() * 2
            n_ops = a.n_ops if nb <= (1 << 20) else a.n_ops_big
            be = {"pynccl": lambda t: nccl.all_reduce(t)}
            if rows not in pynccl_only_rows:
                if fits_1shot(nb):
                    be["rdma_1shot"] = lambda t: rd.h.all_reduce(t, False)
                if fits_2shot(nb):
                    be["rdma_2shot"] = lambda t: rd.h.all_reduce(t, True)
            # correctness first (eager)
            ref = B.exact_ref(nccl, x, world)
            corr = {}
            for k in [k for k in be if k != "pynccl"]:
                y = be[k](x)
                torch.cuda.synchronize()
                ok = bool(torch.equal(y, ref)) and B.same_on_all_ranks(B.sha(y), cpu_group)
                corr[k] = ok
            graphs = {k: graph_for(f, x, n_ops) for k, f in be.items()}
            per = {k: [] for k in be}
            order = list(be)
            for t in range(a.trials):
                for k in (order if t % 2 == 0 else list(reversed(order))):
                    per[k].append(time_graph(graphs[k], n_ops))
            row = dict(rows=rows, bytes=nb, n_ops=n_ops, correct=corr,
                       **{f"{k}_us": statistics.median(v) for k, v in per.items()},
                       **{f"{k}_all": v for k, v in per.items()})
            best = min([row[f"{k}_us"] for k in be if k != "pynccl"], default=None)
            row["best_rdma_us"] = best
            out.append(row)
            msg = f"[{tag}] rows={rows:5d} bytes={nb:9d} pynccl {row['pynccl_us']:8.1f} us"
            for k in ("rdma_1shot", "rdma_2shot"):
                if k in be:
                    msg += f" | {k} {row[k + '_us']:7.1f}{'' if corr[k] else ' (MISMATCH)'}"
            if best:
                msg += f" | speedup {row['pynccl_us'] / best:.2f}x, per-81-op step saving " \
                       f"{81 * (row['pynccl_us'] - best) / 1000:.2f} ms"
            log0(msg)
            del graphs
        return out

    # ---- X1/X2 decode-range ARs beyond the current 512 KiB limit ---------------------------
    rows_list = [int(r) for r in a.rows.split(",") if r]
    res["ar"] = sweep_ar(rows_list, "X2")
    fails += sum(1 for r in res["ar"] for v in r["correct"].values() if not v)

    # ---- X3 all-gathers ----------------------------------------------------------------------
    ag = []
    shapes = [(int(r), VOCAB_SHARD) for r in a.ag_logit_rows.split(",") if r] + \
             [(int(r), 3, 256) for r in a.ag_engram_rows.split(",") if r]
    for shape in shapes:
        x = B.rand_input(1, torch.bfloat16, rank, 12, device,
                         cols=int(torch.tensor(shape).prod())).view(shape)
        nb = x.numel() * 2
        row = dict(shape=shape, bytes_in=nb, routed=bool(rd.should_gather(x)))
        be = {"pynccl": lambda t: nccl.all_gather_stacked(t, world)}
        if row["routed"]:
            y = rd.all_gather_stacked(x)
            ref = nccl.all_gather_stacked(x, world)
            torch.cuda.synchronize()
            row["bitwise"] = bool(torch.equal(y, ref))
            fails += 0 if row["bitwise"] else 1
            be["rdma"] = lambda t: rd.all_gather_stacked(t)
        graphs = {k: graph_for(f, x, a.n_ag_ops) for k, f in be.items()}
        per = {k: [] for k in be}
        for t in range(a.trials):
            for k in (list(be) if t % 2 == 0 else list(reversed(list(be)))):
                per[k].append(time_graph(graphs[k], a.n_ag_ops))
        for k, v in per.items():
            row[f"{k}_us"] = statistics.median(v)
        ag.append(row)
        log0(f"[X3] AG shape={str(shape):16s} in={nb:8d} B  pynccl {row['pynccl_us']:7.1f} us"
             + (f" | rdma {row['rdma_us']:7.1f} us{'' if row['bitwise'] else ' (MISMATCH)'}"
                f"  speedup {row['pynccl_us'] / row['rdma_us']:.2f}x" if row["routed"]
                else "  (not routed: > RDMA_AR_MAX_BYTES or %16)"))
        del graphs
    res["ag"] = ag

    # ---- X4 prefill-medium ARs (optional) ---------------------------------------------------
    if a.big:
        big_rows = [int(r) for r in a.big_rows.split(",") if r]
        res["ar_big"] = sweep_ar(big_rows, "X4", pynccl_only_rows=(4096,))
        fails += sum(1 for r in res["ar_big"] for v in r["correct"].values() if not v)

    # ---- crossover summary --------------------------------------------------------------------
    allrows = res["ar"] + res.get("ar_big", [])
    win = [r for r in allrows if r.get("best_rdma_us") and r["best_rdma_us"] < r["pynccl_us"]]
    two_better = [r["bytes"] for r in allrows
                  if "rdma_2shot_us" in r and "rdma_1shot_us" in r and r["rdma_2shot_us"] < r["rdma_1shot_us"]]
    res["summary"] = dict(
        largest_ar_bytes_rdma_wins=max((r["bytes"] for r in win), default=0),
        smallest_bytes_2shot_beats_1shot=min(two_better, default=None),
    )
    res["rdma_stats"] = rd.stats(reset=True)
    res["failures"] = fails
    allf = [None] * world
    dist.all_gather_object(allf, fails, group=cpu_group)
    res["failures_all_ranks"] = allf
    import json
    with open(f"{a.out}.rank{rank}.json", "w") as f:
        json.dump(res, f, indent=1, default=str)
    log0(f"SUMMARY {res['summary']}")
    log0(f"DONE failures per rank: {allf}  -> {'PASS' if not any(allf) else 'FAIL'}")
    dist.barrier(group=cpu_group)
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
