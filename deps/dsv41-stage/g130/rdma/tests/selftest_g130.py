#!/usr/bin/env python3
"""selftest_g130.py: single-GPU, no-network bitwise test of every kernel variant. GPU job: STOPPED WINDOW ONLY, one process per
Spark (ops/run-selftest8.sh). NOT RUN YET: written offline, only syntax-checked. Uses the .so's SelfTest class: flags and posted_seq
words are preset in the future, so no kernel waits. It proves the data path of each kPub kernel, NOT the protocol (no proxy here;
that is bench_g130.py phases X and S). Reference: legacy kernels (par=False) and mask 0, both production code.
  ~/venvs/vllm032/bin/python -I selftest_g130.py --ext build/rdma_ar_ext.so
Exit 0 and a last line "SELFTEST PASS" only if every comparison is bitwise equal."""
import argparse
import importlib.util

import torch

W = 8
MAX_BYTES = 2 << 20
HIDDEN = 5120


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext", required=True)
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("rdma_ar_ext", a.ext)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    dev = torch.device("cuda", 0)
    g = torch.Generator().manual_seed(20261102)
    fails = checks = 0

    def wide(shape, dtype):
        scale = torch.exp2(torch.randint(-14, 14, shape, generator=g).float())
        return (torch.randn(shape, generator=g) * scale).to(dtype)

    def bytes_of(t):
        return t.contiguous().view(torch.uint8)

    for rank in (0, 3, 7):
        t = m.SelfTest(rank, W, MAX_BYTES, MAX_BYTES, 0)
        for dtype in (torch.bfloat16, torch.float32):
            es = 2 if dtype == torch.bfloat16 else 4
            cols = HIDDEN if dtype == torch.bfloat16 else HIDDEN // 2
            for M in (1, 2, 3, 4, 6, 12, 48):
                x = wide((M, cols), dtype).to(dev)
                nb = x.numel() * es
                img = torch.stack([bytes_of(wide((nb // es,), dtype)) for _ in range(W)]).to(torch.uint8)
                t.set_recv(img)
                img2 = torch.randint(0, 256, (W, t.max_chunk()), generator=g, dtype=torch.uint8)
                t.set_recv2(img2)
                for two in (False, True):
                    outs = {}
                    ref = torch.empty_like(x)
                    t.set_pub(0)
                    t.all_reduce_into(ref, x, two, False)
                    torch.cuda.synchronize()
                    for mask in range(8):
                        out = torch.empty_like(x)
                        t.set_pub(mask)
                        t.all_reduce_into(out, x, two, True)
                        torch.cuda.synchronize()
                        checks += 1
                        ok = torch.equal(bytes_of(out), bytes_of(ref))
                        fails += 0 if ok else 1
                        outs[mask] = ok
                    if not all(outs.values()):
                        print(f"FAIL rank {rank} {dtype} M={M} two={two}: {outs}", flush=True)
        for M in (1, 6, 48):
            x = wide((M, 224), torch.bfloat16).to(dev)
            nb = x.numel() * 2
            t.set_recv(torch.stack([bytes_of(wide((nb // 2,), torch.bfloat16)) for _ in range(W)]).to(torch.uint8))
            ref = torch.empty(W, M, 224, dtype=torch.bfloat16, device=dev)
            t.set_pub(0)
            t.all_gather_into(ref, x, False)
            refc = torch.empty(M, W * 224, dtype=torch.bfloat16, device=dev)
            t.all_gather_cols_into(refc, x, 448, False)
            torch.cuda.synchronize()
            for mask in range(8):
                o1 = torch.empty_like(ref)
                o2 = torch.empty_like(refc)
                t.set_pub(mask)
                t.all_gather_into(o1, x, True)
                t.all_gather_cols_into(o2, x, 448, True)
                torch.cuda.synchronize()
                checks += 2
                a1, a2 = torch.equal(bytes_of(o1), bytes_of(ref)), torch.equal(bytes_of(o2), bytes_of(refc))
                fails += (0 if a1 else 1) + (0 if a2 else 1)
                if not (a1 and a2):
                    print(f"FAIL gather rank {rank} M={M} mask {mask}: {a1} {a2}", flush=True)
        del t
    print(f"{checks} checks, {fails} failures", flush=True)
    print("SELFTEST PASS" if not fails else "SELFTEST FAIL", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
