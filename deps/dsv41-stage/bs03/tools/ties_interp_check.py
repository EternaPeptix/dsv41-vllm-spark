#!/usr/bin/env python3
"""ties_interp_check.py <kit> <outdir>: the live TIES code under the Triton interpreter against the test's integer reference.

CPU only, nothing opens the GPU. For each input kind of test_det_topk.py: take a valid top-512 with a random subset of any
boundary tie (what the stock kernels may return), run dsv41_topk_ties.run_ties as production calls it (split count path, width
> 16384, and single pass), sort as CANON does, and compare with f_ref_row. It checks the test's reference and vectors against
the real tie rule before any GPU window. It says nothing about GPU float compares or about det_topk.
"""
import os
import sys

kit, outdir = sys.argv[1], sys.argv[2]
os.environ["TRITON_INTERPRET"] = "1"
test_path = os.path.join(kit, "test_det_topk.py")
src = open(test_path).read()
tail = "\nmain()\n"
if not src.endswith(tail):
    sys.exit("test_det_topk.py does not end in main()")
sys.argv = [test_path, "--mock", "--out", outdir]
ns = {"__name__": "bs03_test", "__file__": test_path}
exec(compile(src[: -len(tail)], test_path, "exec"), ns)  # definitions only; --mock hides the GPU before torch is imported

torch = ns["torch"]
f_ref_row, K = ns["f_ref_row"], ns["K"]

from vllm.model_executor.layers import dsv41_topk_ties as ties  # noqa: E402

assert ties.TIES_DECODE
ties._supported = lambda *a: None  # its only objection on CPU is the device check
be = ns["MockBackend"]()
vec = ns["Vectors"](be)
I32_MAX = 2147483647


def stock_like(x, length):
    """A valid top-512 of each row with a random subset of a boundary tie: topk over a random column order."""
    rows = x.shape[0]
    out = torch.full((rows, K), -1, dtype=torch.int32)
    for r in range(rows):
        if length <= K:
            out[r, :length] = torch.randperm(length, dtype=torch.int32)
            continue
        perm = torch.randperm(length)
        sel = x[r, perm].topk(K).indices
        out[r] = perm[sel].to(torch.int32)
    return out


def canon(t):
    key = torch.where(t < 0, I32_MAX, t)
    s = torch.sort(key, dim=-1).values
    return s.masked_fill(s == I32_MAX, -1)


bad = checked = tied = 0
rows = 4
for width in (65536, 12288):  # above and below SPLIT_MIN_WIDTH: the split count path and the single pass
    for length in (300, 512, 513, 1163, 4096, 12288, 16385, 65536):
        if length > width:
            continue
        for kind in ns["KINDS"]:
            for rep in range(2):
                x = vec.draw(kind, rows, length)
                logits = torch.full((rows, width), float("inf"))
                logits[:, :length] = x
                ends = torch.full((rows,), length, dtype=torch.int32)
                topk = stock_like(x, length)
                raw = topk.clone()
                ties.run_ties(logits, None, ends, topk, False, True, False)
                got = canon(topk)
                changed = int((got != canon(raw)).any(dim=1).sum())
                tied += changed
                for r in range(rows):
                    checked += 1
                    ref = f_ref_row(x[r], length)
                    if not torch.equal(got[r], ref):
                        bad += 1
                        if bad <= 8:
                            print(f"DIFF width={width} L={length} kind={kind} row={r}: {int((got[r] != ref).sum())} slots", flush=True)
            print(f"width={width} L={length} {kind}: rows so far {checked}, rows the tie rule changed {tied}, differing {bad}", flush=True)
print(f"TIES_INTERP {'OK' if bad == 0 else 'DIFF'}: {checked} rows, {tied} rows the tie rule changed, {bad} differ from the reference", flush=True)
sys.exit(0 if bad == 0 else 1)
