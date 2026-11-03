#!/usr/bin/env python3
"""device_tu.py <rdma_ar.cu> <out.cu> [pub_mask ...]: a torch-free translation unit holding the device code of rdma_ar.cu
VERBATIM (from `constexpr int kSlots` through the end of launch_kernel), for a low-memory device-only compile (nvcc -cubin:
no torch headers, no host compile). It only exists so kernels can be disassembled while the service is up; it is NOT a
substitute for the recorded-flags rebuild, which needs torch headers. Fails loudly on a missing anchor.

With pub masks (g130 source only) it instantiates just those kernel variants, so a batch stays small; mask 0 also brings the
legacy kernels and pingpong_kernel, i.e. the same 13 kernels as the production module."""
import sys

src, out = sys.argv[1], sys.argv[2]
lines = open(src).read().split("\n")


def find(pred, start=0, what=""):
    for i in range(start, len(lines)):
        if pred(lines[i]):
            return i
    sys.exit(f"device_tu.py: anchor not found: {what}")


a = find(lambda s: s.startswith("constexpr int kSlots"), what="kSlots")
k = find(lambda s: s.startswith("void launch_kernel("), a, "launch_kernel")
b = find(lambda s: s == "}", k, "end of launch_kernel")
head = """#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <stdexcept>
#include <string>
namespace {
"""
g130 = any("kPubV1" in s for s in lines[a:b + 1])
masks = [int(x) for x in sys.argv[3:]]
MODES = [("kOneShot", "false"), ("kOneShot", "true"), ("kTwoShot", "false"), ("kTwoShot", "true"), ("kGather", "false"),
         ("kGatherCols", "false")]
calls = []
if masks and g130:
    for m in masks:
        if m == 0:
            calls.append("  pingpong_kernel<<<1, 1, 0, s>>>(c, 1, 1, o);")
            calls += [f"  ar_kernel<{x}, {y}><<<1, kThreads, 0, s>>>(p);" for x, y in MODES]
        for x, y in MODES:
            if (m & 2) and x != "kTwoShot":
                continue  # launch_pl drops V2 outside two-shot: that kernel is the same as mask & ~2
            calls.append(f"  launch_pl<{x}, {y}, {m}>(p, 1, s);")
else:
    calls.append("  pingpong_kernel<<<1, 1, 0, s>>>(c, 1, 1, o);")
    calls += [f"  launch_kernel<{x}, {y}>(p, 512, true, s);" for x, y in MODES]
# DEV_SLICE=i/n keeps every n-th call, so one compile holds few kernels (cicc memory scales with them)
sl = __import__("os").environ.get("DEV_SLICE", "")
if sl:
    si, sn = (int(x) for x in sl.split("/"))
    calls = calls[si::sn]
tail = ("\n}  // namespace\n"
        "void device_tu_inst(const Params& p, cudaStream_t s, Ctrl* c, unsigned long long* o) {\n"
        + "\n".join(calls) + "\n}\n")
open(out, "w").write(head + "\n".join(lines[a:b + 1]) + tail)
print(f"device_tu.py: lines {a + 1}-{b + 1} -> {out.split('/')[-1]} (g130 flags in source: {g130}, masks {masks or 'all'})")
