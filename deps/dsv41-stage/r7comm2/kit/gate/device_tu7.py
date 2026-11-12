#!/usr/bin/env python3
"""device_tu7.py <rdma_ar.cu> <outdir>: torch-free device-only translation units of an rdma_ar.cu (live, r3, r6 or a
mutant), one per launch_kernel<mode, f32> instantiation (+ pingpong), for low-memory `nvcc -cubin` compiles while the
service runs (a full torch-extension build needs ~5 GB of the node's unified memory; one of these needs a few hundred MB).

The device code is copied VERBATIM from `constexpr int kSlots` through the closing brace of launch_kernel (after the
climb-r3 ref/ops/device_tu.py idea), so every kernel template, helper and launch switch is the source's own text; only
the instantiating call differs. Each TU instantiates exactly what the .so's launch_kernel<mode, f32> instantiates: all
ar_kernel_pl masks 0..7, the legacy ar_kernel and (r3 / r6 sources) every ar_kernel_r3 mask of the launch switch.

The SASS of these cubins is evidence for the gate only. ops/dev8-r6.sh first checks the method: the r3 source's TUs must
reproduce the SASS of every kernel of the r3 .so 7ec43164 (and the live kernels of 14d049dd) instruction for instruction.
The shipped .so is always a full build (ops/build8-r6.sh), never assembled from these."""
import os
import sys

src, outdir = sys.argv[1], sys.argv[2]
lines = open(src).read().split("\n")


def find(pred, start=0, what=""):
    for i in range(start, len(lines)):
        if pred(lines[i]):
            return i
    sys.exit(f"device_tu7.py: anchor not found: {what}")


a = find(lambda s: s.startswith("constexpr int kSlots"), what="kSlots")
k = find(lambda s: s.startswith("void launch_kernel("), a, "launch_kernel")
b = find(lambda s: s == "}", k, "end of launch_kernel")
head = """#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <stdexcept>
#include <string>
namespace {
"""
CALLS = {
    "pp": "  pingpong_kernel<<<1, 1, 0, s>>>(c, 1, 1, o);",
    "os_bf16": "  launch_kernel<kOneShot, false>(p, 512, true, s);",
    "os_f32": "  launch_kernel<kOneShot, true>(p, 512, true, s);",
    "ts_bf16": "  launch_kernel<kTwoShot, false>(p, 512, true, s);",
    "ts_f32": "  launch_kernel<kTwoShot, true>(p, 512, true, s);",
    "ag": "  launch_kernel<kGather, false>(p, 512, true, s);",
    "agc": "  launch_kernel<kGatherCols, false>(p, 512, true, s);",
}
os.makedirs(outdir, exist_ok=True)
body = "\n".join(lines[a:b + 1])
for name, call in CALLS.items():
    tail = ("\n}  // namespace\n"
            "void device_tu_inst(const Params& p, cudaStream_t s, Ctrl* c, unsigned long long* o) {\n"
            + call + "\n}\n")
    open(os.path.join(outdir, f"{name}.cu"), "w").write(head + body + tail)
print(f"device_tu7.py: lines {a + 1}-{b + 1} of {os.path.basename(src)} -> {len(CALLS)} TUs in {outdir}")
