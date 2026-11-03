#!/usr/bin/env python3
"""extract_g130.py <rdma_ar.cu> <dev.inc> <host.inc>: the code the CPU emulator runs, taken VERBATIM from the source.
  dev.inc : `constexpr int kSlots` through the end of ar_kernel_pl (constants, structs, helpers, kernels), minus the three
            inline-PTX helpers that emu_prims.h replaces; `asm volatile("trap;");` becomes EMU_TRAP_TIMEOUT();
  host.inc: struct Hca + struct Peer (marked by HOSTSTRUCTS_END), then RdmaAR::post and RdmaAR::proxy_loop.
Fails loudly when an anchor is missing, so a changed source cannot silently test something else."""
import re
import sys

src, dev, host = sys.argv[1:4]
L = open(src).read().split("\n")


def find(pred, start=0, what=""):
    for i in range(start, len(L)):
        if pred(L[i]):
            return i
    sys.exit(f"extract: anchor not found: {what}")


a = find(lambda s: s.startswith("constexpr int kSlots"), what="kSlots")
k = find(lambda s: "ar_kernel_pl(Params p) {" in s and "__global__" in s, a, "ar_kernel_pl")
b = find(lambda s: s == "}", k, "end of ar_kernel_pl")
body = L[a:b + 1]
drop = []
for name in ("ld_acquire_sys(", "st_release_sys(", "now_ns("):
    i = next((j for j, s in enumerate(body) if s.startswith("__device__ __forceinline__") and name in s), None)
    if i is None:
        sys.exit(f"extract: helper not found: {name}")
    drop.append((i, next(j for j in range(i, len(body)) if body[j] == "}")))
text = "\n".join(s for n, s in enumerate(body) if not any(i <= n <= j for i, j in drop))
if text.count('asm volatile("trap;");') < 1:
    sys.exit("extract: no trap found")
text = text.replace('asm volatile("trap;");', "EMU_TRAP_TIMEOUT();")
if re.search(r"\basm\b", text):
    sys.exit("extract: inline asm left in the device region")
open(dev, "w").write(f"// GENERATED from {src.split('/')[-1]} lines {a + 1}-{b + 1}\n{text}\n")
h0 = find(lambda s: s == "struct Hca {", b, "struct Hca")
h1 = find(lambda s: s.startswith("class RdmaAR"), h0, "class RdmaAR")
p0 = find(lambda s: s.startswith("  void post(Peer& pr,"), h1, "post")
p1 = find(lambda s: s == "  int rank_, world_;", p0, "end of proxy_loop")
open(host, "w").write("\n".join(L[h0:h1]) + "\n// HOSTSTRUCTS_END\n" + "\n".join(L[p0:p1]) + "\n")
print(f"extract: dev {a + 1}-{b + 1}, host structs {h0 + 1}-{h1}, post+proxy_loop {p0 + 1}-{p1}")
