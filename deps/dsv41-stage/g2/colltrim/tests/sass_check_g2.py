#!/usr/bin/env python3
"""sass_check_g2.py (g2 colltrim): CPU-only SASS gate for the column-gather rdma_ar_ext.so.

  python3 tests/sass_check_g2.py ~/rdma_ar_dsv41/build-k1pl/rdma_ar_ext.so build/rdma_ar_ext.so

Derived from k1's sass_check.py (same parser). Checks:
  1. EVERY kernel of the live k1 build (legacy ar_kernel<mode,f32>, parallel-load ar_kernel_pl<mode,f32,8>
     and pingpong_kernel) is instruction-identical in the g2 build (addresses / printf constant-bank
     offsets normalised): every all-reduce and dim-0 all-gather runs the same code as today.
  2. the new column-gather instantiations exist (mode 3: ar_kernelILi3ELb0EE, ar_kernel_plILi3ELb0ELi8EE);
     the parallel-load one keeps all 7 peer loads in flight before its stores (sysload depth >= 7, as k1
     gated its gather), and both have no local-memory spill and REG <= 128 (512-thread CTAs).
Exit 0 = PASS, 1 = FAIL. One line per kernel.
"""
import re
import subprocess
import sys

CUOBJDUMP = "/usr/local/cuda/bin/cuobjdump"
INS = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;")
REG = re.compile(r"(?<![A-Za-z0-9_])R(\d+)(?:\.(64|reuse))?(?:\.reuse)?")


def kernel_key(name):
    m = re.search(r"(ar_kernel_pl|ar_kernel|pingpong_kernel)(I\w+?EE)?", name)
    return m.group(0) if m else name


def functions(sass_text):
    out, cur, name = {}, None, None
    for line in sass_text.splitlines():
        if "Function :" in line:
            name = kernel_key(line.split("Function :", 1)[1].strip())
            cur = out.setdefault(name, [])
            continue
        if cur is None:
            continue
        m = INS.search(line)
        if m:
            txt = re.sub(r"\s+", " ", m.group(2)).strip()
            cur.append(txt)
    return out


def normalise(ins):
    s = re.sub(r"c\[0x4\]\[0x[0-9a-f]+\]", "c[0x4][X]", ins)  # printf format-string constants move
    s = re.sub(r"\(\*\"[^\"]*\"\*\)", "", s)  # relocation annotations
    return s


def regs_of(op):
    regs = []
    for m in REG.finditer(op):
        n = int(m.group(1))
        regs.append(n)
        if m.group(2) == "64":
            regs.append(n + 1)
    return regs


def inflight_depth(body):
    """max / per-load outstanding-STRONG.SYS-load depth at first use, over the straight-line order."""
    pending = {}  # reg -> group id
    groups = {}  # group id -> set(regs)
    depths = []
    gid = 0
    for ins in body:
        t = ins.split(" ", 1)
        if t[0].startswith("@"):
            t = t[1].split(" ", 1) if len(t) > 1 else [""]
        opc = t[0]
        ops = [o.strip() for o in t[1].split(",")] if len(t) > 1 else []
        if opc.startswith("LDG.E.128") and ops:
            dest = regs_of(ops[0])
            if not dest:
                continue
            base = dest[0]
            if "STRONG.SYS" in opc:
                # a predicated pair (@P LDG.STRONG / @!P LDG) writes the same group: keep one id
                gid += 1
                g = pending.get(base, gid)
                for r in range(base, base + 4):
                    pending[r] = g
                groups[g] = set(range(base, base + 4))
            continue
        if opc.startswith(("STG", "ST.", "RED", "ATOM")):
            srcs = [r for o in ops for r in regs_of(o)]
        else:
            srcs = [r for o in ops[1:] for r in regs_of(o)]
        hit = {pending[r] for r in srcs if r in pending}
        if hit:
            outstanding = len(set(pending.values()))
            for g in hit:
                depths.append(outstanding)
                for r in groups.pop(g, ()):
                    pending.pop(r, None)
        if opc.startswith(("BAR", "EXIT", "RET")):
            pending.clear()
            groups.clear()
    return (max(depths) if depths else 0), depths


def res_usage(so):
    txt = subprocess.run([CUOBJDUMP, "-res-usage", so], capture_output=True, text=True, check=True).stdout
    out, name = {}, None
    for line in txt.splitlines():
        if line.strip().startswith("Function"):
            name = kernel_key(line.split("Function", 1)[1].strip().rstrip(":"))
        elif name and "REG:" in line:
            d = dict(kv.split(":", 1) for kv in line.split() if ":" in kv)
            out[name] = (int(d.get("REG", 0)), int(d.get("LOCAL", 0)), int(d.get("STACK", 0)))
            name = None
    return out


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    old_so, new_so = sys.argv[1], sys.argv[2]
    old = functions(subprocess.run([CUOBJDUMP, "-sass", old_so], capture_output=True, text=True, check=True).stdout)
    new = functions(subprocess.run([CUOBJDUMP, "-sass", new_so], capture_output=True, text=True, check=True).stdout)
    ru = res_usage(new_so)
    ok = True
    n_same = 0
    for k in sorted(old):
        a = [normalise(i) for i in old[k]]
        b = [normalise(i) for i in new.get(k, [])]
        same = a == b and len(a) > 0
        ndiff = sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))
        print(f"existing {k:30s} live==g2 {'YES' if same else 'NO '} ({len(a)} vs {len(b)} ins, {ndiff} differ)")
        ok &= same
        n_same += same
    new_only = sorted(k for k in new if k not in old)
    want = {"ar_kernelILi3ELb0EE", "ar_kernel_plILi3ELb0ELi8EE"}
    missing = want - set(new_only)
    if missing:
        print(f"MISSING new kernels {sorted(missing)}; new-only = {new_only}")
        ok = False
    for k in new_only:
        reg, local, stack = ru.get(k, (-1, -1, -1))
        d, ds = inflight_depth(new[k])
        hist = {}
        for x in ds:
            hist[x] = hist.get(x, 0) + 1
        good = local == 0 and 0 < reg <= 128 and (d >= 7 if k.startswith("ar_kernel_pl") else True)
        print(f"new      {k:30s} {len(new[k])} ins  sysload depth max {d} hist {dict(sorted(hist.items()))}  "
              f"REG {reg} LOCAL {local}  -> {'ok' if good else 'FAIL'}")
        ok &= good
    print(f"SASS_CHECK_G2 {'PASS' if ok else 'FAIL'}: {n_same}/{len(old)} live kernels identical, "
          f"{len(new_only)} new kernel(s)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
