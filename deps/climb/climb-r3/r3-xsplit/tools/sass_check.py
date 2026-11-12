#!/usr/bin/env python3
"""Static SASS checks for the XSPLIT build. CPU only (reads cuobjdump -sass text).

  sass_check.py <xsplit.sass> <bench_a0r1.sass> <live_dual64.full> [<live_dual32.full>]

1. C0 for the bench reference: the live dual kernel instantiated in the bench library (same live
   headers, same flags) has the same instruction sequence (mnemonic + modifiers) as the live JIT
   kernel dumped from ~/.cache/flashinfer/.../sparse_mla_sm120.so (base/sass/dual64.full, dual32).
   This is what makes the microbenchmark's "live" arm the production kernel.
2. FTZ: every FP32 FADD/FMUL/FFMA/FMNMX/FSETP in the XSPLIT kernels carries .FTZ, like the live
   kernel (-use_fast_math); the __fmul_rn/__fadd_rn/__fmaf_rn pins compiled to FTZ forms.
3. Histogram of FP/MMA/MUFU/atomic/barrier opcodes, XSPLIT vs live, per variant, with the
   expected structural deltas: A + B contain every live per-tile op; B adds the replay loads.
4. Global loads in XSPLIT: no LDG with .CONSTANT (ld.global.nc) on the workspace path: the only
   .CONSTANT loads allowed are the read-only inputs (indices, lengths, sink, KV via __ldg).
Exit 0 when 1 and 2 hold (3 and 4 are printed for the reader).
"""
import collections
import re
import sys

LINE = re.compile(r"^\s*/\*[0-9a-f]+\*/\s+(?:@!?U?P[T0-9]\s+)?([A-Z0-9_][A-Z0-9_.a-z]*)")
FP32 = re.compile(r"^(FADD|FMUL|FFMA|FMNMX|FSETP|FSET)(\.|$)")
HIST = re.compile(r"^(FADD|FMUL|FFMA|FMNMX|FSETP|FSEL|MUFU|HMMA|QMMA|F2FP|ATOMS|ATOMG|ATOM|RED|"
                  r"BAR|SHFL|LDG|STG|LDS|STS|LDSM|SYNCS|MEMBAR|FENCE|CCTL|ERRBAR|NANOSLEEP|BPT)")


def parse(text):
    out, cur = {}, None
    for raw in text.splitlines():
        m = re.match(r"^\s*Function\s*:\s*(\S+)", raw)
        if m:
            cur = m.group(1)
            out[cur] = []
            continue
        if cur is None:
            continue
        m = LINE.match(raw)
        if m and not m.group(1).startswith("NOP"):
            ins = raw.split("/*", 2)[1].split("*/", 1)[1].split(";")[0].strip()
            out[cur].append((m.group(1), ins))
    return out


def parse_plain(text):
    """base/sass/*.full: one instruction per line, no addresses."""
    ops = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s or s.startswith("NOP"):
            continue
        bare = re.sub(r"^@!?U?P[T0-9]\s+", "", s)
        ops.append((bare.split()[0], s))
    return ops


def pick(funcs, pat):
    ks = [k for k in funcs if re.search(pat, k)]
    return ks[0] if ks else None


def hist(ops):
    h = collections.Counter()
    for mn, _ in ops:
        m = HIST.match(mn)
        if m:
            h[mn] += 1
    return h


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    xs = parse(open(sys.argv[1]).read())
    bench = parse(open(sys.argv[2]).read())
    lives = [parse_plain(open(p).read()) for p in sys.argv[3:]]
    ok = True
    # 1. C0 of the bench reference kernel
    for pbsx, live in zip((64, 32), lives):
        k = pick(bench, rf"sparse_mla_prefill_mg_dual_kernelIL9ModelType1EL11ComputeMode1ELi8ELi64ELi{pbsx}ELi1E")
        if k is None:
            print(f"C0 dual{pbsx}: bench kernel not found")
            ok = False
            continue
        # base/sass/*.full was written by the g130-attn census parser, whose line regex drops
        # instructions under a uniform predicate (@UPn / @!UPn); apply the same filter here.
        norm = lambda t: re.sub(r"\s+", " ", t.strip())  # noqa: E731
        a = [norm(ins) for _, ins in bench[k] if not re.match(r"^@!?UP", ins.strip())]
        b = [norm(ins) for _, ins in live]
        same = a == b
        print(f"C0 dual{pbsx}: bench-lib live kernel {len(a)} instr vs JIT live {len(b)} instr: "
              f"{'IDENTICAL with operands' if same else 'DIFFERENT'} (uniform-predicated "
              f"instructions excluded on both sides)")
        if not same:
            ok = False
            for i, (x, y) in enumerate(zip(a, b)):
                if x != y:
                    print(f"   first diff at {i}: {x} vs {y}")
                    break
    # 2. FTZ everywhere in XSPLIT
    for pbsx in (64, 32):
        k = pick(xs, rf"dsv41_xsplit_mg_dual_kernelILi{pbsx}E")
        if k is None:
            print(f"xsplit kernel PBSX={pbsx} not found")
            ok = False
            continue
        ops = xs[k]
        non_ftz = [ins for mn, ins in ops if FP32.match(mn) and ".FTZ" not in mn]
        print(f"FTZ xsplit{pbsx}: {len(ops)} instr, FP32 ops without .FTZ: {len(non_ftz)}")
        for ins in non_ftz[:8]:
            print(f"   {ins}")
        if non_ftz:
            ok = False
        # 3. histograms
        live = lives[0 if pbsx == 64 else min(1, len(lives) - 1)]
        hx, hl = hist(ops), hist(live)
        print(f"   opcode histogram xsplit{pbsx} vs live dual{pbsx} (count xsplit / live):")
        for mn in sorted(set(hx) | set(hl)):
            if hx[mn] != hl[mn]:
                print(f"     {mn:40s} {hx[mn]:5d} / {hl[mn]:5d}")
        # 4. global loads
        ldg = collections.Counter(mn for mn, _ in ops if mn.startswith("LDG"))
        print(f"   LDG flavours: {dict(ldg)}")
    print("SASS CHECK", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
