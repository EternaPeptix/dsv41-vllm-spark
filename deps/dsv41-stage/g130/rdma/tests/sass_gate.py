#!/usr/bin/env python3
"""sass_gate.py <ref.so|cubin> <new.so|cubin> [--expect-new N] [--show-diff KERNEL]: CPU-only SASS compare.

Derived from g2's tests/sass_check_g2.py (same parser and normalisation). For every kernel of <ref>: the kernel of the
same key in <new> must be instruction-identical. Keys are the mangled template arguments, with the g130 kPub == 0 argument
removed (ar_kernel_plILi3ELb0ELi8ELi0EE == ar_kernel_plILi3ELb0ELi8EE), because that is the only name change.
Normalised away, and counted separately so nothing is hidden: instruction addresses, printf format-string constant-bank
offsets (c[0x4][...]), relocation annotations. Prints RAW-identical as well. Kernels only in <new> are listed, with
their instruction delta against the matching kPub == 0 kernel. Exit 0 = every ref kernel identical."""
import argparse
import re
import subprocess

CUOBJDUMP = "/usr/local/cuda/bin/cuobjdump"
INS = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;")


def kernel_key(name):
    m = re.search(r"(ar_kernel_pl|ar_kernel|pingpong_kernel)(I\w+?EE)?", name)
    k = m.group(0) if m else name
    return re.sub(r"ELi0EE$", "EE", k)


def functions(path):
    txt = subprocess.run([CUOBJDUMP, "-sass", path], capture_output=True, text=True, check=True).stdout
    out, cur = {}, None
    for line in txt.splitlines():
        if "Function :" in line:
            cur = out.setdefault(kernel_key(line.split("Function :", 1)[1].strip()), [])
            continue
        if cur is None:
            continue
        m = INS.search(line)
        if m:
            cur.append(re.sub(r"\s+", " ", m.group(2)).strip())
    return out


def normalise(ins):
    s = re.sub(r"c\[0x4\]\[0x[0-9a-f]+\]", "c[0x4][X]", ins)
    return re.sub(r"\(\*\"[^\"]*\"\*\)", "", s)


def counts(body):
    sys_loads = sum(1 for i in body if re.search(r"\bLDG\b.*STRONG\.SYS|LDG\.[^ ]*STRONG\.SYS", i))
    clock = sum(1 for i in body if "SR_GLOBAL" in i or "GLOBALTIMER" in i.upper())
    return sys_loads, clock


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("new")
    ap.add_argument("--expect-new", type=int, default=None)
    ap.add_argument("--show-diff", default=None)
    a = ap.parse_args()
    ref, new = functions(a.ref), functions(a.new)
    ok, nsame, nraw = True, 0, 0
    for k in sorted(ref):
        x, y = [normalise(i) for i in ref[k]], [normalise(i) for i in new.get(k, [])]
        same = x == y and len(x) > 0
        raw = ref[k] == new.get(k, [])
        nd = sum(1 for p, q in zip(x, y) if p != q) + abs(len(x) - len(y))
        print(f"ref {k:36s} identical {'YES' if same else 'NO '} raw {'YES' if raw else 'no '} ({len(x)} vs {len(y)} ins, {nd} differ)")
        ok &= same
        nsame += same
        nraw += raw
    only = sorted(k for k in new if k not in ref)
    for k in only:
        base = re.sub(r"ELi\d+EE$", "EE", k)
        d = ""
        if base in new:
            s0, c0 = counts(new[base])
            s1, c1 = counts(new[k])
            d = (f" vs {base}: ins {len(new[k]) - len(new[base]):+d}, strong.sys loads {s1 - s0:+d}, clock reads {c1 - c0:+d}")
        print(f"new {k:36s} {len(new[k])} ins{d}")
    if a.show_diff and a.show_diff in new:
        base = re.sub(r"ELi\d+EE$", "EE", a.show_diff)
        import difflib
        for line in difflib.unified_diff([normalise(i) for i in new.get(base, [])], [normalise(i) for i in new[a.show_diff]],
                                         base, a.show_diff, lineterm="", n=2):
            print(line)
    if a.expect_new is not None and len(only) != a.expect_new:
        print(f"expected {a.expect_new} new kernels, found {len(only)}")
        ok = False
    print(f"SASS_GATE {'PASS' if ok else 'FAIL'}: {nsame}/{len(ref)} ref kernels identical (raw-identical {nraw}), {len(only)} new-only kernel(s)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
