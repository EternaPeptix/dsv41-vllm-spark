#!/usr/bin/env python3
"""shapes.py: the shapes production routes through rdma_ar, derived from the BYTE BOUNDS of launcher 166d7aac, and the
code-path signature of each shape in ar_kernel_r3 (review B1).

Production bounds (launcher 166d7aac, read-only grep on spark-one 2026-11-09; rdma_all_reduce.py 26a5d834). r8-comm2fix:
production is now 4f3cc1c6, whose only difference from 166d7aac is DSV41_GLUE_RQPAD / DSV41_GLUE_RPSPLIT (diff of
climb-r8/integrate/live/dsv41-tp8.sh against climb-r6/integrate/live/dsv41-tp8.sh), so every bound below still holds:
  RDMA_AR_MAX_REDUCE_BYTES = 2097152   set by the launcher ("RDMA two-shot all-reduce up to 2 MiB ... 204 rows")
                                       -> should_use: bf16/fp32, 0 < bytes <= 2 MiB, bytes % 16 == 0
  RDMA_AR_MAX_BYTES        = 2097152   unset -> the 2 MiB default; should_gather: 0 < bytes <= 2 MiB, bytes % 16 == 0
  RDMA_AR_TWOSHOT_MIN_BYTES = 32768    unset -> two-shot iff bytes >= 32 KiB
  RDMA_AR_VEC_PER_CTA       = 512      unset -> the .so default (device_test.py preflight asserts it)
  DSV41_RDMA_TRIM_S1_BYTES  = 24576    r3 default (the arm never sets it)
So bf16 [M, 5120] all-reduces are served up to M = 204, fp32 up to M = 102, and gathers up to 2 MiB of input. Prefill
chunks of <= 204 tokens and conc17-32 decode reach the large shapes; decode at conc <= 16 stays at M <= 24.

Kernel geometry (src/rdma_ar.cu launch_kernel / ar_kernel_r3): blocks = clamp(ceil(nvec / 512), 1, 16), 512 threads, so
stride = 512 * blocks <= 8192 vectors. The data-dependent paths of ar_kernel_r3 are:
  staging   S1 (nvec * 16 <= s1: CTA 0 alone, 4-lane loop over 512-thread rows; lanes used = ceil(nvec / 512))
            or the grid-wide stage: the 4x-unrolled loop runs for thread 0 iff nvec > 3 * stride, twice iff
            nvec > 7 * stride; then the remainder loop
  consumer  one-shot / gathers: one pass iff nvec <= stride, else a multi-iteration loop
            two-shot: P2 (single-CTA phase 2) iff chunk <= 512; consumer and phase-2 loops multi-iteration iff
            chunk > stride (bf16 M >= 103, fp32 M >= 52)
The signature of a shape is the tuple of those choices. Every list used by device_test.py (c1, mutants, c2, cpu
selfcheck) and bench_r6.py (phase X) is checked here to contain every signature that ANY served shape has
(`python3 shapes.py` prints the table; tests/run_tests.py T18 runs it).
"""
import sys

HIDDEN = 5120
QKV = 224
W = 8
MAX_REDUCE_BYTES = 2097152
MAX_GATHER_BYTES = 2097152
TWOSHOT_MIN_BYTES = 32768
S1_BYTES = 24576
KTHREADS = 512
MAX_CTAS = 16
VPC = 512
SEGS = (448, 16, 32, 1280)  # column-gather segment bytes the device test uses (448 = the served QKV 224 x bf16)
SEGS_C2 = (448,)            # C2 poison / m_waitlate gathers: the served segment only (its cover reaches every signature)
# the review's explicit list (B1), kept verbatim; the boundary rows below are added to it
REVIEW_EXTRA = (25, 32, 38, 39, 40, 48, 51, 52, 64, 96, 102, 103, 128, 160, 204)


def geom(nvec, vpc=VPC):
    blocks = max(1, min(MAX_CTAS, -(-nvec // vpc)))
    return blocks, blocks * KTHREADS


def stage_sig(nvec, s1=S1_BYTES, vpc=VPC):
    _, stride = geom(nvec, vpc)
    if nvec * 16 <= s1:
        return ("S1", min(4, -(-nvec // KTHREADS)))
    return ("grid", 0 if nvec <= 3 * stride else (1 if nvec <= 7 * stride else 2))


def ar_sig(M, f32, two, s1=S1_BYTES, vpc=VPC):
    es = 4 if f32 else 2
    nvec = M * HIDDEN * es // 16
    blocks, stride = geom(nvec, vpc)
    st = stage_sig(nvec, s1, vpc)
    if two:
        chunk = -(-nvec // W)
        cons = ("two", "P2" if chunk <= KTHREADS else "grid", "multi" if chunk > stride else "single")
    else:
        cons = ("one", "multi" if nvec > stride else "single")
    return ("f32" if f32 else "bf16", st, "16cta" if blocks == MAX_CTAS else "<16cta") + cons


def gather_sig(nbytes, kind, s1=S1_BYTES, vpc=VPC):
    nvec = nbytes // 16
    blocks, stride = geom(nvec, vpc)
    return (kind, stage_sig(nvec, s1, vpc), "16cta" if blocks == MAX_CTAS else "<16cta",
            "multi" if nvec > stride else "single")


def max_rows(row_bytes, bound):
    return bound // row_bytes


def served_two(nbytes):
    return nbytes >= TWOSHOT_MIN_BYTES


def boundary_rows(row_bytes, bound, vpc=VPC):
    """rows on both sides of every nvec threshold of the kernel (stride, unroll 1 and 2, P2 / two-shot multi-iteration,
    S1, two-shot routing), clipped to the served bound"""
    mx = max_rows(row_bytes, bound)
    full = MAX_CTAS * KTHREADS
    thr_bytes = [S1_BYTES, TWOSHOT_MIN_BYTES]
    thr_nvec = [KTHREADS * k for k in (1, 2, 3)] + [vpc * MAX_CTAS, full, 3 * full, 7 * full, KTHREADS * W, full * W]
    out = set()
    for tb in thr_bytes + [t * 16 for t in thr_nvec]:
        m = tb // row_bytes
        out |= {m, m + 1}
    return {m for m in out if 1 <= m <= mx}


def ar_ms(f32):
    """bf16: 1..24 + the review list + every boundary row, up to 204; fp32: the same, capped at 102"""
    rb = HIDDEN * (4 if f32 else 2)
    mx = max_rows(rb, MAX_REDUCE_BYTES)
    return sorted({m for m in set(range(1, 25)) | set(REVIEW_EXTRA) | boundary_rows(rb, MAX_REDUCE_BYTES) | {mx}
                   if 1 <= m <= mx})


def gather_ms(segb):
    """rows of a [M, segb / 2] bf16 gather input: 1..24, both sides of every boundary, 1 MiB and the 2 MiB maximum"""
    mx = max_rows(segb, MAX_GATHER_BYTES)
    return sorted({m for m in set(range(1, 25)) | boundary_rows(segb, MAX_GATHER_BYTES) | {(1 << 20) // segb, mx}
                   if 1 <= m <= mx})


def cover(ms, sig):
    """the smallest and the largest member of each signature class of ms"""
    by = {}
    for m in ms:
        by.setdefault(sig(m), []).append(m)
    return sorted({min(v) for v in by.values()} | {max(v) for v in by.values()})


def c2_ar_ms(f32):
    """C2 poison / m_waitlate list: bf16 1..24 + {39, 103, 204} (review) + a cover of every served signature; fp32 a
    cover (review N5)"""
    rb = HIDDEN * (4 if f32 else 2)
    allm = range(1, max_rows(rb, MAX_REDUCE_BYTES) + 1)
    cv = set(cover(allm, lambda m: ar_sig(m, f32, served_two(m * rb))) +
             cover(allm, lambda m: ar_sig(m, f32, not served_two(m * rb))))
    base = set(range(1, 25)) | {39, 103, 204} if not f32 else set()
    return sorted(cv | base)


def c2_gather_ms(segb):
    allm = range(1, max_rows(segb, MAX_GATHER_BYTES) + 1)
    return cover(allm, lambda m: gather_sig(m * segb, "ag"))


# ------------------------------------------------------------------------------------------------ coverage check ---
def served_ar_sigs(f32):
    rb = HIDDEN * (4 if f32 else 2)
    return {ar_sig(m, f32, served_two(m * rb)) for m in range(1, max_rows(rb, MAX_REDUCE_BYTES) + 1)}


_GS = {}


def served_gather_sigs(kind):
    if kind not in _GS:
        _GS[kind] = {gather_sig(nb, kind) for nb in range(16, MAX_GATHER_BYTES + 1, 16)}
    return _GS[kind]



def tested_ar_sigs(ms, f32, both_modes=True):
    rb = HIDDEN * (4 if f32 else 2)
    out = set()
    for m in ms:
        for two in ((False, True) if both_modes else (served_two(m * rb),)):
            out.add(ar_sig(m, f32, two))
    return out


def check(verbose=False):
    """problems (empty = every served signature is in every list), and a report"""
    bad, rep = [], []
    for f32 in (False, True):
        dt = "fp32" if f32 else "bf16"
        served = served_ar_sigs(f32)
        mx = max_rows(HIDDEN * (4 if f32 else 2), MAX_REDUCE_BYTES)
        for name, ms in (("c1/mutants/selfcheck", ar_ms(f32)), ("c2", c2_ar_ms(f32))):
            miss = served - tested_ar_sigs(ms, f32)
            if miss:
                bad.append(f"AR {dt} {name}: {len(miss)} served signatures untested, e.g. {sorted(miss)[0]}")
            if max(ms) != mx:
                bad.append(f"AR {dt} {name}: max M {max(ms)} != served max {mx}")
            rep.append(f"AR {dt:4s} {name:21s} {len(ms):3d} M ({min(ms)}..{max(ms)}; served max {mx}): "
                       f"{len(served)}/{len(served)} served signatures" if not miss else
                       f"AR {dt:4s} {name:21s} MISSING {len(miss)}")
        for M in (24, 25, 38, 39, 51, 52, 102, 103, 204):
            if M <= mx:
                rep.append(f"   {dt} M={M:3d} served {'two' if served_two(M * HIDDEN * (4 if f32 else 2)) else 'one'}-shot "
                           f"sig {ar_sig(M, f32, served_two(M * HIDDEN * (4 if f32 else 2)))}")
    for kind in ("ag", "agc"):
        served = served_gather_sigs(kind)
        for name, lister, segs in (("c1", gather_ms, SEGS), ("c2", c2_gather_ms, SEGS_C2)):
            for segb in segs:  # every seg list on its own must reach every served signature and the 2 MiB bound
                got = {gather_sig(m * segb, kind) for m in lister(segb)}
                miss = served - got
                if miss:
                    bad.append(f"gather {kind} {name} seg {segb}: {len(miss)} served signatures untested, "
                               f"e.g. {sorted(miss)[0]}")
                top = max(lister(segb)) * segb
                if top + segb <= MAX_GATHER_BYTES:
                    bad.append(f"gather {kind} {name} seg {segb}: largest input {top} B is not the largest served")
            rep.append(f"gather {kind:3s} {name:4s} segs {list(segs)} M per seg {[len(lister(s)) for s in segs]} "
                       f"(largest {[max(lister(s)) * s for s in segs]} B): {len(served)}/{len(served)} served signatures "
                       "per seg")
    if verbose:
        print("\n".join(rep))
    return bad, rep


def bench_x_ops():
    """bench_r6.py phase X coverage block: (kind, M) for every served AR shape class and gathers up to 2 MiB.
    ar = [M, 5120] bf16, arf = [M, 5120] fp32, ag / agc = [M, 224] bf16 (seg 448 B) dim-0 / column gather."""
    ops = [("ar", m) for m in ar_ms(False)] + [("arf", m) for m in ar_ms(True)]
    seg = 2 * QKV
    gsel = sorted(set(cover(range(1, max_rows(seg, MAX_GATHER_BYTES) + 1), lambda m: gather_sig(m * seg, "ag"))) | {1, 6})
    ops += [("ag", m) for m in gsel] + [("agc", m) for m in gsel]
    return ops


def check_bench_x():
    ops = bench_x_ops()
    bad = []
    for f32, k in ((False, "ar"), (True, "arf")):
        rb = HIDDEN * (4 if f32 else 2)
        got = {ar_sig(m, f32, served_two(m * rb)) for kk, m in ops if kk == k}  # bench X uses the served policy
        miss = served_ar_sigs(f32) - got
        if miss:
            bad.append(f"bench X {k}: {len(miss)} served signatures untested")
    for k in ("ag", "agc"):
        got = {gather_sig(m * 2 * QKV, k) for kk, m in ops if kk == k}
        miss = served_gather_sigs(k) - got
        if miss:
            bad.append(f"bench X {k}: {len(miss)} served signatures untested, e.g. {sorted(miss)[0]}")
    return bad, ops


if __name__ == "__main__":
    b, _ = check(verbose=True)
    bb, ops = check_bench_x()
    n = {k: sum(1 for kk, _ in ops if kk == k) for k in ("ar", "arf", "ag", "agc")}
    print(f"bench X coverage block: {len(ops)} ops {n}")
    for x in b + bb:
        print("SHAPES FAIL:", x)
    print(f"SHAPES {'PASS' if not (b or bb) else 'FAIL'}: lists derived from the byte bounds "
          f"(AR <= {MAX_REDUCE_BYTES} B: bf16 M <= {max_rows(2 * HIDDEN, MAX_REDUCE_BYTES)}, fp32 M <= "
          f"{max_rows(4 * HIDDEN, MAX_REDUCE_BYTES)}; gathers <= {MAX_GATHER_BYTES} B) cover every served signature")
    sys.exit(0 if not (b or bb) else 1)
