#!/usr/bin/env python3
"""c3_rebase.py: re-base the COLEMB / PACK estimates on the 8-rank production census extracts (launcher 4f3cc1c6 era,
climb-windows/r6skew-census/prod/x<rank>.<capture>.tsv.gz; columns start end stream gx gy gz src name, ns).

Per rank and capture (4/5/6 = code/prose/maths on every rank), medians in us of:
  emb_eager   eager vocab_parallel_embedding_kernel -> next ar_kernel_pl on the same stream (the target embedding AR)
  emb_gap     end of the embedding kernel -> start of that AR
  dr_emb_ar   first ar_kernel_pl after vocab_parallel_embedding_kernel inside a graph whose step also ran the eager one
              (the DSpark embedding AR in the drafter graph)
  rows_ag     the ar_kernel_pl two collectives before each _fused_engram_post_wkv_kernel (Engram rows AG)
  wkv_ag      the ar_kernel_pl right before each _fused_engram_post_wkv_kernel (Engram wkv AG)
  wkv_gemm    the kernel between them that is not a collective and runs longest (the wkv GEMM)
Prints one line per (rank, capture) and a summary. stdlib only.
"""
import glob
import gzip
import os
import statistics as S
import sys

D = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/Documents/dsv41-offline/climb-windows/r6skew-census/prod")
CAP = {4: "code", 5: "prose", 6: "maths"}
POOL = {}
AR = "ar_kernel_pl"


def load(p):
    rows = []
    with gzip.open(p, "rt") as f:
        for l in f:
            if l[0] == "#":
                continue
            a, b, st, gx, gy, gz, src, name = l.rstrip("\n").split("\t")
            rows.append((int(a), int(b), int(st), src, name, int(gx)))
    rows.sort()
    return rows


def med(v):
    return round(S.median(v) / 1e3, 2) if v else None


def per_file(rows):
    bystream = {}
    for r in rows:
        bystream.setdefault(r[2], []).append(r)
    out = {k: [] for k in ("emb_eager", "emb_gap", "dr_emb_ar", "rows_ag", "wkv_ag", "wkv_gemm", "ag_gx")}
    for st, rs in bystream.items():
        for i, r in enumerate(rs):
            if r[4].startswith("vocab_parallel_embedding_kernel"):
                j = next((j for j in range(i + 1, min(i + 6, len(rs))) if rs[j][4].startswith(AR)), None)
                if j is None:
                    continue
                key = "emb_eager" if r[3] != "G" else "dr_emb_ar"
                out[key].append(rs[j][1] - rs[j][0])
                if r[3] != "G":
                    out["emb_gap"].append(rs[j][0] - r[1])
            elif r[4].startswith("_fused_engram_post_wkv_kernel"):
                ars = [j for j in range(i - 1, max(i - 40, -1), -1) if rs[j][4].startswith(AR)][:2]
                if len(ars) < 2:
                    continue
                w, ro = ars
                out["wkv_ag"].append(rs[w][1] - rs[w][0])
                out["rows_ag"].append(rs[ro][1] - rs[ro][0])
                mid = [rs[k] for k in range(ro + 1, w) if not rs[k][4].startswith(AR)]
                if mid:
                    out["wkv_gemm"].append(max(m[1] - m[0] for m in mid))
    return out


def main():
    tot = {}
    files = sorted(glob.glob(os.path.join(D, "x*.tsv.gz")))
    caps = {}
    for p in files:
        rk, cap = os.path.basename(p)[1:].split(".")[:2]
        caps.setdefault(rk, []).append(int(cap))
    for p in files:
        rk, cap = os.path.basename(p)[1:].split(".")[:2]
        cap = int(cap)
        if max(caps[rk]) <= 3:  # ranks with captures 1..3 only (skew8.txt: ranks 0/4 use 4/5/6)
            cap += 3
        if cap not in CAP:
            continue
        o = per_file(load(p))
        print(f"rank {rk} {CAP[cap]:5s} " + " ".join(f"{k} {med(v)} (n {len(v)})" for k, v in o.items() if k != "ag_gx"))
        for k, v in o.items():
            POOL.setdefault((CAP[cap], k), []).extend(v)
            tot.setdefault((CAP[cap], k), []).append(med(v))
    print("== pooled over ranks: p5 / p10 / p50 (us); p5 ~ the transfer alone (the last-arriving rank waits ~0)")
    for w in ("code", "prose", "maths"):
        parts = []
        for k in ("emb_eager", "dr_emb_ar", "rows_ag", "wkv_ag"):
            v = sorted(POOL.get((w, k), []))
            if v:
                q = lambda f: round(v[int(f * (len(v) - 1))] / 1e3, 2)
                parts.append(f"{k} {q(0.05)}/{q(0.10)}/{q(0.50)}")
        print(w, " | ".join(parts))
    print("== medians over ranks of the per-rank medians (us)")
    for w in ("code", "prose", "maths"):
        print(w, " ".join(f"{k} {S.median([x for x in tot.get((w, k), []) if x is not None]) if any(x is not None for x in tot.get((w, k), [])) else None}"
                          for k in ("emb_eager", "emb_gap", "dr_emb_ar", "rows_ag", "wkv_ag", "wkv_gemm")))


main()
