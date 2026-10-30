#!/usr/bin/env python3
"""engram_punch.py: release the Engram-table rows a vLLM 0.32 TP rank never reads (DeepSeek-V4.1, TP8 on 8 Sparks).

Adapted from Blackwellboy's tools/engram_slice.py (recipe DeepSeek-V4.1-Flash-1017-BLACKFOLD-TP2, d905cf0b), but the
rank -> rows map is re-derived from vLLM 0.32 (vllm/models/deepseek_v41/common/engram.py, sha aa8aad1b):

  EngramLayout: per layer, (max_ngram-1) x n_heads primes drawn upward from engram_vocab_size-1, never reused across
                layers; head_sizes = the 24 primes in (ngram, head) order; column j owns rows [sum(p[:j]), sum(p[:j+1])).
  ParallelEngramEmbedding (dp_size 1, TP8): part = cdiv(24, 8) = 3; head_start = tp_rank*3;
                vocab_start = sum(p[:head_start]); vocab_end = sum(p[:head_start+3]).
  loader:       _engram_head_shard_weight_loader -> loaded_weight.narrow(0, vocab_start, part_rows).copy_() for both
                embed.weight (fp8 [N,256]) and embed.scale (e8m0 [N,8] -> weight_scale_inv).
  runtime:      _engram_lookup_kernel reads only the rank's in-RAM copy (mlocked THP); the files are not read after load.

Modes (stdlib only; run with python3 -I):
  --dry-run   (default) headers + config + hole map: ranges, punch plan, bytes to release, free space now/after.
  --check     headers + hole map (~1 s): exit 0 iff this node's shards hold every row --rank reads (whole, or punched
              for this rank). Used by ~/dsv41-tp8.sh before vLLM starts.
  --manifest OUT          sha256 of this rank's kept byte segments (+ full-file sha when the file has no holes). Bulk read.
  --ref-all OUT           (pristine node only) one pass: full sha + kept-segment sha for ALL ranks. Bulk read.
  --punch-in-place --yes-punch [--ref-manifest REF]   punch the other ranks' rows. Refuses on the pristine node, on a
              host whose rank != --rank, while vLLM is running, if kept rows already have holes, if the pre-punch full
              sha256 does not match ~/models/dsv41/uncensored.sha, or if kept segments differ from REF.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import errno
import hashlib
import json
import os
import re
import shutil
import socket
import struct
import sys
import time

HOSTS = ["spark-one", "spark-two", "spark-three", "spark-four",
         "spark-five", "spark-six", "spark-seven", "spark-eight"]  # rank r runs on HOSTS[r] (dsv41-launch-all.sh)
PRISTINE = {"spark-four"}  # keeps the full shards as the restore source; never punched
WORLD = 8
XATTR = "user.dsv41.engram_punch"
DTYPE_BYTES = {"F8_E4M3": 1, "F8_E8M0": 1, "U8": 1, "BF16": 2, "F16": 2, "F32": 4}
CHUNK = 16 << 20


# ---------------------------------------------------------------- layout (vLLM 0.32 engram.py, verbatim logic)
def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 7, 61):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def head_sizes(cfg: dict) -> list[list[int]]:
    seen: set[int] = set()
    out = []
    for _ in cfg["engram_layer_ids"]:
        flat = []
        for _ in range(cfg["engram_max_ngram_size"] - 1):
            cur = cfg["engram_vocab_size"] - 1
            for _ in range(cfg["engram_n_heads"]):
                cur += 1
                while not _is_prime(cur) or cur in seen:
                    cur += 1
                seen.add(cur)
                flat.append(cur)
        out.append(flat)
    return out


def rank_rows(cfg: dict, rank: int, world: int) -> dict[int, tuple[int, int]]:
    """layer id -> [vocab_start, vocab_end) of TP rank `rank` (ParallelEngramEmbedding, dp_size 1)."""
    out = {}
    for j, layer in enumerate(cfg["engram_layer_ids"]):
        p = head_sizes(cfg)[j]
        n = cfg["engram_num_embeddings"][j]
        if sum(p) != n:
            raise SystemExit(f"layer {layer}: primes sum to {sum(p)} but config engram_num_embeddings is {n}; refusing")
        part = -(-len(p) // world)
        if not (world - 1) * part < len(p):
            raise SystemExit(f"layer {layer}: {len(p)} hash heads over {world} ranks leaves a rank empty (vLLM asserts)")
        hs = rank * part
        out[layer] = (sum(p[:hs]), sum(p[:hs + part]))
    return out


def prove_partition(cfg: dict, world: int) -> None:
    for j, layer in enumerate(cfg["engram_layer_ids"]):
        prev = 0
        for r in range(world):
            a, b = rank_rows(cfg, r, world)[layer]
            if a != prev or b <= a:
                raise SystemExit(f"layer {layer}: rank {r} rows [{a},{b}) do not continue at {prev}")
            prev = b
        if prev != cfg["engram_num_embeddings"][j]:
            raise SystemExit(f"layer {layer}: ranks cover [0,{prev}) not [0,{cfg['engram_num_embeddings'][j]})")


# ---------------------------------------------------------------- files
def read_header(path: str) -> tuple[int, dict]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n > 100 << 20:
            raise SystemExit(f"{path}: header length {n} implausible")
        return 8 + n, json.loads(f.read(n))


def find_tables(model: str, cfg: dict) -> dict[int, dict]:
    """layer -> {path, base, w:(lo,hi,N,cols), s:(lo,hi,N,cols), others:[(name,lo,hi)], size} with absolute offsets."""
    with open(os.path.join(model, "model.safetensors.index.json")) as f:
        wmap = json.load(f)["weight_map"]
    out = {}
    for j, layer in enumerate(cfg["engram_layer_ids"]):
        wn, sn = f"layers.{layer}.engram.embed.weight", f"layers.{layer}.engram.embed.scale"
        if wmap.get(wn) is None or wmap.get(wn) != wmap.get(sn):
            raise SystemExit(f"layer {layer}: {wn} and {sn} are not in one shard per the index")
        path = os.path.join(model, wmap[wn])
        base, h = read_header(path)
        N = cfg["engram_num_embeddings"][j]
        ent = {}
        for key, want_dtype, cols in ((wn, "F8_E4M3", cfg["engram_head_dim"]), (sn, "F8_E8M0", cfg["engram_head_dim"] // 32)):
            e = h.get(key)
            if e is None or e["dtype"] != want_dtype or e["shape"] != [N, cols]:
                raise SystemExit(f"{path}: {key} is {e}, expected {want_dtype} [{N}, {cols}]")
            lo, hi = base + e["data_offsets"][0], base + e["data_offsets"][1]
            if hi - lo != N * cols * DTYPE_BYTES[want_dtype]:
                raise SystemExit(f"{path}: {key} byte length does not match its shape")
            ent[key] = (lo, hi, N, cols)
        w, s = ent[wn], ent[sn]
        if s[0] != w[1]:
            raise SystemExit(f"{path}: scale does not follow the weight directly (layout assumption)")
        others = [(k, base + e["data_offsets"][0], base + e["data_offsets"][1])
                  for k, e in h.items() if k not in ("__metadata__", wn, sn)]
        for k, lo, hi in others:
            if lo < s[1] and hi > w[0]:
                raise SystemExit(f"{path}: tensor {k} overlaps the embed tables; refusing")
        size = os.stat(path).st_size
        end = max([s[1]] + [hi for _, _, hi in others])
        if size != end:
            raise SystemExit(f"{path}: size {size} != header end {end}")
        out[layer] = {"path": path, "file": os.path.basename(path), "base": base, "w": w, "s": s,
                      "others": others, "size": size, "meta": h.get("__metadata__")}
    return out


def kept_segments(t: dict, a: int, b: int) -> dict[str, tuple[int, int]]:
    """Exact byte ranges rank [a,b) needs from this shard (rows) plus the parts no rank may lose (header, tail)."""
    w0, _, _, wc = t["w"]
    s0, s1, _, sc = t["s"]
    return {"header": (0, t["base"]), "weight_rows": (w0 + a * wc, w0 + b * wc),
            "scale_rows": (s0 + a * sc, s0 + b * sc), "tail": (s1, t["size"])}


def punch_ranges(t: dict, a: int, b: int, bs: int) -> list[tuple[int, int]]:
    """Byte ranges to punch for rank [a,b), shrunk inward to fs blocks so no kept byte shares a punched block."""
    w0, w1, N, wc = t["w"]
    s0, s1, _, sc = t["s"]
    raw = [(w0, w0 + a * wc), (w0 + b * wc, s0 + a * sc), (s0 + b * sc, s1)]
    out = []
    for lo, hi in raw:
        lo, hi = -(-lo // bs) * bs, hi // bs * bs
        if hi > lo:
            out.append((lo, hi))
    keep = kept_segments(t, a, b)
    for lo, hi in out:
        for name, (klo, khi) in keep.items():
            if lo < khi and hi > klo and khi > klo:
                raise SystemExit(f"internal: punch [{lo},{hi}) overlaps kept {name} [{klo},{khi})")
    return out


def hole_map(path: str) -> list[tuple[int, int]]:
    holes = []
    fd = os.open(path, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size
        off = 0
        while off < size:
            try:
                h = os.lseek(fd, off, os.SEEK_HOLE)
            except OSError as e:  # fail closed: an unreadable hole map must never pass --check
                raise SystemExit(f"{path}: SEEK_HOLE failed at {off}: {e}; cannot map holes, refusing")
            if h >= size:
                break
            try:
                d = os.lseek(fd, h, os.SEEK_DATA)
            except OSError as e:
                if e.errno == errno.ENXIO:
                    d = size
                else:
                    raise
            holes.append((h, d))
            off = d
    finally:
        os.close(fd)
    return holes


def overlap(rngs, lo, hi) -> int:
    return sum(max(0, min(h, hi) - max(l, lo)) for l, h in rngs)


def get_xattr(path: str):
    try:
        if hasattr(os, "getxattr"):
            return json.loads(os.getxattr(path, XATTR))
        return json.load(open(path + ".engram_punch.json"))  # synthetic tests on macOS only
    except (OSError, ValueError):
        return None


def set_xattr(path: str, rec: dict) -> None:
    if hasattr(os, "setxattr"):
        os.setxattr(path, XATTR, json.dumps(rec).encode())
    else:
        json.dump(rec, open(path + ".engram_punch.json", "w"))


def fadvise(fd: int, off: int, n: int, advice: str) -> None:
    if hasattr(os, "posix_fadvise"):
        os.posix_fadvise(fd, off, n, getattr(os, advice))


# ---------------------------------------------------------------- hashing
def hash_pass(path: str, segments: dict[str, tuple[int, int]], full: bool, label: str = "") -> dict:
    """One sequential read; sha256 per segment (+ whole file). Page cache is dropped behind the reader."""
    hs = {k: hashlib.sha256() for k in segments}
    fh = hashlib.sha256() if full else None
    lo_all = 0 if full else min(lo for lo, _ in segments.values())
    hi_all = os.stat(path).st_size if full else max(hi for _, hi in segments.values())
    fd = os.open(path, os.O_RDONLY)
    t0 = time.time()
    try:
        fadvise(fd, 0, 0, "POSIX_FADV_DONTNEED")
        fadvise(fd, 0, 0, "POSIX_FADV_SEQUENTIAL")
        off = lo_all
        last = t0
        while off < hi_all:
            n = min(CHUNK, hi_all - off)
            # skip spans no segment needs (when not hashing the full file)
            if not full and not any(lo < off + n and hi > off for lo, hi in segments.values()):
                nxt = min([lo for lo, _ in segments.values() if lo >= off + n] + [hi_all])
                off = nxt
                continue
            buf = os.pread(fd, n, off)
            if len(buf) != n:
                raise SystemExit(f"{path}: short read at {off}")
            if fh:
                fh.update(buf)
            for k, (lo, hi) in segments.items():
                a, b = max(lo, off), min(hi, off + n)
                if a < b:
                    hs[k].update(memoryview(buf)[a - off:b - off])
            fadvise(fd, off, n, "POSIX_FADV_DONTNEED")
            off += n
            if time.time() - last > 30:
                last = time.time()
                print(f"  {label}{os.path.basename(path)}: {off / 2**30:.1f}/{hi_all / 2**30:.1f} GiB "
                      f"{(off - lo_all) / 2**30 / (last - t0):.2f} GiB/s", flush=True)
    finally:
        os.close(fd)
    out = {"segments": {k: [segments[k][0], segments[k][1], hs[k].hexdigest()] for k in segments}}
    if fh:
        out["full_sha256"] = fh.hexdigest()
    return out


def published_sha(model: str, fname: str) -> str | None:
    p = os.path.join(os.path.dirname(model.rstrip("/")), "uncensored.sha")
    try:
        for line in open(p):
            parts = line.split()
            if len(parts) == 2 and parts[1].lstrip("*") == fname:
                return parts[0]
    except OSError:
        return None
    return None


# ---------------------------------------------------------------- environment guards
def vllm_running() -> list[str]:
    hits = []
    for pid in os.listdir("/proc") if os.path.isdir("/proc") else []:
        if not pid.isdigit() or int(pid) == os.getpid():
            continue
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if "vllm" in cmd and (" serve " in cmd or "Worker" in cmd or "EngineCore" in cmd):
            hits.append(f"{pid}: {cmd[:120]}")
    return hits


def log_rows(rank: int) -> list[tuple[int, int]] | None:
    """(tp index, rows) of the last two 'Engram table offloaded' lines in ~/dsv41-logs/rank{rank}.log."""
    p = os.path.expanduser(f"~/dsv41-logs/rank{rank}.log")
    if not os.path.exists(p):
        return None
    pat = re.compile(rb"Worker_TP(\d+).*Engram table offloaded to pinned host memory: (\d+) rows")
    hits = []
    with open(p, "rb") as f:
        for line in f:
            m = pat.search(line)
            if m:
                hits.append((int(m.group(1)), int(m.group(2))))
    return hits[-2:] if hits else None


def fs_block(path: str) -> int:
    v = os.statvfs(path)
    return max(v.f_frsize or v.f_bsize, 4096)


def free_bytes(path: str) -> int:
    return shutil.disk_usage(path).free


# ---------------------------------------------------------------- punching
FALLOC_FL_KEEP_SIZE, FALLOC_FL_PUNCH_HOLE = 0x01, 0x02


def punch(fd: int, lo: int, hi: int, bs: int) -> None:
    step = 1 << 30
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        fa = libc.fallocate
        fa.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int64, ctypes.c_int64]
        fa.restype = ctypes.c_int
        off = lo
        while off < hi:
            n = min(step, hi - off)
            if fa(fd, FALLOC_FL_KEEP_SIZE | FALLOC_FL_PUNCH_HOLE, off, n) != 0:
                raise SystemExit(f"fallocate punch [{off},{off + n}) failed: {os.strerror(ctypes.get_errno())}")
            off += n
    elif sys.platform == "darwin" and os.environ.get("ENGRAM_PUNCH_TEST") == "1":  # synthetic tests only (APFS)
        import fcntl
        fcntl.fcntl(fd, 99, struct.pack("<IIqq", 0, 0, lo, hi - lo))  # F_PUNCHHOLE, struct fpunchhole
    else:
        raise SystemExit("punching is implemented for Linux only")


# ---------------------------------------------------------------- report
def plan(model: str, cfg: dict, rank: int, world: int) -> dict:
    tabs = find_tables(model, cfg)
    rows = rank_rows(cfg, rank, world)
    bs = fs_block(model)
    rep = {"host": socket.gethostname(), "rank": rank, "world": world, "fs_block": bs, "files": {}}
    for layer, t in tabs.items():
        a, b = rows[layer]
        holes = hole_map(t["path"])
        pr = punch_ranges(t, a, b, bs)
        keep = kept_segments(t, a, b)
        st = os.stat(t["path"])
        rep["files"][t["file"]] = {
            "layer": layer, "rows": [a, b], "n_rows": b - a, "table_rows": t["w"][2],
            "kept_segments": {k: list(v) for k, v in keep.items()},
            "kept_bytes": sum(hi - lo for lo, hi in keep.values()),
            "punch": [list(x) for x in pr],
            "punch_bytes": sum(hi - lo for lo, hi in pr),
            "to_release": sum(hi - lo for lo, hi in pr) - sum(overlap(holes, lo, hi) for lo, hi in pr),
            "holes": [list(x) for x in holes],
            "hole_bytes": sum(d - h for h, d in holes),
            "holes_in_kept": {k: overlap(holes, lo, hi) for k, (lo, hi) in keep.items() if overlap(holes, lo, hi)},
            "size": st.st_size, "allocated": st.st_blocks * 512, "xattr": get_xattr(t["path"]),
        }
    return rep


PROBE_ROWS = 64


def zero_probe(model: str, cfg: dict, rank: int, world: int) -> list[str]:
    """Read PROBE_ROWS evenly spaced rows (first and last included) of this rank's weight and scale rows. Holes
    are caught by the hole map; this catches zeros written as DATA (e.g. an rsync from a punched node). On the pristine
    tables 0 of 2048 random rows per table are all-zero, and an all-zero E8M0 scale row (2^-127 x8) never is."""
    problems = []
    rows = rank_rows(cfg, rank, world)
    for layer, t in find_tables(model, cfg).items():
        a, b = rows[layer]
        w0, _, _, wc = t["w"]
        s0, _, _, sc = t["s"]
        k = min(PROBE_ROWS, b - a)
        idx = sorted({a + (b - 1 - a) * i // max(k - 1, 1) for i in range(k)})
        zs = zw = 0
        fd = os.open(t["path"], os.O_RDONLY)
        try:
            for r in idx:
                zw += os.pread(fd, wc, w0 + r * wc) == bytes(wc)
                zs += os.pread(fd, sc, s0 + r * sc) == bytes(sc)
            fadvise(fd, 0, 0, "POSIX_FADV_DONTNEED")
        finally:
            os.close(fd)
        if zs or 2 * zw > len(idx):
            problems.append(f"{t['file']}: {zs}/{len(idx)} sampled scale rows and {zw}/{len(idx)} weight rows of rank "
                            f"{rank} are all-zero data (copied from a punched node?): restore from spark-four")
    return problems


def check(model: str, cfg: dict, rank: int, world: int) -> list[str]:
    problems = []
    rep = plan(model, cfg, rank, world)
    for fname, f in rep["files"].items():
        if f["holes_in_kept"]:
            problems.append(f"{fname}: rows rank {rank} reads have holes {f['holes_in_kept']} "
                            "(punched for another rank/world): restore from spark-four")
        x = f["xattr"]
        if x and (x.get("rank") != rank or x.get("world") != world):
            problems.append(f"{fname}: punched for rank {x.get('rank')}/{x.get('world')}, not {rank}/{world}")
        if f["hole_bytes"] and not x:
            print(f"engram check WARNING: {fname} has {f['hole_bytes']} B of holes outside rank {rank}'s rows "
                  "but no punch record", file=sys.stderr)
    if not problems:
        problems += zero_probe(model, cfg, rank, world)
    return problems


def deep_check(model: str, cfg: dict, rank: int, world: int, state_dir: str) -> list[str]:
    """Re-hash this rank's kept segments against the manifest saved at punch time (bulk read, ~23.6 GiB)."""
    problems = []
    rows = rank_rows(cfg, rank, world)
    for layer, t in find_tables(model, cfg).items():
        p = os.path.join(state_dir, f"{t['file']}.post.json")
        if not os.path.exists(p):
            problems.append(f"{t['file']}: no saved manifest {p}")
            continue
        want = json.load(open(p))["segments"]
        got = hash_pass(t["path"], kept_segments(t, *rows[layer]), full=False, label="deep ")["segments"]
        for k in want:
            if want[k] != got.get(k):
                problems.append(f"{t['file']}: kept {k} sha256 differs from the punch-time manifest")
    return problems


def human(rep: dict, free_now: int) -> str:
    L = [f"host {rep['host']}  rank {rep['rank']}/{rep['world']}  fs block {rep['fs_block']}"]
    tot = 0
    for fname, f in rep["files"].items():
        a, b = f["rows"]
        L.append(f"  {fname} layer {f['layer']}: keep rows [{a:,}, {b:,}) = {f['n_rows']:,} of {f['table_rows']:,}; "
                 f"kept {f['kept_bytes'] / 2**30:.2f} GiB; release {f['to_release'] / 2**30:.2f} GiB "
                 f"({len(f['punch'])} ranges); holes now {f['hole_bytes']} B; allocated {f['allocated'] / 2**30:.2f} GiB")
        for lo, hi in f["punch"]:
            L.append(f"      punch [{lo:,}, {hi:,})  {(hi - lo) / 2**30:.3f} GiB")
        tot += f["to_release"]
    L.append(f"  total release {tot / 2**30:.2f} GiB ({tot:,} B); free now {free_now / 2**30:.1f} GiB "
             f"-> expected {(free_now + tot) / 2**30:.1f} GiB")
    return "\n".join(L)


# ---------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=os.path.expanduser("~/models/dsv41/uncensored-fp8"))
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world", type=int, default=WORLD)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--deep", action="store_true", help="with --check: also re-hash kept rows vs the punch manifest")
    ap.add_argument("--json", help="dry-run: also write the plan as JSON here")
    ap.add_argument("--manifest", help="write sha256 of this rank's kept segments here (bulk read)")
    ap.add_argument("--ref-all", help="pristine node: write full + all-rank kept-segment sha256 here (bulk read)")
    ap.add_argument("--punch-in-place", action="store_true")
    ap.add_argument("--yes-punch", action="store_true")
    ap.add_argument("--ref-manifest", help="--ref-all output from spark-four; kept segments must match it")
    ap.add_argument("--no-ref", action="store_true", help="punch without a spark-four reference (not recommended)")
    ap.add_argument("--allow-while-serving", action="store_true")
    ap.add_argument("--state-dir", default=os.path.expanduser("~/dsv41-stage/engram-punch/state"))
    ap.add_argument("--test-host", help=argparse.SUPPRESS)  # synthetic tests: pretend to be this host
    a = ap.parse_args(argv)
    try:
        os.nice(19)
    except OSError:
        pass
    if a.test_host and os.environ.get("ENGRAM_PUNCH_TEST") != "1":
        raise SystemExit("--test-host is for the synthetic test only (ENGRAM_PUNCH_TEST=1)")
    host = a.test_host or socket.gethostname()
    if not 0 <= a.rank < a.world:
        raise SystemExit("--rank must be in [0, --world)")
    with open(os.path.join(a.model, "config.json")) as f:
        cfg = json.load(f)
    cfg = cfg.get("text_config", cfg)
    if not cfg.get("engram_layer_ids"):
        print("engram: config has no Engram layers; nothing to do")
        return 0
    prove_partition(cfg, a.world)

    if a.check:
        problems = check(a.model, cfg, a.rank, a.world)
        if a.deep and not problems:
            problems = deep_check(a.model, cfg, a.rank, a.world, a.state_dir)
        for p in problems:
            print(f"ENGRAM CHECK FAILED ({host} rank {a.rank}/{a.world}): {p}", file=sys.stderr)
        if not problems:
            rep = plan(a.model, cfg, a.rank, a.world)
            state = "punched" if any(f["hole_bytes"] for f in rep["files"].values()) else "whole"
            print(f"engram check OK: {host} holds every row rank {a.rank}/{a.world} reads ({state})")
        return 1 if problems else 0

    if a.ref_all:
        if host not in PRISTINE:
            raise SystemExit(f"--ref-all must run on a pristine node {sorted(PRISTINE)}")
        tabs = find_tables(a.model, cfg)
        out = {"host": host, "world": a.world, "time": time.time(), "files": {}}
        for layer, t in tabs.items():
            if hole_map(t["path"]):
                raise SystemExit(f"{t['path']}: pristine copy has holes")
            segs = {}
            for r in range(a.world):
                ra, rb = rank_rows(cfg, r, a.world)[layer]
                k = kept_segments(t, ra, rb)
                segs[f"r{r}.weight_rows"], segs[f"r{r}.scale_rows"] = k["weight_rows"], k["scale_rows"]
                segs["header"], segs["tail"] = k["header"], k["tail"]
            res = hash_pass(t["path"], segs, full=True, label="ref ")
            pub = published_sha(a.model, t["file"])
            res["published_sha256"], res["full_matches_published"] = pub, res["full_sha256"] == pub
            out["files"][t["file"]] = res
            print(f"{t['file']}: full {res['full_sha256'][:16]} published match {res['full_matches_published']}")
        json.dump(out, open(a.ref_all, "w"), indent=1)
        return 0 if all(f["full_matches_published"] for f in out["files"].values()) else 1

    rep = plan(a.model, cfg, a.rank, a.world)
    free0 = free_bytes(a.model)
    lr = log_rows(a.rank)
    if lr is not None:
        want = [(a.rank, f["n_rows"]) for f in sorted(rep["files"].values(), key=lambda f: f["layer"])]
        rep["log_rows"], rep["log_matches"] = lr, lr == want
    rep["host_rank"] = HOSTS.index(host) if host in HOSTS else None
    rep["free_now"], rep["free_expected_after"] = free0, free0 + sum(f["to_release"] for f in rep["files"].values())

    if not a.punch_in_place and not a.manifest:
        print(human(rep, free0))
        print(f"  host map rank {rep['host_rank']}  log rows {rep.get('log_rows')}  log matches {rep.get('log_matches')}")
        if a.json:
            json.dump(rep, open(a.json, "w"), indent=1)
        return 0

    # ---- from here on: bulk reads (manifest) and/or writes (punch)
    if rep["host_rank"] != a.rank:
        raise SystemExit(f"refusing: {host} runs rank {rep['host_rank']}, not --rank {a.rank}")
    if lr is not None and not rep["log_matches"]:
        raise SystemExit(f"refusing: rank log rows {lr} != computed {[(a.rank, f['n_rows']) for f in rep['files'].values()]}")
    tabs = find_tables(a.model, cfg)
    rows = rank_rows(cfg, a.rank, a.world)

    if a.manifest and not a.punch_in_place:
        out = {"host": host, "rank": a.rank, "world": a.world, "time": time.time(), "files": {}}
        for layer, t in tabs.items():
            segs = kept_segments(t, *rows[layer])
            full = not hole_map(t["path"])
            out["files"][t["file"]] = hash_pass(t["path"], segs, full=full)
        json.dump(out, open(a.manifest, "w"), indent=1)
        print(f"wrote {a.manifest}")
        return 0

    # ---- punch
    if not a.yes_punch:
        raise SystemExit("refusing: --punch-in-place is permanent (restore = 2 x 101.5 GB from spark-four); add --yes-punch")
    if host in PRISTINE:
        raise SystemExit(f"refusing: {host} is the pristine restore source")
    running = vllm_running()
    if running and not a.allow_while_serving:
        raise SystemExit("refusing: vLLM is running here:\n  " + "\n  ".join(running[:4]))
    ref = None
    if a.ref_manifest:
        ref = json.load(open(a.ref_manifest))
        if ref.get("host") not in PRISTINE or ref.get("world") != a.world:
            raise SystemExit("refusing: --ref-manifest is not a pristine-node --ref-all for this world size")
    elif not a.no_ref:
        raise SystemExit("refusing: give --ref-manifest (spark-four --ref-all output) or --no-ref")
    os.makedirs(a.state_dir, exist_ok=True)
    bs = rep["fs_block"]
    summary = {"host": host, "rank": a.rank, "world": a.world, "free_before": free0, "files": {}}
    for layer, t in tabs.items():
        f = rep["files"][t["file"]]
        if f["holes_in_kept"]:
            raise SystemExit(f"refusing: {t['file']} kept rows already have holes (prepared for another rank)")
        x = get_xattr(t["path"])
        already = bool(f["hole_bytes"])
        if already and not (x and x.get("rank") == a.rank and x.get("world") == a.world):
            raise SystemExit(f"refusing: {t['file']} has holes without a matching punch record")
        segs = kept_segments(t, *rows[layer])
        print(f"{t['file']}: pre-punch hash pass ({'kept only' if already else 'full + kept'})", flush=True)
        pre = hash_pass(t["path"], segs, full=not already, label="pre ")
        if not already:
            pub = published_sha(a.model, t["file"])
            if pre["full_sha256"] != pub:
                raise SystemExit(f"refusing: {t['file']} full sha256 {pre['full_sha256']} != published {pub}")
        if ref is not None:
            rf = ref["files"][t["file"]]["segments"]
            for k, (lo, hi, sha) in pre["segments"].items():
                rk = k if k in ("header", "tail") else f"r{a.rank}.{k}"
                if rf[rk][:2] != [lo, hi] or rf[rk][2] != sha:
                    raise SystemExit(f"refusing: {t['file']} {k} differs from spark-four reference")
        json.dump(pre, open(os.path.join(a.state_dir, f"{t['file']}.pre.json"), "w"), indent=1)
        blocks0 = os.stat(t["path"]).st_blocks
        set_xattr(t["path"], {"rank": a.rank, "world": a.world, "rows": list(rows[layer]), "state": "punching",
                              "time": int(time.time())})  # written first: a crash mid-punch stays resumable
        fd = os.open(t["path"], os.O_RDWR)
        try:
            for lo, hi in f["punch"]:
                punch(fd, lo, hi, bs)
            os.fsync(fd)
        finally:
            os.close(fd)
        holes = hole_map(t["path"])
        bad = {k: overlap(holes, lo, hi) for k, (lo, hi) in segs.items() if overlap(holes, lo, hi)}
        if bad:
            raise SystemExit(f"{t['file']}: kept segments have holes after punching {bad}: STOP, restore from spark-four")
        missing = sum((hi - lo) - overlap(holes, lo, hi) for lo, hi in f["punch"])
        print(f"{t['file']}: post-punch hash pass (kept segments, page cache dropped)", flush=True)
        post = hash_pass(t["path"], segs, full=False, label="post ")
        for k in segs:
            if post["segments"][k][2] != pre["segments"][k][2]:
                raise SystemExit(f"{t['file']}: kept {k} changed by punching: STOP, restore from spark-four")
        set_xattr(t["path"], {"rank": a.rank, "world": a.world, "rows": list(rows[layer]), "state": "done",
                              "time": int(time.time())})
        blocks1 = os.stat(t["path"]).st_blocks
        summary["files"][t["file"]] = {"released": (blocks0 - blocks1) * 512, "punch_not_hole_bytes": missing,
                                       "kept_verified": list(segs)}
        print(f"{t['file']}: released {(blocks0 - blocks1) * 512 / 2**30:.2f} GiB; kept segments verified "
              f"(sha256 pre == post); punch ranges not reported as holes: {missing} B", flush=True)
        json.dump(post, open(os.path.join(a.state_dir, f"{t['file']}.post.json"), "w"), indent=1)
    summary["free_after"] = free_bytes(a.model)
    json.dump(summary, open(os.path.join(a.state_dir, "punch-summary.json"), "w"), indent=1)
    print(f"free {free0 / 2**30:.1f} -> {summary['free_after'] / 2**30:.1f} GiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
