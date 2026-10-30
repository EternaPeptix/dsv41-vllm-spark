# SPDX-License-Identifier: Apache-2.0
"""DSV41_FAST_LOAD (goal window 4, Loader phase "g4l", 2026-10-27): parallel direct-read
safetensors stream for the DeepSeek-V4.1 TP8 deployment. Default OFF (DSV41_FAST_LOAD=1 enables).

Stock path (vLLM 0.32 + the DSv4.1 VL wrapper): ``safetensors_weights_iterator`` yields lazy
mmap views of all 48 shards, the VL wrapper materialises and sorts the whole stream by mapped
name, then the weight loaders copy from the views in that order: ~93K chains of page faults at
~1.1 GB/s, ~194 GB read per rank, ~168 s.

Fast path: the SAME (name, tensor) sequence in the SAME order (the sort is done here on the
mapped names, identical key, stable), but
  * headers are parsed directly and the VL wrapper takes the stream pre-sorted, so nothing is
    materialised;
  * N reader threads read each tensor (O_DIRECT, aligned superset) into a pinned host buffer ahead
    of the consumer, bounded by a byte window (torch's caching host allocator: a block is reused only
    after the tensor viewing it is gone, and every block is freed when the stream ends;
    DSV41_FAST_LOAD_BUF=mmap uses unmapped-on-free THP mappings instead, which measured slower on
    nodes that had to compact memory for them); expert w1/w3 weights and scales read only the rows
    ``_load_w13`` narrows to on this rank (rows [R/tp*rank, R/tp*(rank+1)) of dim 0), every other
    tensor is read whole;
  * the Engram host tables are filled directly with this rank's rows (buffered pread into the
    parameter memory = ``_engram_head_shard_weight_loader``'s narrow + copy), so a punched hole is
    never read and the 12.7 GB tables are not copied twice.
Every byte a weight loader reads is the file byte; the weight loaders, their order and
process_weights_after_loading are unchanged. Exactness is checked per parameter with
DSV41_WEIGHT_DIGEST=<dir> (sha256 of every parameter after load, one JSON per rank and load).
Setup problems fall back to the stock path; a reader error fails the load loudly.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import mmap
import os
import re
import struct
import threading
import time
from collections.abc import Iterator

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_ST_DTYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "F8_E8M0": torch.float8_e8m0fnu,
    "U64": torch.uint64,
    "U32": torch.uint32,
    "U16": torch.uint16,
}
# Checkpoint names whose dim-0 TP slice _load_w13 takes (weight and MX scale of w1 / w3).
_ROWS_RX = re.compile(r"\.experts\.\d+\.(w1|w3)\.(weight|scale)$")
_ALIGN = 4096
_ENG_CHUNK = 64 << 20
_SLAB = 64 << 20


def _proc_mem() -> dict:
    out = {}
    try:
        with open("/proc/self/status") as f:
            for line in f:
                k = line.split(":", 1)[0]
                if k in ("VmRSS", "RssAnon", "RssFile", "RssShmem"):
                    out[k] = int(line.split()[1]) // 1024
    except OSError:
        pass
    return out


def _new_map(n: int, thp: bool) -> torch.Tensor:
    """Private anonymous mapping as a uint8 tensor; torch.frombuffer keeps the mmap object
    exported, so the mapping is unmapped when the last view of it is freed."""
    mm = mmap.mmap(-1, n, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    if thp:
        try:
            mm.madvise(mmap.MADV_HUGEPAGE)
        except (AttributeError, OSError, ValueError):
            pass
    return torch.frombuffer(mm, dtype=torch.uint8)


def enabled() -> bool:
    return os.environ.get("DSV41_FAST_LOAD", "0") == "1"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _floor(x: int) -> int:
    return x & ~(_ALIGN - 1)


def _ceil(x: int) -> int:
    return (x + _ALIGN - 1) & ~(_ALIGN - 1)


class _Unsupported(Exception):
    pass


class _Item:
    """One yielded tensor: read [start + r0, start + r1) of the file into a buffer whose byte 0
    is file offset a0 (aligned); the tensor views [start - a0, start - a0 + nbytes)."""

    __slots__ = (
        "mapped", "name", "fidx", "dt", "shape", "start", "nbytes", "kind", "r0", "r1",
        "a0", "buf_size", "tensor", "done",
    )

    def __init__(self, mapped, name, fidx, dt, shape, start, nbytes):
        self.mapped, self.name, self.fidx = mapped, name, fidx
        self.dt, self.shape, self.start, self.nbytes = dt, shape, start, nbytes
        self.kind, self.r0, self.r1 = "whole", 0, nbytes
        self.a0 = _floor(start)
        self.buf_size = max(_ceil(start + nbytes) - self.a0, _ALIGN)
        self.tensor = None
        self.done = False


class _EngChunk:
    __slots__ = ("mv", "fidx", "file_off", "n", "done")

    def __init__(self, mv, fidx, file_off, n):
        self.mv, self.fidx, self.file_off, self.n = mv, fidx, file_off, n
        self.done = False


def _is_engram_param(p: torch.Tensor) -> bool:
    if getattr(p, "engram_vocab_start", None) is None or p.device.type != "cpu":
        return False
    try:
        from vllm.models.deepseek_v41.common.engram import (
            _engram_head_shard_weight_loader,
        )
    except Exception:
        return False
    return (
        getattr(p, "weight_loader", None) is _engram_head_shard_weight_loader
        and p.is_contiguous()
        and p.dim() == 2
    )


class FastWeightStream:
    """Weights source for models that declare ``_dsv41_fast_load_ok`` (the DSv4.1 VL wrapper).
    Not a plain iterable on purpose: a consumer that materialised it would hold every tensor."""

    def __init__(self, files: list[str], tp_rank: int, tp_size: int, use_tqdm: bool) -> None:
        from vllm.model_executor.model_loader.weight_utils import _natural_sort_key

        self.files = sorted(files, key=_natural_sort_key)
        self.tp_rank, self.tp_size, self.use_tqdm = tp_rank, tp_size, use_tqdm
        self.threads = max(1, _env_int("DSV41_FAST_LOAD_THREADS", 8))
        self.eng_threads = min(self.threads, max(0, _env_int("DSV41_FAST_LOAD_ENG_THREADS", 2)))
        self.window = max(64, _env_int("DSV41_FAST_LOAD_WINDOW_MB", 2048)) << 20
        self.direct = os.environ.get("DSV41_FAST_LOAD_DIRECT", "1") == "1"
        self.buf_mode = os.environ.get("DSV41_FAST_LOAD_BUF", "pin")  # pin | mmap | malloc
        if self.buf_mode not in ("mmap", "pin", "malloc"):
            self.buf_mode = "pin"
        self.thp = os.environ.get("DSV41_FAST_LOAD_THP", "1") == "1"
        self.slab = None
        self.slab_off = 0
        self.slab_lock = threading.Lock()
        self.n_maps = 0
        self.rows_rule = os.environ.get("DSV41_FAST_LOAD_ROWS", "1") == "1"
        self.engram_direct = os.environ.get("DSV41_FAST_LOAD_ENGRAM", "1") == "1"
        self.sizes = [os.path.getsize(p) for p in self.files]
        self.entries: list[tuple] = []  # stock order: files natural-sorted, keys sorted
        for fidx, path in enumerate(self.files):
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(n))
            hdr.pop("__metadata__", None)
            ds = 8 + n
            for name in sorted(hdr):
                v = hdr[name]
                dt = _ST_DTYPES.get(v["dtype"])
                if dt is None:
                    raise _Unsupported(f"dtype {v['dtype']} of {name}")
                a, b = v["data_offsets"]
                shape = tuple(int(s) for s in v["shape"])
                if ds + b > self.sizes[fidx]:
                    raise _Unsupported(f"{name} extends past the end of {path}")
                self.entries.append((name, fidx, v["dtype"], dt, shape, ds + a, b - a))

    def __iter__(self):
        raise TypeError("FastWeightStream is consumed through dsv41_sorted() only")

    # ---- entry point used by the DSv4.1 VL wrapper (vl_model.py) -------------------------------
    def dsv41_sorted(self, mapper, model) -> Iterator[tuple[str, torch.Tensor]]:
        """== sorted(mapper.apply(stock_iterator), key=name), lazily, with fast reads."""
        mapped = []
        for e in self.entries:
            r = mapper._map_name_with_shard(e[0])
            if r is None:
                continue
            if r[1] is not None:
                return self._stock_sorted(mapper, "stacked shard mapping")
            mapped.append((r[0], e))
        mapped.sort(key=lambda x: x[0])
        try:
            self._plan(mapped, model)
            self._canary()
        except Exception as exc:  # nothing yielded yet: the stock path is still available
            logger.warning("DSV41_FAST_LOAD: falling back to the stock loader (%s: %s)",
                           type(exc).__name__, exc)
            return self._stock_sorted(mapper, "setup failed")
        return self._run()

    def _stock_sorted(self, mapper, why: str):
        from vllm.model_executor.model_loader.weight_utils import (
            safetensors_weights_iterator,
        )

        logger.warning("DSV41_FAST_LOAD: stock iterator (%s)", why)
        it = safetensors_weights_iterator(self.files, self.use_tqdm, None)
        return iter(sorted(mapper.apply(it), key=lambda x: x[0]))

    # ---- plan ---------------------------------------------------------------------------------
    def _plan(self, mapped, model) -> None:
        params = dict(model.named_parameters()) if model is not None else {}
        self.main: list[_Item] = []
        self.eng: list[_EngChunk] = []
        self.eng_tables = []
        self.eng_meta = []
        n_rows = 0
        for mname, (name, fidx, dts, dt, shape, start, nbytes) in mapped:
            p = params.get(mname)
            if self.engram_direct and p is not None and _is_engram_param(p):
                rows, v0 = int(p.shape[0]), int(p.engram_vocab_start)
                if (
                    len(shape) == 2
                    and shape[0] > 0
                    and nbytes % shape[0] == 0
                    and tuple(shape[1:]) == tuple(p.shape[1:])
                    and nbytes // shape[0] == p.shape[1] * p.element_size()
                    and 0 <= v0
                    and v0 + rows <= shape[0]
                ):
                    rb = nbytes // shape[0]
                    mv = memoryview(p.data.view(torch.uint8).reshape(-1).numpy())
                    off0, tot = start + v0 * rb, rows * rb
                    for o in range(0, tot, _ENG_CHUNK):
                        n = min(_ENG_CHUNK, tot - o)
                        self.eng.append(_EngChunk(mv[o:o + n], fidx, off0 + o, n))
                    self.eng_tables.append((mname, v0, rows, tot))
                    self.eng_meta.append((mname, (name, fidx, dts, dt, shape, start, nbytes), p, v0, rows))
                    continue
            it = _Item(mname, name, fidx, dt, shape, start, nbytes)
            if (
                self.rows_rule
                and _ROWS_RX.search(name)
                and len(shape) == 2
                and shape[0] > 0
                and shape[0] % self.tp_size == 0
                and nbytes % shape[0] == 0
            ):
                per, rb = shape[0] // self.tp_size, nbytes // shape[0]
                it.kind, it.r0, it.r1 = "rows", per * self.tp_rank * rb, per * (self.tp_rank + 1) * rb
                n_rows += 1
            self.main.append(it)
        self.n_rows = n_rows

    # ---- reads --------------------------------------------------------------------------------
    def _fd(self, fds: dict, fidx: int, direct: bool) -> int:
        key = (fidx, direct)
        fd = fds.get(key)
        if fd is None:
            flags = os.O_RDONLY | (os.O_DIRECT if direct else 0)
            fd = os.open(self.files[fidx], flags)
            if not direct:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
            fds[key] = fd
        return fd

    @staticmethod
    def _pread_full(fd: int, mv: memoryview, off: int, need: int) -> None:
        """Read until `need` bytes of mv are filled (mv may be longer: aligned O_DIRECT tail)."""
        pos = 0
        while pos < need:
            r = os.preadv(fd, [mv[pos:]], off + pos)
            if r <= 0:
                raise EOFError(f"short read at {off + pos} (got {pos} of {need})")
            pos += r

    def _alloc(self, n: int) -> torch.Tensor:
        """n is a multiple of 4096; the result is 4096-aligned (O_DIRECT)."""
        if self.buf_mode == "mmap":
            try:
                if n > _SLAB // 2:  # big tensor: its own mapping (2 MiB rounded for THP)
                    with self.slab_lock:
                        self.n_maps += 1
                    return _new_map((n + (2 << 20) - 1) & ~((2 << 20) - 1), self.thp)[:n]
                with self.slab_lock:
                    if self.slab is None or self.slab_off + n > _SLAB:
                        self.slab = _new_map(_SLAB, self.thp)
                        self.slab_off = 0
                        self.n_maps += 1
                    t = self.slab[self.slab_off:self.slab_off + n]
                    self.slab_off += n
                return t
            except Exception:
                self.buf_mode = "malloc"
        if self.buf_mode == "pin":
            try:
                return torch.empty(n, dtype=torch.uint8, pin_memory=True)
            except Exception:
                self.buf_mode = "malloc"
        raw = torch.empty(n + _ALIGN, dtype=torch.uint8)
        k = (-raw.data_ptr()) % _ALIGN
        return raw[k:k + n]

    def _read_item(self, it: _Item, fds: dict) -> torch.Tensor:
        buf = self._alloc(it.buf_size)
        mv = memoryview(buf.numpy())
        s, e = it.start + it.r0, it.start + it.r1
        if e > s:
            done = False
            if self.direct and buf.data_ptr() % _ALIGN == 0:
                lo, hi = _floor(s), _ceil(e)
                try:
                    self._pread_full(self._fd(fds, it.fidx, True), mv[lo - it.a0:hi - it.a0], lo, e - lo)
                    done = True
                except OSError:  # EINVAL etc.: buffered read of the same bytes below
                    self.direct_errors += 1
            if not done:
                self._pread_full(self._fd(fds, it.fidx, False), mv[s - it.a0:e - it.a0], s, e - s)
                os.posix_fadvise(self._fd(fds, it.fidx, False), s, e - s, os.POSIX_FADV_DONTNEED)
        if it.nbytes == 0:
            return torch.empty(it.shape, dtype=it.dt)
        d = it.start - it.a0
        raw = buf[d:d + it.nbytes]
        if (raw.data_ptr() % it.dt.itemsize) == 0:
            return raw.view(it.dt).view(it.shape)
        out = torch.empty(it.shape, dtype=it.dt)  # misaligned dtype view: copy (never seen)
        out.view(-1).view(torch.uint8).copy_(raw)
        return out

    def _read_eng(self, ch: _EngChunk, fds: dict) -> None:
        fd = self._fd(fds, ch.fidx, False)
        self._pread_full(fd, ch.mv, ch.file_off, ch.n)
        os.posix_fadvise(fd, ch.file_off, ch.n, os.POSIX_FADV_DONTNEED)

    def _canary(self) -> None:
        """Before any thread starts: per safetensors dtype the smallest whole tensor, plus one
        rows tensor, read through _read_item and compared with safe_open's bytes."""
        from safetensors import safe_open

        pick: dict[str, _Item] = {}
        rows_pick = None
        for it in self.main:
            if it.kind == "rows":
                if rows_pick is None or it.nbytes < rows_pick.nbytes:
                    rows_pick = it
                continue
            k = str(it.dt)
            if it.nbytes > 0 and (k not in pick or it.nbytes < pick[k].nbytes):
                pick[k] = it
        probes = list(pick.values()) + ([rows_pick] if rows_pick is not None else [])
        fds: dict = {}
        self.direct_errors = 0
        try:
            for it in probes:
                mine = self._read_item(it, fds)
                with safe_open(self.files[it.fidx], framework="pt") as f:
                    ref = f.get_tensor(it.name)
                if ref.dtype != mine.dtype or tuple(ref.shape) != tuple(mine.shape):
                    raise _Unsupported(f"canary {it.name}: {mine.dtype}{tuple(mine.shape)} vs "
                                       f"{ref.dtype}{tuple(ref.shape)}")
                a, b = mine.reshape(-1).view(torch.uint8), ref.reshape(-1).view(torch.uint8)
                if it.kind == "rows":
                    a, b = a[it.r0:it.r1], b[it.r0:it.r1]
                if not torch.equal(a, b):
                    raise _Unsupported(f"canary {it.name}: bytes differ")
        finally:
            for fd in fds.values():
                os.close(fd)
        self.canary = len(probes)

    # ---- run ------------------------------------------------------------------------------------
    def _run(self) -> Iterator[tuple[str, torch.Tensor]]:
        self.cv = threading.Condition()
        self.err: BaseException | None = None
        self.stop = False
        self.next_main = 0
        self.next_eng = 0
        self.eng_done = 0
        self.inflight = 0
        self.peak_inflight = 0
        self.read_bytes = 0
        self.eng_bytes = 0
        self.direct_errors = 0
        mem0 = _proc_mem()
        t0 = time.perf_counter()
        threads = [
            threading.Thread(target=self._reader, args=(i < self.eng_threads,),
                             name=f"dsv41-fastload-{i}", daemon=True)
            for i in range(self.threads)
        ]
        for th in threads:
            th.start()
        wait_s = 0.0
        t_eng_end = None
        try:
            for it in self.main:
                tw = time.perf_counter()
                with self.cv:
                    while not it.done and self.err is None:
                        self.cv.wait()
                    if self.err is not None:
                        raise RuntimeError(f"DSV41_FAST_LOAD reader failed on or before "
                                           f"{it.name}") from self.err
                    t = it.tensor
                    it.tensor = None
                wait_s += time.perf_counter() - tw
                yield it.mapped, t
                del t
                with self.cv:
                    self.inflight -= it.buf_size
                    self.cv.notify_all()
            t_main_end = time.perf_counter()
            with self.cv:
                while self.eng_done < len(self.eng) and self.err is None:
                    self.cv.wait()
                if self.err is not None:
                    raise RuntimeError("DSV41_FAST_LOAD Engram fill failed") from self.err
            t_eng_end = time.perf_counter()
            self._engram_spot_check()
        finally:
            with self.cv:
                self.stop = True
                self.cv.notify_all()
            for th in threads:
                th.join(timeout=60)
            with self.slab_lock:
                self.slab = None  # the last slab is unmapped with its last tensor
            if self.buf_mode == "pin":
                try:
                    torch._C._host_emptyCache()
                except Exception as exc:
                    logger.warning("DSV41_FAST_LOAD: host emptyCache failed: %s", exc)
            try:
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass
        mem1 = _proc_mem()
        dt = (t_eng_end or time.perf_counter()) - t0
        rb = self.read_bytes + self.eng_bytes
        logger.info(
            "DSV41_FAST_LOAD: %d tensors (%d row-sliced), %.2f GB + Engram %.2f GB direct "
            "(%d tables) read in %.1f s = %.2f GB/s; consumer waited %.1f s, main stream done "
            "at %.1f s; threads %d (Engram-first %d), window %d MiB, peak in flight %.0f MiB, "
            "O_DIRECT %s (fallbacks %d), buffers %s (%d maps, THP %s), canary %d OK, Engram spot "
            "checks %d OK; worker RSS %s -> %s MB (anon %s -> %s, shmem %s -> %s)",
            len(self.main), self.n_rows, self.read_bytes / 1e9, self.eng_bytes / 1e9,
            len(self.eng_tables), dt, rb / dt / 1e9 if dt > 0 else 0.0, wait_s,
            t_main_end - t0, self.threads, self.eng_threads, self.window >> 20,
            self.peak_inflight / 2**20, self.direct, self.direct_errors, self.buf_mode,
            self.n_maps, self.thp, self.canary, getattr(self, "eng_checked", 0),
            mem0.get("VmRSS"), mem1.get("VmRSS"), mem0.get("RssAnon"), mem1.get("RssAnon"),
            mem0.get("RssShmem"), mem1.get("RssShmem"),
        )

    def _engram_spot_check(self) -> None:
        """Three 64 KiB windows per Engram table (first, middle, last bytes of this rank's rows)
        against safe_open's rows of the same range: catches a wrong row range on every launch."""
        from safetensors import safe_open

        self.eng_checked = 0
        for mname, (name, fidx, dts, dt, shape, start, nbytes), p, v0, rows in self.eng_meta:
            rb = nbytes // shape[0]
            mine = p.data.view(torch.uint8).reshape(-1)
            tot = rows * rb
            with safe_open(self.files[fidx], framework="pt") as f:
                sl = f.get_slice(name)
                for a in sorted({0, max(0, tot // 2 - 32768), max(0, tot - 65536)}):
                    b = min(tot, a + 65536)
                    ra, rz = a // rb, -(-b // rb)
                    ref = sl[v0 + ra:v0 + rz].reshape(-1).view(torch.uint8)
                    if not torch.equal(mine[a:b], ref[a - ra * rb:b - ra * rb]):
                        raise RuntimeError(
                            f"DSV41_FAST_LOAD Engram spot check failed: {mname} bytes "
                            f"[{a}, {b}) of rows [{v0}, {v0 + rows})")
                    self.eng_checked += 1

    def _reader(self, engram_first: bool) -> None:
        fds: dict = {}
        try:
            while True:
                job = None
                with self.cv:
                    while job is None:
                        if self.err is not None or self.stop:
                            return
                        main_ok = self.next_main < len(self.main) and (
                            self.inflight == 0
                            or self.inflight + self.main[self.next_main].buf_size <= self.window
                        )
                        eng_ok = self.next_eng < len(self.eng)
                        if eng_ok and (engram_first or not main_ok):
                            job = self.eng[self.next_eng]
                            self.next_eng += 1
                        elif main_ok:
                            job = self.main[self.next_main]
                            self.next_main += 1
                            self.inflight += job.buf_size
                            self.peak_inflight = max(self.peak_inflight, self.inflight)
                        elif self.next_main >= len(self.main) and not eng_ok:
                            return
                        else:
                            self.cv.wait()
                if isinstance(job, _Item):
                    t = self._read_item(job, fds)
                    with self.cv:
                        job.tensor = t
                        job.done = True
                        self.read_bytes += job.r1 - job.r0
                        self.cv.notify_all()
                else:
                    self._read_eng(job, fds)
                    with self.cv:
                        job.done = True
                        self.eng_done += 1
                        self.eng_bytes += job.n
                        self.cv.notify_all()
        except BaseException as exc:  # noqa: BLE001 - surfaced by the consumer
            with self.cv:
                if self.err is None:
                    self.err = exc
                self.cv.notify_all()
        finally:
            for fd in fds.values():
                try:
                    os.close(fd)
                except OSError:
                    pass


def maybe_stream(loader, model, model_config):
    """FastWeightStream for this load, or None (stock path). Called by DefaultModelLoader."""
    if not enabled():
        return None
    lc = loader.load_config
    why = None
    if not getattr(type(model), "_dsv41_fast_load_ok", False):
        why = f"{type(model).__name__} has no pre-sorted hook"
    elif lc.load_format not in ("auto", "hf", "safetensors"):
        why = f"load_format {lc.load_format}"
    elif lc.safetensors_load_strategy not in (None, "lazy"):
        why = f"safetensors_load_strategy {lc.safetensors_load_strategy}"
    elif (lc.model_loader_extra_config or {}).get("enable_multithread_load"):
        why = "enable_multithread_load"
    elif loader.local_expert_ids is not None:
        why = "EP weight filter"
    elif getattr(loader, "_encoder_only_lm_prefixes", None) is not None:
        why = "mm-encoder-only"
    elif tuple(getattr(model, "secondary_weights", ()) or ()):
        why = "secondary weights"
    if why is not None:
        logger.info("DSV41_FAST_LOAD: stock loader for this load (%s)", why)
        return None
    try:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )
        from vllm.model_executor.model_loader.weight_utils import (
            filter_safetensors_files_by_weight_name,
        )

        _, files, use_st, _ = loader._prepare_weights(
            model_config.model,
            None,
            model_config.revision,
            getattr(model, "fall_back_to_pt_during_load", True),
            getattr(model, "allow_patterns_overrides", None),
        )
        if not use_st:
            raise _Unsupported("checkpoint is not safetensors")
        is_unused = getattr(model, "is_unused_checkpoint_weight", None)
        if is_unused is not None:
            files = filter_safetensors_files_by_weight_name(files, is_unused)
        stream = FastWeightStream(
            files,
            get_tensor_model_parallel_rank(),
            get_tensor_model_parallel_world_size(),
            lc.use_tqdm_on_load,
        )
    except Exception as exc:
        logger.warning("DSV41_FAST_LOAD: stock loader (setup failed: %s: %s)",
                       type(exc).__name__, exc)
        return None
    logger.info(
        "DSV41_FAST_LOAD armed: %d files, %d tensors in headers, tp %d/%d",
        len(stream.files), len(stream.entries), stream.tp_rank, stream.tp_size,
    )
    return stream


# ---- DSV41_WEIGHT_DIGEST: per-parameter sha256 after load ---------------------------------------
_DIGEST_N = [0]


def digest_model(model: torch.nn.Module, prefix: str = "") -> None:
    out_dir = os.path.expanduser(os.environ.get("DSV41_WEIGHT_DIGEST", ""))
    if not out_dir:
        return
    from concurrent.futures import ThreadPoolExecutor

    os.makedirs(out_dir, exist_ok=True)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    tag = os.environ.get("DSV41_WEIGHT_DIGEST_TAG", str(os.getpid()))
    n = _DIGEST_N[0]
    _DIGEST_N[0] += 1
    t0 = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    named = list(model.named_parameters())

    def one(item):
        name, p = item
        x = p.detach()
        if x.device.type == "meta":  # never materialised (e.g. the unused vision tower)
            return name, [str(p.dtype), list(p.shape), "meta", 0, "meta"]
        if not x.is_contiguous():
            x = x.contiguous()
        flat = x.reshape(-1).view(torch.uint8)
        h = hashlib.sha256()
        if flat.device.type == "cpu":
            h.update(memoryview(flat.numpy()))
        else:
            step = 64 << 20
            for a in range(0, flat.numel(), step):
                h.update(memoryview(flat[a:a + step].cpu().numpy()))
        return name, [str(p.dtype), list(p.shape), p.device.type, flat.numel(), h.hexdigest()]

    with ThreadPoolExecutor(max_workers=4) as ex:
        res = dict(ex.map(one, named))
    summ = hashlib.sha256()
    for name in sorted(res):
        summ.update(f"{name}:{res[name][4]};".encode())
    total = sum(v[3] for v in res.values())
    path = os.path.join(out_dir, f"wd-{tag}-r{rank}-{n}.json")
    with open(path + ".tmp", "w") as f:
        json.dump({"rank": rank, "load": n, "prefix": prefix, "model": type(model).__name__,
                   "n_params": len(res), "bytes": total, "summary": summ.hexdigest(),
                   "fast_load": enabled(), "params": res}, f)
    os.replace(path + ".tmp", path)
    logger.info("DSV41_WEIGHT_DIGEST: load %d (%s) %d params %.2f GiB summary %s in %.1f s -> %s",
                n, type(model).__name__, len(res), total / 2**30, summ.hexdigest()[:16],
                time.perf_counter() - t0, path)
