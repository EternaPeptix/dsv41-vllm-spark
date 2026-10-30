# SPDX-License-Identifier: Apache-2.0
# DSv4.1 local addition (2026-10-21): MiMo ibverbs one-/two-shot RDMA all-reduce and one-shot
# all-gather for decode-sized TP collectives on the 8x DGX Spark fabric.
#
# Opt-in: DSV41_RDMA_AR=1 (TP group only). With the flag unset, nothing here is imported.
# Env knobs (all optional):
#   RDMA_AR_EXT                path of rdma_ar_ext.so (default ~/rdma_ar_dsv41/build/rdma_ar_ext.so)
#   RDMA_AR_HCAS               RoCE devices, pair (i,j) uses hcas[(i+j) % n]
#                              (default rocep1s0f0,roceP2p1s0f1: one per PCIe link, as MiMo)
#   RDMA_AR_GID_INDEX          RoCEv2 GID index (default NCCL_IB_GID_INDEX or 5)
#   RDMA_AR_MAX_BYTES          slot size = largest all-gather input (default 2 MiB)
#   RDMA_AR_MAX_REDUCE_BYTES   largest all-reduce taken (default 512 KiB; NCCL above)
#   RDMA_AR_TWOSHOT_MIN_BYTES  all-reduces >= this use two-shot (default 32 KiB; 0 = never)
#   RDMA_AR_FP32               1 (default) also take fp32 all-reduces
#   DSV41_RDMA_AG              1 (default) also route all-gathers that fit
#   RDMA_AR_PROXY_CORE         CPU core for the proxy thread (default -1 = unpinned)
#   RDMA_AR_TIMEOUT_MS         GPU wait before trap (default 600000 here; MiMo used 60000)
#   RDMA_AR_AUDIT              N > 0: log the first N routed calls (op, shape, dtype, stream,
#                              capturing) and every new stream seen; use on the first serve.
#   (read by the .so itself: RDMA_AR_THP, RDMA_AR_VEC_PER_CTA, RDMA_AR_SPIN_US,
#    RDMA_AR_MAX_SLEEP_US, RDMA_AR_TIMING)
#
# Ordering contract: every rank must issue the same rdma ops in the same order on ONE stream
# (the kernel takes its sequence number from a single device counter). NCCL tolerates
# collectives on side streams; this does not, so calls made on vLLM's shared-experts aux stream
# are sent to NCCL, and RDMA_AR_AUDIT reports any other stream.
import importlib.util
import os

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.logger import init_logger

logger = init_logger(__name__)

_DEFAULT_EXT = os.path.join(
    os.path.expanduser("~"), "rdma_ar_dsv41", "build", "rdma_ar_ext.so"
)
_EXT = None


def rdma_ar_requested() -> bool:
    return os.environ.get("DSV41_RDMA_AR", "0") == "1"


def _load_ext():
    global _EXT
    if _EXT is None:
        path = os.environ.get("RDMA_AR_EXT", _DEFAULT_EXT)
        spec = importlib.util.spec_from_file_location("rdma_ar_ext", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"rdma_ar: cannot load {path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _EXT = mod
    return _EXT


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


class RdmaAllReduce:
    """Collective constructor: every rank of `cpu_group` must call it. All ranks end up
    enabled or all disabled (a partial enable would deadlock the first mixed collective)."""

    def __init__(self, cpu_group: ProcessGroup, rank: int, world: int, device: torch.device):
        self.disabled = True
        self.world = world
        self.rank = rank
        self.max_bytes = _env_int("RDMA_AR_MAX_BYTES", 2 * 1024 * 1024)
        self.max_reduce_bytes = min(
            self.max_bytes, _env_int("RDMA_AR_MAX_REDUCE_BYTES", 512 * 1024)
        )
        self.twoshot_min_bytes = _env_int("RDMA_AR_TWOSHOT_MIN_BYTES", 32 * 1024)
        self.allow_fp32 = os.environ.get("RDMA_AR_FP32", "1") == "1"
        self.use_gather = os.environ.get("DSV41_RDMA_AG", "1") == "1"
        self._audit_left = _env_int("RDMA_AR_AUDIT", 0)
        self._streams_seen: set[int] = set()
        self.h = None

        # phase 1: local resources (arena, HCAs, QPs in INIT)
        info = None
        err = ""
        try:
            ext = _load_ext()
            hcas = [
                h
                for h in os.environ.get("RDMA_AR_HCAS", "rocep1s0f0,roceP2p1s0f1").split(",")
                if h
            ]
            gid = _env_int("RDMA_AR_GID_INDEX", _env_int("NCCL_IB_GID_INDEX", 5))
            core = _env_int("RDMA_AR_PROXY_CORE", -1)
            timeout_ms = _env_int("RDMA_AR_TIMEOUT_MS", 600000)
            dev = device.index if device.index is not None else torch.cuda.current_device()
            self.h = ext.RdmaAR(
                rank, world, self.max_bytes, self.max_reduce_bytes, hcas, gid, 1, core,
                timeout_ms, dev,
            )
            info = self.h.local_info()
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
        infos: list = [None] * world
        dist.all_gather_object(infos, (info, err), group=cpu_group)
        bad = [(r, e) for r, (i, e) in enumerate(infos) if i is None]
        if bad:
            logger.warning("rdma_ar disabled on all ranks; setup failed on %s", bad)
            return

        # phase 2: connect QPs, start proxy
        ok = True
        try:
            self.h.connect([i for i, _ in infos])
        except Exception as e:  # noqa: BLE001
            ok = False
            err = f"{type(e).__name__}: {e}"
        oks: list = [None] * world
        dist.all_gather_object(oks, (ok, err), group=cpu_group)
        bad = [(r, e) for r, (o, e) in enumerate(oks) if not o]
        if bad:
            logger.warning("rdma_ar disabled on all ranks; connect failed on %s", bad)
            return
        dist.barrier(group=cpu_group)
        self.disabled = False
        arena = list(self.h.arena_info())
        logger.info(
            "rdma_ar enabled: world=%d max_reduce=%d twoshot>=%d max_gather=%d fp32=%s "
            "gather=%s arena=%s (bytes, thp_bytes, thp_path)",
            world, self.max_reduce_bytes, self.twoshot_min_bytes, self.max_bytes,
            self.allow_fp32, self.use_gather, arena,
        )

    # -- helpers ---------------------------------------------------------------------------
    @staticmethod
    def _on_aux_stream() -> bool:
        try:
            from vllm.utils import torch_utils

            aux = getattr(torch_utils, "_aux_stream", None)
        except Exception:  # noqa: BLE001
            return False
        return aux is not None and torch.cuda.current_stream().cuda_stream == aux.cuda_stream

    def _audit(self, op: str, x: torch.Tensor, extra: str = "") -> None:
        s = torch.cuda.current_stream().cuda_stream
        new = s not in self._streams_seen
        if new:
            self._streams_seen.add(s)
        if self._audit_left > 0 or new:
            if self._audit_left > 0:
                self._audit_left -= 1
            logger.info(
                "rdma_ar audit: %s shape=%s dtype=%s bytes=%d stream=%#x capturing=%s%s%s",
                op, tuple(x.shape), x.dtype, x.numel() * x.element_size(), s,
                torch.cuda.is_current_stream_capturing(), extra,
                " NEW_STREAM(total %d)" % len(self._streams_seen) if new else "",
            )

    # -- all-reduce ------------------------------------------------------------------------
    def should_use(self, x: torch.Tensor) -> bool:
        if self.disabled:
            return False
        nbytes = x.numel() * x.element_size()
        if not (
            (x.dtype == torch.bfloat16 or (self.allow_fp32 and x.dtype == torch.float32))
            and x.is_cuda
            and x.is_contiguous()
            and 0 < nbytes <= self.max_reduce_bytes
            and nbytes % 16 == 0
        ):
            return False
        return not self._on_aux_stream()

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        nbytes = x.numel() * x.element_size()
        two_shot = 0 < self.twoshot_min_bytes <= nbytes
        self._audit("all_reduce", x, " two_shot" if two_shot else "")
        return self.h.all_reduce(x, two_shot)

    # -- all-gather (dim 0 into [world, *x.shape]) ------------------------------------------
    def should_gather(self, x: torch.Tensor) -> bool:
        if self.disabled or not self.use_gather:
            return False
        nbytes = x.numel() * x.element_size()
        if not (x.is_cuda and x.is_contiguous() and 0 < nbytes <= self.max_bytes and nbytes % 16 == 0):
            return False
        return not self._on_aux_stream()

    def all_gather_stacked(self, x: torch.Tensor) -> torch.Tensor:
        """Returns [world, *x.shape]; out[r] = x of rank r."""
        self._audit("all_gather", x)
        out = torch.empty((self.world,) + tuple(x.shape), dtype=x.dtype, device=x.device)
        self.h.all_gather_into(out, x)
        return out

    def stats(self, reset: bool = False) -> list:
        return [] if self.h is None else list(self.h.stats(reset))
