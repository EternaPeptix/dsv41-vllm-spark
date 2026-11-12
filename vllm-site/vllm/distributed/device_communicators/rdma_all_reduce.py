# SPDX-License-Identifier: Apache-2.0
# DSv4.1 local addition (2026-10-21): MiMo ibverbs one-/two-shot RDMA all-reduce and one-shot
# all-gather for decode-sized TP collectives on the 8x DGX Spark fabric.
#
# Opt-in: DSV41_RDMA_AR=1 (TP group only). With the flag unset, nothing here is imported.
# Env knobs (all optional):
#   RDMA_AR_EXT                path of rdma_ar_ext.so (default ~/rdma_ar_dsv41/build/rdma_ar_ext.so);
#                              an explicit value always wins over DSV41_RDMA_PARLOAD
#   DSV41_RDMA_PARLOAD         k1/3-rdma-loads (default 0): 1 = load the parallel-load build
#                              ~/rdma_ar_dsv41/build-k1pl/rdma_ar_ext.so (all peer loads in flight
#                              before accumulating; bitwise identical results). 0 = the build above.
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
#   RDMA_AR_XSTREAM            rel-2 (default off): eager-only cross-stream guard. When an eager
#                              (non-capturing) call arrives on a different stream than the previous
#                              eager call, "detect" logs whether that stream still had work queued
#                              (the DSpark-capture race), "wait" also makes the new stream wait for
#                              it (stream.wait_stream; GPU-side, no host sync). Captures are skipped
#                              (a capturing stream must not wait on an uncaptured event).
#   RDMA_AR_FINGERPRINT        rel-2 (default 0): 1 = all ranks exchange a hash of the rdma-relevant
#                              env, this file, the .so and the l2 prefetch module at init and refuse
#                              to start on any mismatch (a per-rank difference in the rdma call
#                              sequence is a guaranteed seq desync = 600 s hang).
#   DSV41_RDMA_AG_COLS         g2 colltrim (a) (default 0): 1 = all-gathers along dim > 0 (the 40
#                              qkv_a shard gathers per step, logits / Engram / main_proj gathers) are
#                              written in place by the kernel: rank r's [outer, seg] slice lands in
#                              columns [r*seg, (r+1)*seg) of the final tensor, so the [world, ...]
#                              staging tensor and cuda_communicator's movedim/reshape copy disappear.
#                              Pure data movement (bitwise the same tensor). Loads the g2 build
#                              ~/rdma_ar_dsv41/build-g2ct/rdma_ar_ext.so (= the k1 parallel-load
#                              kernels, SASS-identical, + the column mode; DSV41_RDMA_PARLOAD picks
#                              parload vs legacy kernels inside it). Same wire protocol and op sequence.
#   (read by the .so itself: RDMA_AR_THP, RDMA_AR_VEC_PER_CTA, RDMA_AR_SPIN_US,
#    RDMA_AR_MAX_SLEEP_US, RDMA_AR_TIMING)
#   DSV41_RDMA_TRIM            climb r6 C1 = r3 G06 (default 0): 1 = the r6 .so runs ar_kernel_r3 (early posted_seq  # r6-comm
#                              read, seq advance at entry, single-CTA publish <= 24 KiB, single-CTA two-shot phase 2).  # r6-comm
#                              Needs RDMA_AR_EXT = the r6 .so and RDMA_AR_PUB_V1/V2/HYG_POLL=1. Bitwise identical.  # r6-comm
#   DSV41_RDMA_PDL             climb r6 C2 (default 0, needs TRIM): 1/in = ar_kernel_r3 launches with PDL (only  # r6-comm
#                              ctrl->posted_seq is read before griddepcontrol.wait); trig = also triggers its  # r6-comm
#                              dependents after the last flag wait. Scheduling only, bitwise identical.  # r6-comm
#   DSV41_RDMA_SENDBUF         r3 G26 (default 0, needs TRIM): registered send-buffer region; an input that  # r6-comm
#                              IS a send_buffer() is published without staging. BENCH ONLY: no producer uses it.  # r6-comm
#   DSV41_RDMA_PROJ            climb r7 HT13 (default 0, needs TRIM and the r7 .so): an all-reduce of a bf16  # r7-comm2
#                              [rows <= 6, 5120] tensor while dsv41_oblit has armed a direction d also writes  # r7-comm2
#                              y = _proj_rows(out, d) from the same kernel; dsv41_oblit.project(out, d) then  # r7-comm2
#                              returns y instead of launching _proj_rows. Same op order (gate/projgate.py).  # r7-comm2
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

# r4 l2-prefetch (DSV41_L2_PREFETCH=1, default off): side-stream L2 prefetch of the next weights
# while the collective spins. Off = this block does nothing and the hooks below are skipped.
_L2PF = None
_L2PF_AFTER = None  # g2 colltrim: post-collective fork hook (None with an older prefetch module)
if os.environ.get("DSV41_L2_PREFETCH", "0").strip().lower() in ("1", "on", "true"):
    try:
        from vllm.distributed.device_communicators import dsv41_l2_prefetch as _l2pf_mod

        if _l2pf_mod.init():
            _L2PF = _l2pf_mod
            if getattr(_l2pf_mod, "G2", False):
                _L2PF_AFTER = getattr(_l2pf_mod, "after_collective", None)
    except Exception as _e:  # noqa: BLE001 - the prefetch is optional; never block the collective
        logger.warning("DSV41_L2_PREFETCH: disabled (%s)", _e)

_DEFAULT_EXT = os.path.join(
    os.path.expanduser("~"), "rdma_ar_dsv41", "build", "rdma_ar_ext.so"
)
# k1/3-rdma-loads: parallel-load build, installed next to the live one (never over it)
_K1PL_EXT = os.path.join(
    os.path.expanduser("~"), "rdma_ar_dsv41", "build-k1pl", "rdma_ar_ext.so"
)
# g2 colltrim (a): k1 kernels + column all-gather, installed next to the live builds (never over them)
_G2_EXT = os.path.join(
    os.path.expanduser("~"), "rdma_ar_dsv41", "build-g2ct", "rdma_ar_ext.so"
)
_EXT = None
_EXT_PATH = None
_R3_ENV = {k: os.environ.get(k, "").strip().lower() for k in ("DSV41_RDMA_TRIM", "DSV41_RDMA_SENDBUF")}  # r6-comm
_R3_BAD = sorted(k for k, v in _R3_ENV.items() if v not in ("", "0", "off", "false", "no", "1", "on", "true", "yes"))  # r6-comm
if _R3_BAD:  # r6-comm
    raise RuntimeError(f"rdma_ar r6: {_R3_BAD} must be one of 0/1/off/on/false/true/no/yes")  # r6-comm
_R3_TRIM = _R3_ENV["DSV41_RDMA_TRIM"] in ("1", "on", "true", "yes")  # r6-comm
_R3_SENDBUF = _R3_ENV["DSV41_RDMA_SENDBUF"] in ("1", "on", "true", "yes")  # r6-comm
_R6_PDL_ENV = os.environ.get("DSV41_RDMA_PDL", "").strip().lower()  # r6-comm
if _R6_PDL_ENV not in ("", "0", "off", "false", "no", "1", "on", "true", "yes", "in", "trig"):  # r6-comm
    raise RuntimeError(f"rdma_ar r6: DSV41_RDMA_PDL={_R6_PDL_ENV!r} must be one of 0/1/off/on/false/true/no/yes/in/trig")  # r6-comm
_R6_PDL = _R6_PDL_ENV in ("1", "on", "true", "yes", "in", "trig")  # r6-comm
if (_R6_PDL or _R3_SENDBUF) and not _R3_TRIM:  # r6-comm
    raise RuntimeError("rdma_ar r6: DSV41_RDMA_PDL / DSV41_RDMA_SENDBUF need DSV41_RDMA_TRIM=1")  # r6-comm
_R3_ANY = _R3_TRIM or _R3_SENDBUF or _R6_PDL  # r6-comm
_R7_PROJ_ENV = os.environ.get("DSV41_RDMA_PROJ", "").strip().lower()  # r7-comm2
if _R7_PROJ_ENV not in ("", "0", "off", "false", "no", "1", "on", "true", "yes"):  # r7-comm2
    raise RuntimeError(f"rdma_ar r7: DSV41_RDMA_PROJ={_R7_PROJ_ENV!r} must be one of 0/1/off/on/false/true/no/yes")  # r7-comm2
_R7_PROJ = _R7_PROJ_ENV in ("1", "on", "true", "yes")  # r7-comm2
if _R7_PROJ and not _R3_TRIM:  # r7-comm2
    raise RuntimeError("rdma_ar r7: DSV41_RDMA_PROJ needs DSV41_RDMA_TRIM=1")  # r7-comm2
_R7_MAX_ROWS = 6  # r7-comm2
_R7 = {"d": None, "tag": None, "folded": 0, "taken": 0, "seg_folded": 0, "seg_taken": 0, "seg_rows": set(),  # r7-comm2
       "seg_first": None, "eager_logs": 0}  # r7-comm2
  # r7-comm2
  # r7-comm2
def r7_arm(d, rows=-1):  # r7-comm2
    """dsv41_oblit: the next all-reduces may fold the projection by direction d (an fp32 [5120] device tensor);  # r7-comm2
    rows = token rows of the projected tensor (the first arm of a forward names the forward's rows in the log)"""  # r7-comm2
    if _R7_PROJ:  # r7-comm2
        _R7["d"] = d  # r7-comm2
        if _R7["seg_first"] is None:  # r7-comm2
            _R7["seg_first"] = rows  # r7-comm2
  # r7-comm2
  # r7-comm2
def r7_disarm():  # r7-comm2
    """dsv41_oblit.project_head: end of the bound layers. Logs this forward's fold counts (review: the promote  # r7-comm2
    precondition is taken == 76 in every captured decode forward on every rank; folded - taken = wasted tails)"""  # r7-comm2
    _R7["d"] = None  # r7-comm2
    _R7["tag"] = None  # r7-comm2
    if not _R7_PROJ:  # r7-comm2
        return  # r7-comm2
    f, t, rows, fr = _R7["seg_folded"], _R7["seg_taken"], sorted(_R7["seg_rows"]), _R7["seg_first"]  # r7-comm2
    _R7["seg_folded"], _R7["seg_taken"], _R7["seg_rows"], _R7["seg_first"] = 0, 0, set(), None  # r7-comm2
    try:  # r7-comm2
        cap = bool(torch.cuda.is_current_stream_capturing())  # r7-comm2
    except Exception:  # noqa: BLE001  (CPU-only builds)  # r7-comm2
        cap = False  # r7-comm2
    if cap or (f and _R7["eager_logs"] < 8):  # r7-comm2
        _R7["eager_logs"] += 0 if cap else 1  # r7-comm2
        logger.info("rdma_ar-r7 fwd: capturing=%d fwd_rows=%s ar_rows=%s folded=%d taken=%d wasted=%d "  # r7-comm2
                    "(totals folded=%d taken=%d)", int(cap), fr, rows, f, t, f - t, _R7["folded"], _R7["taken"])  # r7-comm2
  # r7-comm2
  # r7-comm2
def r7_take(x, d):  # r7-comm2
    """y when x IS the output of the last all-reduce (same storage, size, unchanged version counter) and that  # r7-comm2
    all-reduce folded direction d (the same tensor object); else None. One use: the tag is consumed. The tag  # r7-comm2
    holds the output tensor itself, so its memory cannot be reused by another tensor while the tag lives."""  # r7-comm2
    t = _R7["tag"]  # r7-comm2
    _R7["tag"] = None  # r7-comm2
    if t is None or not _R7_PROJ:  # r7-comm2
        return None  # r7-comm2
    out, ver, y, dd = t  # r7-comm2
    if (dd is not d or x.dtype != torch.bfloat16 or not x.is_contiguous() or x.data_ptr() != out.data_ptr() or  # r7-comm2
            x.numel() != out.numel() or x.is_inference() or out.is_inference() or  # r7-comm2
            x._version != ver or out._version != ver):  # r7-comm2
        return None  # r7-comm2
    _R7["taken"] += 1  # r7-comm2
    _R7["seg_taken"] += 1  # r7-comm2
    return y  # r7-comm2


def rdma_ar_requested() -> bool:
    return os.environ.get("DSV41_RDMA_AR", "0") == "1"


def _parload_requested() -> bool:
    return os.environ.get("DSV41_RDMA_PARLOAD", "0").strip().lower() in ("1", "on", "true")


def _ag_cols_requested() -> bool:
    return os.environ.get("DSV41_RDMA_AG_COLS", "0").strip().lower() in ("1", "on", "true")


def _ext_path() -> str:
    explicit = os.environ.get("RDMA_AR_EXT", "")
    if explicit:
        return explicit
    if _ag_cols_requested():
        return _G2_EXT
    return _K1PL_EXT if _parload_requested() else _DEFAULT_EXT


def _load_ext():
    global _EXT, _EXT_PATH
    if _EXT is None:
        path = _ext_path()
        _EXT_PATH = path
        spec = importlib.util.spec_from_file_location("rdma_ar_ext", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"rdma_ar: cannot load {path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _EXT = mod
    return _EXT


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


# rel-2 cross-stream guard (RDMA_AR_XSTREAM=off|detect|wait); counters are process-wide.
_XS_MODE = os.environ.get("RDMA_AR_XSTREAM", "off").strip().lower()
if _XS_MODE not in ("detect", "wait"):
    _XS_MODE = "off"
_XS = {"switch": 0, "busy": 0, "wait": 0, "unknown": 0}


def xstream_summary(where: str = "") -> None:
    if _XS_MODE == "off":
        return
    logger.info(
        "rdma_ar xstream (%s, mode=%s): stream switches=%d busy=%d waits=%d unknown=%d",
        where, _XS_MODE, _XS["switch"], _XS["busy"], _XS["wait"], _XS["unknown"],
    )


def _file_sha(path: str) -> str:
    import hashlib

    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except OSError as e:
        return f"unreadable:{type(e).__name__}"


def _fingerprint(ext_path: str) -> str:
    """rel-2 RDMA_AR_FINGERPRINT: what must be identical on every rank for the rdma call sequence."""
    import hashlib

    keys = sorted(
        k for k in os.environ
        if k.startswith(("RDMA_AR_", "DSV41_RDMA", "DSV41_L2_PREFETCH", "DSV41_QKV_A_SHARD"))
    )
    parts = [f"{k}={os.environ[k]}" for k in keys]
    here = os.path.abspath(__file__)
    parts.append("self=" + _file_sha(here))
    parts.append("so=" + _file_sha(ext_path))
    parts.append("l2pf=" + _file_sha(os.path.join(os.path.dirname(here), "dsv41_l2_prefetch.py")))
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16] + " " + ";".join(parts)


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
        self._xs_last = None  # rel-2: stream of the previous eager (non-capturing) call
        self.h = None
        self.ag_cols = False  # g2 colltrim (a): set below once the extension is up

        # rel-2 RDMA_AR_FINGERPRINT=1: refuse to start when ranks differ (exchanged before phase 1)
        if os.environ.get("RDMA_AR_FINGERPRINT", "0") == "1":
            fp = _fingerprint(_ext_path())
            fps: list = [None] * world
            dist.all_gather_object(fps, fp, group=cpu_group)
            heads = [f.split(" ", 1)[0] for f in fps]
            if len(set(heads)) != 1:
                mine = fp.split(" ", 1)[1].split(";")
                diff = {}
                for r, f in enumerate(fps):
                    other = f.split(" ", 1)[1].split(";")
                    d = sorted(set(other) ^ set(mine))
                    if d:
                        diff[r] = d[:6]
                raise RuntimeError(
                    f"rdma_ar fingerprint mismatch across ranks {heads}; rank {rank} differs from: {diff}"
                )
            logger.info("rdma_ar fingerprint %s identical on %d ranks", heads[0], world)

        # phase 1: local resources (arena, HCAs, QPs in INIT)
        info = None
        err = ""
        try:
            ext = _load_ext()
            if _R3_ANY and getattr(ext, "R3_VARIANT", None) is None:  # r6-comm
                raise RuntimeError(f"DSV41_RDMA_TRIM/SENDBUF/PDL set but {_EXT_PATH} has no r3 kernels")  # r6-comm
            if _R6_PDL and getattr(ext, "R6_VARIANT", None) is None:  # r6-comm
                raise RuntimeError(f"DSV41_RDMA_PDL set but {_EXT_PATH} has no r6 (PDL) kernels")  # r6-comm
            if _R7_PROJ and getattr(ext, "R7_VARIANT", None) is None:  # r7-comm2
                raise RuntimeError(f"DSV41_RDMA_PROJ set but {_EXT_PATH} has no r7 projection kernels")  # r7-comm2
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
            if _R3_ANY:  # r6-comm
                raise RuntimeError(f"rdma_ar r6: setup failed on {bad}; refusing to start without rdma_ar")  # r6-comm
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
            if _R3_ANY:  # r6-comm
                raise RuntimeError(f"rdma_ar r6: connect failed on {bad}; refusing to start without rdma_ar")  # r6-comm
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
        # g2 colltrim (a): the g2 build carries both kernel sets; keep DSV41_RDMA_PARLOAD's choice
        if _ag_cols_requested():
            if hasattr(self.h, "all_gather_cols"):
                if _EXT_PATH == _G2_EXT and hasattr(self.h, "set_parload"):
                    self.h.set_parload(_parload_requested())
                self.ag_cols = True
            else:  # identical on every rank (the fingerprint covers env + .so), so no partial enable
                logger.warning("DSV41_RDMA_AG_COLS=1 but %s has no all_gather_cols; column gathers off", _EXT_PATH)
            logger.info("rdma_ar g2 column all-gather: %s (variant %s)", "ON" if self.ag_cols else "off",
                        getattr(_EXT, "G2_VARIANT", None))
        # k1/3-rdma-loads: which build is live (the legacy .so has no parload())
        parload = self.h.parload() if hasattr(self.h, "parload") else None
        logger.info(
            "rdma_ar ext=%s variant=%s parload=%s (DSV41_RDMA_PARLOAD=%s)",
            _EXT_PATH, getattr(_EXT, "K1_VARIANT", "legacy"), parload,
            os.environ.get("DSV41_RDMA_PARLOAD", "0"),
        )
        if _R3_ANY:  # r6-comm
            logger.info("rdma_ar r6 %s %s: r3 [mask, s1_bytes, sendbuf_bytes, used, buffers] = %s, pdl bits = %s",  # r6-comm
                        getattr(_EXT, "R3_VARIANT", None), getattr(_EXT, "R6_VARIANT", None), list(self.h.r3()),  # r6-comm
                        self.h.r6() if hasattr(self.h, "r6") else None)  # r6-comm
        if _R7_PROJ:  # r7-comm2
            logger.info("rdma_ar r7 %s: projection fold on (bf16 [rows <= %d, 5120], DSV41_RDMA_PROJ), so r7 %s",  # r7-comm2
                        getattr(_EXT, "R7_VARIANT", None), _R7_MAX_ROWS, self.h.r7() if hasattr(self.h, "r7") else None)  # r7-comm2

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

    def _xstream_guard(self, op: str, x: torch.Tensor) -> None:
        """rel-2 RDMA_AR_XSTREAM: order eager calls that hop streams (see the header)."""
        if torch.cuda.is_current_stream_capturing():
            return
        cur = torch.cuda.current_stream()
        last = self._xs_last
        self._xs_last = cur
        if last is None or last.cuda_stream == cur.cuda_stream:
            return
        _XS["switch"] += 1
        try:
            busy = not last.query()
        except Exception:  # noqa: BLE001 - e.g. the old stream is capturing elsewhere
            busy = None
        if busy is None:
            _XS["unknown"] += 1
        elif busy:
            _XS["busy"] += 1
        waited = False
        if _XS_MODE == "wait" and busy is not False:
            try:
                cur.wait_stream(last)
                _XS["wait"] += 1
                waited = True
            except Exception as e:  # noqa: BLE001
                logger.warning("rdma_ar xstream: wait_stream failed (%s)", e)
        n = _XS["switch"]
        if n <= 16 or busy or n % 1000 == 0:
            logger.info(
                "rdma_ar xstream #%d (mode=%s): %s shape=%s from stream %#x to %#x; previous "
                "stream busy=%s%s", n, _XS_MODE, op, tuple(x.shape), last.cuda_stream,
                cur.cuda_stream, busy, " -> new stream waits for it" if waited else "",
            )

    def _r7_fold(self, x: torch.Tensor, two_shot: bool):  # r7-comm2
        d = _R7["d"]  # r7-comm2
        if (d is None or x.dtype != torch.bfloat16 or x.dim() != 2 or x.shape[-1] != 5120 or  # r7-comm2
                not (1 <= x.shape[0] <= _R7_MAX_ROWS) or not x.is_contiguous()):  # r7-comm2
            return None  # r7-comm2
        # Served under torch.inference_mode, whose tensors carry no version counter and may not be written  # r7-comm2
        # in place outside it: allocate y and the op's output (torch::empty_like) as normal tensors, so  # r7-comm2
        # r7_take's in-place guard works and no inference tensor is written. x is only read. Same values.  # r7-comm2
        with torch.inference_mode(False):  # r7-comm2
            y = torch.empty_like(x)  # r7-comm2
            out = self.h.all_reduce_proj(x, two_shot, d, y)  # r7-comm2
        _R7["tag"] = None if out.is_inference() else (out, out._version, y, d)  # r7-comm2
        _R7["folded"] += 1  # r7-comm2
        _R7["seg_folded"] += 1  # r7-comm2
        _R7["seg_rows"].add(int(x.shape[0]))  # r7-comm2
        return out  # r7-comm2
  # r7-comm2
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        nbytes = x.numel() * x.element_size()
        two_shot = 0 < self.twoshot_min_bytes <= nbytes
        if _XS_MODE != "off":
            self._xstream_guard("all_reduce", x)
        self._audit("all_reduce", x, " two_shot" if two_shot else "")
        if _L2PF is not None:
            _L2PF.before_collective("ar")
        if _R7_PROJ:  # r7-comm2
            _r7 = self._r7_fold(x, two_shot)  # r7-comm2
            if _r7 is not None:  # r7-comm2
                if _L2PF_AFTER is not None:  # r7-comm2
                    _L2PF_AFTER("ar")  # r7-comm2
                return _r7  # r7-comm2
            _R7["tag"] = None  # r7-comm2
        out = self.h.all_reduce(x, two_shot)
        if _L2PF_AFTER is not None:
            _L2PF_AFTER("ar")
        return out

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
        if _XS_MODE != "off":
            self._xstream_guard("all_gather", x)
        self._audit("all_gather", x)
        out = torch.empty((self.world,) + tuple(x.shape), dtype=x.dtype, device=x.device)
        if _L2PF is not None:
            _L2PF.before_collective("ag")
        self.h.all_gather_into(out, x)
        if _L2PF_AFTER is not None:
            _L2PF_AFTER("ag")
        return out

    def all_gather_cols(self, x: torch.Tensor, dim: int) -> torch.Tensor | None:
        """g2 colltrim (a): concat of every rank's x along `dim` (0 < dim < x.dim()) as one contiguous
        tensor, written in place by the kernel. None = not applicable (the caller then uses
        all_gather_stacked + movedim/reshape); checked before any side effect, so a None call issued
        nothing. Caller has already passed should_gather(x) and x is contiguous."""
        shape = tuple(x.shape)
        if not self.ag_cols or not 0 < dim < len(shape):
            return None
        seg = x.element_size()
        for d in shape[dim:]:
            seg *= d
        if seg <= 0 or seg % 16:
            return None
        if _XS_MODE != "off":
            self._xstream_guard("all_gather", x)
        self._audit("all_gather", x, " cols")
        out = torch.empty(shape[:dim] + (self.world * shape[dim],) + shape[dim + 1:], dtype=x.dtype,
                          device=x.device)
        if _L2PF is not None:
            _L2PF.before_collective("ag")
        self.h.all_gather_cols(out, x, seg)
        if _L2PF_AFTER is not None:
            _L2PF_AFTER("ag")
        return out

    def send_buffer(self, shape, dtype: torch.dtype):  # r6-comm
        """r3 G26: a tensor of `shape`/`dtype` in a registered send buffer (graph capture only, one buffer per  # r6-comm
        call, never reused by another call), or None. A collective whose input IS that tensor skips staging."""  # r6-comm
        if not _R3_SENDBUF or self.disabled or not torch.cuda.is_current_stream_capturing():  # r6-comm
            return None  # r6-comm
        n = torch.Size(shape).numel() * torch.empty((), dtype=dtype).element_size()  # r6-comm
        if n <= 0 or n % 16 or n > self.max_bytes:  # r6-comm
            return None  # r6-comm
        b = self.h.send_buffer(n)  # r6-comm
        return None if b is None else b.view(dtype).view(shape)  # r6-comm
  # r6-comm
    def stats(self, reset: bool = False) -> list:
        return [] if self.h is None else list(self.h.stats(reset))
