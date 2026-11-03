"""DSv4.1 activation steering (Modulo-style) for the dsv41 vLLM stack.

A request opts in with ``vllm_xargs``:

    {"steer": ["curiosity=8", "formality=-4"]}      # name=strength, strength in % of the
                                                    # layer's typical residual norm
    {"steer": "curiosity=8,formality=-4"}           # same, one string
    {"steer": ["random:1234=8"]}                    # random unit direction (control arm)
    {"steer_capture": "ID:START:END"}               # capture mode only (DSV41_STEER_CAPTURE=1):
                                                    # mean stream-averaged residual over prompt
                                                    # positions [START, END) -> CAPTURE_DIR/ID.npz

Steering is applied at one Engram layer (default 14), where the stack already
materialises the full hc stream: ``residual = engram(previous_post)``; the hook adds
``sum_k strength_k/100 * norm_L * unit_k`` to every hc copy of every steered row.
A row is steered when its position >= num_prompt_tokens - 1 (Modulo: from the last
prompt position onward), so prompt KV stays unsteered.

Exactness: rows of unsteered requests get an all-zero coefficient, and x + 0.0 == x
bitwise, so unsteered requests are bit-identical to the unpatched stack. In eager
forwards with no steered row the add is skipped entirely; in CUDA graphs the add is
captured once and replays with zero rows.

Vectors: DSV41_STEER_DIR/manifest.json
    {"layer": 14, "hidden": 5120, "layer_norm": 123.4,
     "vectors": [{"name": "curiosity", "file": "curiosity.npy", "norm": 123.4}, ...]}
Each .npy is a float32 [hidden] direction (normalised at load). The manifest is re-read
when its mtime changes (checked at most every DSV41_STEER_RELOAD_S seconds), so new
vectors can be published without a relaunch: write the .npy files first, then replace
manifest.json atomically, on all 8 nodes.
"""

from __future__ import annotations

import json
import logging
import os
import time
import zlib

import numpy as np
import torch

logger = logging.getLogger("vllm.dsv41_steer")


def _env_bool(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() not in ("0", "", "off", "false", "no")


ENABLED = _env_bool("DSV41_STEER", "1")
LAYER = int(os.environ.get("DSV41_STEER_LAYER", "14"))
DIR = os.path.expanduser(os.environ.get("DSV41_STEER_DIR", "~/dsv41-steer/vectors"))
SLOTS = int(os.environ.get("DSV41_STEER_SLOTS", "32"))
RANDOM_SLOTS = int(os.environ.get("DSV41_STEER_RANDOM_SLOTS", "4"))
MAX_ACTIVE = int(os.environ.get("DSV41_STEER_MAX_ACTIVE", "4"))
MAX_STRENGTH = float(os.environ.get("DSV41_STEER_MAX_STRENGTH", "15"))
RELOAD_S = float(os.environ.get("DSV41_STEER_RELOAD_S", "2"))
CAPTURE = _env_bool("DSV41_STEER_CAPTURE", "0")
CAPTURE_DIR = os.environ.get("DSV41_STEER_CAPTURE_DIR", "/dev/shm/dsv41-steer-capture")


class _Steer:
    def __init__(self) -> None:
        self.ready = False
        self.any_active = False  # some row of the current step has a non-zero coefficient
        self.dirty = 0  # rows[:dirty] may be non-zero
        self.names: dict[str, int] = {}
        self.layer_norm = 0.0
        self.manifest_mtime = -1.0
        self.next_check = 0.0
        self.random_slots: dict[int, int] = {}  # seed -> slot (LRU order)
        self.parsed: dict[str, tuple] = {}  # req_id -> (extra_args id, steer list, capture)
        self.reqs: dict[str, tuple] = {}  # V2 runner: req_id -> (extra_args, prompt_len)
        self.warned: set[str] = set()
        self.cap_specs: list[tuple[str, int, int, int]] = []
        self.cap_acc: dict[str, list] = {}
        self.T = 0
        self.N = 0  # full (padded) token count of the current forward
        self.req_idx: torch.Tensor | None = None
        self.pos: torch.Tensor | None = None

    # ---------------------------------------------------------------- setup
    def ensure(self, max_tokens: int, hidden: int, device: torch.device) -> None:
        if self.ready:
            return
        self.hidden = hidden
        self.device = device
        self.max_tokens = max_tokens
        self.rows = torch.zeros(max_tokens, SLOTS, dtype=torch.float32, device=device)
        self.bank = torch.zeros(SLOTS, hidden, dtype=torch.float32, device=device)
        self.coef_req = torch.zeros(1024, SLOTS, dtype=torch.float32, device=device)
        self.thr_req = torch.zeros(1024, dtype=torch.int64, device=device)
        self.cap_buf = (
            torch.zeros(max_tokens, hidden, dtype=torch.float32, device=device) if CAPTURE else None
        )
        self.ready = True
        self._maybe_reload(force=True)
        logger.warning(
            "DSV41_STEER armed: layer %d, %d slots (%d random), %d named vectors from %s, "
            "max %d active, |strength| <= %.1f, capture=%s",
            LAYER, SLOTS, RANDOM_SLOTS, len(self.names), DIR, MAX_ACTIVE, MAX_STRENGTH, CAPTURE,
        )

    def _maybe_reload(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now < self.next_check:
            return
        self.next_check = now + RELOAD_S
        path = os.path.join(DIR, "manifest.json")
        try:
            mtime = os.stat(path).st_mtime
        except OSError:
            return
        if mtime == self.manifest_mtime:
            return
        try:
            with open(path) as f:
                man = json.load(f)
            if int(man.get("layer", LAYER)) != LAYER:
                raise ValueError(f"manifest layer {man.get('layer')} != DSV41_STEER_LAYER {LAYER}")
            named = SLOTS - RANDOM_SLOTS
            vecs = man.get("vectors", [])[:named]
            bank = torch.zeros(SLOTS, self.hidden, dtype=torch.float32)
            names: dict[str, int] = {}
            for i, v in enumerate(vecs):
                arr = np.load(os.path.join(DIR, v["file"])).astype(np.float32).reshape(-1)
                if arr.shape[0] != self.hidden:
                    raise ValueError(f"{v['file']}: dim {arr.shape[0]} != {self.hidden}")
                t = torch.from_numpy(arr)
                n = float(t.norm())
                if not np.isfinite(n) or n == 0:
                    raise ValueError(f"{v['file']}: zero or non-finite")
                bank[i] = t / n * (float(v["norm"]) / 100.0)
                names[v["name"]] = i
            # keep live random slots as they are
            bank[named:] = self.bank[named:].cpu()
            self.bank.copy_(bank)
            self.names = names
            self.layer_norm = float(man.get("layer_norm", 0.0))
            self.manifest_mtime = mtime
            self.parsed.clear()
            logger.warning("DSV41_STEER loaded %d vectors: %s", len(names), ", ".join(names))
        except Exception as e:  # keep the old bank
            self.manifest_mtime = mtime
            logger.error("DSV41_STEER manifest %s rejected: %s", path, e)

    def _random_slot(self, seed: int) -> int | None:
        if self.layer_norm <= 0 or RANDOM_SLOTS <= 0:
            return None
        if seed in self.random_slots:
            slot = self.random_slots.pop(seed)
            self.random_slots[seed] = slot
            return slot
        base = SLOTS - RANDOM_SLOTS
        used = set(self.random_slots.values())
        free = [s for s in range(base, SLOTS) if s not in used]
        if free:
            slot = free[0]
        else:  # evict the least recently used seed
            old = next(iter(self.random_slots))
            slot = self.random_slots.pop(old)
        g = torch.Generator().manual_seed(seed & 0x7FFFFFFFFFFFFFFF)
        v = torch.randn(self.hidden, generator=g, dtype=torch.float32)
        self.bank[slot].copy_((v / v.norm() * (self.layer_norm / 100.0)).to(self.device))
        self.random_slots[seed] = slot
        return slot

    # --------------------------------------------------------------- parsing
    def _parse(self, req_id: str, extra: dict) -> tuple[list[tuple[int, float]], tuple | None]:
        cached = self.parsed.get(req_id)
        if cached is not None and cached[0] is extra:
            return cached[1], cached[2]
        spec = extra.get("steer")
        items: list[str] = []
        if isinstance(spec, str):
            items = [s for s in spec.split(",") if s.strip()]
        elif isinstance(spec, (list, tuple)):
            items = [str(s) for s in spec]
        coefs: dict[int, float] = {}
        for item in items:
            name, sep, val = item.strip().rpartition("=")
            if not sep:
                continue
            try:
                strength = float(val)
            except ValueError:
                continue
            if not np.isfinite(strength) or strength == 0.0:
                continue
            strength = max(-MAX_STRENGTH, min(MAX_STRENGTH, strength))
            name = name.strip()
            if name.startswith("random:"):
                try:
                    slot = self._random_slot(int(name[7:]))
                except ValueError:
                    slot = None
            else:
                slot = self.names.get(name)
            if slot is None:
                key = f"{req_id}:{name}"
                if key not in self.warned:
                    self.warned.add(key)
                    logger.warning("DSV41_STEER request %s: unknown vector %r ignored", req_id, name)
                continue
            if slot not in coefs and len(coefs) >= MAX_ACTIVE:
                continue
            coefs[slot] = strength
        cap = None
        if CAPTURE and isinstance(extra.get("steer_capture"), str):
            try:
                cid, s, e = extra["steer_capture"].rsplit(":", 2)
                if cid and "/" not in cid:
                    cap = (cid, int(s), int(e))
            except ValueError:
                cap = None
        out = list(coefs.items())
        self.parsed[req_id] = (extra, out, cap)
        if len(self.parsed) > 4096:
            self.parsed.clear()
        return out, cap

    # ---------------------------------------------------------- runner side
    # V2 model runner (vllm/v1/worker/gpu/model_runner.py): requests are registered here
    def add_request(self, req_id: str, sampling_params, prompt_len: int) -> None:
        extra = getattr(sampling_params, "extra_args", None) if sampling_params is not None else None
        if extra and ("steer" in extra or "steer_capture" in extra):
            self.reqs[req_id] = (extra, int(prompt_len))
        else:
            self.reqs.pop(req_id, None)

    def remove_request(self, req_id: str) -> None:
        self.reqs.pop(req_id, None)
        self.parsed.pop(req_id, None)
        self.cap_acc.pop(req_id, None)

    def prepare_v2(self, req_ids, total_tokens: int, query_start_loc: torch.Tensor,
                   positions: torch.Tensor) -> None:
        """After positions are final; query_start_loc[:num_reqs+1] maps tokens to batch rows."""
        if not self.ready:
            return
        self._maybe_reload()
        num_reqs = len(req_ids)
        steered = []
        caps = []
        if self.reqs:
            for i, rid in enumerate(req_ids):
                ent = self.reqs.get(rid)
                if ent is None:
                    continue
                coefs, cap = self._parse(rid, ent[0])
                if coefs:
                    steered.append((i, coefs, ent[1] - 1))
                if cap is not None:
                    caps.append((rid, i, cap[1], cap[2], cap[0]))
        T = total_tokens
        ri = None
        if steered or caps:
            qsl = query_start_loc[1 : num_reqs + 1]
            ri = torch.searchsorted(qsl, torch.arange(T, dtype=qsl.dtype, device=qsl.device), right=True)
        if steered:
            coef = np.zeros((num_reqs, SLOTS), dtype=np.float32)
            thr = np.full(num_reqs, 1 << 62, dtype=np.int64)
            for i, coefs, t in steered:
                for slot, st in coefs:
                    coef[i, slot] = st
                thr[i] = t
            self.coef_req[:num_reqs].copy_(torch.from_numpy(coef))
            self.thr_req[:num_reqs].copy_(torch.from_numpy(thr))
            on = (positions[:T] >= self.thr_req[ri]).to(torch.float32).unsqueeze(1)
            torch.mul(self.coef_req[ri], on, out=self.rows[:T])
            if self.dirty > T:
                self.rows[T : self.dirty].zero_()
            self.dirty = T
            self.any_active = True
        else:
            if self.dirty:
                self.rows[: self.dirty].zero_()
                self.dirty = 0
            self.any_active = False
        self.cap_specs = caps
        if caps:
            self.T = T
            self.req_idx = ri
            self.pos = positions[:T]

    def prepare(self, runner, num_reqs: int, total_tokens: int, req_idx: torch.Tensor,
                positions: torch.Tensor) -> None:
        """Called by the model runner once positions[:total_tokens] are final."""
        if not self.ready:
            return
        self._maybe_reload()
        steered: list[tuple[int, list, int]] = []
        caps: list[tuple[str, int, int, int]] = []
        reqs = runner.requests
        for i, rid in enumerate(runner.input_batch.req_ids[:num_reqs]):
            st = reqs.get(rid)
            sp = getattr(st, "sampling_params", None) if st is not None else None
            extra = getattr(sp, "extra_args", None) if sp is not None else None
            if not extra:
                continue
            coefs, cap = self._parse(rid, extra)
            if coefs:
                steered.append((i, coefs, int(st.num_prompt_tokens) - 1))
            if cap is not None:
                caps.append((rid, i, cap[1], cap[2], cap[0]))
        T = total_tokens
        if steered:
            coef = np.zeros((num_reqs, SLOTS), dtype=np.float32)
            thr = np.full(num_reqs, 1 << 62, dtype=np.int64)
            for i, coefs, t in steered:
                for slot, s in coefs:
                    coef[i, slot] = s
                thr[i] = t
            self.coef_req[:num_reqs].copy_(torch.from_numpy(coef), non_blocking=False)
            self.thr_req[:num_reqs].copy_(torch.from_numpy(thr), non_blocking=False)
            ri = req_idx[:T].long()
            on = (positions[:T] >= self.thr_req[ri]).to(torch.float32).unsqueeze(1)
            torch.mul(self.coef_req[ri], on, out=self.rows[:T])
            if self.dirty > T:
                self.rows[T : self.dirty].zero_()
            self.dirty = T
            self.any_active = True
        else:
            if self.dirty:
                self.rows[: self.dirty].zero_()
                self.dirty = 0
            self.any_active = False
        self.cap_specs = caps
        if caps:
            self.T = T
            self.req_idx = req_idx[:T]
            self.pos = positions[:T]

    def after_forward(self) -> None:
        """Capture mode: fold this step's rows into per-request sums (rank 0 writes)."""
        if not self.cap_specs or self.cap_buf is None:
            return
        from vllm.distributed import get_tensor_model_parallel_rank

        rank0 = get_tensor_model_parallel_rank() == 0
        buf = self.cap_buf[: self.T]
        for rid, i, s, e, cid in self.cap_specs:
            m = (self.req_idx == i) & (self.pos >= s) & (self.pos < e)
            rows = buf[m]
            acc = self.cap_acc.setdefault(rid, [np.zeros(self.hidden, np.float64), 0, 0.0, cid, s, e])
            if rows.shape[0]:
                acc[0] += rows.sum(0, dtype=torch.float64).cpu().numpy()
                acc[1] += int(rows.shape[0])
                acc[2] += float(rows.norm(dim=1).double().sum())
            if rank0:
                os.makedirs(CAPTURE_DIR, exist_ok=True)
                tmp = os.path.join(CAPTURE_DIR, f".{cid}.tmp.npz")
                np.savez(tmp, sum=acc[0], count=acc[1], norm_sum=acc[2], start=s, end=e,
                         layer=LAYER)
                os.replace(tmp, os.path.join(CAPTURE_DIR, f"{cid}.npz"))
        self.cap_specs = []
        if len(self.cap_acc) > 4096:
            self.cap_acc.clear()

    # ----------------------------------------------------------- model side
    def apply(self, residual: torch.Tensor) -> torch.Tensor:
        """residual: [rows, hc, hidden] at the steering layer; rows is either the full
        (graph-padded) batch of self.N tokens or this rank's sequence-parallel chunk."""
        n_full = self.N
        if not self.ready:
            self.ensure(_max_tokens_hint(n_full), residual.shape[-1], residual.device)
        rows_here = residual.shape[0]
        if rows_here == n_full:
            lo = 0
        else:  # sp_shard layout: ceil(N / tp) rows per rank, rank-major
            from vllm.distributed import get_tensor_model_parallel_rank

            lo = get_tensor_model_parallel_rank() * rows_here
        if lo + rows_here > self.rows.shape[0]:
            return residual
        if self.cap_buf is not None:
            full = residual
            if rows_here != n_full:
                from vllm.models.common.ops.sequence_parallel import sp_all_gather

                full = sp_all_gather(residual.contiguous())[:n_full]
            self.cap_buf[:n_full].copy_(full.float().mean(dim=1))
        if not self.any_active and not torch.cuda.is_current_stream_capturing():
            return residual
        rows = self.rows[lo : lo + rows_here]
        delta = torch.mm(rows, self.bank).to(residual.dtype).unsqueeze(1)
        # Unsteered rows keep their exact bits (x + 0.0 would turn -0.0 into +0.0).
        keep = (rows.abs().amax(dim=1) == 0).view(-1, 1, 1)
        return torch.where(keep, residual, residual + delta)


def _max_tokens_hint(num_tokens: int) -> int:
    try:
        from vllm.config import get_current_vllm_config

        sc = get_current_vllm_config().scheduler_config
        n = int(sc.max_num_batched_tokens)
    except Exception:
        n = 16384
    return max(n, num_tokens) + 1024


STEER: _Steer | None = _Steer() if ENABLED else None


def steer_hash(name: str) -> int:  # stable helper for clients that want seeds from names
    return zlib.crc32(name.encode())
