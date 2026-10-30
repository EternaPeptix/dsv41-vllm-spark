# SPDX-License-Identifier: Apache-2.0
# DSv4.1 local addition (2026-10-25, rel-2 = round-2 #2 + #4): startup reliability helpers for
# GPUModelRunner.capture_model. Everything here is startup-only and off unless a flag says otherwise.
#
#   DSV41_CAPTURE_SYNC=1           (default 1 = Codex C02, live since 2026-10-24 18:09) device sync between
#                                  the decoder-replay capture and the DSpark capture, so DSpark's first eager
#                                  rdma_ar op cannot overlap the replay warmup still queued on its private
#                                  stream. 0 = pre-C02 order (test arms only).
#   DSV41_CAPTURE_WATCHDOG_S=N     (default 0 = off) fail fast: if the decoder-replay or the DSpark capture
#                                  phase runs longer than N s, faulthandler dumps every thread's stack to fd 2
#                                  (the rank log) and _exit(1)s the worker. Healthy phases take 1-2 s; without
#                                  this a rdma_ar desync spins until the 600 s device trap.
#   DSV41_CAPTURE_WATCHDOG_AV_S=N  (default 420 when the watchdog is on) budget for the AV profile phase
#                                  (healthy ~100 s full, ~20 s pinned).
#   DSV41_TEST_CAPTURE_SKEW=r:ms   test only: rank r sleeps ms before each decoder-replay graph (forces the
#                                  pre-C02 race window).
#   DSV41_TEST_CAPTURE_HANG=r      test only: rank r blocks at the start of the DSpark capture (exercises the
#                                  watchdog and the supervisor).
#   DSV41_AV_PROFILE=pinned        (default off) when DSV41_AV_COST_JSON covers this capture shape set, skip
#                                  the 5x AV cost profile: rank 0 decides and broadcasts, every rank runs one
#                                  untimed replay of each profiled size (JIT + workspace warmup), then the
#                                  unchanged set_cost_curves path prices the pinned curves. off/force/empty =
#                                  stock (profile, then the pin overrides inside _dsv41_calibrate_curves).
import faulthandler
import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_float(name: str, default: float) -> float:
    v = _env(name)
    try:
        return float(v) if v else default
    except ValueError:
        logger.warning("DSV41 rel: ignoring %s=%r (not a number)", name, v)
        return default


def _tp_rank() -> int:
    try:
        from vllm.distributed.parallel_state import get_tp_group

        return int(get_tp_group().rank_in_group)
    except Exception:  # noqa: BLE001
        return int(_env("RANK", "0") or 0)


# ---------------------------------------------------------------------------------------------
# C02 stream join + fail-fast watchdog
# ---------------------------------------------------------------------------------------------
def capture_sync_enabled() -> bool:
    return _env("DSV41_CAPTURE_SYNC", "1").lower() not in ("0", "off", "false", "no")


WATCHDOG_S = _env_float("DSV41_CAPTURE_WATCHDOG_S", 0.0)
WATCHDOG_AV_S = _env_float("DSV41_CAPTURE_WATCHDOG_AV_S", 420.0) if WATCHDOG_S > 0 else 0.0
_armed: dict[str, Any] = {"phase": None, "t": 0.0}


def wd_arm(phase: str, seconds: float | None = None) -> None:
    """(Re)arm the process-wide fail-fast timer for one capture phase. No-op when off.

    faulthandler's watchdog is a C thread: it fires even while the main thread is blocked inside
    a CUDA call, and exit=True ends the worker with _exit(1) right after the stack dump."""
    s = WATCHDOG_S if seconds is None else seconds
    if WATCHDOG_S <= 0 or s <= 0:
        return
    faulthandler.dump_traceback_later(s, repeat=False, file=2, exit=True)
    _armed["phase"], _armed["t"] = phase, time.monotonic()
    logger.info(
        "DSV41 capture watchdog: phase=%s armed for %.0f s on rank %d (on timeout: stack dump "
        "'Timeout (h:mm:ss)!' to the rank log, then worker _exit(1))", phase, s, _tp_rank(),
    )


def wd_cancel() -> None:
    if WATCHDOG_S <= 0 or _armed["phase"] is None:
        return
    faulthandler.cancel_dump_traceback_later()
    logger.info(
        "DSV41 capture watchdog: disarmed after phase=%s (%.1f s)",
        _armed["phase"], time.monotonic() - _armed["t"],
    )
    _armed["phase"] = None


def _test_rank_ms(name: str) -> tuple[int, float] | None:
    v = _env(name)
    if not v:
        return None
    try:
        r, _, ms = v.partition(":")
        return int(r), float(ms or 0)
    except ValueError:
        logger.warning("DSV41 rel: ignoring %s=%r (want rank[:ms])", name, v)
        return None


@contextmanager
def capture_skew_test(model_state: Any) -> Iterator[None]:
    """DSV41_TEST_CAPTURE_SKEW=r:ms (test only): delay rank r before each decoder-replay graph."""
    t = _test_rank_ms("DSV41_TEST_CAPTURE_SKEW")
    if t is None or t[0] != _tp_rank() or not hasattr(model_state, "_prepare_replay_batch"):
        yield
        return
    rank, ms = t
    orig = model_state._prepare_replay_batch
    n = [0]

    def delayed(*args: Any, **kwargs: Any) -> Any:
        n[0] += 1
        logger.warning("DSV41 TEST capture skew: rank %d sleeps %.0f ms before replay graph %d", rank, ms, n[0])
        time.sleep(ms / 1000.0)
        return orig(*args, **kwargs)

    model_state._prepare_replay_batch = delayed  # instance attribute shadows the method
    try:
        yield
    finally:
        try:
            del model_state._prepare_replay_batch
        except AttributeError:
            pass


def capture_hang_test() -> None:
    """DSV41_TEST_CAPTURE_HANG=r (test only): rank r blocks here (inside the armed DSpark phase)."""
    t = _test_rank_ms("DSV41_TEST_CAPTURE_HANG")
    if t is None or t[0] != _tp_rank():
        return
    logger.warning("DSV41 TEST capture hang: rank %d blocks at the start of the DSpark capture", t[0])
    while True:
        time.sleep(3600)


def xstream_summary() -> None:
    try:
        from vllm.distributed.device_communicators import rdma_all_reduce as _rar

        f = getattr(_rar, "xstream_summary", None)
        if f is not None:
            f("after capture")
    except Exception:  # noqa: BLE001 - diagnostics only
        pass


# ---------------------------------------------------------------------------------------------
# Relaunch diet arm P: skip the AV cost profile when a pin covers the shape set
# ---------------------------------------------------------------------------------------------
def av_profile_mode() -> str:
    v = _env("DSV41_AV_PROFILE").lower()
    return "pinned" if v in ("pinned", "1", "on", "true", "skip") else "profile"


def _tail_sizes(capture_sizes: list[int], max_num_tokens: int) -> set[int]:
    # Mirrors AdaptiveVerificationManager.batches_to_profile.
    size = capture_sizes[-1] if capture_sizes else 0
    tails: set[int] = set()
    if size:
        tails.add(min(size + size // 2, max_num_tokens))
        while size < max_num_tokens:
            size = min(size * 2, max_num_tokens)
            tails.add(size)
        tails -= set(capture_sizes)
    return tails


def pin_shape_key(capture_sizes: list[int], max_num_reqs: int, max_num_tokens: int, num_spec: int, tp: int) -> dict:
    return {
        "capture_sizes": sorted(int(x) for x in capture_sizes),
        "max_num_reqs": int(max_num_reqs),
        "max_num_batched_tokens": int(max_num_tokens),
        "num_speculative_steps": int(num_spec),
        "tp": int(tp),
    }


def check_pin(pin: dict, capture_sizes: list[int], max_num_reqs: int, max_num_tokens: int,
              num_spec: int, tp: int) -> tuple[bool, str]:
    """Pure check (unit-tested): does this pin's raw curve cover the current capture shape set?"""
    try:
        draft = [(int(n), float(ms)) for n, ms in pin.get("draft_ms") or []]
        verify = [(int(n), float(ms)) for n, ms in pin.get("verify_ms") or []]
    except (TypeError, ValueError) as e:
        return False, f"unreadable curves ({e})"
    if not draft or not verify:
        return False, "pin lacks draft_ms or verify_ms"
    if "raw_draft_ms" in pin and "draft_ms" in pin and "source" not in pin:
        return False, "looks like a DSV41_AV_DUMP_JSON dump (already calibrated); pin raw curves only"
    caps = sorted(int(x) for x in capture_sizes)
    if not caps:
        return False, "no captured sizes"
    limit = caps[-1]
    want = pin_shape_key(caps, max_num_reqs, max_num_tokens, num_spec, tp)
    have = pin.get("shape_key")
    if have is not None and have != want:
        return False, f"shape_key mismatch: pin {have} != live {want}"
    vx = [n for n, _ in verify]
    if len(set(vx)) != len(vx) or vx != sorted(vx):
        return False, "verify sizes not strictly increasing"
    if sorted(n for n in vx if n <= limit) != caps:
        return False, f"verify sizes <= {limit} differ from the captured sizes"
    above = {n for n in vx if n > limit}
    tails = _tail_sizes(caps, max_num_tokens)
    if not above or not above <= tails:
        return False, f"verify tail sizes {sorted(above)} not within this profile's tails {sorted(tails)}"
    dx = [n for n, _ in draft]
    if not set(dx) <= set(caps) or max(dx) > max_num_reqs:
        return False, f"draft sizes {dx} not within captured sizes <= max_num_reqs {max_num_reqs}"
    return True, f"covers {len(caps)} captured sizes (limit {limit}), tails {sorted(above)}, draft {dx}"


def av_pinned_curves(av: Any, capture_sizes: list[int]) -> tuple[list, list] | None:
    """Rank 0 decides from DSV41_AV_COST_JSON + the live shape set and broadcasts, so every rank
    takes the same branch (the dummy runs are collectives). None = run the stock profile."""
    if av_profile_mode() != "pinned":
        return None
    from vllm.distributed.parallel_state import get_tp_group

    tp = get_tp_group()
    decision: tuple | None = None
    if tp.rank_in_group == 0:
        path = _env("DSV41_AV_COST_JSON")
        reason = "DSV41_AV_COST_JSON unset"
        if path:
            try:
                with open(path) as f:
                    pin = json.load(f)
                rs = av.req_states
                ok, reason = check_pin(
                    pin, list(capture_sizes), rs.max_num_reqs, rs.max_num_batched_tokens,
                    av.num_speculative_steps, tp.world_size,
                )
                if ok:
                    decision = (
                        path,
                        [(int(n), float(ms)) for n, ms in pin["draft_ms"]],
                        [(int(n), float(ms)) for n, ms in pin["verify_ms"]],
                        reason,
                    )
            except Exception as e:  # noqa: BLE001 - a bad pin means "profile", never a crash
                reason = f"{type(e).__name__}: {e}"
        if decision is None:
            logger.info("DSV41 AV profile: pin miss (%s); running the full startup profile", reason)
    decision = tp.broadcast_object(decision, src=0)
    if decision is None:
        return None
    path, draft, verify, reason = decision
    logger.info(
        "DSV41 AV profile skipped (pin hit, DSV41_AV_PROFILE=pinned): %s %s; one untimed replay per size",
        path, reason,
    )
    return draft, verify


def av_warm_batches(av: Any, capture_sizes: list[int], replays: int | None = None) -> list[dict]:
    """First replay of every profiled size, with the stock kwargs (same seeds), so the hit path
    exercises each shape once (JIT, 16K workspace growth before lock_workspace) and
    batches_to_profile sets _cudagraph_limit as usual."""
    if replays is None:
        import sys

        replays = int(getattr(sys.modules[type(av).__module__], "_PROFILE_REPLAYS", 5))
    out = []
    for batch in av.batches_to_profile(list(capture_sizes)):
        if int(batch.get("cost_profile_seed", -1)) == int(batch["num_tokens"]) * replays:
            out.append(batch)
    return out
