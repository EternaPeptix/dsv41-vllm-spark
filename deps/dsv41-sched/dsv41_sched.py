"""dsv41_sched: AsyncScheduler subclass that stops short requests starving behind
long chunked prefills (policy: dsv41_sched_policy.py).

Load (rank 0 only runs the scheduler; the file is copied to all ranks anyway):
  PYTHONPATH=$HOME/dsv41-sched  DSV41_EXTRA="--scheduler-cls dsv41_sched.DSV41Scheduler"
Params: env DSV41_SCHED_<KEY> (see DEFAULTS), optionally overridden live from the JSON
file named by DSV41_SCHED_CTL (testing only; polled every 0.5 s by mtime).
Stats: DSV41_SCHED_STATS (default /dev/shm/dsv41-sched-stats.json), rewritten <= every 2 s.
DSV41_SCHED_ENABLE=0 gives exact stock AsyncScheduler behaviour.
"""
import json
import os
import sys
import time
import weakref

from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler

from dsv41_sched_policy import DEFAULTS, decide_hold, decide_threshold

logger = init_logger("dsv41_sched")


def _env_params():
    p = dict(DEFAULTS)
    for k in p:
        v = os.environ.get("DSV41_SCHED_" + k.upper())
        if v not in (None, ""):
            try:
                p[k] = int(v)
            except ValueError:
                logger.warning("dsv41_sched: DSV41_SCHED_%s=%r is not an int", k.upper(), v)
    return p


class DSV41Scheduler(AsyncScheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._d_env = _env_params()
        self._d_p = dict(self._d_env)
        self._d_ctl = os.environ.get("DSV41_SCHED_CTL", "")
        self._d_ctl_mtime = None
        self._d_ctl_next = 0.0
        self._d_stats_path = os.environ.get("DSV41_SCHED_STATS", "/dev/shm/dsv41-sched-stats.json")
        self._d_stats_next = 0.0
        self._d_st = {"steps": 0, "short": 0, "dcap": 0, "emu": 0, "hold": 0, "probe": 0,
                      "probe_err": 0, "skip": 0, "hold_timed": 0, "presched": 0, "last_kind": "", "params": self._d_p, "pid": os.getpid()}
        self._d_block = int(getattr(self, "block_size", 64) or 64)
        self._d_hold_dec = {}     # id(scheduler_output) -> bool decision (hold=2)
        self._d_t_sched = {}      # id(scheduler_output) -> monotonic time it was scheduled
        self._d_last_big = 0.0    # duration estimate of the last big step (s)
        self._d_last_done = 0.0   # monotonic time the last big step's output was processed
        self._d_eng = None
        f = sys._getframe(1)
        while f is not None:
            s = f.f_locals.get("self")
            if s is not None and s is not self and any(
                c.__name__.startswith("EngineCore") for c in type(s).__mro__
            ):
                self._d_eng = weakref.ref(s)
                break
            f = f.f_back
        self._d_reload(force=True)
        logger.warning(
            "dsv41_sched armed: params=%s engine_ref=%s ctl=%s",
            self._d_p, self._d_eng is not None, self._d_ctl or "-",
        )

    # ---- params / stats -------------------------------------------------
    def _d_reload(self, force=False):
        if not self._d_ctl:
            return
        now = time.monotonic()
        if not force and now < self._d_ctl_next:
            return
        self._d_ctl_next = now + 0.5
        try:
            m = os.stat(self._d_ctl).st_mtime_ns
        except OSError:
            m = None
        if m == self._d_ctl_mtime:
            return
        self._d_ctl_mtime = m
        p = dict(self._d_env)
        if m is not None:
            try:
                with open(self._d_ctl) as fh:
                    for k, v in json.load(fh).items():
                        if k in p:
                            p[k] = int(v)
            except Exception as e:  # noqa: BLE001
                logger.warning("dsv41_sched: bad ctl file %s: %s", self._d_ctl, e)
                return
        self._d_p = p
        self._d_st["params"] = p
        logger.warning("dsv41_sched params now %s", p)

    def _d_stats(self):
        now = time.monotonic()
        if now < self._d_stats_next:
            return
        self._d_stats_next = now + 2.0
        try:
            tmp = self._d_stats_path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(dict(self._d_st, t=time.time()), fh)
            os.replace(tmp, self._d_stats_path)
        except OSError:
            pass

    # ---- state views ------------------------------------------------------
    def _d_running(self):
        out = []
        for r in self.running:
            pf = r.num_computed_tokens < r.num_prompt_tokens
            out.append((pf, r.num_prompt_tokens - r.num_computed_tokens if pf else 0))
        return out

    def _d_fresh(self, r):
        fresh = r.num_tokens - r.num_computed_tokens
        if (
            self._d_p["probe"]
            and fresh > self._d_p["short_max"]
            and r.num_computed_tokens == 0
        ):
            try:
                _, n_cached, _ = self.kv_cache_manager.get_computed_blocks(r)
                self._d_st["probe"] += 1
                fresh = r.num_tokens - n_cached
            except Exception as e:  # noqa: BLE001
                self._d_st["probe_err"] += 1
                if self._d_st["probe_err"] == 1:
                    logger.warning("dsv41_sched: prefix probe failed (%s); probe off", e)
                self._d_p["probe"] = 0
        return fresh

    # ---- scheduling -------------------------------------------------------
    def schedule(self, throttle_prefills: bool = False):
        self._d_reload()
        p = self._d_p
        if not p["enable"] and p["emu_threshold"] <= 0:
            return super().schedule(throttle_prefills)
        self._d_st["steps"] += 1
        thr, kind = 0, ""
        if p["emu_threshold"] > 0:
            thr, kind = p["emu_threshold"], "emu"
        else:
            running = self._d_running()
            if any(pf for pf, _ in running) and (len(self.waiting) or p["decode_cap"]):
                fresh = []
                if len(self.waiting):
                    for i, r in enumerate(self.waiting):
                        if i >= p["scan"]:
                            break
                        fresh.append(self._d_fresh(r))
                thr, kind = decide_threshold(running, fresh, p, self._d_block)
        if thr <= 0:
            out = super().schedule(throttle_prefills)
        else:
            cfg = self.scheduler_config
            old = cfg.long_prefill_token_threshold
            cfg.long_prefill_token_threshold = thr
            removed = []
            if kind == "short-skip":
                for i, r in enumerate(self.running):
                    if (r.num_computed_tokens < r.num_prompt_tokens
                            and r.num_prompt_tokens - r.num_computed_tokens > thr):
                        removed.append((i, r))
                for _, r in reversed(removed):
                    self.running.remove(r)
                self._d_st["skip"] += 1
            try:
                out = super().schedule(throttle_prefills)
            finally:
                cfg.long_prefill_token_threshold = old
                for i, r in removed:
                    self.running.insert(min(i, len(self.running)), r)
            if kind == "emu":
                self._d_st["emu"] += 1
            elif "short" in kind:
                self._d_st["short"] += 1
            else:
                self._d_st["dcap"] += 1
            self._d_st["last_kind"] = kind
        if out.total_num_scheduled_tokens >= p["hold_min"]:
            self._d_t_sched[id(out)] = time.monotonic()
        self._d_stats()
        return out

    def update_from_output(self, scheduler_output, model_runner_output):
        t0 = self._d_t_sched.pop(id(scheduler_output), None)
        self._d_hold_dec.pop(id(scheduler_output), None)
        if t0 is not None:
            now = time.monotonic()
            # GPU start of this step ~ max(its schedule time, previous big step's end)
            self._d_last_big = now - max(t0, self._d_last_done)
            self._d_last_done = now
        return super().update_from_output(scheduler_output, model_runner_output)

    def _d_timed_hold(self, eng, item):
        """hold=2: wait (polling) until a new request arrives (-> hold), the in-flight
        big step finishes (-> hold, nothing to overlap), or its predicted end minus the
        margin is reached (-> pre-schedule as stock). Decided once per in-flight step."""
        key = id(item[1])
        dec = self._d_hold_dec.get(key)
        if dec is not None:
            return dec
        if len(self.waiting) or not eng.input_queue.empty():
            dec = True
        else:
            t0 = self._d_t_sched.get(key)
            est = self._d_last_big
            dec = False
            if t0 is not None and est > 0:
                start = max(t0, self._d_last_done)
                deadline = start + est - self._d_p["hold_margin_ms"] / 1000.0
                fut = item[0]
                done = getattr(fut, "done", None)
                while time.monotonic() < deadline:
                    if not eng.input_queue.empty():
                        dec = True
                        break
                    if done is not None and done():
                        dec = True
                        break
                    time.sleep(0.002)
        self._d_hold_dec[key] = dec
        self._d_st["hold_timed" if dec else "presched"] += 1
        return dec

    def has_requests(self) -> bool:
        base = super().has_requests()
        if not base or self._d_eng is None:
            return base
        p = self._d_p
        if not p["enable"] or not p["hold"] or p["emu_threshold"] > 0:
            return base
        eng = self._d_eng()
        bq = getattr(eng, "batch_queue", None) if eng is not None else None
        if not bq:
            return base
        inflight = [item[1].total_num_scheduled_tokens for item in list(bq)]
        head = None
        if len(self.waiting):
            r = next(iter(self.waiting), None)
            if r is not None:
                head = r.num_tokens - r.num_computed_tokens
        room = len(self.running) < getattr(self, "max_num_active_reqs", 1 << 30)
        if p["hold"] == 2:
            if len(bq) == 1 and inflight[0] >= p["hold_min"]:
                if self._d_timed_hold(eng, bq[0]):
                    self._d_st["hold"] += 1
                    return False
                return base
        if decide_hold(p, inflight, self._d_running(), head, room):
            self._d_st["hold"] += 1
            return False
        return base
