"""CPU tests for dsv41_rel.py (rel-2). Run: python3 -I test_dsv41_rel.py <dir with dsv41_rel.py> <pin.json>...
vllm is stubbed, so this runs anywhere with a stock Python 3.10+ (no torch, no GPU)."""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import types

here, pins = sys.argv[1], sys.argv[2:]

# --- stub vllm.logger ------------------------------------------------------------------------
vllm = types.ModuleType("vllm")
vlog = types.ModuleType("vllm.logger")


class _L:
    def __init__(self):
        self.lines = []

    def _log(self, *a):
        self.lines.append(a[0] % a[1:] if len(a) > 1 else a[0])

    info = warning = debug = error = _log


LOG = _L()
vlog.init_logger = lambda name: LOG
sys.modules["vllm"] = vllm
sys.modules["vllm.logger"] = vlog


def load(env=None):
    for k in list(os.environ):
        if k.startswith(("DSV41_CAPTURE", "DSV41_TEST_", "DSV41_AV_PROFILE")):
            del os.environ[k]
    os.environ.update(env or {})
    spec = importlib.util.spec_from_file_location("dsv41_rel", os.path.join(here, "dsv41_rel.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


fails = 0


def check(name, cond):
    global fails
    print(("PASS " if cond else "FAIL ") + name)
    fails += 0 if cond else 1


m = load()
check("defaults: sync on, watchdog off, AV profile stock", m.capture_sync_enabled() and m.WATCHDOG_S == 0
      and m.WATCHDOG_AV_S == 0 and m.av_profile_mode() == "profile")
m = load({"DSV41_CAPTURE_SYNC": "0", "DSV41_CAPTURE_WATCHDOG_S": "90", "DSV41_AV_PROFILE": "pinned"})
check("flags: sync off, watchdog 90/420, pinned", (not m.capture_sync_enabled()) and m.WATCHDOG_S == 90
      and m.WATCHDOG_AV_S == 420 and m.av_profile_mode() == "pinned")

# --- check_pin against the real pins -----------------------------------------------------------
for p in pins:
    pin = json.load(open(p))
    caps = [n for n, _ in pin["verify_ms"] if n <= 192]
    ok, why = m.check_pin(pin, caps, 32, 16384, 5, 8)
    check(f"{os.path.basename(p)} covers live shapes ({why})", ok)
    check("dropping a captured size misses", not m.check_pin(pin, caps[:-1], 32, 16384, 5, 8)[0])
    check("max_num_batched_tokens 8192 misses", not m.check_pin(pin, caps, 32, 8192, 5, 8)[0])
    check("max_num_seqs 16 misses", not m.check_pin(pin, caps, 16, 16384, 5, 8)[0])
    dump = {"raw_draft_ms": pin["draft_ms"], "raw_verify_ms": pin["verify_ms"],
            "draft_ms": pin["draft_ms"], "verify_ms": pin["verify_ms"]}
    check("calibrated DUMP file is refused", not m.check_pin(dump, caps, 32, 16384, 5, 8)[0])
    keyed = dict(pin, shape_key=m.pin_shape_key(caps, 32, 16384, 5, 8))
    check("matching shape_key hits", m.check_pin(keyed, caps, 32, 16384, 5, 8)[0])
    check("shape_key with other num_spec misses", not m.check_pin(keyed, caps, 32, 16384, 3, 8)[0])
    check("empty curves miss", not m.check_pin({"draft_ms": [], "verify_ms": pin["verify_ms"]}, caps, 32, 16384, 5, 8)[0])


# --- av_warm_batches == first replay of each stock profile size ---------------------------------
class FakeReq:
    max_num_batched_tokens = 16384
    max_num_reqs = 32


class FakeAV:
    """Copy of AdaptiveVerificationManager.batches_to_profile (7e30eebc) with _PROFILE_REPLAYS=5."""
    req_states = FakeReq()
    num_speculative_steps = 5
    _cudagraph_limit = 0

    def batches_to_profile(self, capture_sizes):
        max_num_tokens = self.req_states.max_num_batched_tokens
        size = self._cudagraph_limit = capture_sizes[-1] if capture_sizes else 0
        tail_sizes = set()
        if size:
            tail_sizes.add(min(size + size // 2, max_num_tokens))
            while size < max_num_tokens:
                size = min(size * 2, max_num_tokens)
                tail_sizes.add(size)
            tail_sizes -= set(capture_sizes)
        for num_tokens in capture_sizes + sorted(tail_sizes):
            for replay in range(5):
                yield {"num_tokens": num_tokens, "context_len": 0,
                       "cost_profile_seed": num_tokens * 5 + replay}


caps = [1, 2, 4, 6, 8, 12, 16, 24, 32] + list(range(40, 193, 8))
av = FakeAV()
full = list(av.batches_to_profile(caps))
av2 = FakeAV()
warm = m.av_warm_batches(av2, caps, 5)
check(f"stock profile = {len(full)} runs, warm = {len(warm)} runs (one per size)", len(full) == 5 * len(warm)
      and len(warm) == len(caps) + 8)
check("warm runs are the stock first replays, same order", warm == [b for b in full if b["cost_profile_seed"] % 5 == 0])
check("warm path sets _cudagraph_limit like the profile", av2._cudagraph_limit == 192)
check("tails mirror batches_to_profile", m._tail_sizes(caps, 16384) == {288, 384, 768, 1536, 3072, 6144, 12288, 16384})

# --- skew hook wraps then restores the method ---------------------------------------------------
os.environ["RANK"] = "0"
m = load({"DSV41_TEST_CAPTURE_SKEW": "0:50"})


class MS:
    def _prepare_replay_batch(self, x):
        return x * 2


ms = MS()
t = time.monotonic()
with m.capture_skew_test(ms):
    r = ms._prepare_replay_batch(3)
dt = time.monotonic() - t
check(f"skew hook on rank 0 delays ({dt*1000:.0f} ms) and passes through", r == 6 and dt >= 0.045)
check("skew hook restores the class method", "_prepare_replay_batch" not in ms.__dict__ and ms._prepare_replay_batch(2) == 4)
m = load({"DSV41_TEST_CAPTURE_SKEW": "3:50"})
t = time.monotonic()
with m.capture_skew_test(ms):
    ms._prepare_replay_batch(1)
check("skew hook inert on other ranks", time.monotonic() - t < 0.03 and "_prepare_replay_batch" not in ms.__dict__)
m = load({"DSV41_TEST_CAPTURE_HANG": "3"})
t = time.monotonic()
m.capture_hang_test()
check("hang hook inert on other ranks", time.monotonic() - t < 0.03)

# --- watchdog: arm/cancel is harmless; an expired timer dumps stacks and exits 1 ----------------
m = load({"DSV41_CAPTURE_WATCHDOG_S": "2"})
m.wd_arm("unit")
time.sleep(0.2)
m.wd_cancel()
time.sleep(2.2)
check("armed then cancelled watchdog does not fire", True)
child = f"""
import sys, types, os, time, importlib.util
v = types.ModuleType('vllm'); l = types.ModuleType('vllm.logger')
class L:
    def info(self, *a): print(a[0] % a[1:], file=sys.stderr, flush=True)
    warning = debug = error = info
l.init_logger = lambda n: L(); sys.modules['vllm'] = v; sys.modules['vllm.logger'] = l
os.environ['DSV41_CAPTURE_WATCHDOG_S'] = '1'
spec = importlib.util.spec_from_file_location('dsv41_rel', {os.path.join(here, 'dsv41_rel.py')!r})
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.wd_arm('dspark_capture')
time.sleep(30)
print('NOT REACHED', flush=True)
"""
t = time.monotonic()
p = subprocess.run([sys.executable, "-I", "-c", child], capture_output=True, text=True, timeout=60)
dt = time.monotonic() - t
check(f"expired watchdog: exit {p.returncode} after {dt:.1f} s, stack dump present",
      p.returncode == 1 and dt < 10 and "Timeout (0:00:01)!" in p.stderr and "NOT REACHED" not in p.stdout)
print("ALL PASS" if fails == 0 else f"{fails} FAILED")
sys.exit(1 if fails else 0)
