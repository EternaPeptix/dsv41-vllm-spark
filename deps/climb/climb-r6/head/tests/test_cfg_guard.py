#!/usr/bin/env python3
"""r6-head tuning-flag guards (stdlib only: a stub `torch` is used when torch is absent; runs on any CPU host).

The device sweeps (device_test.py --part bitwise) run one M per BM / BMX bucket 16, 32, 64, 128, refine warps fixed at
0, and non-default screen shapes with DQ2 only. These checks prove the module cannot serve a tuning value outside that
coverage:
  1. DSV41_R6_HEAD_REFINE_CFG with non-zero warps -> ValueError at import.
  2. DSV41_R6_HEAD_SCREEN_CFG with any non-zero field and SCREEN_DQ != 2 -> ValueError at import; with DQ2 it imports.
  3. Inside the swept buckets (BM / BMX <= SWEPT_MAX_B = 128) the configured screen shape / refine grid / spread serve;
     above them the live screen shape, the default grid (4 x SMs) and spread 1 serve. Refine BK/stages act at BMX 16
     only; warps is always the live XWARPS.
  4. Defaults (no tuning env) give the live values at every bucket.
  5. Must-fail: textual mutants of the module that drop each guard must be caught by the same checks.
Usage: python3 tests/test_cfg_guard.py   (exit 0 = all pass)
"""
import importlib.util
import os
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "files/vllm/models/deepseek_v41/nvidia/dsv41_head_r6.py")
FAILS, N = [], [0]
try:
    import torch  # noqa: F401
except ImportError:  # the module only needs torch inside functions these checks do not call
    sys.modules["torch"] = types.ModuleType("torch")

# the live constants the module reads from dsv41_r4_small (launcher defaults, KIT.json depends_on)
R4 = types.SimpleNamespace(SBN=64, SBK=128, SWARPS=4, SSTAGES=3, XBV=16, XBK=128, XWARPS=4, XSTAGES=3)
ENV_KEYS = ("DSV41_R6_HEAD_REFINE", "DSV41_R6_HEAD_SCREEN", "DSV41_R6_HEAD_REFINE_CFG", "DSV41_R6_HEAD_SCREEN_CFG",
            "DSV41_R6_HEAD_SCREEN_DQ", "DSV41_R6_HEAD_CAND_ROWS", "DSV41_R6_HEAD_SELFCHECK", "DSV41_R6_HEAD_STATS_EVERY")
_n = [0]


def check(name, ok, detail=""):
    N[0] += 1
    print(f"{'PASS' if ok else 'FAIL'} {name}{(': ' + str(detail)) if detail else ''}")
    if not ok:
        FAILS.append(name)
    return ok


def load(env, src=SRC):
    """Import a fresh copy of the module under `env` (only the r6 keys are touched). Returns the module or the
    exception raised at import."""
    saved = {k: os.environ.get(k) for k in ENV_KEYS}
    for k in ENV_KEYS:
        os.environ.pop(k, None)
    os.environ.update(env)
    try:
        _n[0] += 1
        spec = importlib.util.spec_from_file_location(f"r6guard_{_n[0]}", src)
        m = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(m)
        except Exception as e:  # noqa: BLE001
            return e
        m._S6["nsm"] = 48  # GB10: 48 SMs (the default grid is 4 x SMs = 192)
        return m
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def suite(src, report):
    """All guard checks against module source `src`; returns the list of failed check names (report=True prints)."""
    fails = []

    def c(name, ok, detail=""):
        if report:
            check(name, ok, detail)
        if not ok:
            fails.append(name)

    m = load({}, src)
    c("defaults import", not isinstance(m, Exception), repr(m)[:120])
    if isinstance(m, Exception):
        return fails
    c("SWEPT_MAX_B == 128 (largest bucket device_test sweeps; live CERT_MAX_M bucket)", m.SWEPT_MAX_B == 128)
    for B in (16, 32, 64, 128, 256):
        c(f"defaults: screen_cfg(BM={B}) == live 64,128,4,3", m.screen_cfg(B, R4) == (64, 128, 4, 3))
        c(f"defaults: refine_cfg(BMX={B}) == live 128,4,3", m.refine_cfg(B, R4) == (128, 4, 3))
        c(f"defaults: grid(BMX={B}) == 192, spread 1", m._grid_n(None, B) == 192 and m.refine_spread(B) is True)
    e = load({"DSV41_R6_HEAD_REFINE_CFG": "0,8,0,0,1"}, src)
    c("REFINE_CFG warps 8 -> ValueError at import", isinstance(e, ValueError) and "warps" in str(e), repr(e)[:120])
    e = load({"DSV41_R6_HEAD_REFINE_CFG": "256,4,4,96,0"}, src)
    c("REFINE_CFG warps 4 (non-zero, even == live) -> ValueError", isinstance(e, ValueError), repr(e)[:120])
    for cfg in ("128,256,8,4", "0,0,0,2", "32,0,0,0"):
        for dq in (None, "1"):
            env = {"DSV41_R6_HEAD_SCREEN_CFG": cfg}
            if dq:
                env["DSV41_R6_HEAD_SCREEN_DQ"] = dq
            e = load(env, src)
            c(f"SCREEN_CFG={cfg} DQ={dq or 'default(1)'} -> ValueError", isinstance(e, ValueError)
              and "SCREEN_DQ=2" in str(e), repr(e)[:120])
    m = load({"DSV41_R6_HEAD_SCREEN_CFG": "0,0,0,0", "DSV41_R6_HEAD_SCREEN_DQ": "1"}, src)
    c("SCREEN_CFG=0,0,0,0 with DQ1 imports", not isinstance(m, Exception), repr(m)[:120])
    m = load({"DSV41_R6_HEAD_SCREEN_CFG": "128,256,8,4", "DSV41_R6_HEAD_SCREEN_DQ": "2"}, src)
    c("SCREEN_CFG=128,256,8,4 with DQ2 imports", not isinstance(m, Exception), repr(m)[:120])
    if not isinstance(m, Exception):
        for B in (16, 32, 64, 128):
            c(f"swept BM={B}: configured screen shape serves", m.screen_cfg(B, R4) == (128, 256, 8, 4))
        c("BM=256 (outside the sweep): live screen shape serves", m.screen_cfg(256, R4) == (64, 128, 4, 3))
    m = load({"DSV41_R6_HEAD_SCREEN_CFG": "0,0,0,2", "DSV41_R6_HEAD_SCREEN_DQ": "2"}, src)
    if not isinstance(m, Exception):
        c("partial SCREEN_CFG 0,0,0,2: other fields live", m.screen_cfg(64, R4) == (64, 128, 4, 2))
    else:
        c("partial SCREEN_CFG 0,0,0,2 with DQ2 imports", False, repr(m)[:120])
    m = load({"DSV41_R6_HEAD_REFINE_CFG": "256,0,4,96,0"}, src)
    c("REFINE_CFG=256,0,4,96,0 imports", not isinstance(m, Exception), repr(m)[:120])
    if not isinstance(m, Exception):
        c("BMX 16: BK 256 / stages 4 / live warps", m.refine_cfg(16, R4) == (256, 4, 4))
        for B in (32, 64, 128):
            c(f"BMX {B}: live BK / stages, configured grid 96 and spread 0",
              m.refine_cfg(B, R4) == (128, 4, 3) and m._grid_n(None, B) == 96 and m.refine_spread(B) is False)
        c("BMX 256 (outside the sweep): default grid 192 and spread 1",
          m._grid_n(None, 256) == 192 and m.refine_spread(256) is True and m.refine_cfg(256, R4) == (128, 4, 3))
    e = load({"DSV41_R6_HEAD_REFINE": "rows"}, src)
    c("REFINE=rows -> ValueError", isinstance(e, ValueError), repr(e)[:120])
    return fails


def main():
    suite(SRC, report=True)
    src = open(SRC).read()
    mutants = {
        "no-warps-refusal": ("if R_WARPS:\n", "if False:\n"),
        "no-dq2-rule": ("if any((S_BN, S_BK, S_WARPS, S_STAGES)) and SCREEN_DQ != 2:", "if False:"),
        "screen-cfg-everywhere": ("    if BM > SWEPT_MAX_B:\n        return r4.SBN", "    if False:\n        return r4.SBN"),
        "grid-everywhere": ("if R_GRID and BMX <= SWEPT_MAX_B:", "if R_GRID:"),
        "spread-everywhere": ("return bool(R_SPREAD) if BMX <= SWEPT_MAX_B else True", "return bool(R_SPREAD)"),
        "warps-from-env": ("return bk, r4.XWARPS, st", "return bk, (R_WARPS or r4.XWARPS), st"),
        "bk-at-every-bmx": ("    if BMX <= 16:\n        bk = R_BK or bk", "    if True:\n        bk = R_BK or bk"),
    }
    d = tempfile.mkdtemp(prefix="r6guard-")
    for tag, (old, new) in mutants.items():
        if not check(f"mutant {tag}: applies exactly once", src.count(old) == 1, str(src.count(old))):
            continue
        p = os.path.join(d, f"mut_{tag}.py")
        open(p, "w").write(src.replace(old, new))
        if tag == "warps-from-env":  # also drop the import refusal, so the env value reaches refine_cfg
            open(p, "w").write(src.replace(old, new).replace("if R_WARPS:\n", "if False:\n"))
        mf = suite(p, report=False)
        check(f"MUTANT {tag} caught", bool(mf), f"{len(mf)} checks fail, e.g. {mf[:2]}")
    print(f"\n{N[0] - len(FAILS)}/{N[0]} PASS" + (f"; FAILED: {FAILS}" if FAILS else ""))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
