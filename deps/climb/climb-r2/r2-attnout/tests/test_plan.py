#!/usr/bin/env python3
"""CPU plan test for dsv41_attnout's host logic (torch CPU; triton/vllm stubbed, kernels replaced by recorders).

Checks: flag parsing; o_proj returns None (nothing launched) when sonar is off / nothing applies; one
try_project call per projection (L2-prefetch recorder sees the live call count); eager self-check order and
results; capture-before-check runs the live kernels; a bit mismatch disables the part, logs ERROR and uses the
live results; try_project=False falls back to inverse RoPE + torch.bmm with exactly one inverse-RoPE launch;
rq only on eligible layers / G=1 / splits=32.
Run on 512S1: ~/exo-venv313/bin/python3 tests/test_plan.py
"""
import importlib.util
import os
import sys
import types

import torch

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOD = os.path.join(KIT, "files/models/deepseek_v4/nvidia/ops/dsv41_attnout.py")


# ---- stubs ----------------------------------------------------------------------------------------------
class _Log:
    def __init__(self):
        self.records = []

    def _r(self, lvl, msg, *a):
        self.records.append((lvl, msg % a if a else msg))

    def info(self, m, *a):
        self._r("INFO", m, *a)

    def warning(self, m, *a):
        self._r("WARN", m, *a)

    def error(self, m, *a):
        self._r("ERROR", m, *a)


LOG = _Log()


def install_stubs():
    vllm = types.ModuleType("vllm")
    vllm.__path__ = []
    logger = types.ModuleType("vllm.logger")
    logger.init_logger = lambda name: LOG
    tu = types.ModuleType("vllm.triton_utils")

    class _Triton:
        @staticmethod
        def jit(fn=None, **kw):
            return (lambda f: f) if fn is None else fn

        @staticmethod
        def cdiv(a, b):
            return -(-a // b)

    tu.triton = _Triton()
    tu.tl = types.SimpleNamespace(constexpr=lambda v: v)  # module-level tl.constexpr(...) asm strings
    plat = types.ModuleType("vllm.platforms")
    plat.current_platform = types.SimpleNamespace(is_device_capability_family=lambda f: f == 120)
    for name, m in {"vllm": vllm, "vllm.logger": logger, "vllm.triton_utils": tu, "vllm.platforms": plat}.items():
        sys.modules[name] = m


def load(flag):
    os.environ["DSV41_ATTNOUT_OP"] = flag
    spec = importlib.util.spec_from_file_location(f"ao_{len(LOG.records)}_{flag}", MOD)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class FakeSonar:
    MAX_M = 32

    def __init__(self, enabled=True):
        self._en = enabled
        self.calls = 0

    def enabled(self):
        return self._en

    def _splits_for(self, m):
        return 32

    def try_project(self, o_proj_input, wo_a, grouped_weight, z, r2ao=None):
        self.calls += 1
        if not (1 <= o_proj_input.shape[0] <= self.MAX_M):
            return False
        return r2ao.run(o_proj_input, grouped_weight, None, z)


class Rec:
    def __init__(self, m, sonar, capturing=False, bad=None, rq_layer=True):
        self.ev = []
        self.cap = capturing
        m._sonar = lambda: sonar
        m._capturing = lambda: self.cap
        m._rq_layer_ok = lambda wo_b: rq_layer
        m._qa = lambda q, sf, mm, n: ("QA", q, sf)

        def inv(o, a):
            self.ev.append("inv_rope")
            return o.reshape(o.shape[0], 1, -1)

        def part(a, rope, w, s, P, mm, g, n, k, splits, fp8w, new):
            self.ev.append(("partial", "new" if new else "live", "rope" if rope is not None else "-"))
            P.fill_(1.0)
            if new and bad == "partial":
                P.view(-1)[7] = 2.0

        def red(P, z, mm, g, n, splits):
            self.ev.append("reduce")
            z.fill_(3.0)

        def rq(P, mm, n, splits):
            self.ev.append("rq")
            q = torch.zeros((mm, n), dtype=torch.uint8)
            if bad == "rq":
                q[0, 0] = 1
            return q, torch.zeros(128 * n // 32, dtype=torch.uint8)

        def lq(z2d):
            self.ev.append("flashinfer_quant")
            return torch.zeros(z2d.shape, dtype=torch.uint8), torch.zeros(128 * z2d.shape[1] // 32, dtype=torch.uint8)

        m._live_inv_rope = inv
        m._launch_partial = part
        m._launch_live_reduce = red
        m._launch_rq = rq
        m._live_quant = lq
        m._hostile_bf16 = lambda shape, gen: torch.zeros(shape, dtype=torch.bfloat16)
        m._hostile_partials = lambda shape, gen: torch.zeros(shape)


def call(m, M=4):
    o = torch.zeros((M, 8, 512), dtype=torch.bfloat16)
    pos = torch.zeros(M, dtype=torch.int64)
    cs = torch.zeros((64, 64), dtype=torch.float32)
    wo_a = types.SimpleNamespace(weight=torch.zeros((1024, 4096), dtype=torch.bfloat16))
    out = {}

    def wo_b(x):
        out["x"] = x
        return "Y"

    y = m.o_proj(o, pos, cs, wo_a, wo_b, n_groups=1, heads_per_group=8, nope_dim=448, rope_dim=64,
                 o_lora_rank=1024, quant_group_size=128, tma_aligned_scales=False)
    return y, out.get("x")


def main():
    install_stubs()
    m = load("0")
    assert m.PARTS == frozenset()
    for raw, want in [("rope", {"rope"}), ("1", {"rope", "rq"}), ("tune", {"st1", "vs"}),
                      ("rope, rq ,bn128", {"rope", "rq", "bn128"}), ("rope+tune", {"rope", "st1", "vs"})]:
        assert m.parse(raw) == frozenset(want), raw
    try:
        m.parse("rope,bogus")
        raise AssertionError("unknown part accepted")
    except ValueError:
        pass
    print("PASS parse")

    # sonar off -> None, nothing launched
    m = load("rope,rq")
    s = FakeSonar(enabled=False)
    r = Rec(m, s)
    y, _ = call(m)
    assert y is None and r.ev == [] and s.calls == 0
    print("PASS sonar off -> o_proj returns None before any launch")

    # rope, eager first call: synthetic check (4 M values: new+live partial each), then real call new + live ref
    m = load("rope")
    s = FakeSonar()
    r = Rec(m, s)
    y, x = call(m)
    assert y == "Y" and s.calls == 1, (y, s.calls)
    syn = [("partial", "new", "rope"), "inv_rope", ("partial", "live", "-")]
    want = ["inv_rope", ("partial", "new", "rope"), ("partial", "live", "-")] * 4 + [
        "inv_rope", ("partial", "new", "rope"), ("partial", "live", "-"), "reduce"]
    got = r.ev
    # synthetic and real check order: live inv-rope (ref A) first, then new, then live partial
    assert got == want, got
    assert m._ok["partial"] is True and x is not None and x.shape == (4, 1024)
    print("PASS rope eager: synthetic(4 M) + real self-check, one try_project, output = z")

    # capture after a pass: no checks, new partial, live reduce
    r.ev.clear()
    r.cap = True
    call(m)
    assert r.ev == [("partial", "new", "rope"), "reduce"], r.ev
    print("PASS rope in capture after a passed check: folded partial only, no inverse-RoPE launch")

    # capture BEFORE any check: live kernels
    m = load("rope")
    s = FakeSonar()
    r = Rec(m, s, capturing=True)
    call(m)
    assert r.ev == ["inv_rope", ("partial", "live", "-"), "reduce"], r.ev
    print("PASS rope first seen in capture -> live inverse RoPE + live partial")

    # mismatch -> disabled, ERROR, live results; next call live
    m = load("rope")
    s = FakeSonar()
    r = Rec(m, s, bad="partial")
    y, x = call(m)
    assert m._ok["partial"] is False
    assert any(l == "ERROR" and "DISABLED" in t for l, t in LOG.records)
    assert r.ev[-3:] == ["inv_rope", ("partial", "live", "-"), "reduce"], r.ev  # this call: live kernels
    r.ev.clear()
    y, x = call(m)
    assert y is None and r.ev == [], r.ev  # later calls: o_proj declines, the caller's live code runs
    print("PASS partial mismatch -> part disabled (ERROR logged), this call uses live kernels, later calls "
          "return None (live o_proj code)")

    # rq: eager check then QA output; capture uses rq only after pass
    m = load("rq")
    s = FakeSonar()
    r = Rec(m, s)
    y, x = call(m)
    assert x[0] == "QA" and m._ok["rq"] is True, x
    assert r.ev[0] == "inv_rope" and "flashinfer_quant" in r.ev
    r.ev.clear()
    r.cap = True
    y, x = call(m)
    assert r.ev == ["inv_rope", ("partial", "live", "-"), "rq"] and x[0] == "QA", r.ev
    print("PASS rq: eager self-check vs live reduce+flashinfer quant, then capture = live partial + rq -> QuantizedActivation")

    # rq mismatch -> z path
    m = load("rq")
    s = FakeSonar()
    r = Rec(m, s, bad="rq")
    y, x = call(m)
    assert m._ok["rq"] is False and isinstance(x, torch.Tensor)
    print("PASS rq mismatch -> disabled, wo_b gets the live z")

    # rq on an ineligible layer -> live reduce, wo_b gets z
    m = load("rq")
    s = FakeSonar()
    r = Rec(m, s, rq_layer=False)
    y, x = call(m)
    assert isinstance(x, torch.Tensor) and "rq" not in r.ev
    print("PASS rq on an ineligible wo_b layer -> live reduce + quant inside wo_b")

    # try_project False (M > MAX_M) with rope: fallback inverse RoPE once + bmm
    m = load("rope")
    s = FakeSonar()
    r = Rec(m, s)
    y, x = call(m, M=40)
    assert s.calls == 1 and r.ev == ["inv_rope"] and x.shape == (40, 1024), (s.calls, r.ev)
    print("PASS try_project=False -> one inverse-RoPE launch + torch.bmm (live fallback), one try_project call")

    # tune only, sonar on: live inverse RoPE first (same call as live), new partial
    m = load("tune")
    s = FakeSonar()
    r = Rec(m, s, capturing=False)
    m._checks_left = 0
    m._ok["partial"] = True
    call(m)
    assert r.ev == ["inv_rope", ("partial", "new", "-"), "reduce"], r.ev
    print("PASS tune: live inverse RoPE, retuned partial, live reduce")

    # review fix: eager call after the check budget is spent with a part never checked -> live kernels
    for flag, ok_key in (("rope", "partial"), ("tune", "partial"), ("rq", "rq")):
        m = load(flag)
        s = FakeSonar()
        r = Rec(m, s)
        m._checks_left = 0
        assert m._ok[ok_key] is None
        y, x = call(m)
        assert ("partial", "new", "rope") not in r.ev and ("partial", "new", "-") not in r.ev and "rq" not in r.ev, \
            (flag, r.ev)
        assert r.ev[-2:] == [("partial", "live", "-"), "reduce"] and isinstance(x, torch.Tensor), (flag, r.ev)
        assert m._ok[ok_key] is None
    print("PASS budget spent + part never checked (eager) -> live kernels, part stays unchecked (rope, tune, rq)")

    # review fix: an exception in the real-call self-check fails closed (part disabled, live result)
    def boom(*a, **k):
        raise RuntimeError("compile failed")

    m = load("rope")
    s = FakeSonar()
    r = Rec(m, s)
    m._synthetic_done = True  # the synthetic check is done; the real call is the first new launch
    live_part = m._launch_partial

    def part_boom(a, rope, w, sc, P, mm, g, n, k, splits, fp8w, new):
        if new:
            boom()
        live_part(a, rope, w, sc, P, mm, g, n, k, splits, fp8w, new)

    m._launch_partial = part_boom
    y, x = call(m)
    assert y == "Y" and m._ok["partial"] is False and isinstance(x, torch.Tensor), (y, m._ok)
    assert r.ev == ["inv_rope", ("partial", "live", "-"), "reduce"], r.ev
    assert any(l == "ERROR" and "raised" in t for l, t in LOG.records)
    m = load("rq")
    s = FakeSonar()
    r = Rec(m, s)
    m._synthetic_done = True
    m._launch_rq = boom
    y, x = call(m)
    assert y == "Y" and m._ok["rq"] is False and isinstance(x, torch.Tensor), (y, m._ok)
    assert r.ev[-2:] == [("partial", "live", "-"), "reduce"], r.ev
    print("PASS exception in a real-call self-check (partial or rq) -> part disabled, ERROR logged, live result")
    print("ALL PASS test_plan")


if __name__ == "__main__":
    main()
