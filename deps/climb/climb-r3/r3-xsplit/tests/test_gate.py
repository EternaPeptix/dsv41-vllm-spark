#!/usr/bin/env python3
"""CPU tests of the XSPLIT gate (files/flashinfer/mla/dsv41_xsplit.py) with torch on CPU.

The CUDA checks, the stream query and the library are mocked; the envelope logic, capacity rule,
capture rule, self-check and fail-closed behaviour run for real.
"""
import importlib.util
import os
import sys
import types

import torch

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILS = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def load_gate(env=None):
    for k in list(os.environ):
        if k.startswith("DSV41_ATTN_XSPLIT"):
            del os.environ[k]
    os.environ.update(env or {})
    spec = importlib.util.spec_from_file_location(
        "dsv41_xsplit_t", os.path.join(KIT, "files/flashinfer/mla/dsv41_xsplit.py"))
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    g._cuda = lambda t: True
    return g


class FakeLib:
    def __init__(self, mode="same"):
        self.mode = mode
        self.launches = 0

    def dsv41_xsplit_ws_bytes(self, cap):
        return 4096


def inputs(T=4, topkx=512, pbsx=64, sink=True, lengths=True):
    q = torch.zeros(T, 8, 512, dtype=torch.bfloat16)
    kv = torch.zeros(2, 64, 584, dtype=torch.uint8)
    kvx = torch.zeros(topkx // pbsx, pbsx, 584, dtype=torch.uint8)
    idx = torch.zeros(T, 128, dtype=torch.int32)
    idxx = torch.zeros(T, topkx, dtype=torch.int32)
    out = torch.zeros(T, 8, 512, dtype=torch.bfloat16)
    lse = torch.zeros(T, 16, dtype=torch.float32)[:, :8]
    ln = torch.zeros(T, dtype=torch.int32) if lengths else None
    lnx = torch.zeros(T, dtype=torch.int32) if lengths else None
    s = torch.zeros(8, dtype=torch.float32) if sink else None
    return dict(q=q, kv=kv, idx=idx, out=out, lse=lse, mt=1, ln=ln, sink=s, kvx=kvx, idxx=idxx,
                lnx=lnx)


def env_of(g, d):
    return g._envelope(d["q"], d["kv"], d["idx"], d["out"], d["lse"], d["mt"], d["ln"], d["sink"],
                       d["kvx"], d["idxx"], d["lnx"])


def main():
    g = load_gate()
    a = env_of(g, inputs())
    check(a is not None and a["nmax"] == 10 and a["pbsx"] == 64 and a["stride_kv"] == 64 * 584,
          "production dual64 shape accepted (nmax 10)")
    a = env_of(g, inputs(T=6, topkx=1024, pbsx=32))
    check(a is not None and a["nmax"] == 18 and a["pbsx"] == 32, "dual32 topkx 1024 accepted (nmax 18)")
    check(a["stride_lse"] == 16, "strided out_lse view keeps its row stride")
    d = inputs()
    d["kv"] = d["kv"].view(2, 1, 64, 584)
    check(env_of(g, d) is not None, "4D HND main cache accepted")
    bad = {
        "heads 16": lambda d: d.update(q=torch.zeros(4, 16, 512, dtype=torch.bfloat16),
                                       out=torch.zeros(4, 16, 512, dtype=torch.bfloat16)),
        "no lengths": lambda d: d.update(ln=None, lnx=None),
        "no extra cache": lambda d: d.update(kvx=None, idxx=None),
        "pbsx 2": lambda d: d.update(kvx=torch.zeros(256, 2, 584, dtype=torch.uint8)),
        "main pbs 32": lambda d: d.update(kv=torch.zeros(4, 32, 584, dtype=torch.uint8)),
        "T 65": lambda d: d.update(**{k: v for k, v in inputs(T=65).items()}),
        "T 0": lambda d: d.update(q=torch.zeros(0, 8, 512, dtype=torch.bfloat16)),
        "fp8 query": lambda d: d.update(q=torch.zeros(4, 8, 512, dtype=torch.float8_e4m3fn)),
        "model dsv3_2": lambda d: d.update(mt=0),
        "non-contig indices": lambda d: d.update(idx=torch.zeros(128, 4, dtype=torch.int32).t()),
        "topk 96": lambda d: d.update(idx=torch.zeros(4, 96, dtype=torch.int32)),
        "sink f16": lambda d: d.update(sink=torch.zeros(8, dtype=torch.float16)),
        "out wrong shape": lambda d: d.update(out=torch.zeros(4, 8, 448, dtype=torch.bfloat16)),
        "int64 lengths": lambda d: d.update(ln=torch.zeros(4, dtype=torch.int64)),
        "bf16 kv cache": lambda d: d.update(kv=torch.zeros(2, 64, 512, dtype=torch.bfloat16)),
    }
    for name, f in bad.items():
        d = inputs()
        f(d)
        try:
            r = env_of(g, d)
        except Exception as e:  # noqa: BLE001 - try_run wraps this; record it
            r = ("raised", repr(e))
        check(r is None or (isinstance(r, tuple) and r[0] == "raised"),
              f"envelope refuses: {name}")

    # try_run behaviour with mocks
    calls = {"live": 0, "xs": 0}

    class Mod:
        def sparse_mla_sm120_paged_attention(self, q, kv, idx, out, lse, *a):
            calls["live"] += 1
            out.fill_(1.0)
            lse.fill_(2.0)

    def run(g, d, capturing=False, xs_val=1.0, rc=0):
        torch.cuda.current_stream = lambda dev=None: types.SimpleNamespace(cuda_stream=1234)
        torch.cuda.is_current_stream_capturing = lambda: capturing
        torch.cuda.current_device = lambda: 0

        def fake_launch(lib, a, q, kv, idx, out, lse, *rest):
            calls["xs"] += 1
            out.fill_(xs_val)
            lse.fill_(2.0)
            return rc
        g._launch = fake_launch
        return g.try_run(Mod(), 3, d["q"], d["kv"], d["idx"], d["out"], d["lse"], 0.04, d["mt"],
                         d["ln"], d["sink"], d["kvx"], d["idxx"], d["lnx"])

    g = load_gate()
    g._load = lambda: FakeLib()
    d = inputs()
    calls.update(live=0, xs=0)
    r = run(g, d, capturing=True)
    check(r is False and calls["xs"] == 0 and g._stats["fallback_no_ws"] == 1 and not g._pools,
          "first call inside capture: no workspace yet, nothing allocated -> live kernel")
    r = run(g, inputs(T=64, topkx=4096))
    check(r is False and g._stats["fallback_cap"] == 1 and len(g._pools) == 1,
          "first eager call (profile-run shape, refused) still allocates the workspace once")
    g._stats["fallback_cap"] = 0
    r = run(g, d)
    check(r is True and calls == {"live": 1, "xs": 1} and g._stats["checks_ok"] == 1,
          "first eager call: self-check runs live + xsplit, equal -> routed")
    r = run(g, d)
    check(r is True and calls == {"live": 1, "xs": 2}, "checked shape: xsplit only")
    r = run(g, d, capturing=True)
    check(r is True and calls["xs"] == 3, "checked shape inside capture: routed into the graph")
    d6 = inputs(T=6, topkx=1024, pbsx=32)
    r = run(g, d6, capturing=True)
    check(r is False and g._stats["fallback_unchecked"] == 1,
          "unchecked shape inside capture -> live kernel kept")
    d64 = inputs(T=64, topkx=2048, pbsx=64)
    r = run(g, d64)
    check(r is False and g._stats["fallback_cap"] == 1, "T*nmax > cap -> live kernel")
    r = run(g, inputs(lengths=False))
    check(r is False and g._stats["fallback_envelope"] >= 1, "out of envelope -> live kernel")

    g = load_gate()
    g._load = lambda: FakeLib()
    d = inputs()
    try:
        run(g, d, xs_val=3.0)
        raised = False
    except RuntimeError as e:
        raised = "self-check failed" in str(e)
    check(raised and torch.all(d["out"] == 1.0), "mismatch: live result restored, RuntimeError")

    g = load_gate({"DSV41_ATTN_XSPLIT_ON_MISMATCH": "disable"})
    g._load = lambda: FakeLib()
    d = inputs()
    r = run(g, d, xs_val=3.0)
    check(r is True and g._disabled and torch.all(d["out"] == 1.0),
          "mismatch with ON_MISMATCH=disable: live result kept, gate disabled")
    r = run(g, d)
    check(r is False, "disabled gate never routes again")

    g = load_gate({"DSV41_ATTN_XSPLIT_LIB": "/nonexistent"})
    r = run(g, inputs())
    check(r is False and g._stats["fallback_lib"] == 1 and g._lib_failed,
          "missing library -> live kernel, logged once")

    print(f"\n{'ALL PASS' if not FAILS else 'FAILURES: ' + str(len(FAILS))}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
