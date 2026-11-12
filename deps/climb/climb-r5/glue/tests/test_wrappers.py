#!/usr/bin/env python3
"""r5-glue CPU test (torch CPU, no Triton, no GPU): the host-side wrappers' state machines.

The kit modules are imported with stub vllm / triton modules; their Triton kernels are replaced by Python fakes that
write deterministic bytes (a faithful fake of the r2 kernel, an identical "new" kernel, and must-fail mutants). Checked:
  * first eager call at each M runs the self-check (r2 + new on poisoned buffers), returns bytes == r2;
  * later calls / captures use the new kernel only after a passed check; a capture before any check uses r2;
  * a mismatching or raising new kernel DISABLES the lever for the process and r2 bytes are served from then on;
  * RQ: a new kernel that skips one padding row is caught through the 0xFF poison;
  * RP: scope M*6 <= 64 (M > 10 always r2), the warmup path compiles r2 as before and the new kernel additionally,
    a warmup failure of the new kernel disables RPSPLIT without touching r2, every output buffer is compared;
  * synthetic sweep (review N2): the first eager call runs r2 vs new at every M (RQ 1..32 on hostile partials, RP 1..10 x
    2 routings) before anything else; a mutant that is wrong ONLY at M=6 is caught although the first real call is at
    M=2 (production's profile-run size) and a later capture at M=6 then runs r2; the INFO line names the sweep;
  * invalid flag values raise at import.
Run on 512S1:  ~/exo-venv313/bin/python3 tests/test_wrappers.py
"""
import importlib.util
import os
import sys
import types

import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RQ = os.path.join(HERE, "files/vllm/models/deepseek_v4/nvidia/ops/dsv41_r5g_rq.py")
RPM = os.path.join(HERE, "files/vllm/model_executor/layers/fused_moe/dsv41_r5g_rp.py")
FAILS = []
CAPTURING = [False]
LOG = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name}{(': ' + str(detail)) if detail and not ok else ''}")
    if not ok:
        FAILS.append(name)


class FakeKernel:
    def __init__(self, fn):
        self.fn = fn
        self.launches = 0
        self.warmups = 0

    def __getitem__(self, grid):
        def run(*a, **kw):
            self.launches += 1
            return self.fn(grid, *a, **kw)
        return run

    def warmup(self, *a, grid=None, **kw):
        self.warmups += 1
        if getattr(self, "warm_raises", False):
            raise RuntimeError("ptxas failed (fake)")


class Logger:
    def info(self, *a):
        LOG.append(("info", a[0] % a[1:]))

    def error(self, *a):
        LOG.append(("error", a[0] % a[1:]))

    def warning(self, *a):
        LOG.append(("warning", a[0] % a[1:]))


def stub(name, **kw):
    m = types.ModuleType(name)
    m.__dict__.update(kw)
    sys.modules[name] = m
    return m


tl = types.SimpleNamespace(constexpr=lambda x=None: x, int32="i32", float32="f32", uint8="u8", int64="i64")
triton = types.SimpleNamespace(jit=lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f)),
                               next_power_of_2=lambda n: 1 << (int(n) - 1).bit_length(),
                               cdiv=lambda a, b: -(-a // b))
for n in ("vllm", "vllm.models", "vllm.models.deepseek_v4", "vllm.models.deepseek_v4.nvidia",
          "vllm.models.deepseek_v4.nvidia.ops", "vllm.model_executor", "vllm.model_executor.layers",
          "vllm.model_executor.layers.fused_moe"):
    stub(n)
stub("vllm.logger", init_logger=lambda name: Logger())
stub("vllm.triton_utils", tl=tl, triton=triton)
torch.cuda.is_current_stream_capturing = lambda: CAPTURING[0]


# ------------------------------------------------------------------------------------------------------- RQ fakes
def r2_rq(grid, partial, q, sf, m, n, splits, padded_m, sf_cols, inv448, use_pdl, *rest, **kw):
    """Fake r2 reduce-quant: q = f(partial), SF real rows = row+1, padding rows 0 (every byte written)."""
    z = partial.sum(0).reshape(m, n)
    q.copy_((z * 7).to(torch.int64).remainder(251).to(torch.uint8))
    sf2 = sf.view(padded_m, sf_cols)
    sf2[:m] = (torch.arange(m, dtype=torch.uint8) + 1)[:, None]
    sf2[m:] = 0


def new_rq_good(grid, partial, q, sf, m, n, splits, padded_m, sf_cols, inv448, use_pdl, pad_j, **kw):
    r2_rq(grid, partial, q, sf, m, n, splits, padded_m, sf_cols, inv448, use_pdl)


def new_rq_skip_last_pad(grid, partial, q, sf, m, n, splits, padded_m, sf_cols, inv448, use_pdl, pad_j, **kw):
    keep = sf.view(padded_m, sf_cols)[padded_m - 1].clone()
    r2_rq(grid, partial, q, sf, m, n, splits, padded_m, sf_cols, inv448, use_pdl)
    sf.view(padded_m, sf_cols)[padded_m - 1] = keep  # the last padding row is not written


def new_rq_raise(*a, **k):
    raise RuntimeError("launch failed (fake)")


def bits_equal(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.contiguous().view(torch.uint8),
                                                                      b.contiguous().view(torch.uint8))


def load_rq(newfn):
    r2k = FakeKernel(r2_rq)
    ao = stub("vllm.models.deepseek_v4.nvidia.ops.dsv41_attnout", _ASM_FP8="", _ASM_INV_SCALE="", _ASM_UE8M0="",
              INV448_BITS=0x3B124925, _r2_reduce_quant=r2k, _bits_equal=bits_equal,
              _first_diff=lambda a, b: "diff", _hostile_partials=lambda shape, gen: torch.randn(shape, generator=gen))

    def live_launch_rq(partial, m, n, splits):  # dsv41_attnout._launch_rq (verbatim structure)
        padded_m = (m + 127) // 128 * 128
        sf_cols = n // 32
        q = torch.empty((m, n), dtype=torch.uint8)
        sf = torch.empty((padded_m * sf_cols,), dtype=torch.uint8)
        ao._r2_reduce_quant[(n // 256, m)](partial, q, sf, m, n, splits, padded_m, sf_cols, ao.INV448_BITS, True)
        return q.view(torch.float8_e4m3fn), sf

    sys.modules["vllm.models.deepseek_v4.nvidia.ops"].dsv41_attnout = ao
    os.environ["DSV41_GLUE_RQPAD"] = "1"
    spec = importlib.util.spec_from_file_location("r5g_rq_under_test", RQ)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    nk = FakeKernel(newfn)
    m._r5g_reduce_quant = nk
    return m, m.wrap(live_launch_rq), r2k, nk, live_launch_rq


def test_rq():
    torch.manual_seed(0)
    n, splits = 1024, 32
    P = {M: torch.randn(splits, 1, M, n) for M in (1, 2, 4, 5, 6, 7, 16, 32)}
    # good kernel
    LOG.clear()
    m, w, r2k, nk, live = load_rq(new_rq_good)
    q, sf = w(P[1], 1, n, splits)
    qr, sfr = live(P[1], 1, n, splits)
    ns = len(m.SWEEP_M)
    check("RQ SWEEP_M is every sonar M 1..32", m.SWEEP_M == tuple(range(1, 33)))
    check("RQ first eager call M=1: sweep (32 M) + real-data self-check ran both kernels, bytes == r2",
          r2k.launches == ns + 2 and nk.launches == ns + 1 and bits_equal(q, qr) and bits_equal(sf, sfr)
          and m._ok is True and m._swept, (r2k.launches, nk.launches))
    check("RQ INFO line names the sweep: 'bit-identical ... at every M 1..32 (synthetic sweep'",
          any(k == "info" and "bit-identical to _r2_reduce_quant at every M 1..32 (synthetic sweep" in s
              for k, s in LOG), LOG)
    r2k.launches = nk.launches = 0
    q, sf = w(P[1], 1, n, splits)
    check("RQ second call M=1: new kernel only", r2k.launches == 0 and nk.launches == 1 and bits_equal(sf, sfr))
    CAPTURING[0] = True
    q, sf = w(P[7], 7, n, splits)
    CAPTURING[0] = False
    check("RQ capture at an unchecked M after a pass: new kernel, no check", nk.launches == 2 and r2k.launches == 0
          and 7 not in m._checked)
    w(P[2], 2, n, splits)
    check("RQ new eager M=2: checked once", 2 in m._checked and r2k.launches == 1)
    # capture before any check -> r2
    m, w, r2k, nk, live = load_rq(new_rq_good)
    CAPTURING[0] = True
    w(P[4], 4, n, splits)
    CAPTURING[0] = False
    check("RQ capture before any check: r2 kernel", r2k.launches == 1 and nk.launches == 0 and m._ok is None)
    # budget exhausted and no sweep ran (never passed) -> r2
    m._checks_left = 0
    m._swept = True
    w(P[4], 4, n, splits)
    check("RQ no check budget and never passed: r2 kernel", r2k.launches == 2 and nk.launches == 0)
    # must-fail: skipped padding row
    for name, fn in (("skips the last padding row", new_rq_skip_last_pad), ("raises", new_rq_raise)):
        for M in (1, 2, 4, 5, 6, 16, 32):
            LOG.clear()
            m, w, r2k, nk, live = load_rq(fn)
            q, sf = w(P[M], M, n, splits)
            qr, sfr = live(P[M], M, n, splits)
            ok = m._ok is False and bits_equal(sf, sfr) and bits_equal(q, qr) and any(
                k == "error" and "DISABLED" in s for k, s in LOG)
            r2k.launches = nk.launches = 0
            CAPTURING[0] = True
            w(P[M], M, n, splits)
            CAPTURING[0] = False
            ok = ok and nk.launches == 0 and r2k.launches == 1
            if not ok:
                break
        check(f"RQ mutant new kernel that {name}: caught at every M (1,2,4,5,6,16,32), disabled, r2 bytes served", ok)
    # review N2: a mutant wrong ONLY at M=6; the first real call is eager at M=2, M=6 is then served only in captures
    def new_rq_bad_at_6(grid, partial, q, sf, m, n, splits, padded_m, sf_cols, inv448, use_pdl, pad_j, **kw):
        r2_rq(grid, partial, q, sf, m, n, splits, padded_m, sf_cols, inv448, use_pdl)
        if m == 6:
            sf[-1] = 9
    LOG.clear()
    m, w, r2k, nk, live = load_rq(new_rq_bad_at_6)
    q, sf = w(P[2], 2, n, splits)
    qr, sfr = live(P[2], 2, n, splits)
    caught = m._ok is False and any(k == "error" and "synthetic sweep M=6" in s for k, s in LOG)
    r2k.launches = nk.launches = 0
    CAPTURING[0] = True
    w(P[6], 6, n, splits)
    CAPTURING[0] = False
    check("RQ mutant wrong only at M=6, first eager call at M=2: the sweep catches it at M=6, r2 bytes served, a "
          "later capture at M=6 runs r2", caught and bits_equal(sf, sfr) and bits_equal(q, qr)
          and nk.launches == 0 and r2k.launches == 1, LOG)


# ------------------------------------------------------------------------------------------------------- RP fakes
E, TOPK, H = 384, 6, 5120


def r2_rp_write(grid, x, xs, ids, i0, i1, out, os_, sfp, mn, layout, inv, v0, v1, ends, *rest, **kw):
    M = ids.shape[0]
    M_sum = layout.shape[0]
    layout.fill_(-1)
    inv.copy_(torch.arange(M * TOPK, dtype=torch.int32).view(M, TOPK))
    rows = min(M * TOPK, M_sum)
    out[:rows] = (x.repeat_interleave(TOPK, 0)[:rows].float() * 3).to(torch.int64).remainder(256).to(torch.uint8)
    sfp[: rows] = 7
    ends.copy_(torch.arange(E, dtype=torch.int32))


class RPState:
    mutate = None


def new_rp_write(grid, *a, **kw):
    r2_rp_write(grid, *a, **kw)
    if RPState.mutate == "layout":
        a[9][0] = 5
    elif RPState.mutate == "ends":
        a[13][E - 1] = 0
    elif RPState.mutate == "row-byte":
        a[5][0, H - 1] ^= 1
    elif RPState.mutate == "raise":
        raise RuntimeError("launch failed (fake)")


def load_rp(mode):
    r2k = FakeKernel(r2_rp_write)

    def pf_args(w1, sf, E_, unused):
        return 0, 1, 1, (unused, 0, 0, unused, 0, 0, 0)

    parent = stub("vllm.model_executor.layers.fused_moe.dsv41_routeprep", _E=E, _TOPK=TOPK, _ALIGN=32, _H=H,
                  _GROUP=128, _GB=8, _FILL_ROWS=64, _FC1_KB_BYTES=64, EARLY_TRIGGER=False, _PDL=True,
                  _pf_args=pf_args, _rp_prefetch_l2=None, _dsv41_rp_kernel=r2k)

    def live_launch(x, xs, ids, i0, i1, out_u8, os_, sfp, mn, layout, inv, v0, v1, ends, w1, sf, M, M_sum,
                    warmup=False):
        if warmup:
            r2k.warmup()
            return 0
        r2k[(M * TOPK,)](x, xs, ids, i0, i1, out_u8, os_, sfp, mn, layout, inv, v0, v1, ends)
        return 0

    sys.modules["vllm.model_executor.layers.fused_moe"].dsv41_routeprep = parent
    os.environ["DSV41_GLUE_RPSPLIT"] = mode
    spec = importlib.util.spec_from_file_location(f"r5g_rp_under_test_{mode}", RPM)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    nk = FakeKernel(new_rp_write)
    m._r5g_rp_kernel = nk
    return m, m.wrap(live_launch), r2k, nk, live_launch


def rp_bufs(M):
    M_sum = ((M * TOPK + min(M * TOPK, E) * 31) + 31) // 32 * 32
    mn = (M_sum + 3) // 4 * 4
    x = torch.randn(M, H).to(torch.bfloat16)
    ids = torch.randint(0, E, (M, TOPK), dtype=torch.int32)
    out = torch.zeros((M_sum, H), dtype=torch.uint8)
    sfp = torch.zeros((10 * mn,), dtype=torch.int32)
    layout = torch.zeros((M_sum,), dtype=torch.int32)
    inv = torch.zeros((M, TOPK), dtype=torch.int32)
    ends = torch.zeros((E,), dtype=torch.int32)
    return (x, H, ids, TOPK, 1, out, H, sfp, mn, layout, inv, TOPK, 1, ends, None, None, M, M_sum)


def test_rp():
    for mode in ("hoist", "split"):
        RPState.mutate = None
        LOG.clear()
        m, w, r2k, nk, live = load_rp(mode)
        check(f"RP {mode}: MODE/SPLIT parsed", m.MODE == mode and m.SPLIT == (mode == "split"))
        a = rp_bufs(4)
        w(*a)
        b = rp_bufs(4)
        b = (a[0], *b[1:2], a[2], *b[3:])
        live(*b)
        same = all(torch.equal(a[i], b[i]) for i in (5, 7, 9, 10, 13))
        ns = len(m.SWEEP_M) * len(m.SWEEP_ROUTINGS)
        check(f"RP {mode}: sweep covers M 1..10 x 2 routings", m.SWEEP_M == tuple(range(1, 11)) and ns == 20)
        check(f"RP {mode} first eager M=4: sweep (20) + real-data self-check (r2 + new on poisoned copies) then the "
              f"real launch, bytes == r2", same and m._ok is True and m._swept and r2k.launches == ns + 2
              and nk.launches == ns + 2, (r2k.launches, nk.launches))
        check(f"RP {mode}: INFO line names the sweep",
              any(k == "info" and f"DSV41_GLUE_RPSPLIT={mode}: _r5g_rp_kernel bit-identical to _dsv41_rp_kernel at "
                  "every M 1..10 (synthetic sweep" in s for k, s in LOG), LOG)
        grid_seen = []
        nk.fn = lambda grid, *aa, **kw: (grid_seen.append(grid), new_rp_write(grid, *aa, **kw))
        w(*rp_bufs(4))
        check(f"RP {mode}: launch grid {'(24, 5)' if mode == 'split' else '(24, 1)'}",
              grid_seen == [(24, 5 if mode == "split" else 1)], grid_seen)
        nk.fn = new_rp_write
        r2k.launches = nk.launches = 0
        w(*rp_bufs(11))
        check(f"RP {mode}: M=11 (M*6 > 64) always r2, no check", r2k.launches == 1 and nk.launches == 0
              and 11 not in m._checked)
        r2k.warmups = nk.warmups = 0
        w(*rp_bufs(1), warmup=True)
        w(*rp_bufs(16), warmup=True)
        check(f"RP {mode}: warmup compiles r2 as before (2) and the new kernel in scope only (1)",
              r2k.warmups == 2 and nk.warmups == 1)
        CAPTURING[0] = True
        r2k.launches = nk.launches = 0
        w(*rp_bufs(6))
        CAPTURING[0] = False
        check(f"RP {mode}: capture after a pass, unchecked M=6: new kernel, no check", nk.launches == 1
              and r2k.launches == 0)
        # capture before any check
        m, w, r2k, nk, live = load_rp(mode)
        CAPTURING[0] = True
        w(*rp_bufs(2))
        CAPTURING[0] = False
        check(f"RP {mode}: capture before any check -> r2", r2k.launches == 1 and nk.launches == 0)
        # warmup failure of the new kernel
        m, w, r2k, nk, live = load_rp(mode)
        nk.warm_raises = True
        LOG.clear()
        w(*rp_bufs(1), warmup=True)
        w(*rp_bufs(1))
        check(f"RP {mode}: new-kernel precompile failure -> disabled, r2 compiled and serving",
              m._ok is False and r2k.warmups == 1 and r2k.launches == 1 and nk.launches == 0
              and any(k == "error" and "precompile" in s for k, s in LOG))
        # must-fail mutants, every M in scope
        for mut in ("layout", "ends", "row-byte", "raise"):
            ok = True
            for M in range(1, 11):
                RPState.mutate = mut
                LOG.clear()
                m, w, r2k, nk, live = load_rp(mode)
                a = rp_bufs(M)
                w(*a)
                RPState.mutate = None
                b = rp_bufs(M)
                b = (a[0], *b[1:2], a[2], *b[3:])
                live(*b)
                served_r2 = all(torch.equal(a[i], b[i]) for i in (5, 7, 9, 10, 13))
                r2k.launches = nk.launches = 0
                CAPTURING[0] = True
                w(*rp_bufs(M))
                CAPTURING[0] = False
                ok = (m._ok is False and served_r2 and nk.launches == 0 and r2k.launches == 1
                      and any(k == "error" and "DISABLED" in s for k, s in LOG))
                if not ok:
                    break
            check(f"RP {mode} mutant '{mut}': caught at every M 1..10, disabled, r2 bytes served", ok, f"M={M}")
        RPState.mutate = None
        # review N2: a mutant wrong ONLY at M=6 (row byte flipped when inv_perm has 6 rows);
        # first real call eager at M=2, then a capture at M=6
        LOG.clear()
        m, w, r2k, nk, live = load_rp(mode)
        nk.fn = lambda grid, *aa, **kw: (new_rp_write(grid, *aa, **kw),
                                         aa[5].__setitem__((0, 0), aa[5][0, 0] ^ 1) if aa[10].shape[0] == 6 else None)
        a = rp_bufs(2)
        w(*a)
        b = rp_bufs(2)
        b = (a[0], *b[1:2], a[2], *b[3:])
        live(*b)
        served_r2 = all(torch.equal(a[i], b[i]) for i in (5, 7, 9, 10, 13))
        caught = m._ok is False and any(k == "error" and "synthetic sweep M=6" in s for k, s in LOG)
        r2k.launches = nk.launches = 0
        CAPTURING[0] = True
        w(*rp_bufs(6))
        CAPTURING[0] = False
        check(f"RP {mode} mutant wrong only at M=6, first eager call at M=2: the sweep catches it, r2 bytes served, a "
              f"later capture at M=6 runs r2", caught and served_r2 and nk.launches == 0 and r2k.launches == 1, LOG)


def test_flags():
    for var, path, bad in (("DSV41_GLUE_RQPAD", RQ, "2"), ("DSV41_GLUE_RPSPLIT", RPM, "1")):
        os.environ[var] = bad
        try:
            spec = importlib.util.spec_from_file_location("bad_flag", path)
            mm = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mm)
            raised = False
        except ValueError:
            raised = True
        check(f"{var}={bad!r} raises ValueError at import", raised)


if __name__ == "__main__":
    test_rq()
    test_rp()
    test_flags()
    print("ALL PASS" if not FAILS else f"FAILED ({len(FAILS)}): {FAILS}")
    sys.exit(1 if FAILS else 0)
