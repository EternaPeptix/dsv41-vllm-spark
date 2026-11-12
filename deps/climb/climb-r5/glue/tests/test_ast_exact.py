#!/usr/bin/env python3
"""r5-glue CPU test: the new kernels are the live kernels' statements, verbatim, except the parts the levers move.

Parses (no import, no Triton needed) the live r2 modules (base/ = byte copies of the live site files, sha-checked) and the
kit modules, and compares ast.dump of every statement that carries floating-point or index math:

RQPAD  _r5g_reduce_quant vs dsv41_attnout._r2_reduce_quant: same parameters (+ PAD_J), same decorator, every statement
       identical except the final padding loop; the kit launch helpers use the live launch expression.
RPSPLIT _r5g_rp_chunk body == the live `for ch in tl.static_range(...)` body (load moved out, preamble constants the
       same statements); plan, route, prefetch, dead-row and ends statements identical (guards may add `& is0`);
       _launch_new builds the same args tuple and constexpr dict (+ SPLIT, grid dim 1).
Exit 0 when every check passes.
"""
import ast
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIVE_AO = os.path.join(HERE, "base/dsv41_attnout.py")
LIVE_RP = os.path.join(HERE, "base/dsv41_routeprep.py")
KIT_RQ = os.path.join(HERE, "files/vllm/models/deepseek_v4/nvidia/ops/dsv41_r5g_rq.py")
KIT_RP = os.path.join(HERE, "files/vllm/model_executor/layers/fused_moe/dsv41_r5g_rp.py")
SHA = {LIVE_AO: "308942ebbed1fc121f74ea5b9820a941b7fed895edf7aae4d6d63480ba1d1bfc",
       LIVE_RP: "842f49fb5df830793e02c12ff0195fb853165e90fac35a16ff3f136cf7b0eda7"}
FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name}{(': ' + detail) if detail and not ok else ''}")
    if not ok:
        FAILS.append(name)


class Norm(ast.NodeTransformer):
    """_ao.X / _rp.X -> X (the kit reaches the live module's names through a module alias)."""

    def visit_Attribute(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id in ("_ao", "_rp"):
            return ast.copy_location(ast.Name(id=node.attr, ctx=node.ctx), node)
        return node


def d(node):
    return ast.dump(Norm().visit(ast.parse(ast.unparse(node))), annotate_fields=False)


def fn(tree, name):
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    raise KeyError(name)


def same(name, a, b):
    da = [d(x) for x in (a if isinstance(a, list) else [a])]
    db = [d(x) for x in (b if isinstance(b, list) else [b])]
    ok = da == db
    detail = ""
    if not ok:
        for i, (x, y) in enumerate(zip(da, db)):
            if x != y:
                detail = f"stmt {i}: {x[:160]} != {y[:160]}"
                break
        else:
            detail = f"lengths {len(da)} vs {len(db)}"
    check(name, ok, detail)


def params(f):
    return [(a.arg, ast.unparse(a.annotation) if a.annotation else None) for a in f.args.args]


def main(rq_src=None, rp_src=None):
    for p, s in SHA.items():
        check(f"base copy {os.path.basename(p)} is the live file (sha {s[:8]})",
              hashlib.sha256(open(p, "rb").read()).hexdigest() == s)
    ao = ast.parse(open(LIVE_AO).read())
    rq = ast.parse(rq_src if rq_src is not None else open(KIT_RQ).read())
    rpl = ast.parse(open(LIVE_RP).read())
    rpk = ast.parse(rp_src if rp_src is not None else open(KIT_RP).read())

    # ------------------------------------------------------------------ RQPAD
    L, K = fn(ao, "_r2_reduce_quant"), fn(rq, "_r5g_reduce_quant")
    check("RQ params = live + PAD_J: constexpr", params(K) == params(L) + [("PAD_J", "tl.constexpr")])
    same("RQ decorator triton.jit(do_not_specialize=['M'])", L.decorator_list, K.decorator_list)
    same("RQ every statement before the padding loop", L.body[:-1], K.body[:-2])
    lf = L.body[-1]
    check("RQ live tail is `for p in range(M + r, PADDED_M, M)`",
          isinstance(lf, ast.For) and ast.unparse(lf.iter) == "range(M + r, PADDED_M, M)")
    kf = K.body[-1]
    check("RQ kit tail = jj = tl.arange(0, PAD_J); for j0 in range(0, PADDED_M - M - r, PAD_J * M)",
          ast.unparse(K.body[-2]) == "jj = tl.arange(0, PAD_J)" and isinstance(kf, ast.For)
          and ast.unparse(kf.iter) == "range(0, PADDED_M - M - r, PAD_J * M)")
    check("RQ kit pad rows p = M + r + j0 + jj * M (the live rows M+r, M+r+M, ...; tests/test_writesets.py)",
          ast.unparse(kf.body[0]) == "p = M + r + j0 + jj * M")
    # the swizzle formula of a padding row is the live one with p a vector
    live_off = ast.unparse(lf.body[0].value)
    kit_off = ast.unparse(kf.body[1].value)
    check("RQ pad offset = live formula, vectorised",
          live_off == "sf_col + p % 32 * 16 + p % 128 // 32 * 4 + p // 128 * (128 * SF_COLS)"
          and kit_off == "sf_col[None, :] + (p % 32 * 16 + p % 128 // 32 * 4 + p // 128 * (128 * SF_COLS))[:, None]",
          f"{live_off} | {kit_off}")
    check("RQ pad value 0 (live tl.zeros([8], tl.uint8), kit tl.zeros([PAD_J, 8], tl.uint8))",
          ast.unparse(lf.body[1]) == "tl.store(SF + offp, tl.zeros([8], tl.uint8))"
          and ast.unparse(kf.body[2]) == "tl.store(SF + offp, tl.zeros([PAD_J, 8], tl.uint8), mask=(p < PADDED_M)[:, None])")
    # launches
    lr = fn(ao, "_launch_rq")
    live_launch = [s for s in lr.body if isinstance(s, ast.Expr)][0]
    r2into = fn(rq, "_launch_r2_into")
    same("RQ _launch_r2_into: padded_m / sf_cols / launch == live _launch_rq", lr.body[:2] + [live_launch],
         r2into.body[1:])
    newinto = fn(rq, "_launch_new_into")
    nl = Norm().visit(ast.parse(ast.unparse(newinto.body[-1])))
    call = nl.body[0].value
    ok = isinstance(call.func, ast.Subscript) and ast.unparse(call.func.value) == "_r5g_reduce_quant"
    if ok:
        call.func.value = ast.Name(id="_r2_reduce_quant", ctx=ast.Load())
        ok = ast.unparse(call.args[-1]) == "PAD_J"
        call.args = call.args[:-1]
    check("RQ _launch_new_into = live launch with the r5 kernel and + PAD_J", ok and d(nl.body[0]) == d(live_launch))
    same("RQ _alloc: the live allocations, same order", lr.body[:4], fn(rq, "_alloc").body[1:5])

    # ------------------------------------------------------------------ RPSPLIT
    L, K = fn(rpl, "_dsv41_rp_kernel"), fn(rpk, "_r5g_rp_kernel")
    check("RP params = live + SPLIT: constexpr", params(K) == params(L) + [("SPLIT", "tl.constexpr")])
    same("RP decorator", L.decorator_list, K.decorator_list)
    lb = {ast.unparse(s).split("\n")[0]: s for s in L.body}
    kb = {ast.unparse(s).split("\n")[0]: s for s in K.body}

    def stmt(tbl, head):
        if head not in tbl:
            raise KeyError(f"no statement starting {head!r}")
        return tbl[head]

    same("RP pid / nprog / gdc_wait", [stmt(lb, "pid = tl.program_id(0)"), stmt(lb, "nprog = tl.num_programs(0)"),
                                       stmt(lb, "if launch_pdl:")],
         [stmt(kb, "pid = tl.program_id(0)"), stmt(kb, "nprog = tl.num_programs(0)"), stmt(kb, "if launch_pdl:")])
    i0 = L.body.index(stmt(lb, "s = tl.arange(0, SLOTS)"))
    j0 = K.body.index(stmt(kb, "s = tl.arange(0, SLOTS)"))
    same("RP plan (s .. total)", L.body[i0:i0 + 10], K.body[j0:j0 + 10])
    i1 = L.body.index(stmt(lb, "e = tl.load(topk_ids + t * ids_stride0 + k * ids_stride1).to(tl.int32)"))
    j1 = K.body.index(stmt(kb, "e = tl.load(topk_ids + t * ids_stride0 + k * ids_stride1).to(tl.int32)"))
    same("RP route (e .. first)", L.body[i1:i1 + 9], K.body[j1:j1 + 9])
    same("RP t / k", [stmt(lb, "t = pid // TOPK"), stmt(lb, "k = pid % TOPK")],
         [stmt(kb, "t = pid // TOPK"), stmt(kb, "k = pid % TOPK")])
    lpf, kpf = stmt(lb, "if PF_KB > 0:"), stmt(kb, "if PF_KB > 0:")
    check("RP prefetch guard: live `if first:`, kit `if first & is0:`",
          ast.unparse(lpf.body[0].test) == "first" and ast.unparse(kpf.body[0].test) == "first & is0")
    same("RP prefetch body", lpf.body[0].body, kpf.body[0].body)
    same("RP early trigger", stmt(lb, "if EARLY_TRIGGER:"), stmt(kb, "if EARLY_TRIGGER:"))
    llive, klive = stmt(lb, "if live:"), stmt(kb, "if live:")
    check("RP kit: inv_perm / layout stores under `if is0:`", ast.unparse(klive.body[0].test) == "is0")
    same("RP inv_perm / layout stores", llive.body[0:2], klive.body[0].body)
    same("RP xrow (kit: hoisted to the top)", llive.body[2], stmt(kb, "xrow = x_ptr + t.to(tl.int64) * x_row_stride"))
    same("RP orow", llive.body[3], klive.body[1])
    loop = llive.body[-1]
    check("RP live chunk loop is `for ch in tl.static_range(H // (GB * GROUP))`",
          isinstance(loop, ast.For) and ast.unparse(loop.iter) == "tl.static_range(H // (GB * GROUP))")
    ch = fn(rpk, "_r5g_rp_chunk")
    same("RP chunk preamble g / c / col / fmax_v / one_v", llive.body[4:9], ch.body[0:5])
    same("RP chunk offs", loop.body[0], ch.body[5])
    check("RP live load `xf = tl.load(xrow + offs).to(tl.float32)`, kit `xf = xa.to(tl.float32)`",
          ast.unparse(loop.body[1]) == "xf = tl.load(xrow + offs).to(tl.float32)"
          and ast.unparse(ch.body[6]) == "xf = xa.to(tl.float32)")
    same("RP chunk body after the load (absmax .. packed-word store), statement by statement", loop.body[2:],
         ch.body[7:])
    offs_fn = fn(rpk, "_r5g_offs")
    check("RP _r5g_offs(ch) returns the live offs expression",
          ast.unparse(offs_fn.body[-1].value) == ast.unparse(loop.body[0].value)
          and [ast.unparse(s) for s in offs_fn.body[:2]] == ["g = tl.arange(0, GB)", "c = tl.arange(0, GROUP)"])
    ifs = stmt(kb, "if SPLIT:")
    hoist = [ast.unparse(s) for s in ifs.orelse]
    check("RP hoist: xa0..xa4 = tl.load(xrow + _r5g_offs(i, GB, GROUP)) in chunk order",
          hoist == [f"xa{i} = tl.load(xrow + _r5g_offs({i}, GB, GROUP))" for i in range(5)])
    check("RP split: xa = tl.load(xrow + _r5g_offs(y, GB, GROUP))",
          [ast.unparse(s) for s in ifs.body] == ["xa = tl.load(xrow + _r5g_offs(y, GB, GROUP))"])
    kch = klive.body[2]
    calls_h = [ast.unparse(s) for s in kch.orelse]
    args = "orow, sfp, mn_aligned, dest, eps, fp8_min, fp8_max, GB, GROUP"
    check("RP hoist: chunks quantized in the live order 0..4",
          calls_h == [f"_r5g_rp_chunk(xa{i}, {i}, {args})" for i in range(5)])
    check("RP split: one chunk per program (ch = program_id(1))",
          [ast.unparse(s) for s in kch.body] == [f"_r5g_rp_chunk(xa, y, {args})"]
          and ast.unparse(stmt(kb, "y = tl.program_id(1)")) == "y = tl.program_id(1)"
          and ast.unparse(stmt(kb, "is0 = y == 0")) == "is0 = y == 0")
    lfirst, kfirst = stmt(lb, "if first:"), stmt(kb, "if first & is0:")
    same("RP segment padding (kit guard first & is0)", lfirst.body, kfirst.body)
    kis0 = stmt(kb, "if is0:")
    same("RP strided dead-row fill (kit under is0; nprog = num_programs(0) unchanged)",
         [stmt(lb, "rr = tl.arange(0, FILL_ROWS)"),
          stmt(lb, "for r0 in range(total + pid * FILL_ROWS, M_sum, nprog * FILL_ROWS):")], kis0.body)
    same("RP expert ends (kit guard (pid == 0) & is0)", stmt(lb, "if pid == 0:").body,
         stmt(kb, "if (pid == 0) & is0:").body)
    # every live statement is accounted for
    covered = {"pid = tl.program_id(0)", "nprog = tl.num_programs(0)", "if launch_pdl:", "t = pid // TOPK",
               "k = pid % TOPK", "if PF_KB > 0:", "if EARLY_TRIGGER:", "if live:", "if first:",
               "rr = tl.arange(0, FILL_ROWS)", "for r0 in range(total + pid * FILL_ROWS, M_sum, nprog * FILL_ROWS):",
               "if pid == 0:"}
    rest = [h for i, (h, s) in enumerate(lb.items()) if h not in covered
            and not (i0 <= L.body.index(s) < i0 + 10) and not (i1 <= L.body.index(s) < i1 + 9)]
    check("RP every live kernel statement is compared above", not rest, f"uncompared: {rest}")
    # launch builder
    ll, kl = fn(rpl, "_launch"), fn(rpk, "_launch_new")
    lmap = {ast.unparse(s).split(" = ")[0]: s for s in ll.body if isinstance(s, ast.Assign)}
    kmap = {ast.unparse(s).split(" = ")[0]: s for s in kl.body if isinstance(s, ast.Assign)}
    same("RP launch: _pf_args call", lmap["pf_kb, pf_n, sf_step, w_args"], kmap["pf_kb, pf_n, sf_step, w_args"])
    same("RP launch: args tuple", lmap["args"], kmap["args"])
    lkw = {k.arg: d(k.value) for k in lmap["kw"].value.keywords}
    kkw = {k.arg: d(k.value) for k in kmap["kw"].value.keywords}
    extra = {k: v for k, v in kkw.items() if k not in lkw}
    check("RP launch: constexpr/option dict = live + SPLIT", {k: v for k, v in kkw.items() if k in lkw} == lkw
          and list(extra) == ["SPLIT"], f"diff {set(lkw.items()) ^ set(kkw.items())}")
    check("RP launch: grid = (M * TOPK, 5 if split else 1); live (M * TOPK,)",
          ast.unparse(lmap["grid"].value) == "(M * _TOPK,)"
          and ast.unparse(kmap["grid"].value) == "(M * _rp._TOPK, NCH if split else 1)")
    print("ALL PASS" if not FAILS else f"FAILED ({len(FAILS)}): {FAILS}")
    return 1 if FAILS else 0


# must-fail mutants of the kit sources: each one has to make the comparison above fail
MUTANTS = [
    ("rq: bf16 rounding of the split sum dropped", "rq", "zb = (ev + od).to(tl.bfloat16)", "zb = ev + od"),
    ("rq: split chains swapped", "rq", "zb = (ev + od).to(tl.bfloat16)", "zb = (od + ev).to(tl.bfloat16)"),
    ("rq: pad row stride M+1", "rq", "p = M + r + j0 + jj * M", "p = M + r + j0 + jj * (M + 1)"),
    ("rq: pad value 1", "rq", "tl.zeros([PAD_J, 8], tl.uint8)", "tl.full([PAD_J, 8], 1, tl.uint8)"),
    ("rp: amax eps floor dropped", "rp", "amax = tl.maximum(tl.max(tl.abs(xf), axis=1), eps)",
     "amax = tl.max(tl.abs(xf), axis=1)"),
    ("rp: x*inv -> x/scale", "rp", "q = xf * inv[:, None]", "q = xf / scale[:, None]"),
    ("rp: chunks 3/4 loads swapped", "rp", "xa3 = tl.load(xrow + _r5g_offs(3, GB, GROUP))",
     "xa3 = tl.load(xrow + _r5g_offs(4, GB, GROUP))"),
    ("rp: segment padding in every split program", "rp", "if first & is0:\n        r = start_e", "if first:\n        r = start_e"),
    ("rp: SPLIT missing from the launch dict", "rp", "SPLIT=split, num_warps=4", "num_warps=4"),
]


def mutants():
    rq0, rp0 = open(KIT_RQ).read(), open(KIT_RP).read()
    bad = []
    for name, which, old, new in MUTANTS:
        src = rq0 if which == "rq" else rp0
        assert src.count(old) == 1, (name, src.count(old))
        src = src.replace(old, new)
        FAILS.clear()
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                rc = main(rq_src=src if which == "rq" else None, rp_src=src if which == "rp" else None)
            except KeyError:  # a compared statement is missing: caught
                rc = 1
        print(f"{'PASS' if rc else 'FAIL'} mutant caught: {name}{'' if rc else ' (NOT caught)'}")
        if not rc:
            bad.append(name)
    FAILS.clear()
    FAILS.extend(bad)
    return 1 if bad else 0


if __name__ == "__main__":
    rc = main()
    print("== must-fail mutants")
    rc2 = mutants()
    print("ALL PASS (clean + every mutant caught)" if not (rc or rc2) else "FAILED")
    sys.exit(rc or rc2)
