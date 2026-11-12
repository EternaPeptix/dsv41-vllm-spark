"""r8-fixb step CPU tests: what changed from r6 (PACK operand identity) and the prefill-size pass-through claim.

    STEP_R5=... PYTHONPATH=<pytest deps> python3 -m pytest -q tests/test_r8_fixb.py
"""
import ast
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, KIT)
import det_gate as DG  # noqa: E402
import test_step as TS  # noqa: E402

R6 = os.path.join(KIT, "prev", "dsv41_step.py.r6-ebe2f7bb")
R8 = os.path.join(KIT, "files", "vllm", "models", "deepseek_v41", "nvidia", "dsv41_step.py")
CHANGED = {"_PackState.__init__", "_pack_ok", "_ident", "_install_engram", "_wrap_wkv"}


def _defs(path):
    tree = ast.parse(open(path).read())
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            out[node.name] = ast.dump(node, include_attributes=False)
            if isinstance(node, ast.ClassDef):
                for sub in node.body:
                    if isinstance(sub, ast.FunctionDef):
                        out[f"{node.name}.{sub.name}"] = ast.dump(sub, include_attributes=False)
        elif isinstance(node, ast.Assign):
            out["assign:" + ast.unparse(node.targets[0])] = ast.dump(node, include_attributes=False)
    return out


def test_r8_changes_only_the_pack_pairing():
    """COLEMB, C4, the AR map, the flags and the entry points are AST-identical to r6; only the PACK pairing differs."""
    a, b = _defs(R6), _defs(R8)
    diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    diff.discard("_PackState")  # the class node contains __init__
    assert diff == CHANGED, sorted(diff)


def test_r6_module_is_the_installed_r6_bytes():
    import hashlib

    assert hashlib.sha256(open(R6, "rb").read()).hexdigest().startswith("ebe2f7bbebefe916")


@pytest.mark.parametrize("mode", ["look", "all"])
def test_pack_prefill_sizes_are_pure_pass_through(mode):
    """Every chunked-prefill size: both layers run the stock embed + stock wkv, no kit gather, counters unchanged,
    no pairing state left behind; decode forwards before and after still pack."""
    m = TS.load_step({"DSV41_STEP_ENGRAM_PACK": mode})
    a, b, log = TS._engram_world(m, mode)
    for e in (a, b):  # the harness stages 64 rows; serving stages max_num_batched_tokens
        e.staged_rows = torch.zeros(DG.TMAX, 3, 16)
    for T in [6] + DG.PREFILL_T + [6, 129, 128]:
        log.clear()
        k0 = (m.COUNTS["pack_rows"], m.COUNTS["pack_wkv"])
        h = torch.zeros(T, 2, dtype=torch.int64)
        a.forward(h)
        b.forward(h)
        ags = [x for x in log if x[0] == "AG"]
        if T <= m.PACK_MAX_T:
            assert len(ags) == (1 if mode == "look" else 2), (T, log)
        else:
            assert ags == [], (T, log)
            assert ("stock_embed", 0) in log and ("stock_embed", 1) in log
            assert ("stock_wkv", "a") in log and ("stock_wkv", "b") in log
            assert (m.COUNTS["pack_rows"], m.COUNTS["pack_wkv"]) == k0
            assert m.PK.rows is None and m.PK.kv is None and m.PK.a_out is None and m.PK.b_out is None


def test_pack_all_foreign_operand_and_double_call_go_stock():
    m = TS.load_step({"DSV41_STEP_ENGRAM_PACK": "all"})
    a, b, log = TS._engram_world(m, "all")
    h = torch.zeros(5, 2, dtype=torch.int64)
    ea = a.embed(h)
    a.wkv(ea.flatten(-2))
    log.clear()
    a.wkv(ea.flatten(-2))  # a second call of the first layer's wkv in the same forward: stock
    assert ("stock_wkv", "a") in log
    eb = b.embed(h)
    log.clear()
    foreign = torch.ones(5, eb.flatten(-2).shape[1])
    b.wkv(foreign)  # a foreign operand with the right T: stock, not the stored kv
    assert ("stock_wkv", "b") in log
    # and a fresh forward with the true operand takes the stored kv
    ea = a.embed(h)
    a.wkv(ea.flatten(-2))
    eb = b.embed(h)
    log.clear()
    b.wkv(eb.flatten(-2))
    assert ("stock_wkv", "b") not in log


def test_r6_pairing_hole_is_real():
    """Negative control: the r6 module returns the stored kv to a foreign operand with the same T."""
    spec_m = DG.load_module(R6, "dsv41_step_r6_test")
    os.environ["DSV41_STEP_ENGRAM_PACK"] = "all"
    try:
        spec_m.PACK = "all"
        a, b, log = TS._engram_world(spec_m, "all")
        h = torch.zeros(5, 2, dtype=torch.int64)
        ea = a.embed(h)
        a.wkv(ea.flatten(-2))
        eb = b.embed(h)
        log.clear()
        b.wkv(torch.ones(5, eb.flatten(-2).shape[1]))
        assert ("stock_wkv", "b") not in log  # r6 served the stored kv
    finally:
        os.environ.pop("DSV41_STEP_ENGRAM_PACK", None)


def test_colemb_forward_is_unchanged_and_size_gated():
    """The colemb class hook passes through for T > max_t, T == 0, non-integer or non-CUDA ids (source-level check of
    the r8 module; the engaged path is the device test's)."""
    src = open(R8).read()
    assert "if T == 0 or T > ct.max_t or input_.dtype not in (torch.int32, torch.int64) or not input_.is_cuda:" in src
    assert "return orig(self, input_)" in src


def test_det_sim_full_lists_pass_and_catch():
    """The det gate's PACK part on the latest-possible scheduler, full T lists (decode 20 T, prefill 8 T up to 16384)."""
    res = []
    m = TS.load_step({})

    def gate(n, ok, d=""):
        res.append((n, bool(ok), "gate"))

    def mutant(n, ok, d=""):
        res.append((n, bool(ok), "mutant"))

    DG.WKV_OUT = 8  # small stand-in GEMM: the CPU run is about the pairing / ordering logic
    DG.part_pack_det(m, DG.SimRT(torch), gate, mutant, quick=False)
    bad = [r for r in res if not r[1]]
    assert not bad, bad
    assert sum(1 for r in res if r[2] == "mutant") == 7
