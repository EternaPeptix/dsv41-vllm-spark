"""Installer + flag-off tests for r1-eager (CPU only; no vllm import needed).

Run: python3 -m pytest -q tests/   (from the kit folder)
"""
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys

import pytest

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_MR = os.path.join(KIT, "base", "v1", "worker", "gpu", "model_runner.py")
sys.path.insert(0, KIT)
import install as inst  # noqa: E402


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def kit_base(rel):
    return next(f["base_sha256"] for f in json.load(open(os.path.join(KIT, "KIT.json")))["files"] if f["path"] == rel)


@pytest.fixture
def tree(tmp_path):
    v = tmp_path / "vllm"
    (v / "v1" / "worker" / "gpu").mkdir(parents=True)
    shutil.copy2(BASE_MR, v / "v1" / "worker" / "gpu" / "model_runner.py")
    return v


def run(tree, cmd):
    env = dict(os.environ, R1EAGER_VLLM=str(tree))
    return subprocess.run([sys.executable, os.path.join(KIT, "install.py"), cmd], env=env,
                          capture_output=True, text=True)


def test_base_copy_is_live_base():
    assert sha(BASE_MR) == kit_base("vllm/v1/worker/gpu/model_runner.py")


def test_anchors_apply_exact_counts():
    src = open(BASE_MR).read()
    for old, _new, count in inst.EDITS:
        assert src.count(old) == count, old[:60]
    new = inst.patched(src)
    assert new.count(inst.TAG) == 10
    assert inst.unpatched(new) == src


def test_dry_install_idempotent_status_uninstall(tree):
    mr = tree / "v1" / "worker" / "gpu" / "model_runner.py"
    base = sha(mr)
    r = run(tree, "dry")
    assert r.returncode == 0, r.stderr
    assert r.stdout.count(inst.TAG) == 10
    r = run(tree, "install")
    assert r.returncode == 0, r.stderr
    patched_sha = sha(mr)
    assert patched_sha != base
    assert (tree / "v1" / "worker" / "gpu" / "dsv41_dtgraph.py").exists()
    assert sha(tree / "v1" / "worker" / "gpu" / "dsv41_dtgraph.py") == sha(os.path.join(KIT, "dsv41_dtgraph.py"))
    assert os.path.exists(str(mr) + ".bak-r1-eager")
    r = run(tree, "install")  # idempotent
    assert r.returncode == 0 and "already patched" in r.stdout
    assert sha(mr) == patched_sha
    r = run(tree, "status")
    assert "PATCHED" in r.stdout and "(= kit)" in r.stdout
    r = run(tree, "uninstall")
    assert r.returncode == 0
    assert sha(mr) == base
    assert not (tree / "v1" / "worker" / "gpu" / "dsv41_dtgraph.py").exists()
    assert "base" in run(tree, "status").stdout


def test_refuses_other_base(tree):
    mr = tree / "v1" / "worker" / "gpu" / "model_runner.py"
    mr.write_text(mr.read_text() + "# drift\n")
    r = run(tree, "install")
    assert r.returncode != 0 and "refusing" in (r.stderr + r.stdout)
    r = run(tree, "dry")
    assert r.returncode != 0


def test_refuses_tagged_on_other_base(tree):
    mr = tree / "v1" / "worker" / "gpu" / "model_runner.py"
    mr.write_text(inst.patched(mr.read_text()).replace("import torch.nn as nn\n", "import torch.nn as nn  # x\n"))
    r = run(tree, "install")
    assert r.returncode != 0 and "refusing" in (r.stderr + r.stdout)


def _strip_flag_nodes(tree):
    """Remove every node the kit added: the module-level `_DSV41_DTGRAPH = ...` assignment and every
    `if _DSV41_DTGRAPH:` block (any depth)."""

    class T(ast.NodeTransformer):
        def generic_visit(self, node):
            super().generic_visit(node)
            for field in ("body", "orelse", "finalbody"):
                seq = getattr(node, field, None)
                if isinstance(seq, list):
                    keep = []
                    for st in seq:
                        if isinstance(st, ast.If) and isinstance(st.test, ast.Name) and st.test.id == "_DSV41_DTGRAPH":
                            continue
                        if (isinstance(st, ast.Assign) and len(st.targets) == 1
                                and isinstance(st.targets[0], ast.Name) and st.targets[0].id == "_DSV41_DTGRAPH"):
                            continue
                        keep.append(st)
                    setattr(node, field, keep)
            return node
    return T().visit(tree)


def test_flag_off_ast_identical():
    src = open(BASE_MR).read()
    new = inst.patched(src)
    a = ast.dump(ast.parse(src))
    b = ast.dump(_strip_flag_nodes(ast.parse(new)))
    assert a == b


def test_inserted_statements_are_all_guarded():
    """Every tagged line is the flag assignment or inside an `if _DSV41_DTGRAPH:` block."""
    new = inst.patched(open(BASE_MR).read())
    t = ast.parse(new)
    guarded = set()
    for node in ast.walk(t):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "_DSV41_DTGRAPH":
            guarded.update(range(node.lineno, node.end_lineno + 1))
        if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "_DSV41_DTGRAPH"):
            guarded.update(range(node.lineno, node.end_lineno + 1))
    for i, line in enumerate(new.splitlines(), 1):
        if inst.TAG in line:
            assert i in guarded, (i, line)


def test_flag_expression():
    new = inst.patched(open(BASE_MR).read())
    line = next(l for l in new.splitlines() if l.startswith("_DSV41_DTGRAPH = "))
    expr = line.split("=", 1)[1].split(inst.TAG)[0]
    for val, want in ((None, False), ("", False), ("0", False), ("off", False), (" OFF ", False),
                      ("no", False), ("false", False), ("1", True), ("dt", True), ("update", True),
                      ("td", True)):
        env = {} if val is None else {"DSV41_DTGRAPH": val}
        old = os.environ.copy()
        try:
            os.environ.pop("DSV41_DTGRAPH", None)
            os.environ.update(env)
            assert eval(expr) is want, (val, want)
        finally:
            os.environ.clear()
            os.environ.update(old)


def test_region_brackets_the_glue():
    """dt_begin sits right before prepare_inputs in the real-batch branch; dt_end right before the
    inputs_embeds block, i.e. after model_state.prepare_attn and before embedding / lookback."""
    new = inst.patched(open(BASE_MR).read()).splitlines()
    b = next(i for i, l in enumerate(new) if "_dsv41_dtg.dt_begin(" in l)
    e = next(i for i, l in enumerate(new) if "_dsv41_dtg.dt_end(" in l)
    assert "input_batch = self.prepare_inputs(" in new[b + 1]
    body = "\n".join(new[b:e])
    for must in ("self.prepare_attn(input_batch)", "self.model_state.prepare_attn(",
                 "build_slot_mappings_by_layer("):
        assert must in body
    tail = "\n".join(new[e:e + 60])
    assert "prepare_inputs_embeds(" in tail and "self.model_state.prepare_inputs(" in tail
    assert "run_fullgraph(" not in body and "embed" not in body.split("self.model_state.prepare_attn(")[1]
