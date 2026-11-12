"""r1-attn: anchors apply to the live base file, flag-off code path unchanged, installer behaviour.

Run: python3 tests/test_install.py   (CPU only; no vllm import needed)
"""
import ast
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(KIT, "base")
ATTN_REL = "models/deepseek_v41/attention.py"
MOD_REL = "models/deepseek_v41/nvidia/dsv41_idxpar.py"
FLAG = "_DSV41_IDXPAR"


def _load_installer():
    spec = importlib.util.spec_from_file_location("r1attn_install", os.path.join(KIT, "install.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


INST = _load_installer()


def sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _strip_flag_nodes(tree):
    """Remove the flag assignment and every `if _DSV41_IDXPAR...:` statement (no else branch allowed)."""

    class T(ast.NodeTransformer):
        def visit_Assign(self, node):
            if any(isinstance(t, ast.Name) and t.id == FLAG for t in node.targets):
                return None
            return node

        def visit_If(self, node):
            names = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
            if FLAG in names:
                assert not node.orelse, "flag-guarded block must not have an else branch"
                return None
            self.generic_visit(node)
            return node

    return T().visit(tree)


class TestAnchors(unittest.TestCase):
    def setUp(self):
        self.base_path = os.path.join(BASE, ATTN_REL)
        self.base = open(self.base_path).read()
        self.kit = json.load(open(os.path.join(KIT, "KIT.json")))

    def test_base_sha_matches_kit_json(self):
        want = [f for f in self.kit["files"] if f["path"] == ATTN_REL][0]["base_sha256"]
        self.assertEqual(sha(self.base_path), want)
        # recorded from spark-one 2026-11-08 (identical 8/8 per the brief)
        self.assertEqual(want, "f2f48198de81ebc7eb4ead6d145cbeea5bf867c632d9d9aa8b237a6801b857ca")

    def test_anchor_counts(self):
        for old, _new, count in INST.EDITS:
            self.assertEqual(self.base.count(old), count, old[:60])

    def test_patched_compiles_and_strips_back_to_base(self):
        new = INST.patched(self.base)
        compile(new, "attention.py", "exec")
        tagged = [ln for ln in new.splitlines() if INST.TAG in ln]
        self.assertEqual(len(tagged), 10)
        # every inserted line carries the tag: dropping them restores the base byte for byte
        stripped = "\n".join(ln for ln in new.split("\n") if INST.TAG not in ln)
        self.assertEqual(stripped, self.base)
        self.assertEqual(INST.unpatched(new), self.base)
        # the patch is idempotency-safe: the TAG does not occur in the base
        self.assertNotIn(INST.TAG, self.base)

    def test_flag_off_ast_identical(self):
        """With the flag-guarded statements removed, the patched module's AST equals the base AST:
        the flag-off path executes exactly the base statements plus the (false) flag tests."""
        new = INST.patched(self.base)
        a = ast.dump(_strip_flag_nodes(ast.parse(self.base)), include_attributes=False)
        b = ast.dump(_strip_flag_nodes(ast.parse(new)), include_attributes=False)
        self.assertEqual(a, b)
        self.assertNotIn(FLAG, self.base)

    def test_every_tagged_statement_is_guarded(self):
        new = INST.patched(self.base)
        lines = new.split("\n")
        tagged = {i + 1 for i, ln in enumerate(lines) if INST.TAG in ln}
        tree = ast.parse(new)
        guarded = set()
        flag_def = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and FLAG in {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}:
                guarded |= set(range(node.lineno, node.end_lineno + 1))
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == FLAG for t in node.targets):
                flag_def |= set(range(node.lineno, node.end_lineno + 1))
        comments = {i for i in tagged if lines[i - 1].lstrip().startswith("#")}
        self.assertEqual(tagged - guarded - flag_def - comments, set())
        # the helper import happens only inside the guard
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and any(al.name == "dsv41_idxpar" for al in node.names):
                self.assertIn(node.lineno, guarded)

    def test_flag_value_parsing(self):
        expr = INST.E1_NEW.splitlines()[-1].split("  # r1-attn")[0].split("=", 1)[1]
        for val, want in (("", False), ("0", False), (" 0 ", False), ("1", True), ("flat", True)):
            env = {"DSV41_IDXPAR": val}
            got = eval(expr.replace('__import__("os").environ', "env"), {"env": env})
            self.assertEqual(got, want, val)
        got = eval(expr.replace('__import__("os").environ', "env"), {"env": {}})
        self.assertFalse(got)

    def test_helper_compiles(self):
        with open(os.path.join(KIT, "files", MOD_REL)) as f:
            compile(f.read(), "dsv41_idxpar.py", "exec")


class TestInstaller(unittest.TestCase):
    def _tree(self, tmp):
        v = os.path.join(tmp, "vllm")
        os.makedirs(os.path.join(v, "models/deepseek_v41/nvidia"))
        shutil.copy2(os.path.join(BASE, ATTN_REL), os.path.join(v, ATTN_REL))
        return v

    def _run(self, v, cmd, ok=True):
        env = dict(os.environ, R1ATTN_VLLM=v)
        p = subprocess.run([sys.executable, os.path.join(KIT, "install.py"), cmd], env=env,
                           capture_output=True, text=True)
        if ok:
            self.assertEqual(p.returncode, 0, p.stderr + p.stdout)
        return p

    def test_install_idempotent_uninstall(self):
        with tempfile.TemporaryDirectory() as tmp:
            v = self._tree(tmp)
            a = os.path.join(v, ATTN_REL)
            base = sha(a)
            self._run(v, "install")
            once = sha(a)
            self.assertNotEqual(once, base)
            self.assertTrue(os.path.exists(a + ".bak-r1attn"))
            self.assertEqual(sha(os.path.join(v, MOD_REL)), sha(os.path.join(KIT, "files", MOD_REL)))
            p = self._run(v, "install")
            self.assertIn("already patched", p.stdout)
            self.assertEqual(sha(a), once)
            st = self._run(v, "status").stdout
            self.assertIn("PATCHED base=ok", st)
            self._run(v, "dry")
            self._run(v, "uninstall")
            self.assertEqual(sha(a), base)
            self.assertFalse(os.path.exists(os.path.join(v, MOD_REL)))
            self.assertIn("clean", self._run(v, "status").stdout)

    def test_refuses_foreign_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            v = self._tree(tmp)
            a = os.path.join(v, ATTN_REL)
            with open(a, "a") as f:
                f.write("# drift\n")
            p = self._run(v, "install", ok=False)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("refusing", p.stderr + p.stdout)
            self.assertFalse(os.path.exists(a + ".bak-r1attn"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
