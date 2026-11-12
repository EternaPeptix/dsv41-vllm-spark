#!/usr/bin/env python3
"""CPU lint for the review's blocker 1 class of bug: Triton 3.x refuses (NameError at compile time) any module
global read inside an @triton.jit function unless it is a tl.constexpr instance. test_plan.py stubs Triton, so it
cannot see this; this test checks the kit module's source with the AST:

  every Name loaded inside a @triton.jit function that is not an argument, a local, `tl`/`triton` or a Python
  builtin must be a module-level assignment whose value is `tl.constexpr(...)`.

It also runs the same lint on a minimal copy of the shipped-before-fix pattern (plain str global used in
tl.inline_asm_elementwise) to show the lint flags it.

Run: python3 tests/test_jit_globals.py   (stdlib only)
"""
import ast
import builtins
import os

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOD = os.path.join(KIT, "files/models/deepseek_v4/nvidia/ops/dsv41_attnout.py")
ALLOWED = {"tl", "triton"} | set(dir(builtins))


def _is_jit(fn):
    for d in fn.decorator_list:
        t = d.func if isinstance(d, ast.Call) else d
        if isinstance(t, ast.Attribute) and t.attr == "jit":
            return True
    return False


def _is_constexpr_call(v):
    return (isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and v.func.attr == "constexpr"
            and isinstance(v.func.value, ast.Name) and v.func.value.id == "tl")


def lint(src):
    tree = ast.parse(src)
    module_assign = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    module_assign[t.id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            module_assign[node.target.id] = node.value
    bad, checked = [], {}
    for fn in tree.body:
        if not (isinstance(fn, ast.FunctionDef) and _is_jit(fn)):
            continue
        local = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
        for n in ast.walk(fn):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                local.add(n.id)
        used = sorted({n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
                      - local - ALLOWED)
        checked[fn.name] = used
        for name in used:
            v = module_assign.get(name)
            if v is None or not _is_constexpr_call(v):
                bad.append((fn.name, name))
    return bad, checked


def main():
    bad, checked = lint(open(MOD).read())
    assert checked, "no @triton.jit functions found"
    assert set(checked) == {"_r2_wo_a_partial", "_r2_reduce_quant"}, checked
    assert set(checked["_r2_reduce_quant"]) == {"_ASM_UE8M0", "_ASM_INV_SCALE", "_ASM_FP8"}, checked
    assert not bad, f"non-constexpr module globals read inside @triton.jit: {bad}"
    print(f"PASS kit module: every global read inside @triton.jit is tl.constexpr(...): {checked}")

    before_fix = (
        'from vllm.triton_utils import tl, triton\n'
        '_ASM = """\n{ mov.b32 $0, $1; }\n"""\n'
        '@triton.jit\n'
        'def k(X):\n'
        '    x = tl.load(X)\n'
        '    y = tl.inline_asm_elementwise(_ASM, "=r,r", [x], dtype=tl.int32, is_pure=True, pack=1)\n'
    )
    bad, _ = lint(before_fix)
    assert bad == [("k", "_ASM")], bad
    print("PASS lint flags the shipped-before-fix pattern (plain str global in inline_asm_elementwise)")
    print("ALL PASS test_jit_globals")


if __name__ == "__main__":
    main()
