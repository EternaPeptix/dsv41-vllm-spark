#!/usr/bin/env python3
"""CPU test: with DSV41_ATTNOUT_OP unset/0/off the patched files execute exactly the live code.

1. The module-level flag expression evaluates False for unset / "" / "0" / "off" / " OFF " and True otherwise.
2. o_proj.py: deep_gemm_fp8_o_proj's AST minus the one inserted `if _DSV41_ATTNOUT_OP and ...` statement is
   identical to the live function; that statement's test starts with the flag in a short-circuit `and`, so no
   other expression (not even isinstance) is evaluated when the flag is False; the new module is imported only
   inside it. Every other top-level statement of the file is unchanged.
3. sonar_bf16_bmm.py: try_project gains `r2ao=None` (last, defaulted) and one `if r2ao is not None: return ...`
   right before the live grouped_bmm call; minus those, the function AST equals the live one. Every other
   top-level statement is unchanged. No call site in the patched tree passes a 5th positional argument or an
   r2ao keyword except the flag-guarded block (the L2-prefetch wrapper forwards *a/**k unchanged, i.e. nothing
   when the flag is off).
Run: python3 tests/test_flag_off.py  (stdlib only)
"""
import ast
import os
import subprocess
import sys
import tempfile
import shutil

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
O_PROJ = "models/deepseek_v4/nvidia/ops/o_proj.py"
SONAR = "models/deepseek_v4/nvidia/ops/sonar_bf16_bmm.py"


def patched_tree(t):
    for rel in (O_PROJ, SONAR):
        dst = os.path.join(t, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(os.path.join(KIT, "base", rel), dst)
    subprocess.run([sys.executable, os.path.join(KIT, "install.py"), "install"], check=True,
                   env=dict(os.environ, R2AO_VLLM=t), capture_output=True)
    return {rel: open(os.path.join(t, rel)).read() for rel in (O_PROJ, SONAR)}


def func(tree, name):
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    raise AssertionError(name)


def dump(n):
    return ast.dump(n, include_attributes=False)


def top_level_except(tree, skip):
    return [dump(n) for n in tree.body if not (isinstance(n, ast.FunctionDef) and n.name in skip)]


def main():
    with tempfile.TemporaryDirectory() as t:
        new = patched_tree(t)
    old = {rel: open(os.path.join(KIT, "base", rel)).read() for rel in (O_PROJ, SONAR)}

    # 1. flag expression
    flag_line = [ln for ln in new[O_PROJ].splitlines() if ln.startswith("_DSV41_ATTNOUT_OP =")]
    assert len(flag_line) == 1
    expr = flag_line[0].split("=", 1)[1].split("  #")[0]
    for val, want in [(None, False), ("", False), ("0", False), ("off", False), (" OFF ", False),
                      ("rope", True), ("1", True), ("rope,rq", True)]:
        env = dict(os.environ)
        env.pop("DSV41_ATTNOUT_OP", None)
        if val is not None:
            env["DSV41_ATTNOUT_OP"] = val
        got = subprocess.run([sys.executable, "-c", f"print({expr.strip()})"], env=env, capture_output=True,
                             text=True).stdout.strip()
        assert got == str(want), (val, got)
    print("PASS flag expression: unset/''/0/off -> False; rope|1|rope,rq -> True")

    # 2. o_proj.py
    to, tn = ast.parse(old[O_PROJ]), ast.parse(new[O_PROJ])
    fo, fn = func(to, "deep_gemm_fp8_o_proj"), func(tn, "deep_gemm_fp8_o_proj")
    inserted = [s for s in fn.body if isinstance(s, ast.If) and "_DSV41_ATTNOUT_OP" in dump(s.test)]
    assert len(inserted) == 1
    ins = inserted[0]
    assert isinstance(ins.test, ast.BoolOp) and isinstance(ins.test.op, ast.And)
    assert isinstance(ins.test.values[0], ast.Name) and ins.test.values[0].id == "_DSV41_ATTNOUT_OP"
    assert not ins.orelse
    imports = [n for n in ast.walk(tn) if isinstance(n, (ast.Import, ast.ImportFrom)) and "dsv41_attnout" in dump(n)]
    assert len(imports) == 1 and any(imports[0] is n for n in ast.walk(ins)), "new module imported outside the gate"
    fn.body = [s for s in fn.body if s is not ins]
    assert dump(fn) == dump(fo), "deep_gemm_fp8_o_proj differs beyond the gated block"
    tl_old = top_level_except(to, {"deep_gemm_fp8_o_proj"})
    tl_new = [d for d in top_level_except(tn, {"deep_gemm_fp8_o_proj"}) if "_DSV41_ATTNOUT_OP" not in d]
    assert tl_old == tl_new, "o_proj.py top-level statements differ"
    print("PASS o_proj.py: only the flag line + one `if _DSV41_ATTNOUT_OP and ...` (flag evaluated first) added")

    # 3. sonar_bf16_bmm.py
    so, sn = ast.parse(old[SONAR]), ast.parse(new[SONAR])
    go, gn = func(so, "try_project"), func(sn, "try_project")
    assert [a.arg for a in gn.args.args] == [a.arg for a in go.args.args] + ["r2ao"]
    assert len(gn.args.defaults) == len(go.args.defaults) + 1
    assert isinstance(gn.args.defaults[-1], ast.Constant) and gn.args.defaults[-1].value is None
    hooks = [s for s in gn.body if isinstance(s, ast.If) and "r2ao" in dump(s.test)]
    assert len(hooks) == 1
    h = hooks[0]
    assert dump(h.test) == dump(ast.parse("r2ao is not None", mode="eval").body)
    idx = gn.body.index(h)
    assert "grouped_bmm" in dump(gn.body[idx + 1]), "hook is not right before the live grouped_bmm call"
    gn.body = [s for s in gn.body if s is not h]
    gn.args.args = gn.args.args[:-1]
    gn.args.defaults = gn.args.defaults[:-1]
    assert dump(gn) == dump(go), "try_project differs beyond the r2ao hook"
    assert top_level_except(so, {"try_project"}) == top_level_except(sn, {"try_project"})
    print("PASS sonar_bf16_bmm.py: try_project(+r2ao=None) + one `if r2ao is not None` before grouped_bmm; "
          "everything else identical")

    # call sites: only the gated block passes r2ao (5th positional) to try_project
    for rel, src in new.items():
        for n in ast.walk(ast.parse(src)):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "try_project":
                if len(n.args) > 4 or any(k.arg == "r2ao" for k in n.keywords):
                    raise AssertionError(f"{rel}: try_project called with r2ao outside the kit module")
    kit_mod = open(os.path.join(KIT, "files/models/deepseek_v4/nvidia/ops/dsv41_attnout.py")).read()
    calls = [n for n in ast.walk(ast.parse(kit_mod)) if isinstance(n, ast.Call) and
             isinstance(n.func, ast.Attribute) and n.func.attr == "try_project"]
    assert len(calls) == 1 and len(calls[0].args) == 5, "kit module must call try_project exactly once with ctx"
    print("PASS call sites: live code never passes r2ao; the kit module calls try_project once per projection")
    print("ALL PASS test_flag_off")


if __name__ == "__main__":
    main()
