#!/usr/bin/env python3
"""Logic test of tools/compile_check.py WITHOUT Triton or a GPU (torch CPU only; run on 512S1 exo-venv313).

compile_check.py itself can only run on spark-eight (Triton 3.7.1 + ptxas, lock permitting). This test exercises
everything in it except Triton's binder and the real compiler, with a fake `triton` package on sys.path:
  * production arguments are recorded through the REAL launch functions (dsv41_routeprep._launch,
    dsv41_attnout._launch_rq, the kit's _launch_new / _launch_new_into) with CPU tensors;
  * the specialization comes from compile_check's rule model (the fake has no binder, so the fallback path runs),
    and the production-spec assertions (B1) must PASS: divisibility on pointers / x_row_stride / out_row_stride /
    mn_aligned / M_sum, M constexpr 1 at M == 1, plain i32 at M = 4, r2 == kit spec;
  * the PTX/SASS checks run on synthetic listings with the production load widths and must PASS; then must-fail
    modes must each make compile_check exit 1 with the expected FAIL line:
        late      the kit x loads are emitted after the first atom.shared / ATOMS (no hoist)
        spill     an STL in the kit SASS
        nodiv     M_sum made non-16-divisible (production spec not reached)
        scalar    the kit x loads lowered to 8 scalar b16 loads (must still PASS: the byte count is width-generic)
Also: the PTX/SASS byte counters on the real r2-moefront PTX (climb-r2/r2-moefront/ptx/s64_off.ptx, if present):
40 scalar b16 x loads, only the 4-byte ids load before the histogram.

    ~/exo-venv313/bin/python3 tests/test_compile_check_logic.py
"""
import os
import subprocess
import sys
import tempfile
import textwrap

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name}{(': ' + detail[-1500:]) if (detail and not ok) else ''}", flush=True)
    if not ok:
        FAILS.append(name)


FAKE_TRITON = {
    "triton/__init__.py": '''
import inspect, os
__version__ = "fake"
from . import language  # noqa
class _P:
    def __init__(self, i, prm, dns):
        self.num, self.name = i, prm.name
        ann = prm.annotation
        self.is_constexpr = "constexpr" in (ann if isinstance(ann, str) else getattr(ann, "__name__", str(ann)))
        self.do_not_specialize = prm.name in dns
        self.do_not_specialize_on_alignment = False
class _JIT:
    def __init__(self, fn, dns=()):
        self.fn, self.__name__ = fn, fn.__name__
        self.signature = inspect.signature(fn)
        self.arg_names = list(self.signature.parameters)
        self.params = [_P(i, p, dns) for i, p in enumerate(self.signature.parameters.values())]
    def __getitem__(self, grid):
        raise RuntimeError("fake triton: no launches")
    def warmup(self, *a, **k):
        raise RuntimeError("fake triton: no warmup")
def jit(fn=None, **kw):
    if fn is None:
        return lambda f: _JIT(f, tuple(kw.get("do_not_specialize") or ()))
    return _JIT(fn)
def next_power_of_2(n):
    p = 1
    while p < n:
        p *= 2
    return p
def cdiv(a, b):
    return (a + b - 1) // b
class _K:
    def __init__(self, asm):
        self.asm = asm
        self.metadata = type("M", (), {"shared": 2048})()
def compile(src, target=None, options=None):
    from ._synth import synth
    return _K(synth(src, options or {}))
''',
    "triton/language/__init__.py": '''
class constexpr:
    def __init__(self, v):
        self.value = v
''',
    "triton/backends/__init__.py": "",
    "triton/backends/compiler.py": '''
class GPUTarget:
    def __init__(self, backend, arch, warp_size):
        self.backend, self.arch, self.warp_size = backend, arch, warp_size
    def __repr__(self):
        return f"GPUTarget({self.backend}, {self.arch}, {self.warp_size})"
''',
    "triton/compiler/__init__.py": '''
class ASTSource:
    def __init__(self, fn, signature, constexprs=None, attrs=None):
        self.fn, self.signature, self.constexprs, self.attrs = fn, signature, constexprs or {}, attrs or {}
''',
    "triton/compiler/compiler.py": "raise ImportError('fake triton: no make_backend (forces the rule model)')\n",
    "triton/runtime/__init__.py": "",
    "triton/runtime/jit.py": '''
class MockTensor:
    def __init__(self, dtype, shape=None):
        self.dtype, self.shape = dtype, shape or [1]
    @staticmethod
    def data_ptr():
        return 0
''',
    "triton/_synth.py": '''
import os
MODE = os.environ.get("FAKE_MODE", "good")
ASM = os.environ["FAKE_ASM_LINES"].split("\\x1f")
def synth(src, opts):
    fn = src.fn.__name__
    c = {src.fn.arg_names[k[0]]: v for k, v in src.constexprs.items()}
    if "reduce_quant" in fn:
        ptx = "griddepcontrol.launch_dependents;\\ngriddepcontrol.wait;\\nadd.rn.f32 %f1;\\nmul.rn.f32 %f2;\\n" \\
              "cvt.rn.bf16.f32 %rs1;\\n" + "\\n".join(ASM) + "\\nst.global.b8 [%rd1];\\n"
        sass = "LDG.E R2, desc[UR4][R2.64] ;\\nSTG.E.U8 desc[UR4][R4.64], R5 ;\\nBRA 0x10 ;\\n"
        return {"ptx": ptx, "cubin": ("SASS:" + sass).encode()}
    kit = fn == "_r5g_rp_kernel"
    split = bool(c.get("SPLIT"))
    pdl = bool(c.get("launch_pdl"))
    nx = (1 if split else 5) if kit else 0
    vx = "ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd2 + 0 ];"
    sx = "ld.global.b16 { %rs1 }, [ %rd2 + 0 ];"
    xl = [vx] * nx
    gx = ["@P0 LDG.E.128 R8, desc[UR4][R6.64] ;"] * nx
    if kit and MODE == "scalar":
        xl = [sx] * (8 * nx)
        gx = ["LDG.E.U16 R8, desc[UR4][R6.64] ;"] * (8 * nx)
    late = (not kit) or MODE == "late"
    pre = (["griddepcontrol.wait; // dummy"] if pdl else [])
    pre += (["griddepcontrol.launch_dependents;"] if c.get("EARLY_TRIGGER") else [])
    pre += (["prefetch.global.L2 [ %rd9 + 0 ];"] if c.get("PF_KB") else [])
    ids = ["@%p1 ld.global.b32 { %r34 }, [ %rd16 + 0 ];"]
    body = (ids + ["atom.shared.add.u32 %r1, [%r2], %r3;"] + (xl if late else []))
    ptx = "\\n".join(pre + ([] if late else xl) + body + ["div.rn.f32 %f1, %f2, %f3;", "max.f32 %f4, %f5, %f6;",
                                                          "st.global.b8 [ %rd1 + 0 ], %rs1;"]) + "\\n"
    sbody = ["LDG.E R4, desc[UR4][R2.64] ;", "ATOMS.ADD RZ, [R3], R4 ;"] + (gx if late else [])
    sass = "\\n".join(([] if late else gx) + sbody + ["STG.E.U8 desc[UR4][R4.64], R5 ;"]) + "\\n"
    if kit and MODE == "spill":
        sass += "STL [R1+0x4], R9 ;\\n"
    return {"ptx": ptx, "cubin": ("SASS:" + sass).encode()}
''',
}

DRIVER = '''
import os, sys, runpy
sys.argv = ["compile_check.py"]
os.environ["TRITON_CACHE_DIR"] = os.path.join(os.environ["FAKE_ROOT"], "tcache")
g = runpy.run_path(os.environ["CC_PATH"], run_name="cc")

def fake_cuobjdump(cubin, flag):
    t = cubin.decode()[5:]
    return "REG:40 STACK:0 SHARED:2048 LOCAL:0" if flag == "-res-usage" else t

g["main"].__globals__["cuobjdump"] = fake_cuobjdump
if os.environ.get("FAKE_MODE") == "nodiv":
    f0 = g["m_sum_fused"]
    g["main"].__globals__["m_sum_fused"] = lambda M: f0(M) + 8
g["main"]()
'''


def asm_lines():
    """The inline-asm instruction lines compile_check looks for, read from the live dsv41_attnout.py source."""
    import re
    src = open(os.path.join(HERE, "base/dsv41_attnout.py")).read()
    out = []
    for nm in ("_ASM_UE8M0", "_ASM_INV_SCALE", "_ASM_FP8"):
        m = re.search(nm + r' = tl\.constexpr\("""(.*?)"""\)', src, re.S)
        out += [ln.strip() for ln in m.group(1).splitlines() if ln.strip() and "$" not in ln]
    return out


def run(root, mode):
    env = dict(os.environ, FAKE_MODE=mode, FAKE_ROOT=root, CC_PATH=os.path.join(HERE, "tools/compile_check.py"),
               PYTHONPATH=os.path.join(root, "fake"), FAKE_ASM_LINES="\x1f".join(asm_lines()))
    r = subprocess.run([sys.executable, "-c", DRIVER], env=env, capture_output=True, text=True, timeout=600,
                       cwd=root)
    return r.returncode, r.stdout + r.stderr


def main():
    root = tempfile.mkdtemp(prefix="r5g-cc-")
    for rel, body in FAKE_TRITON.items():
        p = os.path.join(root, "fake", rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").write(textwrap.dedent(body).lstrip())
    rc, out = run(root, "good")
    check("good listings + production spec: compile_check exit 0, ALL PASS", rc == 0 and "ALL PASS" in out, out)
    check("rule model used (fake has no binder) and announced", "rule model" in out, out)
    for frag in ("hoist_m1: M specialized to constexpr 1", "split_m4: M = 4 plain i32",
                 "hoist_m4: tt.divisibility 16 on every pointer and on x_row_stride",
                 "split_i64_m1: specialization identical to the r2 kernel's",
                 "rq_r2_m1: tt.divisibility 16 on ['P', 'Q', 'SF']; M plain i32",
                 "rq m4: kit specialization identical", "hoist_m4: PTX global-load bytes/thread before the first",
                 "split_m1: SASS LDG bytes/thread before the first ATOMS", "hoist_pf2_m4: prefetch.global.L2 present",
                 "split_nopdl_m4: launch_dependents absent", "hoist_pdl_m4: launch_dependents present",
                 "rq_kit_m1: no local-memory spills"):
        check(f"good: PASS line '{frag}'", any(ln.startswith("PASS " + frag) for ln in out.splitlines()), out)
    nfail = sum(ln.startswith("FAIL") for ln in out.splitlines())
    check("good: no FAIL line", nfail == 0, out)
    rc, out = run(root, "scalar")
    check("scalar b16 x loads (width-generic byte count): still exit 0", rc == 0, out)
    for mode, frag in (("late", "PTX global-load bytes/thread before the first atom.shared"),
                       ("late", "SASS LDG bytes/thread before the first ATOMS"),
                       ("spill", "no local-memory spills"),
                       ("nodiv", "tt.divisibility 16 on every pointer")):
        rc, out = run(root, mode)
        fl = [ln for ln in out.splitlines() if ln.startswith("FAIL") and frag in ln]
        check(f"must-fail mode {mode!r}: exit 1 with FAIL '{frag}' for hoist and split", rc == 1 and
              any("hoist" in ln for ln in fl) and any("split" in ln for ln in fl), out)
    # real r2-moefront PTX (no divisibility attrs): the byte counter on real text
    real = os.environ.get("R5G_R2PTX") or os.path.expanduser(
        "~/Documents/dsv41-offline/climb-r2/r2-moefront/ptx/s64_off.ptx")
    if os.path.exists(real):
        code = ("import runpy,os;g=runpy.run_path(os.environ['CC_PATH'],run_name='cc');"
                "t=open(os.environ['REAL']).read();"
                "print(g['loads_before'](t,'atom.shared',g['PTX_LD'],g['ptx_ld_bytes']));"
                "import collections;print(sum(g['ptx_ld_bytes'](m.group(1)) for m in g['PTX_LD'].finditer(t)))")
        env = dict(os.environ, PYTHONPATH=os.path.join(root, "fake"), CC_PATH=os.path.join(HERE,
                   "tools/compile_check.py"), REAL=real, FAKE_ASM_LINES="")
        r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, cwd=root)
        check("real r2-moefront s64_off.ptx: 4 B (the ids ld.global.b32) before the histogram, 88 B in all "
              "(40 b16 + 2 b32)", r.stdout.split()[:4] == ["(4,", "{'ld.global.b32':", "1})", "88"],
              r.stdout + r.stderr)
    else:
        print(f"INFO {real} not on this host: real-PTX counter case skipped")
    import shutil
    shutil.rmtree(root, ignore_errors=True)
    print("ALL PASS" if not FAILS else f"FAILED ({len(FAILS)}): {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
