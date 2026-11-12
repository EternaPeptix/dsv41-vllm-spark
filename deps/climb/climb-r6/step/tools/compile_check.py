"""Compile the kit's Triton kernels for sm_121 without a GPU launch and check the SASS instruction kinds.
Run on spark-eight in ~/climb-scratch/r6-step with CUDA_VISIBLE_DEVICES= (no GPU use), nice 19, lock absent:
  source ~/venvs/vllm032/bin/activate; export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
  CUDA_VISIBLE_DEVICES= nice -n 19 python3 tools/compile_check.py"""
import importlib.util, os, re, subprocess, sys, tempfile
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
spec = importlib.util.spec_from_file_location("st", os.path.join(HERE, "files/vllm/models/deepseek_v41/nvidia/dsv41_step.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
spec2 = importlib.util.spec_from_file_location("dt", os.path.join(HERE, "device_test.py"))
dt = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(dt)
def kernel_of(fn):
    for c in fn.__closure__ or ():
        v = c.cell_contents
        if isinstance(v, triton.runtime.jit.JITFunction):
            return v
    raise SystemExit("kernel not found")
tgt = GPUTarget("cuda", 121, 32)
def build(k, sig, const):
    try:
        src = ASTSource(fn=k, signature=sig, constexprs=const)
    except TypeError:
        src = ASTSource(fn=k, signature=sig, constants=const)
    return triton.compile(src, target=tgt)
def sass(cubin):
    with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as f:
        f.write(cubin); p = f.name
    out = subprocess.run(["cuobjdump", "-sass", p], capture_output=True, text=True).stdout
    os.unlink(p)
    ops = re.findall(r"/\*[0-9a-f]{4,}\*/\s+(?:@!?P\d\s+)?([A-Z0-9_.]+)", out)
    return ops
fails = 0
def report(name, ops, must, forbid):
    global fails
    fp = sorted({o for o in ops if re.match(r"^(F[A-Z0-9]*|H[A-Z0-9]+|DADD|DFMA|DMUL|MUFU)", o.split(".")[0]) and not o.startswith(("FSETP","FSEL"))})
    ok = all(any(o.startswith(x) for o in ops) for x in must) and not any(o.startswith(x) for o in ops for x in forbid)
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'} {name}: {len(ops)} SASS instrs; fp-class opcodes {fp}; must {must}; forbid {forbid}")
look = kernel_of(m._colemb_lookup_fn())
c = build(look, {"IDS": "*i32", "TAB": "*i16", "OUT": "*i16", "V": "i32", "N": "constexpr", "BLOCK": "constexpr"}, {"N": 640, "BLOCK": 128})
report("colemb lookup (int16 copy, N=640)", sass(c.asm["cubin"]), [], ["FADD", "FFMA", "FMUL", "F2F", "F2FP", "HADD2", "HFMA2.MMA"])
ar = kernel_of(dt.ar_emulator())
c = build(ar, {"X": "*i16", "OUT": "*bf16", "numel": "i32", "stride_r": "i32", "BLOCK": "constexpr"}, {"BLOCK": 4096})
ops = sass(c.asm["cubin"])
report("device_test AR emulator (FADD no FTZ, F2FP)", ops, ["FADD", "F2FP.BF16.F32.PACK_AB"], ["FADD.FTZ", "FFMA", "FMUL", "FHADD", "HADD2", "HFMA2.MMA"])
print("ALL PASS" if not fails else f"{fails} FAIL")
sys.exit(1 if fails else 0)
