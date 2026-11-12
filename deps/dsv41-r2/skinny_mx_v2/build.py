#!/usr/bin/env python3
"""CPU-only build of skinny_mx_v2.cu -> <dir>/build/skinny_mx_v2_ext*.so for sm_121 (no GPU needed).

Same toolchain and flags as the live ~/dsv41-r2/skinny_mx/build.py (torch cpp_extension, -O3 -lineinfo, no fast
math, so fp32 multiplies keep denormals exactly like v1). Run with the serving venv's python:
    nice -n 19 ~/venvs/vllm032/bin/python build.py
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.1")
os.environ.setdefault("MAX_JOBS", "1")
os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")
from torch.utils.cpp_extension import load  # noqa: E402

here = os.path.dirname(os.path.abspath(__file__))
build = os.path.join(here, "build")
os.makedirs(build, exist_ok=True)
load(name="skinny_mx_v2_ext", sources=[os.path.join(here, "skinny_mx_v2.cu")], build_directory=build,
     extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "-lineinfo", "-Xptxas=-v"], verbose=True)
for f in os.listdir(build):  # keep only the .so (intermediates are large)
    if f.endswith((".o", ".cuda.o", ".ninja_deps", ".ninja_log")) and not f.endswith(".so"):
        try:
            os.remove(os.path.join(build, f))
        except OSError:
            pass
print("BUILD_OK", sorted(f for f in os.listdir(build) if f.endswith(".so")))
sys.exit(0)
