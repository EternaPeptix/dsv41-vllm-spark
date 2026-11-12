#!/usr/bin/env python3
"""CPU-only build of skinny_mx.cu -> <dir>/build/skinny_mx_ext*.so for sm_121 (no GPU needed)."""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.1")
os.environ.setdefault("MAX_JOBS", "1")
os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")
from torch.utils.cpp_extension import load
here = os.path.dirname(os.path.abspath(__file__))
build = os.path.join(here, "build"); os.makedirs(build, exist_ok=True)
load(name="skinny_mx_ext", sources=[os.path.join(here, "skinny_mx.cu")], build_directory=build,
     extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "-lineinfo", "-Xptxas=-v"],
     verbose=True)
print("BUILD_OK", sorted(f for f in os.listdir(build) if f.endswith(".so")))
