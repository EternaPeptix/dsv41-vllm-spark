#!/usr/bin/env python3
"""build.py: compile rdma_ar.cu into <this dir>/build/rdma_ar_ext.so for sm_121 against the
torch in the running interpreter (DSv4.1: ~/venvs/vllm032, torch 2.13+cu130, CUDA 13.0).

CPU-only: TORCH_CUDA_ARCH_LIST is pinned, so no GPU is queried. Run with
  CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH ~/venvs/vllm032/bin/python build.py
Source: MiMo ~/rdma_ar (== published kernels/rdma_ar, sha1 904c8fdd) + fp32 reduce + MADV_DONTFORK.
"""
import os
import shutil
import sys

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.1")
os.environ.setdefault("MAX_JOBS", "2")
from torch.utils.cpp_extension import load  # noqa: E402

here = os.path.dirname(os.path.abspath(__file__))
build = os.path.join(here, "build")
# fresh build dir: never reuse objects compiled against another torch
shutil.rmtree(build, ignore_errors=True)
os.makedirs(build, exist_ok=True)
load(
    name="rdma_ar_ext",
    sources=[os.path.join(here, "rdma_ar.cu")],
    build_directory=build,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "-lineinfo"],
    extra_ldflags=["-L/usr/lib/aarch64-linux-gnu", "-l:libibverbs.so.1"],
    verbose=False,
)
import torch  # noqa: E402

print("BUILD_OK torch", torch.__version__, "cuda", torch.version.cuda, "py", sys.version.split()[0],
      sorted(f for f in os.listdir(build) if f.endswith(".so")))
