#!/usr/bin/env python3
"""build_g2.py (g2 colltrim): CPU-only build of both native libraries into <stage>/build/.

  cd ~/dsv41-stage/g2/colltrim && CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
      CUDA_VISIBLE_DEVICES= nice -n 19 ~/venvs/vllm032/bin/python native/build_g2.py

  build/rdma_ar_ext.so        native/rdma_ar.cu = the k1 parallel-load source (1abfbc63) + the column
                              all-gather mode; same flags as build_k1.py / ~/rdma_ar_dsv41/build.py, same
                              module name (rdma_ar_ext), so rdma_all_reduce.py loads it unchanged.
  build/libdsv41_l2pf_g2.so   native/l2pf_g2.cu (r4 burst kernels verbatim + the paced issuer).
Refuses to build into any live directory (~/rdma_ar_dsv41/build, build-k1pl, build-g2ct). No GPU is queried
(TORCH_CUDA_ARCH_LIST pinned to 12.1, the l2pf build is a plain nvcc call).
"""
import os
import shutil
import subprocess
import sys

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.1")
os.environ.setdefault("MAX_JOBS", "2")

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(here)
build = os.path.join(root, "build")
live = {os.path.realpath(os.path.expanduser(p)) for p in
        ("~/rdma_ar_dsv41/build", "~/rdma_ar_dsv41/build-k1pl", "~/rdma_ar_dsv41/build-g2ct")}
if os.path.realpath(build) in live:
    sys.exit(f"refusing to build into a live directory {build}")
ext_dir = os.path.join(build, "ext")
shutil.rmtree(build, ignore_errors=True)
os.makedirs(ext_dir, exist_ok=True)

# 1) paced L2 prefetch library
nvcc = os.path.join(os.environ.get("CUDA_HOME", "/usr/local/cuda"), "bin", "nvcc")
cmd = [nvcc, "-O3", "-arch=sm_121a", "-shared", "-Xcompiler", "-fPIC", "-lineinfo",
       "-o", os.path.join(build, "libdsv41_l2pf_g2.so"), os.path.join(here, "l2pf_g2.cu")]
print("+", " ".join(cmd), flush=True)
subprocess.run(cmd, check=True)

# 2) rdma_ar_ext with the column mode
from torch.utils.cpp_extension import load  # noqa: E402

mod = load(
    name="rdma_ar_ext",
    sources=[os.path.join(here, "rdma_ar.cu")],
    build_directory=ext_dir,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "-lineinfo"],
    extra_ldflags=["-L/usr/lib/aarch64-linux-gnu", "-l:libibverbs.so.1"],
    verbose=False,
)
so = os.path.join(ext_dir, "rdma_ar_ext.so")
shutil.copy2(so, os.path.join(build, "rdma_ar_ext.so"))
import torch  # noqa: E402

print("BUILD_OK torch", torch.__version__, "cuda", torch.version.cuda, "py", sys.version.split()[0],
      "K1", getattr(mod, "K1_VARIANT", None), "G2", getattr(mod, "G2_VARIANT", None),
      "cols", hasattr(mod.RdmaAR, "all_gather_cols"), "selftest_cols", hasattr(mod.SelfTest, "all_gather_cols_into"),
      sorted(f for f in os.listdir(build) if f.endswith(".so")), flush=True)
