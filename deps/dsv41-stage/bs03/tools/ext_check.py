#!/usr/bin/env python3
"""ext_check.py <kit>: the extension loads and its glue refuses bad input, CPU only (run with CUDA_VISIBLE_DEVICES empty)."""
import importlib.machinery
import importlib.util
import os
import sys

import torch

kit = sys.argv[1]
if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    sys.exit("set CUDA_VISIBLE_DEVICES empty")
name = "vllm_bs03_det_topk_sm121"
loader = importlib.machinery.ExtensionFileLoader(name, os.path.join(kit, "build", name + ".so"))
spec = importlib.util.spec_from_loader(name, loader)
m = importlib.util.module_from_spec(spec)
loader.exec_module(m)
print("functions:", sorted(x for x in dir(m) if x.startswith("det_topk")))
m.det_topk_configure(65536)
try:
    m.det_topk_configure(1)
    sys.exit("NOT refused: cap out of range")
except RuntimeError as e:
    print("refused as expected:", str(e).splitlines()[0][:100])
try:
    m.det_topk_decode(torch.zeros(2, 8), torch.zeros((1, 2), dtype=torch.int32), 2, torch.zeros(2, 512, dtype=torch.int32), 2, 512, 8, 12288)
    sys.exit("NOT refused: CPU tensors")
except RuntimeError as e:
    print("refused as expected:", str(e).splitlines()[0][:100])
print("torch sees", torch.cuda.device_count(), "GPUs in this process")
print("EXT_OK")
