#!/usr/bin/env python3
"""Build a detached, caller-reset FC2 graph binding inside Athena's folder.

This is compile/import only. It does not register a vLLM kernel, launch it,
or establish graph safety, model output behaviour, or throughput.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess

here=Path(os.path.abspath(__file__)).parent
athena=(Path.home()/'Athena Hillclimbing stuff').resolve()
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--build-dir',type=Path,required=True)
p.add_argument('--run-experimental-build',action='store_true')
a=p.parse_args()
if not a.run_experimental_build:
 p.error('explicit experimental-build opt-in required')
if os.environ.get('TORCH_CUDA_ARCH_LIST')!='12.1a':
 p.error('set TORCH_CUDA_ARCH_LIST=12.1a')
if not a.build_dir.is_absolute() or not a.build_dir.resolve().is_relative_to(athena) or a.build_dir.exists() or not a.build_dir.parent.is_dir():
 p.error('build directory must be new and inside Athena')
binding=here/'fc2_packed_graph_binding.generated.cu'
if binding.exists():
 p.error('refusing to overwrite graph source')
subprocess.run([os.environ.get('PYTHON','python3'),str(here/'make_fc2_graph_binding.py'),'--output',str(binding)],check=True)
a.build_dir.mkdir()
from torch.utils.cpp_extension import load
module=load(name='athena_tp8_fc2_packed_graph_sm121',sources=[str(binding)],build_directory=str(a.build_dir),
 extra_cuda_cflags=['-O3','-std=c++20','-DATHENA_EXPERIMENTAL_TP8_NATIVE_FC2=1',
 '-DATHENA_EXPERIMENTAL_TP8_FC2_SERIAL32PACKED_K288=1','-DATHENA_EXPERIMENTAL_TP8_FC2_GRAPH_BINDING=1'],
 extra_cflags=['-O3','-std=c++20'],verbose=True)
print(f'Built/imported {module.__file__}; no kernel execution or serving integration',flush=True)
