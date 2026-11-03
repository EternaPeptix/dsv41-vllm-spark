#!/usr/bin/env python3
"""Build a standalone TP8 FC2 extension in a new Athena-only directory.

This is never invoked by a server. Build/load is not kernel execution, and the
extension is still an unvalidated experiment.
"""
import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    path = args.build_dir
    if (not path.is_absolute() or not path.resolve().is_relative_to(root)
            or path.resolve() == root or not path.parent.is_dir()):
        raise SystemExit('build directory must be a new subdirectory inside Athena folder')
    if os.environ.get('TORCH_CUDA_ARCH_LIST') != '12.1a':
        raise SystemExit('set TORCH_CUDA_ARCH_LIST=12.1a in this separate build process')
    path.mkdir(exist_ok=False)
    from torch.utils.cpp_extension import load
    module = load(name='athena_tp8_fc2_row_sm121',
                  sources=[str(Path(__file__).with_name('fc2_binding.cu'))],
                  build_directory=str(path),
                  extra_cuda_cflags=['-O3', '-std=c++20', '-DATHENA_EXPERIMENTAL_TP8_NATIVE_FC2=1'],
                  extra_cflags=['-O3', '-std=c++20'], verbose=True)
    print(f'Built and imported in this build process: {module.__file__}')
    print('This is not kernel execution or evidence of correctness, graph safety or speed.')


if __name__ == '__main__':
    main()
