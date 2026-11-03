#!/usr/bin/env python3
"""Build a detached FC2 K288 pack extension; never modifies the engine."""
import argparse
import os
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
ATHENA = (Path.home() / "Athena Hillclimbing stuff").resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-experimental-build", action="store_true")
    parser.add_argument("--build-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.run_experimental_build:
        parser.error("explicit --run-experimental-build required")
    path = args.build_dir
    if (not path.is_absolute() or not path.resolve().is_relative_to(ATHENA)
            or path.resolve() == ATHENA or path.exists() or not path.parent.is_dir()):
        parser.error("build directory must be new, with an existing parent, under Athena")
    if os.environ.get("TORCH_CUDA_ARCH_LIST") != "12.1a":
        parser.error("set TORCH_CUDA_ARCH_LIST=12.1a")
    source = HERE / "fc2_pack_binding.cu"
    if not source.is_file() or not (HERE / "fc2_pack_k288_source.cu").is_file():
        parser.error("pack source or binding missing")
    path.mkdir(exist_ok=False)
    from torch.utils.cpp_extension import load
    module = load(name="athena_fc2_k288_pack_sm121", sources=[str(source)], build_directory=str(path),
                  extra_cuda_cflags=["-O3", "-std=c++20", "-DATHENA_EXPERIMENTAL_TP8_FC2_PACK_K288=1"],
                  extra_cflags=["-O3", "-std=c++20"], verbose=True)
    print(f"Built and imported {module.__file__}; no model or GPU kernel was run")


if __name__ == "__main__":
    main()
