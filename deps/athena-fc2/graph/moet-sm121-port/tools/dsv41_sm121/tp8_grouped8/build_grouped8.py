#!/usr/bin/env python3
"""Build a detached grouped-eight TP8 extension in a fresh Athena directory.

This script is opt-in and is not a serving hook. Importing the built extension
is not evidence of execution, numerical validity, ABI compatibility or speed.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def render_binding(stage: str, source: str) -> str:
    if stage not in ("fc1", "fc2"):
        raise ValueError("stage must be fc1 or fc2")
    candidate = str(HERE / f"{stage}_grouped8_candidate.cu")
    if any(c in candidate for c in ('"', "\\", "\n", "\r")):
        raise ValueError("unsupported character in candidate source path")
    width = 10 if stage == "fc1" else 80
    substitutions = [
        (f'#include "{stage}_row_candidate.cu"', f'#include "{candidate}"'),
        (f"const dim3 grid({width}, static_cast<unsigned>(m)), block(128);",
         f"const dim3 grid({width}, static_cast<unsigned>(m / 8)), block(128);"),
        (f"athena_tp8_{stage}_direct::athena_tp8_{stage}_row_candidate<<<",
         f"athena_tp8_{stage}_direct::athena_tp8_{stage}_grouped8_candidate<<<"),
    ]
    for old, new in substitutions:
        if source.count(old) != 1:
            raise ValueError(f"binding anchor absent or ambiguous: {old}")
        source = source.replace(old, new, 1)
    return source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("fc1", "fc2"), required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    path = args.build_dir
    if (not path.is_absolute() or not path.resolve().is_relative_to(root)
            or path.resolve() == root or not path.parent.is_dir()):
        raise SystemExit("build directory must be a new subdirectory inside Athena folder")
    if os.environ.get("TORCH_CUDA_ARCH_LIST") != "12.1a":
        raise SystemExit("set TORCH_CUDA_ARCH_LIST=12.1a in this separate build process")
    original = (HERE.parent / f"tp8_{args.stage}_direct" / f"{args.stage}_binding.cu").read_text()
    generated = render_binding(args.stage, original)
    path.mkdir(exist_ok=False)
    binding = path / f"{args.stage}_grouped8_binding.cu"
    binding.write_text(generated)
    from torch.utils.cpp_extension import load
    upper = args.stage.upper()
    module = load(name=f"athena_tp8_{args.stage}_grouped8_sm121", sources=[str(binding)],
                  build_directory=str(path),
                  extra_cuda_cflags=["-O3", "-std=c++20",
                                     f"-DATHENA_EXPERIMENTAL_TP8_NATIVE_{upper}=1",
                                     f"-DATHENA_EXPERIMENTAL_TP8_{upper}_GROUPED8=1"],
                  extra_cflags=["-O3", "-std=c++20"], verbose=True)
    print(f"Built and imported in this process: {module.__file__}")
    print("No execution, numerical, ABI, graph-safety or performance claim is implied.")


if __name__ == "__main__":
    main()
