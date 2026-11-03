#!/usr/bin/env python3
"""Generate detached caller-reset FC2 graph-binding source; never builds it."""
from __future__ import annotations
import argparse
import importlib.util
import os
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
RESEARCH = HERE.resolve()


def once(source: str, old: str, new: str, label: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"{label}: expected exactly one anchor, got {source.count(old)}")
    return source.replace(old, new, 1)


def generate(original: str, render, candidate: Path) -> str:
    source = render(original, candidate, "serial32packed")
    source = once(source,
        '#include <torch/extension.h>\n',
        '#ifndef ATHENA_EXPERIMENTAL_TP8_FC2_GRAPH_BINDING\n'
        '#error "explicit experimental caller-reset graph binding required"\n'
        '#endif\n#include <torch/extension.h>\n', 'compile-time opt-in')
    source = once(source,
        '  cudaStreamCaptureStatus capture;\n'
        '  C10_CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));\n'
        '  TORCH_CHECK(capture == cudaStreamCaptureStatusNone, "experimental FC2 binding is eager-only");\n',
        '  // Experimental: capture is permitted. Caller owns status reset,\n'
        '  // lifetime, stream ordering and publication; this function does not.\n', 'capture gate')
    source = once(source,
        '  C10_CUDA_CHECK(cudaMemsetAsync(status.data_ptr<int32_t>(), 0, sizeof(int32_t), stream));\n',
        '  // NO status reset here: the forward owner must reset before each step.\n', 'status reset')
    source = source.replace('  // Status clearing and candidate launch are enqueued on the same current\n'
                            '  // stream. The caller must await/inspect status before any output consumer.\n',
                            '  // Candidate atomically accumulates structural error bits.\n', 1)
    source = source.replace('Experimental eager-only TP8 FC2 launch',
                            'Experimental caller-reset TP8 FC2 graph launch', 1)
    return source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True, help="new source path inside research folder")
    args = p.parse_args()
    out = args.output
    if (not out.is_absolute() or not out.resolve().is_relative_to(RESEARCH)
            or out.exists() or not out.parent.is_dir()):
        p.error("output must be a new path in this research folder")
    spec = importlib.util.spec_from_file_location("athena_fc2_build_support", HERE / "build_fc2_k288.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load binding renderer")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    original = HERE.parent / "moet-sm121-port/tools/dsv41_sm121/tp8_fc2_direct/fc2_binding.cu"
    candidate = HERE / "fc2_serial32packed_k288_candidate.cu"
    text = generate(original.read_text(), helper.render_binding, candidate)
    with out.open("x") as file:
        file.write(text)
    print(f"Wrote experimental source {out}; not compiled, loaded, captured or launched")


if __name__ == "__main__":
    main()
