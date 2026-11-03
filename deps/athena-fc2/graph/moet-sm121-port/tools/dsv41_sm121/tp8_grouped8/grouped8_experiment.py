"""Opt-in eager-only grouped-eight launch. Never imported by serving code.

Experimental and unbuilt. All output is invalid if status is nonzero; no
consumer may use output before caller synchronizes and checks that status.
"""
from __future__ import annotations
import functools
import importlib.util
import os
from pathlib import Path


@functools.lru_cache(maxsize=2)
def _extension(stage):
    if stage not in ("fc1", "fc2"):
        raise ValueError("stage must be fc1 or fc2")
    key = f"ATHENA_TP8_{stage.upper()}_GROUPED8_SO"
    raw = os.environ.get(key, "")
    path = Path(raw)
    root = (Path.home() / "Athena Hillclimbing stuff").resolve()
    if not raw or not path.is_absolute() or not path.is_file() or not path.resolve().is_relative_to(root):
        raise RuntimeError(f"{key} must name an absolute prebuilt .so inside Athena folder")
    name = f"athena_tp8_{stage}_grouped8_sm121"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("experimental grouped8 extension is not loadable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "launch", None)):
        raise RuntimeError("grouped8 extension has no launch entry")
    return module


def prepare(stage):
    """Load the explicitly selected extension outside CUDA graph capture."""
    import torch
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("grouped8 extension must be loaded outside capture")
    return _extension(stage)


def launch(stage, aq, aq_scale, group_layout, weight, weight_scale, out, status, *,
           experimental=False, attest_native_encoding=False, attest_grouped_layout=False,
           attest_stream_and_status_handling=False):
    """Enqueue only; attestations are assertions, not verification.

    Caller owns all buffers and stream dependencies, must wait for status, and
    must reject the entire output on any error. Padding rows are untouched.
    """
    if not (experimental and attest_native_encoding and attest_grouped_layout
            and attest_stream_and_status_handling):
        raise RuntimeError("explicit experimental opt-in and all attestations are required")
    import torch
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("grouped8 is eager-only")
    _extension(stage).launch(aq, aq_scale, group_layout, weight, weight_scale, out, status)
    return out, status
