"""Explicit, eager-only TP8 native FC2 experiment; not a vLLM serving hook.

The extension is unbuilt and unexecuted. Caller must preserve all input/output
storage and handle the asynchronous status flag before consuming any output.
"""
from __future__ import annotations
import functools
import importlib.util
import os
from pathlib import Path


@functools.lru_cache(maxsize=1)
def _extension():
    raw = os.environ.get('ATHENA_TP8_FC2_SO', '')
    path = Path(raw)
    root = (Path.home() / 'Athena Hillclimbing stuff').resolve()
    if not raw or not path.is_absolute() or not path.is_file() or not path.resolve().is_relative_to(root):
        raise RuntimeError('ATHENA_TP8_FC2_SO must name an absolute prebuilt .so inside Athena folder')
    name = 'athena_tp8_fc2_row_sm121'
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError('experimental FC2 extension is not loadable')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, 'launch', None)):
        raise RuntimeError('FC2 extension has no launch entry')
    return module


def prepare():
    """Explicitly load the separately built library, outside capture and launch."""
    import torch
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('FC2 extension must be loaded before capture')
    return _extension()


def launch(aq, aq_scale, group_layout, weight, weight_scale, out, status, *,
           experimental: bool = False, attest_native_encoding: bool = False,
           attest_stream_and_status_handling: bool = False):
    """Enqueue candidate; status must be awaited and checked before use.

    Both attestations are developer assertions, not verification. The C++ host
    binding validates shapes/strides/devices/alignment/non-overlap and rejects
    capture, then clears status and launches on the input's current stream.
    Output contains only live grouped rows; negative-layout rows are untouched.
    Nonzero status invalidates all output, including rows written by other CTAs.
    """
    if not (experimental and attest_native_encoding and attest_stream_and_status_handling):
        raise RuntimeError('explicit experimental mode and both contract attestations are required')
    import torch
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('experimental FC2 is eager-only')
    _extension().launch(aq, aq_scale, group_layout, weight, weight_scale, out, status)
    return out, status
