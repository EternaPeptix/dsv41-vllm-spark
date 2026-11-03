#!/usr/bin/env python3
"""live_import_check.py <base|patched-off|patched-on> <kit>: import checks with the GPU hidden. Nothing launches.

base         the live modules the test calls import in a fresh process with the chain's flags; persistent_topk's schema and
             decode_hook's signature match the test's calls; the hook file loads against real vLLM and finds its extension.
patched-off  the patched sparse_attn_indexer.py with DSV41_BS03_DET_TOPK unset: _bs03 is None, and the hook module is not imported.
patched-on   the same file with the flag on: _bs03 is the hook module. The kit's copy is used, as nothing is installed.
"""
import importlib.machinery
import importlib.util
import inspect
import os
import sys

mode, kit = sys.argv[1:3]
if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    sys.exit("set CUDA_VISIBLE_DEVICES empty")
os.environ.update(DSV41_TOPK_CANON="1", DSV41_TOPK_CANON_FUSED="1", DSV41_TOPK_CANON_TIES="all")
os.environ.pop("DSV41_BS03_DET_TOPK", None)
if mode == "patched-on":
    os.environ["DSV41_BS03_DET_TOPK"] = "1"

import torch  # noqa: E402

HOOK_NAME = "vllm.model_executor.layers.dsv41_bs03_det_topk"
SAI_NAME = "vllm.model_executor.layers.sparse_attn_indexer"
HOOK_PATH = os.path.join(kit, "site", "dsv41_bs03_det_topk.py")
FAILS = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)


def load_hook():
    spec = importlib.util.spec_from_file_location(HOOK_NAME, HOOK_PATH)
    m = importlib.util.module_from_spec(spec)
    sys.modules[HOOK_NAME] = m
    spec.loader.exec_module(m)
    return m


def load_patched():
    path = os.path.join(kit, "patch", "sparse_attn_indexer.patched.py.txt")
    loader = importlib.machinery.SourceFileLoader(SAI_NAME, path)  # a .txt name: spec_from_file_location would return None
    spec = importlib.util.spec_from_loader(SAI_NAME, loader)
    m = importlib.util.module_from_spec(spec)
    sys.modules[SAI_NAME] = m
    loader.exec_module(m)
    return m


if mode == "base":
    import vllm._custom_ops  # noqa: F401
    from vllm.model_executor.layers import dsv41_topk_ties as ties
    from vllm.model_executor.layers import sparse_attn_indexer as sai
    from vllm.model_executor.layers.indexer_topk import RADIX_TOPK_WORKSPACE_SIZE, SparseIndexerTopk

    check("chain flags took effect", ties.TIES_DECODE and sai._DSV41_TOPK_CANON and sai._DSV41_TOPK_CANON_FUSED)
    check("fused CANON kernel object exists", sai._CANON_TOPK_512_KERNEL is not None)
    check("workspace size is 1 MiB", RADIX_TOPK_WORKSPACE_SIZE == 1024 * 1024)
    check("SparseIndexerTopk takes a backend", "backend" in inspect.signature(SparseIndexerTopk.__init__).parameters)
    schemas = torch._C._jit_get_schemas_for_operator("_C::persistent_topk")
    print("  ", [str(s) for s in schemas])
    names = [a.name for a in schemas[0].arguments]
    check("persistent_topk(logits, lengths, output, workspace, k, max_seq_len)", names == ["logits", "lengths", "output", "workspace", "k", "max_seq_len"], str(names))
    params = list(inspect.signature(ties.decode_hook).parameters)
    check("decode_hook(logits, seq_lens, next_n, topk, q, w, meta, fwd_key)", params == ["logits", "seq_lens", "next_n", "topk", "q", "w", "meta", "fwd_key"], str(params))
    h = load_hook()
    check("hook ACTIVE under the chain's flags", h.ACTIVE)
    e = h.ext()
    check("hook loads the extension (sha checked)", hasattr(e, "det_topk_decode"))
    logits, seq, out = torch.zeros(2, 8), torch.zeros((1, 2), dtype=torch.int32), torch.zeros(2, 512, dtype=torch.int32)
    check("hook refuses CPU tensors without launching", h.try_decode(logits, seq, 2, out, 512, "auto", None) is False)
elif mode == "patched-off":
    sai = load_patched()
    check("flag unset: _bs03 is None", sai._bs03 is None)
    check("flag unset: hook module not imported", HOOK_NAME not in sys.modules)
    src = inspect.getsource(sai.sparse_attn_indexer)
    check("decode path holds the guard", "bs03_done = _bs03 is not None and _bs03.try_decode(" in src)
    check("no extension loaded", not any("bs03_det_topk_sm121" in k for k in sys.modules))
elif mode == "patched-on":
    h = load_hook()
    sai = load_patched()
    check("flag on: _bs03 is the hook module", sai._bs03 is h)
else:
    sys.exit(__doc__)
print(("LIVE_IMPORT_FAIL " if FAILS else "LIVE_IMPORT_OK ") + mode, flush=True)
sys.exit(1 if FAILS else 0)
