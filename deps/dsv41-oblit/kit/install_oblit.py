#!/usr/bin/env python3
"""Install / uninstall / check the DSV41_OBLIT hooks (OBLITERATUS refusal-direction port) in the vllm032 venv.

    python3 install_oblit.py install    # backs up model.py to model.py.bak-oblit, applies anchored edits
    python3 install_oblit.py uninstall  # restores the backup, removes the module
    python3 install_oblit.py status
Requires the dsv41-steer patch (its import line and the decoder layer's _dsv41_layer line are anchors).
With DSV41_OBLIT unset the patched model is bitwise the unpatched one: every hook is behind `self._oblit_l >= 0`
(-1 when OFF) or `_dsv41_oblit.ON`.
"""
import hashlib
import os
import shutil
import sys

V = os.environ.get("OBLIT_VLLM") or os.path.expanduser("~/venvs/vllm032/lib/python3.12/site-packages/vllm")
MODEL = f"{V}/models/deepseek_v41/nvidia/model.py"
MOD_DST = f"{V}/models/deepseek_v41/nvidia/dsv41_oblit.py"
MOD_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dsv41_oblit.py")
TAG = "# dsv41-oblit"
STEER_IMP = "from vllm.models.deepseek_v41.nvidia import dsv41_steer as _dsv41_steer  # dsv41-steer\n"
STEER_LAYER = ("        self._dsv41_layer = next((int(t) for t in str(prefix).split(\".\") if t.isdigit()), -1)"
               "  # dsv41-steer\n")


def hook(indent, call):
    return (f"{indent}if self._oblit_l >= 0:  {TAG}\n"
            f"{indent}    {call}  {TAG}\n")


EDITS = [
    # module import (after the steer import)
    (STEER_IMP, STEER_IMP + "from vllm.models.deepseek_v41.nvidia import dsv41_oblit as _dsv41_oblit  " + TAG + "\n", 1),
    # decoder layer: bound layer index (-1 = no projection)
    (STEER_LAYER, STEER_LAYER + "        self._oblit_l = _dsv41_oblit.layer_for(prefix)  " + TAG + "\n", 1),
    # 1. normalised attention input (all attention readers: wq_a, wkv, indexer.weights_proj, compressor)
    ("        if dsv41_sp:\n            # r4 prefill-sp: x is this rank's rows; returns this rank's rows.\n",
     hook("        ", "x = _dsv41_oblit.project(x, self._oblit_l)")
     + "        if dsv41_sp:\n            # r4 prefill-sp: x is this rank's rows; returns this rank's rows.\n", 1),
    # 2. attention output (wo_b), before the post mix (linear, so partial sums before a fused all-reduce are fine)
    ("        if self.use_sequence_parallel and self.attn.gemm_rs is None:\n            x = sp_reduce_scatter(x)\n\n",
     "        if self.use_sequence_parallel and self.attn.gemm_rs is None:\n            x = sp_reduce_scatter(x)\n"
     + hook("        ", "x = _dsv41_oblit.project(x, self._oblit_l)") + "\n", 1),
    # 3. normalised FFN input (router, routed w1/w3, shared w1/w3)
    ("        if dsv41_sp:\n            x = _dsv41_sp_ffn(\n",
     hook("        ", "x = _dsv41_oblit.project(x, self._oblit_l)") + "        if dsv41_sp:\n            x = _dsv41_sp_ffn(\n", 1),
    # 4. FFN output (routed w2, shared w2), all three branches
    ("            x = self.ffn(x, input_ids, mega_gate_metadata)\n        if mhc_stream is not None:\n",
     "            x = self.ffn(x, input_ids, mega_gate_metadata)\n"
     + hook("        ", "x = _dsv41_oblit.project_moe(x, self._oblit_l)") + "        if mhc_stream is not None:\n", 1),
    # 5. replay KV write reads the attention input too
    ("        self.attn.forward_kv(positions, x)\n",
     hook("        ", "x = _dsv41_oblit.project(x, self._oblit_l)") + "        self.attn.forward_kv(positions, x)\n", 1),
    # 6. head (reader): compute_logits and compute_logits_local
    ("        if _DSV41_R4:\n            # r4 small-kernels: certified (screened) target head; None -> stock.\n",
     f"        if _dsv41_oblit.ON:  {TAG}\n            hidden_states = _dsv41_oblit.project_head(hidden_states)  {TAG}\n"
     "        if _DSV41_R4:\n            # r4 small-kernels: certified (screened) target head; None -> stock.\n", 1),
    ("    ) -> torch.Tensor:\n        if _DSV41_R4:\n            from vllm.models.deepseek_v41.nvidia import dsv41_r4_small as _r4\n\n"
     "            out = _r4.target_logits(self, hidden_states, skip_gather=True)\n",
     f"    ) -> torch.Tensor:\n        if _dsv41_oblit.ON:  {TAG}\n"
     f"            hidden_states = _dsv41_oblit.project_head(hidden_states)  {TAG}\n"
     "        if _DSV41_R4:\n            from vllm.models.deepseek_v41.nvidia import dsv41_r4_small as _r4\n\n"
     "            out = _r4.target_logits(self, hidden_states, skip_gather=True)\n", 1),
]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8] if os.path.exists(p) else "-"


def patched_text(src):
    for old, new, count in EDITS:
        n = src.count(old)
        if n != count:
            sys.exit(f"{MODEL}: anchor found {n}x, expected {count}: {old[:70]!r}")
        src = src.replace(old, new)
    return src


def install():
    src = open(MODEL).read()
    if TAG in src:
        print(f"{MODEL}: already patched ({sha(MODEL)})")
    else:
        new = patched_text(src)
        bak = MODEL + ".bak-oblit"
        if not os.path.exists(bak):
            shutil.copy2(MODEL, bak)
        tmp = MODEL + ".tmp-oblit"
        open(tmp, "w").write(new)
        shutil.copymode(bak, tmp)
        os.replace(tmp, MODEL)
        print(f"{MODEL}: patched ({sha(bak)} -> {sha(MODEL)}), {new.count(TAG)} tagged lines")
    shutil.copy2(MOD_SRC, MOD_DST)
    print(f"{MOD_DST}: {sha(MOD_DST)}")


def uninstall():
    bak = MODEL + ".bak-oblit"
    if os.path.exists(bak):
        os.replace(bak, MODEL)
        print(f"{MODEL}: restored {sha(MODEL)}")
    if os.path.exists(MOD_DST):
        os.remove(MOD_DST)


def status():
    for p in (MODEL, MOD_DST):
        on = os.path.exists(p) and (p == MOD_DST or TAG in open(p).read())
        print(f"{sha(p)} {'PATCHED' if on else 'clean  '} {p}")


def dry():
    """Apply the edits in memory only and print the result's tagged lines (no file is written)."""
    new = patched_text(open(MODEL).read())
    for i, line in enumerate(new.splitlines(), 1):
        if TAG in line:
            print(f"{i:5d}: {line}")


{"install": install, "uninstall": uninstall, "status": status, "dry": dry}[sys.argv[1]]()
