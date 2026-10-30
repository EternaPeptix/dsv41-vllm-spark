#!/usr/bin/env python3
"""g5det_install.py install|uninstall|status   (ON each Spark, python3 stdlib; goal window 5 Determinism phase)

install  : sparse_attn_indexer.py f2ae649b -> patched (backup .bak-g5-det), + layers/dsv41_topk_ties.py
           (from ~/dsv41-stage/g5det/dsv41_topk_ties.py). Inert unless DSV41_TOPK_CANON_TIES / DSV41_DET_DUMP.
uninstall: restore sparse_attn_indexer.py from .bak-g5-det, remove layers/dsv41_topk_ties.py.
status   : sha256[:8] of both files.
"""
import hashlib
import os
import shutil
import sys

SITE = os.path.expanduser("~/venvs/vllm032/lib/python3.12/site-packages/vllm")
SAI = os.path.join(SITE, "model_executor/layers/sparse_attn_indexer.py")
MOD = os.path.join(SITE, "model_executor/layers/dsv41_topk_ties.py")
SRC_MOD = os.path.expanduser("~/dsv41-stage/g5det/dsv41_topk_ties.py")
BAK = SAI + ".bak-g5-det"
BASE_SHA = "f2ae649b"

A_IMPORT = """from vllm.model_executor.layers.indexer_topk import (
    RADIX_TOPK_WORKSPACE_SIZE,
    get_indexer_topk,
)
"""
B_IMPORT = A_IMPORT + """# g5det (goal window 5, 2026-10-27): boundary-tie canonicalization + debug dump of the indexer
# top-k; inert unless DSV41_TOPK_CANON_TIES or DSV41_DET_DUMP is set (see dsv41_topk_ties.py).
from vllm.model_executor.layers import dsv41_topk_ties as _g5det
"""

A_PREFILL = """                if not g4p_cand:
                    ops.top_k_per_row_prefill(
                        logits,
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                        topk_indices,
                        num_rows,
                        logits.stride(0),
                        logits.stride(1),
                        topk_tokens,
                    )
"""
B_PREFILL = A_PREFILL + """                    if _g5det.ENABLED:
                        _g5det.prefill_hook(
                            logits,
                            cu_seqlen_ks,
                            cu_seqlen_ke,
                            topk_indices,
                            q_slice_cast,
                            weights[row_start:row_end],
                            {
                                "hfp": hidden_states[row_start:row_end],
                                "kfp": k_quant_cast[: chunk.local_total_seq_lens],
                                "ksfp": k_scale_cast[: chunk.local_total_seq_lens],
                                "cand": -1
                                if candidate_blocks is None
                                else int(candidate_write),
                                "row_start": row_start,
                                "row_end": row_end,
                            },
                            forward_context,
                        )
"""

A_DECODE = """        get_indexer_topk(topk_backend)(
            logits,
            seq_lens,
            next_n,
            topk_indices,
            topk_tokens,
            attn_metadata_narrowed.max_seq_len,
        )
"""
B_DECODE = A_DECODE + """        if _g5det.ENABLED:
            _g5det.decode_hook(
                logits,
                seq_lens,
                next_n,
                topk_indices,
                padded_q_quant_cast,
                weights[:num_padded_tokens],
                {
                    "hfp": hidden_states[:num_decode_tokens],
                    "cand": -1 if candidate_blocks is None else int(candidate_write),
                },
                forward_context,
            )
"""


def sha(p):
    if not os.path.exists(p):
        return "absent"
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]


def status():
    print("%s sparse_attn_indexer.py %s bak-g5-det %s dsv41_topk_ties.py %s"
          % (os.uname().nodename, sha(SAI), sha(BAK), sha(MOD)))


def install():
    cur = sha(SAI)
    s = open(SAI).read()
    if "_g5det" in s:
        print("already installed")
        status()
        return
    if cur != BASE_SHA:
        sys.exit("refusing: sparse_attn_indexer.py is %s, expected %s" % (cur, BASE_SHA))
    for a, b in ((A_IMPORT, B_IMPORT), (A_PREFILL, B_PREFILL), (A_DECODE, B_DECODE)):
        if s.count(a) != 1:
            sys.exit("refusing: anchor found %d times:\n%s" % (s.count(a), a[:200]))
        s = s.replace(a, b)
    if not os.path.exists(BAK):
        shutil.copy2(SAI, BAK)
    shutil.copy2(SRC_MOD, MOD)
    tmp = SAI + ".g5tmp"
    open(tmp, "w").write(s)
    os.replace(tmp, SAI)
    status()


def uninstall():
    if not os.path.exists(BAK):
        sys.exit("no backup %s" % BAK)
    if sha(BAK) != BASE_SHA:
        sys.exit("backup sha %s != %s" % (sha(BAK), BASE_SHA))
    shutil.copy2(BAK, SAI)
    if os.path.exists(MOD):
        os.remove(MOD)
    status()


{"install": install, "uninstall": uninstall, "status": status}[sys.argv[1]]()
