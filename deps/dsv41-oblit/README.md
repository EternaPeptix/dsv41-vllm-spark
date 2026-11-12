# DSV41_OBLIT: OBLITERATUS projections at runtime

Since 2026-11-04, production serves the **official** `deepseek-ai/DeepSeek-V4.1-Flash-1017` FP8/FP4 weights
(revision `1bae0e10`). It applies the OBLITERATUS refusal-direction ablation
(`OBLITERATUS/DeepSeek-V4.1-Flash-1017-obliterated`, revision `29a9f998`, MIT) as exact BF16 runtime projections. It
does not load OBLITERATUS's 1.5 TB BF16 checkpoint. This was a deliberate output change. Every speed change made
after it was gated bitwise against the new outputs (see the top-level README).

## What the edit is

OBLITERATUS has one unit direction `d` per layer for layers 2 to 39, plus one for the LM head. Layers 0 and 1, norms,
embeddings, Engram and MTP are left alone. We recovered the directions by diffing their tensors against the official
checkpoint, reading only the needed byte ranges over HTTP:

- Readers `W' = W (I - d dᵀ)`: attention `wq_a`, `wkv`, indexer `weights_proj`, compressor, router gate, routed and
  shared `w1`/`w3`, and `lm_head` (which has its own direction).
- Writers `W' = (I - d dᵀ) W`: attention `wo_b`, routed and shared `w2`.

A projection is linear, so we apply it to activations rather than weights. This stays exact for the FP4 experts too:
the attention input, the attention output, the FFN input and the FFN output are each projected once per edited layer,
and the head input is projected before the LM head. The module is
`vllm-site/vllm/models/deepseek_v41/nvidia/dsv41_oblit.py`; the hooks in `model.py` carry `# dsv41-oblit` tags. When
`DSV41_OBLIT` is empty, the stock code path runs.

## The vectors are not in this repo

`prod/dsv41-tp8.sh` sets `DSV41_OBLIT=$HOME/models/dsv41/oblit/obliteratus_dirs.npz`. That file holds float32
`[5120]` unit vectors named `L2` … `L39` and `head`: 39 arrays, 808,010 bytes,
sha256 `6fe62c07ce7f79a59c14f0604c280c4838d8f29c051a1b90017e2bc715da11f6`, identical on all 8 nodes. It is model
data, so we leave it out like the weights.

To regenerate it: `python3 kit/obl_dirs.py --out DIR`. The script uses `kit/obl_probe.py`'s HTTP Range helpers
against the two public Hugging Face repos. For each layer it takes the direction from the router (BF16 on both
sides), from `wq_a` and from shared `w2`. It requires pairwise |cos| ≥ 0.999 and a fitted alpha within 1 ± 0.01,
then averages the sign-aligned estimates. It also writes `obliteratus_dirs.json` with per-layer fit statistics and
the source revisions. `kit/official27.sha256` lists the official shards that differ from the earlier checkpoint.

`kit/install_oblit.py` applies the anchored `model.py` edits, which need the `dsv41-steer` patch as an anchor.
`kit/test_oblit.py` tests the module and the installer. It runs on CPU, and adds GPU and CUDA-graph cases when a GPU
is free. The live `dsv41_oblit.py` (`8e8596d5`) is the kit version (`b894f54e`) plus lines tagged `# r1-pdl` and
`# r7-comm2`. Those lines are inert in production, because `DSV41_GLUE_PDL` and `DSV41_RDMA_PROJ` are both unset.

## Measured on 2026-11-04 (launcher 39e5c200; same harness for all three)

| Weights | MMLU (14,042 questions, thinking off) | G1 tok/s (code / prose / maths) |
|---|---:|---:|
| official, no projections | 88.50% | 117.2 |
| **official + projections (production)** | **88.10%** | **116.5** (118.0 / 84.3 / 147.3) |
| previous `dealignai/…-UNCENSORED-FP8` weight-edited checkpoint | 84.01% | ~114.8 (11-03 prod) |

Raw summaries: `notes/results-20261112/obl-{A,B,C}/`.
