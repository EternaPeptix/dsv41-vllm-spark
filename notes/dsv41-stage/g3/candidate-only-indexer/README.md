# Candidate-only FP8 prefill scoring

A concrete, default-off patch for the four consumer indexers **24/28/32/36**, plus a runnable isolated GPU qualifier. **No GPU qualification, timing, installation, or service change has been performed.** The patch is against the captured installed source, not the accumulated research composition.

The first arm packs candidate K/scale bits, calls the existing FP8 score implementation once, scatters into the original full-width logits, and retains the original native top-k and CANON. It targets dense scoring and the candidate-mask pass; it deliberately retains full-width selection. Layer 20, K-cache writes, and dense K gathers are unchanged.

Local checks, which import no GPU libraries:

```sh
python3 -B check_candidate.py
python3 -B check_harness.py
python3 -B qualify_gpu.py --plan
```

The future GPU command below is **a recipe, not authorization or evidence that it ran**. Run from a copied artifact directory in an isolated GPU test window using the matching unpatched vLLM environment, never inside a running serving process:

```sh
python3 -B qualify_gpu.py --run --rows 1,3,4,5,128 \
  --widths 131072,524288,1048576 \
  --patterns random,zeros,duplicates,sentinels,edge,short,empty,special \
  --warmup 2 --iterations 10 --samples 4 \
  --cache-dir ./qualification-cache --output ./gpu-results.json
```

The qualifier loads the local helper and calls the installed native backend without installing the patch. It requires the captured installed sources to match, rejects interpreter/override modes and optimized Python, and uses marked private Triton/DeepGEMM/Inductor/vLLM cache directories. It compares every valid compact score **before accepting duplicate scatter**, full masked score bits, raw FP8/scale copies, output padding, baseline repeatability, and final canonical indices. It traps new pack/scatter compilation after the first fixture; widths and output strides are runtime scalars. A failure exits nonzero and saves the failed case, completed records, and whether GPU execution had begun. Diagnostic clones are excluded from timings; baseline/candidate sample order alternates. Raw top-k order is reported separately because the production route requires CANON.

An optional `--input input.pt` accepts a `torch.load(weights_only=True)` tensor dictionary with `q`, `k`, `scales`, `weights`, `starts`, `ends`, and `candidates`. It validates the scoped shapes, dtypes, device, and bounds; use actual 128-row replay inputs for the final component panel. Synthetic fixtures cannot replace that evidence.

Only after component qualification, apply `candidate.patch` to a separate review/test checkout matching `source-manifest.json`, then qualify the actual model route. Both `DSV41_CANDIDATE_PREFILL_FP8=1` **and** `DSV41_TOPK_CANON=1` are required. Observe all four consumer hits with the actual replay shape and PIECEWISE eager-break path; a set flag is not a hit. Verify source 20 and every original K-cache write still execute. Test all eight ranks, request isolation, quantization/context/TP settings and deterministic output streams. Measure a full 1M prompt with synchronized baseline/candidate runs and unchanged request RNG/reduction order. Run exact-E7 interaction as four arms (neither, each alone, both) only for a separately qualified exact version: the old E7_TAIL variant was rejected and must not be blindly enabled.

`report.md` describes the memory/launch tradeoff, profile attribution and remaining native gates. The saved ~2-second projection is not a measured result from this work.
