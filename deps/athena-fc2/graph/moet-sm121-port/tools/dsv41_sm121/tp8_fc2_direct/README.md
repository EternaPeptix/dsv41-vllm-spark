# Experimental TP8 native FC2 tile staging

Status: **source-level component only**. `stage_native_fc2.cuh` is excluded unless `ATHENA_EXPERIMENTAL_TP8_NATIVE_FC2` is defined. It is not a standalone kernel, is not called by vLLM or MoET, has not been compiled or run, and has no performance or numerical result.

## Why this component

The user's six-agent trace identifies expert GEMMs as a major recorded GPU cost. The existing MoET donor (`../moe_fused_wide/moe_fused_sm121_wide.cu:435-545`) expects each FC2 64-output block in shared memory as one K32-major FP4 tile followed by per-slice packed scales. Its original fused implementation is for TP4 intermediate width 576 and 18 K32 blocks. The installed compact TP8 DeepGEMM path instead has 288 logical intermediate values and an FC2 width padded to 384, or 12 K32 blocks. This component proposes **direct native-to-shared staging** of one FC2 64-output block from the inspected TP8 384-wide compact tensor instead of requiring an additional full-model prepacked donor-weight copy.

It uses the following **unverified input contract**, derived from copied installed source and earlier index study:

- FP4 weight bytes `[384,5120,192]`, contiguous, viewed as uint32 `[384,5120,48]`.
- Packed weight scales `[384,5120,3]` int32 with strides `(5120*3, 1, 5120)`, each int32 holding four K32 scale bytes.
- The native FP4 nibbles and packed scale bytes must be compatible with the donor's E2M1 and UE8M0 MMA interpretation. Neither tensor shape nor a matching byte permutation proves that.

For one expert and one of 80 output blocks, 128 CTA threads gather 12 × 256 int32 weight words and 3 × 64 int32 scale words into two caller-owned shared buffers, followed by a CTA barrier. The weight order matches the **candidate** b12x_next/previous tile-stager mapping; the scale order matches the donor's per-K128, per-output-column four-byte loads. The destination occupies 12,288 bytes of weights plus 768 bytes of scales per buffer (26,112 bytes for two), before the rest of the kernel's shared state. These are allocation calculations, not measurements.

A CPU-only exhaustive comparison checked the component's address formulas against the earlier `tp8_weight_layout/padded_n128_mapping.py` index model for all 80 output blocks, 12 K32 blocks and 245,760 word locations, plus the scale-byte coordinates. This is a consistency check between two source-derived mappings, **not an independent verification against installed GPU tensors or the compiled donor**. The source uses scalar 4-byte global gathers rather than the donor's prepacked contiguous 16-byte `cp.async` loads; it could perform substantially worse.

## What must change for an actual candidate

1. Establish the live binary's exact weight/scale representation and nibble ordering on every applicable rank. The helper's pointer, element-count, CTA-shape and range guards cannot inspect encoding, strides, aliasing or physical shared-memory capacity; a host wrapper must establish those before launch.
2. Adapt the donor FC2 consumer from 18 K32 blocks and five activation slices to 12 K32 blocks and three slices, update all shared offsets and bounds, and connect this stager to an actual FC2 MMA implementation. Its barrier is CTA-local, and replacing `cp.async` changes double-buffer scheduling. Current source has no integration point or execution path.
3. Provide the correct installed 128-value activation scaling, BF16 rounding boundary and zero padding, and an unweighted BF16 route epilogue plus installed-style FP32 gather. The existing TP8 helper sketches are not a fused implementation.
4. Design safe routing, CTA residency/forward progress, stream and graph/workspace lifetimes, and an all-rank fallback before any writes. The existing persistent donor can wait for peer CTAs and must not be assumed safe under the live stack's overlap.
5. Only after a real build and minimal coordinated execution on the eight nodes can a matching performance experiment be considered. No KL, GSM8K or hash campaign is implied.

## Separate Marlin 320 path

The main agent reports that the separate `marlin-unpad-candidate` allocated 288-wide FC1 halves and 320-wide FC2, and served a sampled run whose output behavior is unverified. Its source (`marlin-unpad-candidate/marlin_utils_fp4.py`) applies Marlin weight repacking and scale permutation, while `marlin_moe.py` uses `ops.moe_wna16_marlin_gemm` with the selected input dtype and its own routing contract. The observed W4A16/BF16 path, its 10 K32 blocks and Marlin packing/scales are **not inputs to this DeepGEMM W4A8 stager**. A 320-wide implementation would require a separately justified K-tail, arithmetic, scale and weight adapter; changing `12` to `10` here would be incorrect. The source permits other input dtypes, so do not infer W4A16 merely from the generic function signature.

## Detached row-wise FC2 kernel candidate

`fc2_row_candidate.cu` now supplies a compile-time-disabled, actual **FC2-only** kernel body consuming the staging helper. It has no host binding, launcher, or serving integration. Its conceptual grid is `(80 output blocks, M_sum grouped rows)` with 128 threads per CTA. One CTA handles one grouped row and 64 output columns; negative grouped-layout entries return without writing, and invalid positive expert IDs set a caller-owned status flag. Distinct valid `(row, output block)` pairs write disjoint BF16 locations. This avoids cross-CTA dependency waits, but rereads the expert tile per row and could be much slower than a grouped GEMM; no performance expectation is asserted.

For each of three K128 activation groups, the installed copied activation source packs one UE8M0 exponent byte into the row's int32 scale word. The candidate extracts that byte and replicates it four times in the operand supplied to the donor-derived K32 MMA selectors. It executes 12 K32 MMA steps, accumulating into FP32, then stores **unweighted** BF16 values in the grouped-row output. The installed gather would still have to apply router weights using its inverse mapping. This is a source-derived candidate design: the exact native FP4/scale encoding, selector semantics, activation and accumulator arithmetic, non-finite behavior and BF16 rounding relative to the installed GEMM are unproved.

CPU-only arithmetic checks confirmed coverage of the 96 FP8 input words by the proposed lane offsets, replication of three scale bytes across four selectors, and unique coverage of each 64-column output block. These are index checks only, not CUDA compilation, execution or numerical evidence. The kernel requires the caller to validate tensor shapes, dtypes, alignment, non-aliasing, layouts and buffer lifetimes; clear a device status flag; launch on an ordered stream; and prevent every partial output from being used if any CTA reports an error. It has no safe host error-handling mechanism or all-rank coordination. Without those and the ABI evidence above, there is no deployable path and no performance benchmark should be run.

## Detached eager binding and build path

`fc2_binding.cu` is an uncompiled, non-serving PyTorch extension binding for the row-wise kernel. Its host checks restrict inputs to the exact source-derived TP8 contract: contiguous E4M3 `[M,384]`; packed MN-major int32 activation scales `[M,1]`; contiguous int32 grouped layout `[M]`; native FP4 bytes `[384,5120,192]`; packed int32 weight scales `[384,5120,3]` with strides `(15360,1,5120)`; contiguous BF16 output `[M,5120]`; and caller-owned int32 status `[1]`. It requires one CUDA device, 32-aligned `M` in `[32,8192]`, pointer alignment, and disjoint **byte ranges** between each input, output and status. Byte-range checking permits disjoint views within the installed common workspace; it does not establish which views are live or that other streams are finished with them.

After those host checks, the binding rejects an actively captured CUDA stream, enqueues status clearing and the kernel on the input device's current stream, then checks launch submission errors. CUDA execution is asynchronous. The caller **must wait for and inspect status before any output is consumed**; an invalid expert ID or stage error can coexist with partial writes from other CTAs. Negative-layout rows are not written, and must never be gathered. The binding cannot establish encoded FP4/scale contents, producer/consumer dependencies, correct stream use or error-free execution. Direct C++ callers bypass the Python opt-in, although host shape and capture checks still apply.

`fc2_experiment.py` is a standalone wrapper. `launch(...)` is disabled unless passed explicit experimental, native-encoding and stream/status-handling attestations; these are caller assertions, not verification. `prepare()` can load a separately built `.so` explicitly before launch. The path must be an existing absolute file inside the Athena folder. Neither wrapper nor binding is imported by vLLM.

After current GPU work has finished and the toolchain has been inspected, the **unrun** standalone build command on a node is:

```bash
source "$HOME/venvs/vllm032/bin/activate"
PORT="$HOME/Athena Hillclimbing stuff/moet-sm121-port/tools/dsv41_sm121/tp8_fc2_direct"
TORCH_CUDA_ARCH_LIST=12.1a python3 "$PORT/build_prebuilt_fc2.py" \
  --build-dir "$HOME/Athena Hillclimbing stuff/<new-trial>/fc2-build"
```

Replace the placeholder with a new Athena-local path whose parent exists. The builder refuses an existing build directory and sets the compile-time opt-in in the separate build process. Importing a resulting `.so` in that process would not execute the kernel. Torch's copied architecture parser accepts `12.1a`, but actual nvcc, pybind, PTX and binary compatibility are untested.

## Integration sketch, intentionally not a patch

In the copied `DeepGemmFP4Experts.apply`, FC2 runs after `_act_mul_quant` produces `a2q` and packed `a2q_scale`, and before `deepgemm_unpermute_and_reduce`. A future opt-in could allocate a separate FC2 status tensor and call the wrapper with `(a2q, a2q_scale, expert_ids, w2, self.w2_scale, mm2_out, status)` at that point, retaining the normal grouped GEMM as the default. It must choose eligibility and all-rank fallback **before candidate writes**, enforce output and workspace ownership and stream ordering, and prevent the gather from reading output until status is known. A failed asynchronous candidate cannot safely fall back after partial writes.

This sketch is not a practical performance path yet: a host status check before every layer's gather could introduce synchronization overhead, and the row-wise kernel rereads weights for each live row. A usable integration needs a proven error-handling design, exact encoding/rounding and launch behavior, graph and multi-rank lifecycle, and controlled execution on the real machine. Neither this source nor the wrapper is ready for benchmarking.

CPU-only fake-module tests exercised the Python default-off guard, capture refusal and dispatch wiring. The C++ shape/range checks and CUDA code have not been compiled or executed.

[STATUS_DESIGN.md](STATUS_DESIGN.md) traces which explicit failure cases the inspected producers and host binding appear to rule out, and why an asynchronous device-side gate still requires an all-rank step publication barrier. The candidate now ORs its two status bits so concurrent fault classes do not overwrite each other; zero status remains no proof of correct execution.
