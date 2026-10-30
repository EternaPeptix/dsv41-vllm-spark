# Drafter opportunity after the small kernel results

October 24, 2026. Read-only source investigation, not a new deployment or measured
candidate. The ten-experiment roadmap remains D01–D10. Proposal variants below
are follow-up research directions, not additional completed experiments.

## What can plausibly move the current goal

The current fixed C1 anchors are T=0 thinking requests. Code/prose/math measured
87.845/70.27/120.545 tokens/s with mean acceptance lengths 2.583/1.827/3.569;
code/prose hashes repeat. These are workload-specific results, not a universal
performance range. [Fresh control summary](../control/summary.json),
[request parameters and individual results](../control/single.jsonl).

An arithmetic projection AL/throughput gives approximate 29.40/26.00 ms steps
for code/prose. At unchanged AL, reaching 120 needs roughly 7.88/10.77 ms less
per step. Even deleting a full 3 ms drafter would project only 97.8/79.4 tokens/s.
At unchanged step time, AL would need to reach about 3.53/3.12. These ratios use
rounded counters and include different measurement boundaries; they are not
profiler measurements or forecasts. [Calculation artifact](goal-gap.json).

The next useful direction therefore combines target critical-path savings with
better early proposal acceptance. Compact drafter GEMMs are a reasonable exact
increment but not a plausible large removal of a 2–3 ms greedy drafter.

## D09: exact compact FC1 plus M32 is still unimplemented

The live source still gates the native adapter on E384/top6 at
[deep_gemm_moe.py:472](src/deep_gemm_moe.py#L472), compact/alignment geometry at
[553–591](src/deep_gemm_moe.py#L553), and the FC2 shape at
[731–736](src/deep_gemm_moe.py#L731). The allocator gate remains target-only at
[drafter-mxfp4.py:678](src/drafter-mxfp4.py#L678). E128/top3 currently misses both.
All source snapshots were fetched from spark-one and hashed in the manifest.

The isolated implementation should use a separately named drafter geometry and
flag, never indiscriminately widen target guards. Keep E384 behavior unchanged;
add E128/top3 FC1 N576/K5120 and FC2 N5120/K384 to the native/capture manifest,
use actual layer weights and routes, and compare stock, M32-only and M32+compact
at M5/10/20 for C1/C2/C4. Assert TP8, no EP/EPLB, no bias/LoRA, unpadded width288,
SILU and current layout compatibility just as the target allocator does. Retain
the complete K order and validate both live output and gathered output bitwise.
Maintain FC2's logical width384 and zero tail columns for each live input row.
The original private no-zero-padding contract must survive on dead rows.

The historical C1 trace had FC1 128.5 µs and FC2 64.5 µs ×3 MTP layers, a total
of 0.579 ms. Removing 25% of FC1 streamed weights ideally saves only about
0.096 ms if latency scales directly with those bytes. The historical broader
projection was 0.08–0.16 ms at C1. Neither number is a current observation, and
there is no measured C2/C4 projection. The existing live-row SiLU saving cannot
be claimed again. Even an impossible zero-cost routed MoE would recover only
0.579 ms under that old breakdown, about 21% of its old 2.79 ms drafter graph.
[Historical D09 analysis, lines 618–680](../../../outputs/experiment-roadmap-20261024/sources/claude/kernel-ideas-2.md#L618),
[old whole-graph profile, lines 14–30](../../../outputs/experiment-roadmap-20261024/sources/claude/k1/profile-1m.md#L14).

That trace also had about 0.427 ms for the draft head at ~200 GB/s and 0.47 ms
of drafter collectives. The greedy Markov path was already sharded at about
5×16 µs. Those old measurements do not establish current component costs. The
fresh startup AV log reports draft_ms(1)=4.280 raw and 7.680 after calibration;
neither is a measured greedy drafter span, so do not replace the 2–3 ms working
estimate with those calibration values. Capture a current greedy and sampled
component breakdown before allocating more native-kernel work.

## D03 and D10 remain valid lossless-speculation research

There are three distinct contracts:

- **Same arithmetic and same trajectory:** bitwise component outputs, same
  proposal decisions, pinned AV, target tokens and request hashes. D02/D05/D09
  initially target this contract.
- **Same target distribution:** arbitrary changed proposal q is allowed if
  proposal sampling, cached q, acceptance and residual sampling agree, with
  correct randomness and support handling. Sampled output for one seed may
  differ. D03/D10 belong here; that difference does not make them lossy.
- **Same greedy target result with changed proposals:** a changed drafter is
  allowed if the verifier still checks exact target greedy choices and returns
  them on rejection. Batch-shape/target arithmetic drift remains an independent
  implementation concern; prove request hashes under the selected path rather
  than assuming it from the mathematical argument.

D03 can materially help sampled/mixed steps: the current speculator still uses
full-vocabulary base logits and five sequential full Markov biases at
[lines 364–418](src/drafter-speculator.py#L364), while greedy steps use local
shards and small winner gathers at [420–477](src/drafter-speculator.py#L420).
The older 1.0–1.3 ms sampled-path projection is untested on the current system,
and D03 cannot improve the already-sharded T=0 anchors directly. A sampled
prototype must preserve global-ID Gumbel keys, fresh-step noise, padding/ties,
cached proposal logits and the final complete-logit requirement of verification.
An exact global draw does not imply bitwise equivalence if a different GEMM
tiling changes the logits themselves. Cache precisely the logits actually drawn.

D10's temperature multiplier is a low-cost q-family search for sampled rows,
but T=0 argmax is unchanged, so it cannot fix code/prose anchors. A matched q
must reach all verifier sites, including block acceptance and residual sampling.
The cache is explicitly pre-temperature at
[drafter-gumbel.py:237](src/drafter-gumbel.py#L237); the verifier reapplies it at
[rejection_sampler_utils.py:193](src/drafter-rejection_sampler_utils.py#L193),
[309](src/drafter-rejection_sampler_utils.py#L309) and
[874](src/drafter-rejection_sampler_utils.py#L874). A toy sampler-only test misses
speculator-to-verifier wiring errors. Use a deliberately wrong-q negative
control and keep the deployed fresh-step and fresh-uniform fixes.

## More relevant proposal-quality investigation for greedy anchors

A concrete inexpensive q-family to examine offline is the relative Markov
contribution: base_logits + alpha[position] * markov_bias, rather than only a
temperature applied to their sum. Alpha changes top-1 choices at T=0 and can
change overlap with target p for sampled requests. The current addition is at
[speculator.py:404](src/drafter-speculator.py#L404) and the greedy shard equivalent
at [455](src/drafter-speculator.py#L455). This is a hypothesis, with **no evidence
yet that any alpha beats 1**. It does not require checkpoint or target changes.
Do not install a hand-chosen alpha based on the low current acceptance alone.

First capture base, bias, target choice/logits and conditioning prefixes for
positions 0–2 across varied code/prose/math prompts in a root-controlled window.
Rank alpha values offline by held-out first-token agreement and expected
acceptance sum(min(p,q)), accounting for the exact original BF16 addition and
sampling operations. Position-1/2 results from a changed preceding token require
recomputing Markov bias; fixed-logit replay is not a valid new-prefix evaluation.
Keep recorded greedy and sampled data separate. If there is a stable held-out
gain, implement the arithmetic inside the existing add stage, cache that exact
proposal distribution and validate the complete target verifier. Greedy target
hashes, sampled distribution tests and matched latency/AL all gate promotion.
Changed proposal confidence may invalidate the calibrated confidence head; hold
AV/depth fixed for the causal test, then have the scheduler owner requalify it.

This search can improve acceptance without a large new drafter pass, but it
cannot be assumed to supply the roughly +0.95/+1.29 AL needed at fixed code/prose
step time. Perfecting only the first proposal adds at most 0.2601/0.4196 to its
direct first-position acceptance contribution; gains at later prefixes matter.
Those are per-position accounting bounds, not a bound on propagated improvement
in later proposals.

## Existing top-k proposal path is not a ready deployment toggle

Two current-source blockers were found by inspection:

1. The stock sampled path applies _DSV41_DRAFT_FRESH_NOISE at
   [speculator.py:384](src/drafter-speculator.py#L384). The top-k path at
   [479–525](src/drafter-speculator.py#L479) does not copy that key transformation.
   Selecting it under block verification would lose the deployed freshness
   contract unless corrected and tested through the actual proposal path.
2. The top-k path calls model.apply_markov_bias_gathered(), but the fetched
   [DSparkDeepseekV4ForCausalLM](src/drafter-dspark.py#L280), a plain nn.Module,
   defines no such method. No external injection has been verified. It cannot
   be treated as a supported V4.1 configuration from the speculator alone.

Moreover, the existing top-k path still computes and gathers full base logits,
whereas current greedy sharding avoids that gather. Small candidate bias work
does not automatically outweigh reintroducing full-vocabulary traffic. Any
truncated q remains mathematically usable with a correct residual sampler, but
has to earn both an implementation proof and a latency/acceptance trade-off.

## Recommended order

For the current T=0 goal, obtain the current critical-path and proposal evidence
first; use D09 as a bounded exact increment only if its current C1/C2/C4 share
warrants it. Run the offline relative-Markov hypothesis before a costly new
proposal architecture. For sampled service, prioritize D03's complete q-aware
sharded path over assuming D10 sharpening helps. Keep all measured gains and
prediction-only figures visibly separate. Root owns any future GPU captures,
builds, serving tests or rollout.
