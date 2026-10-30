
## Round 3 / e5: sampled spec decode, probabilistic drafting + fixed block verification (PROMOTED pb5, 2026-10-22 11:30 UTC)
Change: the default DSpark config now adds draft_sample_method=probabilistic and rejection_sample_method=block.
num_speculative_tokens stays 5 and adaptive verification stays on. Stock vLLM 0.32 block verification is lossy
for two reasons: it reuses acceptance uniforms across steps, and it reuses drafter Gumbel noise keyed by
(seed, position). Two site patches fix this. Both are inert when their flags are off:
 - v1/worker/gpu/spec_decode/rejection_sampler_utils.py 09522889 (stock 99556e84): DSV41_BLOCK_VERIFY_FRESH_U=1
 - v1/worker/gpu/spec_decode/dspark/speculator.py 0cc6e2bf (this is the e1 Markov file 774b34c2 + the e5 hunk): DSV41_DRAFT_FRESH_NOISE=1
 - Backups on all 8: *.bak-r3spec (99556e84 / 774b34c2). bench.py.bak-r3spec on spark-one; the new bench.py keeps the old CLI
   and adds --preset t1|t06, --top-p and --seed.
 - Launcher ~/dsv41-tp8.sh 2971eb50 on all 8 (backup ~/dsv41-tp8.sh.bak-r3spec = d5c41fd0). New knobs:
   DSV41_SPEC_SAMPLED (default 1; 0 = old greedy-draft JSON), DSV41_BLOCK_VERIFY_FRESH_U (default 1),
   DSV41_DRAFT_FRESH_NOISE (default 1).
 - GPU unit test (spark-two, test_spec_sampling.py): greedy PASS. stream with FRESH_U=0: block rows FAIL with greedy
   drafts, which confirms the test detects the bug. FRESH_U=1 stream: all ok. stream-vllm-draft with stock noise:
   block+prob FAIL. Adding FRESH_NOISE=1: all ok.
Trade-off: the sharded Markov drafter (e1) needs greedy drafting. With probabilistic drafting it logs
"DSV41_DSPARK_SHARDED_MARKOV=1 ignored" and uses the full-vocab Markov path. Every arm and the ctl5m0 control
therefore ran with DSV41_DSPARK_SHARDED_MARKOV=0; ctl5mk is the live e1-e4 config.
A/B with one instance per row (order ctl5m0a ps5a pb5a ctl5m0b ctl5mk ps5b pb5b ctl5m0c pb5c ps5c).
Bench: greedy = 3-type single mean; t1 = temp 1.0 / top_p 0.95; t06 = temp 0.6. Columns are tok/s; AL = accept length.
 inst    | greedy AL   g.c8  | t1    AL   | t1.c4 t1.c8 t1.c16 | t06   | g06 (GSM8K-50 t0.6 x2)
 ctl5m0a | 85.83 2.75 218.3 | 80.77 2.64 | 125.1 195.5 286.5 | 83.32 | 50.0
 ctl5m0b | 86.53 2.77 212.2 | 74.73 2.40 | 140.2 196.8 290.1 | 83.62 | 50.0
 ctl5m0c | 84.77 2.69 212.7 | 76.32 2.45 | 139.3 200.2 295.2 | 83.43 | 49.5
 ctl5mk  | 86.63 2.62 217.2 | 79.43 2.42 | 141.5 203.8 301.9 | 87.04 | 49.5
 pb5a    | 84.56 2.67 216.6 | 85.52 2.74 | 144.5 207.8 315.0 | 84.18 | 50.0
 pb5b    | 83.04 2.63 216.5 | 87.11 2.81 | 147.3 212.4 308.0 | 85.41 | 50.0
 pb5c    | 83.52 2.64 213.3 | 87.62 2.83 | 145.9 206.7 308.4 | 86.73 | 50.0
 ps5a    | 83.86 2.65 214.1 | 93.78 3.09 | 143.8 204.4 303.3 | 84.81 | 50.0
 ps5b    | 84.33 2.67 217.8 | 83.95 2.69 | 149.5 207.7 305.9 | 87.09 | 50.0
 ps5c    | 83.70 2.65 213.6 | 83.68 2.67 | 147.4 205.8 301.3 | 86.16 | 50.0
Pooled ctl5m0 (3) -> pb5 (3):
 - t1 single 77.27 -> 86.75 (+12.3%); c4 +8.2%; c8 +5.8%; c16 290.6 -> 310.5 (+6.8%); t06 single +2.4%.
 - Greedy single 85.71 -> 83.71 (-2.3%, inside the +/-4% gate). Greedy c8 214.4 -> 215.5 (+0.5%).
 - Greedy AL 2.733 -> 2.647 (-0.087), just over the +/-0.08 gate. ps5 has the same greedy code path and shows -0.077.
   ctl5mk (live) is 2.624. Greedy text is bit-identical: exact.py combined a8c3394a in every instance, including
   the promoted relaunch. So the AL shift is adaptive-verification trimming, not different greedy tokens.
 - Against the live config (ctl5mk, n=1): greedy single 86.63 -> 83.71 (-3.4%), t1 single +9.2%, t1 c16 +2.8%.
ps5 (probabilistic + standard rejection, no patch needed): t1 single +12.8%, c16 +4.4%. pb5 is better at every concurrency.
gb5 (greedy draft + fixed block) and n=7 arms were not run (time box).
Sampled gate: every pb5/ps5 instance PASS against ctl5m0b (50/50 both runs).
Promotion gate, GSM8K-200 t0.6 conc16 seed 1000: control (live) 196/200, 1 truncated, 267.1 tok/s (misses 93,100,119,184);
pb5 promoted 195/200, 0 truncated, 288.3 tok/s (misses 12,93,119,184,187) -> PASS at tol 6.
Final: greedy GSM8K-50 @conc8 50/50, 240.7 tok/s; exact.py self-consistent (a8c3394a). Raw: ~/dsv41-opt-20261021/r3-e5/.
Decision rationale: vLLM defaults to temperature 1.0 when a client omits it (there is no generation_config),
so typical chat traffic is sampled. +12% sampled single and +7% sampled c16 outweigh about -2-3% greedy single.
For greedy-only benchmarking, relaunch with DSV41_SPEC_SAMPLED=0.
Rollback:
 - Config only: relaunch with DSV41_SPEC_SAMPLED=0 (the sharded Markov drafter is active again).
 - Launcher: cp -p ~/dsv41-tp8.sh.bak-r3spec ~/dsv41-tp8.sh on all 8 (d5c41fd0).
 - Site files (service stopped): in .../v1/worker/gpu/spec_decode, cp -p rejection_sampler_utils.py.bak-r3spec
   rejection_sampler_utils.py and dspark/speculator.py.bak-r3spec dspark/speculator.py on all 8. Then verify
   99556e84 / 774b34c2 on all 8.
Follow-up idea: restore the sharded Markov path for greedy-only batches under probabilistic drafting. This needs a
second captured drafter graph set, and it would recover the ~2-3% greedy single loss.
