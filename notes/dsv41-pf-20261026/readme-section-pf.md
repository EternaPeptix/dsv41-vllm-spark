
## pf prefill A/B step (2026-10-26 02:05-05:45 UTC): nothing promoted. E7 output-changing, SP v2 not yet exact, gather-add crashes, NCCL arms closed

Scope: the prepared prefill candidates of "pd / prefill offline prep" (`~/dsv41-stage/pd/`), run serially. Every A/B arm used the AV pin `~/dsv41-stage/pdab/av-pin-pdfinal.json`. Raw results on spark-one: `~/dsv41-pf-20261026/`, which holds `STATE.md` (running log), `runs/<label>/` (one per instance) and `scripts/`:
- `psuite.sh` / `psumm.py`: prefill suite and summary;
- `lexact.py`, `lpcmp.py`, `ndprobe.py`: long-prompt and nondeterminism probes;
- `e7x.py`, `e7xp.py`: E7 bitwise tests;
- `spx.py`, `sp*probe.sh`: SP probes with exact row counts;
- `xrb.py`: `xr_bench` plus an E7 opt-out flag;
- `tiddet.py`, `finaltests.sh`;
- installers `e7launcher.sh` and `splauncher.sh` (not used).

Mac scratchpad `pf/` holds the drivers (`runarm.sh`, `chain2.sh`, `e7xrun.sh`, `s1run.sh`, `sxrun*.sh`, `sp1run.sh`, `sp2run.sh`, `b2x.sh`, `teardown.sh`, `finalize.sh`), plus `summ-*.json`, `sp_<label>-rank<i>.txt` (SP compare lines) and `tele/`.

**Live state after this step:** instance `pf-FINAL`, healthy 05:24:14 UTC (+366 s, supervised, first try), launcher **04b6738e** defaults (AV **unpinned**, as before the step). Every site file is back to its pre-step sha256 and identical on 8/8: model.py e7264781, model_state.py 53e4fe39, decoder_replay_layers.py 2fedc0a6, dspark/speculator.py bbcee9e3, moe_runner.py 20b69fe5, deep_gemm_utils.py a4a0486a, sm121_target_m32.py 5acc5fde, sm121_d05_verify.py c04c363d, dsv41_l2_prefetch.py 514dcb23, fp8_utils.py b35c0561 (inert), model_runner.py 316fca44, dsv41_rel.py f373f911, rdma_all_reduce.py 9056dcdf, dspark.py 36811ca9, gate_linear.py 41adaba8; no rgate. Scheduler: `~/dsv41-sched/dsv41_sched.py` 6074389f, no dsv41_pd_policy.py.
- Removal: `pd-all.sh uninstall spg` then `uninstall pf` (digests ca01af51f9c5 / b5760b58fa09, identical 8/8), then `sched-install.sh uninstall` on 8/8.
- Ranks: 8/8 active, 0 restarts, 0 ERROR/Traceback lines, fingerprint c2f008432bd4769f, no test environment.
- exact.py: **9330f03c4f97**, the documented AV-profile alternate (p2 2228532c, p4 8c5e765e), counters 914/464/2320/1373 from its own profile. It gave the same hash again after all the other probes. The same code gives a8c3394af9d2 under the pin (A1/A2 with the pf files inert; X-ray B-age), so this is the alternate profile, not a code change.
- `--enable-prompt-tokens-details` is still on: `cached_tokens` 0 -> 4160 on a repeated 4,205-token chat (streaming and /v1/completions carry it too).
- Quick longctx 16K/32K PASS 10/10, copy 1.000.
- 1M solo 182.86 s with a good chunk curve (2.68 / 2.98 / 3.11 s), so the live instance is currently a good long-context launch.
- Stock token-id determinism probe: 3,000 ids gave 3/3 identical first-token logprobs; 16,384 ids gave 3/3 different (-5.84 / -5.95 / -5.55).

### Two findings that change how prefill changes must be tested
1. **Within-instance prefill nondeterminism (pre-existing, not caused by any pf flag).**
   - The same prompt, sent to one instance with distinct `cache_salt` values, gives an identical first-token logprob only up to about 4.1K prompt tokens: 4,112 tokens gave 3/3 identical; 4,145 / 4,196 / 6,056 / 9,071 tokens each gave 3/3 different.
   - At 40K tokens the first token itself differs between copies. At 128K / 300K the text diverges at token 3 / 7 (A1 `lexact` pair).
   - exact.py only sends short prompts, so it never sees this.
   - **Consequence:** long-prompt exactness cannot be judged by comparing copies or instances. It needs the deterministic regime (rows ≤ ~4.1K, e.g. small chunks) or in-forward compares.
   - **One localised source:** layer 2 (the first ratio-2 sparse-indexer layer) on 16K-row token-id prompts varies from run to run (SPK3: rel_l2 2.4e-4 and 2.0e-4 for two identical prompts). On the stock FINAL instance the same 16,384-id prompt gave 3/3 different first-token logprobs (-5.84 / -5.95 / -5.55), while 3,000 ids gave 3/3 identical.
2. **The rank-5 long-context straggler (X-ray step) is not prevented by the AV pin.**
   - In this step, pinned launches A1 and A2 were slow: 1M TTFT 227.6 / 235.9 s, with 600-700K chunks at 4.0 / 4.2 s.
   - Pinned launches B1, B2 and SP1, SP2 were good: 170-174 s with E7 or SP, and 182.3 s for the stock-tail copy on the same good instance.
   - Tally with the X-ray launches: pinned (this step 4/2, X-ray B and P 2/0), so 6 good / 2 slow in all, unpinned 0/4 distinct launches in the X-ray (pdab-FINAL, C, E, R1) plus this step's FINAL, which was **good** (182.86 s): 1/5 overall, one-sided Fisher p ~0.09. The pin raises the odds but does not prevent the straggler.
   - E7 does not change the launch class: on the good B2 instance, the stock-tail copy was also good.
   - 1 Hz GPU power does not identify the slow rank: on slow A2, all 8 ranks show late/early power 0.79-0.82.

### Per-candidate results (pinned AV; TTFT = cold solo, fresh nonce; deltas vs the flags-off arms A1/A2)

| # | candidate (flag) | test | result | verdict |
|---|---|---|---|---|
| 1 | NCCL arms (env via launcher hook) | `ar_bench`, service stopped, 8 ranks, [T,5120] bf16 | control 16K rows 11.41-11.46 ms (busbw 25.6-25.7 GB/s); QPS4+split 12.51 (worse); QPS2 11.79; BUFFSIZE 16M 11.34 (same sha); 8 channels 11.38 (sha differs); 4 channels 12.18; PROTO=Simple 11.34 / ALGO=Ring 11.40 (same sha as control, so already the defaults); Tree 60.5 | **CLOSED.** No exact arm ≥ 5% of AR time; no launcher hook installed |
| 2 | E7 decoder-tail skip (`DSV41_E7_TAIL=1`) | ABAB A1 B1 A2 B2; B2X; E7X / E7XP (1024-token chunks, deterministic regime) | see "E7" below | **NOT PROMOTED: changes output** |
| 3 | SP v2 exact (`DSV41_PREFILL_SP=exact`) + new split-match gate (`spg`) | DEBUG compares S1, SX, SP1, SPK1-3, SP2; timing SP1/SP2 | see "SP v2" below | **NOT PROMOTED: not yet bitwise.** Biggest exact lever if fixed: -6.1% TTFT |
| 4 | gather-add fold (`DSV41_MOE_GATHER_ADD=1`) | GPU gate (`ga_gpu2.py`, spark-eight); serving G1 | kernel bitwise 15/15, fused 0.78x (M16384 7079 -> 5542 µs). Serving: **startup crash on both tries** (profile_run, `shared_experts.py:155` assert) | **KILLED as staged** |
| 5 | P09 dual quant | - | not run: estimated -0.5%, cannot reach the 3% gate; does not stack with the fold | not run |
| 6 | lone-prefill 24K chunks | - | not run: HOLD on memory; output differs for prompts > 16K (chunk boundaries change), so it is not exact | not run |

Note: the staged `moe-glue/test_gather_add_gpu.py` is invalid. It imports `test_gather_add_interp`, whose `setdefault` forces `TRITON_INTERPRET=1`, so it runs in the interpreter with round-toward-zero and reports 682,485 false mismatches. The self-contained `~/dsv41-pf-20261026/scripts/ga_gpu2.py` is the real GPU gate.

**E7 (decoder-tail skip on prefill chunks that are neither the final one nor the one before it).**
- Performance and gates (2 instances per arm):
  - 128K TTFT 21.617 / 21.645 -> 20.566 / 20.566 s (**-4.9% / -5.0%**).
  - 16K unchanged.
  - Flat-band chunk 2.68 -> 2.52 s (-157 ms).
  - Like-for-like 1M on one instance (B2X): stock tail 182.34 s -> E7 170.54 / 170.36 s (**-6.5%**). The per-chunk saving grows from -152 ms to -223 ms because tail-indexer growth is skipped.
  - Starvation unchanged (E7 skips only lone prefills).
  - C1 32.40 / 27.67 and 32.42 / 27.90 (A) vs 32.57 / 27.68 and 32.44 / 27.68 ms (B), AL identical; conc8 within noise.
  - exact.py a8c3394af9d2 with 905/471/2355/1373 on every instance.
  - GSM8K-50 50/50; longctx 60/60 (copy 1.000); 1M needle 7/7 (copy exact).
- **E7X failed.** Setup: E7 on, `--max-num-batched-tokens=1024`, 2,187-4,009-token prompts (3-4 chunks), deterministic regime.
  - The stock copies (per-request `dsv41_e7_off`) were bitwise identical 3/3 at every length.
  - The E7 copies differed from stock, including the first-token logprob: 2,187 tokens -0.3499 vs -0.6346; 4,009 tokens -0.3889 vs -0.2836.
  - E7XP (prefix-cache variant) failed the same way.
- **Cause.** The replay layers do not own only SWA KV. They also write the indexer K caches of the index-source layers 20/24/28/32/36 (`metadata_prefixes` includes `attn.indexer.k_cache.prefix`).
  - Each chunk's replay writes entries for its last 128 rows, and those entries persist.
  - The top-k at layers 24-39 for the final rows reads them.
  - E7 skips those writes, so the final rows see fewer index-K entries.
  - The `write_batch_kv` call arguments match the stock call; this is a design premise error, not a call-site bug.
- **Exact variant (not built):** keep the replay cache writes for skipped chunks and skip only the post-replay full-T work plus the DSpark draft (about 84 of the 168 ms per chunk).

**SP v2 (layers 0-19 row-sharded across TP; attention and MoE on gathered rows).**
- **Ungated (S1, SX).** In-forward DEBUG compares are far from bitwise on 2,048-8,065-row forwards: hidden rel_l2 0.11-0.19, identical across ranks and copies.
  - Cause 1: `compute_mhc_pre_num_splits` returns 1/4/16 from `n_sms // cdiv(rows, 64)`. The stock path on T rows and the shard on T/8 rows pick different mHC pre-norm split-K counts unless the shard has enough row tiles.
  - The resulting last-bit difference flips routing and top-k, and the error cascades.
  - The new pd tag **spg** (model.py fe7a7d12 -> 8f2904e6, installed for SP1-SP2 only) gates SP on matching split counts.
- **Gated (SP1, SPK1-2).** Layers [0,1) and [0,2) are **bitwise** at 12,296 / 14,000 / 16,380 rows, including the Engram layer 1. So the SP plumbing is exact.
- **Full 20 layers, gated (SP2, chat prompts).**
  - **Bitwise** at 15,022 and 16,032 rows (and S1's 16,032 x2).
  - **Not bitwise** at 12,760 / 13,544 / 14,241 rows (hidden rel_l2 0.073-0.079).
  - So a second row-count-dependent kernel configuration switches between shard sizes 1,781 and 1,878 rows. It is not identified yet.
  - Token-id prompts additionally hit finding 1 at layer 2.
- **Timing (no E7, pinned, good launch).**
  - TTFT 16K 2.523 (**-6.2%**); 128K 20.297 (20.305 / 20.288, **-6.1% / -6.2%**).
  - 1M 172.29 s vs like-for-like stock 182.34 (-5.5%); flat chunks 2.508 vs 2.672-2.683 s.
  - Starvation long 22.78 vs 24.86-24.95 s.
  - Decode unchanged: C1 32.474 / 27.806 ms, conc8 235.4.
  - SP2 repeated this: 16K 2.540, 128K 20.362 (20.418 / 20.307, -5.8% / -5.9%), 1M 174.21 s (good launch). GSM8K-50 50/50, longctx 60/60 (copy 1.000), 1M needle 7/7 (copy exact), 0 errors. SP2's exact.py gave e0feee387e43 (only p3 flipped: e4460f26 vs cd93c706; counters 888/491/2455/1373). It ran after 5 chat probes, and SP cannot engage on exact.py's short prompts (under 2,048 rows, and the gate needs about 12.3K), so this is request-history dependence of the decode path, not SP; S1 and SP1, also after probes, gave a8. exact.py must run first on a fresh instance

### Per instance (pinned unless noted)

| instance | config | exact / counters | C1 coding / prose ms (AL 5.224 / 2.718 on all) | conc8 | TTFT 16K / 128K / 500K / 1M (s) | 1M chunks c2-12 / c37-43 / c50-62 (s) | 1M launch | starvation: long TTFT / short p50 / p95 (s) / decoder gap p50 (ms) |
|---|---|---|---|---|---|---|---|---|
| A1 | flags off | a8c3394af9d2 / 905/471/2355/1373 | 32.396 / 27.668 | 240.9 | 2.690 / 21.617 / 89.98 / 227.57 | 2.681 / 4.013 / 4.838 | slow | 24.86 / 1.667 / 1.979 / 428 |
| B1 | E7 | a8c3394af9d2 / 905/471/2355/1373 | 32.572 / 27.684 | 236.7 | 2.685 / 20.566 / 82.45 / 170.72 | 2.524 / 2.763 / 2.884 | good | 24.84 / 1.659 / 1.960 / 420 |
| A2 | flags off | a8c3394af9d2 / 905/471/2355/1373 | 32.416 / 27.896 | 236.9 | 2.690 / 21.645 / 90.21 / 235.85 | 2.683 / 4.244 / 5.030 | slow | 24.95 / 1.683 / 2.034 / 430 |
| B2 | E7 + gates | a8c3394af9d2 / 905/471/2355/1373 | 32.442 / 27.682 | 239.4 | 2.689 / 20.566 / 82.29 / 170.54 | 2.520 / 2.759 / 2.885 | good | 24.89 / 1.667 / 1.970 / 426 |
| S1 | E7 + SP exact ungated (DEBUG, quick) | a8c3394af9d2 / 905/471/2355/1373 | - / - | - | 2.521 / 19.319 / - / - | - / - / - | - | - / - / - / - |
| SP1 | SP exact + split gate | a8c3394af9d2 / 905/471/2355/1373 | 32.474 / 27.806 | 235.4 | 2.523 / 20.297 / 82.47 / 172.29 | 2.508 / 2.803 / 2.938 | good | 22.78 / 0.973 / 2.567 / 426 |
| SP2 | SP exact + split gate + gates | e0feee387e43 / 888/491/2455/1373 | 32.360 / 27.668 | 234.0 | 2.540 / 20.362 / 82.86 / 174.21 | 2.514 / 2.838 / 2.943 | good | 22.93 / 0.791 / 1.078 / 428 |
| B2X | B2 instance, 1M with per-request `dsv41_e7_off` (stock tail), then E7 again | - | - | - | - / - / - / 182.34 (stock) / 170.36 (E7 again) | 2.672 / 2.973 / 3.108 (stock) | good | - |
| FINAL (live) | launcher 04b6738e defaults, **unpinned**, all pf files removed | 9330f03c4f97 (known AV-profile alternate, p2/p4) / 914/464/2320/1373; same again after all probes | - | - | - / - / - / 182.86 | 2.678 / 2.975 / 3.112 | good | - |

### Rollback and leftovers
- **Live config:** launcher 04b6738e, unchanged. Everything above was env-only or installed and removed within the step (see live state).
- **Staged, not installed:** `~/dsv41-stage/pd/{pf,spg,sched-pd}` on 8/8, plus the installers `~/dsv41-pf-20261026/scripts/{e7launcher.sh,splauncher.sh}` (never run).
- **Re-install with the service stopped:** `ssh spark-one cat ~/dsv41-stage/pd/common/pd-all.sh | bash -s install pf` (then `install spg`), and on each node `bash ~/dsv41-stage/pd/sched-pd/sched-install.sh install`.

### Next steps
1. **SP v2 is the largest exact prefill lever: -6.1% TTFT from 16K to 1M, decode unchanged.**
   - Find the second row-count-dependent kernel (bitwise from 15,022 rows, not at 12,760-14,241) with a `DSV41_PREFILL_SP_LAYERS` bisect over layers 2-19, using ~13.5K-row chat prompts.
   - Compare one full 16,384-row chunk with a chat prompt.
   - Then gate or fix it, and run ABAB; the SP1/SP2 timings carry over.
2. **E7 exact variant:** keep the replay indexer-K writes on skipped chunks.
3. **Gather-add:** the arm must peek `_output[_output_idx]` instead of reading the consuming `output` property. Worth at most ~1.2%.
4. **Root-cause the long-prompt nondeterminism (finding 1),** starting with the layer-2 indexer top-k tie set. It blocks bitwise testing of every long-prompt change.
5. **Straggler:** it is random per launch and the pin does not prevent it. The relaunch-if-slow 700K screen remains the practical guard.
