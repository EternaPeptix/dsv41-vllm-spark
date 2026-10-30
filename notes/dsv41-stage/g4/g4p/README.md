# g4p: goal window 4, exact prefill (Codex items 04 / 05 / 06), staged 2026-10-27

Every feature is default OFF; with no flag set the five files must reproduce the live outputs (exact.py a8 + G1 digest).

| flag | item | what |
|---|---|---|
| `DSV41_PREFILL_SP=exact` | 04 | Codex sp_exact_logical_rows.patch (SP v2 exact + logical-row split hooks + per-(layer, op, rows) bitwise self-check with full-M fallback + DEBUG top-k replay) on live model.py e7264781, plus fixes: the DEBUG top-k replay shadows `indexer_op.forward` (assigning a function to an nn.Module child raised TypeError), bitwise (uint8-view) equality, no self-check verdict from dummy/autotune forwards, compare line reports bitwise flags, top-k replay count and self-check tallies. `DSV41_PREFILL_SP_DEBUG=N`, `DSV41_PREFILL_SP_SELF_CHECK=1|once|0`, `DSV41_PREFILL_SP_LAYERS=k`. |
| `DSV41_E7_EXACT=post_and_draft` | 05 | On an eager step whose every request is a prefill chunk with more than max(max_num_batched_tokens, 128) prompt tokens left after it (so neither F nor F-1): layer 20 write_kv + the 128-row replay run as stock; skipped: the scatter back, MTP buffer copy, hc_collapse, final norm and the DSpark propose. No scheduler change. |
| `DSV41_CAND_ONLY_INDEXER=scatter\|verify` | 06 | Consumer indexers 24/28/32/36, prefill only (decode untouched): pack the 2,048 candidate blocks, score them with the same DeepGEMM kernel (compressed logits), scatter into a -inf full row, stock top-k + CANON. `verify` serves stock and compares the scatter logits bitwise in shadow (log only). |

Per-request opt-outs (same-instance A/B): `vllm_xargs` `{"dsv41_sp_off": 1}`, `{"dsv41_e7_off": 1}`, `{"dsv41_cand_off": 1}`.

Install (service stopped): `ssh spark-one cat ~/dsv41-stage/g4/common/g4-all.sh | bash -s install g4p` (backups `<file>.bak-g4-g4p`); `uninstall g4p` restores.
Tools: `tools/mk_g4p.py` (04+05: `base` dir of live files + Codex-patched model.py -> out), `tools/mk_g4p06.py` (06), `tests/q06.py` (GPU qualification of 06, service stopped), `scripts/` probes (spx.py, ndprobe.py, sptid.py, e7x.py from pf).
