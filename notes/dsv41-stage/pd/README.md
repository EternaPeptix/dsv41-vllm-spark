# ~/dsv41-stage/pd: exact decode candidates, offline prep (2026-10-25)

Nothing here is installed. No restart, GPU job, live site-packages edit or launcher edit was made. Every candidate is flag-gated with the default off. Each one has its own README.

Install/uninstall:
- From the Mac: `ssh spark-one cat ~/dsv41-stage/pd/common/pd-all.sh | bash -s install|uninstall|status <tag>`. It requires the same digest on 8/8.
- On one node: `common/pd-site.sh`.
- Each install backs up `<file>.bak-pd-<tag>` and refuses unless the current sha is the recorded baseline or the staged sha.
- `pd-site.sh` refuses while dsv41-rank is active unless `PD_ALLOW_ACTIVE=1`, in which case the files take effect at the next start.

Staged tree digests are identical on 8/8: d05 542f7004a049, pdl d7d4b1cdb41b (trees f4de75ed9be3), rgate e6380dc23aec, l2pf c14bbd8f4a0d, common 62ce2f0b8000.

| rank | tag | what | flag(s) | expected | readiness / evidence |
|---|---|---|---|---|---|
| 1 | d05 | Codex D05 dead FC2 8-row store-group skip, wired into the M32 serving path. All 40 layers by construction; `verify` = live all-layer shadow compare | `DSV41_MOE_FC2_D05=0/1/verify` | C1 -0.6, C2 -1.0, C4 -1.9 ms (2-4%), more at C16/32 | Codex: 490 + 148 component cases, 21,584 live shadow comparisons with 0 mismatches (2 layers). Ours: CPU dispatch PASS. GPU unit test written (40 layers x tokens 1/4/6/48, eager + graphs, poisoned) but not run. JIT cubin pre-seeded |
| 2 | l2pf | L2 prefetch re-plan arms | env only: S, SQ, SR; PB needs install | 1u -0.3..-0.6, 16u -0.2..-0.5 ms | S measured +2% C1 single (1 pinned pair). PB: Codex CPU 10/10 (500 + 500 randomized). `ops/arm.sh` pins AV |
| 3 | rgate | exact cuBLAS-replica router GEMM, M ≤ 8 (24 CTA x 16 warp HMMA.16816 + smem split reduce) | `DSV41_ROUTER_GATE_REPLICA=0/1/verify` + probe config | 1u -0.35..-0.55 ms if exact; 0 at ≥8u | .so e708c229 built (78 regs, 0 spills). CPU PASS. **Bitwise match unproven**: 8 schedule x 7 reduce hypotheses; the 5-min stopped-window probe decides and writes the config |
| 4 | pdl | FC2 early weight (B/SFB) TMA prologue before griddepcontrol.wait (JIT-header tree, same .so) + SiLU `gdc_launch_dependents` | `DSV41_MOE_PDL_EARLY_B=1`, `DSV41_MOE_PDL_TRIGGER=start/end` | 1u -0.12..-0.24 ms | nvcc on the real JIT kernel.cu: FC1 SASS identical, FC2 +prologue, 168 regs, 0 spills. CPU PASS. GPU test (200 bitwise-checked replays per arm) + sanitizer not run. FC1 prologue and glue-kernel arm A not done |
| 5 | fc2cd05 | FC2-compact v2 (r4-e6, bitwise, 1.10x) + D05 | not built | +0.3 ms C1 / +0.9 ms C4 on top of D05 | spark-one only. FC2c v2 applies cleanly to the M32 source; D05 on top fails 4 host hunks (manual merge). Needs native rebuild, payload wiring and +2.5 GiB/rank tail tensors at 1M |

## File conflicts

- `fused_moe/experts/sm121_target_m32.py`: d05 staged 5acc5fde, which also carries the pdl wrapper half (tree selection), so install d05 for pdl. A future fc2cd05 also needs this file.
- `quantization/utils/fp8_utils.py`: pdl b35c0561. Codex P09 dual-quant (prefill) edits the same file, so it would need a merge.
- `fused_moe/router/gate_linear.py`: rgate only (pristine 41adaba8 today).
- `device_communicators/dsv41_l2_prefetch.py`: l2pf PB only. Live 4da7745b already holds S/SQ/SR.
- `deep_gemm_moe.py` and the launcher: untouched by every candidate. The launcher is now 2fd26e53 (Base agent added `--enable-prompt-tokens-details`). All arms pass env through `launch-conc.sh`.

## Suggested order for the Base agent's windows

1. Stopped window (~20 min):
   - `pd-all install d05; install pdl; install rgate; install l2pf`, flags off;
   - `d05/ops/run-gpu-tests.sh`;
   - `pdl/tests/test_pdl_unit.py` + racecheck;
   - `rgate/ops/run-probe.sh`.
2. Restart on the normal config (all flags off, identical to live), then confirm exact.py = a8 anchor and look at the pinned counters.
3. ABAB with pinned AV:
   - D05 (=1) vs control;
   - then the L2PF arms (S, PB, SQ);
   - then bundle α = rgate + pdl (each alone is under the 0.8% C1 bar).
4. Promote only on the gates: C1 at least 0.8% every pair, or conc16/32 at least 2%; GSM8K-50 at least 49; longctx PASS.

## Status after the decode A/B step (2026-10-25 23:25 UTC; details: ~/dsv41-opt-20261021/README.md "pd decode A/B step")
- d05: PROMOTED (launcher default DSV41_MOE_FC2_D05=1). ABAB C1 coding -2.50/-2.61%, prose -1.72/-2.25%, conc16 +3.1/+5.4%, conc32 +4.5/+6.0%; exact identical; GSM8K-50 50/50; longctx PASS.
- l2pf: arm S PROMOTED (launcher default DSV41_L2_PREFETCH_PRUNE_SHARED=1; site file 514dcb23 installed). ABAB on d05: C1 -2.05/-1.71%, prose -2.11/-2.06%, conc neutral. PB -1.0/-1.35% (behind S), SQ = S.
- rgate: KILLED (bitwise at M2-8 but slower than cuBLAS: warm 0.83-0.96x, cold 0.65x); uninstalled 8/8.
- pdl: HOLD (EB pair 1 -0.33% / -0.16%, below the gates); fp8_utils.py b35c0561 installed, flags off.
- fc2cd05: not built.
- common/pd-all.sh: in-loop ssh now uses -n (backup pd-all.sh.bak-pdab).
