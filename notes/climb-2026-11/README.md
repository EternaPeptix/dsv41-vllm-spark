# G1 climb records, 2026-11-05 to 2026-11-12

These files are the raw evidence behind the climb table in the top-level README.

- `promotions/<tag>/`: one folder per launcher promotion.
  - `run.log` records the launcher edit on 8/8, the supervised relaunch, the verify line (G1, digest, `exact.py`,
    9-prompt determinism check) and the lock release.
  - `g1.json` is the verify G1 summary.
  - `promote-r8head` is the first HEAD attempt. A foreign request shared a batch with prose rep 0 during its verify
    run, so the digest check rolled it back. `promote-r8head2` is the isolated retry, which shipped as `4a04b175`.
- `windows/<window>/`: the serving A/B arms whose result was promoted. Controls (`c*`) bracket the treatment arms,
  and each arm line carries its G1, digest, `exact.py` and determinism result.
  - `w1b`: paced attention/all-gather prefetch and `wo_b`
  - `w2b`: DTGRAPH, R1 prefetch and IDXPAR (the `p5`/`p5b` arms also set `DSV41_MHC_OVERLAP_DRAFT=1`, which was
    not promoted)
  - `w3b`: the five round-2 kernel levers
  - `w4`: XSPLIT plus attention pace 200
  - `r5-w1`, `r5-w2`: glue
  - `r8-w1`: COLEMB and CMPPRIO
  - `r7-w1-c1`: RDMA trim
  - `r8-head`: LM-head screen and refine
- `campaign-20261105-REPORT.md`: the 5 November campaign (FC1 tile deal +1.21%, FC2 barrier 1024 +0.15%, and the
  levers it found dead).

The G1 records for the current launcher, the OBLIT migration evaluation (`obl-A` = official weights,
`obl-B` = official + projections = production, `obl-C` = previous checkpoint) and the 7 November gate summary are in
`../results-20261112/`.
