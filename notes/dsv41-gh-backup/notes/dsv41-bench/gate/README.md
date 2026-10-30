# dsv41 regression gate (Codex ops kit 08), installed 2026-10-27 (goal window 5, ops step)

One command, on spark-one, against the live server:

    bash ~/dsv41-bench/gate/gate.sh --live [--label L] [--no-1m]    # ~9.5 min; exit 0 PASS / 1 FAIL / 3 REFUSED (busy)
    bash ~/dsv41-bench/gate/gate.sh --plan                           # the steps
    bash ~/dsv41-bench/gate/gate.sh --last                           # newest result

Steps (each stops the gate on a hard failure): refuse_if_busy (nothing running/waiting, no request finished in
5 s) -> exact.py (anchor a8c3394af9d2) -> G1 one pass (tok/s + outputs digest, compared with the previous PASS
under the same launcher; informational, because an unpinned production launch can land on either side of the
known prose near-tie) -> ONE cold 1M probe (xrb4 solo, TTFT <= 220 s = good class, MemAvailable min on
spark-one) -> GSM8K-200 (>= 194; reference 196) -> longctx needle + copy 4K-200K (LONGCTX PASS) -> kl_panel
verify / smoke / capture --sets tools,math (81,840 teacher-forced positions; compared --tier equiv with the
previous gate capture under the same launcher, when there is one).
Not in it (nightly set): full KL capture, G1 x3, G4, speed A/Bs.

Results: runs/<launcher sha8>/<UTC>.json (summary) + runs/<launcher sha8>/<UTC>/ (raw). Two launchers never
share a directory. A 2 s /metrics sampler (sampler.txt) and per-step requests sent/finished flag foreign
traffic (the LAN/tailnet API has no auth).

History
- 2026-10-27 19:51 UTC, f214c2c9, instance g5ops-PROD (production, AV unpinned, right after the spark-three
  reboot test): PASS in 570 s. exact a8c3394af9d28249; G1 101.81, digest abcdf1eb96654d71 (code 596b478a,
  math 92e5d8cf, prose e8a524f3); GSM8K-200 195/200 (misses 93,100,119,182,184); longctx 60/60 needles, copy
  1.000; 1M TTFT 168.26 s (good class), MemAvailable min 5.98 GB; KL baseline gate-f214c2c9-20261027T195149Z.
  That run used the kit's order (1M probe last). Its "foreign" flags on longctx and probe_1m are an artifact
  of the first script (it counted 1 request per longctx row and 1 for the probe; the real numbers are 2 per
  row and 3 = xrb4's two 8K settle requests + the 1M request); the sampler shows no foreign request.
- Script v2 (same day): the 1M probe now runs before GSM8K/longctx. After the 200K longctx requests the 1M
  prefill bottomed at MemAvailable 5.98 GB on spark-one (floor 6 GB), against 8.6-8.7 GB when it runs first,
  as in every earlier verification suite (PyTorch caching-allocator state left by the 200K requests). v1 is
  kept as gate.sh.v1.
