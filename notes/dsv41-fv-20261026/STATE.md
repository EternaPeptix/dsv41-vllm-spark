# fv final verification 2026-10-26 (Prefill+decode window)
06:29 live = mm-FINAL (started 06:15:01, healthy 06:20:52), launcher 04b6738e 8/8, 12 prior requests (Marlin step exact.py x2), API idle
06:30-06:37 static checks pre/ (vcheck.sh 8/8): launcher, cmdline flag, env, site shas, manifest ad97d72f180e, 18 backups hash ae661a994ec7, fp c2f008432bd4769f, maps, engram 8/8 OK
06:37:36-06:53:20 runs/FINAL (fsuite.sh): exact a8c3394af9d2 x3 (902/474/2370/1373), usage cached 0->4160, matched live2055 32.116/27.567 (AL 5.226/2.729), pdab 32.254/27.545, single 90.93/72.73/124.63, conc4/8/16/32 162.8/238.9/354.2/514.1, TTFT 16K 2.709 128K 21.785 1M 229.106 (slow class, c37-43 4.04 s), starve long 24.94 p50 1.733 p95 2.051, GSM8K-50 49/50 (idx 12), longctx PASS 60/60; no foreign traffic
06:54 post/ checks 8/8 identical to pre (0 diff lines), units active 0 restarts 0 errors
06:58 README 1c26ed08 (backup README.md.bak-pd-fv = ae153061): pointer line + section "Prefill+decode window"
