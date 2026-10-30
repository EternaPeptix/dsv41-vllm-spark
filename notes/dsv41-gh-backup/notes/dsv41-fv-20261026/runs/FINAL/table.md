| metric | BASELINE | FINAL | delta |
|---|---|---|---|
| matched C1 coding ms/step (label live2055 = BASELINE prompts, x10) | 33.574 (AL 5.13) | 32.116 (sd 0.134, AL 5.226) | -4.3% (better) |
| matched C1 prose ms/step (label live2055, x10) | 28.620 (AL 2.70) | 27.567 (sd 0.338, AL 2.729) | -3.7% (better) |
| matched C1 coding / prose ms/step (label pdab, x10) | pd pre-step pinned 33.555 / 28.680 (AL 5.224 / 2.718) | 32.254 / 27.545 (AL 5.165 / 2.7) | -3.9% (better) / -4.0% (better) |
| single code tok/s (1024 tok x2) | 87.9 | [91.06, 90.8] mean 90.93 (AL [2.554]) | +3.4% (better) |
| single prose tok/s (1024 tok x2) | 69.66 | [72.77, 72.7] mean 72.73 (AL [1.809]) | +4.4% (better) |
| single math tok/s (1024 tok x2) | 120.85 | [124.57, 124.7] mean 124.63 (AL [3.531]) | +3.1% (better) |
| conc4 aggregate tok/s (512 tok x2) | 153.9 (per-req 50.2 / min 40.6) | [163.3, 162.4] mean 162.8 (per-req 54.5 / min 41.22) | +5.8% (better) |
| conc8 aggregate tok/s (512 tok x2) | 228.8 (per-req 37.2 / min 30.0) | [239.5, 238.4] mean 238.9 (per-req 39.9 / min 30.15) | +4.4% (better) |
| conc16 aggregate tok/s (512 tok x2) | 336.0 (per-req 28.5 / min 21.9) | [353.1, 355.2] mean 354.2 (per-req 29.9 / min 22.68) | +5.4% (better) |
| conc32 aggregate tok/s (512 tok x2) | 497.2 (per-req 20.9 / min 15.6) | [515.6, 512.5] mean 514.1 (per-req 21.9 / min 16.87) | +3.4% (better) |
| cold solo TTFT 16k s | ~2.7 (16K chunk 2.71-2.81) | [2.758, 2.629, 2.739] mean 2.709 (pt [16441]) | -1.5% (better) |
| cold solo TTFT 128k s | ~22 (sk solo 21.72-21.76) | [21.82, 21.749] mean 21.785 (pt [131215]) | +0.2% (worse) |
| cold solo TTFT 1M s | ~205-228 (sk mean 227.0) | 229.106 (pt 1024611, 63 chunks; launch slow) | +0.9% (worse) |
| 1M per-chunk s: c2-12 / c20-30 / c37-43 / c50-62 | flat 2.67-2.81, ~3.4 near 1M (good) / 5.0-5.2 (slow) | 2.687 / 2.982 / 4.044 / 4.907 | - |
| 128K starvation: long TTFT / short p50 / p95 / max s | 24.9 / 1.73-1.80 / 2.2 (sk, 2026-10-24) | 24.936 / 1.733 / 2.051 / 2.087 | - |
| exact.py #1 | a8c3394af9d2 | a8c3394af9d2 (902/474/2370/1373) | - |
| exact.py #2 | a8c3394af9d2 | a8c3394af9d2 (902/474/2370/1373) | - |
| exact.py #3 | a8c3394af9d2 | a8c3394af9d2 (902/474/2370/1373) | - |
| GSM8K-50 | 50/50 | 49/50 | - |
| longctx (full) | PASS 60/60 | LONGCTX PASS needle=60/60 copy_mean=1.000 gate(<= 128k: needle>=0.9, copy>=0.9) fails=none | - |
