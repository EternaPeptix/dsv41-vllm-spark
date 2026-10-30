#!/usr/bin/env bash
# run_parity.sh LABEL [steps]  -- run the knapcio-parity benches on spark-one against 127.0.0.1:8000 (model dsv41).
# steps (default "decode nofill speed prefill"): decode  = sparkDash 1.8.8 DecodeBench, knapcio protocol (~3 min)
#                                                nofill  = same prompts at c1 without min_tokens/ignore_eos (~1 min)
#                                                speed   = speedbench 0.3 full, 73 prompts, c1, thinking off (~4-5 min)
#                                                prefill = sparkDash 1.8.8 PrefillBench 4k-128k x3 (~2.5 min)
# Needs an idle server (the clients warn if vllm:num_requests_running > 0). Results: runs/LABEL/.
# Order matters: run decode right after a fresh instance's own exact.py/warm-up, never during an A/B window.
set -euo pipefail
cd "$(dirname "$0")"
L=${1:?label}; STEPS=${2:-"decode nofill speed prefill"}
D=runs/$L; mkdir -p "$D"
curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null || { echo "server not healthy"; exit 1; }
R=$(curl -s -m 5 http://127.0.0.1:8000/metrics | awk '/^vllm:num_requests_running/{s+=$NF} END{print s+0}')
[ "$R" = "0" ] || { echo "server busy ($R running requests): refusing"; exit 1; }
date -u +%FT%TZ > "$D/started"
for s in $STEPS; do
  case $s in
    decode)  python3 -I sparkdash_decode.py --out "$D/sd_decode.jsonl" 2>&1 | tee "$D/sd_decode.log" ;;
    nofill)  python3 -I sparkdash_decode.py --protocol single --conc 1 --no-fill --out "$D/sd_nofill.jsonl" 2>&1 | tee "$D/sd_nofill.log" ;;
    speed)   python3 -I speedbench_c1.py --out "$D/sb_full.jsonl" 2>&1 | tee "$D/sb_full.log" ;;
    prefill) python3 -I sparkdash_prefill.py --out "$D/sd_prefill.jsonl" 2>&1 | tee "$D/sd_prefill.log" ;;
    words)   python3 -I sparkdash_prefill.py --filler words --out "$D/sd_prefill_words.jsonl" 2>&1 | tee "$D/sd_prefill_words.log" ;;
    *) echo "unknown step $s"; exit 2 ;;
  esac
done
date -u +%FT%TZ > "$D/finished"
