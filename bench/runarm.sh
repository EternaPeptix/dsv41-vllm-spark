#!/bin/bash
# runarm.sh <dir> <label> [gsm] — run one A/B arm against the live server on spark-one.
set -u
D=$1; L=$2; G=${3:-}
mkdir -p "$D"; cd ~/dsv41-bench
python3 bench.py single --max-tokens 1024 --repeats 1 --types code,prose,math --out "$D/single.jsonl" --label "$L" > "$D/single.log" 2>&1
python3 bench.py conc --conc 4 --max-tokens 512 --out "$D/conc4.jsonl" --label "$L" > "$D/conc4.log" 2>&1
python3 bench.py conc --conc 8 --max-tokens 512 --out "$D/conc8.jsonl" --label "$L" > "$D/conc8.log" 2>&1
python3 exact.py "$D/exact.json" "$L" > "$D/exact.log" 2>&1
if [ -n "$G" ]; then python3 ~/gsm8k_gate.py "$D/gsm8k.jsonl" 50 8 > "$D/gsm8k.log" 2>&1; fi
echo DONE > "$D/DONE"
