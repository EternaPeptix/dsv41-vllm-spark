#!/bin/bash
# runarm_conc.sh <dir> <label> [gsmconc] — one arm for the conc experiment (adds conc12/conc16 + memavail sampling).
set -u
D=$1; L=$2; G=${3:-}
mkdir -p "$D"; cd ~/dsv41-bench
memsnap() { for h in spark-one spark-two spark-three spark-four spark-five spark-six spark-seven spark-eight; do echo "$1 $h $(ssh -n -o ConnectTimeout=5 $h 'grep MemAvailable /proc/meminfo' | awk '{print $2}')"; done >> "$D/memavail.txt"; }
memsnap start
python3 bench.py single --max-tokens 1024 --repeats 1 --types code,prose,math --out "$D/single.jsonl" --label "$L" > "$D/single.log" 2>&1
python3 bench.py conc --conc 4 --max-tokens 512 --out "$D/conc4.jsonl" --label "$L" > "$D/conc4.log" 2>&1
python3 bench.py conc --conc 8 --max-tokens 512 --out "$D/conc8.jsonl" --label "$L" > "$D/conc8.log" 2>&1
python3 bench.py conc --conc 12 --max-tokens 512 --out "$D/conc12.jsonl" --label "$L" > "$D/conc12.log" 2>&1 &
BP=$!; sleep 20; memsnap during12; wait $BP
python3 bench.py conc --conc 16 --max-tokens 512 --out "$D/conc16.jsonl" --label "$L" > "$D/conc16.log" 2>&1 &
BP=$!; sleep 20; memsnap during16; wait $BP
python3 exact.py "$D/exact.json" "$L" > "$D/exact.log" 2>&1
if [ -n "$G" ]; then python3 ~/gsm8k_gate.py "$D/gsm8k.jsonl" 50 "$G" > "$D/gsm8k.log" 2>&1; fi
memsnap end
echo DONE > "$D/DONE"
