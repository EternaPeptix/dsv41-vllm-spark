#!/bin/bash
# runarm_sampled.sh <label> [quick] — on spark-one, server up. Sampled + greedy measurement for one arm.
# Writes ~/dsv41-bench/r3spec/<label>.jsonl and gsm8k gate files <label>.g06.*  (~25 min full, ~10 min quick)
set -u
L=$1; Q=${2:-}
cd ~/dsv41-bench; mkdir -p r3spec; O=r3spec/$L.jsonl
python3 bench.py single --max-tokens 64 --repeats 1 --types code --preset t1 >/dev/null   # warm top-p/temp kernels
python3 bench.py single --repeats 2 --label $L.greedy --out $O >/dev/null                 # greedy regression
python3 bench.py single --repeats 2 --preset t1  --label $L.t1  --out $O >/dev/null
python3 bench.py single --repeats 2 --preset t06 --label $L.t06 --out $O >/dev/null
for c in 4 8 16; do
  [ -n "$Q" ] && [ $c != 8 ] && continue
  python3 bench.py conc --conc $c --preset t1 --label $L.t1.c$c --out $O >/dev/null
done
[ -z "$Q" ] && python3 bench.py conc --conc 8 --label $L.greedy.c8 --out $O >/dev/null
python3 ~/dsv41-bench/gsm8k_sampled_gate.py run r3spec/$L.g06 --n 50 --conc 8 --temp 0.6 --top-p 1.0 --runs 2 --seed 1000
python3 summ_sampled.py r3spec/$L.jsonl
