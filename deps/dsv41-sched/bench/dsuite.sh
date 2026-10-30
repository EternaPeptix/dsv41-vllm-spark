#!/bin/bash
# dsuite.sh <outdir> <label> <perf|corr|all> : decode non-regression (matched C1, bench single/conc8/conc32)
# and/or correctness (exact.py, GSM8K-50 conc8, longctx default). Run on spark-one.
set -u
D=$1; A=$2; M=$3; mkdir -p $D; cd ~/dsv41-bench
step() { echo "$(date -u +%H:%M:%S) $1" >> $D/progress.txt; }
step warm; python3 bench.py single --max-tokens 64 --repeats 1 --types code --preset t1 > /dev/null 2>&1
if [ "$M" != corr ]; then
  step matched; python3 ~/dsv41-stage/r4/rhys-ep/matched_bench.py --plan coding:1x5,prose:1x3 --out $D/matched.json --label $A > $D/matched.log 2>&1
  step single; python3 bench.py single --max-tokens 1024 --repeats 2 --types code,prose,math --out $D/single.jsonl --label $A > $D/single.log 2>&1
  for c in 8 32; do step conc$c; python3 bench.py conc --conc $c --max-tokens 512 --out $D/conc$c.jsonl --label $A.c$c > $D/conc$c.log 2>&1; done
fi
if [ "$M" != perf ]; then
  step exact; python3 exact.py $D/exact.json $A > $D/exact.log 2>&1
  step gsm50; python3 ~/gsm8k_gate.py $D/gsm8k50.jsonl 50 8 > $D/gsm8k50.log 2>&1
  step longctx; python3 longctx.py --out $D/longctx.jsonl > $D/longctx.log 2>&1
fi
step DONE; echo DONE > $D/DONE
