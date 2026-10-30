#!/bin/bash
# i4b.sh : extra conc8 ABAB (3 reps per arm, alternating) on instance 4. Run on spark-one after i4.sh.
R=~/dsv41-opt-20261021/sched/runs/i4; B=~/dsv41-sched/bench
while ! grep -q "ALL DONE" $R/progress.txt; do sleep 20; done
setctl() {
  echo "$1" > /dev/shm/dsv41-sched.json.tmp && mv /dev/shm/dsv41-sched.json.tmp /dev/shm/dsv41-sched.json
  for i in $(seq 1 20); do
    curl -s -o /dev/null 127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' -d '{"model":"dsv41","prompt":"hi","max_tokens":2}'
    sleep 1
    grep "dsv41_sched params now\|dsv41_sched armed" ~/dsv41-logs/rank0.log | tail -n 1 | grep -q "$2" && return 0
  done
  exit 1
}
cd ~/dsv41-bench
for k in 1 2 3; do
  for a in off sk; do
    if [ $a = sk ]; then setctl '{"enable": 1}' "'enable': 1, 'short_max'"; else setctl '{"enable": 0}' "'enable': 0,"; fi
    python3 bench.py conc --conc 8 --max-tokens 512 --out $R/c8x_$a.jsonl --label $a.k$k > /dev/null 2>&1
    echo "$(date -u +%T) c8 $a $k" >> $R/progress.txt
  done
done
setctl '{"enable": 1}' "'enable': 1, 'short_max'"
echo "$(date -u +%T) C8 DONE" >> $R/progress.txt
