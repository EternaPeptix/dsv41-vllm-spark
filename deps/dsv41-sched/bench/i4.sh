#!/bin/bash
# i4.sh : in-instance ABAB on the promoted launcher (sk = env defaults, off = enable 0). Run on spark-one.
R=~/dsv41-opt-20261021/sched/runs/i4; mkdir -p $R; B=~/dsv41-sched/bench
setctl() {  # $1 json, $2 expected substring in the params log line
  echo "$1" > /dev/shm/dsv41-sched.json.tmp && mv /dev/shm/dsv41-sched.json.tmp /dev/shm/dsv41-sched.json
  for i in $(seq 1 20); do
    curl -s -o /dev/null 127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' -d '{"model":"dsv41","prompt":"hi","max_tokens":2}'
    sleep 1
    grep "dsv41_sched params now\|dsv41_sched armed" ~/dsv41-logs/rank0.log | tail -n 1 | grep -q "$2" && { echo "$(date -u +%T) ctl $1" >> $R/progress.txt; return 0; }
  done
  echo "ctl FAILED $1" >> $R/progress.txt; exit 1
}
SK='{"enable": 1}'; OFF='{"enable": 0}'
setctl "$SK" "'enable': 1, 'short_max': 4096, 'long_cap': 0, 'scan': 32, 'hold': 1,"; bash $B/dsuite.sh $R/d_sk1 sk1 perf
setctl "$OFF" "'enable': 0,"; bash $B/dsuite.sh $R/d_off1 off1 perf
bash $B/dsuite.sh $R/d_off2 off2 perf
setctl "$SK" "'enable': 1, 'short_max': 4096, 'long_cap': 0, 'scan': 32, 'hold': 1,"; bash $B/dsuite.sh $R/d_sk2 sk2 perf
bash $B/dsuite.sh $R/c_sk sk corr
cd ~/dsv41-opt-20261021/sched/runs && python3 $B/abrun.py i4 $B/plan_i4.json
echo "$(date -u +%T) ALL DONE" >> $R/progress.txt
