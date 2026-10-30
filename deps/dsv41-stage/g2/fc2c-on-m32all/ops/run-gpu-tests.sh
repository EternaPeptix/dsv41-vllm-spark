#!/bin/bash
# run-gpu-tests.sh -- g2 fc2c-on-m32all, STOPPED-SERVICE WINDOW, run from the Mac (bash 3.2):
#   ssh spark-one cat ~/dsv41-stage/g2/fc2c-on-m32all/ops/run-gpu-tests.sh | bash
#   ... | QUICK=1 bash        # 1 layer per rank, tokens 1,2,6,24 (~1/5 of the time)
#   ... | BENCH=0 bash        # unit test only;  BENCH_RANK7=0: bench on spark-one only
# Steps: preflight 8/8 (service inactive, GPU idle, PREP.ok) -> prep-node pre (private caches) -> unit test on 8 ranks
# in parallel (5 real layers each = all 40 target layers) -> bench on spark-one (rank 0) and spark-eight (rank 7) in
# parallel -> prep-node post (serving caches unchanged) -> summary. Nothing is installed or seeded (the serving JIT
# cache is the pre-JIT entry built on spark-four; the unit test proves it is a cache HIT for the runtime key).
# Results: ~/dsv41-stage/g2/fc2c-on-m32all/results/{unit,bench}-<TS>.rank<R>.{jsonl,err} on each node.
set -u
HOSTS="spark-one spark-two spark-three spark-four spark-five spark-six spark-seven spark-eight"
G='$HOME/dsv41-stage/g2/fc2c-on-m32all'
PY='$HOME/venvs/vllm032/bin/python'
TS=$(date -u +%Y%m%dT%H%M%SZ)
SSH="ssh -o ConnectTimeout=15 -o BatchMode=yes"
fail=0
for h in $HOSTS; do
  st=$($SSH -n $h "systemctl --user is-active dsv41-rank; nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l; test -f $G/PREP.ok && echo prep-ok || echo no-prep" 2>&1 | tr '\n' ' ')
  set -- $st
  case "${1:-?}" in inactive|failed|unknown) ;; *) echo "[$h] service not stopped: $st"; fail=1;; esac
  [ "${2:-x}" = 0 ] || { echo "[$h] GPU not idle: $st"; fail=1; }
  [ "${3:-x}" = prep-ok ] || { echo "[$h] no PREP.ok: $st"; fail=1; }
done
[ $fail = 0 ] || { echo "RUN-GPU-TESTS abort: stop the service on 8/8 first (and prep-node.sh build)"; exit 2; }
for h in $HOSTS; do $SSH -n $h "bash $G/ops/prep-node.sh pre" 2>&1 | sed "s/^/[$h] /"; done
r=0
for h in $HOSTS; do
  if [ "${QUICK:-0}" = 1 ]; then
    args="--layers all --layer-offset $((r * 5)) --layer-stride 40 --tp-rank $r --tokens 1,2,6,24 --replays 1"
  else
    args="--layers all --layer-offset $r --layer-stride 8 --tp-rank $r"
  fi
  $SSH -n $h "cd $G && $PY tests/test_g2_unit.py $args > results/unit-$TS.rank$r.jsonl 2> results/unit-$TS.rank$r.err" &
  r=$((r + 1))
done
wait
pass=0
r=0
for h in $HOSTS; do
  s=$($SSH -n $h "tail -1 $G/results/unit-$TS.rank$r.jsonl" 2>/dev/null)
  echo "[$h] unit rank$r: $s"
  case "$s" in *'"status": "PASS"'*) pass=$((pass + 1));; esac
  r=$((r + 1))
done
echo "UNIT $pass/8 PASS ($TS)"
if [ "${BENCH:-1}" = 1 ]; then
  if [ $pass != 8 ]; then
    echo "BENCH skipped: unit test not 8/8 PASS"
  else
    $SSH -n spark-one "cd $G && $PY tests/bench_g2.py --tp-rank 0 > results/bench-$TS.rank0.jsonl 2> results/bench-$TS.rank0.err" &
    [ "${BENCH_RANK7:-1}" = 1 ] && $SSH -n spark-eight "cd $G && $PY tests/bench_g2.py --tp-rank 7 > results/bench-$TS.rank7.jsonl 2> results/bench-$TS.rank7.err" &
    wait
    for hr in spark-one:0 spark-eight:7; do
      h=${hr%:*}; rk=${hr#*:}
      $SSH -n $h "test -f $G/results/bench-$TS.rank$rk.jsonl && $PY -I $G/tests/bench_summary.py $G/results/bench-$TS.rank$rk.jsonl" 2>/dev/null | sed "s/^/[$h] /"
    done
  fi
fi
for h in $HOSTS; do $SSH -n $h "bash $G/ops/prep-node.sh post" 2>&1 | sed "s/^/[$h] /"; done
if [ $pass = 8 ]; then
  echo "RUN-GPU-TESTS PASS 8/8 ($TS). Next (service still stopped): ops/g2-all.sh install, then pinned ABAB with DSV41_MOE_FC2C=1."
else
  echo "RUN-GPU-TESTS FAIL ($pass/8 PASS): nothing installed; see results/unit-$TS.rank*.{jsonl,err}"; exit 1
fi
