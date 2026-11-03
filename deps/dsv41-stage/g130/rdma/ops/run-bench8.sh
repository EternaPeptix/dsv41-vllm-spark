#!/bin/bash
# run-bench8.sh <tag> <ext relative to the kit> [bench_g130.py args]: 8 ranks, one process per Spark, real fabric, GPU 0. STOPPED WINDOW ONLY:
# refuses while dsv41-rank is active or vllm/bench/nsys runs anywhere. TIMING=0 for a run without in-kernel timers (default 1).
# Results: results/<tag>.rank<R>.{json,log} on each node, <tag>.allranks.json on spark-one.
set -euo pipefail
TAG=${1:?tag}; EXT=${2:?ext}; shift 2
HOSTS="spark-one spark-two spark-three spark-four spark-five spark-six spark-seven spark-eight"
MASTER=${MASTER_ADDR:-10.77.0.11}; PORT=${MASTER_PORT:-29583}; TIMEOUT=${BENCH_TIMEOUT:-2400}
D='$HOME/dsv41-stage/g130/rdma'
for h in $HOSTS; do
  st=$(ssh -n -o ConnectTimeout=10 "$h" 'systemctl --user is-active dsv41-rank 2>/dev/null || true')
  { [ "$st" = active ] || [ "$st" = activating ]; } && { echo "ABORT: dsv41-rank is $st on $h (needs a stopped window)"; exit 1; }
  n=$(ssh -n "$h" 'pgrep -fc "[v]llm serve|[b]ench_|[s]elftest|[n]sys" || true'); [ "${n:-0}" = 0 ] || { echo "ABORT: $h busy ($n)"; exit 1; }
  ssh -n "$h" "cd $D && sha256sum -c --quiet SHA256SUMS" || { echo "ABORT: $h fails SHA256SUMS"; exit 1; }
done
ARGS="$*"
ENVS="GLOO_SOCKET_IFNAME=enp1s0f0np0 NCCL_SOCKET_IFNAME=enp1s0f0np0 RDMA_AR_THP=1 PYTHONNOUSERSITE=1 RDMA_AR_TIMING=${TIMING:-1}"
r=0
for h in $HOSTS; do
  ssh -n "$h" "mkdir -p $D/results && cd $D && nohup env $ENVS RANK=$r WORLD_SIZE=8 MASTER_ADDR=$MASTER MASTER_PORT=$PORT timeout $TIMEOUT \
    \$HOME/venvs/vllm032/bin/python -u bench/bench_g130.py --out results/$TAG --ext $EXT $ARGS > $D/results/$TAG.rank$r.log 2>&1 < /dev/null &" &
  r=$((r + 1))
done
wait
echo "started 8 ranks (tag=$TAG ext=$EXT); waiting..."
while ssh -n spark-one "pgrep -f '[b]ench_g130.py --out' >/dev/null"; do sleep 10; done
sleep 5
ssh -n spark-one "grep -E '^\[|^init|^DONE|Error|Traceback' $D/results/$TAG.rank0.log | tail -150 | cut -c1-200"
r=0; for h in $HOSTS; do echo "$h: $(ssh -n "$h" "tail -1 $D/results/$TAG.rank$r.log" | cut -c1-140)"; r=$((r + 1)); done
echo "== leftovers (must be none)"; for h in $HOSTS; do ssh -n "$h" "pgrep -fa '[b]ench_g130' || true"; done
