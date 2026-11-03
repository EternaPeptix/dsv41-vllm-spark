#!/bin/bash
# run-bench8.sh <tag> [bench_l2win.py args]: run FROM THE MAC with bash, STOPPED WINDOW ONLY (refuses while dsv41-rank is active anywhere or any vllm /
# bench / nsys process runs). 8 ranks, one process per Spark, real fabric, production rdma_ar_ext.so through the installed wrapper. Installs nothing.
# Results: ~/dsv41-stage/g130/l2win/results/<tag>.rank<R>.{json,log}; the rank-0 log is printed at the end. NOT RUN YET.
set -euo pipefail
TAG=${1:?usage: run-bench8.sh <tag> [bench args]}; shift || true
HOSTS="spark-one spark-two spark-three spark-four spark-five spark-six spark-seven spark-eight"
MASTER=${MASTER_ADDR:-10.77.0.11}
PORT=${MASTER_PORT:-29573}
TIMEOUT=${BENCH_TIMEOUT:-2400}
D='$HOME/dsv41-stage/g130/l2win'
for h in $HOSTS; do
  st=$(ssh -n -o ConnectTimeout=10 "$h" 'timeout 10 systemctl --user is-active dsv41-rank 2>/dev/null || true')
  if [ "$st" = active ] || [ "$st" = activating ]; then echo "ABORT: dsv41-rank is $st on $h (needs a stopped window)"; exit 1; fi
  n=$(ssh -n "$h" 'pgrep -fc "[v]llm serve|[b]ench_|[s]elftest|[n]sys" || true')
  [ "${n:-0}" = 0 ] || { echo "ABORT: $h still has vllm/bench/nsys processes ($n)"; exit 1; }
  ssh -n "$h" "cd $D && timeout 30 sha256sum -c --quiet SHA256SUMS" || { echo "ABORT: $h stage dir fails SHA256SUMS"; exit 1; }
done
ARGS="$*"
# the served rdma_ar env, every rank: the launcher's own export lines for these prefixes (AG_COLS sits on a mixed line)
ENVSRC='eval "$(grep -E "^export (RDMA_AR_|DSV41_RDMA)|^export .*DSV41_RDMA_AG_COLS" $HOME/dsv41-tp8.sh)"'
NCCLENV="NCCL_SOCKET_IFNAME=enp1s0f0np0 GLOO_SOCKET_IFNAME=enp1s0f0np0 NCCL_IB_HCA=rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1 NCCL_IB_GID_INDEX=5 NCCL_DEBUG=WARN"
r=0
for h in $HOSTS; do
  ssh -n "$h" "mkdir -p $D/results && cd $D && $ENVSRC && nohup env $NCCLENV RANK=$r WORLD_SIZE=8 MASTER_ADDR=$MASTER MASTER_PORT=$PORT \
    PATH=/usr/local/cuda/bin:\$PATH timeout -k 10 $TIMEOUT \$HOME/venvs/vllm032/bin/python -u bench/bench_l2win.py --out results/$TAG $ARGS \
    > $D/results/$TAG.rank$r.log 2>&1 < /dev/null &" &
  r=$((r + 1))
done
wait
echo "started 8 ranks (tag=$TAG); waiting..."
while ssh -n spark-one "pgrep -f '[b]ench_l2win.py --out' >/dev/null"; do sleep 10; done
sleep 5
echo "== rank0 log"; ssh -n spark-one "grep -v -E '^(INFO|WARNING|DEBUG)|^\(' $D/results/$TAG.rank0.log | tail -150"
echo "== per-rank last line"
r=0; for h in $HOSTS; do echo "$h: $(ssh -n "$h" "tail -1 $D/results/$TAG.rank$r.log" | cut -c1-200)"; r=$((r + 1)); done
echo "== leftovers (must be none)"; for h in $HOSTS; do ssh -n "$h" "pgrep -fa '[b]ench_l2win' || true"; done
