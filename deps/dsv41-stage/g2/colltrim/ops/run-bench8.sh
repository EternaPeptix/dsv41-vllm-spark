#!/bin/bash
# run-bench8.sh <tag> [bench_g2ct.py args] (g2 colltrim) — run FROM THE MAC with bash, ONLY in a stopped
# window (refuses while dsv41-rank is active anywhere or vllm/bench/nsys runs). 8 ranks, one process per
# Spark, real fabric, staged files only (nothing installed). Default phases A,B,W (~6-10 min).
# Results: ~/dsv41-stage/g2/colltrim/results/<tag>.rank<R>.{json,log}; rank-0 log printed at the end.
set -euo pipefail
TAG=${1:?usage: run-bench8.sh <tag> [bench args]}; shift || true
HOSTS="spark-one spark-two spark-three spark-four spark-five spark-six spark-seven spark-eight"
MASTER=${MASTER_ADDR:-10.77.0.11}
PORT=${MASTER_PORT:-29563}
TIMEOUT=${BENCH_TIMEOUT:-1800}
D='$HOME/dsv41-stage/g2/colltrim'
for h in $HOSTS; do
  st=$(ssh -n "$h" 'systemctl --user is-active dsv41-rank 2>/dev/null || true')
  if [ "$st" = active ] || [ "$st" = activating ]; then echo "ABORT: dsv41-rank is $st on $h (needs a stopped window)"; exit 1; fi
  n=$(ssh -n "$h" 'pgrep -fc "[v]llm serve|[b]ench_|[s]elftest|[n]sys" || true')
  [ "${n:-0}" = 0 ] || { echo "ABORT: $h still has vllm/bench/nsys processes ($n)"; exit 1; }
  ssh -n "$h" "cd $D && sha256sum -c --quiet SHA256SUMS" || { echo "ABORT: $h stage dir fails SHA256SUMS"; exit 1; }
done
ARGS="$*"
NCCLENV="NCCL_SOCKET_IFNAME=enp1s0f0np0 GLOO_SOCKET_IFNAME=enp1s0f0np0 NCCL_IB_HCA=rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1 NCCL_IB_GID_INDEX=5 NCCL_DEBUG=WARN"
FWD=""
for v in RDMA_AR_MAX_REDUCE_BYTES RDMA_AR_TWOSHOT_MIN_BYTES RDMA_AR_VEC_PER_CTA; do
  eval "val=\${$v:-}"; [ -n "$val" ] && FWD="$FWD $v=$val"
done
r=0
for h in $HOSTS; do
  ssh -n "$h" "mkdir -p $D/results && cd $D && nohup env $NCCLENV $FWD RANK=$r WORLD_SIZE=8 MASTER_ADDR=$MASTER \
    MASTER_PORT=$PORT timeout $TIMEOUT \$HOME/venvs/vllm032/bin/python -u bench/bench_g2ct.py --out results/$TAG $ARGS \
    > $D/results/$TAG.rank$r.log 2>&1 < /dev/null &" &
  r=$((r + 1))
done
wait
echo "started 8 ranks (tag=$TAG, fwd:${FWD:- none}); waiting..."
while ssh -n spark-one "pgrep -f '[b]ench_g2ct.py --out' >/dev/null"; do sleep 10; done
sleep 5
echo "== rank0 log"; ssh -n spark-one "grep -v -E '^(INFO|WARNING|DEBUG)|^\(' $D/results/$TAG.rank0.log | tail -120"
echo "== per-rank last line"
r=0; for h in $HOSTS; do echo "$h: $(ssh -n "$h" "tail -1 $D/results/$TAG.rank$r.log")"; r=$((r + 1)); done
echo "== leftovers (must be none)"; for h in $HOSTS; do ssh -n "$h" "pgrep -fa '[b]ench_g2ct' || true"; done
