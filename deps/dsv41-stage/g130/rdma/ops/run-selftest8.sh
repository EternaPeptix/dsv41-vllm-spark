#!/bin/bash
# run-selftest8.sh <tag> [ext, default build/rdma_ar_ext.so]: single-GPU kernel selftest on all 8 at once. STOPPED WINDOW ONLY: refuses
# while dsv41-rank is active or vllm/bench/nsys runs anywhere. Results: results/<tag>.<host>.log on each node.
set -euo pipefail
TAG=${1:?tag}; EXT=${2:-build/rdma_ar_ext.so}
HOSTS="spark-one spark-two spark-three spark-four spark-five spark-six spark-seven spark-eight"
D='$HOME/dsv41-stage/g130/rdma'
for h in $HOSTS; do
  st=$(ssh -n -o ConnectTimeout=10 "$h" 'systemctl --user is-active dsv41-rank 2>/dev/null || true')
  { [ "$st" = active ] || [ "$st" = activating ]; } && { echo "ABORT: dsv41-rank is $st on $h"; exit 1; }
  n=$(ssh -n "$h" 'pgrep -fc "[v]llm serve|[b]ench_|[s]elftest|[n]sys" || true'); [ "${n:-0}" = 0 ] || { echo "ABORT: $h busy ($n)"; exit 1; }
  ssh -n "$h" "cd $D && sha256sum -c --quiet SHA256SUMS" || { echo "ABORT: $h fails SHA256SUMS"; exit 1; }
done
for h in $HOSTS; do
  ssh -n "$h" "cd $D && mkdir -p results && timeout 1500 \$HOME/venvs/vllm032/bin/python -I tests/selftest_g130.py --ext $EXT > results/$TAG.$h.log 2>&1; echo \"$h exit=\$? \$(tail -1 results/$TAG.$h.log)\"" &
done
wait
