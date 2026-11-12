#!/bin/bash
# run-tests.sh <cpu|gpu> [host] (g2 colltrim) — run FROM THE MAC with bash.
#   cpu  (default host spark-four; safe while serving: nice -n 19, no CUDA context): SASS gate vs the live
#        k1 build + the planner test + the column-gather glue test.
#   gpu  (default host spark-two; STOPPED window only: refuses while dsv41-rank is active or any vllm/bench
#        process runs on that host): tests/test_cols_selftest.py (column gather, single GPU, no network) and
#        tests/test_l2pf_g2_gpu.py (paced kernel + g2 fork/join under CUDA graphs). ~3-4 min, ~2 GB.
# Logs: ~/dsv41-stage/g2/colltrim/results/<test>.<host>.log on that host (tail printed here).
set -euo pipefail
MODE=${1:?usage: run-tests.sh cpu|gpu [host]}
D='$HOME/dsv41-stage/g2/colltrim'
PY='$HOME/venvs/vllm032/bin/python'
if [ "$MODE" = cpu ]; then
  h=${2:-spark-four}
  ssh -n "$h" "cd $D && nice -n 19 $PY tests/sass_check_g2.py \$HOME/rdma_ar_dsv41/build-k1pl/rdma_ar_ext.so \
    build/rdma_ar_ext.so > results/sass_check.$h.log 2>&1; tail -1 results/sass_check.$h.log; \
    CUDA_VISIBLE_DEVICES= nice -n 19 $PY tests/test_plan_g2_cpu.py > results/test_plan_g2_cpu.$h.log 2>&1; \
    tail -1 results/test_plan_g2_cpu.$h.log; \
    CUDA_VISIBLE_DEVICES= nice -n 19 $PY tests/test_cols_logic_cpu.py > results/test_cols_logic_cpu.$h.log 2>&1; \
    tail -1 results/test_cols_logic_cpu.$h.log"
  exit 0
fi
h=${2:-spark-two}
st=$(ssh -n "$h" 'systemctl --user is-active dsv41-rank 2>/dev/null || true')
if [ "$st" = active ] || [ "$st" = activating ]; then echo "ABORT: dsv41-rank is $st on $h"; exit 1; fi
n=$(ssh -n "$h" 'pgrep -fc "[v]llm serve|[b]ench_|[t]est_.*gpu|[s]elftest|[n]sys" || true')
[ "${n:-0}" = 0 ] || { echo "ABORT: $h has vllm/bench/test processes ($n)"; exit 1; }
for t in test_cols_selftest test_l2pf_g2_gpu; do
  ssh -n "$h" "cd $D && mkdir -p results && timeout 1200 $PY tests/$t.py > results/$t.$h.log 2>&1; \
    echo \"$h $t exit=\$? \$(tail -1 results/$t.$h.log)\""
done
ssh -n "$h" "cd $D && grep -E 'FAIL|cases bitwise|replayed|saves|issued|L2 effect|cold:|graph us|RESULT' results/test_cols_selftest.$h.log results/test_l2pf_g2_gpu.$h.log | cut -c1-220"
