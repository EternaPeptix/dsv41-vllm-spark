#!/bin/bash
# run-device8-r7.sh [all|quick]: device_test7.py on stopped nodes (service must be inactive on every node), in the
# mandatory environment. Node i runs the r6 c1 part (ranks i and i+3) plus at most one more part (review item 6: no node
# runs more than two parts back to back): spark-one r6 mutants, spark-two proj (ranks 0-3), spark-three proj (ranks 4-7),
# spark-four stage, spark-five r6 c2, spark-six projtime (INFO only: never a gate, never a PASS that proves anything),
# spark-seven projmut (rank 0), spark-eight projmut (rank 5). 16 parts; each prints R7_DEVICE_TEST
# PASS|FAIL|REFUSED|INCOMPLETE as its last line.
set -uo pipefail
TS=$HOME/Documents/dsv41-kits/ts-ssh/bin
[ -x "$TS/ssh" ] && export PATH=$TS:$PATH
WHAT=${1:-all}
HOSTS=(spark-one spark-two spark-three spark-four spark-five spark-six spark-seven spark-eight)
# r8-comm2fix (review B2): ENVP mirrors the production launcher 4f3cc1c6's glue env and must equal device_test6.ENV_PREFIX
# (device_test7 --cpu-selfcheck checks that; its preflight refuses a node whose env or launcher defaults differ).
ENVP='source ~/venvs/vllm032/bin/activate; export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda DSV41_GLUE_RQPAD=1 DSV41_GLUE_RPSPLIT=split'
K='~/dsv41-stage/r7comm2/kit'
for h in "${HOSTS[@]}"; do
  st=$(ssh -n -o ConnectTimeout=10 "$h" 'systemctl --user is-active dsv41-rank 2>/dev/null || true')
  { [ "$st" = active ] || [ "$st" = activating ]; } && { echo "ABORT: dsv41-rank is $st on $h"; exit 1; }
  ssh -n "$h" "cd $K && sha256sum -c --quiet SHA256SUMS.stage" || { echo "ABORT: $h fails SHA256SUMS.stage"; exit 1; }
done
TMO=${DEVICE_TIMEOUT:-9000}
Q=$([ "$WHAT" = quick ] && echo --quick)
run() {  # host part extra-args
  ssh -n -o ServerAliveInterval=30 "$1" "$ENVP; cd $K && mkdir -p results && timeout $TMO python -I device_test7.py --part $2 \
    --cand prebuilt/rdma_ar_ext.so $Q $3 > results/device7-$2-$1.log 2>&1; echo \"$1 $2 rc=\$? \$(tail -1 results/device7-$2-$1.log)\""
}
for i in "${!HOSTS[@]}"; do
  h=${HOSTS[$i]}
  ( run "$h" r6 "--r6-part c1 --ranks $i,$(( (i + 3) % 8 ))"
    case $h in
      spark-one) run "$h" r6 "--r6-part mutants" ;;
      spark-two) run "$h" proj "--ranks 0,1,2,3" ;;
      spark-three) run "$h" proj "--ranks 4,5,6,7" ;;
      spark-four) run "$h" stage "" ;;
      spark-five) run "$h" r6 "--r6-part c2" ;;
      spark-six) run "$h" projtime "" ;;
      spark-seven) run "$h" projmut "--ranks 0" ;;
      spark-eight) run "$h" projmut "--ranks 5" ;;
    esac ) &
done
wait
