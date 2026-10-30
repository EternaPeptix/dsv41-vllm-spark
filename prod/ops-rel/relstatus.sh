#!/bin/bash
# relstatus.sh [rank log] -- one-line startup status of a rank log + /health (rel-2 supervisor helper,
# runs on spark-one). Counts are grep -c line counts; health is the HTTP code of :8000/health.
L=${1:-$HOME/dsv41-logs/rank0.log}
c() { grep -c -e "$1" "$L" 2>/dev/null || true; }
if [ -f "$L" ]; then
  printf 'log=1 dspark=%s greedy=%s avstart=%s avpin=%s avprof=%s capfin=%s start=%s f_rdma=%s f_wd=%s f_eng=%s f_fp=%s f_cuda=%s ' \
    "$(c 'Capturing model for DSpark')" "$(c 'capturing greedy draft graphs')" "$(c 'before_adaptive_calibration')" \
    "$(c 'DSV41 AV profile skipped')" "$(c 'Adaptive verification cost profile:')" \
    "$(c 'Graph capturing finished')" "$(c 'Application startup complete')" \
    "$(c 'timeout waiting for')" "$(c 'Timeout ([0-9]*:[0-9][0-9]:[0-9][0-9])!')" \
    "$(c 'EngineCore failed to start\|Engine core initialization failed\|died unexpectedly')" \
    "$(c 'rdma_ar fingerprint mismatch')" "$(c 'CUDA error')"
else
  printf 'log=0 '
fi
echo "health=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8000/health 2>/dev/null)"
