#!/bin/bash
# r1-attn CPU tests. Usage: tests/run_all.sh [python]   (512S1: ~/exo-venv313/bin/python3, torch 2.11 CPU)
PY=${1:-python3}
cd "$(dirname "$0")/.." || exit 1
rc=0
for t in tests/test_install.py tests/test_plan.py tests/test_mutations.py; do
  echo "=== $t"
  $PY -W ignore "$t" 2>&1 | grep -v "^DSV41_IDXPAR" || true
  $PY -W ignore "$t" >/dev/null 2>&1 || rc=1
done
echo "=== overall rc=$rc"
exit $rc
