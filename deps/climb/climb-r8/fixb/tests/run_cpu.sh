#!/usr/bin/env bash
# r8-fixb CPU tests (no GPU). Run from the kit root on a host with python3 + torch CPU + pytest (512S1:
# PY=~/exo-venv313/bin/python3 PYTHONPATH=~/climb-r1/pydeps STEP_R5=~/climb-r6/climb-r5). Exit 0 = all pass.
set -u
K=$(cd "$(dirname "$0")/.." && pwd); PY=${PY:-python3}; rc=0
run() { echo "== $1"; shift; ( "$@" ) || rc=1; }
run "step pytest (r6 + r8)" bash -c "cd $K/step && $PY -m pytest -q -p no:cacheprovider tests/"
run "step device_test --cpu-selftest" bash -c "cd $K/step && $PY device_test.py --cpu-selftest | tail -1 | grep -q ' 0 failed'"
run "skew r8 T4c/audit" bash -c "cd $K/skew && $PY -m pytest -q -p no:cacheprovider tests/test_r8_t4c.py"
for t in test_cfg_guard test_dequant test_refine_logic; do run "head $t" bash -c "cd $K/head && $PY tests/$t.py | tail -1"; done
run "compose on copies" python3 "$K/integrate/compose_fixb.py"
run "fleet driver (local transport)" bash "$K/integrate/test_fleet.sh"
run "kit pins" python3 "$K/install.py" pins
echo "CPU tests: $([ $rc = 0 ] && echo ALL PASS || echo FAILURES)"; exit $rc
