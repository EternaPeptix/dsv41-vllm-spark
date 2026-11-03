#!/bin/bash
# cpu_check.sh: CPU-only checks of the kit, beside the running service. Nothing here opens the GPU (CUDA_VISIBLE_DEVICES is
# empty), everything runs at nice 19 on one thread, and it stops if MemAvailable falls below the site's 6 GB floor.
#   1. the extension loads, and its glue refuses CPU tensors; the live modules and the patched file import;
#   2. test_det_topk.py --mock end to end, then once per verdict path;
#   3. bs03_install.py against a throwaway copy of the three base files, including its refusal paths.
set -u
KIT=$(cd "$(dirname "$0")/.." && pwd)
PY=$HOME/venvs/vllm032/bin/python
FLOOR_KB=${BS03_MEM_FLOOR_KB:-6291456}
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
OUT=$KIT/out/cpu-check-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"

guarded() {
  setsid nice -n 19 ionice -c3 "$@" &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    local m
    m=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
    if [ "$m" -lt "$FLOOR_KB" ]; then
      echo "MEMORY GUARD: MemAvailable $m kB, killing"
      kill -KILL -- "-$pid" 2>/dev/null
      wait "$pid" 2>/dev/null
      return 99
    fi
    sleep 0.5
  done
  wait "$pid"
}

check_rc() {  # check_rc <name> <expected rc> <got rc> <log>
  if [ "$3" = "$2" ]; then echo "ok    $1 (rc $3): $(grep -a 'VERDICT' "$4" | tail -1 | cut -c1-200)"; else echo "FAIL  $1: rc $3, expected $2"; tail -5 "$4"; FAILS=$((FAILS + 1)); fi
}
FAILS=0

echo "== 1. extension load, CPU only"
guarded "$PY" -I "$KIT/tools/ext_check.py" "$KIT"
rc=$?
[ "$rc" = 0 ] || { echo "FAIL extension load rc=$rc"; FAILS=$((FAILS + 1)); }

echo "== 1b. live modules and the patched file import with the GPU hidden (HOME redirected, no .pyc written)"
for m in base patched-off patched-on; do
  mkdir -p "$OUT/home"
  HOME=$OUT/home TRITON_CACHE_DIR=$OUT/home/triton DSV41_BS03_SO=$KIT/build/vllm_bs03_det_topk_sm121.so guarded "$PY" -I -B "$KIT/tools/live_import_check.py" "$m" "$KIT" > "$OUT/import-$m.log" 2>&1
  rc=$?
  grep -aE '^(ok  |FAIL|   \[|LIVE_IMPORT)|MEMORY GUARD|Error' "$OUT/import-$m.log" | cut -c1-200
  [ "$rc" = 0 ] || { echo "FAIL import check $m rc=$rc"; tail -5 "$OUT/import-$m.log" | cut -c1-200; FAILS=$((FAILS + 1)); }
done

echo "== 2. harness, mock backend"
T="$PY -I $KIT/test_det_topk.py --mock"
guarded $T --out "$OUT/go" > "$OUT/go.log" 2>&1; check_rc "mock, all stages, expects GO" 0 $? "$OUT/go.log"
guarded $T --out "$OUT/hold-us" --stages smoke,timing,exact --replays 2 --mock-det-us 10 > "$OUT/hold-us.log" 2>&1; check_rc "10 us, expects HOLD" 2 $? "$OUT/hold-us.log"
guarded $T --out "$OUT/stop-us" --stages smoke,timing,exact --replays 2 --mock-det-us 14 > "$OUT/stop-us.log" 2>&1; check_rc "14 us, expects STOP" 1 $? "$OUT/stop-us.log"
guarded $T --out "$OUT/stop-mis" --stages smoke,timing,exact --replays 2 --mock-inject mismatch > "$OUT/stop-mis.log" 2>&1; check_rc "injected mismatch, expects STOP" 1 $? "$OUT/stop-mis.log"
guarded $T --out "$OUT/hold-slow" --stages smoke,timing,exact --replays 2 --mock-inject slowshape > "$OUT/hold-slow.log" 2>&1; check_rc "slow shape, expects HOLD" 2 $? "$OUT/hold-slow.log"
guarded $T --out "$OUT/hold-oracle" --stages smoke,timing,exact --replays 2 --mock-inject oracle > "$OUT/hold-oracle.log" 2>&1; check_rc "reference miss, expects HOLD" 2 $? "$OUT/hold-oracle.log"
guarded $T --out "$OUT/hold-inc" --stages timing > "$OUT/hold-inc.log" 2>&1; check_rc "timing only, expects HOLD" 2 $? "$OUT/hold-inc.log"
grep -a -c 'MISMATCH' "$OUT/stop-mis.log" | sed 's/^/mismatch lines in the injected run: /'
ls "$OUT/stop-mis" | grep -c 'mismatch-' | sed 's/^/saved mismatch files: /'
# Refuses a real run when a vLLM process or a wrong base sha is present: the guard must fire, here, on a serving node.
guarded "$PY" -I "$KIT/test_det_topk.py" --stages probe --out "$OUT/refuse" > "$OUT/refuse.log" 2>&1; rc=$?
if [ "$rc" = 3 ] && grep -aq REFUSED "$OUT/refuse.log"; then echo "ok    a real run refuses on this node: $(grep -a REFUSED "$OUT/refuse.log" | cut -c1-120)"; else echo "FAIL  a real run did not refuse: rc $rc"; tail -3 "$OUT/refuse.log"; FAILS=$((FAILS + 1)); fi

echo "== 3. installer on a throwaway copy"
SITE=$HOME/venvs/vllm032/lib/python3.12/site-packages/vllm
C=$OUT/site
mkdir -p "$C/model_executor/layers"
for f in sparse_attn_indexer.py dsv41_topk_ties.py indexer_topk.py; do cp -p "$SITE/model_executor/layers/$f" "$C/model_executor/layers/$f"; done
I="$PY -I $KIT/bs03_install.py"
ok() { if "$@" > "$OUT/inst.log" 2>&1; then echo "ok    $*" | sed "s#$C#<copy>#; s#$KIT/##; s#$PY -I ##"; tail -1 "$OUT/inst.log" | cut -c1-200; else echo "FAIL  $*"; tail -5 "$OUT/inst.log"; FAILS=$((FAILS + 1)); fi; }
no() { if "$@" > "$OUT/inst.log" 2>&1; then echo "FAIL  not refused: $*"; FAILS=$((FAILS + 1)); else echo "ok    refused: $(head -1 "$OUT/inst.log" | cut -c1-140)"; fi; }
ok $I plan --site "$C"
ok $I install --site "$C"
SAI=$C/model_executor/layers/sparse_attn_indexer.py
[ -f "$SAI.bak-bs03" ] && [ -f "$C/model_executor/layers/dsv41_bs03_det_topk.py" ] || { echo "FAIL install left no backup or module"; FAILS=$((FAILS + 1)); }
"$PY" -I -m py_compile "$SAI" && rm -rf "$C/model_executor/layers/__pycache__"
ok $I status --site "$C"
ok $I install --site "$C"   # second install: already installed
ok $I uninstall --site "$C"
cmp "$SAI" "$SITE/model_executor/layers/sparse_attn_indexer.py" && echo "ok    uninstall restored the base bytes"
[ ! -f "$C/model_executor/layers/dsv41_bs03_det_topk.py" ] || { echo "FAIL module left behind"; FAILS=$((FAILS + 1)); }
echo "# changed" >> "$SAI"
no $I plan --site "$C"
no $I install --site "$C"
cp -p "$SITE/model_executor/layers/sparse_attn_indexer.py" "$SAI"
echo "# changed" >> "$C/model_executor/layers/dsv41_topk_ties.py"
no $I install --site "$C"
cp -p "$SITE/model_executor/layers/dsv41_topk_ties.py" "$C/model_executor/layers/dsv41_topk_ties.py"
ok $I install --site "$C"
echo "# changed" >> "$SAI.bak-bs03"
no $I uninstall --site "$C"
rm -rf "$C"
echo "live site untouched: $(cd "$SITE/model_executor/layers" && sha256sum sparse_attn_indexer.py dsv41_topk_ties.py indexer_topk.py | cut -c1-8 | tr '\n' ' '), module $(ls "$SITE/model_executor/layers/dsv41_bs03_det_topk.py" 2>/dev/null | wc -l), backup $(ls "$SITE/model_executor/layers/sparse_attn_indexer.py.bak-bs03" 2>/dev/null | wc -l)"
echo "MemAvailable $(awk '/MemAvailable/{print $2}' /proc/meminfo) kB; logs in $OUT"
[ "$FAILS" = 0 ] && echo CPU_CHECK_OK || echo "CPU_CHECK_FAILS=$FAILS"
