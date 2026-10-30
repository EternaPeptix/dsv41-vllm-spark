#!/bin/bash
# g1/m32-bw GPU runs for the STOPPED-SERVICE window (one node at a time or all 8 in parallel; each is single-GPU):
#   bash ops/run-gpu-tests.sh [unit|site|bench|all]      (default all; results in results/)
#   unit  tests/test_m32bw_unit.py: every variant bitwise vs live (eager + graph replays, poisoned workspaces)
#   site  tests/test_site_loader.py per variant: the staged site file's own loader path, bitwise vs live
#   bench bench/m32bw_bench.py on recorded routes (M 1/6/12/24/48, c4rec) + bench/project.py
# Shared caches (serving M32 / D05 JIT caches, ~/.cache/vllm, ~/.triton) are listed before/after: must be unchanged.
set -u
G=$HOME/dsv41-stage/g1/m32-bw
cd "$G" || exit 1
WHAT=${1:-all}
[ "$(systemctl --user is-active dsv41-rank 2>/dev/null)" != active ] || { echo "RUN FAIL dsv41-rank active"; exit 1; }
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" ] || { echo "RUN FAIL GPU busy"; exit 1; }
[ -f PREP.ok ] || { echo "RUN FAIL no PREP.ok"; exit 1; }
source $HOME/venvs/vllm032/bin/activate
case $(hostname) in spark-one) R=0;; spark-two) R=1;; spark-three) R=2;; spark-four) R=3;;
  spark-five) R=4;; spark-six) R=5;; spark-seven) R=6;; spark-eight) R=7;; *) R=0;; esac
TS=$(date -u +%Y%m%dT%H%M%SZ); O=$G/results; mkdir -p "$O"
snap() { for d in $HOME/dsv41-stage/m32/target-m32-jit-cache-v1 $HOME/dsv41-stage/pd/d05/jit-cache $HOME/.cache/vllm $HOME/.triton/cache; do
  [ -d "$d" ] && (cd "$d" && find . -type f ! -name .committed -printf '%p %s\n' | sort) | sed "s@^@$d @"; done | sha256sum | cut -c1-16; }
pre=$(snap)
rc=0
if [ "$WHAT" = unit ] || [ "$WHAT" = all ]; then
  timeout 2400 python tests/test_m32bw_unit.py --tp-rank $R > "$O/unit-$TS.jsonl" 2> "$O/unit-$TS.err"; r=$?; rc=$((rc|r))
  echo "UNIT rc=$r $(tail -1 "$O/unit-$TS.jsonl")"
fi
if [ "$WHAT" = site ] || [ "$WHAT" = all ]; then
  for v in pair pf la ds all; do
    DSV41_MOE_M32_G1=$v timeout 900 python tests/test_site_loader.py > "$O/site-$v-$TS.jsonl" 2> "$O/site-$v-$TS.err"; r=$?; rc=$((rc|r))
    echo "SITE $v rc=$r $(tail -1 "$O/site-$v-$TS.jsonl")"
  done
fi
if [ "$WHAT" = bench ] || [ "$WHAT" = all ]; then
  timeout 3600 python bench/m32bw_bench.py --tp-rank $R > "$O/bench-$(hostname)-$TS.jsonl" 2> "$O/bench-$(hostname)-$TS.err"; r=$?; rc=$((rc|r))
  echo "BENCH rc=$r $(grep -c '"timing"' "$O/bench-$(hostname)-$TS.jsonl") timing rows"
  python3 bench/project.py "$O/bench-$(hostname)-$TS.jsonl" | tee "$O/project-$(hostname)-$TS.txt"
fi
post=$(snap)
[ "$pre" = "$post" ] && echo "CACHES unchanged ($pre)" || { echo "CACHES CHANGED $pre -> $post"; rc=1; }
echo "RUN $(hostname) rc=$rc results $O/*-$TS*"
exit $rc
