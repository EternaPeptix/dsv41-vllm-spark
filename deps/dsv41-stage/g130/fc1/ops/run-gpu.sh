#!/bin/bash
# g130 fc1 stopped-window GPU steps on this node. Refuses if the service is active (g1lib.guard) or the shared JIT caches change.
# Usage: cd ~/dsv41-stage/g130/fc1 && bash ops/run-gpu.sh probe|exact|race|sync|rec [extra args]
# Results: results/<step>-<host>-<stamp>.{jsonl,log}. Explicit timeouts: a hang here is a KILL for a kernel arm.
set -u
K=$HOME/dsv41-stage/g130/fc1
cd "$K" || exit 1
source "$HOME/venvs/vllm032/bin/activate"
export PATH=/usr/local/cuda/bin:$PATH   # compute-sanitizer is not on the PATH of ssh -n
# race and sync: every G of the model's steps up to 16 -> 17, once each at tokens 6 (nmb 36). Timings under a sanitizer mean nothing.
SAN_G=0-18; SAN_R=19
HOSTN=$(hostname); ST=$(date -u +%Y%m%dT%H%M%SZ); step=${1:?step}; shift
case "$HOSTN" in *one) R=0;; *two) R=1;; *three) R=2;; *four) R=3;; *five) R=4;; *six) R=5;; *seven) R=6;; *eight) R=7;; *) echo "unknown host"; exit 2;; esac
mkdir -p results
O=results/$step-$HOSTN-$ST
snap() { ( cd "$HOME/dsv41-stage/g1/m32-bw/jit/all-base" && find cache -type f ! -name .committed | sort | xargs sha256sum ) | sha256sum | cut -c1-16; }
before=$(snap)
case "$step" in
  probe) timeout -k 20 2700 python bench/fc1_probe.py --tp-rank $R "$@" > "$O.jsonl" 2> "$O.log"; rc=$? ;;
  exact) timeout -k 20 3000 python tests/test_deal_gpu.py --tp-rank $R "$@" > "$O.jsonl" 2> "$O.log"; rc=$? ;;
  race)  timeout -k 20 2400 compute-sanitizer --tool racecheck --error-exitcode 9 python tests/test_deal_gpu.py --tp-rank $R --arms "${1:-deal}" --tokens 6 --replays $SAN_R --gs $SAN_G --sanitizer --real-layer -1 --phase direct --force-gpu-shared > "$O.jsonl" 2> "$O.log"; rc=$? ;;
  sync)  timeout -k 20 2400 compute-sanitizer --tool synccheck --error-exitcode 9 python tests/test_deal_gpu.py --tp-rank $R --arms "${1:-deal}" --tokens 6 --replays $SAN_R --gs $SAN_G --sanitizer --real-layer -1 --phase direct --force-gpu-shared > "$O.jsonl" 2> "$O.log"; rc=$? ;;
  rec)   timeout -k 20 600 python tests/test_route_rec_gpu.py "$@" > "$O.jsonl" 2> "$O.log"; rc=$? ;;
  *) echo "usage: run-gpu.sh probe|exact|race|sync|rec"; exit 2 ;;
esac
after=$(snap)
[ "$before" = "$after" ] || { echo "CACHE-CHANGED production JIT cache all-base moved: $before -> $after"; rc=70; }
case "$step" in
  race|sync)  # the sanitizer shares stdout with the test, so the last line is not the status
    grep -h '"event": "summary"\|ERROR SUMMARY' "$O.jsonl" "$O.log" | cut -c1-300
    [ "$rc" = 0 ] && ! grep -q '"status": "PASS"' "$O.jsonl" && { echo "NO-PASS rc 0 but no PASS line in $O.jsonl"; rc=71; } ;;
  *) tail -1 "$O.jsonl" | cut -c1-300 ;;
esac
echo "$HOSTN $step rc=$rc $O"
exit "$rc"
