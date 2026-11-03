#!/bin/bash
# dev_one.sh <slice name, e.g. m0_s0>: one guarded device-only compile of dev/<name>.cu, bounded by `timeout`, at nice 19 (via
# guard_run.sh). Refuses to start unless MemAvailable >= GUARD_FLOOR_KB + START_MARGIN_KB. Never retries. Prints one line.
# Exit: 0 built, 3 no start margin, 75 stopped by the guard, 124 timeout, else the compiler's status.
set -u
cd "$(dirname "$0")/../dev" || exit 1
b=$1
FLOOR=${GUARD_FLOOR_KB:-8000000}; MARGIN=${START_MARGIN_KB:-150000}; T=${DEV_TIMEOUT:-120}
[ -s "$b.cubin" ] && { echo "$b: already built"; exit 0; }
a=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
if [ "$a" -lt $((FLOOR + MARGIN)) ]; then echo "$b: not started, MemAvailable $a kB under start margin"; exit 3; fi
timeout -k 5 "$T" bash ../ops/dev_build.sh "$b.cu" "$b.cubin" "../logs/dev-$b.log" > /dev/null 2>&1; rc=$?
echo "$(date -u +%T) $b rc=$rc start=$a $(grep -E '^guard: (done|stopped)' "../logs/dev-$b.log" | tail -1 | cut -c8-80)"
exit "$rc"
