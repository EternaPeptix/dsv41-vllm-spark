#!/bin/bash
# guard_run.sh <logfile> -- <command...>
# Runs a CPU build step at nice 19 in its own session and enforces the memory floor. Samples MemAvailable every 100 ms
# and SIGKILLs the whole session as soon as it is below GUARD_FLOOR_KB (default 8000000 kB). Refuses to start below it.
# Exit: the command's status, 75 = stopped by the guard, 76 = refused to start.
# CUDA_VISIBLE_DEVICES is emptied: nothing started here may open the GPU.
# THE FLOOR IS NOT ENOUGH DURING A RELAUNCH: vLLM needs 0.90 x 121.63 = 109.47 GiB free on the device at startup, and on GB10 that
# is the same memory. On 2026-11-02 a 4.9 GB compile here made rank 1 fail that check (109.0 GiB) with MemAvailable at 113 GB.
# Never run a build while a relaunch may be in progress (WINDOW.md).
set -u
LOG=${1:?log}; shift; [ "${1:-}" = "--" ] && shift
FLOOR=${GUARD_FLOOR_KB:-8000000}
avail() { while read -r k v _; do [ "$k" = "MemAvailable:" ] && { echo "$v"; return; }; done < /proc/meminfo; }
a0=$(avail)
echo "guard: $(date -u +%FT%TZ) floor=${FLOOR} kB start MemAvailable=${a0} kB cmd: $*" >> "$LOG"
if [ "$a0" -lt "$FLOOR" ]; then echo "guard: REFUSED, MemAvailable ${a0} < floor ${FLOOR}" | tee -a "$LOG"; exit 76; fi
CUDA_VISIBLE_DEVICES= setsid nice -n 19 "$@" >> "$LOG" 2>&1 &
pid=$!
# an outer `timeout` must not orphan the session (found the hard way: a hung mutant kept spinning after timeout killed us)
trap 'kill -KILL -- "-$pid" 2>/dev/null; echo "guard: terminated, session killed" >> "$LOG"; exit 143' TERM INT HUP
min=$a0; rc=""
while kill -0 "$pid" 2>/dev/null; do
  a=$(avail)
  [ "$a" -lt "$min" ] && min=$a
  if [ "$a" -lt "$FLOOR" ]; then
    { echo "guard: BREACH $(date -u +%FT%TZ) MemAvailable=${a} kB < floor ${FLOOR} kB; killing session ${pid}"
      ps -o pid,rss,comm --sid "$pid" 2>/dev/null | head -12; } >> "$LOG"
    kill -KILL -- "-$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
    echo "guard: stopped by floor; min MemAvailable=${min} kB, used about $((a0 - min)) kB" | tee -a "$LOG"
    exit 75
  fi
  sleep 0.1
done
wait "$pid"; rc=$?
echo "guard: done rc=${rc} min MemAvailable=${min} kB, headroom used about $((a0 - min)) kB" | tee -a "$LOG"
exit "$rc"
