#!/bin/bash
# guard_run.sh <logfile> -- <command...>   (from the g130 rdma kit, plus a service-state check)
# Runs a CPU build step at nice 19 in its own session, one job, with CUDA_VISIBLE_DEVICES empty. Samples MemAvailable every
# 100 ms and SIGKILLs the session below GUARD_FLOOR_KB (default 7500000 kB). Refuses to start below it.
# It also stops if dsv41-rank stops being active, because vLLM needs 109.47 GiB free at startup and a compile beside a
# relaunch made a rank fail that check on 2026-11-02 (read-only is-active call, never changes the unit). GUARD_NO_SVC=1
# skips that check, for a stopped window.
# Exit: the command's status, 75 = stopped by the guard, 76 = refused to start.
set -u
LOG=${1:?log}; shift; [ "${1:-}" = "--" ] && shift
FLOOR=${GUARD_FLOOR_KB:-7500000}
avail() { while read -r k v _; do [ "$k" = "MemAvailable:" ] && { echo "$v"; return; }; done < /proc/meminfo; }
svc() { [ "${GUARD_NO_SVC:-0}" = 1 ] && { echo active; return; }; systemctl --user is-active dsv41-rank 2>/dev/null; }
a0=$(avail)
echo "guard: $(date -u +%FT%TZ) floor=${FLOOR} kB start MemAvailable=${a0} kB cmd: $*" >> "$LOG"
if [ "$a0" -lt "$FLOOR" ]; then echo "guard: REFUSED, MemAvailable ${a0} < floor ${FLOOR}" | tee -a "$LOG"; exit 76; fi
if [ "$(svc)" != active ]; then echo "guard: REFUSED, dsv41-rank not active (relaunch may be running)" | tee -a "$LOG"; exit 76; fi
CUDA_VISIBLE_DEVICES= setsid nice -n 19 "$@" >> "$LOG" 2>&1 &
pid=$!
trap 'kill -KILL -- "-$pid" 2>/dev/null; echo "guard: terminated, session killed" >> "$LOG"; exit 143' TERM INT HUP
min=$a0; i=0
while kill -0 "$pid" 2>/dev/null; do
  a=$(avail)
  [ "$a" -lt "$min" ] && min=$a
  i=$((i + 1))
  bad=""
  [ "$a" -lt "$FLOOR" ] && bad="MemAvailable=${a} kB < floor"
  if [ $((i % 20)) = 0 ] && [ "$(svc)" != active ]; then bad="dsv41-rank left active"; fi
  if [ -n "$bad" ]; then
    echo "guard: BREACH $(date -u +%FT%TZ) $bad; killing session ${pid}" >> "$LOG"
    kill -KILL -- "-$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
    echo "guard: stopped; $bad; min MemAvailable=${min} kB" | tee -a "$LOG"
    exit 75
  fi
  sleep 0.1
done
wait "$pid"; rc=$?
echo "guard: done rc=${rc} min MemAvailable=${min} kB, headroom used about $((a0 - min)) kB" | tee -a "$LOG"
exit "$rc"
