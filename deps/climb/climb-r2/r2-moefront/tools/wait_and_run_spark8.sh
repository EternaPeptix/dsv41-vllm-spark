#!/bin/bash
# Wait (polling every POLL_S seconds, default 60, up to DEADLINE_H hours, default 6) until CLUSTER_LOCK on spark-one is absent for two
# consecutive polls, then run tools/run_spark8_checks.sh once. Exit code = that script's (2/3 = lock, retry later),
# or 9 when the deadline passed with the lock still held.
set -u
export PATH=$HOME/Documents/dsv41-kits/ts-ssh/bin:$PATH
KIT=$(cd "$(dirname "$0")/.." && pwd)
DEADLINE=$(( $(date +%s) + ${DEADLINE_H:-6} * 3600 ))
free=0
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    if ssh -n -o ConnectTimeout=15 spark-one 'test -e ~/dsv41-opt-20261021/CLUSTER_LOCK'; then
        free=0
    elif [ $? -eq 1 ]; then
        free=$((free + 1))
        if [ $free -ge 2 ]; then
            echo "$(date -u +%FT%TZ) lock absent twice: running checks"
            bash "$KIT/tools/run_spark8_checks.sh"
            exit $?
        fi
    fi
    sleep ${POLL_S:-60}
done
echo "$(date -u +%FT%TZ) deadline reached, lock still held"
exit 9
