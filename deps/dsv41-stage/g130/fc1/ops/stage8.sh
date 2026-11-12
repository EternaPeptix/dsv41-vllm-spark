#!/bin/bash
# g130 fc1: copy the kit to all 8 and stage it there (disk only). Run from the Mac, with the Tailscale ssh shim first on PATH.
set -u
K=$(cd "$(dirname "$0")/.." && pwd)
cd "$K" || exit 1
( LC_ALL=C shasum -a 256 -c SHA256SUMS --quiet ) || { echo "mirror fails SHA256SUMS"; exit 1; }
for h in spark-one spark-two spark-three spark-four spark-five spark-six spark-seven spark-eight; do
  ( COPYFILE_DISABLE=1 tar czf - --no-xattrs --exclude trees --exclude work . \
      | ssh -o ConnectTimeout=10 "$h" 'mkdir -p ~/dsv41-stage/g130/fc1 && cd ~/dsv41-stage/g130/fc1 && tar xzf - 2>/dev/null && timeout 300 bash ops/stage-node.sh' \
      | sed "s/^/$h: /" ) &
done
wait
