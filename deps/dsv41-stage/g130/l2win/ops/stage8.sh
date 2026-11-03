#!/bin/bash
# stage8.sh: copy the kit from this Mac's mirror to ~/dsv41-stage/g130/l2win/ on all 8 Sparks and verify SHA256SUMS on each. Disk only: no GPU, no
# service change, safe while serving. NOT RUN on 8 hosts yet. Needs the Tailscale shim on PATH (WINDOW.md).
set -euo pipefail
cd "$(dirname "$0")/.."
HOSTS="spark-one spark-two spark-three spark-four spark-five spark-six spark-seven spark-eight"
shasum -a 256 -c --quiet SHA256SUMS || { echo "ABORT: mirror fails SHA256SUMS"; exit 1; }
FILES=$(cut -c67- SHA256SUMS)
for h in $HOSTS; do
  COPYFILE_DISABLE=1 tar --no-mac-metadata --no-xattrs -czf - $FILES SHA256SUMS | \
    ssh -o ConnectTimeout=10 "$h" 'mkdir -p ~/dsv41-stage/g130/l2win/results && cd ~/dsv41-stage/g130/l2win && nice -n 19 tar xzf - && chmod -R a+rX . && chmod +x ops/*.sh && sha256sum -c --quiet SHA256SUMS && echo "$(hostname) staged OK $(sha256sum SHA256SUMS | cut -c1-16)"'
done
