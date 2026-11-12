#!/bin/bash
# stage.sh (g2 colltrim) — run FROM THE MAC with bash after ops/build.sh. Copies the package to
# ~/dsv41-stage/g2/colltrim/ on all 8 Sparks and verifies every file against SHA256SUMS there.
# Stage dir only: no site-packages, launcher, supervisor or service change. Safe while serving.
set -euo pipefail
cd "$(dirname "$0")/.."
HOSTS="spark-one spark-two spark-three spark-four spark-five spark-six spark-seven spark-eight"
[ -s build/rdma_ar_ext.so ] && [ -s build/libdsv41_l2pf_g2.so ] || { echo "run ops/build.sh first"; exit 1; }
FILES=$(find README.md native site base tests bench ops diffs build results -type f ! -name '.DS_Store' \
  ! -path 'build/ext/*' 2>/dev/null | sort)
shasum -a 256 $FILES > SHA256SUMS
echo "package digest $(shasum -a 256 SHA256SUMS | cut -c1-12) ($(echo $FILES | wc -w | tr -d ' ') files)"
for h in $HOSTS; do
  COPYFILE_DISABLE=1 tar --no-mac-metadata --no-xattrs -czf - $FILES SHA256SUMS | \
    ssh "$h" 'mkdir -p ~/dsv41-stage/g2/colltrim && cd ~/dsv41-stage/g2/colltrim && tar xzf - && chmod +x ops/*.sh && sha256sum -c --quiet SHA256SUMS && echo "$(hostname) staged OK $(sha256sum SHA256SUMS | cut -c1-12)"'
done
