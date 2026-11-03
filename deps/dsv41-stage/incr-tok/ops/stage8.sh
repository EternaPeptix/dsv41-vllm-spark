#!/bin/bash
# stage8.sh: copy the kit from this Mac's mirror to ~/dsv41-stage/incr-tok/ on all 8 Sparks and verify SHA256SUMS there. Disk only: no GPU, no
# service change, safe while serving. Each host also dry-runs the installer, which fails unless renderers/deepseek_v4.py is the f7175f99 base and
# dsv41_incr_tok.py is absent, so re-stage only on a base site. Exit 0 only with 8/8 staged at this mirror's digest. Needs the Tailscale shim on
# PATH (WINDOW.md). NOT RUN YET on 8 hosts.
set -euo pipefail
cd "$(dirname "$0")/.."
HOSTS="spark-one spark-two spark-three spark-four spark-five spark-six spark-seven spark-eight"
shasum -a 256 -c --quiet SHA256SUMS || { echo "ABORT: mirror fails SHA256SUMS"; exit 1; }
FILES=$(cut -c67- SHA256SUMS)
WANT=$(shasum -a 256 SHA256SUMS | cut -c1-12)
n=0
for h in $HOSTS; do
  out=$(COPYFILE_DISABLE=1 tar --no-mac-metadata --no-xattrs -czf - $FILES SHA256SUMS | \
    ssh -o ConnectTimeout=10 "$h" 'mkdir -p ~/dsv41-stage/incr-tok && cd ~/dsv41-stage/incr-tok && nice -n 19 timeout 60 tar xzf - && chmod -R a+rX . && sha256sum -c --quiet SHA256SUMS && python3 -I tools/incr_install.py dry-run >/dev/null && echo "$(hostname) staged OK $(sha256sum SHA256SUMS | cut -c1-12)"') \
    || { echo "ABORT: $h failed to stage: ${out:-no output}"; exit 1; }
  echo "$out"
  case "$out" in *"staged OK $WANT") n=$((n + 1)) ;; *) echo "ABORT: $h digest is not $WANT"; exit 1 ;; esac
done
[ "$n" = 8 ] || { echo "ABORT: staged $n of 8"; exit 1; }
echo "stage8: 8/8 staged OK $WANT"
