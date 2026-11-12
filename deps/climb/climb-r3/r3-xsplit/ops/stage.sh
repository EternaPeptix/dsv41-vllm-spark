#!/usr/bin/env bash
# Copy this kit (prebuilt build/ included) to ~/climb-r3/r3-xsplit on one node for the stage-1
# window (device_test.py + bench). Disk only; touches nothing outside that folder.
# Usage: bash ops/stage.sh spark-five
set -euo pipefail
export PATH=$HOME/Documents/dsv41-kits/ts-ssh/bin:$PATH
H=${1:?node}
KIT=$(cd "$(dirname "$0")/.." && pwd)
COPYFILE_DISABLE=1 tar cf - -C "$KIT" --exclude '*.sass' --exclude __pycache__ \
  KIT.json README.md EXACTNESS.md device_test.py bench build files install.py tools tests src |
  ssh "$H" 'mkdir -p ~/climb-r3/r3-xsplit && cd ~/climb-r3/r3-xsplit && tar xf - 2>/dev/null && find . -name "._*" -delete && cd build && sha256sum -c SHA256SUMS'
