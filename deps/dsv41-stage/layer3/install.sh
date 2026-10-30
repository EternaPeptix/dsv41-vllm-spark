#!/bin/bash
# Install the layer-3 tool-argument streaming patch into this host's vLLM 0.32.0 venv.
#
#   bash install.sh [--venv DIR] [--dry-run]      (default venv: ~/venvs/vllm032)
#
# - Refuses unless every live file is the exact deployed version (sha256 in MANIFEST) or
#   already the patched one; backs each original up as <file>.bak-layer3 first.
# - Changes files only. A running server keeps the code it imported: nothing changes until
#   the server is restarted, and even then the patch is inert unless the API server's
#   environment has DSV41_TOOL_STREAM_NEWLINE=1. This script restarts nothing.
# Undo: bash uninstall.sh. Check: bash status.sh.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
VENV=$HOME/venvs/vllm032
DRY=0
while (( $# )); do
  case $1 in
    --venv) VENV=$2; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    *) echo "usage: install.sh [--venv DIR] [--dry-run]" >&2; exit 2 ;;
  esac
done
SITE=$(ls -d "$VENV"/lib/python3*/site-packages 2>/dev/null | head -1)
[[ -n $SITE && -d $SITE/vllm ]] || { echo "install: no vllm in $VENV" >&2; exit 1; }
sha() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi 2>/dev/null | cut -d' ' -f1; }

# Check everything before changing anything.
todo=()
while read -r rel orig patched; do
  [[ -z $rel || $rel == \#* ]] && continue
  live=$SITE/$rel
  payload=$HERE/payload/$rel
  [[ -f $live ]] || { echo "install: missing $live" >&2; exit 1; }
  [[ $(sha "$payload") == "$patched" ]] || { echo "install: payload $rel does not match MANIFEST" >&2; exit 1; }
  now=$(sha "$live")
  if [[ $now == "$patched" ]]; then
    echo "already patched: $rel"
    continue
  fi
  [[ $now == "$orig" ]] || { echo "install: $live is neither the deployed nor the patched version ($now); not touching it" >&2; exit 1; }
  if [[ -e $live.bak-layer3 && $(sha "$live.bak-layer3") != "$orig" ]]; then
    echo "install: $live.bak-layer3 exists and is not the deployed version; not touching it" >&2; exit 1
  fi
  todo+=("$rel")
done < "$HERE/MANIFEST"

for rel in "${todo[@]}"; do
  live=$SITE/$rel
  if (( DRY )); then echo "would patch: $live (backup $live.bak-layer3)"; continue; fi
  [[ -e $live.bak-layer3 ]] || cp -p "$live" "$live.bak-layer3"
  cp "$HERE/payload/$rel" "$live.layer3-tmp"
  mv -f "$live.layer3-tmp" "$live"
  echo "patched: $live"
done
(( DRY )) && exit 0
bash "$HERE/status.sh" --venv "$VENV"
echo "Nothing was restarted. To enable: start the vLLM API server with DSV41_TOOL_STREAM_NEWLINE=1 in its environment."
