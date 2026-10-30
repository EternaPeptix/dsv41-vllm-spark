#!/bin/bash
# Restore the deployed vLLM parser files from <file>.bak-layer3 (see install.sh).
#   bash uninstall.sh [--venv DIR] [--dry-run]
# Refuses unless the live file is the patched version (or already the deployed one) and the
# backup is the deployed version. Restarts nothing.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
VENV=$HOME/venvs/vllm032
DRY=0
while (( $# )); do
  case $1 in
    --venv) VENV=$2; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    *) echo "usage: uninstall.sh [--venv DIR] [--dry-run]" >&2; exit 2 ;;
  esac
done
SITE=$(ls -d "$VENV"/lib/python3*/site-packages 2>/dev/null | head -1)
[[ -n $SITE && -d $SITE/vllm ]] || { echo "uninstall: no vllm in $VENV" >&2; exit 1; }
sha() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi 2>/dev/null | cut -d' ' -f1; }
todo=()
while read -r rel orig patched; do
  [[ -z $rel || $rel == \#* ]] && continue
  live=$SITE/$rel
  now=$(sha "$live")
  if [[ $now == "$orig" ]]; then echo "already deployed version: $rel"; continue; fi
  [[ $now == "$patched" ]] || { echo "uninstall: $live is neither version ($now); not touching it" >&2; exit 1; }
  [[ -f $live.bak-layer3 && $(sha "$live.bak-layer3") == "$orig" ]] || { echo "uninstall: no valid $live.bak-layer3" >&2; exit 1; }
  todo+=("$rel")
done < "$HERE/MANIFEST"
for rel in "${todo[@]}"; do
  live=$SITE/$rel
  if (( DRY )); then echo "would restore: $live from $live.bak-layer3"; continue; fi
  cp -p "$live.bak-layer3" "$live.layer3-tmp"
  mv -f "$live.layer3-tmp" "$live"
  rm -f "$live.bak-layer3"
  echo "restored: $live"
done
(( DRY )) && exit 0
bash "$HERE/status.sh" --venv "$VENV"
echo "Nothing was restarted; a running server keeps the code it imported."
