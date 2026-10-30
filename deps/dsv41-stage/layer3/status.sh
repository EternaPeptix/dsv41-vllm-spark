#!/bin/bash
# Report the layer-3 parser patch state on this host (read-only):
# which version each file is, whether a backup exists, and whether a running vLLM process
# has DSV41_TOOL_STREAM_NEWLINE in its environment.
#   bash status.sh [--venv DIR]
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
VENV=$HOME/venvs/vllm032
[[ ${1:-} == --venv ]] && VENV=$2
SITE=$(ls -d "$VENV"/lib/python3*/site-packages 2>/dev/null | head -1)
sha() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi 2>/dev/null | cut -d' ' -f1; }
echo "host $(hostname)  venv $VENV"
while read -r rel orig patched; do
  [[ -z $rel || $rel == \#* ]] && continue
  live=$SITE/$rel
  now=$(sha "$live")
  state=unknown
  [[ $now == "$orig" ]] && state=deployed
  [[ $now == "$patched" ]] && state=patched
  bak=none
  [[ -f $live.bak-layer3 ]] && bak=$([[ $(sha "$live.bak-layer3") == "$orig" ]] && echo ok || echo MISMATCH)
  printf '  %-40s %-8s backup=%s\n' "$rel" "$state" "$bak"
done < "$HERE/MANIFEST"
found=0
for pid in $(pgrep -u "$(id -u)" -f 'vllm' 2>/dev/null); do
  [[ -r /proc/$pid/environ ]] || continue
  flag=$(tr '\0' '\n' < /proc/$pid/environ | grep '^DSV41_TOOL_STREAM' | tr '\n' ' ')
  cmd=$(tr '\0' ' ' < /proc/$pid/cmdline | cut -c1-80)
  printf '  pid %-8s %s  [%s]\n' "$pid" "${flag:-flag unset}" "$cmd"
  found=1
done
(( found )) || echo "  no vllm process of this user"
