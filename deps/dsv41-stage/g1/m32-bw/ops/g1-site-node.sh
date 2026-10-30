#!/bin/bash
# g1/m32-bw per-node site action (called by ops/g1-site.sh): bash g1-site-node.sh <install|uninstall|status> [FORCE]
set -u
A=${1:?action}; FORCE=${2:-0}
LIVE=5acc5fde33bc7e7e922efeea92611d594db9c13e9d9b309781ccd58de787bb92
STAGED=db659d809cbcc3949f94dd810d9f00359f1770af082aadd8f7037816371f02da
SP=$HOME/venvs/vllm032/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/experts
F=$SP/sm121_target_m32.py; B=$F.bak-g1-m32bw; G=$HOME/dsv41-stage/g1/m32-bw
SRC=$G/site/vllm/model_executor/layers/fused_moe/experts/sm121_target_m32.py
sha() { sha256sum "$1" 2>/dev/null | cut -c1-64; }
active=$(systemctl --user is-active dsv41-rank 2>/dev/null)
case $A in
  status) echo "G1 OK status $(sha $F)";;
  install)
    [ "$active" != active ] || [ "$FORCE" = 1 ] || { echo "G1 FAIL service active"; exit 1; }
    [ -f $G/PREP.ok ] || { echo "G1 FAIL no PREP.ok (run ops/prep-node.sh)"; exit 1; }
    [ "$(sha $SRC)" = $STAGED ] || { echo "G1 FAIL staged file sha $(sha $SRC)"; exit 1; }
    cur=$(sha $F)
    if [ "$cur" = $STAGED ]; then echo "G1 OK install (already) $cur"; exit 0; fi
    [ "$cur" = $LIVE ] || { echo "G1 FAIL live file is $cur, expected $LIVE"; exit 1; }
    [ -f $B ] || cp -p $F $B
    [ "$(sha $B)" = $LIVE ] || { echo "G1 FAIL backup sha $(sha $B)"; exit 1; }
    cp $SRC $F.g1tmp && mv $F.g1tmp $F
    if [ "$(sha $F)" = $STAGED ]; then echo "G1 OK install $(sha $F)"; else echo "G1 FAIL after install $(sha $F)"; exit 1; fi;;
  uninstall)
    [ "$active" != active ] || [ "$FORCE" = 1 ] || { echo "G1 FAIL service active"; exit 1; }
    cur=$(sha $F)
    if [ "$cur" = $LIVE ]; then echo "G1 OK uninstall (already) $cur"; exit 0; fi
    { [ -f $B ] && [ "$(sha $B)" = $LIVE ]; } || { echo "G1 FAIL no valid backup $B"; exit 1; }
    cp -p $B $F.g1tmp && mv $F.g1tmp $F
    if [ "$(sha $F)" = $LIVE ]; then echo "G1 OK uninstall $(sha $F)"; else echo "G1 FAIL after uninstall $(sha $F)"; exit 1; fi;;
  *) echo "G1 FAIL bad action $A"; exit 1;;
esac
