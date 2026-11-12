#!/usr/bin/env bash
# serve.sh  (from the Mac) -- the r8-fixb serving window: COLEMB solo (e) and PACK=all solo (p), C-T-C-T-C x 2.
#   EXPECT=<production sha8> [WIN=r8-fixb-w1] bash ops/serve.sh          (refuses without EXPECT)
# Preconditions: the device window passed (fixb-step-dt rc=0, SUMMARY 0 failed) and r6-step is installed 8/8 flags-off.
# KIT_INSTALL swaps dsv41_step.py to the r8 module 8/8 (lock-checked; rollback on failure); the identity arm then proves
# the flags-off kit inert (exact.py + G1 digest + the 9 long-prompt cases). EVERY T arm runs the 9-case long-prompt
# determinism check (DET_ARMS) -- the check s1 failed in r6-w1. G4 on c1 / e1 / p1.
# Boot lines to read in the rank logs (8/8): "DSV41_STEP_COLEMB rank r: full-table check ... PASS" (e arms) and
# "DSV41_STEP_ENGRAM_PACK=all: engram layers 1 and 14 share one rows gather and one wkv gather" (p arms).
# The kit stays installed flags-off after the window (identity proven); remove with: bash ops/fleet.sh uninstall (lock).
set -u
D=$(cd "$(dirname "$0")" && pwd); KIT=$(cd "$D/.." && pwd); CW=$(cd "$KIT/../.." && pwd)/climb-common/climb-window.sh
EXPECT=${EXPECT:?set EXPECT=<production launcher sha8>}
export EXPECT WIN=${WIN:-r8-fixb-w1}
export KIT_INSTALL="bash $KIT/ops/fleet.sh install" KIT_UNINSTALL="bash $KIT/ops/fleet.sh uninstall"
export DET_ARMS="e1 p1 e2 p2" G4_ARMS="c1 e1 p1"
exec bash "$CW" $(grep -v '^#' "$D/arms-fixb.txt" | tr '\n' ' ')
