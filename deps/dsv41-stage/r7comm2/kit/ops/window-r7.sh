#!/bin/bash
# window-r7.sh plan | pre | dw | serve-cmd : the r7-comm2 windows (printed by `plan`; the lead runs them).
#  plan / serve-cmd need EXPECT=<production launcher sha8> in the environment (no default): EXPECT=4f3cc1c6 bash window-r7.sh plan
#  pre  (no window; while serving, only when no CLUSTER_LOCK exists): device-only cubins of the r7 source, r6 control and
#       the 15 mutants on spark-eight (ops/dev8-r7.sh: ~100-200 MB per TU, guard floor 8.5 GB, nice 19, lock re-checked
#       before every TU) -> gate R0-R4, MR1-MR8, MP1-MP5, MO1-MO3 on 512S1. Fails early on a compile error, before any window.
#  dw   (stopped device window, ~3.5-4 h): full builds (14 .so, spark-eight), gate incl. D7, stage 8/8, device tests.
set -euo pipefail
cd "$(dirname "$0")/.."
KIT=$(pwd)
GH=${R7_GATE_HOST-jeweled@<tailscale-ip>}
GR=${GATE_REMOTE:-climb-r8/comm2fix}  # r8-comm2fix: the 512S1 copy the gate runs in (was climb-r7/comm2)
# r8-comm2fix (review B1): the printed window commands carry EXPECT=<production launcher sha8>, which both runners pass
# to the relaunch supervisor as --expect-launcher for every arm AND for the restore. There is NO default: plan /
# serve-cmd refuse unless EXPECT is set (EXPECT=4f3cc1c6 on 2026-11-09, climb-r8/BRIEF.md) and refuse known-old shas.
STALE_EXPECT="166d7aac 08a249da fda6d05a"
need_expect() {
  [[ "${EXPECT:-}" =~ ^[0-9a-f]{8}$ ]] || { echo "window-r7: set EXPECT=<production launcher sha8> (no default)" >&2; exit 2; }
  for s in $STALE_EXPECT; do
    [ "$EXPECT" != "$s" ] || { echo "window-r7: EXPECT=$EXPECT is an old launcher, not production" >&2; exit 2; }
  done
}
gate() {  # run the suite on 512S1 (python3 with no extra packages needed); copy results back
  rsync -a --exclude __pycache__ --exclude results gate sass ref KIT.json audit "$GH:$GR/" >/dev/null
  rsync -a ../../climb-r6/comm/sass/ "$GH:climb-r6/comm/sass/" >/dev/null
  ssh "$GH" "cd ~/$GR && nice -n 10 ~/exo-venv313/bin/python3 -I gate/run_gate7.py --sass6 ~/climb-r6/comm/sass $*" | tee "build-logs/gate7-$(date -u +%H%M%S).txt"
  rsync -a "$GH:$GR/gate/results/" gate/results/
}
case "${1:-plan}" in
  pre)
    bash ops/dev8-r7.sh r7 ctl_r6 m_rev m_seed0 m_negzero m_ffma m_tree m_waitlate m_stage mp_first mp_warp mp_l2 mp_unfused mp_chunk \
      mo_nofence mo_first mo_noacq
    gate --skip-missing
    for c in R0_live_vs_devr7 R1_devctlr6_vs_devr7 R2_pdl_devr7 R3_proj_devr7 R4_ord_devr7 MR1 MR2 MR3 MR4 MR5 MR6 MR7 MR8 \
      V1 V2 V3 V4 V5 V6 MP1 MP2 MP3 MP4 MP5 MO1 MO2 MO3 O7; do
      grep -q "^$c[^ ]* *OK" gate/results/SUMMARY.txt || { echo "window-r7 pre: $c not OK"; exit 1; }
    done
    echo "window-r7 pre: every device-TU case OK (no .so built; D7 waits for the stopped window)"
    ;;
  dw)
    set -x
    bash ops/build8-r7.sh
    gate
    grep -q "^RUN_GATE7 PASS" gate/results/SUMMARY.txt
    grep -q "^GATED_SO_SHA256 $(python3 -c 'import json;print([f for f in json.load(open("KIT.json"))["files"] if f["path"].endswith("r7comm2/build/rdma_ar_ext.so")][0]["sha256"])') " gate/results/SUMMARY.txt
    bash ops/stage8-r7.sh
    bash ops/run-device8-r7.sh all | tee build-logs/device7-dw.txt
    [ "$(grep -c 'R7_DEVICE_TEST PASS' build-logs/device7-dw.txt)" = 16 ]
    grep PROJTIME build-logs/device7-dw.txt || true
    ;;
  plan)
    need_expect
    SERVE=$(bash "$0" serve-cmd) || exit 2
    cat <<EOF2
== 0. pre (no lock, service may run; ~30-45 min, spark-eight device-only compiles + 512S1 gate)
   bash $KIT/ops/window-r7.sh pre
   pass = "every device-TU case OK". This is the first time the r7 source meets nvcc: a compile error stops here.
== 1. stopped device window (service stopped 8/8; ~3.5-4 h; restores production at the end)
   echo "r7dw|bash $KIT/ops/window-r7.sh dw" > /tmp/r7dw.mac.plan
   cd ~/Documents/dsv41-offline/climb-common && WIN=r7-dw EXPECT=$EXPECT NODE_PLAN= MAC_PLAN=/tmp/r7dw.mac.plan bash device-window.sh
   pass = RUN_GATE7 PASS with GATED_SO_SHA256 of the recorded .so, 16 x R7_DEVICE_TEST PASS (r6 c1 x8, r6 mutants,
   r6 c2, stage, projmut x2, proj x2, projtime). PROJTIME lines = the measured fold time per call (README §5): INFO,
   never a gate; do not run the q/p pair if PROJTIME shows no gain. No node runs more than 2 parts (review item 6).
   This window replaces r6-comm's window C (the r7 .so carries C1 + C2 unchanged; the r6 device test runs on it).
   RETIRED (review item 8): climb-r6/comm/gate/run_gate.py (its sassir reads a predicated BRA.U as unconditional and
   skips the S1 staging region) and r6-comm's window C. Gate with gate/run_gate7.py and run r7-dw instead.
== 2. serving window (after 1; KILL stops the HT13 arms only)
$SERVE
EOF2
    ;;
  serve-cmd)
    need_expect
    E=/home/jeweled/dsv41-stage/r7comm2/build/rdma_ar_ext.so
    cat <<EOF2
   cd ~/Documents/dsv41-offline/climb-common && WIN=r7-w1 EXPECT=$EXPECT \\
     KIT_INSTALL="bash $KIT/ops/install8-r7.sh install" KIT_UNINSTALL="bash $KIT/ops/install8-r7.sh uninstall" \\
     DET_ARMS="t1 t2 p1 p2" bash climb-window.sh \\
       c1:RDMA_AR_AUDIT=200\;RDMA_AR_EXT=$E \\
       t1:RDMA_AR_AUDIT=200\;RDMA_AR_EXT=$E\;DSV41_RDMA_TRIM=1 \\
       c2:RDMA_AR_AUDIT=200\;RDMA_AR_EXT=$E \\
       t2:RDMA_AR_AUDIT=200\;RDMA_AR_EXT=$E\;DSV41_RDMA_TRIM=1 \\
       q1:RDMA_AR_AUDIT=200\;RDMA_AR_EXT=$E\;DSV41_RDMA_TRIM=1 \\
       p1:RDMA_AR_AUDIT=200\;RDMA_AR_EXT=$E\;DSV41_RDMA_TRIM=1\;DSV41_RDMA_PROJ=1 \\
       q2:RDMA_AR_AUDIT=200\;RDMA_AR_EXT=$E\;DSV41_RDMA_TRIM=1 \\
       p2:RDMA_AR_AUDIT=200\;RDMA_AR_EXT=$E\;DSV41_RDMA_TRIM=1\;DSV41_RDMA_PROJ=1
   C1 pair (t vs c), then the HT13 pair on top of C1 (p vs q). Promote at >= +0.3 % G1 in BOTH pairs with exact.py
   ec2f4437f8e1ed09, G1 digest 2f7fdccbd388225a and det ALL==REF on every T arm; boot lines counted 8/8 per T arm:
   "rdma_ar r6 ... pdl bits = 0" (t, p) and "rdma_ar r7 r7comm2-ht13-v1: projection fold on" (p).
   HT13 precondition (review item 1): run   bash $KIT/ops/takecheck-r7.sh --watch   next to the window; it must print
   TAKECHECK_R7 PASS for p1 (taken == 76, wasted == 0 on every captured decode forward, 8/8 ranks). If it FAILS, the
   folds are not being taken: stop the window (KILL) before p2 and do not promote HT13.
   Environment rules for the q / p arms (review B1, item 3): never DSV41_GLUE_PDL (any value that enables its 'proj'
   family: the r5 combined-env.json, all1/all2, pdl1/pdl2) and never DSV41_OBLIT_FOLD together with DSV41_RDMA_PROJ=1.
   dsv41_oblit refuses that at import (the launch fails closed); the q arms must not inherit them either, so q and p
   differ by DSV41_RDMA_PROJ only. HT13 composes with glue-PDL only after the PDL _proj_rows cubin's projgate record is
   proven equal to f29c9ff9 (not done).
   Installer order (review item 5, LIFO): r7-comm2 is installed LAST and removed FIRST. With r7 installed,
   climb-r6/install-all.sh status / stack-install / stack-uninstall see rdma_all_reduce.py as FOREIGN and refuse: run
   bash $KIT/ops/install8-r7.sh uninstall before ANY install-all.sh operation, and install8-r7.sh install after it.
   Never run r6-comm's own install.py uninstall on a node where r7 is installed (it leaves r7 lines without r6 lines;
   install.py status then reports HYBRID and exits 4). OPEN before promotion: install-all.sh's comm entry still names
   r6-comm; swapping it to r7-comm2 (SHIP pins, check_comm, dsv41_oblit.py as a target) is not done in this kit.
   C2 pair (only after the C2 audit closes: audit/out/acqcheck.txt N2 PASS; today OPEN for 4 kernels, README §2):
       r1:...;DSV41_RDMA_TRIM=1[;DSV41_RDMA_PROJ=1]   s1:...same;DSV41_RDMA_PDL=1   r2 / s2 likewise
   Never DSV41_RDMA_PDL=trig (N1 OPEN: flashinfer MXFP8 quant, README §1). Never DSV41_RDMA_SENDBUF.
EOF2
    ;;
  *) echo "usage: window-r7.sh plan | pre | dw | serve-cmd"; exit 2;;
esac
