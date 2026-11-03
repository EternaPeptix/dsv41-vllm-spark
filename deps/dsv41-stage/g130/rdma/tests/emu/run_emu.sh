#!/bin/bash
# run_emu.sh [--mutants] [--build-only]: build and run the 8-rank CPU emulator ON SPARK-TWO, on a few A725 cores at nice 19,
# under the memory guard (never on the Mac). Verbatim source: extract_g130.py. No GPU. Mutants: planted bugs that must be caught.
set -u
D=$(cd "$(dirname "$0")" && pwd)
R=$(cd "$D/../.." && pwd)
CORES=${EMU_CORES:-0-3}
B=$R/tests/emu/build
mkdir -p "$B"
G="bash $R/ops/guard_run.sh"
build() {  # <source> <outdir>
  mkdir -p "$2"
  python3 -I "$D/extract_g130.py" "$1" "$2/dev.inc" "$2/host.inc" || return 2
  cp "$2/dev.inc" "$2/emu_dev.inc"
  sed -n '1,/HOSTSTRUCTS_END/p' "$2/host.inc" > "$2/emu_hoststructs.inc"
  sed '1,/HOSTSTRUCTS_END/d' "$2/host.inc" > "$2/emu_host.inc"
  : > "$2/build.log"
  timeout -k 5 ${EMU_BUILD_TIMEOUT:-300} $G "$2/build.log" -- taskset -c "$CORES" g++ -std=c++20 -O1 -pthread -w -I "$2" -I "$D" -o "$2/emu" "$D/emu_g130.cpp"
  rc=$?
  [ $rc = 0 ] || { grep -E "error|guard:" "$2/build.log" | head -20 | cut -c1-200; return $rc; }
}
[ -x "$B/emu" ] && [ "$B/emu" -nt "$D/emu_g130.cpp" ] && [ "$B/emu" -nt "$R/native/rdma_ar.cu" ] || build "$R/native/rdma_ar.cu" "$B" || exit $?
[ "${1:-}" = "--build-only" ] && exit 0
if [ "${1:-}" != "--mutants" ]; then
  L=$R/results/emu.out; : > "$L"
  EMU_OPS=${EMU_OPS:-40} timeout -k 5 ${EMU_RUN_TIMEOUT:-540} $G "$L" -- taskset -c "$CORES" "$B/emu"
  rc=$?
  grep -vE '^guard:' "$L" | cut -c1-150
  exit $rc
fi
mut() {  # <name> <case that must fail> <python replace old> <new>
  local n=$1 c=$2 o=$3 w=$4 M=$B/mut-$1
  mkdir -p "$M"
  python3 -I - "$R/native/rdma_ar.cu" "$M/src.cu" "$o" "$w" <<'PY' || { echo "MUTANT $n: did not apply (BAD MUTANT)"; bad=$((bad+1)); return; }
import sys
s=open(sys.argv[1]).read(); o,w=sys.argv[3].encode().decode("unicode_escape"),sys.argv[4].encode().decode("unicode_escape")
if s.count(o)<1: sys.exit(1)
open(sys.argv[2],"w").write(s.replace(o,w))
PY
  build "$M/src.cu" "$M" > /dev/null || { echo "MUTANT $n: build failed or stopped"; bad=$((bad+1)); return; }
  : > "$M/run.log"
  EMU_OPS=${EMU_OPS:-40} timeout -k 5 120 $G "$M/run.log" -- taskset -c "$CORES" "$M/emu" "$c"
  rc=$?
  if [ $rc = 1 ] && grep -q "FAIL" "$M/run.log"; then echo "MUTANT $n ($c): KILLED: $(grep -m1 'FAIL' "$M/run.log" | cut -c1-110)"
  elif [ $rc = 124 ] || [ $rc = 143 ]; then echo "MUTANT $n ($c): KILLED by hang (120 s limit; the proxy never honours stop)"
  else echo "MUTANT $n ($c): SURVIVED rc=$rc"; bad=$((bad+1)); fi
}
bad=0
# the phase-1 guard is load-bearing: with it gone a rank overwrites the mailbox before its stalled proxy read it
mut no_phase1_guard v1/victim 'G130_SPIN(&p.ctrl->posted_seq, seq - 1, -1, 0);' ';'
# V1's compare applied to phase 2: send2_seq does not move on other modes, so this spin never ends
mut v1_on_phase2 v1v2/nostall 'if constexpr ((kPub & kPubV2) == 0)\n        G130_SPIN(&p.ctrl->posted2_seq, ld_acquire_sys(&p.ctrl->send2_seq), -2, 0);' 'G130_SPIN(&p.ctrl->posted2_seq, seq - 1, -2, 0);'
# V2's premise broken: peers do not wait for this rank's phase-2 flag, so nothing orders its proxy against the next publish
mut v2_premise_broken v2/victim '    if (threadIdx.x < p.world && threadIdx.x != p.rank)\n      G130_SPIN(p.flags2 + slot * p.world + threadIdx.x, seq, threadIdx.x, 2);' '    if (threadIdx.x < p.world && threadIdx.x != p.rank && threadIdx.x != 3)\n      G130_SPIN(p.flags2 + slot * p.world + threadIdx.x, seq, threadIdx.x, 2);'
# split post in the wrong order: the flag can land before the payload
mut split_flag_first split/nostall 'if (ibv_post_send(pr.qp, &wr[0], &bad) || ibv_post_send(pr.qp, &wr[1], &bad)) {' 'if (ibv_post_send(pr.qp, &wr[1], &bad) || (usleep(50), ibv_post_send(pr.qp, &wr[0], &bad))) {'
# hygiene without its fallback, spinning on the wrong word forever: must hang, proving the fallback is load-bearing
mut hyg_no_fallback hygproxy/nostall 'for (int k = 0; k < kHygSpin && __atomic_load_n(db, __ATOMIC_ACQUIRE) == old; ++k) {' 'for (;__atomic_load_n(&ctrl_->send_seq, __ATOMIC_ACQUIRE) == last1;) {'
# sum order: reversed r in the two-shot accumulate must change the bits
mut sum_reversed v1/nostall '      for (int r = 0; r < kW; ++r) accv<kF32>(acc, v[r]);\n      const uint4 o = packv<kF32>(acc);' '      for (int r = kW - 1; r >= 0; --r) accv<kF32>(acc, v[r]);\n      const uint4 o = packv<kF32>(acc);'
echo "MUTANTS: $bad not killed"
[ $bad = 0 ]
