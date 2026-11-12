#!/bin/bash
# g2ct-site.sh <preflight|install|uninstall|status> [--purge] (g2 colltrim) — runs ON ONE NODE; normally called
# by ops/g2ct-all.sh from the Mac for all 8 nodes. Service must be stopped for install/uninstall
# (G2CT_ALLOW_ACTIVE=1 overrides: a file swap is inert for a running process, keep it for windows).
#
#   site  vllm/distributed/device_communicators/cuda_communicator.py   52fc8cf5 -> staged (column-gather branch)
#         vllm/distributed/device_communicators/rdma_all_reduce.py     9056dcdf -> staged (all_gather_cols, after hook)
#         vllm/distributed/device_communicators/dsv41_l2_prefetch.py   998af3d7 -> staged (pacing / re-plan arms)
#         backups <file>.bak-g2-colltrim (made once, must hold the base sha)
#   libs  ~/rdma_ar_dsv41/build-g2ct/rdma_ar_ext.so        (NEW dir; build/ and build-k1pl/ untouched)
#         ~/rdma_ar_dsv41/build-g2ct/libdsv41_l2pf_g2.so   (the live ~/dsv41-stage/r4/l2-prefetch lib untouched)
# Every flag defaults off: installed and inert until DSV41_RDMA_AG_COLS=1 / DSV41_L2_PREFETCH_PACE_GBPS>0 /
# DSV41_L2_PREFETCH_REPLAN=... (the rdma fingerprint changes with the files, identically on 8/8).
# uninstall restores the three backups (libs stay; --purge removes build-g2ct too).
set -euo pipefail
CMD=${1:?usage: g2ct-site.sh preflight|install|uninstall|status [--purge]}
PURGE=${2:-}
STAGE=$HOME/dsv41-stage/g2/colltrim
SITE=$HOME/venvs/vllm032/lib/python3.12/site-packages/vllm/distributed/device_communicators
RT=$HOME/rdma_ar_dsv41/build-g2ct
BAK=bak-g2-colltrim
FILES="cuda_communicator.py rdma_all_reduce.py dsv41_l2_prefetch.py"
LIBS="rdma_ar_ext.so libdsv41_l2pf_g2.so"
H=$(hostname)

base_of() {
  case "$1" in
    cuda_communicator.py) echo 52fc8cf52d74742f4d4cfb448e3a52ea44ddfc427889fee595feac2f8573a006 ;;
    rdma_all_reduce.py)   echo 9056dcdf39617ee46f9a3d3d7462c06c23f2814fa2a8e1889abd2636a297b9f4 ;;
    dsv41_l2_prefetch.py) echo 998af3d7fe38e92915206af5cbec795a88b4427be416073786ef8bbfaea7e3c7 ;;
  esac
}
sha() { if [ -e "$1" ]; then sha256sum "$1" | cut -c1-64; else echo missing; fi; }
new_of() { sha "$STAGE/site/vllm/distributed/device_communicators/$1"; }

service_ok() {
  local st
  st=$(systemctl --user is-active dsv41-rank 2>/dev/null || true)
  if [ "$st" = active ] || [ "$st" = activating ]; then
    [ "${G2CT_ALLOW_ACTIVE:-0}" = 1 ] || { echo "ABORT $H: dsv41-rank is $st (stop the service first)"; return 1; }
  fi
  return 0
}

preflight() {
  local ok=0 f cur b n bk
  (cd "$STAGE" && sha256sum -c --quiet SHA256SUMS) || { echo "ABORT $H: stage dir fails SHA256SUMS"; return 1; }
  for f in $FILES; do
    cur=$(sha "$SITE/$f"); b=$(base_of "$f"); n=$(new_of "$f")
    if [ "$cur" != "$b" ] && [ "$cur" != "$n" ]; then
      echo "ABORT $H: $f is ${cur:0:8} (expected base ${b:0:8} or staged ${n:0:8}; another package changed it?)"; ok=1
    fi
    if [ -e "$SITE/$f.$BAK" ]; then
      bk=$(sha "$SITE/$f.$BAK")
      [ "$bk" = "$b" ] || { echo "ABORT $H: existing backup $f.$BAK is ${bk:0:8}, not base ${b:0:8}"; ok=1; }
    fi
  done
  for f in $LIBS; do [ -s "$STAGE/build/$f" ] || { echo "ABORT $H: $STAGE/build/$f missing"; ok=1; }; done
  return $ok
}

status_line() {
  local f cur b n s out=""
  for f in $FILES; do
    cur=$(sha "$SITE/$f"); b=$(base_of "$f"); n=$(new_of "$f")
    if [ "$cur" = "$b" ]; then s=base; elif [ "$cur" = "$n" ]; then s=g2; else s=OTHER; fi
    out="$out $f=${cur:0:8}($s)"
    [ -e "$SITE/$f.$BAK" ] && out="$out bak=$(sha "$SITE/$f.$BAK" | cut -c1-8)"
  done
  for f in $LIBS; do
    cur=$(sha "$RT/$f"); n=$(sha "$STAGE/build/$f")
    if [ "$cur" = "$n" ]; then s=ok; elif [ "$cur" = missing ]; then s=absent; else s=DIFF; fi
    out="$out $f=${cur:0:8}($s)"
  done
  echo "$H$out service=$(systemctl --user is-active dsv41-rank 2>/dev/null || true)"
}

case "$CMD" in
  preflight)
    service_ok || exit 1
    preflight || exit 1
    echo "PREFLIGHT_OK $H" ;;
  status)
    status_line ;;
  install)
    service_ok || exit 1
    preflight || exit 1
    for f in $FILES; do
      cur=$(sha "$SITE/$f"); n=$(new_of "$f")
      [ "$cur" = "$n" ] && continue
      [ -e "$SITE/$f.$BAK" ] || cp -p "$SITE/$f" "$SITE/$f.$BAK"
      cp "$STAGE/site/vllm/distributed/device_communicators/$f" "$SITE/$f.g2tmp"
      mv -f "$SITE/$f.g2tmp" "$SITE/$f"
    done
    mkdir -p "$RT"
    for f in $LIBS; do
      [ "$(sha "$RT/$f")" = "$(sha "$STAGE/build/$f")" ] && continue
      cp "$STAGE/build/$f" "$RT/$f.g2tmp"
      mv -f "$RT/$f.g2tmp" "$RT/$f"
    done
    find "$SITE/__pycache__" -name 'cuda_communicator.*.pyc' -o -name 'rdma_all_reduce.*.pyc' -o \
      -name 'dsv41_l2_prefetch.*.pyc' 2>/dev/null | xargs -r rm -f
    status_line ;;
  uninstall)
    service_ok || exit 1
    for f in $FILES; do
      cur=$(sha "$SITE/$f"); b=$(base_of "$f"); n=$(new_of "$f")
      [ "$cur" = "$b" ] && continue
      [ "$cur" = "$n" ] || { echo "ABORT $H: $f is ${cur:0:8}, neither base nor staged: not touched"; exit 1; }
      [ "$(sha "$SITE/$f.$BAK")" = "$b" ] || { echo "ABORT $H: backup $f.$BAK missing or not base"; exit 1; }
      cp -p "$SITE/$f.$BAK" "$SITE/$f.g2tmp"
      mv -f "$SITE/$f.g2tmp" "$SITE/$f"
    done
    find "$SITE/__pycache__" -name 'cuda_communicator.*.pyc' -o -name 'rdma_all_reduce.*.pyc' -o \
      -name 'dsv41_l2_prefetch.*.pyc' 2>/dev/null | xargs -r rm -f
    [ "$PURGE" = --purge ] && rm -rf "$RT"
    status_line ;;
  *) echo "unknown command $CMD"; exit 2 ;;
esac
