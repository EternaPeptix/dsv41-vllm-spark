#!/bin/bash
# g130 fc1 site install on this node (SERVICE STOPPED). install: base sha guard, backup, patched file, then verify. uninstall: restores the
# backup, and refuses if the live file is neither ours nor the base. status: prints which it is. check: computes the patched sha, writes
# nothing outside the kit. Flags stay unset, so after install the service is production by construction.
set -u
K=$HOME/dsv41-stage/g130/fc1
F=$HOME/venvs/vllm032/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/experts/sm121_target_m32.py
BASE=ea4a8814986c3618d26d7acb748561d75e50e839cbea225d9f06154febb290e5
cd "$K" || exit 1
mkdir -p work
python3 -I site/make_site.py site/base_sm121_target_m32.py work/sm121_target_m32.patched.py MANIFESTS.json > /dev/null || exit 1
NEW=$(sha256sum work/sm121_target_m32.patched.py | cut -c1-64)
CUR=$(sha256sum "$F" | cut -c1-64)
active() { systemctl --user is-active dsv41-rank 2>/dev/null; }
case "${1:-status}" in
  check)  echo "$(hostname) live ${CUR:0:12} base ${BASE:0:12} patched ${NEW:0:12}" ;;
  status) if [ "$CUR" = "$NEW" ]; then echo "$(hostname) INSTALLED ${NEW:0:12}"; elif [ "$CUR" = "$BASE" ]; then echo "$(hostname) BASE ${BASE:0:12}"; else echo "$(hostname) OTHER ${CUR:0:12}"; fi ;;
  install)
    [ "$(active)" = active ] && { echo "$(hostname) REFUSED: service active"; exit 1; }
    [ "$CUR" = "$NEW" ] && { echo "$(hostname) already installed"; exit 0; }
    [ "$CUR" = "$BASE" ] || { echo "$(hostname) REFUSED: live file ${CUR:0:12} is not base ${BASE:0:12}"; exit 1; }
    cp -a "$F" "$F.bak-g130-fc1" && cp work/sm121_target_m32.patched.py "$F" || exit 1
    [ "$(sha256sum "$F" | cut -c1-64)" = "$NEW" ] && echo "$(hostname) installed ${NEW:0:12}" ;;
  uninstall)
    [ "$(active)" = active ] && { echo "$(hostname) REFUSED: service active"; exit 1; }
    [ "$CUR" = "$BASE" ] && { echo "$(hostname) already base"; exit 0; }
    [ "$CUR" = "$NEW" ] || { echo "$(hostname) REFUSED: live file ${CUR:0:12} is not ours"; exit 1; }
    [ "$(sha256sum "$F.bak-g130-fc1" | cut -c1-64)" = "$BASE" ] || { echo "$(hostname) REFUSED: backup is not base"; exit 1; }
    cp -a "$F.bak-g130-fc1" "$F" && [ "$(sha256sum "$F" | cut -c1-64)" = "$BASE" ] && echo "$(hostname) restored ${BASE:0:12}" ;;
  *) echo "usage: site-node.sh check|status|install|uninstall"; exit 2 ;;
esac
