#!/bin/bash
# g2-site.sh <install|uninstall|status>  -- run ON one Spark (driver: g2-all.sh from the Mac).
# Candidate g2 fc2c-on-m32all (~/dsv41-stage/g2/fc2c-on-m32all). Same contract as the g1 site scripts:
#  FILES lines:  <path relative to site-packages>  <accepted-before sha256[,...]>  <staged sha256>
#  install:   current == staged -> no-op; current in accepted-before -> backup <file>.bak-g2-fc2c-on-m32all, copy,
#             verify (requires PREP.ok: tree built and pinned, pre-JIT list pinned, CPU test PASS on this node)
#  uninstall: current == staged -> restore the backup, verify it is an accepted baseline
#  status:    path current-sha8 state, plus PREP.ok, the tree manifest and the serving pre-JIT entry
# Refuses install/uninstall while dsv41-rank is active unless G2_ALLOW_ACTIVE=1 (applies at the next start).
# With DSV41_MOE_FC2C unset the installed file is call-for-call identical to db659d80 (tests/test_g2_cpu.py), so it is
# inert until a launch passes DSV41_MOE_FC2C=1 (with the live DSV41_MOE_M32_G1=all and DSV41_MOE_FC2_D05=1).
set -u
ACTION=${1:?action}
TAG=fc2c-on-m32all
STAGE=$HOME/dsv41-stage/g2/$TAG
SITE=$HOME/venvs/vllm032/lib/python3.12/site-packages
HOST=$(hostname)
[ -f "$STAGE/FILES" ] || { echo "G2 $HOST $TAG $ACTION FAIL no $STAGE/FILES"; exit 2; }
if [ "$ACTION" != status ] && [ "${G2_ALLOW_ACTIVE:-0}" != 1 ] && systemctl --user is-active --quiet dsv41-rank; then
  echo "G2 $HOST $TAG $ACTION FAIL dsv41-rank active (G2_ALLOW_ACTIVE=1 to stage for the next start)"; exit 3
fi
if [ "$ACTION" = install ] && [ ! -f "$STAGE/PREP.ok" ]; then
  echo "G2 $HOST $TAG install FAIL no PREP.ok (bash $STAGE/ops/prep-node.sh build)"; exit 3
fi
sha() { [ -e "$1" ] && sha256sum "$1" | cut -d' ' -f1 || echo ABSENT; }
rc=0; digest=""
while read -r rel before staged; do
  case "$rel" in ''|'#'*) continue;; esac
  f=$SITE/$rel; src=$STAGE/site/$rel; bak=$f.bak-g2-$TAG
  cur=$(sha "$f")
  [ "$(sha "$src")" = "$staged" ] || { echo "  $rel: staged copy sha mismatch"; rc=1; continue; }
  case "$ACTION" in
    status)
      st=other
      [ "$cur" = "$staged" ] && st=staged
      case ",$before," in *",$cur,"*) st=baseline;; esac
      echo "  $rel ${cur:0:8} $st$( [ -e "$bak" ] && echo " (backup $(sha "$bak" | cut -c1-8))")"
      digest+="$rel:${cur:0:8};";;
    install)
      if [ "$cur" = "$staged" ]; then :
      elif case ",$before," in *",$cur,"*) true;; *) false;; esac; then
        [ -e "$bak" ] || cp -p "$f" "$bak"
        cp "$src" "$f.g2-tmp" && mv -f "$f.g2-tmp" "$f"
      else
        echo "  $rel: refusing, current ${cur:0:8} is neither staged nor an accepted baseline ($before)"; rc=1; continue
      fi
      cur=$(sha "$f"); [ "$cur" = "$staged" ] || { echo "  $rel: post-install sha ${cur:0:8} != staged"; rc=1; }
      digest+="$rel:${cur:0:8};";;
    uninstall)
      if [ "$cur" = "$staged" ]; then
        if [ -e "$bak" ]; then cp -p "$bak" "$f.g2-tmp" && mv -f "$f.g2-tmp" "$f"
        else echo "  $rel: no backup $bak"; rc=1; continue; fi
      fi
      cur=$(sha "$f")
      case ",$before," in *",$cur,"*) ;; *) echo "  $rel: post-uninstall sha ${cur:0:8} not an accepted baseline"; rc=1;; esac
      digest+="$rel:${cur:0:8};";;
    *) echo "unknown action $ACTION"; exit 2;;
  esac
done < "$STAGE/FILES"
if [ "$ACTION" = status ]; then
  files=$(cd "$STAGE" && sha256sum -c --quiet MANIFEST.sha256 >/dev/null 2>&1 && echo files-ok || echo files-BAD)
  tree=$( [ -f "$STAGE/tree/manifest.json" ] && sha256sum "$STAGE/tree/manifest.json" | cut -c1-8 || echo none)
  jit=$(ls "$STAGE/jit-cache/cache" 2>/dev/null | tr '\n' ' ')
  echo "  stage $files, PREP.ok $( [ -f "$STAGE/PREP.ok" ] && cut -c1-16 "$STAGE/PREP.ok" || echo missing), tree $tree, pre-JIT: $jit"
fi
d=$(printf %s "$digest" | sha256sum | cut -c1-12)
[ $rc = 0 ] && echo "G2 $HOST $TAG $ACTION OK $d" || echo "G2 $HOST $TAG $ACTION FAIL $d"
exit $rc
