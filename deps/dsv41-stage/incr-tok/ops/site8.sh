#!/bin/bash
# site8.sh install|uninstall|status [patched|base]: run tools/incr_install.py on all 8 in parallel. Passes only when every host answered, exited 0,
# printed a STATE line and both file lines, all 8 agree, and the state is the required one: install wants patched, uninstall wants base, status wants
# what it is given (otherwise any one state, never UNKNOWN). install and uninstall need the service stopped on every host (the tool refuses
# otherwise). Run from the Mac. Exit 0 pass, 1 fail, 2 bad usage. NOT RUN YET on the real 8.
set -euo pipefail
CMD=${1:?install|uninstall|status [patched|base]}
WANT=${2:-}
case "$CMD" in
  install) [ -z "$WANT" ] || { echo "install takes no state"; exit 2; }; WANT=patched ;;
  uninstall) [ -z "$WANT" ] || { echo "uninstall takes no state"; exit 2; }; WANT=base ;;
  status) case "$WANT" in ''|patched|base) ;; *) echo "bad state $WANT"; exit 2 ;; esac ;;
  *) echo "bad command $CMD"; exit 2 ;;
esac
HOSTS="spark-one spark-two spark-three spark-four spark-five spark-six spark-seven spark-eight"
S=$(mktemp -d "${TMPDIR:-/tmp}/incrtok8.XXXXXX"); trap 'rm -rf "$S"' EXIT
if [ "$CMD" = install ]; then  # all 8 must pass the precheck (base sha, anchors, service stopped) before any host is written
  bad=""
  for h in $HOSTS; do
    ( ssh -n -o ConnectTimeout=10 -o BatchMode=yes "$h" "cd ~/dsv41-stage/incr-tok && python3 -I tools/incr_install.py precheck" > "$S/$h.pre" 2>&1; echo $? > "$S/$h.prerc" ) &
  done
  wait
  for h in $HOSTS; do
    if [ "$(cat "$S/$h.prerc" 2>/dev/null)" != 0 ] || ! grep -q '^precheck OK' "$S/$h.pre"; then
      echo "FAIL $h precheck: $(tail -1 "$S/$h.pre" 2>/dev/null | cut -c1-200)"; bad=1
    fi
  done
  [ -z "$bad" ] || { echo "site8 install: precheck failed, nothing written on any host"; exit 1; }
fi
REMOTE="cd ~/dsv41-stage/incr-tok && python3 -I tools/incr_install.py $CMD"
[ "$CMD" = status ] && REMOTE="$REMOTE $WANT"
for h in $HOSTS; do
  ( rc=0; ssh -n -o ConnectTimeout=10 -o BatchMode=yes "$h" "$REMOTE" > "$S/$h.out" 2>&1 || rc=$?; echo "$rc" > "$S/$h.rc" ) &
done
wait
rc=0
for h in $HOSTS; do
  why=""
  if [ ! -s "$S/$h.rc" ]; then why="no exit code"
  elif [ "$(cat "$S/$h.rc")" != 0 ]; then why="exit $(cat "$S/$h.rc")"
  elif ! grep -q '^STATE ' "$S/$h.out"; then why="no STATE line"
  elif [ "$(grep -cE '^renderers/' "$S/$h.out")" -lt 2 ]; then why="missing file lines"
  fi
  state=$(grep '^STATE ' "$S/$h.out" 2>/dev/null | tail -1 | cut -d' ' -f2 || true)
  if [ -z "$why" ] && [ -n "$WANT" ] && [ "$state" != "$WANT" ]; then why="state ${state:-none}, want $WANT"; fi
  if [ -z "$why" ] && [ "$state" = UNKNOWN ]; then why="state UNKNOWN"; fi
  if [ -n "$why" ]; then
    echo "FAIL $h: $why"
    tail -6 "$S/$h.out" 2>/dev/null | cut -c1-200 | sed 's/^/    /' || true
    rc=1
  fi
done
# The last two file lines and the STATE line must be identical on all 8.
n=$(for h in $HOSTS; do { grep -E '^renderers/' "$S/$h.out" 2>/dev/null | tail -2 || true; grep '^STATE ' "$S/$h.out" 2>/dev/null | tail -1 || true; } | shasum -a 256; done | sort -u | wc -l | tr -d ' ')
if [ "$n" != 1 ]; then
  echo "FAIL hosts differ:"
  for h in $HOSTS; do echo "== $h"; cut -c1-200 "$S/$h.out" 2>/dev/null || true; done
  rc=1
fi
if [ "$rc" = 0 ]; then
  grep -E '^renderers/' "$S/spark-one.out" | tail -2
  echo "site8 $CMD: 8/8 exit 0, identical, state $(grep '^STATE ' "$S/spark-one.out" | tail -1 | cut -d' ' -f2)"
else
  echo "site8 $CMD: FAILED"
fi
exit $rc
