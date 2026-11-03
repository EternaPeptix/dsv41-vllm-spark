#!/bin/bash
# site8.sh install|uninstall|status [patched|base]: run tools/boundary_install.py on all 8. Passes only when every host answered, exited 0, printed a STATE line,
# all 8 agree, and the state is the required one: install wants patched, uninstall wants base, status wants what it is given (otherwise any one state, not
# UNKNOWN). install and uninstall need the service stopped on every host (the tool refuses otherwise). Run from the Mac. Exit 0 pass, 1 fail, 2 bad usage.
# Tested against stub hosts by tests/test_ops_stub.py; NOT RUN YET on the real 8.
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
S=$(mktemp -d "${TMPDIR:-/tmp}/boundary8.XXXXXX"); trap 'rm -rf "$S"' EXIT
REMOTE="cd ~/dsv41-stage/g130/boundary && python3 -I tools/boundary_install.py $CMD"
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
  elif [ ! -s "$S/$h.out" ]; then why="no output"
  elif ! grep -q '^STATE ' "$S/$h.out"; then why="no STATE line"
  elif [ "$(grep -cE '^(models|v1)/' "$S/$h.out")" -lt 3 ]; then why="missing file lines"
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
# Last three file lines and the STATE line: the three hashes and the state must match on all 8.
n=$(for h in $HOSTS; do { grep -E '^(models|v1)/' "$S/$h.out" 2>/dev/null | tail -3 || true; grep '^STATE ' "$S/$h.out" 2>/dev/null | tail -1 || true; } | shasum -a 256; done | sort -u | wc -l | tr -d ' ')
if [ "$n" != 1 ]; then
  echo "FAIL hosts differ:"
  for h in $HOSTS; do echo "== $h"; cut -c1-200 "$S/$h.out" 2>/dev/null || true; done
  rc=1
fi
if [ "$rc" = 0 ]; then
  grep -E '^(models|v1)/' "$S/spark-one.out" | tail -3
  echo "site8 $CMD: 8/8 exit 0, identical, state $(grep '^STATE ' "$S/spark-one.out" | tail -1 | cut -d' ' -f2)"
else
  echo "site8 $CMD: FAILED"
fi
exit $rc
