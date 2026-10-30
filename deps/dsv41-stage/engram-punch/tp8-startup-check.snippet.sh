# --- Engram row check (insert in ~/dsv41-tp8.sh after "source .../activate", before ARGS=(serve ...)) ---
# After engram_punch.py a node holds only its own rank's Engram rows; a node started as another rank (or another
# TP size) would load zeros silently (vLLM 0.32 has no row check). Headers + hole map only, ~1 s. ENGRAM_CHECK=0 skips
# it -- never do that on a punched node.
if [ "${ENGRAM_CHECK:-1}" != 0 ]; then
  EP=$HOME/dsv41-stage/engram-punch/engram_punch.py
  [ -f "$EP" ] || { echo "refusing: no $EP for the Engram check (ENGRAM_CHECK=0 skips it)" | tee -a "$LOGDIR/rank$R.log" >&2; exit 1; }
  if ! python3 -I "$EP" --model "$MODEL" --rank "$R" --world 8 --check >> "$LOGDIR/engram-check.log" 2>&1; then
    echo "refusing: $(hostname) Engram tables lack rank $R's rows (see $LOGDIR/engram-check.log; restore: engram-punch/restore_from_four.sh)" | tee -a "$LOGDIR/rank$R.log" >&2
    exit 1
  fi
fi
# --- end Engram row check ---
