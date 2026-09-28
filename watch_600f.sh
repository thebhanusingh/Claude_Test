#!/bin/bash
# Watches run_600f.sh (PID given as $1). Writes a detailed status line every 15s to
# run_600f_status.log (useful after a silent crash), and prints a summary to stdout
# every 5 min, on every stage change, on errors, and when the pipeline exits.
PID=$1
DONE_PAT=${3:-"=== DONE"}
LOG=${2:-~/Claude_Test/run_600f.log}
STATUS=${LOG%.log}_status.log
last_stage=""; last_err=0; n=0
while true; do
  now=$(date '+%T')
  gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader 2>/dev/null || echo "nvidia-smi failed")
  ram=$(free -m | awk '/Mem:/{printf "%.1f/%.1f GB used", $3/1024, $2/1024}')
  stage=$(grep -a "^=== \[" "$LOG" | tail -1 | cut -c1-40)
  step=$(tr '\r' '\n' < "$LOG" | grep -aE "^[0-9]+ \([0-9.]+%\)" | tail -1 | awk '{print $1, $2}')
  echo "$now | $stage | step $step | GPU $gpu | RAM $ram" >> "$STATUS"

  if [[ "$stage" != "$last_stage" ]]; then
    echo "STAGE CHANGE $now: $stage | GPU $gpu | RAM $ram"; last_stage=$stage
  fi
  errs=$(grep -acE "Traceback|Error|FAILED|Killed|out of memory|OutOfMemory" "$LOG")
  if (( errs > last_err )); then
    echo "ERROR SEEN $now: $(grep -aE 'Traceback|Error|FAILED|Killed|out of memory|OutOfMemory' "$LOG" | tail -3 | tr '\n' ' ' | cut -c1-300)"
    last_err=$errs
  fi
  if grep -aq "$DONE_PAT" "$LOG"; then
    echo "PIPELINE FINISHED $now: $(grep -a "$DONE_PAT" "$LOG" | cut -c1-200)"; exit 0
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "PIPELINE PROCESS GONE $now (no DONE line). Last log: $(tr '\r' '\n' < "$LOG" | grep -av '^\s*$' | tail -3 | tr '\n' ' ' | cut -c1-300)"; exit 1
  fi
  if (( n % 20 == 0 )); then
    echo "UPDATE $now: $stage | step ${step:-n/a} | GPU $gpu | RAM $ram"
  fi
  n=$((n+1)); sleep 15
done
