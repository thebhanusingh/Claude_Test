#!/bin/bash
# Watches a GaussianShader run (tqdm progress). Args: PID LOG TOTAL_ITERS
# Status line every 15 s to <log>_status.log; stdout every 5 min, on errors, finish, or crash.
PID=$1; L=$2; TOTAL=${3:-30000}
STATUS=${L%.log}_status.log
n=0; lasterr=$(grep -acE "Traceback|Error|out of memory|Killed" "$L")
while true; do
  prog=$(tr '\r' '\n' < "$L" | grep -aoE "[0-9]+/$TOTAL \[[^]]*\]" | tail -1)
  gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader 2>/dev/null)
  ram=$(free -m | awk '/Mem:/{printf "%.1f/%.1f GB, avail %.1f", $3/1024, $2/1024, $7/1024}')
  echo "$(date +%T) | $prog | GPU $gpu | RAM $ram" >> "$STATUS"
  e=$(grep -acE "Traceback|Error|out of memory|Killed" "$L")
  if (( e > lasterr )); then
    echo "ERROR $(date +%T): $(grep -aE 'Error|out of memory|Killed' "$L" | tail -2 | tr '\n' ' ' | cut -c1-250)"
    lasterr=$e
  fi
  if grep -aqE "=== DONE|FAILED" "$L"; then
    echo "FINISHED $(date +%T): $(grep -aE '=== DONE|FAILED' "$L" | tail -1)"; exit 0
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "PROCESS GONE $(date +%T): $(tr '\r' '\n' < "$L" | grep -v '^\s*$' | tail -2 | tr '\n' ' ' | cut -c1-250)"; exit 1
  fi
  (( n % 20 == 0 )) && echo "UPDATE $(date +%T): ${prog:-starting} | GPU $gpu | RAM $ram"
  n=$((n+1)); sleep 15
done
