#!/bin/sh
# Safety net for long runs on a 16 GB machine: kill the pipeline's Python jobs if
# system free memory drops below a floor, instead of letting macOS run out of
# memory (which caused a kernel panic once). Usage: sh src/memguard.sh [floor_pct]
FLOOR=${1:-12}
while true; do
  FREE=$(memory_pressure | awk -F': ' '/free percentage/ {gsub("%","",$2); print $2}')
  if [ -n "$FREE" ] && [ "$FREE" -lt "$FLOOR" ]; then
    echo "$(date '+%H:%M:%S') free memory ${FREE}% < ${FLOOR}% -> stopping pipeline jobs"
    pkill -f "src/(run_blocking|train_model|predict|eval_country_transfer).py"
  fi
  sleep 5
done
