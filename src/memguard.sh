#!/bin/sh
# Secondary safety net (the primary one is the pipeline lock in src/ber/guard.py).
# Both kernel panics happened when macOS's memory COMPRESSOR filled up, while the
# "free memory %" still looked fine. So this watches the compressor size (vm_stat)
# and stops the pipeline's Python jobs if it grows past a limit.
# Usage: sh src/memguard.sh [max_compressed_GB]   (default 7 on a 16 GB Mac)
LIMIT_GB=${1:-7}
while true; do
  COMP_GB=$(vm_stat | awk '/page size of/ {ps=$8} /occupied by compressor/ {gsub("\\.","",$5); print int($5*ps/1073741824)}')
  if [ -n "$COMP_GB" ] && [ "$COMP_GB" -ge "$LIMIT_GB" ]; then
    echo "$(date '+%H:%M:%S') compressed memory ${COMP_GB} GB >= ${LIMIT_GB} GB -> stopping pipeline jobs"
    pkill -f "src/(run_blocking|train_model|predict|error_analysis|eval_country_transfer|build_cache|audit_normalization|compare_normalization|learn_translit).py"
  fi
  sleep 3
done
