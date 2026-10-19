#!/usr/bin/env bash
# Low-overhead GPU telemetry recorder for a single experiment.
set -euo pipefail

usage() {
    echo "usage: $0 --log FILE [--gpus 0,1] [--pid PID] [--interval SEC]" >&2
    exit 2
}

LOG_FILE=""
GPU_LIST=""
PARENT_PID=""
INTERVAL=10
while [[ $# -gt 0 ]]; do
    case "$1" in
        --log) LOG_FILE="$2"; shift 2 ;;
        --gpus) GPU_LIST="$2"; shift 2 ;;
        --pid) PARENT_PID="$2"; shift 2 ;;
        --interval) INTERVAL="$2"; shift 2 ;;
        *) usage ;;
    esac
done
[[ -n "$LOG_FILE" ]] || usage

mkdir -p "$(dirname "$LOG_FILE")"
if [[ ! -s "$LOG_FILE" ]]; then
    echo "timestamp,index,memory_used_mib,memory_total_mib,utilization_gpu_pct,temperature_c,power_w" >"$LOG_FILE"
fi

filter=""
if [[ -n "$GPU_LIST" ]]; then
    filter="$(echo "$GPU_LIST" | tr ',' '|')"
fi
while true; do
    if [[ -n "$PARENT_PID" ]] && ! kill -0 "$PARENT_PID" 2>/dev/null; then
        exit 0
    fi
    timestamp="$(date -Is)"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw \
        --format=csv,noheader,nounits | awk -F, -v ts="$timestamp" -v wanted="$filter" '
            BEGIN {
                if (wanted != "") {
                    count = split(wanted, selected, "|")
                    for (i = 1; i <= count; ++i) include[selected[i]] = 1
                }
            }
            {
                gsub(/^ +| +$/, "", $1)
                if (wanted == "" || include[$1]) {
                    for (i = 2; i <= NF; ++i) gsub(/^ +| +$/, "", $i)
                    print ts "," $1 "," $2 "," $3 "," $4 "," $5 "," $6
                }
            }' >>"$LOG_FILE"
    sleep "$INTERVAL"
done
