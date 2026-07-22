#!/bin/sh
CSV="${1:-power_log.csv}"

./power_profiler.sh "$CSV" &
PROFILER_PID=$!

python3 onnx_benchmark.py

kill "$PROFILER_PID" 2>/dev/null
pkill -P "$PROFILER_PID" 2>/dev/null
pkill -f power_profiler.sh
wait "$PROFILER_PID" 2>/dev/null

echo "Done. Power log saved to $CSV"