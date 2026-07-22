#!/bin/bash
OUTFILE="${1:-power_log.csv}"
RAILS=(3V7_WL_SW 3V3_SYS 1V8_SYS DDR_VDD2 DDR_VDDQ 1V1_SYS 0V8_SW VDD_CORE 3V3_DAC 3V3_ADC 0V8_AON HDMI)
echo "timestamp,total_power_w,$(IFS=,; echo "${RAILS[*]}")" > "$OUTFILE"
trap 'exit 0' SIGINT
while true; do
    RAW=$(vcgencmd pmic_read_adc)
    TS=$(date +%s.%N)
    TOTAL=0
    ROW=()
    for RAIL in "${RAILS[@]}"; do
        C=$(echo "$RAW" | grep "${RAIL}_A" | grep -oP 'current\(\d+\)=\K[0-9.]+')
        V=$(echo "$RAW" | grep "${RAIL}_V" | grep -oP 'volt\(\d+\)=\K[0-9.]+')
        P=$(awk -v v="${V:-0}" -v i="${C:-0}" 'BEGIN{printf "%.6f", v*i}')
        ROW+=("$P")
        TOTAL=$(awk -v t="$TOTAL" -v p="$P" 'BEGIN{printf "%.6f", t+p}')
    done
    echo "$TS,$TOTAL,$(IFS=,; echo "${ROW[*]}")" >> "$OUTFILE"
    sleep 0.05
done