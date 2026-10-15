import os
import cv2
import time
import numpy as np
import onnx
import onnxruntime as ort
from onnx_tool import model_profile
import subprocess   # BARU
import pandas as pd # BARU

def preprocess(img):
    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0)
    return img

# ===================== BARU: pencatat daya otomatis =====================
SCRIPT = os.path.abspath("power_log.sh")
CSV = os.path.abspath("power_log.csv")

with open(SCRIPT, "w") as f:
    f.write(r'''#!/bin/bash
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
    sleep 0.1
done
''')

if os.path.exists(CSV):
    os.remove(CSV)
p = subprocess.Popen(["bash", SCRIPT, CSV])

# tunggu sampai CSV benar-benar berisi data
for _ in range(50):
    if os.path.exists(CSV) and sum(1 for _ in open(CSV)) > 2:
        break
    time.sleep(0.2)
else:
    p.terminate()
    raise SystemExit("power_log.sh gagal menghasilkan data (cek: vcgencmd pmic_read_adc)")

# ukur daya idle
idle_start = time.time()
time.sleep(5)
idle_end = time.time()
hasil = []
# ========================================================================

for file in os.listdir("onnx_model"):
    model = f'onnx_model/{file}'

    so = ort.SessionOptions()

    session = ort.InferenceSession(
        model,
        sess_options=so,
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name

    dummy_rgb = np.random.randint(
        0,
        256,
        (200, 300, 3),
        dtype=np.uint8
    )

    dummy_input = preprocess(dummy_rgb)

    session.run(None, {input_name: dummy_input})

    session1 = ort.InferenceSession(
        model,
        providers=["CPUExecutionProvider"]
    )

    input_name = session1.get_inputs()[0].name

    for _ in range(50):
        h = np.random.randint(64, 512)
        w = np.random.randint(64, 512)

        img = np.random.randint(
            0,
            256,
            (h, w, 3),
            dtype=np.uint8
        )

        inp = preprocess(img)

        session1.run(None, {input_name: inp})

    iterations = 200  # DIUBAH dari 10 supaya ada cukup sampel daya

    start = time.time()

    for _ in range(iterations):
        h = np.random.randint(64, 512)
        w = np.random.randint(64, 512)

        img = np.random.randint(
            0,
            256,
            (h, w, 3),
            dtype=np.uint8
        )

        inp = preprocess(img)

        session1.run(None, {input_name: inp})

    end = time.time()

    fps = iterations / (end - start)

    print(f"FPS (preprocess + inference): {fps} | {file}")

    hasil.append((file, fps, start, end))  # BARU
    time.sleep(10)                         # BARU: cooldown

    model_onnx = onnx.load(model)

    """model_profile(
        model_onnx,
        {
            "input": np.random.randn(1, 1, 64, 64).astype(np.float32)
        }
    )"""

# ===================== BARU: hitung daya & simpan hasil =====================
p.terminate()
time.sleep(1)

d = pd.read_csv(CSV, on_bad_lines="skip")
d["timestamp"] = pd.to_numeric(d["timestamp"], errors="coerce")
d["total_power_w"] = pd.to_numeric(d["total_power_w"], errors="coerce")

idle = d[(d.timestamp > idle_start) & (d.timestamp < idle_end)].total_power_w.mean()

out = []
for file, fps, s, e in hasil:
    w = d[(d.timestamp > s) & (d.timestamp < e)].total_power_w.mean()
    out.append([file, fps, w, idle, (w - idle) / fps])

df = pd.DataFrame(out, columns=["model", "fps", "watt", "idle_w", "joule_per_inf"])
df.to_csv("hasil.csv", index=False)
print(df)