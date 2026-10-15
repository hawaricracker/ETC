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

# BARU: nyalakan pencatat daya + ukur idle
p = subprocess.Popen(["bash", "power_log.sh", "power_log.csv"])
idle_start = time.time()
time.sleep(10)
idle_end = time.time()
hasil = []

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

    iterations = 10

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
    time.sleep(30)                         # BARU: cooldown

    model_onnx = onnx.load(model)

    """model_profile(
        model_onnx,
        {
            "input": np.random.randn(1, 1, 64, 64).astype(np.float32)
        }
    )"""

# BARU: matikan pencatat, lalu hitung daya per model
p.terminate()
time.sleep(1)

d = pd.read_csv("power_log.csv")
idle = d[(d.timestamp > idle_start) & (d.timestamp < idle_end)].total_power_w.mean()

out = []
for file, fps, s, e in hasil:
    w = d[(d.timestamp > s) & (d.timestamp < e)].total_power_w.mean()
    out.append([file, fps, w, idle, (w - idle) / fps])

pd.DataFrame(out, columns=["model", "fps", "watt", "idle_w", "joule_per_inf"]).to_csv("hasil.csv", index=False)
print(pd.read_csv("hasil.csv"))