import time
import threading
import numpy as np
import psutil
import onnxruntime as ort

model = "Yolo_final.onnx"

def preprocess(img):
    import cv2
    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0)
    return img

cpu_samples = []
stop_flag = False

def monitor_cpu():
    while not stop_flag:
        cpu_samples.append(psutil.cpu_percent(interval=0.1, percpu=True))

session = ort.InferenceSession(
    model,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

for _ in range(50):
    h = np.random.randint(64, 512)
    w = np.random.randint(64, 512)
    img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    inp = preprocess(img)
    session.run(None, {input_name: inp})

iterations = 10000

monitor_thread = threading.Thread(target=monitor_cpu)
monitor_thread.start()

start = time.time()

for _ in range(iterations):
    h = np.random.randint(64, 512)
    w = np.random.randint(64, 512)
    img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    inp = preprocess(img)
    session.run(None, {input_name: inp})

end = time.time()

stop_flag = True
monitor_thread.join()

fps = iterations / (end - start)

cpu_array = np.array(cpu_samples)
avg_per_core = cpu_array.mean(axis=0)
avg_overall = cpu_array.mean()

print(f"FPS (preprocess + inference): {fps:.2f}")
print(f"Total time: {end - start:.2f} s")
print(f"CPU samples: {len(cpu_samples)}")
print(f"Average CPU usage (overall): {avg_overall:.2f}%")
print(f"Average CPU usage (per core): {np.round(avg_per_core, 2)}")

#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