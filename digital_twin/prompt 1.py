from ultralytics import YOLO
import cv2
import pandas as pd
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import stats

from config import TOTAL_SLOTS, LOG_INTERVAL, CONFIDENCE_THRESHOLD, CSV_FILE

# Membuat folder jika belum ada
os.makedirs('data', exist_ok=True)
os.makedirs('output', exist_ok=True)

# Load model YOLOv8 Nano (ringan untuk Raspberry Pi)
model = YOLO('yolov8n.pt')

# Buka kamera
cap = cv2.VideoCapture(0)

# Buat file CSV jika belum ada
if not os.path.exists(CSV_FILE):
    pd.DataFrame(columns=[
        'timestamp',
        'vehicle_count',
        'occupied_slots',
        'available_slots',
        'occupancy_percent'
    ]).to_csv(CSV_FILE, index=False)

last_log_time = 0

print('Smart Parking System berjalan. Tekan q untuk keluar.')

while True:
    ret, frame = cap.read()
    if not ret:
        print('Kamera tidak terbaca.')
        break

    # Deteksi objek
    results = model(frame, conf=CONFIDENCE_THRESHOLD)

    vehicle_count = 0

    for box in results[0].boxes:
        cls = int(box.cls[0])
        label = model.names[cls]

        if label in ['car', 'motorcycle', 'truck', 'bus']:
            vehicle_count += 1

    # Asumsi sederhana: jumlah kendaraan = slot terisi
    occupied_slots = min(vehicle_count, TOTAL_SLOTS)
    available_slots = TOTAL_SLOTS - occupied_slots
    occupancy_percent = (occupied_slots / TOTAL_SLOTS) * 100

    # Tampilkan hasil
    annotated = results[0].plot()

    cv2.putText(
        annotated,
        f'Terisi: {occupied_slots}/{TOTAL_SLOTS}',
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated,
        f'Kosong: {available_slots}',
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2
    )

    cv2.putText(
        annotated,
        f'Okupansi: {occupancy_percent:.1f}%',
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    cv2.imshow('Smart Parking', annotated)

    # Simpan data setiap LOG_INTERVAL detik
    current_time = time.time()
    if current_time - last_log_time >= LOG_INTERVAL:
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        new_data = pd.DataFrame([{
            'timestamp': now,
            'vehicle_count': vehicle_count,
            'occupied_slots': occupied_slots,
            'available_slots': available_slots,
            'occupancy_percent': occupancy_percent
        }])

        new_data.to_csv(CSV_FILE, mode='a', header=False, index=False)
        print(f'Data tersimpan: {now}')
        last_log_time = current_time

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# =============================
# ANALISIS STATISTIK
# =============================

print('Membuat analisis statistik...')

df = pd.read_csv(CSV_FILE)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Statistik deskriptif
mean_occ = df['occupancy_percent'].mean()
median_occ = df['occupancy_percent'].median()
mode_occ = stats.mode(df['occupancy_percent'], keepdims=True).mode[0]
std_occ = df['occupancy_percent'].std()
max_occ = df['occupancy_percent'].max()
min_occ = df['occupancy_percent'].min()

print('\n===== HASIL ANALISIS =====')
print(f'Rata-rata okupansi : {mean_occ:.2f}%')
print(f'Median             : {median_occ:.2f}%')
print(f'Modus              : {mode_occ:.2f}%')
print(f'Standar deviasi    : {std_occ:.2f}')
print(f'Maksimum           : {max_occ:.2f}%')
print(f'Minimum            : {min_occ:.2f}%')

# Grafik okupansi
plt.figure(figsize=(10, 5))
plt.plot(df['timestamp'], df['occupancy_percent'], marker='o')
plt.xlabel('Waktu')
plt.ylabel('Okupansi (%)')
plt.title('Grafik Okupansi Parkir')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('output/occupancy_plot.png')

print('Grafik disimpan ke output/occupancy_plot.png')
