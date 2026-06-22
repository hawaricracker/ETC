from ultralytics import YOLO
import cv2
import json
import numpy as np

# =====================================================
# CONFIG
# =====================================================

IMAGE_PATH = "parkir.jpg"
JSON_PATH = "parking_slots.json"
OUTPUT_IMAGE = "hasil_smart_parking.jpg"

# =====================================================
# LOAD MODEL
# =====================================================

print("Loading YOLOv11n...")

model = YOLO("yolo11n.pt")

vehicle_classes = [
    "car",
    "truck",
    "bus",
    "motorcycle"
]

# =====================================================
# LOAD IMAGE
# =====================================================

img = cv2.imread(IMAGE_PATH)

if img is None:
    print("Gagal membaca gambar.")
    exit()

h, w = img.shape[:2]

print(f"Resolusi gambar: {w} x {h}")

# =====================================================
# LOAD SLOT JSON
# =====================================================

with open(JSON_PATH, "r") as f:
    parking_data = json.load(f)

print(f"Jumlah slot: {len(parking_data['slots'])}")

# =====================================================
# YOLO DETECTION
# =====================================================

# Gunakan resolusi tinggi agar mobil kecil lebih mudah terdeteksi
imgsz = max(h, w)

# Batasi agar tidak terlalu berat
if imgsz > 2560:
    imgsz = 2560

print(f"YOLO Image Size: {imgsz}")

results = model.predict(
    source=img,
    conf=0.10,
    imgsz=imgsz,
    augment=True,
    verbose=False
)

occupied_slots = set()
detected_vehicles = 0

# =====================================================
# DETEKSI KENDARAAN
# =====================================================

for result in results:

    for box in result.boxes:

        cls_id = int(box.cls[0])

        label = model.names[cls_id]

        if label not in vehicle_classes:
            continue

        conf = float(box.conf[0])

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        width = x2 - x1
        height = y2 - y1

        box_area = width * height

        # Filter noise kecil
        if box_area < 1500:
            continue

        detected_vehicles += 1

        # Bounding box
        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            3
        )

        cv2.putText(
            img,
            f"{label} {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

        # Titik bawah kendaraan
        cx = int((x1 + x2) / 2)
        cy = int(y2 - 5)

        cv2.circle(
            img,
            (cx, cy),
            6,
            (255, 0, 255),
            -1
        )

        # Cek slot
        for slot in parking_data["slots"]:

            polygon = np.array(
                slot["polygon"],
                np.int32
            )

            inside = cv2.pointPolygonTest(
                polygon,
                (cx, cy),
                False
            )

            if inside >= 0:

                occupied_slots.add(
                    slot["id"]
                )

# =====================================================
# VISUALISASI SLOT
# =====================================================

for slot in parking_data["slots"]:

    slot_id = slot["id"]

    polygon = np.array(
        slot["polygon"],
        np.int32
    )

    if slot_id in occupied_slots:
        color = (0, 0, 255)  # merah
    else:
        color = (0, 255, 0)  # hijau

    cv2.polylines(
        img,
        [polygon],
        True,
        color,
        4
    )

    center_x = int(np.mean(polygon[:, 0]))
    center_y = int(np.mean(polygon[:, 1]))

    cv2.putText(
        img,
        slot_id,
        (center_x, center_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        3
    )

# =====================================================
# STATISTIK
# =====================================================

total_slots = len(parking_data["slots"])
occupied = len(occupied_slots)
vacant = total_slots - occupied

occupancy_rate = (
    occupied / total_slots
) * 100

print("\n========== HASIL ==========")
print(f"Kendaraan Terdeteksi : {detected_vehicles}")
print(f"Total Slot           : {total_slots}")
print(f"Occupied             : {occupied}")
print(f"Vacant               : {vacant}")
print(f"Occupancy Rate       : {occupancy_rate:.2f}%")
print("===========================\n")

# =====================================================
# PANEL INFORMASI
# =====================================================

cv2.rectangle(
    img,
    (20, 20),
    (900, 300),
    (255, 255, 255),
    -1
)

cv2.putText(
    img,
    f"Vehicles : {detected_vehicles}",
    (40, 80),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.5,
    (255, 0, 0),
    3
)

cv2.putText(
    img,
    f"Occupied : {occupied}",
    (40, 150),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.5,
    (0, 0, 255),
    3
)

cv2.putText(
    img,
    f"Vacant : {vacant}",
    (40, 220),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.5,
    (0, 180, 0),
    3
)

cv2.putText(
    img,
    f"Occupancy : {occupancy_rate:.1f}%",
    (40, 290),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.3,
    (0, 0, 0),
    3
)

# =====================================================
# SAVE
# =====================================================

cv2.imwrite(
    OUTPUT_IMAGE,
    img
)

print(f"Hasil disimpan ke {OUTPUT_IMAGE}")

# =====================================================
# DISPLAY
# =====================================================

cv2.namedWindow(
    "Smart Parking",
    cv2.WINDOW_NORMAL
)

cv2.imshow(
    "Smart Parking",
    img
)

cv2.waitKey(0)
cv2.destroyAllWindows()