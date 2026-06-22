from ultralytics import YOLO
import cv2

def is_overlapping(box1, box2):
    # box = (x1, y1, x2, y2)
    return not (box1[2] < box2[0] or box1[0] > box2[2] or box1[1] > box2[3] or box1[3] < box2[1])

def get_vehicle_size(x1, y1, x2, y2, threshold=50000):
    # Hitung luas bounding box
    luas = (x2 - x1) * (y2 - y1)
    return "Besar" if luas >= threshold else "Kecil"

# Load model YOLO
model = YOLO("yolo26n.pt")

# Input path gambar dari user
image_path = input("Masukkan path gambar (contoh: Gambar165.jpg): ")

# Load gambar
image = cv2.imread(image_path)
if image is None:
    print(f"Gambar '{image_path}' tidak ditemukan. Pastikan path benar.")
    exit()

# List untuk slot dan kendaraan
list_slot = []  # list of (x1, y1, x2, y2, status) status: 0=kosong, 1=terisi
list_vehicle = []  # list of (x1, y1, x2, y2)

# Deteksi objek dengan confidence 0.3 (lebih sensitif)
results = model(image, conf=0.3)

for result in results:
    for box in result.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if label == "parking space":
            list_slot.append((x1, y1, x2, y2, 0))  # status 0 = kosong

        elif label in ["car", "truck", "bus"]:
            list_vehicle.append((x1, y1, x2, y2))

# Tentukan status slot berdasarkan overlap dengan kendaraan
for i, slot in enumerate(list_slot):
    for vehicle in list_vehicle:
        if is_overlapping(slot[:4], vehicle):
            list_slot[i] = slot[:4] + (1,)  # status 1 = terisi
            break

# Hitung kapasitas, terisi, sisa
kapasitas = len(list_slot)
terisi = sum(1 for s in list_slot if s[4] == 1)
sisa = kapasitas - terisi

# Gambar kotak untuk slot
for slot in list_slot:
    x1, y1, x2, y2, status = slot
    if status == 1:
        color = (0, 0, 255)  # merah untuk terisi
        text = "Terisi"
    else:
        color = (255, 0, 0)  # biru untuk kosong
        text = "Kosong"
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Gambar kotak untuk kendaraan
for vehicle in list_vehicle:
    x1, y1, x2, y2 = vehicle
    ukuran = get_vehicle_size(x1, y1, x2, y2)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # hijau untuk kendaraan
    cv2.putText(image, f"Kendaraan ({ukuran})", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Tampilkan info di layar
cv2.putText(image, f"Kapasitas Parkir: {kapasitas}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

cv2.putText(image, f"Slot Terisi: {terisi}", (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

cv2.putText(image, f"Slot Kosong: {sisa}", (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Hitung jumlah kendaraan besar dan kecil
kendaraan_besar = sum(1 for v in list_vehicle if get_vehicle_size(v[0], v[1], v[2], v[3]) == "Besar")
kendaraan_kecil = len(list_vehicle) - kendaraan_besar

cv2.putText(image, f"Kendaraan Besar: {kendaraan_besar}", (10, 150),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

cv2.putText(image, f"Kendaraan Kecil: {kendaraan_kecil}", (10, 190),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

# Tampilkan hasil
cv2.imshow("Deteksi Slot Parkir", image)
cv2.waitKey(0)
cv2.destroyAllWindows()