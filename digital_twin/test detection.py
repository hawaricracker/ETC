from ultralytics import YOLO
import cv2

# Load model YOLO
model = YOLO("yolo26n.pt")

# Load gambar
image = cv2.imread("parkir.jpg")

# Counter kendaraan
count = 0

# Deteksi objek
results = model(image)

for result in results:
    for box in result.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]

        # Filter kendaraan
        if label in ["car", "truck", "bus"]:
            count += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Gambar kotak
            cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(image, label, (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

# Kapasitas parkir (ubah sesuai kebutuhan)
kapasitas = 20
sisa = kapasitas - count

# Tampilkan info di layar
cv2.putText(image, f"Jumlah Kendaraan: {count}", (10,30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

cv2.putText(image, f"Sisa Parkir: {sisa}", (10,70),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

# Tampilkan hasil
cv2.imshow("Deteksi Kendaraan", image)
cv2.waitKey(0)
cv2.destroyAllWindows()