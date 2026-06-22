import argparse
import os
import json
import cv2
import numpy as np
from ultralytics import YOLO

# Default paths
IMAGE_PATH = "parkir.JPG"
JSON_PATH = "parking_slots.json"
OUTPUT_IMAGE = "hasil_smart_parking_yolov11.jpg"
MODEL_NAME = "yolov11n.pt"
FALLBACK_MODEL = "yolo26n.pt"

vehicle_classes = [
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "van",
]

MIN_BOX_AREA = 1500


def load_yolo_model(model_name: str) -> YOLO:
    if os.path.exists(model_name):
        print(f"Loading local model {model_name}...")
        return YOLO(model_name)

    try:
        print(f"Loading model {model_name} from Ultralytics hub...")
        return YOLO(model_name)
    except Exception as exc:
        print(f"Warning: gagal load {model_name}: {exc}")
        print(f"Menggunakan fallback model {FALLBACK_MODEL}...")
        return YOLO(FALLBACK_MODEL)


def draw_label(img, text: str, x: int, y: int, color: tuple[int, int, int]):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    text_size, _ = cv2.getTextSize(text, font, scale, thickness)
    text_w, text_h = text_size
    x1 = max(x, 0)
    y1 = max(y - text_h - 8, 0)
    x2 = x1 + text_w + 10
    y2 = y1 + text_h + 8
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), -1)
    cv2.putText(img, text, (x1 + 5, y2 - 4), font, scale, color, thickness, cv2.LINE_AA)


def iou(box1, box2) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0.0


def suppress_duplicates(detections, iou_threshold=0.3):
    kept = []
    for det in sorted(detections, key=lambda d: d["conf"], reverse=True):
        if any(iou(det["xyxy"], other["xyxy"]) > iou_threshold for other in kept):
            continue
        kept.append(det)
    return kept


def main() -> None:
    model = load_yolo_model(MODEL_NAME)

    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"Gagal membaca gambar: {IMAGE_PATH}")
        return

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if isinstance(raw_data, dict):
        slots = raw_data.get("slots", [])
    elif isinstance(raw_data, list):
        slots = raw_data
    else:
        print("Format JSON tidak dikenal. Gunakan array object atau object dengan key 'slots'.")
        return

    if len(slots) == 0:
        print("Tidak ada slot parkir di JSON.")
        return

    h, w = img.shape[:2]
    imgsz = min(max(h, w), 2560)

    print(f"Resolusi: {w}x{h}, imgsz={imgsz}")
    results = model.predict(
        source=img,
        conf=0.12,
        imgsz=imgsz,
        augment=True,
        verbose=False,
    )

    detections = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            if label not in vehicle_classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            width, height = x2 - x1, y2 - y1
            area = width * height
            if area < MIN_BOX_AREA:
                continue

            detections.append({
                "label": label,
                "conf": float(box.conf[0]),
                "xyxy": (x1, y1, x2, y2),
            })

    detections = suppress_duplicates(detections, iou_threshold=0.35)
    occupied_slots = set()

    for det in detections:
        x1, y1, x2, y2 = det["xyxy"]
        label = det["label"]
        conf = det["conf"]
        color = (255, 0, 0)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        draw_label(img, f"{label} {conf:.2f}", x1, y1, color)

        cx = int((x1 + x2) / 2)
        cy = y2 - 8
        cv2.circle(img, (cx, cy), 6, (255, 0, 255), -1)

        for slot in slots:
            polygon = np.array(slot["polygon"], np.int32)
            if cv2.pointPolygonTest(polygon, (cx, cy), False) >= 0:
                occupied_slots.add(slot["id"])
                break

    overlay = img.copy()
    for slot in slots:
        slot_id = slot["id"]
        polygon = np.array(slot["polygon"], np.int32)
        occupied = slot_id in occupied_slots
        color = (0, 0, 255) if occupied else (0, 255, 0)

        cv2.fillPoly(overlay, [polygon], color)
        cv2.polylines(img, [polygon], True, color, 3)

        center_x = int(np.mean(polygon[:, 0]))
        center_y = int(np.mean(polygon[:, 1]))
        draw_label(img, slot_id, center_x - 30, center_y - 20, color)

    cv2.addWeighted(overlay, 0.12, img, 0.88, 0, img)

    total_slots = len(slots)
    occupied = len(occupied_slots)
    vacant = total_slots - occupied
    occupancy_rate = (occupied / total_slots * 100) if total_slots else 0.0

    info_text = [
        f"Vehicles : {len(detections)}",
        f"Occupied : {occupied}",
        f"Vacant   : {vacant}",
        f"Occupancy: {occupancy_rate:.1f}%",
    ]

    cv2.rectangle(img, (20, 20), (420, 180), (255, 255, 255), -1)
    for idx, line in enumerate(info_text):
        cv2.putText(
            img,
            line,
            (35, 60 + idx * 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.95,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(OUTPUT_IMAGE, img)
    print(f"Hasil disimpan di {OUTPUT_IMAGE}")

    cv2.namedWindow("Smart Parking YOLOv11", cv2.WINDOW_NORMAL)
    cv2.imshow("Smart Parking YOLOv11", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
