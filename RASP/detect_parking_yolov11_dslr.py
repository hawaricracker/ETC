import argparse
import json
import os

import cv2
import numpy as np
from ultralytics import YOLO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IMAGE_PATHS = [
    os.path.join(SCRIPT_DIR, "rooftop 80 day 1/dslr.JPG"),
    os.path.join(SCRIPT_DIR, "rooftop 80 day 2/dslr.JPG"),
    os.path.join(SCRIPT_DIR, "dslr.JPG"),
]
DEFAULT_JSON_PATH = os.path.join(SCRIPT_DIR, "parking_slots.json")
DEFAULT_OUTPUT_IMAGE = os.path.join(SCRIPT_DIR, "hasil_smart_parking_yolov11_dslr.jpg")
DEFAULT_MODEL = os.path.join(SCRIPT_DIR, "yolov11n.pt")
FALLBACK_MODEL = os.path.join(SCRIPT_DIR, "yolo26n.pt")

VEHICLE_CLASSES = [
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "van",
]
MIN_BOX_AREA = 1500
MIN_SLOT_OVERLAP_RATIO = 0.15
MIN_SLOT_COVERAGE_RATIO = 0.05


def parse_args():
    parser = argparse.ArgumentParser(
        description="Deteksi parkir dari gambar DSLR menggunakan YOLOv11."
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Path ke gambar DSLR. Jika tidak diset, akan mencoba default DSLR paths.",
    )
    parser.add_argument(
        "--json",
        default=DEFAULT_JSON_PATH,
        help="Path ke file JSON slot parkir.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_IMAGE,
        help="Path file hasil output.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model YOLO yang akan dipakai.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.12,
        help="Confidence threshold untuk deteksi YOLO.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Ukuran input YOLO (px). Jika tidak di-set, gunakan max dim sampai 2560.",
    )
    return parser.parse_args()


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


def suppress_duplicates(detections, iou_threshold=0.35):
    kept = []
    for det in sorted(detections, key=lambda d: d["conf"], reverse=True):
        if any(iou(det["xyxy"], other["xyxy"]) > iou_threshold for other in kept):
            continue
        kept.append(det)
    return kept


def find_default_dslr_image():
    for path in DEFAULT_IMAGE_PATHS:
        if os.path.exists(path):
            return path
    return None


def load_parking_slots(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if isinstance(raw_data, dict):
        return raw_data.get("slots", [])
    if isinstance(raw_data, list):
        return raw_data

    raise ValueError("Format JSON tidak dikenal. Gunakan array object atau object dengan key 'slots'.")


def create_slot_masks(img_shape, slots):
    h, w = img_shape[:2]
    slot_polygons = []
    slot_masks = []
    slot_areas = []

    for slot in slots:
        polygon = np.array(slot["polygon"], np.int32)
        slot_polygons.append(polygon)

        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [polygon], 1)
        slot_masks.append(mask)
        slot_areas.append(int(np.count_nonzero(mask)))

    return slot_polygons, slot_masks, slot_areas


def point_in_slot(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0


def box_slot_overlap(box, slot_mask):
    x1, y1, x2, y2 = box
    bbox_mask = np.zeros_like(slot_mask)
    cv2.rectangle(bbox_mask, (x1, y1), (x2, y2), 1, -1)
    intersection = cv2.bitwise_and(slot_mask, bbox_mask)
    return int(np.count_nonzero(intersection))


def match_detection_to_slot(box, slot_polygons, slot_masks, slot_areas, overlap_ratio_threshold=MIN_SLOT_OVERLAP_RATIO, coverage_ratio_threshold=MIN_SLOT_COVERAGE_RATIO):
    x1, y1, x2, y2 = box
    cx = int((x1 + x2) / 2)
    cy = y2 - 8

    best_slot_index = None
    best_overlap = 0

    for index, (polygon, mask) in enumerate(zip(slot_polygons, slot_masks)):
        if point_in_slot((cx, cy), polygon):
            return index

        overlap = box_slot_overlap(box, mask)
        if overlap > best_overlap:
            best_overlap = overlap
            best_slot_index = index

    box_area = max((x2 - x1) * (y2 - y1), 1)
    if best_slot_index is None:
        return None

    slot_area = max(slot_areas[best_slot_index], 1)
    box_overlap_ratio = best_overlap / box_area
    slot_coverage_ratio = best_overlap / slot_area

    if box_overlap_ratio >= overlap_ratio_threshold and slot_coverage_ratio >= coverage_ratio_threshold:
        return best_slot_index

    return None


def main() -> None:
    args = parse_args()
    image_path = args.image or find_default_dslr_image()

    if image_path is None:
        print("Gagal menemukan gambar DSLR. Silakan set --image path_ke_gambar.")
        return

    if not os.path.exists(image_path):
        print(f"Gagal menemukan gambar: {image_path}")
        return

    if not os.path.exists(args.json):
        print(f"Gagal menemukan file JSON slot parkir: {args.json}")
        return

    model = load_yolo_model(args.model)

    img = cv2.imread(image_path)
    if img is None:
        print(f"Gagal membaca gambar: {image_path}")
        return

    slots = load_parking_slots(args.json)
    if len(slots) == 0:
        print("Tidak ada slot parkir di JSON.")
        return

    h, w = img.shape[:2]
    imgsz = args.imgsz if args.imgsz is not None else min(max(h, w), 2560)
    imgsz = min(imgsz, 2560)

    slot_polygons, slot_masks, slot_areas = create_slot_masks(img.shape, slots)

    print(f"Gambar DSLR: {image_path}")
    print(f"Resolusi: {w}x{h}, imgsz={imgsz}")
    print(f"Slot parkir: {len(slots)}")

    results = model.predict(
        source=img,
        conf=args.conf,
        imgsz=imgsz,
        augment=False,
        verbose=False,
    )

    detections = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            if label not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            if area < MIN_BOX_AREA:
                continue

            detections.append({
                "label": label,
                "conf": float(box.conf[0]),
                "xyxy": (x1, y1, x2, y2),
            })

    detections = suppress_duplicates(detections)
    occupied_slots = set()

    for det in detections:
        x1, y1, x2, y2 = det["xyxy"]
        label = det["label"]
        conf = det["conf"]
        color = (255, 0, 0)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        draw_label(img, f"{label} {conf:.2f}", x1, y1, color)

        slot_index = match_detection_to_slot(det["xyxy"], slot_polygons, slot_masks, slot_areas)
        if slot_index is not None:
            occupied_slots.add(slots[slot_index]["id"])

        cx = int((x1 + x2) / 2)
        cy = y2 - 8
        cv2.circle(img, (cx, cy), 6, (255, 0, 255), -1)

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

    cv2.imwrite(args.output, img)
    print(f"Hasil DSLR disimpan di {args.output}")


if __name__ == "__main__":
    main()
