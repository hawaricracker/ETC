"""
Live camera detection using best.pt model.
Reads from USB camera, runs YOLO inference every frame, displays predictions in real-time.

Controls:
  q / ESC  — quit
  s        — save current frame as screenshot
  c        — toggle center-point marker
  p        — toggle pause
"""

import cv2
import os
import time
from datetime import datetime
from ultralytics import YOLO

# =====================================================
# CONFIG
# =====================================================

MODEL_PATH = "best.pt"
CAMERA_INDEX = 0                # 0 = default camera; change if multiple USB cameras
CONF_THRESHOLD = 0.25           # minimum confidence to draw
IMGSZ = 640                     # inference resolution (higher = slower but more accurate)
SAVE_DIR = "screenshots"

# Colors: BGR
COLOR_BOX = (0, 255, 0)         # green bounding box
COLOR_TEXT = (0, 255, 0)
COLOR_CENTER = (0, 0, 255)      # red center dot
COLOR_INFO_BG = (40, 40, 40)
COLOR_INFO_TEXT = (255, 255, 255)

os.makedirs(SAVE_DIR, exist_ok=True)


# =====================================================
# LOAD MODEL
# =====================================================

def load_model(path: str) -> YOLO:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    print(f"Loading model: {path}")
    model = YOLO(path)
    print(f"Model loaded. Classes: {len(model.names)}")
    for i, name in model.names.items():
        print(f"  {i}: {name}")
    return model


# =====================================================
# DRAWING HELPERS
# =====================================================

def draw_detections(frame, results, show_center: bool = True):
    """Draw bounding boxes and labels from YOLO results onto frame."""
    if results is None or results[0].boxes is None:
        return frame, []

    detections = []

    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < CONF_THRESHOLD:
                continue

            cls_id = int(box.cls[0])
            label = result.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BOX, 2)

            # Label with confidence
            text = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), COLOR_BOX, -1)
            cv2.putText(frame, text, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_INFO_BG, 2, cv2.LINE_AA)

            # Center point
            if show_center:
                cv2.circle(frame, (cx, cy), 4, COLOR_CENTER, -1)

            detections.append({
                "label": label,
                "conf": conf,
                "xyxy": (x1, y1, x2, y2),
                "center": (cx, cy),
            })

    return frame, detections


def draw_info_panel(frame, detections, fps: float, paused: bool, show_center: bool):
    """Draw stats overlay panel at top-left."""
    h = 30 + len(detections) * 22 + 40
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (300, h), COLOR_INFO_BG, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    y = 32
    cv2.putText(frame, f"FPS: {fps:.1f}", (16, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_INFO_TEXT, 1, cv2.LINE_AA)
    y += 20

    if paused:
        cv2.putText(frame, "[PAUSED]", (16, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        y += 22

    cv2.putText(frame, f"Detections: {len(detections)}", (16, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_INFO_TEXT, 1, cv2.LINE_AA)
    y += 22

    for d in detections[:8]:  # max 8 items in panel
        cv2.putText(frame, f"  {d['label']} ({d['conf']:.2f})", (16, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_INFO_TEXT, 1, cv2.LINE_AA)
        y += 18

    controls_y = h - 14
    cv2.putText(frame, "q:quit  s:save  c:center  p:pause",
                (16, controls_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)


def save_screenshot(frame, detections):
    """Save current frame as JPEG with timestamp."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SAVE_DIR, f"capture_{ts}.jpg")
    cv2.imwrite(path, frame)
    print(f"Screenshot saved: {path}")
    return path


# =====================================================
# MAIN LOOP
# =====================================================

def main():
    # --- Load model ---
    model = load_model(MODEL_PATH)

    # --- Open camera ---
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        # Try index 1
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("Error: Cannot open USB camera. Check connection and camera index.")
            return

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera opened: {actual_w}x{actual_h} @ {actual_fps:.0f}fps")

    # --- State ---
    paused = False
    show_center = True
    last_result = None
    window_name = "Live Detection - best.pt"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    print("\nLive stream started.")
    print("  q / ESC  — quit")
    print("  s        — save screenshot")
    print("  c        — toggle center point")
    print("  p        — toggle pause")
    print("=" * 50)

    fps = 0.0
    prev_tick = cv2.getTickCount()

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Frame read failed. Retrying...")
                time.sleep(0.1)
                continue

            # Inference
            results = model.predict(
                source=frame,
                conf=CONF_THRESHOLD,
                imgsz=IMGSZ,
                verbose=False,
            )
            last_result = (frame.copy(), results)

            # Draw
            frame, detections = draw_detections(frame, results, show_center)

            # FPS calculation
            tick = cv2.getTickCount()
            elapsed = (tick - prev_tick) / cv2.getTickFrequency()
            if elapsed > 0:
                fps = fps * 0.85 + (1.0 / elapsed) * 0.15  # exponential moving average
            prev_tick = tick

            draw_info_panel(frame, detections, fps, paused, show_center)

            # Display
            cv2.imshow(window_name, frame)
        else:
            # Paused — keep showing last frame
            if last_result:
                frame_show, dets_show = draw_detections(
                    last_result[0].copy(), last_result[1], show_center
                )
                draw_info_panel(frame_show, dets_show, fps, paused, show_center)
                cv2.imshow(window_name, frame_show)

        # --- Key handling ---
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:  # q or ESC
            break
        elif key == ord('s'):
            if last_result:
                frame_save, dets_save = draw_detections(
                    last_result[0].copy(), last_result[1], show_center
                )
                draw_info_panel(frame_save, dets_save, fps, paused, show_center)
                save_screenshot(frame_save, dets_save)
            else:
                print("No frame to save yet.")
        elif key == ord('c'):
            show_center = not show_center
            print(f"Center point: {'ON' if show_center else 'OFF'}")
        elif key == ord('p'):
            paused = not paused
            print(f"Paused: {'ON' if paused else 'OFF'}")

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
