"""
Multi-camera live detection using best.pt model.
Reads from multiple USB cameras, runs YOLO inference on each frame, displays in separate windows.

Controls (per window):
  q / ESC  — quit all cameras
  s        — save current frame as screenshot (from active window)
  c        — toggle center-point marker (per camera)
  p        — toggle pause (per camera)
"""

import cv2
import os
import time
import threading
from datetime import datetime
from ultralytics import YOLO

# =====================================================
# CONFIG
# =====================================================

MODEL_PATH = "best.pt"
CAMERA_INDICES = [0, 1]            # USB camera indices to use; set more for more cameras
CONF_THRESHOLD = 0.25              # minimum confidence to draw
IMGSZ = 640                        # inference resolution
SAVE_DIR = "screenshots"

# Colors: BGR
COLOR_BOX = (0, 255, 0)            # green bounding box
COLOR_TEXT = (0, 255, 0)
COLOR_CENTER = (0, 0, 255)         # red center dot
COLOR_INFO_BG = (40, 40, 40)
COLOR_INFO_TEXT = (255, 255, 255)

os.makedirs(SAVE_DIR, exist_ok=True)

# Global stop signal — any thread can set it, all threads stop
STOP_FLAG = threading.Event()


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
# CAMERA DISCOVERY
# =====================================================

def discover_cameras(max_cams: int = 8) -> list:
    """Probe camera indices and return list of available ones."""
    available = []
    for idx in range(max_cams):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            available.append(idx)
            cap.release()
    return available


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


def draw_info_panel(frame, detections, fps: float, paused: bool,
                    show_center: bool, cam_label: str):
    """Draw stats overlay panel at top-left with camera label."""
    rows = min(len(detections), 8)
    h = 50 + rows * 22 + 40
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (300, h), COLOR_INFO_BG, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    y = 32
    cv2.putText(frame, f"[{cam_label}]  FPS: {fps:.1f}", (16, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_INFO_TEXT, 1, cv2.LINE_AA)
    y += 20

    if paused:
        cv2.putText(frame, "[PAUSED]", (16, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        y += 22

    cv2.putText(frame, f"Detections: {len(detections)}", (16, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_INFO_TEXT, 1, cv2.LINE_AA)
    y += 22

    for d in detections[:8]:
        cv2.putText(frame, f"  {d['label']} ({d['conf']:.2f})", (16, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_INFO_TEXT, 1, cv2.LINE_AA)
        y += 18

    controls_y = h - 14
    cv2.putText(frame, "q:quit  s:save  c:center  p:pause",
                (16, controls_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)


def save_screenshot(frame, detections, cam_label: str):
    """Save current frame as JPEG with timestamp and camera label."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SAVE_DIR, f"{cam_label}_{ts}.jpg")
    cv2.imwrite(path, frame)
    print(f"[{cam_label}] Screenshot saved: {path}")
    return path


# =====================================================
# SINGLE-CAMERA WORKER (one thread per camera)
# =====================================================

def camera_worker(model: YOLO, camera_index: int):
    """Capture + inference loop for one USB camera. Runs in its own thread."""
    cam_label = f"cam{camera_index}"
    window_name = f"Camera {camera_index} - Live Detection"

    # --- Open camera ---
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[{cam_label}] Error: Cannot open camera {camera_index}.")
        return

    # Set camera properties (best-effort; actual values depend on hardware)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[{cam_label}] Camera opened: {actual_w}x{actual_h} @ {actual_fps:.0f}fps")

    # --- Per-camera state ---
    paused = False
    show_center = True
    last_result = None

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    fps = 0.0
    prev_tick = cv2.getTickCount()

    while not STOP_FLAG.is_set():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print(f"[{cam_label}] Frame read failed. Retrying...")
                time.sleep(0.1)
                continue

            # Inference — model.predict is read-only, safe to share across threads
            results = model.predict(
                source=frame,
                conf=CONF_THRESHOLD,
                imgsz=IMGSZ,
                verbose=False,
            )
            last_result = (frame.copy(), results)

            # Draw detections
            frame, detections = draw_detections(frame, results, show_center)

            # FPS (exponential moving average)
            tick = cv2.getTickCount()
            elapsed = (tick - prev_tick) / cv2.getTickFrequency()
            if elapsed > 0:
                fps = fps * 0.85 + (1.0 / elapsed) * 0.15
            prev_tick = tick

            draw_info_panel(frame, detections, fps, paused, show_center, cam_label)
            cv2.imshow(window_name, frame)
        else:
            # Paused — keep showing last frame
            if last_result:
                frame_show, dets_show = draw_detections(
                    last_result[0].copy(), last_result[1], show_center
                )
                draw_info_panel(frame_show, dets_show, fps, paused, show_center, cam_label)
                cv2.imshow(window_name, frame_show)

        # --- Key handling ---
        # cv2.waitKey needed in each thread for window responsiveness
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:        # q or ESC — quit all
            STOP_FLAG.set()
            break
        elif key == ord('s'):
            if last_result:
                frame_save, dets_save = draw_detections(
                    last_result[0].copy(), last_result[1], show_center
                )
                draw_info_panel(frame_save, dets_save, fps, paused, show_center, cam_label)
                save_screenshot(frame_save, dets_save, cam_label)
            else:
                print(f"[{cam_label}] No frame to save yet.")
        elif key == ord('c'):
            show_center = not show_center
            print(f"[{cam_label}] Center point: {'ON' if show_center else 'OFF'}")
        elif key == ord('p'):
            paused = not paused
            print(f"[{cam_label}] Paused: {'ON' if paused else 'OFF'}")

    # --- Cleanup ---
    cap.release()
    cv2.destroyWindow(window_name)
    print(f"[{cam_label}] Stopped.")


# =====================================================
# MAIN
# =====================================================

def main():
    # --- Discover available cameras ---
    available = discover_cameras()
    print(f"Cameras found: {available}")

    if not available:
        print("Error: No USB cameras detected.")
        return

    # Use configured indices that actually exist
    active_indices = [i for i in CAMERA_INDICES if i in available]
    if not active_indices:
        print(f"Warning: CAMERA_INDICES {CAMERA_INDICES} not available. "
              f"Falling back to all discovered: {available}")
        active_indices = available

    print(f"Active cameras: {active_indices}")

    # --- Load model once (shared across all camera threads) ---
    model = load_model(MODEL_PATH)

    # --- Start one thread per camera ---
    threads = []
    for cam_idx in active_indices:
        t = threading.Thread(
            target=camera_worker,
            args=(model, cam_idx),
            name=f"cam{cam_idx}",
            daemon=True,
        )
        t.start()
        threads.append(t)

    print(f"\n{len(active_indices)} camera(s) streaming.")
    print("  q / ESC  — quit all cameras")
    print("  s        — save screenshot (active window)")
    print("  c        — toggle center point")
    print("  p        — toggle pause (per camera)")
    print("=" * 50)

    # --- Wait for all threads to finish ---
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("\nInterrupted. Stopping all cameras...")
        STOP_FLAG.set()

    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
