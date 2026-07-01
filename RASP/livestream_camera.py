"""
Multi-camera live detection using best.pt model.
Discovers USB cameras, then spawns one independent process per camera.
Each process runs its own GUI window — no threading issues.

Usage:
  python livestream_camera.py            # auto-discover all cameras
  python livestream_camera.py --cam 0    # run single camera directly
  python livestream_camera.py --list     # only list available cameras

Controls (per window):
  q / ESC  — close that camera (all quit via main launcher)
  s        — save screenshot
  c        — toggle center-point marker
  p        — toggle pause
"""

import cv2
import os
import sys
import time
import subprocess
import argparse
from datetime import datetime
from ultralytics import YOLO

# =====================================================
# CONFIG
# =====================================================

MODEL_PATH = "best.pt"
CONF_THRESHOLD = 0.25
IMGSZ = 640
SAVE_DIR = "screenshots"

# Colors: BGR
COLOR_BOX = (0, 255, 0)
COLOR_TEXT = (0, 255, 0)
COLOR_CENTER = (0, 0, 255)
COLOR_INFO_BG = (40, 40, 40)
COLOR_INFO_TEXT = (255, 255, 255)

os.makedirs(SAVE_DIR, exist_ok=True)


# =====================================================
# CAMERA DISCOVERY
# =====================================================

def discover_cameras(max_cams: int = 8) -> list:
    """Probe camera indices 0..max_cams-1, return list of available ones."""
    available = []
    for idx in range(max_cams):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            available.append(idx)
            cap.release()
    return available


# =====================================================
# LOAD MODEL
# =====================================================

def load_model(path: str) -> YOLO:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    print(f"[cam] Loading model: {path}")
    model = YOLO(path)
    print(f"[cam] Model loaded. Classes: {len(model.names)}")
    return model


# =====================================================
# DRAWING HELPERS
# =====================================================

def draw_detections(frame, results, show_center: bool = True):
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
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BOX, 2)
            text = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), COLOR_BOX, -1)
            cv2.putText(frame, text, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_INFO_BG, 2, cv2.LINE_AA)
            if show_center:
                cv2.circle(frame, (cx, cy), 4, COLOR_CENTER, -1)

            detections.append({
                "label": label, "conf": conf,
                "xyxy": (x1, y1, x2, y2), "center": (cx, cy),
            })
    return frame, detections


def draw_info_panel(frame, detections, fps: float, paused: bool,
                    show_center: bool, cam_label: str):
    h = 50 + min(len(detections), 8) * 22 + 40
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
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SAVE_DIR, f"{cam_label}_{ts}.jpg")
    cv2.imwrite(path, frame)
    print(f"[{cam_label}] Screenshot saved: {path}")
    return path


# =====================================================
# SINGLE-CAMERA LOOP (runs in its own process)
# =====================================================

def run_single_camera(camera_index: int):
    """Full capture + inference loop for one camera. Called via subprocess."""
    cam_label = f"cam{camera_index}"
    window_name = f"Camera {camera_index} - Live Detection"

    # Load model
    model = load_model(MODEL_PATH)

    # Open camera
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[{cam_label}] Error: Cannot open camera {camera_index}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[{cam_label}] Camera opened: {actual_w}x{actual_h} @ {actual_fps:.0f}fps")

    paused = False
    show_center = True
    last_result = None

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    print(f"[{cam_label}] Live stream started.")
    print(f"[{cam_label}] Controls: q/ESC=quit  s=save  c=center  p=pause")

    fps = 0.0
    prev_tick = cv2.getTickCount()

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print(f"[{cam_label}] Frame read failed. Retrying...")
                time.sleep(0.1)
                continue

            results = model.predict(
                source=frame, conf=CONF_THRESHOLD, imgsz=IMGSZ, verbose=False,
            )
            last_result = (frame.copy(), results)
            frame, detections = draw_detections(frame, results, show_center)

            tick = cv2.getTickCount()
            elapsed = (tick - prev_tick) / cv2.getTickFrequency()
            if elapsed > 0:
                fps = fps * 0.85 + (1.0 / elapsed) * 0.15
            prev_tick = tick

            draw_info_panel(frame, detections, fps, paused, show_center, cam_label)
            cv2.imshow(window_name, frame)
        else:
            if last_result:
                frame_show, dets_show = draw_detections(
                    last_result[0].copy(), last_result[1], show_center)
                draw_info_panel(frame_show, dets_show, fps, paused, show_center, cam_label)
                cv2.imshow(window_name, frame_show)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            if last_result:
                frame_save, dets_save = draw_detections(
                    last_result[0].copy(), last_result[1], show_center)
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

    cap.release()
    cv2.destroyAllWindows()
    print(f"[{cam_label}] Stopped.")


# =====================================================
# LAUNCHER — discovers cameras, spawns subprocesses
# =====================================================

def launch_multi_camera(camera_indices: list):
    """Spawn one subprocess per camera index. Each runs its own GUI."""
    script = os.path.abspath(__file__)
    python = sys.executable
    processes = []

    for idx in camera_indices:
        print(f"Launching camera {idx}...")
        p = subprocess.Popen(
            [python, script, "--cam", str(idx)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        processes.append((idx, p))

    print(f"\n{len(processes)} camera(s) launched.")
    print("Close all camera windows to exit, or press Ctrl+C here.\n")
    print("=" * 50)

    # Wait for all subprocesses to finish
    try:
        for idx, p in processes:
            p.wait()
    except KeyboardInterrupt:
        print("\nTerminating all cameras...")
        for idx, p in processes:
            if p.poll() is None:
                p.terminate()
        for idx, p in processes:
            p.wait(timeout=5)

    print("All cameras stopped.")


# =====================================================
# ENTRY POINT
# =====================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-camera YOLO live detection")
    parser.add_argument("--cam", type=int, default=None,
                        help="Run single camera by index (used internally by subprocess)")
    parser.add_argument("--list", action="store_true",
                        help="List available cameras and exit")
    parser.add_argument("--indices", type=str, default=None,
                        help="Comma-separated camera indices to use (e.g. '0,1,2')")
    args = parser.parse_args()

    # --- Mode: single camera (called by subprocess) ---
    if args.cam is not None:
        run_single_camera(args.cam)
        return

    # --- Mode: list cameras ---
    available = discover_cameras()
    print(f"Cameras found: {available}")

    if args.list:
        for idx in available:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"  Camera {idx}: {w}x{h}")
                cap.release()
        return

    if not available:
        print("Error: No cameras detected.")
        return

    # --- Determine active cameras ---
    if args.indices:
        requested = [int(x.strip()) for x in args.indices.split(",")]
        active = [i for i in requested if i in available]
        if not active:
            print(f"None of requested indices {requested} are available.")
            return
    else:
        active = available  # use all discovered

    print(f"Active cameras: {active}")

    # --- Launch subprocesses ---
    launch_multi_camera(active)


if __name__ == "__main__":
    main()
