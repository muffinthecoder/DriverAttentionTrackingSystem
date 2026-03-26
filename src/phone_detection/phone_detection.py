# This file contains the main code for the Phone detection sub-system of DATS+
# This code also contains a simple GUI in order to test just this unit alone later.
# Code provided by: Fatima Faisal


# ── Imports ──────────────────────────────────────────────────────────────────
import cv2
import argparse
import os
import platform
import time
import numpy as np
from ultralytics import YOLO
import sys

# Add parent folder (src/) to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import stats, announce_violation

# ── Audio alert ───────────────────────────────────────────────────────────────
frequency = 2000
duration = 1500


def play_beep():
    system = platform.system()
    if system == "Windows":
        import winsound
        winsound.Beep(frequency, duration)
    elif system == "Darwin":
        os.system('afplay /System/Library/Sounds/Ping.aiff &')
    else:
        print('\a')


# ── Detection zone helpers ────────────────────────────────────────────────────
def draw_detection_zone(frame, width, height):
    zone_points = np.array(
        [[0, 0], [width // 2, 0], [width // 2, height], [0, height]], np.int32
    )
    overlay = frame.copy()
    cv2.fillPoly(overlay, [zone_points], (0, 0, 255))
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
    cv2.polylines(frame, [zone_points], True, (0, 0, 255), 2)
    return frame, zone_points


def is_in_zone(bbox, zone_points):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return cv2.pointPolygonTest(zone_points, (cx, cy), False) >= 0


# ── Lazy-load the YOLO model once ────────────────────────────────────────────
_yolo_model = None


def _get_model():
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO("yolov8n.pt")
    return _yolo_model


# ── Per-frame state ───────────────────────────────────────────────────────────
_phone_start = None  # timestamp when phone first appeared in zone


# ── Integration entry-point ──────────────────────────────────────────────────
def process_phone_frame(frame, alerts_flags=None):
    """
    Run YOLOv8 phone detection on *frame*.
    Annotates the frame and updates stats["phone_time"].
    Returns the annotated frame.
    """
    global _phone_start

    model = _get_model()
    h, w = frame.shape[:2]
    frame, zone_points = draw_detection_zone(frame, w, h)

    results = model(frame, verbose=False, conf=0.25, iou=0.5)

    phone_in_zone = False

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            confidence = float(box.conf[0])

            if class_name == "cell phone" and confidence > 0.25:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                in_zone = is_in_zone([x1, y1, x2, y2], zone_points)
                color = (0, 0, 255) if in_zone else (0, 255, 0)

                if in_zone:
                    phone_in_zone = True

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f"phone {confidence:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            elif class_name == "person" and confidence > 0.5:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 165, 0), 2)
                cv2.putText(frame, f"person {confidence:.2f}", (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

    # ── Time tracking ─────────────────────────────────────────────────────────
    now = time.time()
    if phone_in_zone:
        if _phone_start is None:
            _phone_start = now
        # accumulate elapsed since last frame
        stats["phone_time"] += now - _phone_start
        _phone_start = now

        play_beep()
        cv2.putText(frame, "PHONE DETECTED IN ZONE!", (10, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        if alerts_flags is not None and not alerts_flags.get("phone"):
            announce_violation("Phone detected while driving!")
            alerts_flags["phone"] = True
    else:
        _phone_start = None

    # HUD
    cv2.putText(frame, f"Phone Time: {stats['phone_time']:.1f}s", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return frame


# ── Stand-alone test entry-point ──────────────────────────────────────────────
def parse_arguments():
    parser = argparse.ArgumentParser(description="YOLOv8 live phone detection")
    parser.add_argument("--webcam-resolution", default=[1280, 720], nargs=2, type=int)
    return parser.parse_args()


def main():
    args = parse_arguments()
    fw, fh = args.webcam_resolution
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, fw)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, fh)

    print("Phone detection started. Press ESC to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = process_phone_frame(frame)
        cv2.imshow("Phone Detection – ESC to quit", frame)
        if cv2.waitKey(30) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. Total phone time: {stats['phone_time']:.1f}s")

#GUI
if __name__ == "__main__":
    main()
