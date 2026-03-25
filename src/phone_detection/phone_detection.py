import cv2
import numpy as np
import time
import os
import platform
from ultralytics import YOLO

# Load YOLO model ONCE
model = YOLO("yolov8n.pt")

# Beep setup
def play_beep():
    system = platform.system()
    if system == "Windows":
        import winsound
        winsound.Beep(2000, 300)
    elif system == "Darwin":
        os.system('afplay /System/Library/Sounds/Ping.aiff &')
    else:
        print('\a')

# Tracking variables (GLOBAL STATE)
phone_in_zone_start_time = None
total_phone_time = 0.0
last_phone_detected = False
last_beep_time = 0  # prevent spam beeping

def draw_detection_zone(frame, width, height):
    zone_points = np.array([[0, 0], [width // 2, 0], [width // 2, height], [0, height]], np.int32)

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

def process_phone(frame):
    global phone_in_zone_start_time, total_phone_time
    global last_phone_detected, last_beep_time

    h, w = frame.shape[:2]

    # Draw detection zone
    frame, zone_points = draw_detection_zone(frame, w, h)

    results = model(frame, verbose=False, conf=0.25)

    phone_detected_in_zone = False
    phone_count = 0

    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])

            if class_name == "cell phone" and confidence > 0.25:
                phone_count += 1

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                if is_in_zone([x1, y1, x2, y2], zone_points):
                    phone_detected_in_zone = True
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    # Time tracking
    current_time = time.time()

    if phone_detected_in_zone:
        if not last_phone_detected:
            phone_in_zone_start_time = current_time
        last_phone_detected = True
    else:
        if last_phone_detected and phone_in_zone_start_time:
            total_phone_time += current_time - phone_in_zone_start_time
            phone_in_zone_start_time = None
        last_phone_detected = False

    # Active time
    active_time = total_phone_time
    if phone_detected_in_zone and phone_in_zone_start_time:
        active_time += current_time - phone_in_zone_start_time

    # Controlled beep (every 2 sec max)
    if phone_detected_in_zone and current_time - last_beep_time > 2:
        play_beep()
        last_beep_time = current_time

    return frame, phone_detected_in_zone, active_time, phone_count