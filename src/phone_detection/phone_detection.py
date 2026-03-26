# This file contains the main code for the Phone detection sub-system of DATS+
# This code also contains a simple GUI in order to test just this unit alone later.
# Code provided by: Fatima Faisal


# Imports
import cv2
import argparse
import os
import platform
from ultralytics import YOLO
import numpy as np
import time

# Beep Sound Setup
frequency = 2000
duration = 1500

def play_beep():
    """Cross-platform beep function"""
    system = platform.system()
    if system == "Windows":
        import winsound
        winsound.Beep(frequency, duration)
    elif system == "Darwin":  # macOS
        os.system('afplay /System/Library/Sounds/Ping.aiff &')
    else:
        print('\a')

# Argument Parsing
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live phone detection")
    parser.add_argument(
        "--webcam-resolution",
        default=[1280, 720],
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args

# Detection Zone
def draw_detection_zone(frame, width, height):
    """Draw the detection zone on the frame"""
    zone_points = np.array([[0, 0], [width // 2, 0], [width // 2, height], [0, height]], np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [zone_points], (0, 0, 255))
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
    cv2.polylines(frame, [zone_points], True, (0, 0, 255), 2)
    return frame, zone_points

def is_in_zone(bbox, zone_points):
    """Check if bounding box center is inside the zone"""
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return cv2.pointPolygonTest(zone_points, (center_x, center_y), False) >= 0

# Main Loop
def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    # Load latest YOLOv8 model (nano)
    model = YOLO("yolov8n.pt")  # Download automatically if not present

    # Time tracking variables
    phone_in_zone_start_time = None
    total_phone_time = 0.0
    last_phone_detected = False

    print("Phone detection started. Press ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        # Draw detection zone
        frame, zone_points = draw_detection_zone(frame, w, h)

        # YOLOv8 Detection
        results = model(frame, verbose=False, conf=0.25, iou=0.5)  # Lowered confidence threshold for better detection, added IOU for NMS

        phone_detected_in_zone = False
        phone_count = 0
        person_count = 0

        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])

                # Detect cell phones with lower confidence threshold
                if class_name == "cell phone" and confidence > 0.25:
                    phone_count += 1
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                    if is_in_zone([x1, y1, x2, y2], zone_points):
                        phone_detected_in_zone = True
                        color = (0, 0, 255)  # Red
                    else:
                        color = (0, 255, 0)  # Green

                    # Draw bounding box + label
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    label = f"{class_name} {confidence:.2f}"
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (int(x1), int(y1) - label_h - 10),
                                  (int(x1) + label_w, int(y1)), color, -1)
                    cv2.putText(frame, label, (int(x1), int(y1) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Detect people
                if class_name == "person" and confidence > 0.5:
                    person_count += 1
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    color = (255, 165, 0)  # Orange

                    # Draw bounding box + label
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    label = f"{class_name} {confidence:.2f}"
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (int(x1), int(y1) - label_h - 10),
                                  (int(x1) + label_w, int(y1)), color, -1)
                    cv2.putText(frame, label, (int(x1), int(y1) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Track time phone is in zone
        current_time = time.time()
        if phone_detected_in_zone:
            if not last_phone_detected:
                phone_in_zone_start_time = current_time
            last_phone_detected = True
        else:
            if last_phone_detected and phone_in_zone_start_time is not None:
                total_phone_time += current_time - phone_in_zone_start_time
                phone_in_zone_start_time = None
            last_phone_detected = False

        # Calculate active phone time (if currently in zone)
        active_phone_time = total_phone_time
        if phone_detected_in_zone and phone_in_zone_start_time is not None:
            active_phone_time += current_time - phone_in_zone_start_time

        # Display counters in top left corner
        counter_y = 30
        cv2.putText(frame, f"Phones: {phone_count}", (10, counter_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        counter_y += 30
        cv2.putText(frame, f"People: {person_count}", (10, counter_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        counter_y += 30
        cv2.putText(frame, f"Phone Time: {active_phone_time:.1f}s", (10, counter_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Play beep if phone detected in zone
        if phone_detected_in_zone:
            play_beep()
            cv2.putText(frame, "PHONE DETECTED IN ZONE!", (10, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Display frame
        cv2.imshow("Phone Detection - Press ESC to quit", frame)

        if cv2.waitKey(30) == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Phone detection stopped. Total phone time in zone: {total_phone_time:.1f} seconds")

#GUI launch
if __name__ == "__main__":
    main()
