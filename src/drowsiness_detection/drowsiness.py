from scipy.spatial import distance
import cv2
import numpy as np
import time
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import urllib.request
import os
import subprocess
import threading

# ── Helper functions ──────────────────────────────────────────────────────────
def eye_aspect_ratio(eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

def mouth_aspect_ratio(shape):
        top_lip = shape[50:53]
        top_lip = np.concatenate((top_lip, shape[61:64]))
        low_lip = shape[56:59]
        low_lip = np.concatenate((low_lip, shape[65:68]))
        top_mean = np.mean(top_lip, axis=0)
        low_mean = np.mean(low_lip, axis=0)
        return abs(top_mean[1] - low_mean[1])

def head_droop_ratio(shape):
        nose_bridge = shape[27]
        nose_tip    = shape[33]
        chin        = shape[8]
        upper = nose_tip[1] - nose_bridge[1]
        lower = chin[1]     - nose_tip[1]
        if upper == 0:
                return 0
        return lower / upper

# ── Thresholds ────────────────────────────────────────────────────────────────
EAR_THRESH          = 0.25
YAWN_THRESH         = 40    # raised from 20 to reduce false positives
ALERT_GENTLE_SEC    = 1.5
ALERT_STRONG_SEC    = 3
ALERT_LOUD_SEC      = 5
ALERT_AUTH_SEC      = 10
HEAD_DROOP_THRESH   = 1.2

# ── Load models ───────────────────────────────────────────────────────────────
haar_path     = os.path.join(os.path.dirname(cv2.__file__), 'data', 'haarcascade_frontalface_default.xml')
face_detector = cv2.CascadeClassifier(haar_path)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "shape_predictor_68_face_landmarks.dat")

predictor = dlib.shape_predictor(model_path)
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# ── State ─────────────────────────────────────────────────────────────────────
cap             = cv2.VideoCapture(0)
closed_start    = None
sunglasses_mode = False

print("DATS+ Drowsiness Detection running.")
print("Press 'S' to toggle sunglasses mode.")
print("Press 'Q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    # Resize frame for faster processing
    frame = cv2.resize(frame, (450, int(frame.shape[0] * 450 / frame.shape[1])))
    img_h, img_w = frame.shape[:2]

    # Convert to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # Run face landmark detection
    detection = face_landmarker.detect(mp_image)

    # Default UI state each frame
    current_alert = "Driver Alert"
    alert_color   = (0, 200, 0)
    ear = None
    mar = None

    if detection.face_landmarks:
        lm = detection.face_landmarks[0]

        # Calculate metrics
        ear = (eye_aspect_ratio(lm, LEFT_EYE, img_w, img_h) +
               eye_aspect_ratio(lm, RIGHT_EYE, img_w, img_h)) / 2.0
        mar = mouth_aspect_ratio(lm, img_w, img_h)

        # --- YAWN DETECTION ---
        if mar > YAWN_THRESH:
            current_alert = "YAWNING - Stay Alert!"
            alert_color   = (0, 140, 255)

            now = time.time()
            if now > yawn_cooldown:
                beep_then_speak(800, 200, "Yawn detected. Please stay alert.")
                yawn_cooldown = now + 8
                print(f"Yawn detected - MAR: {mar:.1f}")

        # --- EYES CLOSED DETECTION ---
        if ear < EAR_THRESH:
            if closed_start is None:
                closed_start = time.time()
                for k in alerts_fired:
                    alerts_fired[k] = False

            elapsed = time.time() - closed_start

            # Update UI state based on severity
            if elapsed >= ALERT_GENTLE_SEC:
                current_alert = "WARNING: Eyes Closing"
                alert_color   = (0, 140, 255)

            if elapsed >= ALERT_STRONG_SEC:
                current_alert = "WARNING: DROWSY"
                alert_color   = (0, 0, 220)

            if elapsed >= ALERT_LOUD_SEC:
                current_alert = "DANGER: VERY DROWSY"
                alert_color   = (0, 0, 200)

            if elapsed >= ALERT_AUTH_SEC:
                current_alert = "CRITICAL: ALERTING FLEET"
                alert_color   = (0, 0, 180)

            # Fire audio/speech alerts (unchanged)
            if elapsed >= ALERT_GENTLE_SEC and not alerts_fired["gentle"]:
                beep_then_speak(1000, 400, "Stay alert.")
                alerts_fired["gentle"] = True

            if elapsed >= ALERT_STRONG_SEC and not alerts_fired["strong"]:
                beep_then_speak(1200, 600, "Warning. Wake up!")
                alerts_fired["strong"] = True

            if elapsed >= ALERT_LOUD_SEC and not alerts_fired["loud"]:
                beep_then_speak(1500, 1000, "Danger! Wake up now!")
                alerts_fired["loud"] = True

            if elapsed >= ALERT_AUTH_SEC and not alerts_fired["auth"]:
                beep_then_speak(1800, 1500, "Critical alert. Alerting fleet management.")
                print(f"CRITICAL: Driver drowsy for {elapsed:.0f}s - alerting authorities")
                alerts_fired["auth"] = True

        else:
            # Eyes open — reset timer
            closed_start = None

    else:
        # No face detected
        current_alert = "FACE NOT DETECTED"
        alert_color   = (0, 0, 200)

        now = time.time()
        if now > no_face_cooldown:
            beep_then_speak(1000, 300, "Eyes not visible. Please remove obstruction.")
            no_face_cooldown = now + 6
            print("WARNING: Eyes not visible")

    # Render UI overlay
    frame = draw_ui(frame, current_alert, alert_color, ear, mar)

    # Show window
    cv2.imshow("DATS+ Drowsiness Detection", frame)

    # Exit on Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cv2.destroyAllWindows()
cap.release()
face_landmarker.close()
ps_process.stdin.close()
ps_process.terminate()