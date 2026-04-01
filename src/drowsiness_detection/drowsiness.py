# This file contains the main code for the Drowsiness sub-system of DATS+
# This code also contains an individual GUI in order to test just this unit alone later.
# Type "python src/drowsiness_detection/drowsiness.py" on the terminal to run it individually
# Code provided by: Minal Haque
# Additional credits: Code dependancy fixes by Fatima Faisal

#imports
from scipy.spatial import distance
import cv2
import numpy as np
import time
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import urllib.request
import subprocess
import threading
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import stats, announce_violation

# Detect platform for cross-platform audio/speech support
import platform
_PLATFORM = platform.system()  # 'Windows', 'Darwin' (macOS), or 'Linux'

# Persistent PowerShell process — Windows only
if _PLATFORM == "Windows":
    ps_process = subprocess.Popen(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", "-"],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    # Load speech engine once to avoid re-initializing it every alert
    ps_process.stdin.write(b"Add-Type -AssemblyName System.Speech\n")
    ps_process.stdin.flush()
    _tts = "(New-Object System.Speech.Synthesis.SpeechSynthesizer)"
else:
    ps_process = None

# Runs beep + speech in a separate thread so it doesn't block the video loop
def beep_then_speak(freq, duration_ms, text):
    def _run():
        if _PLATFORM == "Windows":
            cmd = f"[console]::beep({freq},{duration_ms}); {_tts}.Speak('{text}')\n"
            ps_process.stdin.write(cmd.encode())
            ps_process.stdin.flush()
        elif _PLATFORM == "Darwin":
            # macOS: use built-in 'say' for TTS
            subprocess.run(["say", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            # Linux: use espeak if available
            subprocess.run(["espeak", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    threading.Thread(target=_run, daemon=True).start()

# Model file path for MediaPipe face landmark detection
MODEL_PATH = "face_landmarker.task"

# Download model if it doesn't exist locally
if not os.path.exists(MODEL_PATH):
    print("Downloading face landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        MODEL_PATH
    )
    print("Model downloaded.")

# Configure MediaPipe face landmarker
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
face_landmarker = vision.FaceLandmarker.create_from_options(options)

# Landmark indices for eyes and mouth based on MediaPipe face mesh
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH_TOP = 13
MOUTH_BOT = 14

# Computes Eye Aspect Ratio (EAR) - Used to detect if eyes are closed (low EAR means closed eyes)
def eye_aspect_ratio(lm, eye_indices, img_w, img_h):
    pts = np.array([[lm[i].x * img_w, lm[i].y * img_h] for i in eye_indices])
    A = distance.euclidean(pts[1], pts[5])
    B = distance.euclidean(pts[2], pts[4])
    C = distance.euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C)

# Computes mouth opening distance
def mouth_aspect_ratio(lm, img_w, img_h):
    top = np.array([lm[MOUTH_TOP].x * img_w, lm[MOUTH_TOP].y * img_h])
    bot = np.array([lm[MOUTH_BOT].x * img_w, lm[MOUTH_BOT].y * img_h])
    return abs(top[1] - bot[1])

# Threshold values for detection
EAR_THRESH = 0.25  # eyes considered closed below this value
YAWN_THRESH = 15   # mouth distance considered a yawn

# Time thresholds for triggering different alert levels
ALERT_GENTLE_SEC = 1.5
ALERT_STRONG_SEC = 3
ALERT_LOUD_SEC   = 5
ALERT_AUTH_SEC   = 10


# Landmark drawings
def draw_landmarks(frame, lm, img_w, img_h):
    """Draw eye outlines and mouth markers on the frame."""
    original_h, original_w = frame.shape[:2]  # save original size

    frame = cv2.resize(frame, (450, int(frame.shape[0] * 450 / frame.shape[1])))
    # Left eye — cyan outline + dots
    pts_l = np.array([[int(lm[i].x * img_w), int(lm[i].y * img_h)] for i in LEFT_EYE], np.int32)
    cv2.polylines(frame, [pts_l], isClosed=True, color=(255, 255, 0), thickness=1)
    for pt in pts_l:
        cv2.circle(frame, tuple(pt), 2, (255, 255, 0), -1)

    # Right eye — cyan outline + dots
    pts_r = np.array([[int(lm[i].x * img_w), int(lm[i].y * img_h)] for i in RIGHT_EYE], np.int32)
    cv2.polylines(frame, [pts_r], isClosed=True, color=(255, 255, 0), thickness=1)
    for pt in pts_r:
        cv2.circle(frame, tuple(pt), 2, (255, 255, 0), -1)

    # Mouth top/bottom — magenta dots + connecting line
    mt = (int(lm[MOUTH_TOP].x * img_w), int(lm[MOUTH_TOP].y * img_h))
    mb = (int(lm[MOUTH_BOT].x * img_w), int(lm[MOUTH_BOT].y * img_h))
    cv2.circle(frame, mt, 3, (255, 0, 255), -1)
    cv2.circle(frame, mb, 3, (255, 0, 255), -1)
    cv2.line(frame, mt, mb, (255, 0, 255), 1)

    frame = cv2.resize(frame, (original_w, original_h))  # resize back
    return frame

# UI Helper function
def draw_ui(frame, alert_text, color, ear=None, mar=None):
    h, w = frame.shape[:2]

    # Thin semi-transparent top bar — title only
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 24), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, "DRIVER MONITORING SYSTEM",
                (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

    # Compact semi-transparent status bar — pinned to bottom
    bar_h = 30
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - bar_h), (w, h), color, -1)
    cv2.addWeighted(overlay2, 0.75, frame, 0.25, 0, frame)
    cv2.putText(frame, alert_text,
                (8, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)

    # EAR / MAR inline on right side of status bar
    if ear is not None and mar is not None:
        metrics = f"EAR:{ear:.2f}  MAR:{mar:.1f}"
        cv2.putText(frame, metrics,
                    (w - 150, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Thin colored border
    cv2.rectangle(frame, (0, 0), (w, h), color, 2)

    return frame

# Shared state for when called externally via process_drowsiness_frame()
_ext_closed_start     = None
_ext_no_face_cooldown = 0
_ext_yawn_cooldown    = 0
_ext_alerts_fired     = {"gentle": False, "strong": False, "loud": False, "auth": False}

# CALLABLE FUNCTION FOR main.py
def process_drowsiness_frame(frame, alerts_flags=None):
    """
    Process a single frame for drowsiness detection.
    Intended to be called from main.py each frame.

    Args:
        frame:        BGR frame from OpenCV capture.
        alerts_flags: Optional external dict to override internal alert state.
                      Must contain keys: 'gentle', 'strong', 'loud', 'auth'.
                      If None, internal shared state is used.

    Returns:
        annotated_frame: Frame with UI overlay drawn on it.
        status: dict with keys:
            - 'alert_text'  (str)
            - 'alert_color' (BGR tuple)
            - 'ear'         (float or None)
            - 'mar'         (float or None)
            - 'drowsy'      (bool)
            - 'yawning'     (bool)
            - 'face_found'  (bool)
    """
    global _ext_closed_start, _ext_no_face_cooldown, _ext_yawn_cooldown, _ext_alerts_fired

    # Use external alerts_flags if provided, otherwise use internal state
    alerts = alerts_flags if alerts_flags is not None else _ext_alerts_fired

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
    drowsy     = False
    yawning    = False
    face_found = bool(detection.face_landmarks)

    if detection.face_landmarks:
        lm = detection.face_landmarks[0]

        # Draw eye and mouth landmarks on frame
        frame = draw_landmarks(frame, lm, img_w, img_h)

        # Calculate metrics
        ear = (eye_aspect_ratio(lm, LEFT_EYE, img_w, img_h) +
               eye_aspect_ratio(lm, RIGHT_EYE, img_w, img_h)) / 2.0
        mar = mouth_aspect_ratio(lm, img_w, img_h)

        # yawn detection
        if mar > YAWN_THRESH:
            yawning       = True
            current_alert = "YAWNING - Stay Alert!"
            alert_color   = (0, 140, 255)

            now = time.time()
            if now > _ext_yawn_cooldown:
                beep_then_speak(800, 200, "Yawn detected. Please stay alert.")
                _ext_yawn_cooldown = now + 8
                print(f"Yawn detected - MAR: {mar:.1f}")

        # eyes closed detection
        if ear < EAR_THRESH:
            drowsy = True
            if _ext_closed_start is None:
                _ext_closed_start = time.time()
                for k in alerts:
                    alerts[k] = False

            elapsed = time.time() - _ext_closed_start

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

            # Fire audio/speech alerts
            if elapsed >= ALERT_GENTLE_SEC and not alerts["gentle"]:
                beep_then_speak(1000, 400, "Stay alert.")
                alerts["gentle"] = True

            if elapsed >= ALERT_STRONG_SEC and not alerts["strong"]:
                beep_then_speak(1200, 600, "Warning. Wake up!")
                alerts["strong"] = True

            if elapsed >= ALERT_LOUD_SEC and not alerts["loud"]:
                beep_then_speak(1500, 1000, "Danger! Wake up now!")
                alerts["loud"] = True

            if elapsed >= ALERT_AUTH_SEC and not alerts["auth"]:
                beep_then_speak(1800, 1500, "Critical alert. Alerting fleet management.")
                print(f"CRITICAL: Driver drowsy for {elapsed:.0f}s - alerting authorities")
                alerts["auth"] = True

        else:
            # Eyes open — reset timer
            _ext_closed_start = None

    else:
        # No face detected
        current_alert = "FACE NOT DETECTED"
        alert_color   = (0, 0, 200)

        now = time.time()
        if now > _ext_no_face_cooldown:
            beep_then_speak(1000, 300, "Eyes not visible. Please remove obstruction.")
            _ext_no_face_cooldown = now + 6
            print("WARNING: Eyes not visible")

    # Render UI overlay
    frame = draw_ui(frame, current_alert, alert_color, ear, mar)

    status = {
        "alert_text":  current_alert,
        "alert_color": alert_color,
        "ear":         ear,
        "mar":         mar,
        "drowsy":      drowsy,
        "yawning":     yawning,
        "face_found":  face_found,
    }

    return frame, status


# MAIN LOOP SETUP
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    print("DATS+ Drowsiness Detection running (MediaPipe).")
    print("Press 'Q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        frame, status = process_drowsiness_frame(frame)
        cv2.imshow("DATS+ Drowsiness Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    cap.release()
    face_landmarker.close()
    if ps_process:
        ps_process.stdin.close()
        ps_process.terminate()