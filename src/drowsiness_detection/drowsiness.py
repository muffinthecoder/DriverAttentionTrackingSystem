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

# Persistent PowerShell process used for playing sound + speech without reopening each time
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

# Runs beep + speech in a separate thread so it doesn't block the video loop
def beep_then_speak(freq, duration_ms, text):
    def _run():
        cmd = f"[console]::beep({freq},{duration_ms}); {_tts}.Speak('{text}')\n"
        ps_process.stdin.write(cmd.encode())
        ps_process.stdin.flush()
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

# Computes Eye Aspect Ratio (EAR)
# Used to detect if eyes are closed (low EAR means closed eyes)
def eye_aspect_ratio(lm, eye_indices, img_w, img_h):
    pts = np.array([[lm[i].x * img_w, lm[i].y * img_h] for i in eye_indices])
    A = distance.euclidean(pts[1], pts[5])
    B = distance.euclidean(pts[2], pts[4])
    C = distance.euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C)

# Computes mouth opening distance
# Used to detect yawning based on vertical mouth gap
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

# =========================
# UI HELPER FUNCTION
# =========================
def draw_ui(frame, alert_text, color, ear=None, mar=None):
    h, w = frame.shape[:2]

    # Top bar background
    cv2.rectangle(frame, (0, 0), (w, 60), (30, 30, 30), -1)
    cv2.putText(frame, "DRIVER MONITORING SYSTEM",
                (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2)

    # Status/alert panel
    cv2.rectangle(frame, (10, 70), (w - 10, 145), color, -1)
    cv2.putText(frame, alert_text,
                (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

    # Metrics box
    if ear is not None and mar is not None:
        cv2.rectangle(frame, (10, 155), (220, 215), (50, 50, 50), -1)
        cv2.putText(frame, f"EAR: {ear:.2f}",
                    (20, 183), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"MAR: {mar:.1f}",
                    (20, 208), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Quit hint
    cv2.putText(frame, "Q: quit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    # Colored border glow
    cv2.rectangle(frame, (0, 0), (w, h), color, 4)

    return frame

# =========================
# MAIN LOOP SETUP
# =========================
cap = cv2.VideoCapture(0)

closed_start = None
no_face_cooldown = 0
yawn_cooldown = 0

alerts_fired = {
    "gentle": False,
    "strong": False,
    "loud":   False,
    "auth":   False,
}

print("DATS+ Drowsiness Detection running (MediaPipe).")
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