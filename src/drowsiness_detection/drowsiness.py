# This file contains the main code for the Drowsiness detection sub-system of DATS+
# This code also contains a simple GUI in order to test just this unit alone later.
# Code written by: Minal Haque

# ── Imports ───────────────────────────────────────────────────────────────────
import cv2
import os
import platform
import time
import numpy as np
from scipy.spatial import distance
import mediapipe as mp
import sys
import os

# Add parent folder (src/) to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import stats, announce_violation

# ── Thresholds ────────────────────────────────────────────────────────────────
EAR_THRESH = 0.25
MAR_THRESH = 0.6
ALERT_GENTLE_SEC = 1.5
ALERT_STRONG_SEC = 3
ALERT_LOUD_SEC = 5
ALERT_AUTH_SEC = 10

# ── MediaPipe landmark indices ────────────────────────────────────────────────
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 146, 91, 181, 84, 17, 314, 405]


# ── Helpers ───────────────────────────────────────────────────────────────────
def eye_aspect_ratio(eye_points):
    A = distance.euclidean(eye_points[1], eye_points[5])
    B = distance.euclidean(eye_points[2], eye_points[4])
    C = distance.euclidean(eye_points[0], eye_points[3])
    return (A + B) / (2.0 * C)


def mouth_aspect_ratio(mouth_points):
    A = distance.euclidean(mouth_points[2], mouth_points[6])
    B = distance.euclidean(mouth_points[3], mouth_points[7])
    C = distance.euclidean(mouth_points[0], mouth_points[4])
    return (A + B) / (2.0 * C)


def beep_and_speak(message):
    system = platform.system()
    if system == "Darwin":
        os.system('afplay /System/Library/Sounds/Ping.aiff &')
        os.system(f'say "{message}" &')
    else:
        print(f"ALERT: {message}")


# ── Lazy-load MediaPipe once ──────────────────────────────────────────────────
_face_mesh = None


def _get_face_mesh():
    global _face_mesh
    if _face_mesh is None:
        _face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    return _face_mesh


# ── Per-frame state ───────────────────────────────────────────────────────────
_closed_start = None
_alerts_fired = {"gentle": False, "strong": False, "loud": False, "auth": False}


# ── Integration entry-point ───────────────────────────────────────────────────
def process_drowsiness_frame(frame, alerts_flags=None):
    """
    Run drowsiness / yawn detection on *frame*.
    Annotates the frame and updates stats["drowsy_time"].
    Returns the annotated frame.
    """
    global _closed_start, _alerts_fired

    face_mesh = _get_face_mesh()
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    current_alert = "Driver Alert"
    alert_color = (0, 200, 0)
    ear = mar = None

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark

        left_eye = np.array([[int(lm[i].x * w), int(lm[i].y * h)] for i in LEFT_EYE])
        right_eye = np.array([[int(lm[i].x * w), int(lm[i].y * h)] for i in RIGHT_EYE])
        mouth_pts = np.array([[int(lm[i].x * w), int(lm[i].y * h)] for i in MOUTH])

        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
        mar = mouth_aspect_ratio(mouth_pts)

        # Draw contours
        cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
        cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)
        cv2.polylines(frame, [mouth_pts], True, (0, 255, 0), 1)

        # ── Yawn ──────────────────────────────────────────────────────────────
        if mar > MAR_THRESH:
            current_alert = "YAWNING – Stay Alert!"
            alert_color = (0, 140, 255)
            beep_and_speak("Yawn detected")

        # ── Eyes closed ───────────────────────────────────────────────────────
        if ear < EAR_THRESH:
            now = time.time()
            if _closed_start is None:
                _closed_start = now
                _alerts_fired = {k: False for k in _alerts_fired}

            elapsed = now - _closed_start
            stats["drowsy_time"] += now - (_closed_start if elapsed == 0 else now - 0.03)

            if elapsed >= ALERT_AUTH_SEC:
                current_alert = "CRITICAL: ALERTING FLEET"
                alert_color = (0, 0, 180)
                if not _alerts_fired["auth"]:
                    beep_and_speak("Critical alert")
                    if alerts_flags is not None:
                        announce_violation("Driver critically drowsy!")
                    _alerts_fired["auth"] = True

            elif elapsed >= ALERT_LOUD_SEC:
                current_alert = "DANGER: VERY DROWSY"
                alert_color = (0, 0, 200)
                if not _alerts_fired["loud"]:
                    beep_and_speak("Danger! Wake up!")
                    _alerts_fired["loud"] = True

            elif elapsed >= ALERT_STRONG_SEC:
                current_alert = "WARNING: DROWSY"
                alert_color = (0, 0, 220)
                if not _alerts_fired["strong"]:
                    beep_and_speak("Warning. Wake up!")
                    if alerts_flags is not None and not alerts_flags.get("drowsy"):
                        announce_violation("Driver is drowsy!")
                        alerts_flags["drowsy"] = True
                    _alerts_fired["strong"] = True

            elif elapsed >= ALERT_GENTLE_SEC:
                current_alert = "WARNING: Eyes Closing"
                alert_color = (0, 140, 255)
                if not _alerts_fired["gentle"]:
                    beep_and_speak("Stay alert")
                    _alerts_fired["gentle"] = True
        else:
            _closed_start = None

    else:
        current_alert = "FACE NOT DETECTED"
        alert_color = (0, 0, 200)

    # ── HUD ───────────────────────────────────────────────────────────────────
    cv2.putText(frame, current_alert, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, alert_color, 2)
    if ear is not None:
        cv2.putText(frame, f"EAR: {ear:.2f}  MAR: {mar:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame


# ── Stand-alone test entry-point ──────────────────────────────────────────────
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    print("Drowsiness Detection running. Press 'Q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (640, 480))
        frame = process_drowsiness_frame(frame)
        cv2.imshow("Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()