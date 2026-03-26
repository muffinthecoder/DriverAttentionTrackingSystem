# This file contains the main code for the Drowsiness detection sub-system of DATS+
# This code also contains a simple GUI in order to test just this unit alone later.
# Code written by: Minal Haque

#Imports
from scipy.spatial import distance
import cv2
import numpy as np
import time
import mediapipe as mp
import os
import platform


# Helper functions
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

# Thresholds
EAR_THRESH = 0.25
MAR_THRESH = 0.6
ALERT_GENTLE_SEC = 1.5
ALERT_STRONG_SEC = 3
ALERT_LOUD_SEC = 5
ALERT_AUTH_SEC = 10

# Load MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# MediaPipe indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 146, 91, 181, 84, 17, 314, 405]

# State
cap = cv2.VideoCapture(0)
closed_start = None
alerts_fired = {"gentle": False, "strong": False, "loud": False, "auth": False}

print("Drowsiness Detection running (MediaPipe)")
print("Press 'Q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.resize(frame, (640, 480))
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    current_alert = "Driver Alert"
    alert_color = (0, 200, 0)
    ear, mar = None, None

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # Get eye points
        left_eye = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in LEFT_EYE])
        right_eye = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in RIGHT_EYE])
        mouth = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in MOUTH])

        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
        mar = mouth_aspect_ratio(mouth)

        # Draw contours
        cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
        cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)
        cv2.polylines(frame, [mouth], True, (0, 255, 0), 1)

        # Yawn detection
        if mar > MAR_THRESH:
            current_alert = "YAWNING - Stay Alert!"
            alert_color = (0, 140, 255)
            beep_and_speak("Yawn detected")

        # Eyes closed detection
        if ear < EAR_THRESH:
            if closed_start is None:
                closed_start = time.time()
                alerts_fired = {k: False for k in alerts_fired}

            elapsed = time.time() - closed_start

            if elapsed >= ALERT_AUTH_SEC:
                current_alert = "CRITICAL: ALERTING FLEET"
                alert_color = (0, 0, 180)
                if not alerts_fired["auth"]:
                    beep_and_speak("Critical alert")
                    alerts_fired["auth"] = True
            elif elapsed >= ALERT_LOUD_SEC:
                current_alert = "DANGER: VERY DROWSY"
                alert_color = (0, 0, 200)
                if not alerts_fired["loud"]:
                    beep_and_speak("Danger! Wake up!")
                    alerts_fired["loud"] = True
            elif elapsed >= ALERT_STRONG_SEC:
                current_alert = "WARNING: DROWSY"
                alert_color = (0, 0, 220)
                if not alerts_fired["strong"]:
                    beep_and_speak("Warning. Wake up!")
                    alerts_fired["strong"] = True
            elif elapsed >= ALERT_GENTLE_SEC:
                current_alert = "WARNING: Eyes Closing"
                alert_color = (0, 140, 255)
                if not alerts_fired["gentle"]:
                    beep_and_speak("Stay alert")
                    alerts_fired["gentle"] = True
        else:
            closed_start = None
    else:
        current_alert = "FACE NOT DETECTED"
        alert_color = (0, 0, 200)

    # Display info
    cv2.putText(frame, current_alert, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, alert_color, 2)
    if ear is not None:
        cv2.putText(frame, f"EAR: {ear:.2f}  MAR: {mar:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()