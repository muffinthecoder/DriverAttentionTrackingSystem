from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import numpy as np
import time
import os

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

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
predictor = dlib.shape_predictor(os.path.join(BASE_DIR, "models", "shape_predictor_68_face_landmarks.dat"))

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

        if not ret or frame is None or frame.size == 0:
                continue

        frame = imutils.resize(frame, width=450)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray  = np.ascontiguousarray(gray, dtype=np.uint8)

        faces = face_detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5,
                minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in faces:
                rect  = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
                shape = face_utils.shape_to_np(predictor(gray, rect))

                # ── Eyes ─────────────────────────────────────────────────────
                leftEye  = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                ear      = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0

                cv2.drawContours(frame, [cv2.convexHull(leftEye)],  -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)

                # ── Mouth ─────────────────────────────────────────────────────
                mar = mouth_aspect_ratio(shape)
                cv2.drawContours(frame, [shape[48:60]], -1, (0, 255, 0), 1)

                # ── Head droop ────────────────────────────────────────────────
                droop = head_droop_ratio(shape)

                # ── Choose mode ───────────────────────────────────────────────
                if sunglasses_mode:
                        mode   = "HEAD POSE [sunglasses mode]"
                        drowsy = droop < HEAD_DROOP_THRESH
                else:
                        mode   = "EAR"
                        drowsy = ear < EAR_THRESH

                # ── On screen info ────────────────────────────────────────────
                mode_color = (0, 165, 255) if sunglasses_mode else (255, 255, 0)
                cv2.putText(frame, f"Mode: {mode}",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, mode_color, 1)
                cv2.putText(frame, f"EAR: {ear:.2f}  MAR: {mar:.1f}  Droop: {droop:.2f}",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(frame, "S: sunglasses mode  Q: quit",
                        (10, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

                # ── Yawn alert ────────────────────────────────────────────────
                if mar > YAWN_THRESH:
                        cv2.putText(frame, "Yawning Detected - Stay Alert!",
                                (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 165, 255), 2)
                        print(f"Yawn detected - MAR: {mar:.1f}")

                # ── Drowsiness timer ──────────────────────────────────────────
                if drowsy:
                        if closed_start is None:
                                closed_start = time.time()

                        elapsed = time.time() - closed_start

                        cv2.putText(frame, f"Eyes closed: {elapsed:.1f}s",
                                (10, 75), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), 1)

                        if elapsed >= ALERT_GENTLE_SEC:
                                cv2.putText(frame, "Drowsy - Stay Alert",
                                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.65, (0, 165, 255), 2)

                        if elapsed >= ALERT_STRONG_SEC:
                                cv2.putText(frame, "WARNING - Wake Up!",
                                        (10, 130), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7, (0, 100, 255), 2)

                        if elapsed >= ALERT_LOUD_SEC:
                                cv2.putText(frame, "!! DANGER - WAKE UP NOW !!",
                                        (10, 160), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7, (0, 0, 255), 2)

                        if elapsed >= ALERT_AUTH_SEC:
                                cv2.putText(frame, "!! ALERTING FLEET MANAGEMENT !!",
                                        (10, 190), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6, (0, 0, 255), 2)
                                print(f"CRITICAL: Driver drowsy for {elapsed:.0f}s - alerting authorities")

                else:
                        closed_start = None

        cv2.imshow("DATS+ Drowsiness Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
                break
        elif key == ord("s"):
                sunglasses_mode = not sunglasses_mode
                print(f"Sunglasses mode: {'ON' if sunglasses_mode else 'OFF'}")

cv2.destroyAllWindows()
cap.release()