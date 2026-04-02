
# This file contains the main code for the Phone detection sub-system of DATS+
# Type "python src/phone_detection/phone_detection.py " on the terminal to run it individually
# This code also contains a simple GUI in order to test just this unit alone later.
# Code provided by: Fatima Faisal
# Edited by: Pooja Gurnani


#Imports
import cv2
import argparse
import os
import platform
import time
import numpy as np
from ultralytics import YOLO
import sys
import threading
import subprocess






# Add parent folder (src/) to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import stats, announce_violation


# Audio alert
frequency = 2000
duration = 1500
_PLATFORM = platform.system()


if _PLATFORM == "Windows":
   ps_process = subprocess.Popen(
       ["powershell", "-NoProfile", "-NonInteractive", "-Command", "-"],
       stdin=subprocess.PIPE,
       stdout=subprocess.DEVNULL,
       stderr=subprocess.DEVNULL
   )
   ps_process.stdin.write(b"Add-Type -AssemblyName System.Speech\n")
   ps_process.stdin.flush()
   _tts = "(New-Object System.Speech.Synthesis.SpeechSynthesizer)"
else:
   ps_process = None


def beep_then_speak(freq, duration_ms, text):
   def _run():
       if _PLATFORM == "Windows":
           cmd = f"[console]::beep({freq},{duration_ms}); {_tts}.Speak('{text}')\n"
           ps_process.stdin.write(cmd.encode())
           ps_process.stdin.flush()
       elif _PLATFORM == "Darwin":
           subprocess.run(["say", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
       else:
           subprocess.run(["espeak", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
   threading.Thread(target=_run, daemon=True).start()


def play_beep():
   def _beep():
       system = platform.system()
       if system == "Windows":
           import winsound
           winsound.Beep(frequency, duration)
       elif system == "Darwin":
           os.system('afplay /System/Library/Sounds/Ping.aiff &')
       else:
           print('\a')
   threading.Thread(target=_beep, daemon=True).start()




# Lazy-load the YOLO model once
_yolo_model = None
_base_model = None


def _get_base_model():
   global _base_model
   if _base_model is None:
       _base_model = YOLO(os.path.join(
           os.path.dirname(os.path.abspath(__file__)), "yolov8n.pt"
       ))
   return _base_model


def _get_model():
   global _yolo_model
   if _yolo_model is None:
       _yolo_model = YOLO(os.path.join(
           os.path.dirname(os.path.abspath(__file__)),
           "runs", "phone_finetune_v23", "weights", "best.pt"
       ))
   return _yolo_model




# Per-frame state
_phone_start = None
_frame_count = 0
_last_boxes = []
_box_persist = 0
_last_person_boxes = []
_last_beep_time = 0




def process_phone_frame(frame, alerts_flags=None):
   global _phone_start, _frame_count, _last_boxes, _box_persist, _last_person_boxes, _last_beep_time


   h, w = frame.shape[:2]


   # Always draw persistent boxes BEFORE frame skip check
   for (x1, y1, x2, y2, confidence) in (_last_boxes if _box_persist > 0 else []):
       cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
       cv2.putText(frame, f"phone {confidence:.2f}", (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


   for (x1, y1, x2, y2, confidence) in _last_person_boxes:
       cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
       cv2.putText(frame, f"person {confidence:.2f}", (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)


   cv2.putText(frame, f"Phone Time: {stats['phone_time']:.1f}s", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


   _frame_count += 1
   if _frame_count % 3 != 0:  # Skip YOLO but boxes still drawn above
       return frame


   # YOLO runs every 3rd frame only
   model = _get_model()
   base_model = _get_base_model()
   _last_person_boxes = []


   # Fine-tuned model for phone
   results = model(frame, verbose=False, conf=0.5, iou=0.5)
   # Base model for person
   base_results = base_model(frame, verbose=False, conf=0.5, iou=0.5)


   phone_detected = False
   current_boxes = []


   for r in results:
       for box in r.boxes:
           cls_id = int(box.cls[0])
           class_name = model.names[cls_id]
           confidence = float(box.conf[0])
           if class_name == "phone" and confidence > 0.5:
               x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
               phone_detected = True
               current_boxes.append((int(x1), int(y1), int(x2), int(y2), confidence))
   for r in base_results:
       for box in r.boxes:
           cls_id = int(box.cls[0])
           class_name = base_model.names[cls_id]
           confidence = float(box.conf[0])
           if class_name == "person" and confidence > 0.5:
               x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
               _last_person_boxes.append((int(x1), int(y1), int(x2), int(y2), confidence))


   now = time.time()


   if current_boxes:
       _last_boxes = current_boxes
       _box_persist = 6
   elif _box_persist > 0:
       _box_persist -= 1
       phone_detected = True


   if phone_detected:
       if _phone_start is None:
           _phone_start = now
       stats["phone_time"] += now - _phone_start
       _phone_start = now


       if now - _last_beep_time > 3:
           beep_then_speak(2000, 500, "Phone usage detected")
           _last_beep_time = now


       cv2.putText(frame, "PHONE DETECTED!", (10, h - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)


       if alerts_flags is not None and not alerts_flags.get("phone"):
           announce_violation("Phone detected while driving!")
           alerts_flags["phone"] = True
   else:
       _phone_start = None


   return frame




# Stand-alone test entry-point
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


   print("Phone detection started. Press ESC or Q to quit.")
   while True:
       ret, frame = cap.read()
       if not ret:
           break
       frame = process_phone_frame(frame)
       cv2.imshow("Phone Detection – ESC to quit", frame)
       key = cv2.waitKey(30) & 0xFF
       if key == 27 or key == ord('q'):
           break


   cap.release()
   cv2.destroyAllWindows()
   cv2.waitKey(1)
   print(f"Done. Total phone time: {stats['phone_time']:.1f}s")


#GUI
if __name__ == "__main__":
   main()
