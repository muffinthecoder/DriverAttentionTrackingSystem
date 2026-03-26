# This file contains the main code for the Attendance sub-system of DATS+
# This code also contains an individual GUI in order to test just this unit alone later.
# Code provided by: Pooja Gurnani
# Additional credits: Code dependancies and GUI fixes by Fatima Faisal

# ── Imports ───────────────────────────────────────────────────────────────────
import cv2
import os
import csv
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time
import sys

# Add parent folder (src/) to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import stats, announce_violation

# ── Base directory (relative to this file) ───────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_path(relative_path):
    return os.path.join(BASE_DIR, relative_path)


def assure_path_exists(relative_path):
    full = get_path(relative_path)
    if not os.path.exists(full):
        os.makedirs(full)


# ── Face-image helpers ────────────────────────────────────────────────────────
def getImagesAndLabels(path):
    """Return (faces, ids) from training-image folder."""
    image_paths = [os.path.join(path, f) for f in os.listdir(path)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    faces, ids = [], []
    for img_path in image_paths:
        pil_img = Image.open(img_path).convert('L')
        img_np = np.array(pil_img, 'uint8')
        # filename: name.serial.id.sample.jpg  → serial is index [1]
        try:
            label = int(os.path.basename(img_path).split(".")[1])
        except (IndexError, ValueError):
            continue
        faces.append(img_np)
        ids.append(label)
    return faces, ids


# ── Public API (called from main.py) ─────────────────────────────────────────

def TakeImages(user_id, user_name):
    """
    Capture 100 face images for *user_id* / *user_name* and save to
    TrainingImage/.  Returns a status string.
    """
    haarcascade = get_path("haarcascade_frontalface_default.xml")
    if not os.path.isfile(haarcascade):
        return "ERROR: haarcascade_frontalface_default.xml not found"

    if not (user_name.replace(" ", "").isalpha()):
        return "ERROR: Name must contain letters only"

    assure_path_exists("DriverDetails")
    assure_path_exists("TrainingImage")

    columns = ['SERIAL NO.', '', 'ID', '', 'NAME']
    csv_path = get_path("DriverDetails/DriverDetails.csv")

    if os.path.isfile(csv_path):
        with open(csv_path, 'r', newline='') as f:
            rows = [r for r in csv.reader(f) if any(field.strip() for field in r)]
        serial = max(0, len(rows) - 1)
    else:
        with open(csv_path, 'a+', newline='') as f:
            csv.writer(f).writerow(columns)
        serial = 0

    serial += 1

    detector = cv2.CascadeClassifier(haarcascade)
    cam = cv2.VideoCapture(0)
    sample_num = 0

    while sample_num < 100:
        ret, img = cam.read()
        if not ret:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            sample_num += 1
            img_path = get_path(
                f"TrainingImage/{user_name}.{serial}.{user_id}.{sample_num}.jpg"
            )
            cv2.imwrite(img_path, gray[y:y + h, x:x + w])
            if sample_num >= 100:
                break

    cam.release()

    with open(csv_path, 'a+', newline='') as f:
        csv.writer(f).writerow([serial, '', user_id, '', user_name])

    return f"Captured {sample_num} images for {user_name} (ID: {user_id})"


def TrainImages():
    """
    Train LBPH recognizer on saved images.
    Returns a status string.
    """
    haarcascade = get_path("haarcascade_frontalface_default.xml")
    if not os.path.isfile(haarcascade):
        return "ERROR: haarcascade_frontalface_default.xml not found"

    assure_path_exists("TrainingImageLabel")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, ids = getImagesAndLabels(get_path("TrainingImage"))

    if not faces:
        return "ERROR: No training images found. Please register a driver first."

    recognizer.train(faces, np.array(ids))
    recognizer.save(get_path("TrainingImageLabel/Trainner.yml"))
    return f"Model trained on {len(faces)} images."


# ── Per-frame state ───────────────────────────────────────────────────────────
_recognizer = None
_face_cascade = None
_attendance_marked = set()  # IDs already logged this session


def _load_models():
    global _recognizer, _face_cascade
    if _face_cascade is None:
        haarcascade = get_path("haarcascade_frontalface_default.xml")
        _face_cascade = cv2.CascadeClassifier(haarcascade)

    if _recognizer is None:
        trainner = get_path("TrainingImageLabel/Trainner.yml")
        if os.path.isfile(trainner):
            _recognizer = cv2.face.LBPHFaceRecognizer_create()
            _recognizer.read(trainner)


def process_attendance_frame(frame, alerts_flags=None):
    """
    Run face recognition on *frame*.
    Annotates the frame and updates stats["attendance_logged"].
    Returns the annotated frame.
    """
    _load_models()

    if _face_cascade is None or _recognizer is None:
        cv2.putText(frame, "TRAIN MODEL FIRST", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame

    # Load driver CSV for name look-up
    csv_path = get_path("DriverDetails/DriverDetails.csv")
    driver_df = None
    if os.path.isfile(csv_path):
        try:
            driver_df = pd.read_csv(csv_path)
            driver_df.columns = driver_df.columns.str.strip()
            driver_df = driver_df.dropna(axis=1, how='all')
            driver_df = driver_df.loc[:, ~driver_df.columns.str.startswith('Unnamed')]
            driver_df['SERIAL NO.'] = pd.to_numeric(
                driver_df['SERIAL NO.'], errors='coerce').fillna(0).astype(int)
        except Exception:
            driver_df = None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (225, 0, 0), 2)
        serial, conf = _recognizer.predict(gray[y:y + h, x:x + w])

        if conf < 50:
            name = str(serial)
            if driver_df is not None:
                row = driver_df[driver_df['SERIAL NO.'] == int(serial)]
                if len(row) > 0:
                    name = str(row['NAME'].values[0]).strip()

            cv2.putText(frame, name, (x, y + h),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Log attendance once per ID per session
            if serial not in _attendance_marked:
                _attendance_marked.add(serial)
                _save_attendance(serial, name)
                stats["attendance_logged"] = True
                if alerts_flags is not None:
                    alerts_flags["attendance"] = True
        else:
            cv2.putText(frame, "Unknown", (x, y + h),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame


def _save_attendance(serial, name):
    assure_path_exists("Attendance")
    ts = time.time()
    date_str = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
    time_str = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    att_file = get_path(f"Attendance/Attendance_{date_str}.csv")
    row = [serial, name, date_str, time_str]
    col_names = ['Id', 'Name', 'Date', 'Time']
    exists = os.path.isfile(att_file)

    with open(att_file, 'a+', newline='') as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(col_names)
        w.writerow(row)


# ── Stand-alone tkinter GUI (only when run directly) ─────────────────────────
if __name__ == "__main__":
    import tkinter as tk
    from tkinter import ttk, messagebox as mess
    import tkinter.simpledialog as tsd


    def check_haarcascadefile():
        if not os.path.isfile(get_path("haarcascade_frontalface_default.xml")):
            mess.showerror('File missing', 'haarcascade_frontalface_default.xml not found')
            window.destroy()


    def update_registration_count():
        res = 0
        csv_path = get_path("DriverDetails/DriverDetails.csv")
        if os.path.isfile(csv_path):
            with open(csv_path, 'r', newline='') as f:
                rows = [r for r in csv.reader(f) if any(x.strip() for x in r)]
            res = max(0, len(rows) - 1)
        message.configure(text=f'Total Registrations: {res}')


    def gui_take_images():
        uid = txt.get().strip()
        name = txt2.get().strip()
        if not uid or not name:
            mess.showwarning('Input', 'Please enter both ID and Name')
            return
        result = TakeImages(uid, name)
        mess.showinfo('Done', result)
        update_registration_count()


    def gui_train():
        result = TrainImages()
        mess.showinfo('Done', result)


    def gui_track():
        check_haarcascadefile()
        _load_models()
        if _recognizer is None:
            mess.showerror('Error', 'Train model first!')
            return
        csv_path = get_path("DriverDetails/DriverDetails.csv")
        if not os.path.isfile(csv_path):
            mess.showerror('Error', 'No driver details found!')
            return
        driver_df = pd.read_csv(csv_path)
        driver_df.columns = driver_df.columns.str.strip()
        driver_df = driver_df.dropna(axis=1, how='all')
        driver_df['SERIAL NO.'] = pd.to_numeric(
            driver_df['SERIAL NO.'], errors='coerce').fillna(0).astype(int)

        cam = cv2.VideoCapture(0)
        while True:
            ret, im = cam.read()
            if not ret:
                break
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = _face_cascade.detectMultiScale(gray, 1.2, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
                serial, conf = _recognizer.predict(gray[y:y + h, x:x + w])
                if conf < 50:
                    row = driver_df[driver_df['SERIAL NO.'] == int(serial)]
                    name = str(row['NAME'].values[0]).strip() if len(row) > 0 else str(serial)
                    cv2.putText(im, name, (x, y + h), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(im, 'Unknown', (x, y + h), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Attendance – press Q to quit', im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if cv2.getWindowProperty('Attendance – press Q to quit', cv2.WND_PROP_VISIBLE) < 1:
                break
        cam.release()
        cv2.destroyAllWindows()


    # ── GUI layout ────────────────────────────────────────────────────────────
    window = tk.Tk()
    window.title("Driver Attendance System")
    window.geometry("600x400")
    window.configure(bg="#f5f7fa")

    style = ttk.Style()
    style.theme_use('clam')

    ttk.Label(window, text="Driver Attendance System",
              font=("Segoe UI", 18, "bold")).pack(pady=15)

    frame_reg = ttk.Frame(window)
    frame_reg.pack(pady=10)

    ttk.Label(frame_reg, text="Driver ID:").grid(row=0, column=0, padx=5, pady=5, sticky='e')
    txt = ttk.Entry(frame_reg, width=25)
    txt.grid(row=0, column=1, padx=5, pady=5)

    ttk.Label(frame_reg, text="Driver Name:").grid(row=1, column=0, padx=5, pady=5, sticky='e')
    txt2 = ttk.Entry(frame_reg, width=25)
    txt2.grid(row=1, column=1, padx=5, pady=5)

    btn_frame = ttk.Frame(window)
    btn_frame.pack(pady=10)

    ttk.Button(btn_frame, text="Take Images", command=gui_take_images).grid(row=0, column=0, padx=10)
    ttk.Button(btn_frame, text="Train Model", command=gui_train).grid(row=0, column=1, padx=10)
    ttk.Button(btn_frame, text="Take Attendance", command=gui_track).grid(row=0, column=2, padx=10)
    ttk.Button(btn_frame, text="Quit", command=window.destroy).grid(row=0, column=3, padx=10)

    message = ttk.Label(window, text="")
    message.pack(pady=5)

    update_registration_count()
    window.mainloop()

