# This file contains the main code for the Attendance sub-system of DATS+
# This code also contains an individual GUI in order to test just this unit alone later.
# Type "python src/attendance_system/face_detection.py" on the terminal to run it individually
# Code provided by: Pooja Gurnani
# Additional credits: Code dependancy fixes by Fatima Faisal


#Imports
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox as mess
import tkinter.simpledialog as tsd
import cv2, os
import csv
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import stats, announce_violation


# This makes sure all files are always found relative to where face_detection.py is saved
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# CSV schema
# Three columns only — no empty padding columns.
CSV_COLUMNS = ['SERIAL NO.', 'ID', 'NAME']

def get_path(relative_path):
    return os.path.join(BASE_DIR, relative_path)

#Functions
def assure_path_exists(relative_path):
    full_path = get_path(relative_path)
    if not os.path.exists(full_path):
        os.makedirs(full_path)

def tick():
    time_string = time.strftime('%H:%M:%S')
    clock.config(text=time_string)
    clock.after(200, tick)

def contact():
    mess._show(title='Contact us', message="Please contact us on : 'xxxxxxxxxxxxx@gmail.com' ")

def check_haarcascadefile():
    exists = os.path.isfile(get_path("haarcascade_frontalface_default.xml"))
    if not exists:
        mess._show(title='Some file missing', message='Please contact us for help')
        window.destroy()

def save_pass():
    assure_path_exists("TrainingImageLabel")
    exists1 = os.path.isfile(get_path("TrainingImageLabel/psd.txt"))
    if exists1:
        tf = open(get_path("TrainingImageLabel/psd.txt"), "r")
        key = tf.read()
        tf.close()
    else:
        master.destroy()
        new_pas = tsd.askstring('Old Password not found', 'Please enter a new password below', show='*')
        if new_pas is None:
            mess._show(title='No Password Entered', message='Password not set!! Please try again')
        else:
            tf = open(get_path("TrainingImageLabel/psd.txt"), "w")
            tf.write(new_pas)
            tf.close()
            mess._show(title='Password Registered', message='New password was registered successfully!!')
            return
    op = old.get()
    newp = new.get()
    nnewp = nnew.get()
    if op == key:
        if newp == nnewp:
            txf = open(get_path("TrainingImageLabel/psd.txt"), "w")
            txf.write(newp)
            txf.close()
        else:
            mess._show(title='Error', message='Confirm new password again!!!')
            return
    else:
        mess._show(title='Wrong Password', message='Please enter correct old password.')
        return
    mess._show(title='Password Changed', message='Password changed successfully!!')
    master.destroy()

def change_pass():
    global master
    master = tk.Tk()
    master.geometry("400x160")
    master.resizable(False, False)
    master.title("Change Password")
    master.configure(background="white")
    lbl4 = tk.Label(master, text='    Enter Old Password', bg='white', font=('times', 12, ' bold '))
    lbl4.place(x=10, y=10)
    global old
    old = tk.Entry(master, width=25, fg="black", relief='solid', font=('times', 12, ' bold '), show='*')
    old.place(x=180, y=10)
    lbl5 = tk.Label(master, text='   Enter New Password', bg='white', font=('times', 12, ' bold '))
    lbl5.place(x=10, y=45)
    global new
    new = tk.Entry(master, width=25, fg="black", relief='solid', font=('times', 12, ' bold '), show='*')
    new.place(x=180, y=45)
    lbl6 = tk.Label(master, text='Confirm New Password', bg='white', font=('times', 12, ' bold '))
    lbl6.place(x=10, y=80)
    global nnew
    nnew = tk.Entry(master, width=25, fg="black", relief='solid', font=('times', 12, ' bold '), show='*')
    nnew.place(x=180, y=80)
    cancel = tk.Button(master, text="Cancel", command=master.destroy, fg="black", bg="red", height=1, width=25,
                       activebackground="white", font=('times', 10, ' bold '))
    cancel.place(x=200, y=120)
    save1 = tk.Button(master, text="Save", command=save_pass, fg="black", bg="#3ece48", height=1, width=25,
                      activebackground="white", font=('times', 10, ' bold '))
    save1.place(x=10, y=120)
    master.mainloop()

def psw():
    assure_path_exists("TrainingImageLabel")
    exists1 = os.path.isfile(get_path("TrainingImageLabel/psd.txt"))
    if exists1:
        tf = open(get_path("TrainingImageLabel/psd.txt"), "r")
        key = tf.read()
        tf.close()
    else:
        new_pas = tsd.askstring('Old Password not found', 'Please enter a new password below', show='*')
        if new_pas is None:
            mess._show(title='No Password Entered', message='Password not set!! Please try again')
        else:
            tf = open(get_path("TrainingImageLabel/psd.txt"), "w")
            tf.write(new_pas)
            tf.close()
            mess._show(title='Password Registered', message='New password was registered successfully!!')
            return
    password = tsd.askstring('Password', 'Enter Password', show='*')
    if password == key:
        TrainImages()
    elif password is None:
        pass
    else:
        mess._show(title='Wrong Password', message='You have entered wrong password')

def clear():
    txt.delete(0, 'end')
    message1.configure(text="1)Take Images  >>>  2)Save Profile")

def clear2():
    txt2.delete(0, 'end')
    message1.configure(text="1)Take Images  >>>  2)Save Profile")

def update_registration_count():
    res = 0
    csv_path = get_path("DriverDetails/DriverDetails.csv")
    if os.path.isfile(csv_path):
        with open(csv_path, 'r', newline='', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            rows = [r for r in reader if any(field.strip() for field in r)]
        res = max(0, len(rows) - 1)  # subtract header
    message.configure(text='Total Registrations till now  : ' + str(res))

def _load_or_create_driver_csv():
    """
    Return the path to DriverDetails.csv, creating it with the correct
    3-column header if it does not exist yet.
    """
    assure_path_exists("DriverDetails")
    csv_path = get_path("DriverDetails/DriverDetails.csv")
    if not os.path.isfile(csv_path):
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            csv.writer(f).writerow(CSV_COLUMNS)
    return csv_path

def _next_serial(csv_path):
    """Return the next available serial number (max existing + 1, min 1)."""
    serial = 0
    if os.path.isfile(csv_path):
        with open(csv_path, 'r', newline='', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            rows = [r for r in reader if any(field.strip() for field in r)]
        serial = max(0, len(rows) - 1)   # subtract header row
    return serial + 1

def TakeImages(user_id, user_name, processor=None):
    """Capture face images for registration (works with Streamlit)."""
    try:
        Id   = str(user_id).strip()
        name = str(user_name).strip()

        if not (name.replace(" ", "").isalpha() and Id != ""):
            return False, "Enter a valid name and ID"

        check_haarcascadefile()
        assure_path_exists("TrainingImage")

        csv_path = _load_or_create_driver_csv()
        serial   = _next_serial(csv_path)

        if processor:
            # ── Streamlit path: configure processor to collect frames in recv() ──
            processor.user_id     = Id
            processor.user_name   = name
            processor.serial      = serial
            processor.saved_count = 0
            processor.done        = False
            processor.capturing   = True
            return True, f"Capturing started for {name} (ID: {Id})"

        else:
            # ── Standalone / tkinter path ──
            cam             = cv2.VideoCapture(0)
            harcascadePath  = get_path("haarcascade_frontalface_default.xml")
            detector        = cv2.CascadeClassifier(harcascadePath)
            sampleNum       = 0
            while sampleNum < 100:
                ret, img = cam.read()
                if not ret:
                    break
                gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    sampleNum += 1
                    img_path = get_path(
                        f"TrainingImage/{name}.{serial}.{Id}.{sampleNum}.jpg"
                    )
                    cv2.imwrite(img_path, gray[y:y + h, x:x + w])
            cam.release()
            cv2.destroyAllWindows()

            # Write only 3 columns with utf-8-sig encoding
            with open(csv_path, 'a', newline='', encoding='utf-8-sig') as f:
                csv.writer(f).writerow([serial, Id, name])

            return True, f"Images Taken for ID: {Id}"
    except Exception as e:
        return False, f"Error: {str(e)}"

def TrainImages(reg_placeholder=None):
    check_haarcascadefile()
    assure_path_exists("TrainingImageLabel")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    harcascadePath = get_path("haarcascade_frontalface_default.xml")
    detector = cv2.CascadeClassifier(harcascadePath)
    faces, ID = getImagesAndLabels(get_path("TrainingImage"))
    try:
        recognizer.train(faces, np.array(ID))
    except Exception:
        return False, "No Registrations. Please Register someone first!!!"

    recognizer.save(get_path("TrainingImageLabel/Trainner.yml"))

    # update registration count
    res = 0
    csv_path = get_path("DriverDetails/DriverDetails.csv")
    if os.path.isfile(csv_path):
        with open(csv_path, 'r', newline='', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            rows = [r for r in reader if any(field.strip() for field in r)]
        res = max(0, len(rows) - 1)

    if reg_placeholder:
        reg_placeholder.text(f'Total Registrations till now  : {res}')

    return True, "Profile Saved Successfully"

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp  = np.array(pilImage, 'uint8')
        # filename format: name.serial.id.samplenum.jpg  → index [1] is serial
        ID = int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(ID)
    return faces, Ids

def TrackImages():
    check_haarcascadefile()
    assure_path_exists("Attendance")
    assure_path_exists("DriverDetails")
    for k in tv.get_children():
        tv.delete(k)
    i = 0
    attendance = []

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    trainner_path = get_path("TrainingImageLabel/Trainner.yml")
    if os.path.isfile(trainner_path):
        recognizer.read(trainner_path)
    else:
        mess._show(title='Data Missing', message='Please click on Save Profile to reset data!!')
        return

    harcascadePath = get_path("haarcascade_frontalface_default.xml")
    faceCascade    = cv2.CascadeClassifier(harcascadePath)
    cam            = cv2.VideoCapture(0)
    font           = cv2.FONT_HERSHEY_SIMPLEX
    col_names      = ['Id', 'Name', 'Date', 'Time']

    csv_path = get_path("DriverDetails/DriverDetails.csv")
    if os.path.isfile(csv_path):
        df = _read_driver_csv(csv_path)
    else:
        mess._show(title='Details Missing', message='Students details are missing, please check!')
        cam.release()
        cv2.destroyAllWindows()
        return

    while True:
        ret, im = cam.read()
        if not ret:
            break
        gray  = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
            serial, conf = recognizer.predict(gray[y:y + h, x:x + w])
            if conf < 50:
                ts        = time.time()
                date      = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                row = df[df['SERIAL NO.'] == int(serial)]
                if len(row) > 0:
                    bb = str(row['NAME'].values[0]).strip()
                    ID = str(row['ID'].values[0]).strip()
                else:
                    bb = 'Unknown'
                    ID = 'Unknown'
                attendance = [ID, bb, date, timeStamp]
                cv2.putText(im, bb, (x, y + h), font, 1, (0, 255, 0), 2)
            else:
                cv2.putText(im, 'Unknown', (x, y + h), font, 1, (0, 0, 255), 2)
        cv2.imshow('Taking Attendance', im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty('Taking Attendance', cv2.WND_PROP_VISIBLE) < 1:
            break

    cam.release()
    cv2.destroyAllWindows()

    if not attendance:
        mess._show(title='No Attendance', message='No face recognized. Please try again.')
        return

    ts       = time.time()
    date     = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
    att_file = get_path("Attendance/Attendance_" + date + ".csv")
    exists   = os.path.isfile(att_file)

    with open(att_file, 'a+', newline='', encoding='utf-8-sig') as csvFile1:
        w = csv.writer(csvFile1)
        if not exists:
            w.writerow(col_names)
        w.writerow(attendance)

    with open(att_file, 'r', newline='', encoding='utf-8-sig') as csvFile1:
        reader1 = csv.reader(csvFile1)
        next(reader1)  # skip header
        for lines in reader1:
            if len(lines) >= 4:
                tv.insert('', 0, text=str(lines[0]), values=(str(lines[1]), str(lines[2]), str(lines[3])))

    mess._show(title='Attendance Marked', message='Attendance marked successfully!')

# Shared helpers
def _read_driver_csv(csv_path):
    """
    Read DriverDetails.csv robustly regardless of encoding or column name variations.
    Always returns a DataFrame with SERIAL NO., ID, NAME columns.
    """
    df = pd.read_csv(csv_path, encoding='utf-8-sig')

    # Strip whitespace + any remaining BOM character from ALL column names
    df.columns = [c.strip().replace('\ufeff', '') for c in df.columns]

    # Drop columns whose name is empty or purely whitespace
    df = df.loc[:, ~df.columns.str.match(r'^\s*$')]

    # Drop any remaining Unnamed columns
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]

    # Rename serial column — match anything containing SERIAL (case-insensitive)
    for col in list(df.columns):
        if 'SERIAL' in col.upper() and col != 'SERIAL NO.':
            df = df.rename(columns={col: 'SERIAL NO.'})
            break

    # If STILL no SERIAL NO. column, assume first column is it
    if 'SERIAL NO.' not in df.columns:
        df = df.rename(columns={df.columns[0]: 'SERIAL NO.'})

    # Coerce SERIAL NO. to int
    df['SERIAL NO.'] = pd.to_numeric(df['SERIAL NO.'], errors='coerce').fillna(0).astype(int)

    print("✅ CSV columns after cleaning:", list(df.columns))
    print(df)
    return df

# Shared state for process_attendance_frame()
_att_recognizer       = None
_att_face_cascade     = None
_att_driver_df        = None
_att_last_recognition = None
_att_col_names        = ['Id', 'Name', 'Date', 'Time']

def _att_load_models():
    """Lazy-load the recognizer and cascade once, on first call."""
    global _att_recognizer, _att_face_cascade, _att_driver_df

    harcascadePath = get_path("haarcascade_frontalface_default.xml")
    if not os.path.isfile(harcascadePath):
        return False, "Haar cascade file missing"

    trainner_path = get_path("TrainingImageLabel/Trainner.yml")
    if not os.path.isfile(trainner_path):
        return False, "No trained model found - please Save Profile first"

    csv_path = get_path("DriverDetails/DriverDetails.csv")
    if not os.path.isfile(csv_path):
        return False, "Driver details CSV missing"

    _att_face_cascade = cv2.CascadeClassifier(harcascadePath)

    _att_recognizer = cv2.face.LBPHFaceRecognizer_create()
    _att_recognizer.read(trainner_path)

    # DEBUG — shows raw bytes so we can see any hidden BOM or encoding issues
    print("RAW CSV content (first 200 bytes):")
    with open(csv_path, 'rb') as f:
        print(repr(f.read(200)))

    df = _read_driver_csv(csv_path)
    _att_driver_df = df

    print("✅ Loaded driver CSV columns:", list(df.columns))
    print("✅ Driver data:\n", df)

    return True, "OK"

def process_attendance_frame(frame, alerts_flags=None):
    """
    Process a single frame for attendance/face recognition.
    Intended to be called from main.py each frame.

    Args:
        frame:        BGR frame from OpenCV capture.
        alerts_flags: Optional dict (unused here but kept consistent with
                      the other process_*_frame signatures).

    Returns:
        annotated_frame: Frame with recognition overlay drawn on it.
        status: dict with keys:
            - 'face_found'        (bool)
            - 'recognized'        (bool)
            - 'driver_id'         (str or None)
            - 'driver_name'       (str or None)
            - 'confidence'        (float or None)
            - 'attendance_marked' (bool)  True only on the frame it was written
    """
    global _att_recognizer, _att_face_cascade, _att_driver_df, _att_last_recognition

    status = {
        "face_found":        False,
        "recognized":        False,
        "driver_id":         None,
        "driver_name":       None,
        "confidence":        None,
        "attendance_marked": False,
    }

    # Lazy-load models on first call
    if _att_recognizer is None:
        ok, msg = _att_load_models()
        if not ok:
            cv2.putText(frame, f"Attendance: {msg}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 200), 2)
            return frame, status

    # Ensure attendance folder always exists before any write attempt
    assure_path_exists("Attendance")

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _att_face_cascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        status["face_found"] = True
        cv2.rectangle(frame, (x, y), (x + w, y + h), (225, 0, 0), 2)

        serial, conf = _att_recognizer.predict(gray[y:y + h, x:x + w])
        status["confidence"] = conf

        if conf < 50:
            row = _att_driver_df[_att_driver_df['SERIAL NO.'] == int(serial)]
            if len(row) > 0:
                driver_name = str(row['NAME'].values[0]).strip()
                driver_id   = str(row['ID'].values[0]).strip()
            else:
                driver_name, driver_id = 'Unknown', 'Unknown'

            status["recognized"]  = True
            status["driver_id"]   = driver_id
            status["driver_name"] = driver_name

            cv2.putText(frame, f"{driver_name} (ID:{driver_id})",
                        (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

            # Write attendance only once per unique recognition
            ts        = time.time()
            date_str  = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
            time_str  = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
            record    = (driver_id, driver_name)

            if _att_last_recognition != record:
                _att_last_recognition = record
                att_file = get_path(f"Attendance/Attendance_{date_str}.csv")
                exists   = os.path.isfile(att_file)
                row_data = [driver_id, driver_name, date_str, time_str]

                with open(att_file, 'a+', newline='', encoding='utf-8-sig') as f:
                    w_csv = csv.writer(f)
                    if not exists:
                        w_csv.writerow(_att_col_names)
                    w_csv.writerow(row_data)

                status["attendance_marked"] = True
                print(f"✅ Attendance marked: {driver_name} ({driver_id}) at {time_str}")

        else:
            cv2.putText(frame, 'Unknown', (x, y + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return frame, status


# GUI entry point — ONLY runs when executed directly, never on import
if __name__ == '__main__':
    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
    day, month, year = date.split("-")

    mont = {'01': 'January', '02': 'February', '03': 'March', '04': 'April',
            '05': 'May', '06': 'June', '07': 'July', '08': 'August',
            '09': 'September', '10': 'October', '11': 'November', '12': 'December'}

    window = tk.Tk()
    window.title("Driver Attendance System")
    window.geometry("1200x700")
    window.configure(bg="#f5f7fa")

    style = ttk.Style()
    style.theme_use('clam')

    style.configure("TFrame", background="#f5f7fa")
    style.configure("TLabel", background="#f5f7fa", font=("Segoe UI", 11))
    style.configure("Header.TLabel", font=("Segoe UI", 20, "bold"))
    style.configure("SubHeader.TLabel", font=("Segoe UI", 14, "bold"))
    style.configure("TButton", font=("Segoe UI", 11), padding=6)

    header = ttk.Label(window, text="Driver Attendance System", style="Header.TLabel")
    header.pack(pady=15)

    top_frame = ttk.Frame(window)
    top_frame.pack(fill="x", padx=20)

    datef = ttk.Label(top_frame, text=f"{day}-{mont[month]}-{year}", font=("Segoe UI", 12))
    datef.pack(side="left")

    clock = ttk.Label(top_frame, font=("Segoe UI", 12))
    clock.pack(side="right")
    tick()

    container = ttk.Frame(window)
    container.pack(fill="both", expand=True, padx=20, pady=20)

    left_frame = ttk.Frame(container)
    left_frame.pack(side="left", fill="both", expand=True, padx=10)

    right_frame = ttk.Frame(container)
    right_frame.pack(side="right", fill="both", expand=True, padx=10)

    ttk.Label(left_frame, text="Attendance", style="SubHeader.TLabel").pack(pady=10)

    trackImg = ttk.Button(left_frame, text="Take Attendance", command=TrackImages)
    trackImg.pack(pady=10, fill="x")

    tv = ttk.Treeview(left_frame, columns=('name', 'date', 'time'))
    tv.heading('#0', text='ID')
    tv.heading('name', text='Name')
    tv.heading('date', text='Date')
    tv.heading('time', text='Time')
    tv.column('#0', width=80)
    tv.column('name', width=120)
    tv.column('date', width=120)
    tv.column('time', width=120)
    tv.pack(fill="both", expand=True, pady=10)

    scroll = ttk.Scrollbar(left_frame, orient="vertical", command=tv.yview)
    tv.configure(yscrollcommand=scroll.set)
    scroll.pack(side="right", fill="y")

    quitWindow = ttk.Button(left_frame, text="Quit", command=window.destroy)
    quitWindow.pack(pady=10, fill="x")

    ttk.Label(right_frame, text="New Registration", style="SubHeader.TLabel").pack(pady=10)

    ttk.Label(right_frame, text="Driver ID").pack(anchor="w", padx=5)
    txt = ttk.Entry(right_frame)
    txt.pack(fill="x", padx=5, pady=5)

    ttk.Label(right_frame, text="Driver Name").pack(anchor="w", padx=5)
    txt2 = ttk.Entry(right_frame)
    txt2.pack(fill="x", padx=5, pady=5)

    clearButton = ttk.Button(right_frame, text="Clear ID", command=clear)
    clearButton.pack(pady=5, fill="x")

    clearButton2 = ttk.Button(right_frame, text="Clear Name", command=clear2)
    clearButton2.pack(pady=5, fill="x")

    def _take_images_gui():
        success, msg = TakeImages(txt.get(), txt2.get())
        message1.configure(text=msg)

    takeImg = ttk.Button(right_frame, text="Take Images", command=_take_images_gui)
    takeImg.pack(pady=10, fill="x")

    def save_profile_callback():
        success, msg = TrainImages()
        message1.configure(text=msg)

    trainImg = ttk.Button(right_frame, text="Save Profile", command=save_profile_callback)
    trainImg.pack(pady=5, fill="x")

    message1 = ttk.Label(right_frame, text="1) Take Images  →  2) Save Profile")
    message1.pack(pady=10)

    message = ttk.Label(right_frame, text="")
    message.pack(pady=5)

    menubar  = tk.Menu(window)
    filemenu = tk.Menu(menubar, tearoff=0)
    filemenu.add_command(label='Change Password', command=change_pass)
    filemenu.add_command(label='Contact Us', command=contact)
    filemenu.add_separator()
    filemenu.add_command(label='Exit', command=window.destroy)
    menubar.add_cascade(label='Help', menu=filemenu)
    window.config(menu=menubar)

    update_registration_count()
    window.mainloop()