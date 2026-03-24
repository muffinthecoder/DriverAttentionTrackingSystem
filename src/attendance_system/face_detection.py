############################################# IMPORTING ################################################
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

############################################# BASE DIRECTORY FIX ################################################

# This makes sure all files are always found relative to where main.py is saved
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_path(relative_path):
    return os.path.join(BASE_DIR, relative_path)

############################################# FUNCTIONS ################################################

def assure_path_exists(relative_path):
    full_path = get_path(relative_path)
    if not os.path.exists(full_path):
        os.makedirs(full_path)

##################################################################################

def tick():
    time_string = time.strftime('%H:%M:%S')
    clock.config(text=time_string)
    clock.after(200, tick)

###################################################################################

def contact():
    mess._show(title='Contact us', message="Please contact us on : 'xxxxxxxxxxxxx@gmail.com' ")

###################################################################################

def check_haarcascadefile():
    exists = os.path.isfile(get_path("haarcascade_frontalface_default.xml"))
    if not exists:
        mess._show(title='Some file missing', message='Please contact us for help')
        window.destroy()

###################################################################################

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

###################################################################################

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

#####################################################################################

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

######################################################################################

def clear():
    txt.delete(0, 'end')
    message1.configure(text="1)Take Images  >>>  2)Save Profile")

def clear2():
    txt2.delete(0, 'end')
    message1.configure(text="1)Take Images  >>>  2)Save Profile")

#######################################################################################

def update_registration_count():
    res = 0
    csv_path = get_path("DriverDetails/DriverDetails.csv")
    if os.path.isfile(csv_path):
        with open(csv_path, 'r', newline='') as f:
            reader = csv.reader(f)
            rows = [r for r in reader if any(field.strip() for field in r)]
        res = max(0, len(rows) - 1)  # subtract header
    message.configure(text='Total Registrations till now  : ' + str(res))

#######################################################################################

def TakeImages():
    check_haarcascadefile()
    columns = ['SERIAL NO.', '', 'ID', '', 'NAME']
    assure_path_exists("DriverDetails")
    assure_path_exists("TrainingImage")
    serial = 0
    csv_path = get_path("DriverDetails/DriverDetails.csv")
    if os.path.isfile(csv_path):
        with open(csv_path, 'r', newline='') as csvFile1:
            reader1 = csv.reader(csvFile1)
            rows = [r for r in reader1 if any(field.strip() for field in r)]
        serial = max(0, len(rows) - 1)
    else:
        with open(csv_path, 'a+', newline='') as csvFile1:
            csv.writer(csvFile1).writerow(columns)
        serial = 0

    serial = serial + 1

    Id = txt.get()
    name = txt2.get()

    if (name.isalpha()) or (' ' in name):
        cam = cv2.VideoCapture(0)
        harcascadePath = get_path("haarcascade_frontalface_default.xml")
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0
        while True:
            ret, img = cam.read()
            if not ret:
                break
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                sampleNum += 1
                img_path = get_path("TrainingImage/" + name + "." + str(serial) + "." + Id + '.' + str(sampleNum) + ".jpg")
                cv2.imwrite(img_path, gray[y:y + h, x:x + w])
                cv2.imshow('Taking Images', img)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            elif sampleNum > 100:
                break
        cam.release()
        cv2.destroyAllWindows()
        row = [serial, '', Id, '', name]
        with open(csv_path, 'a+', newline='') as csvFile:
            csv.writer(csvFile).writerow(row)
        message1.configure(text="Images Taken for ID : " + Id)
        update_registration_count()
    else:
        message1.configure(text="Enter Correct name")

########################################################################################

def TrainImages():
    check_haarcascadefile()
    assure_path_exists("TrainingImageLabel")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    harcascadePath = get_path("haarcascade_frontalface_default.xml")
    detector = cv2.CascadeClassifier(harcascadePath)
    faces, ID = getImagesAndLabels(get_path("TrainingImage"))
    try:
        recognizer.train(faces, np.array(ID))
    except Exception:
        mess._show(title='No Registrations', message='Please Register someone first!!!')
        return
    recognizer.save(get_path("TrainingImageLabel/Trainner.yml"))
    message1.configure(text="Profile Saved Successfully")
    update_registration_count()

############################################################################################

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        # filename format: name.serial.id.samplenum.jpg
        ID = int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(ID)
    return faces, Ids

###########################################################################################

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
    faceCascade = cv2.CascadeClassifier(harcascadePath)
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id', 'Name', 'Date', 'Time']

    csv_path = get_path("DriverDetails/DriverDetails.csv")
    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
        # FIX: strip column names and drop empty columns
        df.columns = df.columns.str.strip()
        df = df.dropna(axis=1, how='all')
        df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
        # FIX: make SERIAL NO. int for comparison
        df['SERIAL NO.'] = pd.to_numeric(df['SERIAL NO.'], errors='coerce').fillna(0).astype(int)
    else:
        mess._show(title='Details Missing', message='Students details are missing, please check!')
        cam.release()
        cv2.destroyAllWindows()
        return

    while True:
        ret, im = cam.read()
        if not ret:
            break
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
            serial, conf = recognizer.predict(gray[y:y + h, x:x + w])
            if conf < 50:
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                # FIX: proper int comparison for serial number
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
        # FIX: also close when camera window X is clicked
        if cv2.getWindowProperty('Taking Attendance', cv2.WND_PROP_VISIBLE) < 1:
            break

    cam.release()
    cv2.destroyAllWindows()

    if not attendance:
        mess._show(title='No Attendance', message='No face recognized. Please try again.')
        return

    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
    att_file = get_path("Attendance/Attendance_" + date + ".csv")
    exists = os.path.isfile(att_file)

    if exists:
        with open(att_file, 'a+', newline='') as csvFile1:
            csv.writer(csvFile1).writerow(attendance)
    else:
        with open(att_file, 'a+', newline='') as csvFile1:
            w = csv.writer(csvFile1)
            w.writerow(col_names)
            w.writerow(attendance)

    with open(att_file, 'r', newline='') as csvFile1:
        reader1 = csv.reader(csvFile1)
        next(reader1)  # skip header
        for lines in reader1:
            if len(lines) >= 4:
                tv.insert('', 0, text=str(lines[0]), values=(str(lines[1]), str(lines[2]), str(lines[3])))

    mess._show(title='Attendance Marked', message='Attendance marked successfully!')

######################################## USED STUFFS ############################################

global key
key = ''

ts = time.time()
date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
day, month, year = date.split("-")

mont = {'01': 'January', '02': 'February', '03': 'March', '04': 'April',
        '05': 'May', '06': 'June', '07': 'July', '08': 'August',
        '09': 'September', '10': 'October', '11': 'November', '12': 'December'}

######################################## GUI FRONT-END ###########################################

window = tk.Tk()
window.geometry("1280x720")
window.resizable(True, False)
window.title("Attendance System")
window.configure(background='#262523')

frame1 = tk.Frame(window, bg="#00aeff")
frame1.place(relx=0.11, rely=0.17, relwidth=0.39, relheight=0.80)

frame2 = tk.Frame(window, bg="#00aeff")
frame2.place(relx=0.51, rely=0.17, relwidth=0.38, relheight=0.80)

message3 = tk.Label(window, text="Face Recognition Based Attendance System", fg="white", bg="#262523", width=55,
                    height=1, font=('times', 29, ' bold '))
message3.place(x=10, y=10)

frame3 = tk.Frame(window, bg="#c4c6ce")
frame3.place(relx=0.52, rely=0.09, relwidth=0.09, relheight=0.07)

frame4 = tk.Frame(window, bg="#c4c6ce")
frame4.place(relx=0.36, rely=0.09, relwidth=0.16, relheight=0.07)

datef = tk.Label(frame4, text=day + "-" + mont[month] + "-" + year + "  |  ", fg="orange", bg="#262523", width=55,
                 height=1, font=('times', 22, ' bold '))
datef.pack(fill='both', expand=1)

clock = tk.Label(frame3, fg="orange", bg="#262523", width=55, height=1, font=('times', 22, ' bold '))
clock.pack(fill='both', expand=1)
tick()

head2 = tk.Label(frame2, text="                       For New Registrations                       ", fg="black",
                 bg="#3ece48", font=('times', 17, ' bold '))
head2.grid(row=0, column=0)

head1 = tk.Label(frame1, text="                       For Already Registered                       ", fg="black",
                 bg="#3ece48", font=('times', 17, ' bold '))
head1.place(x=0, y=0)

lbl = tk.Label(frame2, text="Enter ID", width=20, height=1, fg="black", bg="#00aeff", font=('times', 17, ' bold '))
lbl.place(x=80, y=55)

txt = tk.Entry(frame2, width=32, fg="black", font=('times', 15, ' bold '))
txt.place(x=30, y=88)

lbl2 = tk.Label(frame2, text="Enter Name", width=20, fg="black", bg="#00aeff", font=('times', 17, ' bold '))
lbl2.place(x=80, y=140)

txt2 = tk.Entry(frame2, width=32, fg="black", font=('times', 15, ' bold '))
txt2.place(x=30, y=173)

message1 = tk.Label(frame2, text="1)Take Images  >>>  2)Save Profile", bg="#00aeff", fg="black", width=39, height=1,
                    activebackground="yellow", font=('times', 15, ' bold '))
message1.place(x=7, y=230)

message = tk.Label(frame2, text="", bg="#00aeff", fg="black", width=39, height=1, activebackground="yellow",
                   font=('times', 16, ' bold '))
message.place(x=7, y=450)

lbl3 = tk.Label(frame1, text="Attendance", width=20, fg="black", bg="#00aeff", height=1, font=('times', 17, ' bold '))
lbl3.place(x=100, y=115)

##################### MENUBAR #################################

menubar = tk.Menu(window, relief='ridge')
filemenu = tk.Menu(menubar, tearoff=0)
filemenu.add_command(label='Change Password', command=change_pass)
filemenu.add_command(label='Contact Us', command=contact)
filemenu.add_command(label='Exit', command=window.destroy)
menubar.add_cascade(label='Help', font=('times', 29, ' bold '), menu=filemenu)

################## TREEVIEW ATTENDANCE TABLE ####################

tv = ttk.Treeview(frame1, height=13, columns=('name', 'date', 'time'))
tv.column('#0', width=82)
tv.column('name', width=130)
tv.column('date', width=133)
tv.column('time', width=133)
tv.grid(row=2, column=0, padx=(0, 0), pady=(150, 0), columnspan=4)
tv.heading('#0', text='ID')
tv.heading('name', text='NAME')
tv.heading('date', text='DATE')
tv.heading('time', text='TIME')

###################### SCROLLBAR ################################

scroll = ttk.Scrollbar(frame1, orient='vertical', command=tv.yview)
scroll.grid(row=2, column=4, padx=(0, 100), pady=(150, 0), sticky='ns')
tv.configure(yscrollcommand=scroll.set)

###################### BUTTONS ##################################

clearButton = tk.Button(frame2, text="Clear", command=clear, fg="black", bg="#ea2a2a", width=11,
                        activebackground="white", font=('times', 11, ' bold '))
clearButton.place(x=335, y=86)
clearButton2 = tk.Button(frame2, text="Clear", command=clear2, fg="black", bg="#ea2a2a", width=11,
                         activebackground="white", font=('times', 11, ' bold '))
clearButton2.place(x=335, y=172)
takeImg = tk.Button(frame2, text="Take Images", command=TakeImages, fg="white", bg="blue", width=34, height=1,
                    activebackground="white", font=('times', 15, ' bold '))
takeImg.place(x=30, y=300)
trainImg = tk.Button(frame2, text="Save Profile", command=psw, fg="white", bg="blue", width=34, height=1,
                     activebackground="white", font=('times', 15, ' bold '))
trainImg.place(x=30, y=380)
trackImg = tk.Button(frame1, text="Take Attendance", command=TrackImages, fg="black", bg="yellow", width=35, height=1,
                     activebackground="white", font=('times', 15, ' bold '))
trackImg.place(x=30, y=50)
quitWindow = tk.Button(frame1, text="Quit", command=window.destroy, fg="black", bg="red", width=35, height=1,
                       activebackground="white", font=('times', 15, ' bold '))
quitWindow.place(x=30, y=450)

##################### END ######################################

# Load registration count on startup
update_registration_count()

window.configure(menu=menubar)
window.mainloop()

####################################################################################################