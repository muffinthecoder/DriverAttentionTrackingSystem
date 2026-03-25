import sys
import os
import cv2
import time
import threading
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QStackedLayout, QFrame, QTextEdit
)
from PyQt5.QtGui import QPixmap, QImage, QFont, QPalette, QColor
from PyQt5.QtCore import QTimer
import pyttsx3

# ----------------------------
# AUDIO ALERT
# ----------------------------
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def announce_violation(msg):
    threading.Thread(target=lambda: engine.say(msg) or engine.runAndWait()).start()

# ----------------------------
# GLOBAL VARIABLES
# ----------------------------
attendance_image_path = "attendance_dataset"
if not os.path.exists(attendance_image_path):
    os.makedirs(attendance_image_path)

stats = {
    "phone_count": 0,
    "drowsy_count": 0,
    "attendance_logged": False,
    "start_time": None,
    "end_time": None
}

# ----------------------------
# METRIC CARD WIDGET
# ----------------------------
class MetricCard(QFrame):
    def __init__(self, title, color):
        super().__init__()
        self.setFixedHeight(100)
        self.setStyleSheet(f"""
            background-color: #161b22;
            border: 1px solid #30363d;
            border-radius: 10px;
        """)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("color: #8b949e; font-size: 12px;")
        self.layout.addWidget(self.title_label)

        self.value_label = QLabel("0")
        self.value_label.setStyleSheet(f"font-family: 'Share Tech Mono'; font-size: 24px; color: {color};")
        self.layout.addWidget(self.value_label)

    def update_value(self, val):
        self.value_label.setText(str(val))

# ----------------------------
# PAGE 1: Attendance Photo
# ----------------------------
class AttendancePage(QWidget):
    def __init__(self, parent_layout):
        super().__init__()
        self.parent_layout = parent_layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Header
        self.header = QLabel("🚗 Driver Monitoring System\nAI Safety Dashboard")
        self.header.setAlignment(Qt.AlignCenter)
        self.header.setStyleSheet("""
            font-family: 'Share Tech Mono';
            font-size: 24px;
            color: #f97316;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                        stop:0 #161b22, stop:1 #0d1117);
            padding: 15px;
            border-bottom: 2px solid #f97316;
        """)
        self.layout.addWidget(self.header)

        self.label = QLabel("Take your attendance photo")
        self.label.setStyleSheet("color: #c9d1d9; font-size: 14px;")
        self.layout.addWidget(self.label)

        self.capture_btn = QPushButton("Capture Photo")
        self.capture_btn.setStyleSheet("""
            background-color: #21262d;
            color: white;
            border-radius: 6px;
            padding: 8px;
        """)
        self.capture_btn.clicked.connect(self.capture_photo)
        self.layout.addWidget(self.capture_btn)

        self.next_btn = QPushButton("Next")
        self.next_btn.setStyleSheet("""
            background-color: #f97316;
            color: white;
            border-radius: 6px;
            padding: 8px;
        """)
        self.next_btn.clicked.connect(self.go_next)
        self.layout.addWidget(self.next_btn)

        self.image_label = QLabel()
        self.layout.addWidget(self.image_label)

    def capture_photo(self):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if ret:
            filename = os.path.join(attendance_image_path, f"user_{int(time.time())}.png")
            cv2.imwrite(filename, frame)
            self.image_label.setPixmap(self.cv2pixmap(frame))
            stats['attendance_logged'] = True
            announce_violation("Attendance photo captured")
        else:
            self.label.setText("Failed to capture image, try again")

    def cv2pixmap(self, img):
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def go_next(self):
        self.parent_layout.setCurrentIndex(1)
        stats['start_time'] = time.time()

# ----------------------------
# PAGE 2: Live Monitoring
# ----------------------------
class MonitoringPage(QWidget):
    def __init__(self, parent_layout):
        super().__init__()
        self.parent_layout = parent_layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Video panel
        self.video_panel = QFrame()
        self.video_panel.setStyleSheet("""
            background-color: #161b22;
            border: 1px solid #30363d;
            border-radius: 10px;
        """)
        self.video_layout = QVBoxLayout()
        self.video_panel.setLayout(self.video_layout)
        self.layout.addWidget(self.video_panel)

        self.video_label = QLabel()
        self.video_layout.addWidget(self.video_label)

        # Metrics cards
        self.cards_layout = QHBoxLayout()
        self.phone_card = MetricCard("Phones", "#f97316")
        self.drowsy_card = MetricCard("Drowsiness", "#ef4444")
        self.attendance_card = MetricCard("Attendance", "#22c55e")
        self.cards_layout.addWidget(self.phone_card)
        self.cards_layout.addWidget(self.drowsy_card)
        self.cards_layout.addWidget(self.attendance_card)
        self.layout.addLayout(self.cards_layout)

        # Stop button
        self.stop_btn = QPushButton("Stop Monitoring")
        self.stop_btn.setStyleSheet("""
            background-color: #ef4444;
            color: white;
            border-radius: 6px;
            padding: 8px;
        """)
        self.stop_btn.clicked.connect(self.stop_monitoring)
        self.layout.addWidget(self.stop_btn)

        # Video capture
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # ----------------------------
            # Placeholder: phone detection
            if np.random.rand() < 0.01:
                stats['phone_count'] += 1
                announce_violation("Phone usage detected")
                cv2.putText(frame, "PHONE", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            # Placeholder: drowsiness detection
            if np.random.rand() < 0.01:
                stats['drowsy_count'] += 1
                announce_violation("Drowsiness detected")
                cv2.putText(frame, "DROWSY", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

            self.video_label.setPixmap(self.cv2pixmap(frame))
            self.update_metrics()

    def cv2pixmap(self, img):
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def update_metrics(self):
        self.phone_card.update_value(stats['phone_count'])
        self.drowsy_card.update_value(stats['drowsy_count'])
        self.attendance_card.update_value("Yes" if stats['attendance_logged'] else "No")

    def stop_monitoring(self):
        self.timer.stop()
        self.cap.release()
        stats['end_time'] = time.time()
        self.parent_layout.setCurrentIndex(2)

# ----------------------------
# PAGE 3: Statistics
# ----------------------------
class StatsPage(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.label = QLabel("Monitoring Report")
        self.label.setStyleSheet("font-family: 'Share Tech Mono'; font-size: 18px; color: #f97316;")
        self.layout.addWidget(self.label)

        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setStyleSheet("""
            background-color: #161b22;
            color: #c9d1d9;
            border: 1px solid #30363d;
            border-radius: 10px;
        """)
        self.layout.addWidget(self.text)

        self.show_stats()

    def show_stats(self):
        duration = stats['end_time'] - stats['start_time'] if stats['end_time'] else 0
        report = f"""
Attendance logged: {'Yes' if stats['attendance_logged'] else 'No'}
Monitoring duration: {duration:.1f} seconds
Phone usage detected: {stats['phone_count']} times
Drowsiness detected: {stats['drowsy_count']} times
"""
        self.text.setText(report)

# ----------------------------
# MAIN APP
# ----------------------------
from PyQt5.QtCore import Qt

class DriverMonitoringApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Driver Monitoring System")
        self.setGeometry(100, 100, 900, 700)
        self.setStyleSheet("background-color: #0d1117;")

        self.layout = QStackedLayout()
        self.setLayout(self.layout)

        self.page1 = AttendancePage(self.layout)
        self.page2 = MonitoringPage(self.layout)
        self.page3 = StatsPage()

        self.layout.addWidget(self.page1)
        self.layout.addWidget(self.page2)
        self.layout.addWidget(self.page3)

# ----------------------------
# RUN
# ----------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DriverMonitoringApp()
    window.show()
    sys.exit(app.exec_())