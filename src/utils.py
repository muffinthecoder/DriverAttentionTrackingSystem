import pyttsx3
import threading
import os

engine = pyttsx3.init()
engine.setProperty('rate', 150)

stats = {
    "phone_time": 0.0,
    "drowsy_time": 0.0,
    "attendance_logged": False,
    "start_time": None,
    "end_time": None
}

def announce_violation(msg):
    """Speak alert in a separate thread"""
    threading.Thread(target=lambda: engine.say(msg) or engine.runAndWait()).start()


def get_path(relative_path):
    """
    Returns the absolute path relative to project root.
    Ensures directories exist.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(root, relative_path)
    dir_name = os.path.dirname(full_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return full_path