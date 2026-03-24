import streamlit as st
import cv2
import numpy as np
from datetime import datetime
import time
from PIL import Image

# IMPORTANT: keep these imports AFTER fixing environment
from ultralytics import YOLO
import mediapipe as mp

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Driver Monitoring System",
    layout="wide"
)

# ─────────────────────────────────────────────
# LOAD CSS
# ─────────────────────────────────────────────
def load_css():
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "phone_count" not in st.session_state:
    st.session_state.phone_count = 0
if "drowsy_count" not in st.session_state:
    st.session_state.drowsy_count = 0
if "attendance" not in st.session_state:
    st.session_state.attendance = False

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    yolo = YOLO("yolov8n.pt")

    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh()

    return yolo, face_mesh

model, face_mesh = load_models()

# ─────────────────────────────────────────────
# DETECTION FUNCTIONS
# ─────────────────────────────────────────────
def detect_phone(img):
    results = model(img)
    count = 0

    for r in results:
        for box in r.boxes:
            if model.names[int(box.cls[0])] == "cell phone":
                count += 1

    return count


def detect_drowsiness(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        return True

    return False


# ─────────────────────────────────────────────
# UI HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="dms-header">
    <h1 class="dms-title">🚗 Driver Monitoring System</h1>
    <p class="dms-subtitle">AI Safety Dashboard</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="card-grid">
    <div class="metric-card phone">
        <div class="card-title">Phones</div>
        <div class="card-value">{st.session_state.phone_count}</div>
    </div>
    <div class="metric-card drowsy">
        <div class="card-title">Drowsiness</div>
        <div class="card-value">{st.session_state.drowsy_count}</div>
    </div>
    <div class="metric-card attend">
        <div class="card-title">Attendance</div>
        <div class="card-value">{'Yes' if st.session_state.attendance else 'No'}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CAMERA
# ─────────────────────────────────────────────
st.markdown('<div class="video-panel">', unsafe_allow_html=True)

photo = st.camera_input("")

if photo:
    img = Image.open(photo)
    img = np.array(img)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # detections
    phones = detect_phone(img_bgr)
    drowsy = detect_drowsiness(img_bgr)

    st.session_state.phone_count = phones

    if drowsy:
        st.session_state.drowsy_count += 1

    if phones > 0:
        cv2.putText(img_bgr, "PHONE DETECTED", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

else:
    st.info("Take a photo to start")

st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.write("ICT304 Project - DATS AI")