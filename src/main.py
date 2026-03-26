# main.py

# ----------------------------
# STANDARD LIBRARIES
# ----------------------------
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import stats, announce_violation
from camera import Camera, get_rgb_frame

import streamlit as st
import time
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# ----------------------------
# SUBSYSTEMS
# ----------------------------
from phone_detection.phone_detection import process_phone_frame
from drowsiness_detection.drowsiness import process_drowsiness_frame
from attendance_system.face_detection import TakeImages, TrainImages, process_attendance_frame

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Driver Monitoring System", layout="wide")

# ----------------------------
# LOAD CSS
# ----------------------------
def load_css():
    css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets/style.css")
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ----------------------------
# HEADER + LOGO
# ----------------------------
col1, col2 = st.columns([1, 6])

with col1:
    try:
        st.image("assets/logo.png", width=80)
    except Exception:
        pass  # logo is optional

with col2:
    st.markdown("""
    <div class="dms-header">
        <div>
            <h1 class="dms-title">Driver Monitoring System</h1>
            <p class="dms-subtitle">AI-powered Safety Dashboard</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------
# TABS
# ----------------------------
tab1, tab2 = st.tabs(["📸 Register Face", "📊 Monitoring"])

# ----------------------------
# TAB 1: FACE REGISTRATION
# ----------------------------
with tab1:
    st.subheader("Register Driver")
    user_id   = st.text_input("Enter ID")
    user_name = st.text_input("Enter Name")

    if st.button("Capture Images"):
        if user_id and user_name:
            result = TakeImages(user_id, user_name)
            st.success(result if result else "Images Captured")
        else:
            st.warning("Please enter both ID and Name.")

    if st.button("Train Model"):
        msg = TrainImages()
        st.success(msg if msg else "Model Trained")

# ----------------------------
# TAB 2: MONITORING
# ----------------------------
with tab2:

    # ── Live metric placeholders ──
    col1, col2, col3 = st.columns(3)
    phone_placeholder  = col1.empty()
    drowsy_placeholder = col2.empty()
    attend_placeholder = col3.empty()

    def render_metrics():
        phone_placeholder.metric("Phone Time (s)",  f"{stats['phone_time']:.1f}")
        drowsy_placeholder.metric("Drowsy Time (s)", f"{stats['drowsy_time']:.1f}")
        attend_placeholder.metric("Attendance",      "Yes" if stats['attendance_logged'] else "No")

    render_metrics()

    st.markdown("### Live Feed")

    # ── Shared alert flags (accessed by the video processor) ──
    if "alerts_flags" not in st.session_state:
        st.session_state.alerts_flags = {"phone": False, "drowsy": False, "attendance": False}

    # ----------------------------
    # WEBRTC VIDEO PROCESSOR
    # Uses OpenCV — runs in its own thread, no Streamlit conflict
    # ----------------------------
    class DriverMonitorProcessor(VideoProcessorBase):
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            # Convert incoming frame to a BGR numpy array (OpenCV format)
            img = frame.to_ndarray(format="bgr24")

            # ── Run your OpenCV-based subsystems ──
            img = process_phone_frame(img,      st.session_state.alerts_flags)
            img = process_drowsiness_frame(img, st.session_state.alerts_flags)
            img = process_attendance_frame(img, st.session_state.alerts_flags)

            # Return processed frame back to the browser stream
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    # ICE/STUN config — needed for WebRTC to work on most networks
    RTC_CONFIG = RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })

    webrtc_ctx = webrtc_streamer(
        key="driver-monitor",
        video_processor_factory=DriverMonitorProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # ── Refresh metrics while stream is active ──
    if webrtc_ctx.state.playing:
        if not stats.get("start_time"):
            stats["start_time"] = time.time()
            stats["phone_time"]        = 0.0
            stats["drowsy_time"]       = 0.0
            stats["attendance_logged"] = False

        render_metrics()

    elif stats.get("start_time") and not stats.get("end_time"):
        stats["end_time"] = time.time()
        render_metrics()

# ----------------------------
# SESSION REPORT
# ----------------------------
st.markdown("### Session Report")
duration = 0.0
if stats.get("start_time") and stats.get("end_time"):
    duration = stats["end_time"] - stats["start_time"]

st.write(f"""
- **Phone Usage Time:** {stats['phone_time']:.2f} sec  
- **Drowsiness Time:** {stats['drowsy_time']:.2f} sec  
- **Attendance Logged:** {"Yes" if stats['attendance_logged'] else "No"}  
- **Session Duration:** {duration:.2f} sec  
""")

# ----------------------------
# FOOTER
# ----------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1.5rem 0 1rem 0;">
    <img src="app/static/logo.png" width="55" style="margin-bottom: 0.6rem; opacity: 0.85;" />
    <p style="font-size: 0.85rem; color: #888; margin: 0.2rem 0;">
        Driver Monitoring System &nbsp;|&nbsp; AI-powered Safety Dashboard
    </p>
    <p style="font-size: 0.8rem; color: #aaa; margin: 0.4rem 0 0 0;">
        Built by &nbsp;
        <strong>Fatima Faisal</strong> &nbsp;·&nbsp;
        <strong>Minal Haque</strong> &nbsp;·&nbsp;
        <strong>Pooja Gurnani</strong>
    </p>
</div>
""", unsafe_allow_html=True)