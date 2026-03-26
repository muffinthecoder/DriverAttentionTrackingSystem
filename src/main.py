# main.py

# STANDARD LIBRARIES

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

# SUBSYSTEMS

from phone_detection.phone_detection import process_phone_frame
from drowsiness_detection.drowsiness import process_drowsiness_frame
from attendance_system.face_detection import TakeImages, TrainImages, process_attendance_frame

# PAGE CONFIG
st.set_page_config(page_title="Driver Attention Tracking System (DATS+)", layout="wide")

# LOAD CSS
def load_css():
    css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "style.css")
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# HEADER

st.markdown("""
<div class="dms-header">
    <div>
        <h1 class="dms-title">Driver Attention Tracking System (DATS+)</h1>
        <p class="dms-subtitle">AI-powered Safety Dashboard</p>
    </div>
</div>
""", unsafe_allow_html=True)

# TABS
tab1, tab2, tab3 = st.tabs(["📸 Register Face", "📊 Monitoring", "👥 About Us"])

# TAB 1: FACE REGISTRATION
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

# TAB 2: MONITORING
with tab2:

    # Live metric placeholders
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

    # Shared alert flags (accessed by the video processor)
    if "alerts_flags" not in st.session_state:
        st.session_state.alerts_flags = {"phone": False, "drowsy": False, "attendance": False}

    # WEBRTC VIDEO PROCESSOR
    # Uses OpenCV - need to change this later!!!
    class DriverMonitorProcessor(VideoProcessorBase):
        def __init__(self):
            self.alerts_flags = {"phone": False, "drowsy": False, "attendance": False}

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            img = process_phone_frame(img, self.alerts_flags)
            img = process_drowsiness_frame(img, self.alerts_flags)
            img = process_attendance_frame(img, self.alerts_flags)
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

    # Refresh metrics while stream is active
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

    # SESSION REPORT
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


with tab3:
    st.markdown(
"""<div class="about-wrapper">

<h2 class="about-title">About Team Attenzen</h2>
<p class="about-subtitle">A student-led initiative building smarter, safer roads through AI.</p>

<div class="about-card">
<h3>🚗 The Problem We're Solving</h3>
<p>Driver fatigue and inattentiveness are among the leading causes of road accidents worldwide.
In commercial transport — taxis, public buses, trams, delivery trucks — drivers often work long
shifts with little oversight. Employers have no reliable way to know if a driver is alert, or
even present, at the wheel. Manual attendance systems are slow, error-prone, and easy to game.
<br><br>
<strong>Attenzen</strong> is our answer to that problem.</p>
</div>

<div class="about-card">
<h3>🧠 What This System Does</h3>
<p>The Driver Attention Tracking System (DATS) is a real-time, AI-powered monitoring platform
designed as a <em>proof of concept</em> for commercial vehicle deployment. It combines three
intelligent subsystems into a single unified dashboard:</p>
<ul>
<li><strong>Drowsiness Detection</strong> — monitors eye closure and blink rate to detect fatigue and trigger alerts before an incident occurs.</li>
<li><strong>Phone Usage Detection</strong> — uses object detection to identify when a driver picks up or holds a mobile device while driving.</li>
<li><strong>Automatic Attendance</strong> — recognises the registered driver's face at session start and logs their attendance without any manual input.</li>
</ul>
<p>All three run simultaneously on a live camera feed, with a session report generated at the
end of every drive — giving fleet operators a clear, timestamped record of driver behaviour.</p>
</div>

<div class="about-card">
<h3>🚌 Real-World Applications</h3>
<p>While DATS is currently a proof of concept, the underlying technology is directly applicable to:</p>
<ul>
<li><strong>Ride-hailing & taxis</strong> — ensure drivers are who they claim to be and are alert throughout a trip.</li>
<li><strong>Public buses & trams</strong> — monitor fatigue on long urban routes and flag incidents automatically.</li>
<li><strong>Delivery & logistics</strong> — track driver hours accurately and reduce liability from distracted driving.</li>
<li><strong>School transport</strong> — give parents and operators peace of mind on every route.</li>
<li><strong>Long-haul trucking</strong> — combat highway fatigue, one of the deadliest risks in freight transport.</li>
</ul>
</div>

<h3 style="font-size: 1.3rem; font-weight: 700; margin-bottom: 1.2rem;">👩‍💻 Meet the Team</h3>

<div class="team-grid">
<div class="team-card">
<div class="team-emoji">👩‍💻</div>
<h4>Fatima Faisal</h4>
<p class="team-role">Lead Developer & Systems Integrator</p>
<p>Responsible for the overall system architecture, Streamlit dashboard,
WebRTC integration, and connecting the three AI subsystems into a single
cohesive pipeline. Developed the Phone detection model. Handled CSS theming and deployment setup.</p>
</div>
<div class="team-card">
<div class="team-emoji">👩‍🔬</div>
<h4>Pooja Gurnani</h4>
<p class="team-role">Attendance & Face Recognition Engineer</p>
<p>Designed and built the face recognition attendance subsystem using
OpenCV's LBPH algorithm. Implemented driver registration, model training,
and real-time identity verification with CSV-based logging.</p>
</div>
<div class="team-card">
<div class="team-emoji">👩‍🎓</div>
<h4>Minal Haque</h4>
<p class="team-role">Safety Detection & Alert Systems</p>
<p>Developed the drowsiness detection module,
combining facial landmark analysis and object detection to provide
real-time safety alerts and session-level behavioural reporting.</p>
</div>
</div>

<div class="about-disclaimer">
<p>🎓 &nbsp;This project was developed as an academic proof of concept.<br>
<span>DATS is not yet a commercial product. All detections are experimental
and should not be relied upon for safety-critical decisions without
further validation and regulatory approval.</span></p>
</div>

</div>""", unsafe_allow_html=True)



# FOOTER
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>Driver Attention Tracking System (DATS+) &nbsp;|&nbsp; AI-powered Safety Dashboard</p>
    <p>Built by &nbsp;
        <strong>Fatima Faisal</strong> &nbsp;·&nbsp;
        <strong>Minal Haque</strong> &nbsp;·&nbsp;
        <strong>Pooja Gurnani</strong>
    </p>
</div>
""", unsafe_allow_html=True)