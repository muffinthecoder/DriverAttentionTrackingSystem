# Unit: ICT 304
# Final Project - DATS+ System
# This is the main.py which contains the code for the streamlit dashboard.
# All three subsystems are integrated here.
# Code by: Fatima Faisal and Minal Haque


# STANDARD LIBRARIES
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import stats, announce_violation
from camera import Camera, get_rgb_frame
import base64
from tkinter import filedialog
import pandas as pd
import streamlit as st
import time
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration


# SUBSYSTEMS
from phone_detection.phone_detection import process_phone_frame
from drowsiness_detection.drowsiness import process_drowsiness_frame
from attendance_system.face_detection import (
   TakeImages,
   TrainImages,
   process_attendance_frame
)


# logo imported
logo_path = os.path.join(
   os.path.dirname(os.path.abspath(__file__)),
   "..",
   "assets",
   "logo.png"
)
# Page configuration
st.set_page_config(page_title="Driver Attention Tracking System (DATS+)", layout="wide")


# Load the CSS
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
tab1, tab2, tab3, tab4 = st.tabs(["📸 Register Face", "📊 Monitoring", "👤 Admin", "👥 About Us"])

# TAB 1: FACE REGISTRATION
class RegisterProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame       = None
        self.capturing   = False
        self.saved_count = 0
        self.done        = False
        self.user_id     = ""
        self.user_name   = ""
        self.serial      = 0
        self._detector   = cv2.CascadeClassifier(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "attendance_system",
                "haarcascade_frontalface_default.xml"
            )
        )

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame = img

        # Passively collect frames when capturing flag is set
        if self.capturing and not self.done:
            gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self._detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                self.saved_count += 1
                img_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "attendance_system", "TrainingImage",
                    f"{self.user_name}.{self.serial}.{self.user_id}.{self.saved_count}.jpg"
                )
                cv2.imwrite(img_path, gray[y:y + h, x:x + w])

            if self.saved_count >= 100:
                self.capturing = False
                self.done      = True

                # Save to CSV once done — FIX: no empty padding columns
                csv_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "attendance_system", "DriverDetails", "DriverDetails.csv"
                )
                row = [self.serial, self.user_id, self.user_name]
                file_exists = os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0

                with open(csv_path, 'a+', newline='', encoding='utf-8-sig') as f:
                    import csv
                    writer = csv.writer(f)

                    if not file_exists:
                        writer.writerow(['SERIAL NO.', 'ID', 'NAME'])  # ✅ ADD HEADER

                    writer.writerow(row)

        return frame

if "show_camera" not in st.session_state:
   st.session_state.show_camera = False

with tab1:
   st.markdown("""
   <div style="
       background-color:#f0f6ff;
       padding:15px;
       border-radius:10px;
       border-left:5px solid #4a90e2;
       margin-bottom:15px;
   ">
   <h4 style="margin-bottom:8px;">📘 How Registration Works</h4>
   <p style="margin:0; font-size:14px;">
   To enable automatic attendance tracking, you must first register the driver's face.<br><br>

   <strong>Steps:</strong><br>
   1️⃣ Enter Driver ID and Name<br>
   2️⃣ Click <strong>"Open Camera"</strong> and then the <strong>"Start"</strong> button to start the camera<br>
   3️⃣ Click <strong>"Start Capturing"</strong> to collect face images<br>
   4️⃣ Wait for a few seconds then click <strong>"Train Model"</strong> to save the profile<br><br>

   ⚠️ <strong>Note:</strong> Keep your face clearly visible during capture. The system collects the face samples automatically.
   </p>
   </div>
   """, unsafe_allow_html=True)
   st.subheader("Register Driver")
   registration_count_placeholder = st.empty()

   user_id   = st.text_input("Enter ID")
   user_name = st.text_input("Enter Name")

   st.info("📸 Camera is live. Click 'Start Capturing' to begin registering your face.")

   if st.button("Open Camera"):
       if user_id and user_name:
           st.session_state.show_camera = True
       else:
           st.warning("Enter ID and Name")

   if st.session_state.show_camera:
       st.markdown("### Camera Active")

       RTC_CONFIG = RTCConfiguration({
           "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
       })

       ctx = webrtc_streamer(
           key="register",
           video_processor_factory=RegisterProcessor,
           rtc_configuration=RTC_CONFIG,
           media_stream_constraints={"video": True, "audio": False},
       )

       if st.button("Start Capturing"):
           if ctx and ctx.video_processor:
               if user_id and user_name:
                   vp = ctx.video_processor
                   # Configure the processor then flip the flag — no blocking loop
                   success, msg = TakeImages(user_id, user_name, vp)
                   if success:
                       st.info(msg + " — keep your face visible to the camera...")
                   else:
                       st.error(msg)
               else:
                   st.warning("Enter ID and Name first")
           else:
               st.warning("Camera not ready yet — wait a moment and try again")

       # Poll: show success once recv() finishes collecting 100 frames
       if ctx and ctx.video_processor and ctx.video_processor.done:
           st.success(f"✅ Images captured for {user_name}! Now click 'Train Model'.")
           st.session_state.show_camera = False

   if st.button("Train Model"):
       success, msg = TrainImages(reg_placeholder=registration_count_placeholder)
       if success:
           st.success(msg)
       else:
           st.error(msg)

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
   class DriverMonitorProcessor(VideoProcessorBase):
       def __init__(self):
           self.frame_count = 0
           self.attendance_done = False
           self.alerts_flags = {"phone": False, "drowsy": False, "attendance": False}
           self.drowsy_start = None

       def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
           try:
               img = frame.to_ndarray(format="bgr24")
               self.frame_count += 1

               # PHASE 1: ATTENDANCE ONLY
               if not self.attendance_done:
                   if self.frame_count % 10 == 0:  # process every 10 frames
                       try:
                           small_img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
                           result = process_attendance_frame(small_img, self.alerts_flags)
                           if isinstance(result, tuple) and len(result) == 2:
                               img, status = result
                           else:
                               img = result
                               status = {}
                           status = status or {}

                           if status.get("recognized", False):
                               self.attendance_done = True
                               stats["attendance_logged"] = True
                               print("✅ Attendance marked successfully")
                           elif self.frame_count > 200:
                               self.attendance_done = True
                               print("⚠️ Attendance not recognized, moving to monitoring")
                       except Exception as e:
                           print("⚠️ Attendance error:", e)

                   # Return frame only for attendance phase
                   return av.VideoFrame.from_ndarray(img, format="bgr24")

               # PHASE 2: MONITORING (Phone + Drowsiness)
               if self.frame_count % 3 == 0:  # throttle monitoring for performance
                   try:
                       # Downscale for speed
                       small_img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

                       # Phone detection
                       small_img = process_phone_frame(small_img, self.alerts_flags)

                       # Drowsiness detection
                       small_img, drowsy_status = process_drowsiness_frame(small_img, self.alerts_flags)

                       if drowsy_status and drowsy_status.get("drowsy", False):
                           if self.drowsy_start is None:
                               self.drowsy_start = time.time()
                       else:
                           if self.drowsy_start is not None:
                               stats["drowsy_time"] += time.time() - self.drowsy_start
                               self.drowsy_start = None

                       # Upscale back to original for display
                       img = cv2.resize(small_img, (img.shape[1], img.shape[0]))
                   except Exception as e:
                       print("⚠️ Monitoring error:", e)

               return av.VideoFrame.from_ndarray(img, format="bgr24")

           except Exception as e:
               print("Error in recv:", e)
               return frame

   # ICE/STUN config
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

       vp = webrtc_ctx.video_processor

       if vp and getattr(vp, "drowsy_start", None) is not None:
           stats["drowsy_time"] += time.time() - vp.drowsy_start

   elif stats.get("start_time") and not stats.get("end_time"):
       stats["end_time"] = time.time()
       render_metrics()

   # SESSION REPORT
   duration = 0.0
   if stats.get("start_time") and stats.get("end_time"):
       duration = stats["end_time"] - stats["start_time"]

   attendance_text = "Yes" if stats['attendance_logged'] else "No"
   attendance_color = "#16a34a" if stats['attendance_logged'] else "#dc2626"

   phone_time = f"{stats['phone_time']:.1f}"
   drowsy_time = f"{stats['drowsy_time']:.1f}"
   duration_s = f"{duration:.1f}"

   html = f"""
   <div style="padding:0.5rem 0;">
     <div style="display:flex;align-items:center;gap:10px;margin-bottom:1.25rem;">
       <div style="width:3px;height:20px;background:#2563eb;border-radius:2px;"></div>
       <span style="font-size:15px;font-weight:600;color:#1a2b4a;">Session Report</span>
       <span style="margin-left:auto;font-size:12px;color:#6b84a8;background:#eef3fb;border:1px solid #dbe6f5;border-radius:6px;padding:3px 10px;">Completed</span>
     </div>
     <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px;">
       <div style="background:#f0f4f9;border-radius:8px;padding:1rem;">
         <p style="font-size:11px;color:#6b84a8;margin:0 0 6px 0;text-transform:uppercase;letter-spacing:0.06em;">Phone usage</p>
         <p style="font-size:26px;font-weight:600;margin:0;color:#2563eb;">{phone_time}</p>
         <p style="font-size:12px;color:#9bacc4;margin:4px 0 0 0;">seconds detected</p>
       </div>
       <div style="background:#f0f4f9;border-radius:8px;padding:1rem;">
         <p style="font-size:11px;color:#6b84a8;margin:0 0 6px 0;text-transform:uppercase;letter-spacing:0.06em;">Drowsiness</p>
         <p style="font-size:26px;font-weight:600;margin:0;color:#dc2626;">{drowsy_time}</p>
         <p style="font-size:12px;color:#9bacc4;margin:4px 0 0 0;">seconds detected</p>
       </div>
       <div style="background:#f0f4f9;border-radius:8px;padding:1rem;">
         <p style="font-size:11px;color:#6b84a8;margin:0 0 6px 0;text-transform:uppercase;letter-spacing:0.06em;">Session duration</p>
         <p style="font-size:26px;font-weight:600;margin:0;color:#1a2b4a;">{duration_s}</p>
         <p style="font-size:12px;color:#9bacc4;margin:4px 0 0 0;">seconds total</p>
       </div>
       <div style="background:#f0f4f9;border-radius:8px;padding:1rem;">
         <p style="font-size:11px;color:#6b84a8;margin:0 0 6px 0;text-transform:uppercase;letter-spacing:0.06em;">Attendance</p>
         <p style="font-size:26px;font-weight:600;margin:0;color:{attendance_color};">{attendance_text}</p>
         <p style="font-size:12px;color:#9bacc4;margin:4px 0 0 0;">logged status</p>
       </div>
     </div>
     <div style="border-top:1px solid #dbe6f5;padding-top:10px;display:flex;align-items:center;gap:6px;">
       <div style="width:6px;height:6px;border-radius:50%;background:#16a34a;"></div>
       <span style="font-size:12px;color:#9bacc4;">Report generated at end of session</span>
     </div>
   </div>
   """

   st.markdown(html, unsafe_allow_html=True)

# TAB 3: ADMIN (Attendance CSV Viewer)
with tab3:
   st.subheader("📋 Admin: Attendance Logs")

   BASE_DIR = os.path.dirname(os.path.abspath(__file__))
   attendance_dir = os.path.join(BASE_DIR, "attendance_system", "Attendance")

   # Auto-create the folder if it doesn't exist yet
   os.makedirs(attendance_dir, exist_ok=True)

   files = sorted(
       [f for f in os.listdir(attendance_dir) if f.endswith(".csv")],
       reverse=True
   )
   if not files:
       st.info("No attendance records found yet.")
   else:
       selected_file = st.selectbox("Select Attendance File", files)
       file_path = os.path.join(attendance_dir, selected_file)

       try:
           df = pd.read_csv(file_path, encoding='utf-8-sig')
           df.columns = df.columns.str.strip()
           st.success(f"Showing: {selected_file}")
           st.dataframe(df, use_container_width=True)
       except Exception as e:
           st.error(f"Failed to load file: {e}")

   # Quick View (Latest Attendance)
   st.markdown("### 🟢 Latest Attendance (Auto View)")

   files = sorted(
       [f for f in os.listdir(attendance_dir) if f.endswith(".csv")],
       reverse=True
   )
   if files:
       latest_file = os.path.join(attendance_dir, files[0])
       try:
           df_latest = pd.read_csv(latest_file, encoding='utf-8-sig')
           df_latest.columns = df_latest.columns.str.strip()
           st.write(f"Latest file: **{files[0]}**")
           st.dataframe(df_latest.tail(10), use_container_width=True)
       except Exception as e:
           st.error(f"Error reading latest file: {e}")
   else:
       st.info("No attendance records found yet.")

# TAB 4: ABOUT US
with tab4:
   st.markdown(
"""<div class="about-wrapper">

<h2 class="about-title">About Team Attenzen</h2>
<p class="about-subtitle">A student-led initiative building smarter, safer roads through AI.</p>

<div class="about-card">
<h3>The Problem We're Solving</h3>
<p>Driver fatigue and inattentiveness are among the leading causes of road accidents worldwide.
In commercial transport — taxis, public buses, trams, delivery trucks — drivers often work long
shifts with little oversight. Employers have no reliable way to know if a driver is alert, or
even present, at the wheel. Manual attendance systems are slow, error-prone, and easy to game.
<br><br>
<strong>Attenzen</strong> is our answer to that problem.</p>
</div>

<div class="about-card">
<h3>What This System Does</h3>
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
<h3>Real-World Applications</h3>
<p>While DATS is currently a proof of concept, the underlying technology is directly applicable to:</p>
<ul>
<li><strong>Ride-hailing & taxis</strong> — ensure drivers are who they claim to be and are alert throughout a trip.</li>
<li><strong>Public buses & trams</strong> — monitor fatigue on long urban routes and flag incidents automatically.</li>
<li><strong>Delivery & logistics</strong> — track driver hours accurately and reduce liability from distracted driving.</li>
<li><strong>School transport</strong> — give parents and operators peace of mind on every route.</li>
<li><strong>Long-haul trucking</strong> — combat highway fatigue, one of the deadliest risks in freight transport.</li>
</ul>
</div>

<h3 style="font-size: 1.3rem; font-weight: 700; margin-bottom: 1.2rem;">Meet the Team</h3>

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
and real-time identity verification with CSV-based logging. </p>
</div>
<div class="team-card">
<div class="team-emoji">👩‍🎓</div>
<h4>Minal Haque</h4>
<p class="team-role">Safety Detection & Alert Systems</p>
<p>Developed the drowsiness detection module,
combining facial landmark analysis and object detection to provide
real-time safety alerts and session-level behavioural reporting.Integrated the Attendance Subsystem into the final product.</p>
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
with open(logo_path, "rb") as f:
   logo_base64 = base64.b64encode(f.read()).decode()

st.markdown(f"""
<div style="text-align:center; margin-top:0.5px; margin-bottom:0.5px;">
   <img src="data:image/png;base64,{logo_base64}"
        style="height:300px; margin-bottom:5px;">
</div>
""", unsafe_allow_html=True)

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