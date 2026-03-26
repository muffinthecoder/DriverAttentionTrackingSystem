import cv2


class Camera:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None

    def start(self):
        """Initialize camera"""
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            raise RuntimeError("❌ Cannot open camera")

    def read_frame(self):
        """Read a single frame"""
        if self.cap is None:
            raise RuntimeError("Camera not started")

        ret, frame = self.cap.read()
        if not ret:
            return None

        return frame

    def release(self):
        """Release camera"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None


# ----------------------------
# HELPER FUNCTION (for Streamlit)
# ----------------------------

def get_rgb_frame(frame):
    """Convert BGR → RGB for Streamlit display"""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)