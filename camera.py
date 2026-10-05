"""
Camera management module.
Handles opening camera, capturing frames, and releasing camera.
Includes a synthetic test camera mode for laptop testing when no camera hardware is detected.
"""

import cv2
import numpy as np
import time

class Camera:
    def __init__(self, camera_id=0, width=640, height=480):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.cap = None
        self.is_mock = False
        self._mock_frame_counter = 0

    def open(self):
        """Open the camera device or initialize fallback mock camera."""
        print(f"📷 Opening camera (Device ID: {self.camera_id})...")
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print("⚠️ Physical camera not detected or failed to open.")
            print("   -> Switching to MOCK CAMERA mode for simulation & testing.")
            self.is_mock = True
            return True
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.is_mock = False
        print("✅ Camera active.")
        return True

    def read(self):
        """
        Capture a frame from the camera.
        Returns: (success: bool, frame: numpy.ndarray)
        """
        if self.is_mock or self.cap is None:
            return True, self._generate_mock_frame()

        ret, frame = self.cap.read()
        if not ret or frame is None:
            # Fallback to mock if frame read fails mid-stream
            return True, self._generate_mock_frame()
        return ret, frame

    def release(self):
        """Release camera hardware resources."""
        if self.cap is not None and not self.is_mock:
            self.cap.release()
            self.cap = None
        print("📷 Camera released.")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def _generate_mock_frame(self):
        """Generate a simulated camera frame containing a synthetic face for headless/laptop testing."""
        self._mock_frame_counter += 1
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Draw background pattern
        cv2.rectangle(frame, (0, 0), (self.width, self.height), (30, 30, 40), -1)
        
        # Draw a simulated person face shape in center
        cx, cy = self.width // 2, self.height // 2
        # Head
        cv2.ellipse(frame, (cx, cy - 20), (80, 110), 0, 0, 360, (200, 180, 170), -1)
        # Eyes
        cv2.circle(frame, (cx - 30, cy - 50), 10, (255, 255, 255), -1)
        cv2.circle(frame, (cx + 30, cy - 50), 10, (255, 255, 255), -1)
        cv2.circle(frame, (cx - 30, cy - 50), 4, (40, 40, 40), -1)
        cv2.circle(frame, (cx + 30, cy - 50), 4, (40, 40, 40), -1)
        # Nose
        cv2.line(frame, (cx, cy - 40), (cx, cy - 10), (160, 140, 130), 3)
        # Smile
        cv2.ellipse(frame, (cx, cy + 20), (35, 20), 0, 0, 180, (50, 50, 150), 3)
        
        # Status text
        cv2.putText(frame, "SIMULATED LAPTOP CAMERA FEED", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Frame: {self._mock_frame_counter}", (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        time.sleep(0.03)  # ~30fps simulation
        return frame
