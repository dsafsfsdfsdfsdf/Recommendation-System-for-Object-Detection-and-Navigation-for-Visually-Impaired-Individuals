import cv2
import numpy as np
from sensors.camera_module import CameraModule
import time

class PseudoLidar2D:
    def __init__(self, scan_line_ratio=0.6, fov_deg=60):
        """
        Simulate a 2D LiDAR by scanning a horizontal line in the image.

        Args:
            scan_line_ratio (float): Relative height (0~1) of scan line from top.
            fov_deg (float): Simulated horizontal field of view (degrees).
        """
        self.scan_line_ratio = scan_line_ratio
        self.fov_rad = np.deg2rad(fov_deg)

    def scan(self, frame):
        """
        Perform a 2D pseudo-lidar scan by sampling pixel brightness along one row.

        Args:
            frame (np.ndarray): BGR image from camera.

        Returns:
            List of (angle, normalized distance) tuples.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        y = int(h * self.scan_line_ratio)

        scan_line = gray[y, :]
        scan_line = cv2.GaussianBlur(scan_line, (5, 1), 0)  # Smooth noise
        distances = 1.0 - (scan_line.astype(np.float32) / 255.0)  # Inverted brightness
        distances = np.clip(distances, 0.0, 1.0)

        angles = np.linspace(-self.fov_rad/2, self.fov_rad/2, w)
        scan_data = list(zip(angles, distances))

        return scan_data

    def visualize_scan(self, frame, scan_data):
        """
        Draw a bar visualization of the pseudo-LiDAR scan.

        Args:
            frame (np.ndarray): Original camera frame.
            scan_data (list): List of (angle, distance) pairs.
        """
        vis = frame.copy()
        h, w, _ = vis.shape
        y = int(h * self.scan_line_ratio)

        for i, (_, dist) in enumerate(scan_data):
            bar_len = int(dist * 50)
            cv2.line(vis, (i, y), (i, y - bar_len), (0, 255, 255), 1)

        cv2.imshow("Pseudo LiDAR 2D Scan", vis)
        cv2.waitKey(1)

# --- Direct run here ---
if __name__ == "__main__":
    cam = CameraModule(preview=False)
    lidar2d = PseudoLidar2D()

    print("[INFO] Starting live 2D LiDAR scan. Press Ctrl+C to stop.")

    try:
        while True:
            frame = cam.get_frame()
            scan_data = lidar2d.scan(frame)
            lidar2d.visualize_scan(frame, scan_data)
            time.sleep(0.5)  # Control refresh rate
    except KeyboardInterrupt:
        print("[INFO] Exiting...")
        cam.release()
        cv2.destroyAllWindows()
