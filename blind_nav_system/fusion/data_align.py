
from sensors.camera_module import CameraModule
from sensors.pseudo_lidar2d import PseudoLidar2D

class InputAligner:
    def __init__(self):
        self.camera = CameraModule(preview=False)
        self.lidar = PseudoLidar2D()

    def get_aligned_inputs(self):
        """
        Capture a single frame and generate its corresponding scan.

        Returns:
            tuple: (RGB frame, scan_data) where scan_data = [(angle, distance), ...]
        """
        frame = self.camera.get_frame()
        scan_data = self.lidar.scan(frame)
        return frame, scan_data

    def release(self):
        self.camera.release()
