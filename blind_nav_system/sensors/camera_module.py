import cv2  # Import OpenCV library for video capturing and image processing

class CameraModule:
    """
    A class to manage camera operations using OpenCV.
    
    This class provides functionalities to initialize a camera, capture frames,
    display live preview (optional), and release resources properly.
    
    Attributes:
        cap (cv2.VideoCapture): VideoCapture object to interact with the camera.
        preview (bool): Flag indicating whether to show live video preview.
    """

    def __init__(self, camera_id=0, width=640, height=480, preview=True):
        """
        Initialize the camera with specified settings.

        :param camera_id: ID/index of the camera to open (default: 0).
                          Useful when multiple cameras are connected.
        :param width: Desired width of captured frames (default: 640 px).
        :param height: Desired height of captured frames (default: 480 px).
        :param preview: Whether to enable live preview window (default: False).
        """
        # Create a VideoCapture object for the specified camera index
        self.cap = cv2.VideoCapture(camera_id)
        
        # Set the desired frame width property of the camera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        
        # Set the desired frame height property of the camera
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Store the preview flag for later use in get_frame method
        self.preview = preview

        # Check if the camera was successfully opened
        if not self.cap.isOpened():
            # If camera failed to open, raise an IOError with descriptive message
            raise IOError("Cannot open camera")

        # Print initialization success message
        print("[INFO] Camera initialized.")

    def get_frame(self):
        """
        Capture and return a single frame from the camera.

        :return: Captured frame as a NumPy array.
        :raises IOError: If frame reading fails.
        """
        # Read a frame from the camera
        ret, frame = self.cap.read()
        
        # Check if frame was successfully captured
        if not ret:
            # If reading frame failed, raise an IOError
            raise IOError("Failed to read frame from camera")

        # If preview mode is enabled, display the frame
        if self.preview:
            # Show the frame in a window named "Camera Preview"
            cv2.imshow("Camera Preview", frame)
            
            # Wait for 1ms and check if 'q' key was pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                # If 'q' was pressed, release camera resources and exit program
                self.release()
                exit(0)

        # Return the captured frame
        return frame

    def release(self):
        """
        Release camera resources and close any OpenCV windows.
        """
        # Release the VideoCapture object
        self.cap.release()
        
        # Close all OpenCV windows
        cv2.destroyAllWindows()
        
        # Print resource release confirmation message
        print("[INFO] Camera released.")


# Main execution block (for testing purposes)
if __name__ == "__main__":
    # Create a CameraModule instance with preview enabled
    cam = CameraModule(preview=True)
    
    # Continuous frame capturing loop
    while True:
        # Get a frame from the camera
        frame = cam.get_frame()