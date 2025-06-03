
from fusion.data_align import InputAligner
from fusion.simple_fusion import scan_to_bev
from fusion.bev_map import show_bev_map
from perception.yolov5 import YoloV5Detector
from navigation.simple_planner import astar_path_planning, draw_path_on_bev, simulate_and_speak_alerts
import time
import cv2
import numpy as np

def main():
    # Initialize sensors: camera + pseudo LiDAR
    aligner = InputAligner()
    detector = YoloV5Detector(model_size="yolov5m")

    print("[INFO] Blind Navigation System Started.")

    try:
        while True:
            # Step 1: Get frame and scan
            frame, scan = aligner.get_aligned_inputs()

            # Step 2: YOLO detection
            detections = detector.detect(frame)

            # Step 3: BEV from LiDAR
            bev_map = scan_to_bev(scan)

            # Step 4: Path planning
            bev_binary = (bev_map < 128).astype(np.uint8)
            h, w = bev_binary.shape
            start = (h - 1, 0)
            goal = (h - 1, w - 1)
            path = astar_path_planning(bev_binary, start, goal)

            # Step 5: Show BEV + path
            path_img = draw_path_on_bev(bev_binary, path)
            cv2.imshow("BEV Map with Path", path_img)

            # Step 6: Show YOLO detections
            debug_img = detector.draw_detections(frame, detections)
            cv2.imshow("YOLOv5 Detections", debug_img)

            # Step 7: Obstacle distance and voice alerts
            simulate_and_speak_alerts(bev_binary, path, scan)

            if cv2.waitKey(1) == 27:  # ESC
                break

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user. Exiting...")
        aligner.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
