# perception/yolov5.py
"""
This module loads a YOLOv5 model (e.g., yolov5m) for object detection,
focusing on pedestrians, vehicles, and buildings, and returns bounding boxes.
Designed for integration into the blind_nav_system perception pipeline.
"""

import torch
import cv2
import numpy as np

class YoloV5Detector:
    def __init__(self, model_size="yolov5m", conf_threshold=0.4):
        """
        Load YOLOv5 model via PyTorch Hub.

        Args:
            model_size (str): 'yolov5s', 'yolov5m', etc.
            conf_threshold (float): Minimum confidence to keep detections
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("ultralytics/yolov5", model_size, pretrained=True).to(self.device)
        self.model.eval()

        # Classes of interest (from COCO): 0 = person, 2 = car, 14 = bench (proxy for building)
        self.target_ids = [0, 2, 14]
        self.label_map = {0: "person", 2: "car", 14: "building"}
        self.conf_threshold = conf_threshold

        print(f"[INFO] YOLOv5 {model_size} model loaded on {self.device}")

    def detect(self, frame):
        """
        Run object detection on a frame.

        Args:
            frame (np.ndarray): BGR image from OpenCV

        Returns:
            list of dicts: [{'label': str, 'conf': float, 'bbox': (x1,y1,x2,y2)}, ...]
        """
        results = self.model(frame)
        detections = results.xyxy[0].cpu().numpy()

        filtered = []
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det
            cls_id = int(cls_id)
            if cls_id in self.target_ids and conf >= self.conf_threshold:
                label = self.label_map.get(cls_id, f"class_{cls_id}")
                filtered.append({
                    "label": label,
                    "conf": float(conf),
                    "bbox": (int(x1), int(y1), int(x2), int(y2))
                })

        return filtered

    def draw_detections(self, frame, objects):
        """
        Draw detection results onto a copy of the input image.

        Args:
            frame (np.ndarray): Original image
            objects (list): Output of detect()

        Returns:
            np.ndarray: Annotated image
        """
        annotated = frame.copy()
        for obj in objects:
            x1, y1, x2, y2 = obj["bbox"]
            label = f"{obj['label']} {obj['conf']:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        return annotated
