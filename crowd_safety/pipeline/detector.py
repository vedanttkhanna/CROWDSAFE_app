"""
Person detection and tracking pipeline.
YOLOv8 detects people per frame.
DeepSORT assigns persistent IDs across frames.
Output: list of (track_id, x_centre, y_centre, bbox) per frame.
"""

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np

class CrowdTracker:
    def __init__(self, yolo_model="yolov8n.pt", max_age=30):
        self.detector = YOLO(yolo_model)   # downloads automatically first run
        self.tracker  = DeepSort(max_age=max_age)

    def process_frame(self, frame):
        """
        frame : numpy array (H, W, 3) BGR from OpenCV
        Returns list of dicts:
          { track_id, bbox: [x1,y1,x2,y2], centre: (cx, cy) }
        """
        results    = self.detector(frame, classes=[0], verbose=False)[0]
        detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf            = float(box.conf[0])
            if conf < 0.4:
                continue
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "person"))

        tracks = self.tracker.update_tracks(detections, frame=frame)
        output = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            x1, y1, x2, y2 = track.to_ltrb()
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            output.append({
                "track_id": track.track_id,
                "bbox":     [x1, y1, x2, y2],
                "centre":   (cx, cy),
            })

        return output