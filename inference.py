import cv2
import torch
from ultralytics import YOLO
import numpy as np
from utils.distance_calculator import calculate_distance_and_angle

class ObjectDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        
    def detect(self, image):
        results = self.model(image, verbose=False)
        
        annotated_frame = results[0].plot()
        
        return annotated_frame
    
    def detect_objects(self, image):
        results = self.model(image)
        frame_width = image.shape[1]
        
        # print("Available classes:", self.model.names)
        
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0]
                cls = box.cls[0]
                class_name = self.model.names[int(cls)]
                # print(f"Detected class: {class_name}")  # Debug print
                
                bbox = [float(x1), float(y1), float(x2), float(y2)]
                distance, angle = calculate_distance_and_angle(
                    bbox, class_name, frame_width
                )
                                
                detections.append({
                    'bbox': bbox,
                    'confidence': float(conf),
                    'class_name': class_name,
                    'distance': distance,
                    'angle': angle
                })
                
        return detections 