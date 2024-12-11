# Import required libraries
import cv2
import torch
from ultralytics import YOLO
import numpy as np
from utils.distance_calculator import calculate_distance_and_angle

class ObjectDetector:
    """
    Class for detecting and tracking objects in images using YOLO model.
    """
    def __init__(self, model_path):
        """
        Initialize the object detector with a YOLO model.
        
        Args:
            model_path (str): Path to the trained YOLO model weights
        """
        self.model = YOLO(model_path)
        
    def detect(self, image):
        """
        Detect and track objects in a single image frame.
        
        Args:
            image (np.ndarray): Input image array
            
        Returns:
            np.ndarray: Annotated image with detection boxes and tracking info
        """
        # Track objects in image using YOLO model
        results = self.model.track(image, show=True)  # Tracking with default tracker
        
        # Get annotated frame with detection visualization
        annotated_frame = results[0].plot()
        
        return annotated_frame
    
    def detect_objects(self, image):
        """
        Detect objects and calculate their spatial properties.
        
        Args:
            image (np.ndarray): Input image array
            
        Returns:
            list: List of dictionaries containing detection info for each object:
                - bbox: Bounding box coordinates [x1,y1,x2,y2]
                - confidence: Detection confidence score
                - class_name: Detected object class
                - distance: Estimated distance to object
                - angle: Estimated angle to object
        """
        # Run object detection on image
        results = self.model(image)
        frame_width = image.shape[1]
        
        detections = []
        # Process each set of detections in results
        for r in results:
            boxes = r.boxes
            # Process each detected bounding box
            for box in boxes:
                # Extract box coordinates, confidence and class
                x1, y1, x2, y2 = box.xyxy[0]  # Get box coordinates in x1,y1,x2,y2 format
                conf = box.conf[0]  # Get confidence score
                cls = box.cls[0]  # Get class index
                class_name = self.model.names[int(cls)]  # Convert class index to name
                
                # Convert box coordinates to list of floats
                bbox = [float(x1), float(y1), float(x2), float(y2)]
                
                # Calculate spatial properties of detected object
                distance, angle = calculate_distance_and_angle(
                    bbox, class_name, frame_width
                )
                                
                # Store detection info in dictionary
                detections.append({
                    'bbox': bbox,
                    'confidence': float(conf),
                    'class_name': class_name,
                    'distance': distance,
                    'angle': angle
                })
                
        return detections 