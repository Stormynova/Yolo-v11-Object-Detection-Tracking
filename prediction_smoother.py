from collections import deque
from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass
import time

@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2] coordinates
    distance: float    # Distance from camera in meters
    angle: float      # Angle from camera center in degrees
    timestamp: float  # Unix timestamp of detection

class PredictionSmoother:
    def __init__(self, 
                 buffer_size: int = 5, 
                 confidence_threshold: float = 0.25,
                 min_persistence: int = 3):
        """
        Initialize prediction smoothing with temporal filtering.
        
        Args:
            buffer_size: Number of frames to store in history buffer
            confidence_threshold: Minimum confidence score to consider a detection valid
            min_persistence: Number of consecutive frames object must appear to be considered stable
        """
        self.buffer_size = buffer_size
        self.confidence_threshold = confidence_threshold
        self.min_persistence = min_persistence
        self.detection_buffer = deque(maxlen=buffer_size)
        
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Process new detections and apply temporal smoothing.
        
        Args:
            detections: List of detection dictionaries containing class, confidence, bbox etc.
            
        Returns:
            List of smoothed/filtered detections that have persisted across multiple frames
        """
        current_time = time.time()
        
        # Convert raw detections into Detection objects and filter by confidence
        formatted_detections = [
            Detection(
                class_name=det['class_name'],
                confidence=det['confidence'],
                bbox=det['bbox'],
                distance=det['distance'],
                angle=det['angle'],
                timestamp=current_time
            )
            for det in detections
            if det['confidence'] >= self.confidence_threshold
        ]
        
        self.detection_buffer.append(formatted_detections)
        
        # Get detections that have persisted across min_persistence frames
        persistent_detections = self._get_persistent_detections()
        
        # Convert Detection objects back to dictionary format
        return [
            {
                'class_name': det.class_name,
                'confidence': det.confidence,
                'bbox': det.bbox,
                'distance': det.distance,
                'angle': det.angle
            }
            for det in persistent_detections
        ]
    
    def _get_persistent_detections(self) -> List[Detection]:
        """
        Identify detections that have persisted across multiple frames.
        
        Returns:
            List of Detection objects that appear in min_persistence consecutive frames
        """
        if len(self.detection_buffer) < self.min_persistence:
            return []
        
        # Track how many times each object class appears
        class_counts = {}
        
        # Only examine the most recent frames up to min_persistence
        recent_frames = list(self.detection_buffer)[-self.min_persistence:]
        
        for frame_detections in recent_frames:
            for det in frame_detections:
                class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1
        
        # Filter for classes that appear in enough consecutive frames
        persistent_classes = {
            cls: count for cls, count in class_counts.items()
            if count >= self.min_persistence
        }
        
        if not persistent_classes:
            return []
        
        # Return the most recent detections for objects that have persisted
        latest_detections = recent_frames[-1]
        return [
            det for det in latest_detections
            if det.class_name in persistent_classes
        ]