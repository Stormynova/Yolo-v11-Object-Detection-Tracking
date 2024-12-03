from collections import deque
from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass
import time

@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: List[float]
    distance: float
    angle: float
    timestamp: float

class PredictionSmoother:
    def __init__(self, 
                 buffer_size: int = 5, 
                 confidence_threshold: float = 0.25,
                 min_persistence: int = 3):
        self.buffer_size = buffer_size
        self.confidence_threshold = confidence_threshold
        self.min_persistence = min_persistence
        self.detection_buffer = deque(maxlen=buffer_size)
        
    def update(self, detections: List[Dict]) -> List[Dict]:
        current_time = time.time()
        
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
        
        persistent_detections = self._get_persistent_detections()
        
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
        if len(self.detection_buffer) < self.min_persistence:
            return []
        
        class_counts = {}
        
        recent_frames = list(self.detection_buffer)[-self.min_persistence:]
        
        for frame_detections in recent_frames:
            for det in frame_detections:
                class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1
        
        persistent_classes = {
            cls: count for cls, count in class_counts.items()
            if count >= self.min_persistence
        }
        
        if not persistent_classes:
            return []
        
        latest_detections = recent_frames[-1]
        return [
            det for det in latest_detections
            if det.class_name in persistent_classes
        ]