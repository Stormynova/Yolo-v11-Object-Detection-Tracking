from collections import deque
from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass
import time

# Data class to store detection information in a structured format
@dataclass
class Detection:
    class_name: str      # Name/type of detected object (e.g. "cup", "bottle")
    confidence: float    # Model's confidence score for this detection (0-1)
    bbox: List[float]    # Bounding box coordinates [x1, y1, x2, y2] in pixels
    distance: float      # Estimated distance from camera to object in meters
    angle: float        # Angle between camera center and object in degrees
    timestamp: float    # Unix timestamp when detection occurred

class PredictionSmoother:
    def __init__(self, 
                 buffer_size: int = 5, 
                 confidence_threshold: float = 0.25,
                 min_persistence: int = 3):
        """
        Initialize prediction smoothing with temporal filtering.
        
        Maintains a buffer of recent detections and filters out unstable/noisy detections
        by requiring objects to persist across multiple consecutive frames.
        
        Args:
            buffer_size: Number of frames to store in history buffer
            confidence_threshold: Minimum confidence score to consider a detection valid
            min_persistence: Number of consecutive frames object must appear to be considered stable
        """
        self.buffer_size = buffer_size
        self.confidence_threshold = confidence_threshold
        self.min_persistence = min_persistence
        # Double-ended queue to store detection history, automatically drops old frames
        self.detection_buffer = deque(maxlen=buffer_size)
        
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Process new detections and apply temporal smoothing.
        
        Takes raw detections from object detector, filters by confidence threshold,
        and returns only detections that have persisted across multiple frames.
        This helps reduce false positives and jitter.
        
        Args:
            detections: List of detection dictionaries containing class, confidence, bbox etc.
            
        Returns:
            List of smoothed/filtered detections that have persisted across multiple frames
        """
        current_time = time.time()
        
        # Convert raw detections into Detection objects and filter by confidence threshold
        # This standardizes the detection format and removes low-confidence predictions
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
        
        # Add current frame's detections to history buffer
        self.detection_buffer.append(formatted_detections)
        
        # Get detections that have persisted across min_persistence frames
        persistent_detections = self._get_persistent_detections()
        
        # Convert Detection objects back to dictionary format for output
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
        
        Analyzes detection history to find objects that appear consistently.
        This helps filter out spurious detections while keeping stable ones.
        
        Returns:
            List of Detection objects that appear in min_persistence consecutive frames
        """
        # Return empty list if we don't have enough frame history yet
        if len(self.detection_buffer) < self.min_persistence:
            return []
        
        # Track how many times each object class appears in recent frames
        class_counts = {}
        
        # Only examine the most recent frames up to min_persistence
        # This creates a sliding window of frame history
        recent_frames = list(self.detection_buffer)[-self.min_persistence:]
        
        # Count occurrences of each object class across recent frames
        for frame_detections in recent_frames:
            for det in frame_detections:
                class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1
        
        # Filter for classes that appear in enough consecutive frames
        # This identifies which object classes are stable detections
        persistent_classes = {
            cls: count for cls, count in class_counts.items()
            if count >= self.min_persistence
        }
        
        if not persistent_classes:
            return []
        
        # Return the most recent detections for objects that have persisted
        # This gives us the current position/state of stable objects
        latest_detections = recent_frames[-1]
        return [
            det for det in latest_detections
            if det.class_name in persistent_classes
        ]