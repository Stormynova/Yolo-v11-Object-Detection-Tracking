# Import required libraries and modules
from flask import Flask, render_template, Response, jsonify  # Web framework and response types
import cv2  # OpenCV for image processing
from inference import ObjectDetector  # Custom object detection module
import numpy as np  # Numerical computing
import atexit  # Register cleanup functions
from collections import deque  # Efficient queue data structure
from typing import Dict, List, Optional  # Type hints
import time  # Time utilities
from dataclasses import dataclass  # Data class decorator
from prediction_smoother import PredictionSmoother  # Custom detection smoothing
import os  # Operating system interface

# Initialize Flask application
app = Flask(__name__)

# Create object detector instance using trained model weights
detector = ObjectDetector('runs/detect/household_objects-batch32-v11-alldata_e200/weights/best.pt')

# Set minimum confidence threshold for valid detections
CONFIDENCE_THRESHOLD = 0.5  # Only keep detections above 50% confidence

# Initialize prediction smoothing to reduce jitter and false positives
smoother = PredictionSmoother(
    buffer_size=5,  # Keep track of last 5 frames
    confidence_threshold=CONFIDENCE_THRESHOLD,  # Use same threshold as detector
    min_persistence=3  # Object must appear in 3 consecutive frames
)

# Global video capture object
camera = None

def init_camera():
    """
    Initialize video capture with fallback sources.
    
    Attempts to connect to cameras in priority order:
    1. System webcam
        ... add more
    
    Returns:
        bool: True if connection successful, False otherwise
    """
    global camera
    
    # List of video sources to try in order
    sources = [
        0,  # System webcam (default camera)
    ]
    
    # Try each source until we find a working camera
    for source in sources:
        try:
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                # Validate camera works by reading a test frame
                ret, _ = cap.read()
                if ret:
                    camera = cap
                    print(f"Successfully connected to camera source: {source}")
                    return True
                cap.release()
        except Exception as e:
            print(f"Failed to connect to source {source}: {e}")
            continue
            
    print("No camera sources available")
    return False

def get_camera():
    """
    Retrieve or initialize camera connection.
    
    Returns:
        cv2.VideoCapture: Active camera instance if available
        None: If no camera can be initialized
    """
    global camera
    # Check if camera needs to be initialized
    if camera is None or not camera.isOpened():
        if not init_camera():
            return None
    return camera

def generate_frames():
    """
    Stream video frames with real-time object detection.
    
    Performs:
    - Object detection on each frame
    - Temporal smoothing of detections
    - Visualization with bounding boxes and labels
    - MJPEG stream encoding
    
    Yields:
        bytes: Encoded video frame with annotations
    """
    while True:
        # Get camera instance
        cam = get_camera()
        if cam is None:
            yield b''
            continue
            
        # Read frame from camera
        success, frame = cam.read()
        if not success:
            continue
        
        try:
            # Run object detection on frame
            detections = detector.detect_objects(frame)
            
            # Filter out low confidence detections
            filtered_detections = [
                det for det in detections 
                if det['confidence'] >= CONFIDENCE_THRESHOLD
            ]
            
            # Apply temporal smoothing
            smoothed_detections = smoother.update(filtered_detections)
            
            # Create visualization frame
            annotated_frame = frame.copy()
            
            # Draw detections
            for det in smoothed_detections:
                # Get bounding box coordinates
                x1, y1, x2, y2 = [int(coord) for coord in det['bbox']]
                
                # Draw semi-transparent box fill
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
                cv2.addWeighted(overlay, 0.2, annotated_frame, 0.8, 0, annotated_frame)
                
                # Draw 3D border effect
                thickness = 2
                alpha = 0.3
                for i in range(3):
                    cv2.rectangle(annotated_frame, 
                                (x1-i, y1-i), (x2+i, y2+i), 
                                (0, 255, 0), 
                                thickness)
                    
                # Draw confidence score badge
                conf_text = f"{det['confidence']:.2f}"
                text_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Calculate badge dimensions
                badge_padding = 5
                badge_x1 = x1
                badge_y1 = y1 - text_size[1] - 2 * badge_padding
                badge_x2 = badge_x1 + text_size[0] + 2 * badge_padding
                badge_y2 = y1
                
                # Draw badge background
                cv2.rectangle(annotated_frame, 
                            (badge_x1, badge_y1), 
                            (badge_x2, badge_y2), 
                            (0, 200, 0), 
                            -1)
                
                # Draw confidence score text
                cv2.putText(annotated_frame, 
                          conf_text,
                          (badge_x1 + badge_padding, badge_y2 - badge_padding),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.6,
                          (255, 255, 255),
                          2)
                
                # Draw class label
                label = det['class_name']
                cv2.putText(annotated_frame,
                          label,
                          (x1, y2 + 25),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.7,
                          (0, 255, 0),
                          2)
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()
            
            # Yield frame in MJPEG format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                   
        except Exception as e:
            print(f"Frame generation error: {e}")
            continue

@app.route('/')
def index():
    """Serve the main web interface"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Stream live video with object detection"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect', methods=['POST'])
def detect():
    """
    Process single frame object detection.
    
    Returns:
        JSON containing detection results or error message
    """
    # Get camera instance
    cam = get_camera()
    if cam is None:
        return jsonify({'error': 'No camera available'})
        
    # Read single frame
    success, frame = cam.read()
    if not success:
        return jsonify({'error': 'Failed to capture frame'})
    
    try:
        # Run detection pipeline
        detections = detector.detect_objects(frame)
        filtered_detections = [d for d in detections if d['confidence'] >= CONFIDENCE_THRESHOLD]
        smoothed_detections = smoother.update(filtered_detections)
        
        return jsonify({'detections': smoothed_detections})
    except Exception as e:
        return jsonify({'error': f'Detection failed: {str(e)}'})

@app.route('/camera/status')
def camera_status():
    """
    Query camera connection status.
    
    Returns:
        JSON with camera availability and frame info
    """
    cam = get_camera()
    return jsonify({
        'available': cam is not None and cam.isOpened(),
        'index': camera.get(cv2.CAP_PROP_POS_FRAMES) if camera else None
    })

@app.route('/update_threshold/<float:threshold>', methods=['POST'])
def update_threshold(threshold):
    """
    Update detection confidence threshold.
    
    Args:
        threshold: New threshold value (0-100)
        
    Returns:
        JSON confirmation with new threshold
    """
    global CONFIDENCE_THRESHOLD
    CONFIDENCE_THRESHOLD = threshold / 100  # Convert percentage to decimal
    return jsonify({'status': 'success', 'new_threshold': CONFIDENCE_THRESHOLD})

def cleanup():
    """Release camera resources before shutdown"""
    global camera
    if camera is not None:
        camera.release()

# Register cleanup function to run on shutdown
atexit.register(cleanup)

@app.route('/camera/check', methods=['GET'])
def check_camera():
    """
    Detailed camera diagnostics.
    
    Returns:
        JSON containing:
        - Connection status
        - Resolution and FPS
        - Backend driver info
        - Error details if applicable
    """
    cam = get_camera()
    
    if cam is None:
        return jsonify({
            'status': 'error',
            'message': 'No camera available',
            'available': False
        })
    
    try:
        # Get camera properties
        props = {
            'width': int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': int(cam.get(cv2.CAP_PROP_FPS)),
            'backend': cam.getBackendName()
        }
        
        return jsonify({
            'status': 'success',
            'available': True,
            'properties': props
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'available': False
        })

if __name__ == '__main__':
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 5000))
    
    # Start Flask development server
    app.run(
        host='0.0.0.0',  # Allow external connections
        port=port,
        debug=False,  # Production mode
        threaded=True  # Enable request concurrency
    ) 