from flask import Flask, render_template, Response, jsonify
import cv2
from inference import ObjectDetector
import numpy as np
import atexit
from collections import deque
from typing import Dict, List, Optional
import time
from dataclasses import dataclass
from prediction_smoother import PredictionSmoother
import os

app = Flask(__name__)

# Initialize object detector with optimized model weights
detector = ObjectDetector('runs/detect/household_objects-batch32-v11-alldata_e200/weights/best.pt')

# Confidence threshold for filtering weak detections (0-1)
CONFIDENCE_THRESHOLD = 0.5

# Configure temporal smoothing to reduce detection jitter
smoother = PredictionSmoother(
    buffer_size=5,  # Frame history size
    confidence_threshold=CONFIDENCE_THRESHOLD,
    min_persistence=3  # Required consecutive detections
)

# Global video capture device
camera = None

def init_camera():
    """
    Initialize video capture with fallback sources.
    
    Attempts to connect to cameras in priority order:
    1. System webcam
    2. Local network stream
    3. RTSP stream
    4. Linux video device
    
    Returns:
        bool: True if connection successful, False otherwise
    """
    global camera
    
    # Prioritized video sources
    sources = [
        0,  # System webcam
    ]
    
    for source in sources:
        try:
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                # Validate connection with test frame
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
        cam = get_camera()
        if cam is None:
            yield b''
            continue
            
        success, frame = cam.read()
        if not success:
            continue
        
        try:
            # Run detection pipeline
            detections = detector.detect_objects(frame)
            smoothed_detections = smoother.update(detections)
            
            # Render detection visualizations
            annotated_frame = frame.copy()
            for det in smoothed_detections:
                x1, y1, x2, y2 = [int(coord) for coord in det['bbox']]
                
                # Semi-transparent bounding box
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
                cv2.addWeighted(overlay, 0.2, annotated_frame, 0.8, 0, annotated_frame)
                
                # 3D border effect
                thickness = 2
                alpha = 0.3
                for i in range(3):
                    cv2.rectangle(annotated_frame, 
                                (x1-i, y1-i), (x2+i, y2+i), 
                                (0, 255, 0), 
                                thickness)
                    
                # Confidence score badge
                conf_text = f"{det['confidence']:.2f}"
                text_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                badge_padding = 5
                badge_x1 = x1
                badge_y1 = y1 - text_size[1] - 2 * badge_padding
                badge_x2 = badge_x1 + text_size[0] + 2 * badge_padding
                badge_y2 = y1
                
                cv2.rectangle(annotated_frame, 
                            (badge_x1, badge_y1), 
                            (badge_x2, badge_y2), 
                            (0, 200, 0), 
                            -1)
                
                cv2.putText(annotated_frame, 
                          conf_text,
                          (badge_x1 + badge_padding, badge_y2 - badge_padding),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.6,
                          (255, 255, 255),
                          2)
                
                # Object class label
                label = det['class_name']
                cv2.putText(annotated_frame,
                          label,
                          (x1, y2 + 25),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.7,
                          (0, 255, 0),
                          2)
            
            # Encode frame for streaming
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            print(f"Error in generate_frames: {e}")
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
    cam = get_camera()
    if cam is None:
        return jsonify({'error': 'No camera available'})
        
    success, frame = cam.read()
    if not success:
        return jsonify({'error': 'Failed to capture frame'})
    
    try:
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
    # Production server configuration
    port = int(os.environ.get('PORT', 5000))
    
    app.run(
        host='0.0.0.0',  # Allow external connections
        port=port,
        debug=False,  # Production mode
        threaded=True  # Enable request concurrency
    ) 