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

# Initialize YOLOv11 object detector with pretrained weights
detector = ObjectDetector('runs/detect/household_objects-batch32-v11-alldata_e200/weights/best.pt')

# Set minimum confidence threshold for detections (0-1 range)
CONFIDENCE_THRESHOLD = 0.5

# Initialize prediction smoothing with temporal filtering
smoother = PredictionSmoother(
    buffer_size=5,  # Number of frames to consider for smoothing
    confidence_threshold=CONFIDENCE_THRESHOLD,
    min_persistence=3  # Minimum frames an object must be detected to be considered valid
)

# Global camera instance for video capture
camera = None

def init_camera():
    """
    Initialize camera connection with fallback options.
    Tries multiple video sources in priority order until a working connection is established.
    Returns True if successful, False otherwise.
    """
    global camera
    
    # Video sources in priority order
    sources = [
        0,  # Primary system camera/webcam
        'http://localhost:8080/video',  # Local network camera stream
        'rtsp://username:password@camera-ip:554/stream1',  # RTSP network stream
        '/dev/video0'  # Direct Linux video device
    ]
    
    for source in sources:
        try:
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                # Verify camera works by reading a test frame
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
    Get existing camera instance or initialize a new one.
    Returns camera object if available, None otherwise.
    """
    global camera
    if camera is None or not camera.isOpened():
        if not init_camera():
            return None
    return camera

def generate_frames():
    """
    Generator function that yields video frames with object detection visualization.
    Implements real-time object detection, tracking and annotation of video stream.
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
            # Perform object detection on current frame
            detections = detector.detect_objects(frame)
            
            # Apply temporal smoothing to reduce jitter
            smoothed_detections = smoother.update(detections)
            
            # Create annotated frame with detection visualizations
            annotated_frame = frame.copy()
            for det in smoothed_detections:
                x1, y1, x2, y2 = [int(coord) for coord in det['bbox']]
                
                # Create semi-transparent bounding box fill
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
                cv2.addWeighted(overlay, 0.2, annotated_frame, 0.8, 0, annotated_frame)
                
                # Add multi-layer border for 3D effect
                thickness = 2
                alpha = 0.3
                for i in range(3):
                    cv2.rectangle(annotated_frame, 
                                (x1-i, y1-i), (x2+i, y2+i), 
                                (0, 255, 0), 
                                thickness)
                    
                # Add floating confidence score badge
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
                
                # Add confidence score text
                cv2.putText(annotated_frame, 
                          conf_text,
                          (badge_x1 + badge_padding, badge_y2 - badge_padding),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.6,
                          (255, 255, 255),
                          2)
                
                # Add class label below bounding box
                label = det['class_name']
                cv2.putText(annotated_frame,
                          label,
                          (x1, y2 + 25),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.7,
                          (0, 255, 0),
                          2)
            
            # Convert annotated frame to JPEG bytes
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()
            
            # Yield frame in MJPEG stream format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            print(f"Error in generate_frames: {e}")
            continue

@app.route('/')
def index():
    """Serve main application page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Stream video feed endpoint"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect', methods=['POST'])
def detect():
    """
    Endpoint for single-frame object detection.
    Returns detection results as JSON.
    """
    cam = get_camera()
    if cam is None:
        return jsonify({'error': 'No camera available'})
        
    success, frame = cam.read()
    if not success:
        return jsonify({'error': 'Failed to capture frame'})
    
    try:
        # Perform detection with position estimation
        detections = detector.detect_objects(frame)
            
        # Apply confidence threshold filter
        detections = [d for d in detections if d['confidence'] >= CONFIDENCE_THRESHOLD]
        
        # Apply temporal smoothing
        smoothed_detections = smoother.update(detections)
        
        return jsonify({'detections': smoothed_detections})
    except Exception as e:
        return jsonify({'error': f'Detection failed: {str(e)}'})

@app.route('/camera/status')
def camera_status():
    """
    Get current camera status.
    Returns availability and frame position information.
    """
    cam = get_camera()
    return jsonify({
        'available': cam is not None and cam.isOpened(),
        'index': camera.get(cv2.CAP_PROP_POS_FRAMES) if camera else None
    })

@app.route('/update_threshold/<float:threshold>', methods=['POST'])
def update_threshold(threshold):
    """Update confidence threshold for detections"""
    global CONFIDENCE_THRESHOLD
    CONFIDENCE_THRESHOLD = threshold / 100  # Convert percentage to decimal
    return jsonify({'status': 'success', 'new_threshold': CONFIDENCE_THRESHOLD})

def cleanup():
    """Release camera resources on application shutdown"""
    global camera
    if camera is not None:
        camera.release()

atexit.register(cleanup)

@app.route('/camera/check', methods=['GET'])
def check_camera():
    """
    Detailed camera diagnostic endpoint.
    Returns camera properties and capabilities if available.
    """
    cam = get_camera()
    
    if cam is None:
        return jsonify({
            'status': 'error',
            'message': 'No camera available',
            'available': False
        })
    
    try:
        # Get detailed camera specifications
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
    # Configure server for production deployment
    port = int(os.environ.get('PORT', 5000))
    
    app.run(
        host='0.0.0.0',  # Enable external network access
        port=port,
        debug=False,  # Disable debug mode for security
        threaded=True  # Enable multi-threading for concurrent connections
    ) 