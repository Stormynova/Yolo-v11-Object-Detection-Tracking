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

detector = ObjectDetector('runs/detect/household_objects/weights/best.pt')

CONFIDENCE_THRESHOLD = 0.5  # 50% confidence threshold

smoother = PredictionSmoother(
    buffer_size=5,
    confidence_threshold=CONFIDENCE_THRESHOLD,
    min_persistence=3
)

camera = None

def init_camera():
    global camera
    
    # make this work with different camera sources (maybe like a deployed web app)
    source = 0
    try:
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            # Test reading a frame
            ret, _ = cap.read()
            if ret:
                camera = cap
                print(f"Successfully connected to camera source: {source}")
                return True
            cap.release()
    except Exception as e:
        print(f"Failed to connect to source {source}: {e}")
            
    print("No camera sources available")
    return False

def get_camera():
    """Get or initialize camera"""
    global camera
    if camera is None or not camera.isOpened():
        if not init_camera():
            return None
    return camera

def generate_frames():
    while True:
        cam = get_camera()
        if cam is None:
            yield b''
            continue
            
        success, frame = cam.read()
        if not success:
            continue
        
        try:
            detections = detector.detect_objects(frame)
            
            smoothed_detections = smoother.update(detections)
            
            annotated_frame = frame.copy()
            for det in smoothed_detections:
                x1, y1, x2, y2 = [int(coord) for coord in det['bbox']]
                
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
                cv2.addWeighted(overlay, 0.2, annotated_frame, 0.8, 0, annotated_frame)
                
                thickness = 2
                alpha = 0.3
                for i in range(3):
                    cv2.rectangle(annotated_frame, 
                                (x1-i, y1-i), (x2+i, y2+i), 
                                (0, 255, 0), 
                                thickness)
                    
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
                
                label = det['class_name']
                cv2.putText(annotated_frame,
                          label,
                          (x1, y2 + 25),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.7,
                          (0, 255, 0),
                          2)
            
            # Convert to bytes
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            print(f"Error in generate_frames: {e}")
            continue

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect', methods=['POST'])
def detect():
    cam = get_camera()
    if cam is None:
        return jsonify({'error': 'No camera available'})
        
    success, frame = cam.read()
    if not success:
        return jsonify({'error': 'Failed to capture frame'})
    
    try:
        detections = detector.detect_objects(frame)
            
        detections = [d for d in detections if d['confidence'] >= CONFIDENCE_THRESHOLD]
        
        smoothed_detections = smoother.update(detections)
        
        # print(f"Smoothed detections: {smoothed_detections}")
        return jsonify({'detections': smoothed_detections})
    except Exception as e:
        return jsonify({'error': f'Detection failed: {str(e)}'})

@app.route('/camera/status')
def camera_status():
    """Endpoint to check camera status"""
    cam = get_camera()
    return jsonify({
        'available': cam is not None and cam.isOpened(),
        'index': camera.get(cv2.CAP_PROP_POS_FRAMES) if camera else None
    })

@app.route('/update_threshold/<float:threshold>', methods=['POST'])
def update_threshold(threshold):
    global CONFIDENCE_THRESHOLD
    CONFIDENCE_THRESHOLD = threshold / 100  # Convert percentage to decimal
    return jsonify({'status': 'success', 'new_threshold': CONFIDENCE_THRESHOLD})

def cleanup():
    global camera
    if camera is not None:
        camera.release()

atexit.register(cleanup)

@app.route('/camera/check', methods=['GET'])
def check_camera():
    """Check camera availability and capabilities"""
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
    port = int(os.environ.get('PORT', 5000))
    
    app.run(
        host='0.0.0.0',  # Allow external access
        port=port,
        debug=False
    ) 