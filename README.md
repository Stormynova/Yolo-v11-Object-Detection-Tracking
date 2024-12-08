# YOLO Object Detection
A real-time object detection system using YOLOv11 for identifying and tracking household objects with distance and angle estimation.

## Features

- Real-time object detection using YOLOv11
- Distance and angle estimation for detected objects
- Web-based interface with live video feed
- Confidence threshold adjustment
- Temporal smoothing for stable detections
- Camera status monitoring and error handling

## Object types
Spoon, Cap, Comb, Watch, Mug

## About

## Installation

1. Clone the repository: 

```bash
git clone https://github.com/Stormynova/Object-detection-tracking.git
cd Object-detection-tracking

pip install -r requirements.txt
```

## Project Structure

- `app.py`: Main Flask application with video processing and API endpoints
- `inference.py`: YOLO model implementation and object detection logic
- `prediction_smoother.py`: Temporal smoothing for stable detections
- `utils/`: Helper utilities including distance calculation
- `config/`: Configuration files for object sizes and camera parameters
- `static/`: CSS styles and JavaScript files
- `templates/`: HTML templates for the web interface

## Usage
1. Start the application:
```bash 
python app.py
```

2. Open your web browser and navigate to:
[http://localhost:5000](http://localhost:5000)

## Training

To train the model on your own dataset:

1. Prepare your dataset in YOLOv11 format
2. Configure training parameters in `train.py`
3. Run training:

```bash
python train.py --augment # Use --augment flag for data augmentation
```

## [License](./LICENSE)