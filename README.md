# YOLO 11 Object Detection & Tracking 🎯
[![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![YOLOv11](https://img.shields.io/badge/YOLO-v11-darkgreen.svg)](https://github.com/ultralytics/ultralytics)
[![Flask](https://img.shields.io/badge/flask-%23000.svg?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)

A real-time object detection system using YOLOv11 for identifying and tracking household objects with distance and angle estimation.

<center><img src="./assets/cover.gif" alt="yolo detection gif" width="500"></center>

## ✨ Features

- 🎥 Real-time object detection using YOLOv11
- 📏 Distance and angle estimation for detected objects  
- 🌐 Web-based interface with live video feed
- 🎚️ Confidence threshold adjustment
- 🔄 Temporal smoothing for stable detections
- 📊 Camera status monitoring and error handling

## 🎯 Object types
🥄 Spoon, 🧢 Cap, 📏 Comb, ⌚ Watch, ☕ Mug

## 📖 About

## 🚀 Installation

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

## YOLOv11 Training Results
### Training Plots
![](./runs/detect/household_objects-batch32-v11-alldata_e200/results.png)
![](./runs/detect/household_objects-batch32-v11-alldata_e200/confusion_matrix_normalized.png)
![](./runs/detect/household_objects-batch32-v11-alldata_e200/labels.jpg)
![](./runs/detect/household_objects-batch32-v11-alldata_e200/labels_correlogram.jpg)
![](./runs/detect/household_objects-batch32-v11-alldata_e200/F1_curve.png)
![](./runs/detect/household_objects-batch32-v11-alldata_e200/P_curve.png)
![](./runs/detect/household_objects-batch32-v11-alldata_e200/R_curve.png)
![](./runs/detect/household_objects-batch32-v11-alldata_e200/PR_curve.png)

### Training Predictions
![](./runs/detect/household_objects-batch32-v11-alldata_e200/train_batch0.jpg)

### Validation Predictions
![](./runs/detect/household_objects-batch32-v11-alldata_e200/val_batch1_pred.jpg)

![](./runs/detect/household_objects-batch32-v11-alldata_e200/val_batch0_pred.jpg)


## [License](./LICENSE)
