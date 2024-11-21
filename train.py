from ultralytics import YOLO
import os
from roboflow import Roboflow

def train_yolo():
    data_dir = 'Household-small-objects'
        
    # Load a pretrained YOLO model
    model = YOLO('yolo11n.pt')
    
    # Train the model with augmented dataset
    results = model.train(
        data=f'{data_dir}/data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        name='household_objects'
    )
    
    return results

if __name__ == "__main__":
    
    rf = Roboflow(api_key=os.environ.get('ROBOFLOW_API_KEY'))
    project = rf.workspace("householdobjectsyolo").project("household-small-objects-5")
    version = project.version(4)
    dataset = version.download("yolov11")
    
    train_yolo()