from ultralytics import YOLO
import os
from roboflow import Roboflow
from data_preprocessing import DataPreprocessor
from visualization import visualize_dataset, visualize_statistics
import argparse
import yaml

def train_yolo(augment_dataset = False):
    data_dir = 'Household-small-objects'
    
    preprocessor = DataPreprocessor(data_dir)
    
    if augment_dataset:
        print("Performing data augmentation...")
        preprocessor.augment_dataset(num_augmentations=3)
        
    print("Generating dataset visualizations...")
    visualize_dataset(data_dir)
    visualize_statistics(data_dir)

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
    
    parser = argparse.ArgumentParser(description="Train YOLO model with or without data augmentation")
    parser.add_argument("--augment", action="store_true", help="Perform data augmentation")
    args = parser.parse_args()
    augment_dataset = args.augment
    train_yolo(augment_dataset)