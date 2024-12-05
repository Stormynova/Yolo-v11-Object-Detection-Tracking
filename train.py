from data_preprocessing import DataPreprocessor
from visualization import visualize_dataset, visualize_statistics
from ultralytics import YOLO
import yaml
from pathlib import Path
import argparse
import os
from roboflow import Roboflow

def train_yolo(augment_dataset=False):
    """
    Train a YOLO model on household objects dataset with optional data augmentation
    
    Args:
        augment_dataset (bool): Whether to perform data augmentation before training
        
    Returns:
        dict: Training results and metrics
    """
    data_dir = 'Household-small-objects-5-4'
    
    # Initialize preprocessor for data augmentation and preparation
    preprocessor = DataPreprocessor(data_dir)
    
    # Optionally augment training data to increase dataset size and variety
    if augment_dataset:
        print("Performing data augmentation...")
        stats = preprocessor.augment_dataset(num_augmentations=3)
    
    # Generate visualizations of dataset samples and statistics
    print("Generating dataset visualizations...")
    visualize_dataset(data_dir)
    visualize_statistics(data_dir)
    
    # Initialize pretrained YOLOv11-nano model as starting point
    model = YOLO('yolo11n.pt')
    
    # Fine-tune model on household objects dataset
    results = model.train(
        data=f'{data_dir}/data.yaml',
        epochs=100,
        imgsz=640,
        batch=32,
        name='household_objects-batch32-v11-alldata'
    )
    
    return results

if __name__ == "__main__":
    # Download dataset from Roboflow
    rf = Roboflow(api_key=os.environ.get('ROBOFLOW_API_KEY'))
    project = rf.workspace("householdobjectsyolo").project("household-small-objects-5")
    version = project.version(4)
    dataset = version.download("yolov11")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train YOLO model with optional data augmentation")
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation to increase dataset variety")
    args = parser.parse_args()
    
    augment_dataset = args.augment
    train_yolo(augment_dataset)