# Import required libraries and modules
from data_preprocessing import DataPreprocessor  # Custom data preprocessing utilities
from visualization import visualize_dataset, visualize_statistics  # Custom visualization functions
from ultralytics import YOLO  # YOLO object detection model
import yaml  # For parsing YAML configuration files
from pathlib import Path  # File path operations
import argparse  # Command line argument parsing
import os  # Operating system interface
from roboflow import Roboflow  # Roboflow API client for dataset management

def train_yolo(augment_dataset=False):
    """
    Train a YOLO model on household objects dataset with optional data augmentation
    
    Args:
        augment_dataset (bool): Whether to perform data augmentation before training
        
    Returns:
        dict: Training results and metrics including mAP, precision, recall etc.
    """
    # Directory containing the dataset
    data_dir = 'Household-small-objects-5-4'
    
    # Initialize preprocessor for data augmentation and preparation
    preprocessor = DataPreprocessor(data_dir)
    
    # Optionally augment training data to increase dataset size and variety
    # This creates additional training samples through transformations like rotation, scaling etc.
    if augment_dataset:
        print("Performing data augmentation...")
        stats = preprocessor.augment_dataset(num_augmentations=3)
    
    # Generate visualizations of dataset samples and statistics
    # This helps in understanding data distribution and verifying augmentations
    print("Generating dataset visualizations...")
    visualize_dataset(data_dir)  # Displays sample images with annotations
    visualize_statistics(data_dir)  # Shows class distribution and other metrics
    
    # Initialize pretrained YOLOv11-nano model as starting point
    # Using nano variant for faster training while maintaining decent performance
    model = YOLO('yolo11n.pt')
    
    # Fine-tune model on household objects dataset
    # Configure training parameters and start training process
    results = model.train(
        data=f'{data_dir}/data.yaml',  # Path to dataset configuration
        epochs=100,  # Number of training epochs
        imgsz=640,  # Input image size
        batch=32,  # Batch size for training
        name='household_objects-batch32-v11-alldata'  # Name for saving training results
    )
    
    return results

if __name__ == "__main__":
    # Download dataset from Roboflow platform
    # API key should be set in environment variables for security
    rf = Roboflow(api_key=os.environ.get('ROBOFLOW_API_KEY'))
    project = rf.workspace("householdobjectsyolo").project("household-small-objects-5")
    version = project.version(4)  # Using version 4 of the dataset
    dataset = version.download("yolov11")  # Download in YOLOv11 format
    
    # Set up command line argument parsing
    # Allows users to enable/disable data augmentation from command line
    parser = argparse.ArgumentParser(description="Train YOLO model with optional data augmentation")
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation to increase dataset variety")
    args = parser.parse_args()
    
    # Extract augmentation flag from command line arguments
    augment_dataset = args.augment
    # Start training process with specified configuration
    train_yolo(augment_dataset)