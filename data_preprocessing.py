# Import required libraries
import albumentations as A  # Library for image augmentation
import cv2 
import numpy as np
from pathlib import Path
import yaml
import shutil
import os
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

class DataPreprocessor:
    def __init__(self, data_dir):
        """
        Initialize data preprocessor for augmenting object detection dataset.
        
        Args:
            data_dir: Path to dataset directory containing images and labels
        """
        # Store dataset directory path and create path for augmented data
        self.data_dir = Path(data_dir)
        self.augmented_dir = self.data_dir / 'augmented'
        
    def create_augmentation_pipeline(self):
        """
        Create an augmentation pipeline using Albumentations library.
        Includes geometric transforms, color adjustments, and noise additions.
        
        Returns:
            Albumentations Compose object with configured augmentation pipeline
        """
        # Define sequence of augmentation transforms with probabilities
        return A.Compose([
            A.RandomRotate90(p=0.5),  # 90 degree rotations
            A.Rotate(limit=30, p=0.5),  # Random rotation up to 30 degrees
            A.HorizontalFlip(p=0.5),  # Horizontal mirroring
            A.VerticalFlip(p=0.3),  # Vertical mirroring
            A.RandomScale(scale_limit=0.2, p=0.5),  # Random image scaling
            A.RandomBrightnessContrast(p=0.4),  # Adjust brightness and contrast
            A.GaussNoise(p=0.3),  # Add Gaussian noise
            A.Perspective(scale=(0.05, 0.1), p=0.4),  # Perspective transformation
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),  # Color adjustments
            A.Blur(blur_limit=3, p=0.3),  # Image blurring
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))  # Configure for YOLO format boxes

    def prepare_directories(self):
        """
        Create clean directory structure for augmented dataset.
        Removes existing augmented directory if present.
        """
        # Remove old augmented data if it exists
        if self.augmented_dir.exists():
            shutil.rmtree(self.augmented_dir)
        
        # Create fresh directories for images and labels
        (self.augmented_dir / 'images').mkdir(parents=True)
        (self.augmented_dir / 'labels').mkdir(parents=True)

    def load_yolo_bbox(self, label_path):
        """
        Load bounding box annotations in YOLO format from label file.
        
        Args:
            label_path: Path to label file containing YOLO format annotations
            
        Returns:
            Tuple of (bounding boxes array, class label list)
        """
        bboxes = []  # Store bounding box coordinates
        class_labels = []  # Store class IDs
        
        # Read and parse label file if it exists
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    # Parse YOLO format: class_id x_center y_center width height
                    class_id, x, y, w, h = map(float, line.strip().split())
                    bboxes.append([x, y, w, h])
                    class_labels.append(int(class_id))
                    
        return np.array(bboxes), class_labels

    def save_augmented_data(self, image, bboxes, class_labels, image_path, idx):
        """
        Save augmented image and corresponding label file.
        
        Args:
            image: Augmented image array
            bboxes: List of bounding boxes in YOLO format
            class_labels: List of class IDs for each bbox
            image_path: Original image path for naming
            idx: Augmentation index for unique filenames
        """
        # Save augmented image with unique name
        aug_image_path = self.augmented_dir / 'images' / f'aug_{idx}_{image_path.name}'
        cv2.imwrite(str(aug_image_path), image)
        
        # Save corresponding label file with matching name
        aug_label_path = self.augmented_dir / 'labels' / f'aug_{idx}_{image_path.stem}.txt'
        with open(aug_label_path, 'w') as f:
            # Write each bbox and class in YOLO format
            for bbox, class_id in zip(bboxes, class_labels):
                f.write(f'{class_id} {" ".join(map(str, bbox))}\n')

    def augment_dataset(self, num_augmentations=3):
        """
        Perform dataset augmentation by applying transforms to images and labels.
        
        Args:
            num_augmentations: Number of augmented copies to generate per image
            
        Returns:
            Dictionary containing augmentation statistics and class distribution
        """
        # Prepare directory structure and augmentation pipeline
        self.prepare_directories()
        transform = self.create_augmentation_pipeline()
        
        # Set up paths and counters
        images_dir = self.data_dir / 'train/images'
        labels_dir = self.data_dir / 'train/labels'
        
        image_files = list(images_dir.glob('*.jpg'))
        total_images = len(image_files)
        processed_images = 0
        skipped_images = 0
        class_distribution = {}  # Track class frequencies
        
        print(f"Starting dataset augmentation...")
        print(f"Found {total_images} total images")
        
        # Process each image with progress tracking
        with tqdm(total=total_images, desc="Augmenting dataset") as pbar:
            for image_path in image_files:
                # Load and validate image
                image = cv2.imread(str(image_path))
                if image is None:
                    pbar.write(f"Warning: Could not read image {image_path}")
                    skipped_images += 1
                    pbar.update(1)
                    continue
                
                # Load corresponding labels
                label_path = labels_dir / f'{image_path.stem}.txt'
                bboxes, class_labels = self.load_yolo_bbox(label_path)
                
                # Update class frequency counts
                for class_id in class_labels:
                    class_distribution[class_id] = class_distribution.get(class_id, 0) + 1
                
                # Skip images without annotations
                if len(bboxes) == 0:
                    pbar.write(f"Skipping {image_path.name} - No bounding boxes found")
                    skipped_images += 1
                    pbar.update(1)
                    continue
                
                # Generate multiple augmented versions
                for i in range(num_augmentations):
                    # Apply augmentation transforms
                    transformed = transform(
                        image=image,
                        bboxes=bboxes.tolist(),
                        class_labels=class_labels
                    )
                    
                    # Save augmented data
                    self.save_augmented_data(
                        transformed['image'],
                        transformed['bboxes'],
                        transformed['class_labels'],
                        image_path,
                        i
                    )
                
                processed_images += 1
                pbar.update(1)
        
        # Compile augmentation statistics
        stats = {
            'total_images': total_images,
            'processed_images': processed_images,
            'skipped_images': skipped_images,
            'generated_augmentations': processed_images * num_augmentations,
            'class_distribution': class_distribution
        }
        
        # Save results and update configuration
        self.save_statistics(stats)
        self.update_data_yaml()
        return stats

    def save_statistics(self, stats):
        """
        Save augmentation statistics to YAML file for later visualization.
        
        Args:
            stats: Dictionary containing augmentation statistics
        """
        # Write statistics to YAML file
        stats_file = self.data_dir / 'augmentation_stats.yaml'
        with open(stats_file, 'w') as f:
            yaml.dump(stats, f)

    def update_data_yaml(self):
        """
        Update data.yaml configuration to include augmented dataset paths.
        Uses augmented data for training and original data for validation.
        """
        # Load existing configuration
        yaml_path = self.data_dir / 'data.yaml'
        with open(yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
            
        # Update paths to use augmented data for training
        data_config['train'] = str(self.augmented_dir / 'images')
        data_config['val'] = str(self.data_dir / 'images')  # Keep original data as validation
        
        # Save updated configuration
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f) 
