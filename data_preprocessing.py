import albumentations as A
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
        self.data_dir = Path(data_dir)
        self.augmented_dir = self.data_dir / 'augmented'
        
    def create_augmentation_pipeline(self):
        return A.Compose([
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomScale(scale_limit=0.2, p=0.5),
            A.RandomBrightnessContrast(p=0.4),
            A.GaussNoise(p=0.3),
            A.Perspective(scale=(0.05, 0.1), p=0.4),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
            A.Blur(blur_limit=3, p=0.3),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    def prepare_directories(self):
        if self.augmented_dir.exists():
            shutil.rmtree(self.augmented_dir)
        
        (self.augmented_dir / 'images').mkdir(parents=True)
        (self.augmented_dir / 'labels').mkdir(parents=True)

    def load_yolo_bbox(self, label_path):
        bboxes = []
        class_labels = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    class_id, x, y, w, h = map(float, line.strip().split())
                    bboxes.append([x, y, w, h])
                    class_labels.append(int(class_id))
                    
        return np.array(bboxes), class_labels

    def save_augmented_data(self, image, bboxes, class_labels, image_path, idx):
        # Save image
        aug_image_path = self.augmented_dir / 'images' / f'aug_{idx}_{image_path.name}'
        cv2.imwrite(str(aug_image_path), image)
        
        # Save labels
        aug_label_path = self.augmented_dir / 'labels' / f'aug_{idx}_{image_path.stem}.txt'
        with open(aug_label_path, 'w') as f:
            for bbox, class_id in zip(bboxes, class_labels):
                f.write(f'{class_id} {" ".join(map(str, bbox))}\n')

    def augment_dataset(self, num_augmentations=3):
        self.prepare_directories()
        transform = self.create_augmentation_pipeline()
        
        # Process each image in the dataset
        images_dir = self.data_dir / 'train/images'
        labels_dir = self.data_dir / 'train/labels'
        
        image_files = list(images_dir.glob('*.jpg'))
        total_images = len(image_files)
        processed_images = 0
        skipped_images = 0
        class_distribution = {}
        
        print(f"Starting dataset augmentation...")
        print(f"Found {total_images} total images")
        
        # Main augmentation loop with progress bar
        with tqdm(total=total_images, desc="Augmenting dataset") as pbar:
            for image_path in image_files:
                # Load image and labels
                image = cv2.imread(str(image_path))
                if image is None:
                    pbar.write(f"Warning: Could not read image {image_path}")
                    skipped_images += 1
                    pbar.update(1)
                    continue
                
                label_path = labels_dir / f'{image_path.stem}.txt'
                bboxes, class_labels = self.load_yolo_bbox(label_path)
                
                # Update class distribution
                for class_id in class_labels:
                    class_distribution[class_id] = class_distribution.get(class_id, 0) + 1
                
                # Skip images without bounding boxes
                if len(bboxes) == 0:
                    pbar.write(f"Skipping {image_path.name} - No bounding boxes found")
                    skipped_images += 1
                    pbar.update(1)
                    continue
                
                # Create augmentations
                for i in range(num_augmentations):
                    transformed = transform(
                        image=image,
                        bboxes=bboxes.tolist(),
                        class_labels=class_labels
                    )
                    
                    self.save_augmented_data(
                        transformed['image'],
                        transformed['bboxes'],
                        transformed['class_labels'],
                        image_path,
                        i
                    )
                
                processed_images += 1
                pbar.update(1)
        
        # Save statistics for visualization
        stats = {
            'total_images': total_images,
            'processed_images': processed_images,
            'skipped_images': skipped_images,
            'generated_augmentations': processed_images * num_augmentations,
            'class_distribution': class_distribution
        }
        
        self.save_statistics(stats)
        self.update_data_yaml()
        return stats

    def save_statistics(self, stats):
        stats_file = self.data_dir / 'augmentation_stats.yaml'
        with open(stats_file, 'w') as f:
            yaml.dump(stats, f)

    def update_data_yaml(self):
        yaml_path = self.data_dir / 'data.yaml'
        with open(yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
            
        # Add augmented paths
        data_config['train'] = str(self.augmented_dir / 'images')
        data_config['val'] = str(self.data_dir / 'images')  # Keep original data as validation
        
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f) 
