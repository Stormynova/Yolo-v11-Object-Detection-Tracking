import cv2
import numpy as np
from pathlib import Path
import random
import matplotlib.pyplot as plt
import yaml
import seaborn as sns

def visualize_dataset(data_dir, num_samples=2):
    """
    Visualize random samples from both original and augmented dataset with bounding boxes and labels.
    Displays object detection annotations overlaid on images to validate data quality.
    
    Args:
        data_dir (str): Path to dataset directory containing images and labels
        num_samples (int): Number of random samples to visualize from each set (original and augmented)
    """
    data_dir = Path(data_dir)
    
    # Load object class names from dataset configuration
    with open(data_dir / 'data.yaml', 'r') as f:
        data_config = yaml.safe_load(f)
        class_names = data_config.get('names', [])
    
    # Locate image files in both original and augmented directories
    original_image_paths = list((data_dir / 'train/images').glob('*.jpg'))
    augmented_image_paths = list((data_dir / 'augmented/images').glob('*.jpg'))
    
    # Configure visualization layout with original and augmented samples
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 8))
    
    def plot_image_with_boxes(ax, img_path, label_path, title):
        """Helper function to render a single image with its bounding box annotations"""
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        ax.imshow(image)
        
        # Parse and visualize object detection annotations
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    
                    # Transform normalized YOLO coordinates to pixel space
                    img_height, img_width = image.shape[:2]
                    x1 = int((x_center - width/2) * img_width)
                    y1 = int((y_center - height/2) * img_height)
                    x2 = int((x_center + width/2) * img_width)
                    y2 = int((y_center + height/2) * img_height)
                    
                    # Render bounding box rectangle
                    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                      fill=False, color='red', linewidth=2)
                    ax.add_patch(rect)
                    
                    # Add class label with white background
                    class_name = class_names[int(class_id)]
                    ax.text(x1, y1-5, class_name, 
                          color='red', fontsize=10, 
                          bbox=dict(facecolor='white', alpha=0.8))
        
        ax.axis('off')
        ax.set_title(title)
    
    # Visualize randomly selected original training samples
    if original_image_paths:
        selected_originals = random.sample(original_image_paths, 
                                         min(num_samples, len(original_image_paths)))
        for idx, img_path in enumerate(selected_originals):
            label_path = data_dir / 'train/labels' / f'{img_path.stem}.txt'
            plot_image_with_boxes(axes[0, idx], img_path, label_path, f'Original {idx+1}')
    
    # Visualize randomly selected augmented samples
    if augmented_image_paths:
        selected_augmented = random.sample(augmented_image_paths, 
                                         min(num_samples, len(augmented_image_paths)))
        for idx, img_path in enumerate(selected_augmented):
            label_path = data_dir / 'augmented/labels' / f'{img_path.stem}.txt'
            plot_image_with_boxes(axes[1, idx], img_path, label_path, f'Augmented {idx+1}')
    
    plt.tight_layout()
    
    # Save visualization results
    save_dir = Path('visualization_results')
    save_dir.mkdir(exist_ok=True)
    
    plt.savefig(save_dir / 'dataset_samples_with_augmentation.png', 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Visualization saved to {save_dir}/dataset_samples_with_augmentation.png")

def visualize_statistics(data_dir):
    """
    Generate statistical visualizations summarizing dataset composition and augmentation results.
    Creates pie charts and bar plots showing class distribution and augmentation metrics.
    
    Args:
        data_dir (str): Path to dataset directory containing augmentation statistics
    """
    data_dir = Path(data_dir)
    stats_file = data_dir / 'augmentation_stats.yaml'
    
    if not stats_file.exists():
        print("No statistics file found. Run augmentation first.")
        return
    
    # Load augmentation statistics
    with open(stats_file, 'r') as f:
        stats = yaml.safe_load(f)
    
    # Configure multi-panel visualization layout
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2)
    
    # Create pie chart showing proportion of processed vs skipped images
    ax1 = fig.add_subplot(gs[0, 0])
    pie_data = [stats['processed_images'], stats['skipped_images']]
    ax1.pie(pie_data, labels=['Processed', 'Skipped'], autopct='%1.1f%%')
    ax1.set_title('Image Processing Distribution')
    
    # Create bar chart showing distribution of object classes
    ax2 = fig.add_subplot(gs[0, 1])
    class_dist = stats['class_distribution']
    
    with open(data_dir / 'data.yaml', 'r') as f:
        data_config = yaml.safe_load(f)
        class_names = data_config.get('names', [])
    
    classes = [class_names[int(k)] for k in class_dist.keys()]
    counts = list(class_dist.values())
    
    sns.barplot(x=classes, y=counts, ax=ax2)
    ax2.set_title('Class Distribution')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    
    # Create bar chart comparing dataset sizes before/after augmentation
    ax3 = fig.add_subplot(gs[1, :])
    data = [stats['total_images'], stats['processed_images'], 
            stats['generated_augmentations']]
    labels = ['Total Images', 'Processed Images', 'Generated Augmentations']
    
    sns.barplot(x=labels, y=data, ax=ax3)
    ax3.set_title('Augmentation Impact')
    
    plt.tight_layout()
    
    # Save statistical visualization results
    save_dir = Path('visualization_results')
    save_dir.mkdir(exist_ok=True)
    plt.savefig(save_dir / 'dataset_statistics.png')
    plt.close()
    
    print(f"Statistical visualization saved to {save_dir}/dataset_statistics.png")