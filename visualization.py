import cv2
import numpy as np
from pathlib import Path
import random
import matplotlib.pyplot as plt
import yaml
import seaborn as sns

def visualize_dataset(data_dir, num_samples=2):
    data_dir = Path(data_dir)
    
    # Load class names from data.yaml
    with open(data_dir / 'data.yaml', 'r') as f:
        data_config = yaml.safe_load(f)
        class_names = data_config.get('names', [])
    
    # Get paths for both original and augmented images
    original_image_paths = list((data_dir / 'train/images').glob('*.jpg'))
    augmented_image_paths = list((data_dir / 'augmented/images').glob('*.jpg'))
    
    # Create a subplot grid (2 rows: original and augmented)
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 8))
    
    # Helper function to plot image with bounding boxes
    def plot_image_with_boxes(ax, img_path, label_path, title):
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Plot image
        ax.imshow(image)
        
        # Read and plot bounding boxes
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    
                    # Convert YOLO format to pixel coordinates
                    img_height, img_width = image.shape[:2]
                    x1 = int((x_center - width/2) * img_width)
                    y1 = int((y_center - height/2) * img_height)
                    x2 = int((x_center + width/2) * img_width)
                    y2 = int((y_center + height/2) * img_height)
                    
                    # Draw bounding box
                    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                      fill=False, color='red', linewidth=2)
                    ax.add_patch(rect)
                    
                    # Add label
                    class_name = class_names[int(class_id)]
                    ax.text(x1, y1-5, class_name, 
                          color='red', fontsize=10, 
                          bbox=dict(facecolor='white', alpha=0.8))
        
        ax.axis('off')
        ax.set_title(title)
    
    # Plot original samples
    if original_image_paths:
        selected_originals = random.sample(original_image_paths, 
                                         min(num_samples, len(original_image_paths)))
        for idx, img_path in enumerate(selected_originals):
            label_path = data_dir / 'train/labels' / f'{img_path.stem}.txt'
            plot_image_with_boxes(axes[0, idx], img_path, label_path, f'Original {idx+1}')
    
    # Plot augmented samples
    if augmented_image_paths:
        selected_augmented = random.sample(augmented_image_paths, 
                                         min(num_samples, len(augmented_image_paths)))
        for idx, img_path in enumerate(selected_augmented):
            label_path = data_dir / 'augmented/labels' / f'{img_path.stem}.txt'
            plot_image_with_boxes(axes[1, idx], img_path, label_path, f'Augmented {idx+1}')
    
    plt.tight_layout()
    
    # Create directory for visualization results
    save_dir = Path('visualization_results')
    save_dir.mkdir(exist_ok=True)
    
    # Save the plot
    plt.savefig(save_dir / 'dataset_samples_with_augmentation.png', 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Visualization saved to {save_dir}/dataset_samples_with_augmentation.png")

def visualize_statistics(data_dir):
    data_dir = Path(data_dir)
    stats_file = data_dir / 'augmentation_stats.yaml'
    
    if not stats_file.exists():
        print("No statistics file found. Run augmentation first.")
        return
    
    with open(stats_file, 'r') as f:
        stats = yaml.safe_load(f)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2)
    
    # 1. Pie chart of processed vs skipped images
    ax1 = fig.add_subplot(gs[0, 0])
    pie_data = [stats['processed_images'], stats['skipped_images']]
    ax1.pie(pie_data, labels=['Processed', 'Skipped'], autopct='%1.1f%%')
    ax1.set_title('Image Processing Distribution')
    
    # 2. Bar chart of class distribution
    ax2 = fig.add_subplot(gs[0, 1])
    class_dist = stats['class_distribution']
    
    # Load class names from data.yaml
    with open(data_dir / 'data.yaml', 'r') as f:
        data_config = yaml.safe_load(f)
        class_names = data_config.get('names', [])
    
    classes = [class_names[int(k)] for k in class_dist.keys()]
    counts = list(class_dist.values())
    
    sns.barplot(x=classes, y=counts, ax=ax2)
    ax2.set_title('Class Distribution')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    
    # 3. Augmentation impact
    ax3 = fig.add_subplot(gs[1, :])
    data = [stats['total_images'], stats['processed_images'], 
            stats['generated_augmentations']]
    labels = ['Total Images', 'Processed Images', 'Generated Augmentations']
    
    sns.barplot(x=labels, y=data, ax=ax3)
    ax3.set_title('Augmentation Impact')
    
    plt.tight_layout()
    
    # Save the statistical visualization
    save_dir = Path('visualization_results')
    save_dir.mkdir(exist_ok=True)
    plt.savefig(save_dir / 'dataset_statistics.png')
    plt.close()
    
    print(f"Statistical visualization saved to {save_dir}/dataset_statistics.png")