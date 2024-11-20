from ultralytics import YOLO

def train_yolo():
    data_dir = 'Household-small-objects'
        
    # Load a pretrained YOLO model
    model = YOLO('yolov8n.pt')
    
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
    train_yolo()