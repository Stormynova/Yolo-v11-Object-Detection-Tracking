import math
from config.object_sizes import REFERENCE_SIZES, FOCAL_LENGTH

def calculate_distance_and_angle(bbox, class_name, frame_width):
    """Calculate distance and angle of detected object"""
    if class_name not in REFERENCE_SIZES:
        return None, None

    # Extract bbox coordinates
    x1, y1, x2, y2 = bbox
    
    # Calculate pixel dimensions
    pixel_width = x2 - x1
    pixel_height = y2 - y1
    
    # Get reference size
    ref_size = REFERENCE_SIZES[class_name]
    
    # Calculate distance using width and height
    distance_by_width = (ref_size['width'] * FOCAL_LENGTH) / pixel_width
    distance_by_height = (ref_size['height'] * FOCAL_LENGTH) / pixel_height
    
    # Use average of both measurements
    distance = (distance_by_width + distance_by_height) / 2
    
    # Calculate angle from center
    object_center_x = (x1 + x2) / 2
    frame_center_x = frame_width / 2
    
    # Convert pixel difference to angle
    angle_rad = math.atan2(object_center_x - frame_center_x, FOCAL_LENGTH)
    angle_deg = math.degrees(angle_rad)

    # print(f"Distance: {round(distance, 2)}m, Angle: {round(angle_deg, 1)}°")
    return round(distance, 2), round(angle_deg, 1)
