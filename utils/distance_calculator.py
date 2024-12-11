import math
from config.object_sizes import REFERENCE_SIZES, FOCAL_LENGTH

def calculate_distance_and_angle(bbox, class_name, frame_width):
    """
    Calculate the distance and angle of a detected object using its bounding box dimensions.
    
    Uses the pinhole camera model and known real-world object dimensions to estimate distance.
    Angle is calculated based on object's horizontal offset from frame center.
    
    Args:
        bbox (tuple): Bounding box coordinates (x1, y1, x2, y2) in pixels
        class_name (str): Class name of detected object to look up reference size
        frame_width (int): Width of the image frame in pixels
        
    Returns:
        tuple: (distance, angle) where:
            - distance (float): Estimated distance to object in meters, rounded to 2 decimals
            - angle (float): Angle from frame center in degrees, rounded to 1 decimal
            Returns (None, None) if class_name not found in reference sizes
    """
    # Check if we have reference dimensions for this object class
    if class_name not in REFERENCE_SIZES:
        return None, None

    # Extract bounding box corner coordinates
    x1, y1, x2, y2 = bbox
    
    # Calculate object dimensions in pixels from bounding box
    pixel_width = x2 - x1  # Width in pixels
    pixel_height = y2 - y1  # Height in pixels
    
    # Get real-world reference dimensions for this object class
    ref_size = REFERENCE_SIZES[class_name]
    
    # Calculate distance using similar triangles principle:
    # real_size / distance = pixel_size / focal_length
    # Therefore: distance = (real_size * focal_length) / pixel_size
    distance_by_width = (ref_size['width'] * FOCAL_LENGTH) / pixel_width
    distance_by_height = (ref_size['height'] * FOCAL_LENGTH) / pixel_height
    
    # Average the two distance calculations for better accuracy
    # This helps account for any perspective distortion
    distance = (distance_by_width + distance_by_height) / 2
    
    # Find horizontal center points
    object_center_x = (x1 + x2) / 2  # Center x-coordinate of object
    frame_center_x = frame_width / 2  # Center x-coordinate of frame
    
    # Calculate angle using arctangent:
    # opposite = distance from center in pixels
    # adjacent = focal length in pixels
    angle_rad = math.atan2(object_center_x - frame_center_x, FOCAL_LENGTH)
    # Convert angle from radians to degrees
    angle_deg = math.degrees(angle_rad)

    # Return rounded values for cleaner output
    return round(distance, 2), round(angle_deg, 1)
