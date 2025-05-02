import numpy as np

def calculate_eye_aspect_ratio(eye_landmarks):
    """
    Calculate the eye aspect ratio to determine if the eye is open or closed.
    """
    # Calculate vertical distances
    distance_1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    distance_2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    
    # Calculate horizontal distance
    distance_3 = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    
    # Calculate eye aspect ratio
    ear_value = (distance_1 + distance_2) / (2.0 * distance_3)
    return ear_value

def calculate_mouth_aspect_ratio(mouth_landmarks):
    """
    Calculate the mouth aspect ratio to determine if the mouth is open or closed.
    """
    # Calculate vertical distances
    distance_1 = np.linalg.norm(mouth_landmarks[13] - mouth_landmarks[19])
    distance_2 = np.linalg.norm(mouth_landmarks[14] - mouth_landmarks[18])
    distance_3 = np.linalg.norm(mouth_landmarks[15] - mouth_landmarks[17])
    
    # Calculate horizontal distance
    distance_4 = np.linalg.norm(mouth_landmarks[12] - mouth_landmarks[16])
    
    # Calculate mouth aspect ratio
    mar_value = (distance_1 + distance_2 + distance_3) / (2 * distance_4)
    return mar_value

def determine_direction(nose_coordinate, anchor_coordinate, width, height, multiplier=1):
    """
    Determine the direction of movement based on nose position relative to anchor point.
    """
    nx, ny = nose_coordinate
    x, y = anchor_coordinate
    
    # Determine direction based on nose position
    if nx > x + multiplier * width:
        return 'right'
    elif nx < x - multiplier * width:
        return 'left'
    if ny > y + multiplier * height:
        return 'down'
    elif ny < y - multiplier * height:
        return 'up'
    return '-' 
