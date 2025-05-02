
import cv2
import dlib
import imutils
import numpy as np
import pyautogui as pag
from imutils import face_utils
from utils import calculate_eye_aspect_ratio, calculate_mouth_aspect_ratio, determine_direction

# Disable PyAutoGUI's fail-safe
pag.FAILSAFE = False

# Thresholds and parameters
mouth_aspect_ratio_threshold = 0.3
mouth_aspect_ratio_consecutive_frames = 3
eye_aspect_ratio_threshold = 0.25
eye_aspect_ratio_consecutive_frames = 5
wink_aspect_ratio_diff_threshold = 0.05
wink_aspect_ratio_close_threshold = 0.2
wink_consecutive_frames = 2
scroll_threshold = 0.4  # Threshold for scrolling action
scroll_speed = 10  # Pixels to scroll per frame

# Initialize counters and flags
mouth_counter = 0
eye_counter = 0
wink_counter = 0
input_mode = False
eye_click = False
left_wink = False
right_wink = False
scroll_mode = False
scroll_counter = 0
anchor_point = (0, 0)

# Color definitions
white_color = (255, 255, 255)
yellow_color = (0, 255, 255)
red_color = (0, 0, 255)
green_color = (0, 255, 0)
blue_color = (255, 0, 0)
black_color = (0, 0, 0)

# Load facial landmark predictor
shape_predictor_path = "model\\shape_predictor_68_face_landmarks.dat"
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(shape_predictor_path)

# Define facial landmark indices
(left_eye_start, left_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(right_eye_start, right_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(nose_start, nose_end) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
(mouth_start, mouth_end) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# Initialize video capture
video_capture = cv2.VideoCapture(0)
screen_resolution_width = 1366
screen_resolution_height = 768
camera_width = 640
camera_height = 480
width_unit = screen_resolution_width / camera_width
height_unit = screen_resolution_height / camera_height

while True:
    # Read frame from video capture
    _, frame = video_capture.read()
    frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame, width=camera_width, height=camera_height)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_rects = face_detector(gray_frame, 0)

    if len(face_rects) > 0:
        face_rect = face_rects[0]
    else:
        cv2.imshow("Frame", frame)
        key_pressed = cv2.waitKey(1) & 0xFF
        continue

    # Get facial landmarks
    face_shape = landmark_predictor(gray_frame, face_rect)
    face_shape = face_utils.shape_to_np(face_shape)

    # Extract landmarks for different facial features
    mouth_landmarks = face_shape[mouth_start:mouth_end]
    left_eye_landmarks = face_shape[left_eye_start:left_eye_end]
    right_eye_landmarks = face_shape[right_eye_start:right_eye_end]
    nose_landmarks = face_shape[nose_start:nose_end]

    # Swap left and right eye landmarks due to mirror effect
    temp_eye = left_eye_landmarks
    left_eye_landmarks = right_eye_landmarks
    right_eye_landmarks = temp_eye

    # Calculate aspect ratios
    mouth_aspect_ratio_value = calculate_mouth_aspect_ratio(mouth_landmarks)
    left_eye_aspect_ratio_value = calculate_eye_aspect_ratio(left_eye_landmarks)
    right_eye_aspect_ratio_value = calculate_eye_aspect_ratio(right_eye_landmarks)
    average_eye_aspect_ratio = (left_eye_aspect_ratio_value + right_eye_aspect_ratio_value) / 2.0
    difference_eye_aspect_ratio = np.abs(left_eye_aspect_ratio_value - right_eye_aspect_ratio_value)

    # Get nose tip coordinates
    nose_tip = (nose_landmarks[3, 0], nose_landmarks[3, 1])

    # Draw facial landmarks
    mouth_hull = cv2.convexHull(mouth_landmarks)
    left_eye_hull = cv2.convexHull(left_eye_landmarks)
    right_eye_hull = cv2.convexHull(right_eye_landmarks)
    cv2.drawContours(frame, [mouth_hull], -1, yellow_color, 1)
    cv2.drawContours(frame, [left_eye_hull], -1, yellow_color, 1)
    cv2.drawContours(frame, [right_eye_hull], -1, yellow_color, 1)

    # Draw points on facial landmarks
    for (x, y) in np.concatenate((mouth_landmarks, left_eye_landmarks, right_eye_landmarks), axis=0):
        cv2.circle(frame, (x, y), 2, green_color, -1)

    # Handle eye winks for mouse clicks
    if difference_eye_aspect_ratio > wink_aspect_ratio_diff_threshold:
        if left_eye_aspect_ratio_value < right_eye_aspect_ratio_value:
            if left_eye_aspect_ratio_value < eye_aspect_ratio_threshold:
                wink_counter += 1
                if wink_counter > wink_consecutive_frames:
                    pag.click(button='left')  # Left click on left eye wink
                    wink_counter = 0
        elif left_eye_aspect_ratio_value > right_eye_aspect_ratio_value:
            if right_eye_aspect_ratio_value < eye_aspect_ratio_threshold:
                wink_counter += 1
                if wink_counter > wink_consecutive_frames:
                    pag.click(button='right')  # Right click on right eye wink
                    wink_counter = 0
        else:
            wink_counter = 0
    else:
        if average_eye_aspect_ratio <= eye_aspect_ratio_threshold:
            eye_counter += 1
            if eye_counter > eye_aspect_ratio_consecutive_frames:
                scroll_mode = not scroll_mode
                eye_counter = 0
        else:
            eye_counter = 0
        wink_counter = 0

    # Handle scrolling based on mouth movement
    if scroll_mode:
        cv2.putText(frame, "SCROLL MODE!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, red_color, 2)
        if mouth_aspect_ratio_value > scroll_threshold:
            # Scroll up when mouth is open
            pag.scroll(scroll_speed)
        elif mouth_aspect_ratio_value < 0.2:
            # Scroll down when mouth is closed
            pag.scroll(-scroll_speed)

    # Handle input mode toggle
    if mouth_aspect_ratio_value > mouth_aspect_ratio_threshold:
        mouth_counter += 1
        if mouth_counter >= mouth_aspect_ratio_consecutive_frames:
            input_mode = not input_mode
            mouth_counter = 0
            anchor_point = nose_tip
    else:
        mouth_counter = 0

    # Handle cursor movement in input mode
    if input_mode:
        cv2.putText(frame, "READING INPUT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, red_color, 2)
        anchor_x, anchor_y = anchor_point
        nose_x, nose_y = nose_tip
        rectangle_width, rectangle_height = 60, 35
        cv2.rectangle(frame, (anchor_x - rectangle_width, anchor_y - rectangle_height),
                      (anchor_x + rectangle_width, anchor_y + rectangle_height), green_color, 2)
        cv2.line(frame, anchor_point, nose_tip, blue_color, 2)
        direction_value = determine_direction(nose_tip, anchor_point, rectangle_width, rectangle_height)
        cv2.putText(frame, direction_value.upper(), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, red_color, 2)
        drag_value = 18
        if direction_value == 'right':
            pag.moveRel(drag_value, 0, duration=0.1)
        elif direction_value == 'left':
            pag.moveRel(-drag_value, 0, duration=0.1)
        elif direction_value == 'up':
            pag.moveRel(0, -drag_value, duration=0.1)
        elif direction_value == 'down':
            pag.moveRel(0, drag_value, duration=0.1)

    # Display frame and handle exit
    cv2.imshow("Frame", frame)
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

# Clean up
video_capture.release()
cv2.destroyAllWindows()