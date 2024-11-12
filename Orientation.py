import numpy as np
import cv2
import math 

def correct_perspective(image, roll, pitch, yaw, focal_length_px):
    """
    Correct drone image perspective using roll, pitch, and yaw angles.
    """
    # Convert angles from degrees to radians
    roll_rad = math.radians(roll)
    pitch_rad = math.radians(pitch)
    yaw_rad = math.radians(yaw)

    # Image dimensions
    h, w = image.shape[:2]

    # Camera intrinsic matrix (assuming principal point at the center)
    K = np.array([[focal_length_px, 0, w / 2],
                  [0, focal_length_px, h / 2],
                  [0, 0, 1]])

    # Rotation matrices around the X, Y, and Z axes
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll_rad), -np.sin(roll_rad)],
                   [0, np.sin(roll_rad), np.cos(roll_rad)]])
    Ry = np.array([[np.cos(pitch_rad), 0, np.sin(pitch_rad)],
                   [0, 1, 0],
                   [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]])
    Rz = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad), 0],
                   [np.sin(yaw_rad), np.cos(yaw_rad), 0],
                   [0, 0, 1]])

    # Combined rotation matrix
    R = Rz @ Ry @ Rx

    # Homography matrix
    H = K @ R @ np.linalg.inv(K)
    H = H / H[2, 2]  # Normalize so that H[2,2] == 1

    # Warp the image using the homography matrix
    corrected_image = cv2.warpPerspective(image, H, (w, h), flags=cv2.INTER_LINEAR)

    return corrected_image