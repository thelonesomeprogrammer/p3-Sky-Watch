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

def calculate_gsd(altitude_m, image_width_px, SENSOR_WIDTH, FOCAL_LENGTH):
    """
    Calculate the Ground Sampling Distance (GSD) for the drone image.
    """
    # Convert Sensor Width and Focal Length from mm to meters
    sensor_width_m = SENSOR_WIDTH / 1000.0  # mm to meters
    focal_length_m = FOCAL_LENGTH / 1000.0  # mm to meters

    # GSD = (Sensor Width * Altitude) / (Focal Length * Image Width)
    gsd = (sensor_width_m * altitude_m) / (focal_length_m * image_width_px)
    print(f"Drone altitude (m): {altitude_m}")
    print(f"Image width (px): {image_width_px}")
    print(f"Calculated drone GSD (m/px): {gsd}")

    return gsd  # in meters per pixel

def estimate_scale(drone_gsd, satellite_gsd):
    """
    Estimate the scale factor between drone and satellite images.
    """
    scale_factor = drone_gsd / satellite_gsd
    return scale_factor