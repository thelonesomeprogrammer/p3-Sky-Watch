import cv2
import numpy as np

# Define 3D points (in object coordinates) and 2D points (in image coordinates)
object_points = np.array([
    [9.872404, 56.890317, 0],
    [9.872010, 56.889679, 0],
    [9.870411, 56.890067, 0],
    [9.871091, 56.889904, 0],
], dtype=np.float32)

image_points = np.array([
    [3467, 520],
    [217, 735],
    [3264, 2055],
    [1643, 1413]
], dtype=np.float32)

# Camera intrinsic parameters (example values)
camera_matrix = np.array([
    [13.6, 0, 3840],
    [0, 13.6, 2160],
    [0, 0, 1]
], dtype=np.float32)

# Distortion coefficients (assuming no distortion for simplicity)
dist_coeffs = np.zeros(4)

# Solve for rotation and translation vectors
success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
 # Convert rotation vector to rotation matrix
rotation_matrix, _ = cv2.Rodrigues(rvec)

# Calculate the camera position in world coordinates
camera_position = -np.dot(rotation_matrix.T, tvec)

print(success)
print(rvec)
print(tvec)
print(camera_position) 
