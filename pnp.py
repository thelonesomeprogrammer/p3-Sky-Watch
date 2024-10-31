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
f=13.6
camera_matrix = np.array([
    [800, 0, 3840],
    [0, 800, 2160],
    [0, 0, 1]
], dtype=np.float32)

# Distortion coefficients (assuming no distortion for simplicity)
dist_coeffs = np.zeros(4)

# Solve for rotation and translation vectors
success, R, T = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
 # Convert rotation vector to rotation matrix
#rotation_matrix, _ = cv2.Rodrigues(rvec)

# Calculate the camera position in world coordinates
## camera_position = -np.dot(rotation_matrix.T, tvec)


# R = np.array([[-0.0813445156856268], [-2.5478950926311636], [1.7376856594745234]], dtype=np.float32)
# T = np.array([[10.838262901867047], [-6.506593974297687], [60.308121310607724]], dtype=np.float32)

world_point = [13, 0, 0]

rvec_matrix = cv2.Rodrigues(R)[0]
rmat = np.matrix(rvec_matrix)
tmat = np.matrix(T)
pmat = np.matrix(np.array([[world_point[0]], [world_point[1]], [world_point[2]]], dtype=np.float32))

# world coordinate to camera coordinate
cam_point = rmat * pmat + tmat
print(cam_point)

# camera coordinate to world coordinate
world_point = rmat ** -1 * (cam_point - tmat)
print(world_point)
