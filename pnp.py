import cv2
import numpy as np

def pnp(object_points, image_points, camera_matrix, dist_coeffs):

    cam_coords = []
    for i, v in enumerate(object_points):
        success, R, T = cv2.solvePnP(v, image_points[i], camera_matrix, dist_coeffs) # Solve for rotation and translation vectors
        rotation_matrix, _ = cv2.Rodrigues(R) # Convert rotation vector to rotation matrix

        cam_coords.append(-np.dot(rotation_matrix.T, T)) # Calculate the camera position in world coordinates

    return cam_coords


def sky_test(object_points, image_points):
    camera_matrix = np.array([[3400, 0, 1929], [0, 3400, 1091], [0, 0, 1]], dtype=np.float32) # Camera intrinsic parameters
    distortion_coeffs = np.zeros(4) # Distortion coefficients (assuming no distortion for simplicity)

    print(pnp(object_points, image_points, camera_matrix, distortion_coeffs))

def vpair_test(object_points, image_points):
    camera_matrix = np.array([[750.62614972, 0, 402.41007535], [0, 750.26301185, 292.98832147], [0, 0, 1]])
    #distortion_coeffs = np.array([-0.11592226392258145, 0.1332261251415265, -0.00043977637330175616, 0.0002380609784102606])
    distortion_coeffs = np.zeros(4) # Distortion coefficients (assuming no distortion for simplicity)
    print(pnp(object_points, image_points, camera_matrix, distortion_coeffs))

if __name__ == "__main__":
    sky_object_points = np.array([
        [
            [9.873987, 56.892982, 0],
            [9.875228, 56.892752, 0],
            [9.874458, 56.892741, 0],
            [9.874862, 56.893140, 0],
        ],[
            [9.872404, 56.890317, 0],
            [9.872010, 56.889679, 0],
            [9.870411, 56.890067, 0],
            [9.871091, 56.889904, 0],
        ]
    ], dtype=np.float32) # lat long alt af punkter

    sky_image_points = np.array([
        [
            [3178, 1711],
            [1700, 331],
            [2840, 806],
            [1812, 1665]
        ],[
            [3467, 520],
            [217, 735],
            [3264, 2055],
            [1643, 1413]
        ]], dtype=np.float32) # x y i billeder 
    # sky_test(sky_object_points, sky_image_points)

    vpair_object_points = np.array([
        [
            [7.228260, 50.615783, 0],
            [7.226373, 50.616064, 0],
            [7.226471, 50.614621, 0],
            [7.227210, 50.616696, 0],
        ],
        ], dtype=np.float32)
    vpair_image_points = np.array([
        [
            [553, 272],
            [242, 173],
            [222, 551],
            [394, 21 ],
        ],
        ], dtype=np.float32)
    vpair_test(vpair_object_points, vpair_image_points)
