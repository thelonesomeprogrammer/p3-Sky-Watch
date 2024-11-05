import cv2
import numpy as np

def pnp(object_points, image_points):
    camera_matrix = np.array([[3400, 0, 1928], [0, 3400, 1090], [0, 0, 1]], dtype=np.float32) # Camera intrinsic parameters
    #
    dist_coeffs = np.zeros(4) # Distortion coefficients (assuming no distortion for simplicity)

    # camera_matrix = np.array([[1.11919004e+03, 0.00000000e+00, 6.32063538e+02], [0.00000000e+00, 1.11777157e+03, 3.68072720e+02], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    # dist_coeffs = np.array([[-3.83754695e-01, 5.46362102e-01, -2.01165698e-04,  1.94257579e-03,  -2.09598257e+00]])

    cam_coords = []
    for i, v in enumerate(object_points):
        success, R, T = cv2.solvePnP(v, image_points[i], camera_matrix, dist_coeffs) # Solve for rotation and translation vectors
        rotation_matrix, _ = cv2.Rodrigues(R) # Convert rotation vector to rotation matrix

        cam_coords.append(-np.dot(rotation_matrix.T, T)) # Calculate the camera position in world coordinates

    return cam_coords

if __name__ == "__main__":
    object_points = np.array([
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

    image_points = np.array([
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

    print(pnp(object_points, image_points))


# funky_matix = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]], dtype=np.float32)
# funky_matix[0:3,0:3] = rotation_matrix
# funky_matix[0:3,3:] = T
# funky_matix[3,3] = 1
# print(np.dot(funky_matix,object_points[0]))
