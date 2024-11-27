import cv2
import numpy as np

class PnP:
    def __init__(self, camera_matrix, dist_coeffs, ransac = False, ransac_iterations_count = 1000, ransac_reprojection_error = 2.0):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.ransac = ransac
        self.ransac_iterations_count = ransac_iterations_count
        self.ransac_reprojection_error = ransac_reprojection_error

    def pnp(self, object_points, image_points):
        cam_coords = []
        for i, v in enumerate(object_points):
            if self.ransac:
                success, R, T, inlie = cv2.solvePnPRansac(v, image_points[i], self.camera_matrix, self.dist_coeffs, 
                                                          iterationsCount = self.ransac_iterations_count, 
                                                          reprojectionError = self.ransac_reprojection_error) # Solve for rotation and translation vectors
            else:
                success, R, T = cv2.solvePnP(v, image_points[i], self.camera_matrix, self.dist_coeffs) # Solve for rotation and translation vectors
            rotation_matrix, _ = cv2.Rodrigues(R) # Convert rotation vector to rotation matrix

            cam_coords.append(-np.dot(rotation_matrix.T, T)) # Calculate the camera position in world coordinates

        return cam_coords
    

    @classmethod
    def vsky_init(cls, ransac = False, ransac_iterations_count = 1000, ransac_reprojection_error = 2.0):
        camera_matrix = np.array([[3400, 0, 1929], [0, 3400, 1091], [0, 0, 1]], dtype=np.float32) # Camera intrinsic parameters
        distortion_coeffs = np.zeros(4) # Distortion coefficients (assuming no distortion for simplicity)
        camera_matrix,  droi  = cv2.getOptimalNewCameraMatrix(camera_matrix,distortion_coeffs,(3840,2160),1,(3840,2160))
        return cls(camera_matrix, distortion_coeffs, ransac, ransac_iterations_count, ransac_reprojection_error)

    @classmethod
    def dsky_init(cls, ransac = False, ransac_iterations_count = 1000, ransac_reprojection_error = 2.0):
        camera_matrix = np.array([[3.37557838e+03, 0.00000000e+00, 1.90060064e+03], [0.00000000e+00, 3.37627765e+03, 1.02106476e+03], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        distortion_coeffs = np.array([[-5.68718021e-02, -3.77364195e+00, -2.33949181e-03,  9.64380021e-03,   1.04120413e+01]])
        camera_matrix,  droi  = cv2.getOptimalNewCameraMatrix(camera_matrix,distortion_coeffs,(3840,2160),1,(3840,2160))
        return cls(camera_matrix, distortion_coeffs, ransac, ransac_iterations_count, ransac_reprojection_error)

    @classmethod
    def vpair_init(cls, ransac = False, ransac_iterations_count = 1000, ransac_reprojection_error = 2.0):
        camera_matrix = np.array([[750.62614972, 0, 402.41007535], [0, 750.26301185, 292.98832147], [0, 0, 1]])
        distortion_coeffs = np.array([-0.11592226392258145, 0.1332261251415265, -0.00043977637330175616, 0.0002380609784102606])
        return cls(camera_matrix, distortion_coeffs, ransac, ransac_iterations_count, ransac_reprojection_error)


if __name__ == "__main__":
    pnpv = PnP.vsky_init()
    pnpd = PnP.dsky_init()
    pnpvpair = PnP.vpair_init()
    sky_object_points = np.array([
        [
            [ 9.86831,  56.887398,  0.      ],
            [ 9.869039, 56.884796,  0.      ],
            [ 9.868003, 56.88553,   0.      ],
            [ 9.86891,  56.886036,  0.      ],
        ],[
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
            [2656., 1621.],
            [3143., 1996.],
            [2240., 1891.],
            [2761., 1850.],
        ],[
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
    pnpv.pnp(sky_object_points, sky_image_points)
    pnpd.pnp(sky_object_points, sky_image_points)

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
    pnpvpair.pnp(vpair_object_points, vpair_image_points)
