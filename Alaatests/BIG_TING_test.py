import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.cluster import DBSCAN
import matplotlib.cm as cm
import time

# ---------------------------------------------------------
# HELPER FUNCTIONS FOR LOADING BOUNDARIES AND COORDINATES
# ---------------------------------------------------------

def load_bonuds(file):  # load the boundaries from a CSV
    with open(file) as csv_file:
        anno = []
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            anno.append(float(row[1]))
        return anno

def xy_to_coords(boundaries, sat_res, feature_coords):
    """
    Convert pixel coordinates (x, y) on the satellite image to geographic coordinates (longitude, latitude).
    boundaries: [north, south, east, west]
    sat_res: [height, width] of satellite image
    feature_coords: List of [x, y] pixel coordinates
    """
    north_south = abs(boundaries[0] - boundaries[1])
    east_west = abs(boundaries[2] - boundaries[3])
    pix_lat_long_eq = [east_west / sat_res[1], north_south / sat_res[0]]
    loc_lat_long = []
    for i in feature_coords:
        # longitude = west boundary + (x_pixels * degrees_per_pixel_longitude)
        # latitude = north boundary - (y_pixels * degrees_per_pixel_latitude)
        longitude = boundaries[3] + (i[0] * pix_lat_long_eq[0])
        latitude = boundaries[0] - (i[1] * pix_lat_long_eq[1])
        loc_lat_long.append([longitude, latitude, 0])
    return loc_lat_long
def lonlat_to_ecef(lon, lat, alt=0):
    # WGS84 ellipsoid constants
    a = 6378137.0  # Semi-major axis (meters)
    e2 = 6.69437999014e-3  # Square of eccentricity

    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    N = a / np.sqrt(1 - e2 * (np.sin(lat_rad) ** 2))

    X = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    Y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    Z = (N * (1 - e2) + alt) * np.sin(lat_rad)

    return X, Y, Z

def xy_to_ecef_coords(boundaries, sat_res, feature_coords):
    loc_geo_coords = xy_to_coords(boundaries, sat_res, feature_coords)
    loc_ecef_coords = []
    for coord in loc_geo_coords:
        lon, lat, alt = coord  # alt is 0 in this case
        X, Y, Z = lonlat_to_ecef(lon, lat, alt)
        loc_ecef_coords.append([X, Y, Z])
    return loc_ecef_coords
# ---------------------------------------------------------
# PNP CLASS EXAMPLE (ADJUST AS NEEDED)
# ---------------------------------------------------------
class PnP:
    def __init__(self, camera_matrix, dist_coeffs, ransac=True, ransac_iterations_count=100, ransac_reprojection_error=8.0):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.ransac = ransac
        self.ransac_iterations_count = ransac_iterations_count
        self.ransac_reprojection_error = ransac_reprojection_error

    def pnp(self, object_points, image_points):
        # object_points: list of arrays of shape (N, 1, 3)
        # image_points: list of arrays of shape (N, 1, 2)
        cam_coords = []
        for i, v in enumerate(object_points):
            if v.shape[0] < 4:  # Check if we have enough points
                cam_coords.append([0,0,0])
                continue
            if self.ransac:
                success, R, T, inlie = cv2.solvePnPRansac(v, image_points[i], self.camera_matrix, self.dist_coeffs, 
                                                          iterationsCount=self.ransac_iterations_count,
                                                          reprojectionError=self.ransac_reprojection_error)
            else:
                success, R, T = cv2.solvePnP(v, image_points[i], self.camera_matrix, self.dist_coeffs)
            if not success:
                # Failed to estimate pose
                cam_coords.append([0,0,0])
                continue
            rotation_matrix, _ = cv2.Rodrigues(R)
            cam_pos = -np.dot(rotation_matrix.T, T)
            cam_coords.append(cam_pos)
        return cam_coords

# ---------------------------------------------------------
# IMAGE PROCESSING AND CLUSTERING PARAMETERS
# ---------------------------------------------------------
TILE_SIZE = 1500
TILE_OVERLAP = 300
TOP_N_MATCHES = 1

def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(4,4))
    clahe_image = clahe.apply(image)
    return clahe_image

def divide_into_tiles(image, tile_size, overlap):
    tiles = []
    h, w = image.shape[:2]
    step = tile_size - overlap
    for y in range(0, h - tile_size + 1, step):
        for x in range(0, w - tile_size + 1, step):
            tile = image[y:y + tile_size, x:x + tile_size]
            tiles.append((tile, (x, y)))
    return tiles

# ---------------------------------------------------------
# MAIN PROCESS
# ---------------------------------------------------------
def process_images(img1, drone_images_path_list, flight_data_csv, boundaries_file, pnp_solver):
    start_time = time.time()

    # Load boundaries and prepare sat_res
    boundaries = load_bonuds(boundaries_file)
    sat_res = [img1.shape[0], img1.shape[1]]

    # Divide the satellite image into tiles
    sat_tiles = divide_into_tiles(img1, TILE_SIZE, TILE_OVERLAP)
    print(f"Total satellite tiles: {len(sat_tiles)}")

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Precompute SIFT features for all satellite tiles
    sat_tiles_features = []
    for tile_idx, (tile, (x, y)) in enumerate(sat_tiles):
        kp_sat, des_sat = sift.detectAndCompute(tile, None)
        sat_tiles_features.append((tile, kp_sat, des_sat, (x, y)))

    FLANN_INDEX_KDTREE = 5
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # We'll store the best matches (based on score) for each drone image
    # so we can run PnP on the best match directly.
    best_matches_for_drone = []

    # Iterate over drone images
    for idx, drone_image_path in enumerate(drone_images_path_list):
        print(f"\nProcessing {drone_image_path}")

        # Load drone image
        drone = cv2.imread(drone_image_path, cv2.IMREAD_GRAYSCALE)
        kp_drone, des_drone = sift.detectAndCompute(drone, None)
        drone = cv2.GaussianBlur(drone, (3,3), 0)
        drone = apply_clahe(drone)

        top_matches = []
        matches_found = False

        # Iterate over all satellite tiles
        for tile_idx, (tile, kp_sat, des_sat, (x, y)) in enumerate(sat_tiles_features):
            if des_sat is None or kp_sat is None or len(kp_sat) < 4:
                continue

            matches = flann.knnMatch(des_drone, des_sat, k=2)
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

            if len(good_matches) < 4:
                continue

            src_pts = np.float32([kp_drone[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_sat[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(src_pts, dst_pts,
                                         method=cv2.RANSAC,
                                         ransacReprojThreshold=5,
                                         maxIters=10000,
                                         confidence=0.9999)
            if H is None or mask is None:
                continue

            matches_mask = mask.ravel().tolist()
            good_matches_filtered = [good_matches[i] for i in range(len(good_matches)) if matches_mask[i]]

            if len(good_matches_filtered) < 4:
                continue

            score = len(good_matches_filtered)
            inlier_dst_pts = dst_pts[mask.ravel() == 1]
            inlier_src_pts = src_pts[mask.ravel() == 1]

            # Convert tile-local coordinates to full image coordinates for satellite
            inlier_dst_pts_full = inlier_dst_pts + np.array([x, y])

            matches_found = True
            top_matches.append((score, tile_idx, H, good_matches_filtered, kp_sat, (x, y), inlier_dst_pts_full, inlier_src_pts))

        if not matches_found:
            print("No sufficient matches found across all satellite tiles.")
        else:
            print(f"Matches found for {drone_image_path}")
            top_matches.sort(key=lambda x: x[0], reverse=True)
            top_N_matches = top_matches[:TOP_N_MATCHES]

            # For each top match, we can now directly run PnP:
            for score, tile_idx, H, good_matches_filtered, kp_sat, (x, y), inlier_dst_pts_full, inlier_src_pts in top_N_matches:
                tile = sat_tiles[tile_idx][0]
                print(f"Tile ({x}, {y}) - Number of matches: {score}")
                end_time = time.time()
                print(f"Time taken so far: {end_time - start_time:.2f} seconds")

                # Optional visualization
                match_img = cv2.drawMatches(drone, kp_drone,
                                            tile, kp_sat,
                                            good_matches_filtered, None,
                                            matchColor=(0, 255, 0),
                                            singlePointColor=None,
                                            flags=2)
                plt.figure(figsize=(15, 10))
                plt.imshow(match_img, cmap='gray')
                plt.title(f"Drone Image: {drone_image_path} <--> Satellite Tile: ({x}, {y}) | Matches: {score}")
                plt.axis('off')
                plt.show()

                # ------------------------------
                # Run PnP directly here
                # ------------------------------
                # Convert satellite pixel coords to geographic coords
                cluster_geo_coords = xy_to_coords(boundaries, sat_res, inlier_dst_pts_full.reshape(-1, 2))
                
                # object_points: Nx1x3
                object_points = np.array(cluster_geo_coords, dtype=np.float32).reshape(-1, 1, 3)
                image_points = np.array(inlier_src_pts, dtype=np.float32).reshape(-1, 1, 2)

                # Run PnP
                cam = np.array(pnp_solver.pnp([object_points], [image_points])[0])
                print("Estimated camera position (in geographic reference):", cam)

                # Check if camera position is within expected bounds (if needed)
                # Assuming: cam = [longitude, latitude, 0]
                if (cam[1] < boundaries[0] and cam[1] > boundaries[1] and
                    cam[0,0] < boundaries[2] and cam[0,0] > boundaries[3]):
                    print("Camera position is within expected bounds.")
                else:
                    print("Camera position is out of expected bounds.")

    # Since we are no longer using clustering, remove or comment out the clustering code below this point.
    # The code that performed DBSCAN, computed clusters, and plotted them can be removed.
    # -----------------------------------------
    # REMOVED CLUSTERING AND PNP WITH CLUSTERS
    # -----------------------------------------

# ---------------------------------------------------------
# USAGE EXAMPLE
# ---------------------------------------------------------
if __name__ == "__main__":
    satellite_image_path = "vpair.jpg"  # Replace with your satellite image path
    drone_images_path_list = [
        "00359.png", 
        # Add more drone images here if desired
    ]
    flight_data_csv = "poses.csv"  # Replace with your flight data CSV path
    boundaries_file = 'boundaries.txt'

    img1 = cv2.imread(satellite_image_path, cv2.IMREAD_GRAYSCALE)
    img1 = cv2.GaussianBlur(img1, (3,3), 0)
    img1 = apply_clahe(img1)

    
    
    camera_matrix =np.array([[750.62614972, 0, 402.41007535], [0, 750.26301185, 292.98832147], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array([-0.11592226392258145, 0.1332261251415265, -0.00043977637330175616, 0.0002380609784102606], dtype=np.float32)


    pnp_solver = PnP(camera_matrix, dist_coeffs, ransac=True, ransac_iterations_count=100, ransac_reprojection_error=5.0)
    process_images(img1, drone_images_path_list, flight_data_csv, boundaries_file, pnp_solver)
