import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.cluster import DBSCAN
import matplotlib.cm as cm
import time
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

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
        longitude = boundaries[3] + (i[0] * pix_lat_long_eq[0])
        latitude = boundaries[0] - (i[1] * pix_lat_long_eq[1])
        loc_lat_long.append([longitude, latitude, 0])
    return loc_lat_long

# ---------------------------------------------------------
# PNP CLASS
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

def perform_clustering(satellite_points, eps=0.4, min_samples=5):
    scaler = StandardScaler()
    satellite_points_scaled = scaler.fit_transform(satellite_points)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(satellite_points_scaled)
    labels = db.labels_

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Estimated number of clusters: {n_clusters_}")

    return labels, scaler

def visualize_clusters(satellite_points, labels, image_path):
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    colors = cm.rainbow(np.linspace(0, 1, max(n_clusters,1)))
    sat_image_original_color = cv2.imread(image_path)
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(sat_image_original_color, cv2.COLOR_BGR2RGB))
    for label, color in zip(unique_labels, colors):
        if label == -1:
            continue  # Skip noise
        cluster_points = satellite_points[labels == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=color, s=10, label=f'Cluster {label}')
    plt.legend()
    plt.title('Clusters of Matched Points on Satellite Image')
    plt.axis('off')
    # save the cluster into the file clusters
    plt.savefig(f'clusters\\{image_path}', bbox_inches='tight')
    # plt.show()f

def estimate_camera_pose(clusters, satellite_points, drone_points, boundaries, sat_res, pnp_solver):
    cluster_confidence = {}
    for label, items in clusters.items():
        if len(items) < 4:
            print(f"Cluster {label} skipped: insufficient points for PnP.")
            continue
        sat_pts = np.array([item['sat_pt'] for item in items])
        drone_pts = np.array([item['drone_pt'] for item in items])
        cluster_geo_coords = xy_to_coords(boundaries, sat_res, sat_pts)
        object_points = np.array(cluster_geo_coords, dtype=np.float32).reshape(-1, 1, 3)
        image_points = np.array(drone_pts, dtype=np.float32).reshape(-1, 1, 2)
        cam = np.array(pnp_solver.pnp([object_points], [image_points])[0]).flatten()
        cluster_confidence[label] = {
            'cam_position': cam,
            'num_points': len(items)
        }
        print(f"Cluster {label}: Camera Position: {cam}, Points: {len(items)}")
    return cluster_confidence

def is_within_bounds(cam, boundaries):
    """
    Check if the camera position (longitude, latitude) is within the given boundaries.
    boundaries = [north, south, east, west]
    cam = [longitude, latitude, altitude]
    """
    cam_longitude, cam_latitude, _ = cam
    north, south, east, west = boundaries
    return (south < cam_latitude < north) and (west < cam_longitude < east)

def select_best_cluster(cluster_confidence, boundaries, last_known_position=None):
    """
    Select the best cluster among the available ones:
    - Discard clusters with position [0,0,0]
    - Discard clusters out of bounds
    - If last_known_position is provided, choose cluster closest to last_known_position.
    - If last_known_position is None, pick any valid cluster (e.g., the one with most points).
    """

    valid_clusters = []
    for label, info in cluster_confidence.items():
        cam_pos = info['cam_position']
        # Check for zero position
        if np.allclose(cam_pos, [0,0,0]):
            continue
        # Check if within boundaries
        if not is_within_bounds(cam_pos, boundaries):
            continue
        valid_clusters.append((label, info))

    if len(valid_clusters) == 0:
        print("No valid clusters found within expected bounds.")
        return None, None

    if last_known_position is None:
        # If no last known position, choose the cluster with the most points
        valid_clusters.sort(key=lambda x: x[1]['num_points'], reverse=True)
        best_cluster = valid_clusters[0]
        return best_cluster[0], best_cluster[1]['cam_position']
    else:
        # Choose cluster closest to last_known_position
        def distance(p1, p2):
            "harvesine distance"
            import math
            R = 6371000.0
            lat1, lon1 = p1[1], p1[0]
            lat2, lon2 = p2[1], p2[0]
            lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            distance = R * c
            return distance
        
        last_known_arr = np.array(last_known_position)

        "we need to change to sorting into using harvesine distance"

        valid_clusters.sort(key=lambda x: distance(x[1]['cam_position'], last_known_arr))
        best_cluster = valid_clusters[0]
        return best_cluster[0], best_cluster[1]['cam_position']

# ---------------------------------------------------------
# MAIN PROCESS
# ---------------------------------------------------------

import cv2
#img = cv2.imread('tysk.jpg')

def rotate_image(mat, angle):
    """
    Rotates an image (angle in radians) and expands image to avoid cropping
    """
   
    h=np.rad2deg(angle)
    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, h, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    #debug  
    # print(f"Rotated image by {h} degrees")
    # cv2.imshow('Rotated Image', rotated_mat)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return rotated_mat  # Also return rotation_mat

import os
def process_images(img1, drone_images_path_list, flight_data, boundaries_file, pnp_solver):
    start_time = time.time()
    os.makedirs('camera_positions', exist_ok=True)
    os.makedirs('SIFT_FLANN_TEST', exist_ok=True)
    os.makedirs('clusters', exist_ok=True)
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

    # FLANN_INDEX_KDTREE = 5
    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=1 )
    # search_params = dict(checks=50 )
    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    pred_geo = []
    # Set a manual starting position for the first run (e.g., known GPS start)
    # Format: [longitude, latitude, altitude]
    last_known_position = np.array([7.12, 50.75, 0.0])  # Example coordinate, adjust as needed
    for idx, drone_image_path in enumerate(drone_images_path_list):
        print(f"\nProcessing {drone_image_path}")

        # Extract image number from drone_image_path
        filename = os.path.basename(drone_image_path)  # e.g., '00359.png'
        image_number = os.path.splitext(filename)[0]   # e.g., '00359'
        image_number = image_number + '.png'
        # Retrieve rotation parameters from flight data
        if image_number not in flight_data:
            print(f"No flight data found for {image_number}. Skipping rotation correction.")
            pred_geo.append([idx, 0, 0])
            continue

        roll, pitch, yaw = flight_data[image_number]
        print(f"Applying rotation correction: Roll={roll}, Pitch={pitch}, Yaw={yaw}")

        all_satellite_match_points = []
        match_scores = []
        corresponding_drone_points = []

        # Load drone image
        drone = cv2.imread(drone_image_path, cv2.IMREAD_GRAYSCALE)
        drone= rotate_image(drone, -yaw)
        # print(f"Rotated image {drone_image_path} by {yaw} degrees")
        # cv2.imshow('Rotated Image', drone)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        kp_drone, des_drone = sift.detectAndCompute(drone, None)
        drone = cv2.GaussianBlur(drone, (3,3), 0)
        drone = apply_clahe(drone)

        top_matches = []
        matches_found = False

        # Iterate over all satellite tiles
        for tile_idx, (tile, kp_sat, des_sat, (x, y)) in enumerate(sat_tiles_features):
            if des_sat is None or kp_sat is None or len(kp_sat) < 4:
                continue

            matches = bf.knnMatch(des_drone, des_sat, k=2)
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

            all_satellite_match_points.extend(inlier_dst_pts_full.reshape(-1, 2))
            corresponding_drone_points.extend(inlier_src_pts.reshape(-1, 2))
            match_scores.extend([score] * len(inlier_dst_pts_full))
            matches_found = True
            top_matches.append((score, tile_idx, H, good_matches_filtered, kp_sat, (x, y)))

        if not matches_found:
            print("No sufficient matches found across all satellite tiles.")
            continue
        else:
            print(f"Matches found for {drone_image_path}")
            top_matches.sort(key=lambda x: x[0], reverse=True)
            top_N_matches = top_matches[:TOP_N_MATCHES]

            for score, tile_idx, H, good_matches_filtered, kp_sat, (x, y) in top_N_matches:
                tile = sat_tiles[tile_idx][0]
                match_img = cv2.drawMatches(drone, kp_drone,
                                            tile, kp_sat,
                                            good_matches_filtered, None,
                                            matchColor=(0, 255, 0),
                                            singlePointColor=None,
                                            flags=2)
                print(f"Tile ({x}, {y}) - Number of matches: {score}")
                end_time = time.time()
                print(f"Time taken so far: {end_time - start_time:.2f} seconds")
                saving_path = f"SIFT_FLANN_TEST\\{idx}.png"
                saved = cv2.imwrite(saving_path, match_img) 
                if saved:
                    print(f"Matches saved to {saving_path}")
                else:
                    print(f"Failed to save matches to {saving_path}")
                # Visualization disabled for brevity. Re-enable if needed:
                # plt.figure(figsize=(15, 10))
                # plt.imshow(match_img, cmap='gray')
                # plt.title(f"Drone Image: {drone_image_path} <--> Satellite Tile: ({x}, {y}) | Matches: {score}")
                # plt.axis('off')
                # plt.show()

        if len(all_satellite_match_points) > 0:
            satellite_points = np.array(all_satellite_match_points)
            match_scores_array = np.array(match_scores)
            drone_points_arr = np.array(corresponding_drone_points)

            labels, scaler = perform_clustering(satellite_points)
            unique_labels = set(labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            colors = cm.rainbow(np.linspace(0, 1, max(n_clusters,1)))
            sat_image_original_color = cv2.imread("vpair.jpg")
            plt.figure(figsize=(15, 10))
            plt.imshow(cv2.cvtColor(sat_image_original_color, cv2.COLOR_BGR2RGB))
            for label, color in zip(unique_labels, colors):
                if label == -1:
                    continue  # Skip noise
                cluster_points = satellite_points[labels == label]
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=color, s=10, label=f'Cluster {label}')
            plt.legend()
            plt.title('Clusters of Matched Points on Satellite Image')
            plt.axis('off')
            plt.savefig(f'clusters\\{idx}.jpg', bbox_inches='tight')
            # plt.show()

            # Organize clusters
            clusters = defaultdict(list)
            for label, sat_pt, d_pt, score in zip(labels, satellite_points, drone_points_arr, match_scores_array):
                if label == -1:
                    continue
                clusters[label].append({'sat_pt': sat_pt, 'drone_pt': d_pt, 'score': score})

            # Estimate camera pose for each cluster
            cluster_confidence = estimate_camera_pose(clusters, satellite_points, drone_points_arr, boundaries, sat_res, pnp_solver)

            # Select the best cluster based on last known position
            best_label, best_cam_pos = select_best_cluster(cluster_confidence, boundaries, last_known_position)
            if best_cam_pos is None:
                best_cam_pos = last_known_position
            # Save the camera position to a file with the image number 
            with open('camera_positions\\camera_positions.csv', 'a') as f:
                f.write(f"{image_number},{best_cam_pos[0]},{best_cam_pos[1]}\n")
                
            if best_cam_pos is not None:
                print(f"Selected Cluster {best_label} as best cluster with cam pos: {best_cam_pos}")
                last_known_position = best_cam_pos
            else:
                print("No valid cluster found. Keeping last known position unchanged.")
        else:
            print("No matches found to perform clustering or PnP.")

# ---------------------------------------------------------
# USAGE EXAMPLE
def read_flight_data(csv_file):
    """
    Reads the flight data CSV and returns a dictionary mapping filenames to their corresponding
    roll, pitch, yaw values.
    
    Parameters:
        csv_file (str): Path to the CSV file.
        
    Returns:
        dict: A dictionary where keys are filenames and values are tuples of (roll, pitch, yaw).
    """
    flight_data = {}
    with open(csv_file, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            filename = row['filename']
            roll = float(row['roll'])
            pitch = float(row['pitch'])
            yaw = float(row['yaw'])
            flight_data[filename] = (roll, pitch, yaw)
    return flight_data
# ---------------------------------------------------------
if __name__ == "__main__":
    satellite_image_path = "vpair.jpg"  # Replace with your satellite image path
    drone_images_path="drones\\"
    drone_images_path_list = []
    for i in range(359, 459):
        drone_images_path_list.append(f"{drone_images_path}{i:05}.png")

    flight_data_csv = "poses.csv"  # Replace with your flight data CSV path
    #load flight data
    flight_data = read_flight_data(flight_data_csv)
    boundaries_file = 'boundaries.txt'

    img1 = cv2.imread(satellite_image_path, cv2.IMREAD_GRAYSCALE)
    img1 = cv2.GaussianBlur(img1, (3,3), 0)
    img1 = apply_clahe(img1)

    # Define camera intrinsics and distortion coefficients here (example values)
    # Replace these with your actual calibration data
    camera_matrix = np.array([[750.62614972, 0, 402.41007535],
                              [0, 750.26301185, 292.98832147],
                              [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array([-0.11592226392258145, 0.1332261251415265, -0.00043977637330175616, 0.0002380609784102606], dtype=np.float32)

    pnp_solver = PnP(camera_matrix, dist_coeffs, ransac=True, ransac_iterations_count=100, ransac_reprojection_error=5.0)
    process_images(img1, drone_images_path_list, flight_data, boundaries_file, pnp_solver)
