import cv2
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms
from skimage.feature import local_binary_pattern
import os

# Constants (adjust these according to your camera specifications)
FOCAL_LENGTH = 4.5 / 1000          # Focal length in mm
SENSOR_WIDTH = 6.17 / 1000         # Sensor width in mm
MAX_IMAGE_DIMENSION = 8000         # Maximum allowed image dimension (adjust as needed)
TILE_SIZE = 1000                   # Size of each tile in pixels (adjust based on your needs)
TILE_OVERLAP = 500                 # Overlap between tiles in pixels (optional)
TOP_N_MATCHES = 10                 # Number of top matching tiles to retrieve

# LBP Parameters
             # Weight for LBP similarity in combined score
SIFT_WEIGHT = 1                 # Weight for SIFT inlier ratio in combined score

# Function to read the flight data CSV file

# Function to apply CLAHE
def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(image)
    return clahe_image

# Function to extract edges using Canny Edge Detector
def extract_edges(image):
    edges = cv2.Canny(image, threshold1=170, threshold2=200)
    return edges

# Function to divide image into tiles
def divide_into_tiles(image, tile_size, overlap):
    tiles = []
    h, w = image.shape[:2]
    step = tile_size - overlap
    for y in range(0, h - tile_size + 1, step):
        for x in range(0, w - tile_size + 1, step):
            tile = image[y:y + tile_size, x:x + tile_size]
            tiles.append((tile, (x, y)))
    return tiles


# Main processing function
def process_images(satellite_image_path, drone_images_path_list, flight_data_csv):
    # Load satellite image
    sat_image_original = cv2.imread(satellite_image_path, cv2.IMREAD_GRAYSCALE)
    if sat_image_original is None:
        print("Error loading satellite image.")
        return

    # Read flight data
    try:
        flight_data = pd.read_csv(flight_data_csv)
        # Process flight data as needed
    except Exception as e:
        print(f"Error reading flight data CSV: {e}")
        return

    # Placeholder: Satellite GSD (meters per pixel)
    satellite_gsd = 0.5  # Example value (adjust based on your data)

    # Divide the satellite image into tiles
    sat_tiles = divide_into_tiles(sat_image_original, TILE_SIZE, TILE_OVERLAP)
    print(f"Total satellite tiles: {len(sat_tiles)}")

    # Initialize SIFT detector
    sift = cv2.SIFT_create(nfeatures=1000, nOctaveLayers=3,
                           contrastThreshold=0.06, edgeThreshold=10, sigma=1.6)

    # Initialize FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Iterate over drone images
    for idx, drone_image_path in enumerate(drone_images_path_list):
        print(f"\nProcessing {drone_image_path}")

        # Load drone image
        drone_image = cv2.imread(drone_image_path)
        if drone_image is None:
            print(f"Error loading {drone_image_path}")
            continue

        # Convert drone image to grayscale
        drone = cv2.cvtColor(drone_image, cv2.COLOR_BGR2GRAY)

        # Drone image dimensions
        image_width_px = drone.shape[1]
        image_height_px = drone.shape[0]
        scale_factor = 1  
        print(f"Scale factor to scale down satellite image: {scale_factor}")

        # Detect and compute features in drone image
        kp_drone, des_drone = sift.detectAndCompute(drone, None)

        # Extract keypoint coordinates from kp_drone
        kp_drone_pts = np.array([kp.pt for kp in kp_drone], dtype=np.float32)

        # List to keep track of matches for each tile
        top_matches = []  # Each element: (score, tile_idx, H, good_matches_filtered, kp_sat, (x, y))
        matches_found = False  # Flag to track if any matches are found

        # Iterate over all satellite tiles sequentially
        for tile_idx, (tile, (x, y)) in enumerate(sat_tiles):
            kp_sat, des_sat = sift.detectAndCompute(tile, None)
            if des_sat is None or len(kp_sat) < 4:
                # Not enough features in this tile
                continue
           
            # Extract keypoint coordinates from kp_sat
            kp_sat_pts = np.array([kp.pt for kp in kp_sat], dtype=np.float32)
            matches = flann.knnMatch(des_drone, des_sat, k=2)
           
            # Apply Lowe's ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

            if len(good_matches) < 4:
                continue  # Not enough matches to compute homography

            # Extract matched keypoints
            src_pts = kp_drone_pts[[m.queryIdx for m in good_matches]]
            dst_pts = kp_sat_pts[[m.trainIdx for m in good_matches]]

            # Compute homography
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
           
            if mask is None:
                continue  # Homography computation failed

            # Filter matches using the homography mask
            matches_mask = mask.ravel().tolist()
            good_matches_filtered = [good_matches[i] for i in range(len(good_matches)) if matches_mask[i]]

            if len(good_matches_filtered) < 4:
                continue  # Not enough inliers after RANSAC

            matches_found = True  # At least one valid match found

            # Compute score (number of inliers)
            score = len(good_matches_filtered)
            top_matches.append((score, tile_idx, H, good_matches_filtered, kp_sat, (x, y)))

        if not matches_found:
            print("No sufficient matches found across all satellite tiles.")
        else:
            print(f"Matches found for {drone_image_path}")
            # Sort the matches by score (number of inliers) in descending order
            top_matches.sort(key=lambda x: x[0], reverse=True)
            
            # Select the top N matches
            top_N_matches = top_matches[:TOP_N_MATCHES]
            
            # Process top N matches
            for score, tile_idx, H, good_matches_filtered, kp_sat, (x, y) in top_N_matches:
                # Get the satellite tile
                tile = sat_tiles[tile_idx][0]
                
                # Draw matches
                match_img = cv2.drawMatches(drone, kp_drone,
                                            tile, kp_sat,
                                            good_matches_filtered, None,
                                            matchColor=(0, 255, 0),
                                            singlePointColor=None,
                                            flags=2)

                # Display matches (optional)
                print(f"Tile ({x}, {y}) - Number of matches: {score}")

                plt.figure(figsize=(15, 10))
                plt.imshow(match_img, cmap='gray')
                plt.title(f"Drone Image: {drone_image_path} <--> Satellite Tile: ({x}, {y}) | Matches: {score}")
                plt.axis('off')
                plt.show()


            # Optionally, save the results
            # save_dir = "match_results"
            # os.makedirs(save_dir, exist_ok=True)
            # match_image_path = os.path.join(save_dir, f"{os.path.splitext(os.path.basename(drone_image_path))[0]}_match_tile_{x}_{y}.jpg")
            # cv2.imwrite(match_image_path, match_img)

            # Collect top matches based on your scoring criteria
            # Example: Append to top_matches list
            # combined_score = LBP_WEIGHT * lbp_score + SIFT_WEIGHT * sift_score
            # top_matches.append((combined_score, tile, tile_idx, H, good_matches_filtered, kp_sat))

        if not matches_found:
            print("No sufficient matches found across all satellite tiles.")
        else:
            print(f"Matches found for {drone_image_path}")
            # Process top_matches as needed, e.g., sort and select top N

if __name__ == "__main__":
    satellite_image_path = "tysk.jpg"  # Replace with your satellite image path
    drone_images_path_list = [
        "1313.jpeg"
    ]
    flight_data_csv = "poses.csv"  # Replace with your flight data CSV path

    process_images(satellite_image_path, drone_images_path_list, flight_data_csv)
