import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.cluster import DBSCAN
import matplotlib.cm as cm
import time # For measuring time taken for processing
# Constants (adjust these according to your camera specifications)
TILE_SIZE = 1500                   # Size of each tile in pixels (adjust based on your needs)
TILE_OVERLAP = 300                 # Overlap between tiles in pixels (optional)
TOP_N_MATCHES = 3                  # Number of top matching tiles to retrieve

# Function to apply CLAHE
def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(4,4))
    clahe_image = clahe.apply(image)
    return clahe_image

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

def process_images(img1, drone_images_path_list, flight_data_csv):
    start_time = time.time()
    # Load satellite image
    sat_image_original = img1

    # Divide the satellite image into tiles
    sat_tiles = divide_into_tiles(sat_image_original, TILE_SIZE, TILE_OVERLAP)
    print(f"Total satellite tiles: {len(sat_tiles)}")

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Precompute SIFT features for all satellite tiles
    sat_tiles_features = []
    for tile_idx, (tile, (x, y)) in enumerate(sat_tiles):
        kp_sat, des_sat = sift.detectAndCompute(tile, None)
        sat_tiles_features.append((tile, kp_sat, des_sat, (x, y)))

    FLANN_INDEX_KDTREE = 5
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=1 )
    search_params = dict(checks=50 )
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    all_satellite_match_points = []
    match_scores = []

    # Iterate over drone images
    for idx, drone_image_path in enumerate(drone_images_path_list):
        print(f"\nProcessing {drone_image_path}")

        # Load drone image
        drone = cv2.imread(drone_image_path, cv2.IMREAD_GRAYSCALE)
        # drone = cv2.GaussianBlur(drone, (3,3), 0)
        # drone = apply_clahe(drone)

        # Detect and compute features in drone image
        kp_drone, des_drone = sift.detectAndCompute(drone, None)

        # List to keep track of matches for each tile
        top_matches = []  # Each element: (score, tile_idx, H, good_matches_filtered, kp_sat, (x, y))
        matches_found = False  # Flag to track if any matches are found

        # Iterate over all satellite tiles
        for tile_idx, (tile, kp_sat, des_sat, (x, y)) in enumerate(sat_tiles_features):
            if des_sat is None or len(kp_sat) < 4:
                # Not enough features in this tile
                continue

            # matches = bf.knnMatch(des_drone, des_sat, k=2)
            matches=flann.knnMatch(des_drone,des_sat,k=2)
            # Apply Lowe's ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

            if len(good_matches) < 4:
                continue  # Not enough matches to compute homography

            # Extract matched keypoints
            src_pts = np.float32([kp_drone[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_sat[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Compute homography
            H, mask = cv2.findHomography(
                src_pts,
                dst_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=5,
                maxIters=10000,
                confidence=0.9999,
            )
            if mask is None:
                continue  # Homography computation failed

            # Filter matches using the homography mask
            matches_mask = mask.ravel().tolist()
            good_matches_filtered = [good_matches[i] for i in range(len(good_matches)) if matches_mask[i]]

            if len(good_matches_filtered) < 4:
                continue  # Not enough inliers after RANSAC

            # Compute score (number of inliers)
            score = len(good_matches_filtered)

            inlier_dst_pts = dst_pts[mask.ravel() == 1]
            inlier_dst_pts_full = inlier_dst_pts + np.array([x, y])

            # Collect these points
            all_satellite_match_points.extend(inlier_dst_pts_full.reshape(-1, 2))

            # Collect the score for these points
            match_scores.extend([score] * len(inlier_dst_pts_full))

            matches_found = True  # At least one valid match found

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
                end_time = time.time()
                print(f"Time taken: {end_time - start_time:.2f} seconds")
                plt.figure(figsize=(15, 10))
                plt.imshow(match_img, cmap='gray')
                plt.title(f"Drone Image: {drone_image_path} <--> Satellite Tile: ({x}, {y}) | Matches: {score}")
                plt.axis('off')
                plt.show()

    # After processing all drone images and tiles, perform clustering
    if len(all_satellite_match_points) > 0:
        # Convert to numpy array
        satellite_points = np.array(all_satellite_match_points)
        match_scores_array = np.array(match_scores)

        # Perform clustering
        db = DBSCAN(eps=25, min_samples=3).fit(satellite_points)
        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"Number of clusters found: {n_clusters}")

        # Collect clusters and compute confidence
        clusters = defaultdict(list)
        for label, point, score in zip(labels, satellite_points, match_scores_array):
            if label == -1:
                continue  # Noise point
            clusters[label].append({'point': point, 'score': score})

        cluster_confidence = {}
        for label, items in clusters.items():
            total_score = sum(item['score'] for item in items)
            num_points = len(items)
            average_score = total_score / num_points
            cluster_points = np.array([item['point'] for item in items])
            center = np.mean(cluster_points, axis=0)
            cluster_confidence[label] = {
                'total_score': total_score,
                'num_points': num_points,
                'average_score': average_score,
                'center': center
            }

        # Plot clusters
        colors = cm.rainbow(np.linspace(0, 1, n_clusters))
        sat_image_original_color = cv2.imread("SIFTOGBF.jpg")
        plt.figure(figsize=(15, 10))
        plt.imshow(sat_image_original_color, cmap='gray')
        for label, color in zip(clusters.keys(), colors):
            cluster_points = np.array([item['point'] for item in clusters[label]])
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=color, s=10, label=f'Cluster {label}')
            center = cluster_confidence[label]['center']
            confidence = cluster_confidence[label]['average_score']
            plt.text(center[0], center[1], f'{confidence:.2f}', color='white', fontsize=12, ha='center', va='center')

        plt.legend()
        plt.title('Clusters of Matched Points on Satellite Image')
        plt.axis('off')
        plt.show()
    else:
        print("No matches found to perform clustering.")

if __name__ == "__main__":
    satellite_image_path = "SIFTOGB.jpg"  # Replace with your satellite image path
    drone_images_path_list = [
        "00359.png",
        # "vpair\\00367.png",
        # "vpair\\00368.png",
        # "vpair\\00369.png",
        # "vpair\\00370.png",
        # "vpair\\00371.png",
    ]
    flight_data_csv = "poses.csv"  # Replace with your flight data CSV path

    img1 = cv2.imread(satellite_image_path, cv2.IMREAD_GRAYSCALE)
    img1 = cv2.GaussianBlur(img1, (3,3), 0)
    img1 = apply_clahe(img1)

    process_images(img1, drone_images_path_list, flight_data_csv)
