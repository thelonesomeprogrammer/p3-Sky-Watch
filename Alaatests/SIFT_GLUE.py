import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from LightGlue.lightglue.lightglue import LightGlue
from LightGlue.lightglue.utils import load_image, rbd
from LightGlue.lightglue import viz2d
from tqdm import tqdm
import cv2
from sklearn.cluster import DBSCAN
from collections import defaultdict
from matplotlib import cm
import time

def setup_device():
    """Sets up the computation device."""
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
    else:
        print("CUDA is not available. Using CPU.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def initialize_sift_extractor():
    """Initializes the SIFT feature extractor."""
    sift = cv2.SIFT_create(nfeatures=2000)
    print("SIFT extractor initialized.")
    return sift

def visualize_matches(image0_np, image1_np, feats0_np, feats1_np, matches, title=''):
    """Visualizes the matched keypoints between two images."""
    kpts0 = feats0_np["keypoints"]
    kpts1 = feats1_np["keypoints"]
    matched_indices = matches.get("matches0")

    if matched_indices is None or len(matched_indices) == 0:
        print("No matches found to visualize.")
        return

    # Extract matched keypoints
    valid = matched_indices > -1
    m_kpts0 = kpts0[valid]
    m_kpts1 = kpts1[matched_indices[valid]]

    print(f"Number of matches: {len(m_kpts0)}")

    # Create KeyPoint objects for matched keypoints
    img0_keypoints = [cv2.KeyPoint(x=float(k[0]), y=float(k[1]), size=1) for k in m_kpts0]
    img1_keypoints = [cv2.KeyPoint(x=float(k[0]), y=float(k[1]), size=1) for k in m_kpts1]

    # Create DMatch objects
    matches_draw = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _imgIdx=0, _distance=0) for i in range(len(img0_keypoints))]

    # Draw matches using OpenCV
    img_matches = cv2.drawMatches(
        image0_np, img0_keypoints,
        image1_np, img1_keypoints,
        matches_draw, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Display the matches
    plt.figure(figsize=(15, 10))
    plt.imshow(img_matches)
    plt.title(title)
    plt.axis('off')
    plt.show()

def load_images(images_dir, image_names):
    """Loads images from the specified directory."""
    images_path = Path(images_dir)
    loaded_images = {}
    for name in image_names:
        img_path = images_path / name
        if not img_path.exists():
            raise FileNotFoundError(f"Image '{name}' not found in {images_dir}.")
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        loaded_images[name] = img
    print(f"Images loaded from {images_dir}.")
    return loaded_images

def extract_features_sift(extractor, images):
    """Extracts features from the provided images using SIFT."""
    features = {}
    for name, img in images.items():
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = extractor.detectAndCompute(img_gray, None)
        if keypoints is None or descriptors is None:
            # No keypoints detected
            feats = {
                'keypoints': np.empty((0, 2)),
                'descriptors': np.empty((0, 128)),
                'scales': np.empty((0,)),
                'oris': np.empty((0,)),
            }
        else:
            # Convert keypoints to numpy array
            kpts = np.array([kp.pt for kp in keypoints], dtype=np.float32)
            # Extract scales and orientations from keypoints
            scales = np.array([kp.size for kp in keypoints], dtype=np.float32) * 0.5  # Radius = size * 0.5
            oris = np.array([kp.angle for kp in keypoints], dtype=np.float32)
            # Convert descriptors to numpy array
            descs = descriptors.astype(np.float32)  # [N, 128]
            # Package into features dict
            feats = {
                'keypoints': kpts,        # [N, 2]
                'descriptors': descs,     # [N, 128]
                'scales': scales,         # [N]
                'oris': oris,             # [N]
            }
        features[name] = feats
    print("Feature extraction with SIFT completed.")
    return features

def extract_features_from_tiles_sift(tiles, extractor):
    """Extracts features from each tile individually using SIFT."""
    features = []
    for tile, x, y in tqdm(tiles, desc="Extracting features from tiles"):
        tile_np = tile
        img_gray = cv2.cvtColor(tile_np, cv2.COLOR_RGB2GRAY)
        # Extract keypoints and descriptors
        keypoints, descriptors = extractor.detectAndCompute(img_gray, None)
        if keypoints is None or descriptors is None:
            # No keypoints detected in this tile
            feats = {
                'keypoints': np.empty((0, 2)),
                'descriptors': np.empty((0, 128)),
                'scales': np.empty((0,)),
                'oris': np.empty((0,)),
            }
        else:
            # Convert keypoints to numpy array (remain in tile-local coordinates)
            kpts = np.array([kp.pt for kp in keypoints], dtype=np.float32)
            # Extract scales and orientations from keypoints
            scales = np.array([kp.size for kp in keypoints], dtype=np.float32) * 0.5  # Radius = size * 0.5
            oris = np.array([kp.angle for kp in keypoints], dtype=np.float32)
            # Convert descriptors to numpy array
            descs = descriptors.astype(np.float32)  # [N, 128]
            # Package into features dict
            feats = {
                'keypoints': kpts,        # [N, 2]
                'descriptors': descs,     # [N, 128]
                'scales': scales,         # [N]
                'oris': oris,             # [N]
            }
        features.append(feats)
    print("Feature extraction for tiles completed.")
    return features

def match_features(matcher, feats0_np, feats1_np, device):
    """Matches features using LightGlue."""
    # Prepare features for LightGlue
    feats0 = {
        'keypoints': torch.from_numpy(feats0_np['keypoints']).unsqueeze(0).to(device).float(),
        'descriptors': torch.from_numpy(feats0_np['descriptors']).unsqueeze(0).to(device).float(),
        'scales': torch.from_numpy(feats0_np['scales']).unsqueeze(0).to(device).float(),
        'oris': torch.from_numpy(feats0_np['oris']).unsqueeze(0).to(device).float(),
    }
    feats1 = {
        'keypoints': torch.from_numpy(feats1_np['keypoints']).unsqueeze(0).to(device).float(),
        'descriptors': torch.from_numpy(feats1_np['descriptors']).unsqueeze(0).to(device).float(),
        'scales': torch.from_numpy(feats1_np['scales']).unsqueeze(0).to(device).float(),
        'oris': torch.from_numpy(feats1_np['oris']).unsqueeze(0).to(device).float(),
    }
    data = {
        "image0": feats0,
        "image1": feats1,
    }
    # Matching
    try:
        with torch.no_grad():
            matches = matcher(data)
    except Exception as e:
        raise RuntimeError(f"An error occurred during feature matching: {e}")

    # Post-processing
    processed_matches = {}
    for k, v in matches.items():
        if isinstance(v, torch.Tensor):
            processed_matches[k] = v.squeeze(0).cpu().numpy()
        else:
            processed_matches[k] = v

    return processed_matches

def split_image_into_tiles(image, tile_size=(512, 512), overlap=128):
    """Splits the image into overlapping tiles."""
    tiles = []
    h, w, _ = image.shape  # Assuming image shape is [H, W, C]
    stride_x = tile_size[1] - overlap
    stride_y = tile_size[0] - overlap
    for y in range(0, h - tile_size[0] + 1, stride_y):
        for x in range(0, w - tile_size[1] + 1, stride_x):
            tile = image[y:y + tile_size[0], x:x + tile_size[1], :]
            if tile.shape[0] == tile_size[0] and tile.shape[1] == tile_size[1]:
                tiles.append((tile, x, y))  # Store tile with its top-left position
    print(f"Image split into {len(tiles)} tiles.")
    return tiles

def view_all_clusters(sat_kpts, drone_kpts, scores, satellite_image_path, plot=False):
    """
    Visualizes all clusters of matched keypoints using DBSCAN.

    Args:
        sat_kpts (np.ndarray): Satellite image keypoints coordinates [N, 2].
        drone_kpts (np.ndarray): Drone image keypoints coordinates [N, 2].
        scores (np.ndarray, optional): Matching scores [N].
        satellite_image_path (str): Path to the satellite image file.
        plot (bool, optional): Whether to plot the clusters.

    Returns:
        cluster_dict (dict): Dictionary where keys are cluster labels and values are dictionaries containing:
            - 'sat_kpts': Satellite keypoints in the cluster [M, 2].
            - 'drone_kpts': Corresponding drone keypoints [M, 2].
            - 'scores': Matching scores for the selected matches [M].
    """
    sat_image_original_color = cv2.cvtColor(cv2.imread(str(satellite_image_path)), cv2.COLOR_BGR2RGB)
    image_height, image_width, _ = sat_image_original_color.shape

    # Verify that keypoints are within image bounds
    print(f"Satellite image size: width={image_width}, height={image_height}")

    # Perform DBSCAN clustering on satellite keypoints
    db = DBSCAN(eps=50, min_samples=2).fit(sat_kpts)
    labels = db.labels_

    # Get unique labels (clusters)
    unique_labels = set(labels)
    unique_labels.discard(-1)  # Remove noise label if present

    if len(unique_labels) == 0:
        print("No clusters found.")
        return {}

    cluster_dict = {}

    for cluster_label in unique_labels:
        # Select matches in the current cluster
        mask = labels == cluster_label
        cluster_sat_kpts = sat_kpts[mask]
        cluster_drone_kpts = drone_kpts[mask]
        cluster_scores = scores[mask] if scores is not None else None
        cluster_dict[cluster_label] = {
            'sat_kpts': cluster_sat_kpts,
            'drone_kpts': cluster_drone_kpts,
            'scores': cluster_scores
        }

    if plot and sat_image_original_color is not None:
        plt.figure(figsize=(10, 8))
        plt.imshow(sat_image_original_color)
        colors = cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        for cluster_label, color in zip(unique_labels, colors):
            mask = labels == cluster_label
            cluster_sat_kpts_plot = sat_kpts[mask]
            plt.scatter(cluster_sat_kpts_plot[:, 0], cluster_sat_kpts_plot[:, 1], color=color, s=30, label=f'Cluster {cluster_label}')
        plt.title("All Clusters of Matches")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.axis('equal')
        plt.legend()
        plt.show()

    return cluster_dict

def main():
    """Main function to execute the image processing pipeline."""

    start_time = time.time()
    # Configuration
    drone_images_dir = "path_to_drone_images"  # Replace with your drone images directory
    satellite_images_dir = "path_to_satellite_images"  # Replace with your satellite images directory
    drone_image_names = ["00359.png"]  # List of drone images
    satellite_image_names = ["SIFTOGB.jpg"]  # List of satellite images

    tile_size = (1500, 1500)  # Adjusted tile size for better performance
    overlap = 250  # Reduced overlap to decrease number of tiles
    num_tile_matches_to_visualize = 1

    # Setup
    device = setup_device()
    extractor = initialize_sift_extractor()
    matcher = LightGlue(features="sift", input_dim=128).eval().to(device)

    # Load Images
    drone_images = load_images(drone_images_dir, drone_image_names)
    satellite_images = load_images(satellite_images_dir, satellite_image_names)

    # Extract Features from Drone Images using SIFT
    drone_features = extract_features_sift(extractor, drone_images)

    # Split Satellite Images into Tiles and Extract Features
    satellite_tiles = {}
    satellite_features = {}
    for sat_image_name, sat_image in satellite_images.items():
        tiles = split_image_into_tiles(sat_image, tile_size, overlap)
        satellite_tiles[sat_image_name] = tiles
        print(f"Extracting features for satellite image '{sat_image_name}'")
        features_tiles = extract_features_from_tiles_sift(tiles, extractor)
        satellite_features[sat_image_name] = features_tiles

    for drone_image_name, drone_feats_np in drone_features.items():
        drone_image_np = drone_images[drone_image_name]
        # For each Satellite Image
        for sat_image_name in satellite_image_names:
            tiles = satellite_tiles[sat_image_name]
            features_tiles = satellite_features[sat_image_name]
            sat_image = satellite_images[sat_image_name]

            print(f"Matching drone image '{drone_image_name}' with satellite image '{sat_image_name}'")

            # Initialize lists to store matches and their confidence scores
            all_matches = []
            all_match_scores = []

            # Use tqdm for progress bar
            for i, tile_feats_np in enumerate(tqdm(features_tiles, desc=f"Matching tiles for '{sat_image_name}'")):
                # Match features
                matches = match_features(matcher, drone_feats_np, tile_feats_np, device)
                matched_indices = matches.get("matches0")
                # Assuming LightGlue provides 'matching_scores0' as confidence scores
                match_scores = matches.get("matching_scores0")  # Adjust key based on actual LightGlue output

                # Handle cases where matched_indices might be empty
                if matched_indices is not None and len(matched_indices) > 0:
                    valid = matched_indices > -1
                    num_valid = np.sum(valid)

                    if num_valid > 0:
                        # Extract valid matches
                        valid_matches = matched_indices[valid]

                        if match_scores is not None:
                            valid_scores = match_scores[valid]
                        else:
                            # If no scores are provided, assign a default score
                            valid_scores = np.ones_like(valid_matches, dtype=np.float32)

                        all_matches.append((i, matches))
                        all_match_scores.append(valid_scores)
                        # Debugging print
                        print(f"Tile {i} has {num_valid} matches.")

            # Define the function to get the number of valid matches
            def get_num_valid_matches(match_tuple):
                matches = match_tuple[1]
                matched_indices = matches.get("matches0")
                if matched_indices is not None and len(matched_indices) > 0:
                    valid = matched_indices > -1
                    return np.sum(valid)
                else:
                    return 0

            # Sort all_matches based on the number of valid matches
            all_matches_sorted = sorted(all_matches, key=get_num_valid_matches, reverse=True)

            print(f"Drone image '{drone_image_name}' matched with {len(all_matches_sorted)} satellite tiles.")

            # Verify the sorting
            for idx, (tile_idx, matches) in enumerate(all_matches_sorted):
                num_matches = get_num_valid_matches((tile_idx, matches))
                print(f"Tile {tile_idx} has {num_matches} matches.")

            # Visualize Matches for Selected Tile Pairs
            num_tiles_to_visualize = min(1, len(all_matches_sorted))  # Ensure we don't exceed available tiles
            for idx in range(num_tiles_to_visualize):
                tile_idx, matches = all_matches_sorted[idx]
                tile_np, x, y = tiles[tile_idx]
                tile_feats_np = features_tiles[tile_idx]
                title = f"Drone: {drone_image_name} | Satellite Tile: {tile_idx+1}"
                end_time = time.time()
                print(f"Time taken: {end_time - start_time:.2f} seconds")
                visualize_matches(drone_image_np, tile_np, drone_feats_np, tile_feats_np, matches, title=title)

            # Collect matches from all tiles and adjust satellite keypoints
            all_m_kpts0 = []  # Matched keypoints from drone image
            all_m_kpts1 = []  # Matched keypoints from satellite image
            all_match_scores_final = []  # Corresponding match scores

            for tile_idx, matches in all_matches_sorted:
                matched_indices = matches.get("matches0")
                match_scores = matches.get("matching_scores0")  # Adjust key based on actual LightGlue output
                valid = matched_indices > -1
                m_kpts0 = drone_feats_np["keypoints"][valid]
                m_kpts1 = features_tiles[tile_idx]["keypoints"][matched_indices[valid]]

                # Adjust satellite keypoints to global coordinates
                _, x, y = tiles[tile_idx]
                m_kpts1[:, 0] += x
                m_kpts1[:, 1] += y

                all_m_kpts0.append(m_kpts0)
                all_m_kpts1.append(m_kpts1)

                # Extract corresponding scores
                if match_scores is not None:
                    scores = match_scores[valid]
                else:
                    scores = np.ones(len(m_kpts0), dtype=np.float32)  # Default scores if not provided

                all_match_scores_final.append(scores)

            # Concatenate all matched keypoints and scores
            if all_m_kpts0:
                all_m_kpts0 = np.vstack(all_m_kpts0)
                all_m_kpts1 = np.vstack(all_m_kpts1)
                all_match_scores_final = np.concatenate(all_match_scores_final)
            else:
                print("No matches to plot.")
                continue  # Proceed to next satellite image if any

            # Path to the satellite image
            satellite_image_path = Path(satellite_images_dir) / sat_image_name

            # Visualize clusters on the satellite image
            view_all_clusters(
                sat_kpts=all_m_kpts1,
                drone_kpts=all_m_kpts0,
                scores=all_match_scores_final,
                satellite_image_path=satellite_image_path,
                plot=True
            )

if __name__ == "__main__":
    main()
