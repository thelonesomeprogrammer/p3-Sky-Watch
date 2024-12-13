import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
from tqdm import tqdm  # For progress bars
import time  # For measuring time taken for processing
import matplotlib.cm as cm
from LightGlue.lightglue import SuperPoint  # Adjust this import as per your environment
from sklearn.cluster import DBSCAN  # Import DBSCAN for clustering¨'
from collections import defaultdict
from LightGlue.lightglue import SuperPoint  # Adjust this import as per your environment


def view_all_clusters(sat_kpts, drone_kpts, scores, satellite_image_path, plot=False):
    """
    Visualizes all clusters of matched keypoints using DBSCAN.

    Args:
        sat_kpts (np.ndarray): Satellite image keypoints coordinates [N, 2].
        drone_kpts (np.ndarray): Drone image keypoints coordinates [N, 2].
        scores (np.ndarray, optional): Matching scores [N].
        sat_img (np.ndarray, optional): Satellite image as a NumPy array for plotting.
        plot (bool, optional): Whether to plot the clusters.

    Returns:
        cluster_dict (dict): Dictionary where keys are cluster labels and values are dictionaries containing:
            - 'sat_kpts': Satellite keypoints in the cluster [M, 2].
            - 'drone_kpts': Corresponding drone keypoints [M, 2].
            - 'scores': Matching scores for the selected matches [M].
    """
    sat_image_original_color = cv2.cvtColor(cv2.imread(satellite_image_path), cv2.COLOR_BGR2RGB)
    # Perform DBSCAN clustering on satellite keypoints
    db = DBSCAN(eps=10, min_samples=1).fit(sat_kpts)
    labels = db.labels_

    # Get unique labels (clusters)
    unique_labels = set(labels)
    unique_labels.discard(-1)  # Remove noise label

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
        plt.imshow(sat_image_original_color, cmap='gray')
        colors = cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        for cluster_label, color in zip(unique_labels, colors):
            mask = labels == cluster_label
            cluster_sat_kpts = sat_kpts[mask]
            plt.scatter(cluster_sat_kpts[:, 0], cluster_sat_kpts[:, 1], color=color, s=30, label=f'Cluster {cluster_label}')
        plt.title("All Clusters of Matches")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()
        plt.show()

    return cluster_dict



def filter_matches_with_ransac(kpts0, kpts1, matches, ransac_thresh=2.0):
    """
    Filters matches using RANSAC to find inliers based on homography.

    Args:
        kpts0 (np.ndarray): Keypoints from the first image of shape [N0, 2].
        kpts1 (np.ndarray): Keypoints from the second image of shape [N1, 2].
        matches (dict): Dictionary containing 'matches0' (matched indices) and 'matching_scores0'.
        ransac_thresh (float): RANSAC reprojection threshold.

    Returns:
        inlier_matches (list of tuples): List of tuples containing matched keypoint indices that are inliers.
        homography (np.ndarray or None): Estimated homography matrix if successful, else None.
    """
    import cv2

    # Extract matched keypoint coordinates
    matched_kpts0 = []
    matched_kpts1 = []
    for i, match_idx in enumerate(matches['matches0']):
        if match_idx == -1:
            continue  # Skip unmatched keypoints
        matched_kpts0.append(kpts0[i])
        matched_kpts1.append(kpts1[match_idx])

    if len(matched_kpts0) < 4:
        print("Not enough matches to compute homography.")
        return [], None  # Homography requires at least 4 matches

    matched_kpts0 = np.array(matched_kpts0)
    matched_kpts1 = np.array(matched_kpts1)

    # Compute homography using RANSAC
    homography, mask = cv2.findHomography(
        matched_kpts0, matched_kpts1, cv2.RANSAC, ransac_thresh
    )

    if homography is None:
        print("Homography could not be computed.")
        return [], None

    # mask is a byte array where 1 indicates inlier and 0 indicates outlier
    inlier_mask = mask.ravel().astype(bool)
    inlier_matches = []
    for i, is_inlier in enumerate(inlier_mask):
        if is_inlier:
            inlier_matches.append((i, matches['matches0'][i]))

    num_inliers = np.sum(inlier_mask)
    print(f"Number of inlier matches after RANSAC: {num_inliers}/{len(matched_kpts0)}")

    return inlier_matches, homography


def setup_device():
    """Sets up the computation device."""
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
    else:
        print("CUDA is not available. Using CPU.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def initialize_models(device, max_keypoints=2048):
    """Initializes the feature extractor."""
    extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)
    print("Extractor initialized.")
    return extractor

def load_image(path):
    """Loads an image and converts it to a torch tensor."""
    from PIL import Image
    import torchvision.transforms as transforms
    image = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    img_tensor = transform(image)
    return img_tensor

def load_images(images_dir, image_names):
    """Loads images from the specified directory."""
    images_path = Path(images_dir)
    loaded_images = {}
    for name in image_names:
        img_path = images_path / name
        if not img_path.exists():
            raise FileNotFoundError(f"Image '{name}' not found in {images_dir}.")
        loaded_images[name] = load_image(str(img_path))
    # Convert images to grayscale
    for name in loaded_images:
        img_tensor = loaded_images[name]
        if img_tensor.shape[0] == 3:
            # Using standard weights for RGB to grayscale conversion
            weights = torch.tensor([0.2989, 0.5870, 0.1140], device=img_tensor.device).view(3, 1, 1)
            gray = (img_tensor * weights).sum(dim=0, keepdim=True)
            loaded_images[name] = gray
    print(f"Images converted to grayscale and loaded from {images_dir}.")
    return loaded_images
def resize_image_to_match(image1, image2):
        """
        Resizes image1 to have the same dimensions as image2.
        
        Args:
            image1 (torch.Tensor or np.ndarray): The image to be resized.
            image2 (torch.Tensor or np.ndarray): The reference image with desired dimensions.
        
        Returns:
            torch.Tensor or np.ndarray: The resized image1.
        """
        # Determine the target size from image2
        if isinstance(image2, torch.Tensor):
            _, h2, w2 = image2.shape  # Assuming shape is [C, H, W]
        elif isinstance(image2, np.ndarray):
            h2, w2 = image2.shape[:2]  # Assuming shape is [H, W, C] or [H, W]
        else:
            raise TypeError("image2 must be a torch.Tensor or np.ndarray")
        
        # Resize image1 to match the size of image2
        if isinstance(image1, torch.Tensor):
            image1_resized = torch.nn.functional.interpolate(
                image1.unsqueeze(0), size=(h2, w2), mode='bilinear', align_corners=False
            ).squeeze(0)
        elif isinstance(image1, np.ndarray):
            image1_resized = cv2.resize(image1, (w2, h2), interpolation=cv2.INTER_LINEAR)
        else:
            raise TypeError("image1 must be a torch.Tensor or np.ndarray")
        
        return image1_resized
def rbd(feats):
    """Removes batch dimension from feature dict."""
    feats_no_batch = {k: v.squeeze(0) for k, v in feats.items()}
    return feats_no_batch

def extract_features(extractor, images, device):
    """Extracts features from the provided images."""
    features = {}
    with torch.no_grad():
        for name, img in images.items():
            img = img.to(device)
            feats = extractor.extract(img.unsqueeze(0))  # Add batch dimension
            feats = rbd(feats)  # Remove batch dimension
            features[name] = feats
    print("Feature extraction completed.")
    return features

def split_image_into_tiles(image, tile_size=(512, 512), overlap=128):
    """Splits the image into overlapping tiles."""
    tiles = []
    _, h, w = image.shape  # Assuming image shape is [C, H, W]
    stride_x = tile_size[1] - overlap
    stride_y = tile_size[0] - overlap
    for y in range(0, h - tile_size[0] + 1, stride_y):
        for x in range(0, w - tile_size[1] + 1, stride_x):
            tile = image[:, y:y + tile_size[0], x:x + tile_size[1]]
            if tile.shape[1] == tile_size[0] and tile.shape[2] == tile_size[1]:
                tiles.append((tile, x, y))  # Store tile with its top-left position
    print(f"Image split into {len(tiles)} tiles.")
    return tiles

def extract_features_from_tiles(tiles, extractor, device):
    """Extracts features from each tile individually."""
    features = []
    for tile, x, y in tqdm(tiles, desc="Extracting features from tiles"):
        tile = tile.unsqueeze(0).to(device)  # Add batch dimension
        with torch.no_grad():
            feats = extractor.extract(tile)
            feats = rbd(feats)  # Remove batch dimension
            # Adjust keypoints by tile position
            feats["keypoints"] += torch.tensor([x, y]).to(device)
            features.append(feats)
    print("Feature extraction for tiles completed.")
    return features



def visualize_clusters(satellite_image_path, satellite_points, match_scores_array, eps=25, min_samples=3):
    """
    Visualizes clustered matches on the satellite image.
    
    Args:
        satellite_image_path (str): Path to the satellite image file.
        satellite_points (np.ndarray): Array of matched points on the satellite image (shape [N, 2]).
        match_scores_array (np.ndarray): Array of match scores for the points (shape [N]).
        eps (float): Maximum distance between two points for clustering (DBSCAN parameter).
        min_samples (int): Minimum number of points in a cluster (DBSCAN parameter).
    """
    # Perform clustering
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(satellite_points)
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Number of clusters found: {n_clusters}")

    if n_clusters == 0:
        print("No clusters found.")
        return

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
    sat_image_original_color = cv2.cvtColor(cv2.imread(satellite_image_path), cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(15, 10))
    plt.imshow(sat_image_original_color)
    for label, color in zip(clusters.keys(), colors):
        cluster_points = np.array([item['point'] for item in clusters[label]])
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=color, s=10, label=f'Cluster {label}')
        # Annotate cluster center with confidence
        center = cluster_confidence[label]['center']
        confidence = cluster_confidence[label]['average_score']
        plt.text(center[0], center[1], f'{confidence:.2f}', color='white', fontsize=12, ha='center', va='center')

    plt.legend()
    plt.title('Clusters of Matched Points on Satellite Image')
    plt.axis('off')
    plt.show()

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
from tqdm import tqdm  # For progress bars
import time  # For measuring time taken for processing
import matplotlib.cm as cm
from LightGlue.lightglue import SuperPoint  # Adjust this import as per your environment
from sklearn.cluster import DBSCAN  # Import DBSCAN for clustering
from collections import defaultdict
from LightGlue.lightglue import SuperPoint  # Adjust this import as per your environment


def match_features(feats0, feats1, k=2, ratio_thresh=0.75):
    """
    Matches features between two feature sets using KNN matching with ratio test.

    Args:
        feats0 (dict): Features from the first image, containing 'keypoints' and 'descriptors'.
                       - 'keypoints': Tensor of shape [N0, 2]
                       - 'descriptors': Tensor of shape [N0, D]
        feats1 (dict): Features from the second image, containing 'keypoints' and 'descriptors'.
                       - 'keypoints': Tensor of shape [N1, 2]
                       - 'descriptors': Tensor of shape [N1, D]
        k (int, optional): Number of nearest neighbors to find. Defaults to 2.
        ratio_thresh (float, optional): Threshold for Lowe's ratio test. Defaults to 0.75.

    Returns:
        dict: A dictionary containing matches and related information.
              - 'matches0': Tensor of shape [N0], where each element is the index of the matched descriptor in feats1 or -1.
              - 'matching_scores0': Tensor of shape [N0], containing the distance of the match.
    """
    # Convert descriptors to numpy arrays and ensure type float32
    desc0 = feats0['descriptors'].cpu().numpy().astype(np.float32)  # Shape [N0, D]
    desc1 = feats1['descriptors'].cpu().numpy().astype(np.float32)  # Shape [N1, D]

    # Debugging: Check shapes and types
    print(f"Descriptor 0 shape: {desc0.shape}, dtype: {desc0.dtype}")
    print(f"Descriptor 1 shape: {desc1.shape}, dtype: {desc1.dtype}")

    # Check if descriptors are empty
    if desc0.size == 0 or desc1.size == 0:
        print("Descriptors are empty, skipping matching.")
        return {
            'matches0': torch.full((len(feats0['keypoints']),), -1, dtype=torch.int32, device=feats0['keypoints'].device),
            'matching_scores0': torch.zeros(len(feats0['keypoints']), dtype=torch.float32, device=feats0['keypoints'].device)
        }

    # Ensure descriptors have the same dimension D
    if desc0.shape[1] != desc1.shape[1]:
        raise ValueError(f"Descriptor dimension mismatch: {desc0.shape[1]} vs {desc1.shape[1]}")

    # Create Flann object
    FLANN_INDEX_KDTREE = 1  # Changed to KDTree for better performance with KNN
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)  # Increased number of trees for better accuracy
    search_params = dict(checks=50)  # Number of times the tree(s) in the index should be recursively traversed
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Perform KNN matching
    knn_matches = flann.knnMatch(desc0, desc1, k=k)

    # Prepare matches0 array
    N0 = len(feats0['keypoints'])
    matches0 = -np.ones(N0, dtype=int)
    matching_scores0 = np.zeros(N0, dtype=float)

    # Apply Lowe's ratio test
    for i, match_pair in enumerate(knn_matches):
        if len(match_pair) < 2:
            continue  # Not enough matches to apply ratio test
        m, n = match_pair
        if m.distance < ratio_thresh * n.distance:
            matches0[i] = m.trainIdx
            matching_scores0[i] = m.distance  # You can invert or normalize this score if needed

    # Convert to torch tensors
    matches0 = torch.from_numpy(matches0).to(feats0['keypoints'].device)
    matching_scores0 = torch.from_numpy(matching_scores0).to(feats0['keypoints'].device)

    # Prepare the output dictionary
    processed_matches = {
        'matches0': matches0,  # [N0] tensor
        'matching_scores0': matching_scores0,  # [N0] tensor
    }

    return processed_matches


def plot_matches(image0, image1, kpts0, kpts1, scores=None, layout="lr"):
    """ 
    Plot matches between two images. If score is not None, then red: bad match, green: good match 
    :param image0: reference image 
    :param image1: current image 
    :param kpts0: keypoints in reference image 
    :param kpts1: keypoints in current image 
    :param scores: matching score for each keypoint pair, higher is better 
    :param layout: 'lr': left right; 'ud': up down 
    :return: 
    """

    image0= resize_image_to_match(image0, image1)
    # Convert images from tensors to NumPy arrays
    if torch.is_tensor(image0):
        image0 = image0.permute(1, 2, 0).cpu().numpy()
    if torch.is_tensor(image1):
        image1 = image1.permute(1, 2, 0).cpu().numpy()
    # Ensure images are in uint8 format
    image0 = (image0 * 255).astype(np.uint8)
    image1 = (image1 * 255).astype(np.uint8)
    H0, W0 = image0.shape[0], image0.shape[1] 
    H1, W1 = image1.shape[0], image1.shape[1] 

    if layout == "lr": 
        H, W = max(H0, H1), W0 + W1 
        out = 255 * np.ones((H, W, 3), np.uint8) 
        out[:H0, :W0, :] = image0 
        out[:H1, W0:, :] = image1 
    elif layout == "ud": 
        H, W = H0 + H1, max(W0, W1) 
        out = 255 * np.ones((H, W, 3), np.uint8) 
        out[:H0, :W0, :] = image0 
        out[H0:, :W1, :] = image1 
    else: 
        raise ValueError("The layout must be 'lr' or 'ud'!") 

    # Convert keypoints to NumPy arrays if they are tensors
    if isinstance(kpts0, torch.Tensor):
        kpts0 = kpts0.cpu().numpy()
    if isinstance(kpts1, torch.Tensor):
        kpts1 = kpts1.cpu().numpy()
    kpts0 = np.round(kpts0).astype(int)
    kpts1 = np.round(kpts1).astype(int)

    # Get color
    if scores is not None:
        # Convert scores to numpy array if it's a tensor
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        # Normalize and invert scores to be between 0 and 1
        # Lower distances are better matches, so we invert the distances
        smin, smax = scores.min(), scores.max()
        if smin == smax:
            # All scores are the same, set to 1 (best match)
            scores = np.ones_like(scores)
        else:
            # Invert and normalize scores
            scores = (smax - scores) / (smax - smin)
        smin, smax = scores.min(), scores.max()
        assert (0 <= smin <= 1 and 0 <= smax <= 1)
        color = cm.gist_rainbow(scores * 0.4)
        color = (np.array(color[:, :3]) * 255).astype(int)[:, ::-1]
    else:
        color = np.zeros((kpts0.shape[0], 3), dtype=int)
        color[:, 1] = 255

    for (x0, y0), (x1, y1), c in zip(kpts0, kpts1, color):
        c = c.tolist()
        if layout == "lr":
            cv2.line(out, (x0, y0), (x1 + W0, y1), color=c, thickness=1, lineType=cv2.LINE_AA)
            # Display line end-points as circles
            cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x1 + W0, y1), 2, c, -1, lineType=cv2.LINE_AA)
        elif layout == "ud":
            cv2.line(out, (x0, y0), (x1, y1 + H0), color=c, thickness=1, lineType=cv2.LINE_AA)
            # Display line end-points as circles
            cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x1, y1 + H0), 2, c, -1, lineType=cv2.LINE_AA)

    return out


def main():
    start_time = time.time()
    """Main function to execute the image processing pipeline."""
    # Configuration 
    drone_images_dir = "path_to_drone_images"  # Directory containing drone images
    satellite_images_dir = "path_to_satellite_images"  # Directory containing satellite images
    drone_image_names = ["00359.png"]  # List of drone images
    satellite_image_names = ["SIFTOGB.jpg"]  # List of satellite images

    tile_size = (1500, 1500)  # Adjusted tile size for better performance
    overlap = 250  # Reduced overlap to decrease number of tiles
    num_tile_matches_to_visualize = 1

    # Setup
    device = setup_device()
    extractor = initialize_models(device)

    # Load Images
    drone_images = load_images(drone_images_dir, drone_image_names)
    satellite_images = load_images(satellite_images_dir, satellite_image_names)

    # Extract Features from Drone Images
    drone_features = extract_features(extractor, drone_images, device)

    # Split Satellite Images into Tiles and Extract Features
    satellite_tiles = {}
    satellite_features = {}
    for sat_image_name, sat_image in satellite_images.items():
        tiles = split_image_into_tiles(sat_image, tile_size, overlap)
        satellite_tiles[sat_image_name] = tiles
        print(f"Extracting features for satellite image '{sat_image_name}'")
        features_tiles = extract_features_from_tiles(tiles, extractor, device)
        satellite_features[sat_image_name] = features_tiles
        print(f"Features extracted for {len(features_tiles)} tiles. First tile descriptor shape: {features_tiles[0]['descriptors'].shape}")
    
    # For each Drone Image
    for drone_image_name, drone_feats in drone_features.items():
        drone_image = drone_images[drone_image_name]
        # For each Satellite Image
        for sat_image_name in satellite_image_names:
            tiles = satellite_tiles[sat_image_name]
            features_tiles = satellite_features[sat_image_name]
            sat_image = satellite_images[sat_image_name]

            print(f"Matching drone image '{drone_image_name}' with satellite image '{sat_image_name}'")

            # Initialize list to store matches
            all_matches = []

            # Use tqdm for progress bar
            for i, tile_feats in enumerate(tqdm(features_tiles, desc=f"Matching tiles for '{sat_image_name}'")):
                # Match features
                matches = match_features(drone_feats, tile_feats)
                matched_indices = matches["matches0"]
                if matched_indices is not None:
                    if isinstance(matched_indices, torch.Tensor):
                        num_valid = (matched_indices > -1).sum().item()
                    elif isinstance(matched_indices, np.ndarray):
                        num_valid = np.sum(matched_indices > -1)
                    else:
                        num_valid = len([m for m in matched_indices if m > -1])

                    if num_valid > 0:
                        # Apply RANSAC to filter out outlier matches
                        inlier_matches, homography = filter_matches_with_ransac(
                            drone_feats['keypoints'].cpu().numpy(),
                            tile_feats['keypoints'].cpu().numpy(),
                            matches,
                            ransac_thresh=5.0  # You can adjust the threshold
                        )
                        if len(inlier_matches) > 0:
                            all_matches.append((i, inlier_matches))
                            # Debugging print
                            print(f"Tile {i} has {len(inlier_matches)} inlier matches after RANSAC.")
            
            # Sort all_matches based on the number of inlier matches
            all_matches.sort(key=lambda x: len(x[1]), reverse=True)

            print(f"Drone image '{drone_image_name}' matched with {len(all_matches)} satellite tiles after RANSAC filtering.")

            # Verify the sorting
            for idx, (tile_idx, inlier_matches) in enumerate(all_matches):
                print(f"Tile {tile_idx} has {len(inlier_matches)} inlier matches.")

            # Visualize Matches for Selected Tile Pairs
            num_tiles_to_visualize = min(num_tile_matches_to_visualize, len(all_matches))
            for idx in range(num_tiles_to_visualize):
                tile_idx, inlier_matches = all_matches[idx]
                tile, x, y = tiles[tile_idx]
                tile_image = tile  # Get the tile image tensor
            
                # Convert inlier_matches to NumPy array
                inlier_matches = np.array(inlier_matches)
            
                # Extract inlier keypoints
                drone_kpts = drone_feats['keypoints'][inlier_matches[:, 0]]
                satellite_kpts = features_tiles[tile_idx]['keypoints'][inlier_matches[:, 1]]
            
                title = f"Drone: {drone_image_name} | Satellite Tile: {tile_idx+1}"

                # Plot matches
                out_image = plot_matches(
                    drone_image,
                    tile_image,
                    drone_kpts,
                    satellite_kpts,
                    scores=None,  # Optional: You can pass matching scores if needed
                    layout="lr"
                )

                # Save or display the image
                output_path = f'matches_{drone_image_name}_{sat_image_name}_tile_{tile_idx}.png'
                cv2.imwrite(output_path, out_image)
            print(f"Saved match visualization to {output_path}")

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Processing completed in {elapsed_time:.2f} seconds.")
            satellite_image_path = Path(satellite_images_dir) / sat_image_name

            # Visualize clusters on the satellite image
            satellite_points = satellite_kpts.cpu().numpy().astype(np.float32)
            match_scores_array = np.ones(len(satellite_points))
            visualize_clusters(satellite_image_path, satellite_points, match_scores_array)
            view_all_clusters(satellite_points, drone_kpts, match_scores_array, satellite_image_path, plot=True)

                

if __name__ == "__main__":
    main()