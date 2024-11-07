import os

# Set this at the very beginning of your script, before importing any libraries that use OpenMP
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from LightGlue.lightglue.lightglue import LightGlue
from LightGlue.lightglue.utils import load_image, rbd
from LightGlue.lightglue import viz2d, SuperPoint
from tqdm import tqdm  # For progress bars

def setup_device():
    """Sets up the computation device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def initialize_models(device, max_keypoints=2048):
    """Initializes the feature extractor and matcher."""
    extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)
    print("Models initialized.")
    return extractor, matcher

def load_images(images_dir, image_names):
    """Loads images from the specified directory."""
    images_path = Path(images_dir)
    loaded_images = {}
    for name in image_names:
        img_path = images_path / name
        if not img_path.exists():
            raise FileNotFoundError(f"Image '{name}' not found in {images_dir}.")
        loaded_images[name] = load_image(str(img_path))
    print(f"Images loaded from {images_dir}.")
    return loaded_images

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


import torch

import torch

import torch

def match_features(matcher, feats0, feats1, device):
    """
    Matches features between two feature sets using LightGlue.

    Args:
        matcher (LightGlue): The LightGlue matcher instance.
        feats0 (dict): Features from the first image, containing 'keypoints' and 'descriptors'.
                       - 'keypoints': Tensor of shape [N0, 2]
                       - 'descriptors': Tensor of shape [D, N0]
        feats1 (dict): Features from the second image, containing 'keypoints' and 'descriptors'.
                       - 'keypoints': Tensor of shape [N1, 2]
                       - 'descriptors': Tensor of shape [D, N1]
        device (torch.device): The device to perform computations on (e.g., torch.device('cuda')).

    Returns:
        dict: A dictionary containing matches and related information with batch dimension removed.
              Keys may include:
              - 'matches': Tensor of shape [K, 2] where K is the number of matches
              - Other relevant match information as provided by LightGlue
    """
    # ---------------------------
    # 1. Input Validation
    # ---------------------------
    required_keys = ['keypoints', 'descriptors']
    
    for i, (feats, name) in enumerate(zip([feats0, feats1], ['feats0', 'feats1'])):
        for key in required_keys:
            assert key in feats, f"'{key}' not found in {name}. Required keys: {required_keys}"
            assert isinstance(feats[key], torch.Tensor), f"'{key}' in {name} must be a torch.Tensor"
        
        # Check keypoints shape: [N, 2]
        assert feats['keypoints'].ndim == 2 and feats['keypoints'].shape[1] == 2, \
            f"'keypoints' in {name} must have shape [N, 2], got {feats['keypoints'].shape}"
        
        # Check descriptors shape: [D, N]
        assert feats['descriptors'].ndim == 2, \
            f"'descriptors' in {name} must have 2 dimensions [D, N], got {feats['descriptors'].shape}"
    
    # ---------------------------
    # 2. Data Preparation
    # ---------------------------
    # Move features to the specified device and ensure correct data types
    # Also, add batch dimension by unsqueezing at dim=0

    def prepare_features(feats, name):
        prepared = {}
        for key, tensor in feats.items():
            # Ensure tensor is of type float32
            tensor = tensor.float()
            
            # Move to the specified device
            tensor = tensor.to(device)
            
            # Add batch dimension
            tensor = tensor.unsqueeze(0)  # Shape becomes [1, ..., ...]
            
            prepared[key] = tensor
            print(f"Prepared {name}['{key}']: {tensor.shape}, dtype: {tensor.dtype}, device: {tensor.device}")
        return prepared

    prepared_feats0 = prepare_features(feats0, 'feats0')
    prepared_feats1 = prepare_features(feats1, 'feats1')

    # Prepare the data dictionary as expected by LightGlue
    data = {
        "image0": prepared_feats0,  # Contains 'keypoints' and 'descriptors'
        "image1": prepared_feats1,  # Contains 'keypoints' and 'descriptors'
    }

    # ---------------------------
    # 3. Feature Matching
    # ---------------------------
    try:
        with torch.no_grad():
            matches = matcher(data)
    except AssertionError as e:
        # Provide more context if a known assertion fails
        if "Missing key" in str(e):
            raise AssertionError(f"Input data to matcher is missing required keys: {e}")
        else:
            raise e
    except Exception as e:
        # Catch-all for other exceptions
        raise RuntimeError(f"An error occurred during feature matching: {e}")

    # ---------------------------
    # 4. Post-processing
    # ---------------------------
    # Remove batch dimension from all tensor outputs
    processed_matches = {}
    for k, v in matches.items():
        if isinstance(v, torch.Tensor):
            processed_v = v.squeeze(0)
            processed_matches[k] = processed_v
            print(f"Processed matches['{k}']: {processed_v.shape}")
        else:
            processed_matches[k] = v  # Non-tensor outputs are copied as is
    
    return processed_matches






def visualize_matches(image0, image1, feats0, feats1, matches, title=''):
    """Visualizes the matched keypoints between two images."""
    kpts0, kpts1 = feats0["keypoints"], feats1["keypoints"]
    matched_indices = matches["matches0"]  # Update key if necessary

    if matched_indices is None or len(matched_indices) == 0:
        print("No matches found to visualize.")
        return

    # Extract matched keypoints
    valid = matched_indices > -1
    m_kpts0 = kpts0[valid]
    m_kpts1 = kpts1[matched_indices[valid]]

    print(f"Number of matches: {len(m_kpts0)}")

    # Plot images without unpacking
    viz2d.plot_images([image0, image1])

    # Plot matches
    viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.5)

    # Add text
    viz2d.add_text(0, title, fs=15)

    # Show plot
    plt.show()


def main():
    """Main function to execute the image processing pipeline."""
    # Configuration
    drone_images_dir = "drone"  # Directory containing drone images
    satellite_images_dir = "sat"  # Directory containing satellite images
    drone_image_names = ["00078.png"]  # List of drone images
    satellite_image_names = ["h.jpg"]  # List of satellite images

    tile_size = (1500, 1500)  # Adjusted tile size for better performance
    overlap = 250  # Reduced overlap to decrease number of tiles
    num_tile_matches_to_visualize = 1
    num_matches_per_tile = 10

    # Setup
    device = setup_device()
    extractor, matcher = initialize_models(device)

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
        print(f"Features extracted for {len(features_tiles)} tiles. and shape of features is {features_tiles[0]['keypoints'].shape}")
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
        matches = match_features(matcher, drone_feats, tile_feats, device)
        matched_indices = matches["matches0"]
        print(type(matches["matches0"]))
        if matched_indices is not None:
            if isinstance(matched_indices, torch.Tensor):
                num_valid = (matched_indices > -1).sum().item()
            elif isinstance(matched_indices, np.ndarray):
                num_valid = np.sum(matched_indices > -1)
            else:
                num_valid = len([m for m in matched_indices if m > -1])
            
            if num_valid > 0:
                all_matches.append((i, matches))
                # Debugging print
                print(f"Tile {i} has {num_valid} matches.")

    # Define the function to get the number of valid matches
    def get_num_valid_matches(match_tuple):
        matches = match_tuple[1]
        matched_indices = matches["matches0"]
        if matched_indices is not None:
            valid = matched_indices > -1
            if isinstance(valid, torch.Tensor):
                return valid.sum().item()
            elif isinstance(valid, np.ndarray):
                return np.sum(valid)
            else:
                return len([m for m in valid if m])
        else:
            return 0

    # Sort all_matches based on the number of valid matches
    all_matches.sort(key=get_num_valid_matches, reverse=True)

    print(f"Drone image '{drone_image_name}' matched with {len(all_matches)} satellite tiles.")

    # Verify the sorting
    for idx, (tile_idx, matches) in enumerate(all_matches):
        num_matches = get_num_valid_matches((tile_idx, matches))
        print(f"Tile {tile_idx} has {num_matches} matches.")

    # Visualize Matches for Selected Tile Pairs
    num_tiles_to_visualize = min(num_tile_matches_to_visualize, len(all_matches))
    for idx in range(num_tiles_to_visualize):
        tile_idx, matches = all_matches[idx]
        tile, x, y = tiles[tile_idx]
        tile_feats = features_tiles[tile_idx]
        title = f"Drone: {drone_image_name} | Satellite Tile: {tile_idx+1}"
        visualize_matches(drone_image, tile, drone_feats, tile_feats, matches, title=title)

if __name__ == "__main__":
    main()
