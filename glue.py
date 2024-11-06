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

# After extracting features
def extract_features(extractor, images, device):
    """Extracts features from the provided images."""
    features = {}
    with torch.no_grad():
        for name, img in images.items():
            img = img.to(device)
            feats = extractor.extract(img.unsqueeze(0))  # Add batch dimension
            feats = rbd(feats)  # Remove batch dimension
            # Debug statements
            print(f"Extracted features for {name}:")
            for key in feats:
                if isinstance(feats[key], torch.Tensor):
                    print(f"{key}: {feats[key].shape}")
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

def match_features(matcher, feats0, feats1, image0, image1, device):
    """Matches features between two feature sets."""
    # Preprocess images to ensure they are grayscale and have the correct dimensions
    def preprocess_image(img):
        if img.dim() == 3 and img.shape[0] == 3:
            # Convert RGB to grayscale
            img = 0.2989 * img[0, :, :] + 0.5870 * img[1, :, :] + 0.1140 * img[2, :, :]
        elif img.dim() == 3 and img.shape[0] == 1:
            img = img[0, :, :]
        else:
            raise ValueError("Image tensor has unexpected shape: {}".format(img.shape))
        img = img.unsqueeze(0).unsqueeze(0).to(device)  # Shape: [1, 1, H, W]
        return img

    image0 = preprocess_image(image0)
    image1 = preprocess_image(image1)

    data = {
        "image0": image0,  # [1, 1, H0, W0]
        "image1": image1,  # [1, 1, H1, W1]
        "keypoints0": feats0["keypoints"].unsqueeze(0).to(device),  # [1, N0, 2]
        "keypoints1": feats1["keypoints"].unsqueeze(0).to(device),  # [1, N1, 2]
        "descriptors0": feats0["descriptors"].unsqueeze(0).to(device),  # [1, N0, D]
        "descriptors1": feats1["descriptors"].unsqueeze(0).to(device),  # [1, N1, D]
    }

    for key in data:
        if isinstance(data[key], torch.Tensor):
            print(f"{key}: {data[key].shape}")
            
    # Match features
    with torch.no_grad():
        matches = matcher(data)

    # Remove batch dimension from matches
    matches = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v for k, v in matches.items()}
    return matches



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
    drone_images_dir = "drone_views"  # Directory containing drone images
    satellite_images_dir = "satellite_views"  # Directory containing satellite images
    drone_image_names = ["00078.png", "00077.png"]  # List of drone images
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
            for i, (tile_feats, (tile, x, y)) in enumerate(tqdm(zip(features_tiles, tiles), desc=f"Matching tiles for '{sat_image_name}'", total=len(tiles))):
                # Match features
                matches = match_features(matcher, drone_feats, tile_feats, drone_image, tile, device)
                if matches["matches0"] is not None and len(matches["matches0"]) > 0:
                    all_matches.append((i, matches))

            print(f"Drone image '{drone_image_name}' matched with {len(all_matches)} satellite tiles.")


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
