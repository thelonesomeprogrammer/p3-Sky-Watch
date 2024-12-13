import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from LightGlue.lightglue.lightglue import LightGlue
from LightGlue.lightglue.utils import load_image, rbd
from LightGlue.lightglue import viz2d
from tqdm import tqdm  # For progress bars
import cv2

# Configuration
tile_size = (1500, 1500)  # Adjusted tile size for better performance                                       
overlap = 250  # Reduced overlap to decrease number of tiles

def setup_device():
    """Sets up the computation device."""
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
    else:
        print("CUDA is not available. Using CPU.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def initialize_lightglue(device):
    """Initializes the LightGlue matcher for custom features."""
    matcher = LightGlue(features='custom').eval().to(device)
    print("LightGlue matcher initialized with custom features.")
    return matcher

def initialize_sift_extractor():
    """Initializes the SIFT feature extractor."""
    sift = cv2.SIFT_create(nfeatures=1000)
    print("SIFT extractor initialized.")
    return sift

def extract_features_sift(extractor, images):
    """Extracts features from the provided images using SIFT."""
    features = {}
    for name, img in images.items():
        # Convert image to grayscale if needed
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img
        # Debugging print
        print(f"Processing image '{name}', img_gray shape: {img_gray.shape}, dtype: {img_gray.dtype}")
        # Extract keypoints and descriptors
        keypoints, descriptors = extractor.detectAndCompute(img_gray, None)
        if keypoints is None or descriptors is None:
            # No keypoints detected
            feats = {
                'keypoints': torch.empty((0, 2)),
                'descriptors': torch.empty((0, 128)),
            }
        else:
            # Convert keypoints to numpy array
            kpts = np.array([kp.pt for kp in keypoints], dtype=np.float32)
            # Convert descriptors to numpy array
            descs = descriptors.astype(np.float32)  # [N, 128]
            # Package into features dict
            feats = {
                'keypoints': torch.from_numpy(kpts),     # [N, 2]
                'descriptors': torch.from_numpy(descs),  # [N, 128]
            }
        features[name] = feats
    print("Feature extraction with SIFT completed.")
    return features

def divide_into_tiles(image, tile_size, overlap):
    """Divides an image into overlapping tiles."""
    tiles = []
    h, w = image.shape[:2]
    tile_h, tile_w = tile_size
    step_h = tile_h - overlap
    step_w = tile_w - overlap
    for y in range(0, h - tile_h + 1, step_h):
        for x in range(0, w - tile_w + 1, step_w):
            tile = image[y:y + tile_h, x:x + tile_w]
            tiles.append((tile, (x, y)))
    return tiles

def main():
    """Main function to execute the image processing pipeline."""
    # Configuration 
    drone_images_dir = "path_to_drone_images"  # Replace with your drone images directory
    satellite_images_dir = "path_to_satellite_images"  # Replace with your satellite images directory
    drone_image_names = ["00359.png"]  # List of drone images
    satellite_image_names = ["SIFTOGB.jpg"]  # List of satellite images

    tile_size = (1500, 1500)  # Adjusted tile size for better performance
    overlap = 250  # Reduced overlap to decrease number of tiles
    num_tile_matches_to_visualize = 1

    # Load images
    drone_image_path = os.path.join(drone_images_dir, drone_image_names[0])
    satellite_image_path = os.path.join(satellite_images_dir, satellite_image_names[0])

    drone_image = cv2.imread(drone_image_path)
    satellite_image = cv2.imread(satellite_image_path)

    # Initialize device and models
    device = setup_device()
    sift_extractor = initialize_sift_extractor()
    matcher = initialize_lightglue(device)

    # Prepare images dictionary
    drone_images = {'drone': drone_image}

    # Extract features from drone images
    print("Extracting features from drone image...")
    drone_feats = extract_features_sift(sift_extractor, drone_images)
    feats0 = drone_feats['drone']
    # Move descriptors to device
    feats0['descriptors'] = feats0['descriptors'].to(device)
    feats0['keypoints'] = feats0['keypoints'].to(device)

    # Divide satellite image into tiles
    print("Dividing satellite image into tiles...")
    sat_tiles = divide_into_tiles(satellite_image, tile_size, overlap)

    # Match features between drone image and each satellite tile
    for tile_idx, (tile, (x, y)) in enumerate(sat_tiles):
        print(f"Processing tile {tile_idx + 1}/{len(sat_tiles)} at position ({x}, {y})...")
        # Extract features from satellite tile
        tile_images = {'tile': tile}
        tile_feats = extract_features_sift(sift_extractor, tile_images)
        feats1 = tile_feats['tile']
        # Move descriptors to device
        feats1['descriptors'] = feats1['descriptors'].to(device)
        feats1['keypoints'] = feats1['keypoints'].to(device)

        # Prepare data for LightGlue
        data = {
            'keypoints0': feats0['keypoints'].unsqueeze(0),
            'keypoints1': feats1['keypoints'].unsqueeze(0),
            'descriptors0': feats0['descriptors'].unsqueeze(0),
            'descriptors1': feats1['descriptors'].unsqueeze(0),
        }

        # Check if there are enough keypoints to match
        if data['keypoints0'].shape[1] == 0 or data['keypoints1'].shape[1] == 0:
            print("Not enough keypoints to match.")
            continue

        with torch.no_grad():
            matches01 = matcher(data)

        # Remove batch dimension
        feats0_cpu = {'keypoints': feats0['keypoints'].cpu()}
        feats1_cpu = {'keypoints': feats1['keypoints'].cpu()}
        matches01_cpu = {k: v[0].cpu() for k, v in matches01.items()}

        kpts0, kpts1, matches = feats0_cpu["keypoints"], feats1_cpu["keypoints"], matches01_cpu["matches"]

        # Filter out unmatched points
        valid_matches = matches > -1
        m_kpts0 = kpts0[valid_matches]
        m_kpts1 = kpts1[matches[valid_matches]]

        if len(m_kpts0) == 0:
            print("No matches found.")
            continue

        # Visualize the matches
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(cv2.cvtColor(drone_image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Drone Image')
        axes[0].axis('off')
        axes[1].imshow(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f'Satellite Tile ({x}, {y})')
        axes[1].axis('off')

        # Plot matches
        for i in range(len(m_kpts0)):
            pt0 = m_kpts0[i].numpy()
            pt1 = m_kpts1[i].numpy()
            con_color = np.random.rand(3,)
            axes[0].plot(pt0[0], pt0[1], 'o', color=con_color)
            axes[1].plot(pt1[0], pt1[1], 'o', color=con_color)
            # Draw line between points
            fig.canvas.draw()
            transFigure = fig.transFigure.inverted()
            coord1 = axes[0].transData.transform(pt0)
            coord2 = axes[1].transData.transform(pt1)
            coord1 = transFigure.transform(coord1)
            coord2 = transFigure.transform(coord2)
            line = plt.Line2D((coord1[0], coord2[0]), (coord1[1], coord2[1]), transform=fig.transFigure, color=con_color)
            fig.lines.append(line)

        plt.show()

        # Break after visualizing the desired number of tiles
        num_tile_matches_to_visualize -= 1
        if num_tile_matches_to_visualize <= 0:
            break

if __name__ == "__main__":
    main()
