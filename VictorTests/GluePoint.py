import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

# Import SuperPoint and SuperGlue models
from lightglue.superpoint import SuperPoint
from lightglue.utils import load_image, rbd
from lightglue.lightglue import LightGlue

def read_image(path, device, scale_factor=0.4, rotation=False):
    """
    Read image as grayscale, rescale, and normalize.

    Parameters:
        path (str): Path to the image.
        device (str): 'cuda' or 'cpu'.
        scale_factor (float): Scaling factor to resize the image.

    Returns:
        img_tensor (torch.Tensor): Rescaled and normalized image tensor, shape [1, 1, H, W].
        original_size (tuple): Original image size as (height, width).
        scale_factor (float): The scale factor used for resizing.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f'Image {path} not found')

    original_size = img.shape  # (H, W)

    # Rescale the image
    new_size = (int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor))
    img_rescaled = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

    if rotation == True:
        # Rotate the image by 180 degrees
        img_rescaled = cv2.rotate(img_rescaled, cv2.ROTATE_180)

        # cut 30% of the image border
        # img_rescaled = img_rescaled[int(img_rescaled.shape[0]*0.3):int(img_rescaled.shape[0]*0.7), int(img_rescaled.shape[1]*0.3):int(img_rescaled.shape[1]*0.7)]

    # Normalize the image
    img_normalized = img_rescaled.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_normalized)[None, None].to(device)  # Shape: [1, 1, H, W]

    return img_tensor, original_size, scale_factor

def draw_matches(img1, img2, kp1, kp2, mask=None):
    """
    Draw matches between two images.

    Parameters:
        img1 (np.ndarray): First image in grayscale.
        img2 (np.ndarray): Second image in grayscale.
        kp1 (np.ndarray): Keypoints from the first image, shape [N, 2].
        kp2 (np.ndarray): Keypoints from the second image, shape [N, 2].
        mask (np.ndarray, optional): Boolean mask to filter matches.

    Returns:
        out_img (np.ndarray): Image with matches drawn.
    """
    # Convert images to color
    img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    # Ensure both images have the same height
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    if h1 != h2:
        # Calculate resize factor for img2 to match img1's height
        resize_factor2 = h1 / h2
        new_w2 = int(w2 * resize_factor2)
        img2_color = cv2.resize(img2_color, (new_w2, h1), interpolation=cv2.INTER_AREA)
        img2 = cv2.resize(img2, (new_w2, h1), interpolation=cv2.INTER_AREA)
        h2, w2 = img2.shape
    else:
        resize_factor2 = 1.0  # No resizing needed

    # Concatenate images horizontally
    height = max(h1, h2)
    out_img = np.zeros((height, w1 + w2, 3), dtype=np.uint8)
    out_img[:h1, :w1] = img1_color
    out_img[:h2, w1:w1 + w2] = img2_color

    # If mask is provided, use it to filter matches
    if mask is not None:
        kp1 = kp1[mask]
        kp2 = kp2[mask]

    # Draw lines between matching keypoints
    for pt1, pt2 in zip(kp1, kp2):
        pt1 = (int(pt1[0]), int(pt1[1]))
        pt2 = (int(pt2[0] + w1), int(pt2[1]))  # Adjust pt2's x-coordinate
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.line(out_img, pt1, pt2, color, 1)
        cv2.circle(out_img, pt1, 4, color, -1)
        cv2.circle(out_img, pt2, 4, color, -1)

    return out_img

def remove_bad_matches(mkpts0, mkpts1, reproj_thresh=4.0):
    """
    Remove bad matches using RANSAC to estimate homography and filter out outliers.

    Parameters:
        mkpts0 (np.ndarray): Matched keypoints from image1, shape [M, 2].
        mkpts1 (np.ndarray): Matched keypoints from image2, shape [M, 2].
        reproj_thresh (float): RANSAC reprojection threshold. Default is 4.0.

    Returns:
        mkpts0_filtered (np.ndarray): Filtered keypoints from image1, shape [N, 2].
        mkpts1_filtered (np.ndarray): Filtered keypoints from image2, shape [N, 2].
        mask (np.ndarray): Boolean mask indicating inliers (True) and outliers (False).
    """
    # Ensure there are enough matches to compute homography
    if len(mkpts0) < 4 or len(mkpts1) < 4:
        print("Not enough matches to compute homography. Returning original matches.")
        return mkpts0, mkpts1, np.ones(len(mkpts0), dtype=bool)

    # Compute homography using RANSAC
    H, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, reproj_thresh)

    # Check if homography was found
    if mask is None:
        print("Homography could not be computed. Returning original matches.")
        return mkpts0, mkpts1, np.ones(len(mkpts0), dtype=bool)

    # Flatten the mask to a 1D array of booleans
    mask = mask.ravel().astype(bool)

    # Apply the mask to the keypoints
    mkpts0_filtered = mkpts0[mask]
    mkpts1_filtered = mkpts1[mask]

    print(f"Total matches: {len(mkpts0)}")
    print(f"Good matches after RANSAC: {len(mkpts0_filtered)}")

    return mkpts0_filtered, mkpts1_filtered, mask

def initialize_models(device, max_keypoints=2048):
    """Initializes the feature extractor and matcher."""
    extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)
    print("Models initialized.")
    return extractor, matcher

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

def main():

    # Paths to the two images
    img_path1 = 'drone_views/00072.png'  # Replace with your first image path
    img_path2 = 'SatData/bonn_72_sat_slice_2.jpg'   # Replace with your second image path

    # Check if image paths exist
    if not os.path.exists(img_path1):
        raise FileNotFoundError(f'Image {img_path1} does not exist.')
    if not os.path.exists(img_path2):
        raise FileNotFoundError(f'Image {img_path2} does not exist.')

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # Read and preprocess images with scaling
    img1_tensor, original_size1, scale_factor1 = read_image(img_path1, device, scale_factor=0.4)
    img2_tensor, original_size2, scale_factor2 = read_image(img_path2, device, scale_factor=0.4, rotation=False)

    # Initialize SuperPoint and SuperGlue models
    extractor, LightGlue = initialize_models(device)

    # Extract keypoints and descriptors from both images
    pred1 = extractor({'image': img1_tensor})
    pred2 = extractor({'image': img2_tensor})

    print(pred1.keys())

    # Prepare data for lightGlue by extracting tensors from lists
    data = {
        'keypoints0': pred1['keypoints'][0].unsqueeze(0),      # Shape: [1, N0, 2]
        'keypoints1': pred2['keypoints'][0].unsqueeze(0),      # Shape: [1, N1, 2]
        'scores0': pred1['keypoint_scores'][0].unsqueeze(0),            # Shape: [1, N0]
        'scores1': pred2['keypoint_scores'][0].unsqueeze(0),            # Shape: [1, N1]
        'descriptors0': pred1['descriptors'][0].unsqueeze(0),  # Shape: [1, N0, D]
        'descriptors1': pred2['descriptors'][0].unsqueeze(0),  # Shape: [1, N1, D]
        'image0': img1_tensor,                                 # Shape: [1, 1, H, W]
        'image1': img2_tensor                                  # Shape: [1, 1, H, W]
    }

    # Ensure all tensors are on the correct device
    for key in ['keypoints0', 'keypoints1', 'scores0', 'scores1', 'descriptors0', 'descriptors1']:
        data[key] = data[key].to(device)

    # Match with SuperGlue
    with torch.no_grad():
        matches = LightGlue(data)

    # Extract matches and valid mask
    matches_idx = matches['matches0'][0].cpu().numpy()  # Shape: [N0]
    valid = matches_idx > -1
    mkpts0 = data['keypoints0'][0][valid].cpu().numpy()             # Shape: [M, 2]
    mkpts1 = data['keypoints1'][0][matches_idx[valid]].cpu().numpy() # Shape: [M, 2]

    print(f"Number of initial matches: {len(mkpts0)}")

    # Remove bad matches using RANSAC
    mkpts0_filtered, mkpts1_filtered, inlier_mask = remove_bad_matches(mkpts0, mkpts1, reproj_thresh=4.0)

    print(f"Number of good matches after filtering: {len(mkpts0_filtered)}")

    # Scale keypoints back to original image sizes
    mkpts0_scaled = mkpts0_filtered / scale_factor1
    mkpts1_scaled = mkpts1_filtered / scale_factor2

    # Load original images for visualization
    img1_orig = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
    img2_orig = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)

    # Ensure both images have the same height
    h1_orig, w1_orig = img1_orig.shape
    h2_orig, w2_orig = img2_orig.shape
    if h1_orig != h2_orig:
        # Calculate resize factor for img2 to match img1's height
        resize_factor2 = h1_orig / h2_orig
        new_w2_orig = int(w2_orig * resize_factor2)
        img2_orig = cv2.resize(img2_orig, (new_w2_orig, h1_orig), interpolation=cv2.INTER_AREA)
        h2_orig, w2_orig = img2_orig.shape
        # Scale keypoints in image2 to account for resizing
        mkpts1_scaled = mkpts1_scaled * resize_factor2
    else:
        resize_factor2 = 1.0  # No resizing needed

    # Visualize all initial matches (scaled back to original sizes)
    img_matches_initial = draw_matches(
        img1_orig, img2_orig,
        mkpts0 / scale_factor1,
        mkpts1 / scale_factor2
    )
    plt.figure(figsize=(20, 15))
    plt.imshow(img_matches_initial, cmap='gray')
    plt.title('Initial SuperPoint + SuperGlue Matches')
    plt.axis('off')
    plt.show()

    # Visualize good matches after filtering (scaled back and adjusted for resizing)

    #img2 with 30% border cut
    img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
    # img2 = cv2.rotate(img2, cv2.ROTATE_180)
    # img2 = img2[int(img2.shape[0]*0.3):int(img2.shape[0]*0.7), int(img2.shape[1]*0.3):int(img2.shape[1]*0.7)]

    img_matches_filtered = draw_matches(
        img1_orig, img2,
        mkpts0_scaled,
        mkpts1_scaled
    )
    plt.figure(figsize=(20, 15))
    plt.imshow(img_matches_filtered, cmap='gray')
    plt.title('Filtered SuperPoint + SuperGlue Matches (Inliers Only)')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()
