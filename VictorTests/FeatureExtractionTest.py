import numpy as np
import cv2 as cv
import enum as Enum
import matplotlib.pyplot as plt
import torch

from lightglue.utils import numpy_image_to_torch
from lightglue.lightglue import LightGlue
from lightglue import SuperPoint

class FeatureType(Enum.Enum):
    SIFT = 0
    ORB = 1
    BRISK = 2
    SUPERPOINT = 3

def getFeatures(featureType, extractor=None):
    """
    Initialize the feature extractor based on the FeatureType.
    """
    if featureType == FeatureType.SIFT:
        return cv.SIFT_create(nfeatures=1000)
    elif featureType == FeatureType.ORB:
        return cv.ORB_create()
    elif featureType == FeatureType.BRISK:
        return cv.BRISK_create(thresh=80)
    elif featureType == FeatureType.SUPERPOINT:
        assert extractor is not None, "SuperPoint extractor not initialized"
        return extractor

def getBFMatcher(featureType):
    """
    Initialize the BFMatcher based on the FeatureType.
    """
    if featureType == FeatureType.SIFT:
        return cv.BFMatcher(cv.NORM_L2, crossCheck=False)
    elif featureType in (FeatureType.ORB, FeatureType.BRISK):
        return cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
    elif featureType == FeatureType.SUPERPOINT:
        return None  # LightGlue will handle matching for SuperPoint

def getMatches(des1, des2, featureType, superpoint_matcher=None, kp1=None, kp2=None):
    """
    Match descriptors between two images using either BFMatcher or LightGlue.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if featureType == FeatureType.SUPERPOINT:
        assert superpoint_matcher is not None, "LightGlue matcher must be provided"

        # Ensure kp1 and kp2 are NumPy arrays of shape [num_kpts, 2]
        # and des1 and des2 are NumPy arrays of shape [num_kpts, desc_dim]
        if kp1.size == 0 or kp2.size == 0:
            return []

        kp1_tensor = torch.from_numpy(kp1).unsqueeze(0).to(device)  # Shape: [1, num_kpts, 2]
        kp2_tensor = torch.from_numpy(kp2).unsqueeze(0).to(device)  # Shape: [1, num_kpts, 2]
        des1_tensor = torch.from_numpy(des1).unsqueeze(0).to(device)  # Shape: [1, num_kpts, desc_dim]
        des2_tensor = torch.from_numpy(des2).unsqueeze(0).to(device)  # Shape: [1, num_kpts, desc_dim]

        # Prepare input dictionary for LightGlue
        data = {
            "image0": {"keypoints": kp1_tensor, "descriptors": des1_tensor},
            "image1": {"keypoints": kp2_tensor, "descriptors": des2_tensor},
        }

        # Debugging: print the keys of data
        print(f"Passing keys to LightGlue: {list(data.keys())}")  # Should be ['image0', 'image1']

        # Pass the data to LightGlue
        img_matches = superpoint_matcher(data)

        # Debugging: print the keys of img_matches
        print(f"Matches returned by LightGlue: {list(img_matches.keys())}")

        # Extract matches: matches0 contains indices of matches for image0
        if "matches0" not in img_matches:
            print("Error: 'matches0' not found in LightGlue output.")
            return []
        matches0 = img_matches["matches0"][0].cpu().numpy()
        good_matches = [(i, m) for i, m in enumerate(matches0) if m != -1]
        return good_matches
    else:
        # OpenCV-based matchers
        bf = getBFMatcher(featureType)
        if bf is None:
            return []
        if des1 is None or des2 is None:
            return []
        matches = bf.knnMatch(des1, des2, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
        return good_matches

def getKeypoints(kp1, kp2, scale, featureType):
    """
    Scale keypoints based on the transformation applied.
    """
    kp1_scaled = kp1 * scale
    kp2_scaled = kp2 * scale
    return kp1_scaled, kp2_scaled

def getResizedImages(img, scale):
    """
    Resize the image by the given scale factor.
    """
    return cv.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=cv.INTER_AREA)

def getRotatedImages(img, angle):
    """
    Rotate the image by the given angle.
    """
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)
    rot_mat = cv.getRotationMatrix2D(center, angle, 1.0)
    result = cv.warpAffine(img, rot_mat, (w, h))
    return result

def runFeatureExtractionTest(TestType, img, superpoint_extractor, superpoint_matcher):
    """
    Run feature extraction and matching tests for different feature types.
    """
    # Define number of iterations based on test type
    if TestType == 'ScaleTest':
        Iterations = 20
    elif TestType == 'RotationTest':
        Iterations = 36
    else:
        raise ValueError("TestType must be 'ScaleTest' or 'RotationTest'")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    pltArray = np.empty((4, Iterations))  # Now 4 feature types

    # Calculate match rate for each feature type
    for i in range(4):  # Iterate over feature types
        FeatureTypeInstance = FeatureType(i)
        print(f'Feature Type: {FeatureTypeInstance.name}')

        if FeatureTypeInstance == FeatureType.SUPERPOINT:
            # Prepare image tensor for SuperPoint
            img_tensor = numpy_image_to_torch(img).to(device)
            features = superpoint_extractor.extract(img_tensor.unsqueeze(0))
            kp1 = features["keypoints"][0].cpu().numpy()  # Shape: [num_kpts, 2]
            des1 = features["descriptors"][0].cpu().numpy()
        else:
            # Use OpenCV feature extractors
            algorithm = getFeatures(FeatureTypeInstance)
            kp1_cv, des1 = algorithm.detectAndCompute(img, None)
            if kp1_cv is None or des1 is None:
                kp1 = np.array([], dtype=np.float32).reshape(-1, 2)
                des1 = np.array([], dtype=np.float32).reshape(-1, algorithm.descriptorSize())
            else:
                kp1 = np.array([kp.pt for kp in kp1_cv], dtype=np.float32)

        # Initialize transformation parameters
        angle = 0
        scale = 0.2 if TestType == 'ScaleTest' else 1

        for j in range(Iterations):
            if TestType == 'ScaleTest':
                TransformedSrc = getResizedImages(img, scale)
                current_scale = scale
                scale = round(scale + 0.1, 1)
            elif TestType == 'RotationTest':
                TransformedSrc = getRotatedImages(img, angle)
                current_angle = angle
                angle += 10

            # Extract features from the transformed image
            if FeatureTypeInstance == FeatureType.SUPERPOINT:
                src_tensor = numpy_image_to_torch(TransformedSrc).to(device)
                features2 = superpoint_extractor.extract(src_tensor.unsqueeze(0))
                kp2 = features2["keypoints"][0].cpu().numpy()
                des2 = features2["descriptors"][0].cpu().numpy()
            else:
                algorithm = getFeatures(FeatureTypeInstance)
                kp2_cv, des2 = algorithm.detectAndCompute(TransformedSrc, None)
                if kp2_cv is None or des2 is None:
                    kp2 = np.array([], dtype=np.float32).reshape(-1, 2)
                    des2 = np.array([], dtype=np.float32).reshape(-1, algorithm.descriptorSize())
                else:
                    kp2 = np.array([kp.pt for kp in kp2_cv], dtype=np.float32)

            # Convert descriptors to the same type if necessary
            if des1.dtype != des2.dtype:
                des1 = des1.astype(np.float32)
                des2 = des2.astype(np.float32)

            # Rescale keypoints based on scale to match resized image
            if TestType == 'ScaleTest':
                kp1_scaled, kp2_scaled = getKeypoints(kp1, kp2, current_scale, FeatureTypeInstance)
            else:
                # For RotationTest, scale remains 1
                kp1_scaled, kp2_scaled = getKeypoints(kp1, kp2, 1, FeatureTypeInstance)

            # Match features
            if FeatureTypeInstance == FeatureType.SUPERPOINT:
                if kp1_scaled.size == 0 or kp2_scaled.size == 0:
                    matches = []
                else:
                    matches = getMatches(
                        des1, des2, FeatureTypeInstance,
                        superpoint_matcher=superpoint_matcher,
                        kp1=kp1_scaled, kp2=kp2_scaled
                    )
            else:
                if kp1_scaled.size == 0 or kp2_scaled.size == 0:
                    matches = []
                else:
                    matches = getMatches(des1, des2, FeatureTypeInstance)

            # Calculate match rate as percentage of keypoints matched
            MatchRate = len(matches) / len(kp1_scaled) * 100 if len(kp1_scaled) > 0 else 0

            pltArray[i, j] = MatchRate

            # Debugging output
            print(f'Iteration {j+1}/{Iterations}')
            print(f'Number of Keypoints (Image 1): {len(kp1_scaled)}')
            print(f'Number of Keypoints (Image 2): {len(kp2_scaled)}')
            if TestType == 'ScaleTest':
                print(f'Scale: {current_scale}')
            elif TestType == 'RotationTest':
                print(f'Angle: {current_angle}')
            print(f'Number of Matches: {len(matches)}')
            print(f'Match Rate: {MatchRate:.2f}%\n')

    return pltArray
def plotMatchRate(pltArray, TestType):
    """
    Plot the match rate for different feature types across transformations.
    """
    if TestType == 'ScaleTest':
        x = np.arange(0.2, 2.2, 0.1)
    elif TestType == 'RotationTest':
        x = np.arange(0, 360, 10)

    labels = ['SIFT', 'ORB', 'BRISK', 'SuperPoint']

    for i in range(4):
        plt.plot(x, pltArray[i], label=labels[i])

    plt.legend()
    plt.xticks(x[::2])    
    plt.xlabel('Scale' if TestType == "ScaleTest" else 'Angle')
    plt.ylabel('Match Rate (%)')
    plt.title(f'Match Rate vs {"Scale" if TestType == "ScaleTest" else "Rotation"}')
    plt.grid(True)
    plt.show()

def __main__():
    """
    Main function to execute the feature extraction and matching tests.
    """
    img_path = "datasets/vpair0-100/00074.png"
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    assert img is not None, f"Image not found at {img_path}"

    TestType = "ScaleTest" 

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using device: {device}')

    # Initialize SuperPoint extractor and LightGlue matcher
    superpoint_extractor = SuperPoint(max_num_keypoints=1024).eval().to(device)
    superpoint_matcher = LightGlue(features="superpoint").eval().to(device)

    # Run the feature extraction and matching test
    pltArray = runFeatureExtractionTest(TestType, img, superpoint_extractor, superpoint_matcher)

    # Plot the match rates
    plotMatchRate(pltArray, TestType)

if __name__ == "__main__":
    __main__()
