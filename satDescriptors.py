import cv2 as cv
import pickle
import os

sift = cv.SIFT_create()

def serialize_keypoints(keypoints):
    serialized = []
    for kp in keypoints:
        serialized.append((
            kp.pt,      # (x, y) coordinates
            kp.size,    # Diameter of the meaningful keypoint neighborhood
            kp.angle,   # Orientation of the keypoint
            kp.response, # The response by which the most strong keypoints have been selected
            kp.octave,  # Octave (pyramid layer) from which the keypoint has been extracted
            kp.class_id # Object class (if any)
        ))
    return serialized

# Path to your high-resolution satellite image
image_path = 'SatData/Stovring8K(1).jpg'  # Replace with your image file name

# Read the image in grayscale
image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError(f"Image file '{image_path}' not found.")

# Extract keypoints and descriptors
keypoints, descriptors = sift.detectAndCompute(image, None)
print(f"Extracted {len(keypoints)} keypoints.")

# Serialize keypoints
serialized_keypoints = serialize_keypoints(keypoints)

# Prepare data dictionary
data = {
    'keypoints': serialized_keypoints,
    'descriptors': descriptors
}

# Define the feature file path
feature_file = 'satellite_image_features.pkl'  # You can change the file name as needed

# Save the serialized data using pickle
with open(feature_file, 'wb') as f:
    pickle.dump(data, f)

print(f"Features saved to '{feature_file}'.")