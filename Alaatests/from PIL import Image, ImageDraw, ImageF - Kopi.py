import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

drone_image_path = '00359.png'
sat_image_path = 'SIFTOGB.jpg'

# Load images in color
img_drone = cv2.imread(drone_image_path, cv2.IMREAD_COLOR)
img_sat = cv2.imread(sat_image_path, cv2.IMREAD_COLOR)

if img_drone is None or img_sat is None:
    raise IOError("Check that the image paths are correct and images exist.")

# Convert to grayscale
gray_drone = cv2.cvtColor(img_drone, cv2.COLOR_BGR2GRAY)
gray_sat = cv2.cvtColor(img_sat, cv2.COLOR_BGR2GRAY)

# Template matching works best on normalized data
# Optionally, you can apply some contrast normalization or smoothing
# gray_drone = cv2.equalizeHist(gray_drone)
# gray_sat = cv2.equalizeHist(gray_sat)

h, w = gray_drone.shape[:2]

# Parameters for brute-force search
scale_start = 0.5   # start at half size
scale_end = 2.0     # up to double size
scale_step = 0.1    # scale step
angle_step = 15      # rotate every 15 degrees from -90 to +90 degrees

best_val = -1
best_loc = None
best_scale = None
best_angle = None
best_rotated_template = None

print("Starting brute-force search... This could take a while.")

# Loop over scales
scale = scale_start
while scale <= scale_end:
    # Resize drone image at this scale
    scaled_h = int(h * scale)
    scaled_w = int(w * scale)
    if scaled_h < 10 or scaled_w < 10:
        scale += scale_step
        continue
    resized_template = cv2.resize(gray_drone, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
    
    # Loop over angles
    for angle in range(-90, 91, angle_step):
        # Rotate the resized template around its center
        M = cv2.getRotationMatrix2D((scaled_w/2, scaled_h/2), angle, 1.0)
        rotated_template = cv2.warpAffine(resized_template, M, (scaled_w, scaled_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        
        # If the rotated template is larger than the satellite image in any dimension, skip
        sat_h, sat_w = gray_sat.shape[:2]
        if rotated_template.shape[0] > sat_h or rotated_template.shape[1] > sat_w:
            continue

        # Perform template matching
        # Use TM_CCOEFF_NORMED to get a score between -1 and 1
        res = cv2.matchTemplate(gray_sat, rotated_template, cv2.TM_CCOEFF_NORMED)
        # Find the best match location
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # Update best match if this one is better
        if max_val > best_val:
            best_val = max_val
            best_loc = max_loc
            best_scale = scale
            best_angle = angle
            best_rotated_template = rotated_template.copy()

    scale += scale_step

if best_loc is not None and best_val > 0.7:  # Threshold depends on the scenario
    print(f"Found a good match with value={best_val:.3f}, scale={best_scale}, angle={best_angle} degrees.")
    # Draw rectangle on the satellite image
    top_left = best_loc
    t_h, t_w = best_rotated_template.shape[:2]
    bottom_right = (top_left[0] + t_w, top_left[1] + t_h)

    img_result = img_sat.copy()
    cimage=cv2.rectangle(img_result, top_left, bottom_right, (0, 255, 0), 3)
    plt.imshow(cimage)
    plt.axis('off')
    plt.show()

else:
    print("No suitable match found with this brute-force approach. Consider adjusting parameters.")
