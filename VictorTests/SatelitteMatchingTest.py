import cv2 as cv
import numpy as np
#04325

srcDrone = cv.imread('Billed Data/05671.jpg', cv.IMREAD_COLOR)
srcSat = cv.imread('SatData/StovringRoadCorner.jpg', cv.IMREAD_COLOR)

#resize

srcDrone = cv.resize(srcDrone, (int(srcDrone.shape[1] * 0.2), int(srcDrone.shape[0] * 0.2)), interpolation = cv.INTER_AREA)
srcSat = cv.resize(srcSat, (int(srcSat.shape[1] * 0.2), int(srcSat.shape[0] * 0.2)), interpolation = cv.INTER_AREA)

droneImg_NoGreen = srcDrone.copy()
droneImg_NoGreen[:, :, 1] = droneImg_NoGreen[:, :, 1]/2

satImage_NoGreen = srcSat.copy()
satImage_NoGreen[:, :, 1] = satImage_NoGreen[:, :, 1]/2

# src grayscale

droneImage_gray = cv.cvtColor(srcDrone, cv.COLOR_BGR2GRAY)
SatImage_gray = cv.cvtColor(srcSat, cv.COLOR_BGR2GRAY)

# grayscale no green

sat_BR = cv.cvtColor(satImage_NoGreen, cv.COLOR_BGR2GRAY)
drone_BR = cv.cvtColor(droneImg_NoGreen, cv.COLOR_BGR2GRAY)

srcDroneBlurred = cv.GaussianBlur(srcDrone, (15, 15), 3)
srcSatBlurred = cv.GaussianBlur(srcSat, (15, 15), 3)

edgesDrone = cv.Canny(srcDroneBlurred, threshold1=30, threshold2=200, apertureSize=7)
edgesSat = cv.Canny(srcSatBlurred, threshold1=30, threshold2=200, apertureSize=7)


# cv.imshow('Drone Image', edgesDrone)
# cv.imshow('no green', edgesSat)

# cv.waitKey(0)
# cv.destroyAllWindows()

# Initiate SIFT detector
sift = cv.SIFT_create(nfeatures=100000, contrastThreshold=0.01, edgeThreshold=3)

# find the keypoints and descriptors with SIFT
def __main__():

    print("status1")

    kp1, des1 = sift.detectAndCompute(edgesDrone, None)
    kp2, des2 = sift.detectAndCompute(edgesSat, None)

    print(f'Found {len(kp1)} keypoints in the drone image.')
    print(f'Found {len(kp2)} keypoints in the satellite image.')

    # display both images with keypoints

    print("status2")

    # droneImage = cv.drawKeypoints(droneImage, kp1, None)
    # SatImage = cv.drawKeypoints(SatImage, kp2, None)

    # cv.imshow('Drone Image', droneImage)
    # cv.imshow('Sat Image', SatImage)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    print("status3")

    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    print('Matches:', len(matches))

    # show matches

    print("status4")

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    print('Good matches:', len(good_matches))

    # cv.drawMatchesKnn expects list of lists as matches.

    # matched_image = cv.drawMatches(
    #     droneImage, kp1,
    #     SatImage, kp2,
    #     good_matches, None,
    #     flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    # )

    # cv.imshow('Matches', matched_image)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    if len(good_matches) >= 4:  # RANSAC needs at least 4 points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Use RANSAC to estimate homography and filter inliers
        homography, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        inliers = [good_matches[i] for i in range(len(good_matches)) if mask[i]]

        # Draw only inlier matches for better accuracy
        matched_image = cv.drawMatches(srcDrone, kp1, srcSat, kp2, inliers, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        print('Inliers:', len(inliers))

        cv.imshow('Inlier Matches with RANSAC', matched_image)
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        print("Not enough matches found to apply RANSAC.")

# __main__()

scale = 1.4            # Scale factor, <1 detects shorter lines, >1 detects longer lines
sigma_scale = 1.1     # Controls Gaussian smoothing (higher = more smoothing)
quant = 2.2           # Quantization of the gradient angle, lower values detect more lines
ang_th = 16          # Angle tolerance in degrees, lower values are more selective
log_eps = 1          # Detection threshold for small lines, larger for longer lines
density_th = 0    # Minimum density of region pixels, 0 to 1, higher values are stricter
n_bins = 1024          # Number of bins in quantization of orientation, default is 1024

# Initialize LSD detector
lsd = cv.createLineSegmentDetector(cv.LSD_REFINE_STD, scale, sigma_scale, quant, ang_th, log_eps, density_th, n_bins)

# Detect lines in the grayscale image
lines = lsd.detect(droneImage_gray)[0]  # `lines` is a list of detected line segments

lines_mask = np.zeros_like(droneImage_gray)

extension_length = 10

# Draw the detected lines on a copy of the source image
line_image = srcSat.copy()
if lines is not None:
    for line in lines:
        x0, y0, x1, y1 = map(int, line[0])  # Line coordinates
        cv.line(lines_mask, (x0, y0), (x1, y1), 255, 1, cv.LINE_AA)

        dx = x1 - x0
        dy = y1 - y0
        line_length = np.sqrt(dx ** 2 + dy ** 2)

        # Normalize direction and extend both endpoints
        if line_length > 0:  # Avoid division by zero
            extend_x = int(extension_length * (dx / line_length))
            extend_y = int(extension_length * (dy / line_length))

            # New extended line endpoints
            new_x0 = x0 - extend_x
            new_y0 = y0 - extend_y
            new_x1 = x1 + extend_x
            new_y1 = y1 + extend_y

            # Draw the extended line on the mask
            cv.line(lines_mask, (new_x0, new_y0), (new_x1, new_y1), 255, 1, cv.LINE_AA)  # Use 255 for white in grayscale

contours, _ = cv.findContours(lines_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contours_mask = np.zeros_like(droneImage_gray)
cv.drawContours(contours_mask, contours, -1, 255, thickness=cv.FILLED)

cv.imshow('mask2', lines_mask)
cv.imshow('mask', contours_mask)
cv.waitKey(0)
cv.destroyAllWindows()
 
print('Number of contours:', len(contours))
#size of contours

for contour in contours:
    area = cv.contourArea(contour)

    if area < 2000 or area > 200000:
        cv.drawContours(contours_mask, [contour], -1, (0, 0, 0), thickness=cv.FILLED)
    else: print('Contour area:', area)


inverted_mask = cv.bitwise_not(contours_mask)

# Step 2: Perform connected component analysis
num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(inverted_mask, connectivity=8)

# Step 3: Create a mask to hold the filtered components
filtered_mask = np.zeros_like(inverted_mask)

# Define area thresholds
min_area = 10000  # Minimum area to keep
max_area = 100000  # Maximum area to keep

# Step 4: Iterate over each component, filtering by area
for i in range(1, num_labels):  # Start from 1 to skip the background
    area = stats[i, cv.CC_STAT_AREA]
    if min_area <= area <= max_area:
        # Keep the component if its area is within the defined range
        filtered_mask[labels == i] = 255

# Step 5: Invert the mask back to the original style (black regions as holes)
final_mask = cv.bitwise_not(filtered_mask)

# Display results
cv.imshow("Original Contours Mask", contours_mask)
cv.imshow("Filtered Black Regions", final_mask)
cv.waitKey(0)
cv.destroyAllWindows()

#apply opening on contours binary image

# filled_mask = cv.morphologyEx(contours_mask, cv.MORPH_CLOSE, kernel)

# # Set up the SimpleBlobDetector with parameters
# params = cv.SimpleBlobDetector_Params()

# # Filter by Area to focus on significant blobs
# params.filterByArea = True
# params.minArea = 1  # Minimum blob area
# params.maxArea = 1000000  # Maximum blob area

# _, binary_image = cv.threshold(closed_mask, 127, 255, cv.THRESH_BINARY)

# params.filterByColor = True
# params.blobColor = 0  # Detect black blobs

# # Create a blob detector with the specified parameters
# detector = cv.SimpleBlobDetector_create(params)

# # Detect blobs in the binary image
# keypoints = detector.detect(closed_mask)

# print(f'Found {len(keypoints)} blobs.')

# # Draw detected blobs as red circles on the original image for visualization
# blob_image = cv.cvtColor(closed_mask, cv.COLOR_GRAY2BGR)
# for kp in keypoints:
#     x, y = int(kp.pt[0]), int(kp.pt[1])
#     radius = int(kp.size / 2)
#     cv.circle(blob_image, (x, y), radius, (0, 0, 255), 2)  # Red circles for blobs


# lines2 = lsd.detect(line_image)[0]

# line_image2 = line_image.copy()
# if lines is not None:
#     for line in lines:
#         x0, y0, x1, y1 = map(int, line[0])  # Line coordinates
#         cv.line(line_image, (x0, y0), (x1, y1), (0, 255, 0), 1, cv.LINE_AA)

# Display the original and line-detected images
# cv.imshow("Original Image", srcDrone)
# cv.imshow("Detected Lines", line_image)
# cv.waitKey(0)
# cv.destroyAllWindows()