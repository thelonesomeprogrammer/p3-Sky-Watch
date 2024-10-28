import cv2 as cv
import numpy as np
#04325

droneImage_color = cv.imread('Billed Data/05921.jpg', cv.IMREAD_COLOR)
SatImage_color = cv.imread('SatData/StovringRoadCorner.jpg', cv.IMREAD_COLOR)

droneImage_gray = cv.cvtColor(droneImage_color, cv.COLOR_BGR2GRAY)
SatImage_gray = cv.cvtColor(SatImage_color, cv.COLOR_BGR2GRAY)

# Initiate SIFT detector
sift = cv.SIFT_create(nfeatures=100000, contrastThreshold=0.1, edgeThreshold=5, sigma = 1.4)

# find the keypoints and descriptors with SIFT

print("status1")

kp1, des1 = sift.detectAndCompute(droneImage_gray, None)
kp2, des2 = sift.detectAndCompute(SatImage_gray, None)

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
    matched_image = cv.drawMatches(droneImage_color, kp1, SatImage_color, kp2, inliers, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    cv.imshow('Inlier Matches with RANSAC', matched_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
else:
    print("Not enough matches found to apply RANSAC.")