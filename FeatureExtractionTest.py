import numpy as np
import cv2 as cv
 
img = cv.imread('Billed Data/01899.jpg')
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
rotated = cv.rotate(gray, cv.ROTATE_90_CLOCKWISE)
blur = cv.GaussianBlur(gray,(9,9),0)

scale = 1

resizedSrc = cv.resize(gray, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation = cv.INTER_AREA)
resizedTransformed = cv.resize(blur, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation = cv.INTER_AREA)

sift = cv.SIFT_create(nfeatures=1000)

kp1, des1 = sift.detectAndCompute(resizedSrc,None)
kp2, des2 = sift.detectAndCompute(resizedTransformed,None)
 
bf = cv.BFMatcher()

# OrbMatches = bf.match(des1,des2)
SiftMatches = knnMatch = bf.knnMatch(des1,des2, k=2)

src=cv.drawKeypoints(resizedSrc,kp1,img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
srcTransformed=cv.drawKeypoints(resizedTransformed,kp2,blur, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#print number of keypoints

#Ratio Test:

good = []
for m,n in SiftMatches:
    if m.distance < 0.75*n.distance:
        good.append([m])

print(f'Number of Keypoints (Image 1): {len(kp1)}')
print(f'Number of Keypoints (Image 2): {len(kp2)}')
print(f'Number of Matches: {len(good)}')

#resize but keep aspect ratio

resizedSrc = cv.resize(src, (int(img.shape[1] * 0.3), int(img.shape[0] * 0.3)), interpolation = cv.INTER_AREA)
resizedTransformed = cv.resize(srcTransformed, (int(img.shape[1] * 0.3), int(img.shape[0] * 0.3)), interpolation = cv.INTER_AREA)

cv.imshow('sift_keypoints_src.jpg', resizedSrc)
cv.imshow('sift_keypoints_transformed.jpg', resizedTransformed)

cv.waitKey(0)

cv.destroyAllWindows()