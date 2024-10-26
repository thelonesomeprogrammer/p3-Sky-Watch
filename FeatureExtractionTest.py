import numpy as np
import cv2 as cv
import enum as Enum
import matplotlib.pyplot as plt
 
img = cv.imread('Billed Data/04345.jpg', cv.IMREAD_GRAYSCALE)

print(img.shape)

class FeatureType(Enum.Enum):
    SIFT = 0
    ORB = 1
    BRISK = 2

def getFeatures(img, featureType):
    if featureType == FeatureType.SIFT:
        return cv.SIFT_create(nfeatures=1000)
    elif featureType == FeatureType.ORB:
        return cv.ORB_create()
    elif featureType == FeatureType.BRISK:
        return cv.BRISK_create(thresh=80)
    
def getMatches(des1, des2, featureType):
    bf = cv.BFMatcher()

    if featureType == FeatureType.SIFT:
        matches = bf.knnMatch(des1, des2, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
        return good_matches
    
    elif featureType == FeatureType.ORB or featureType == FeatureType.BRISK:
        matches = bf.knnMatch(des1, des2, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
        return good_matches
    
def getKeypoints(kp1, kp2, scale):
    for kp in kp1:
        kp.size *= (1 * scale)

    for kp in kp2:
        kp.size *= (1 * scale)
        
    return kp1, kp2

def getResizedImages(img, scale):
    return cv.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation = cv.INTER_AREA)

def getRotatedImages(img, angle):
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)
    rot_mat = cv.getRotationMatrix2D(center, angle, 1.0)
    result = cv.warpAffine(img, rot_mat, (w, h))
    return result

def getKeypointImages(img, kp1, kp2):
    return cv.drawKeypoints(img, kp1, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS), cv.drawKeypoints(img, kp2, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

def getGoodMatches(matches, featureType):
    if featureType == FeatureType.ORB:
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
        return good
    # elif featureType == FeatureType.SIFT or featureType == FeatureType.SURF or featureType == FeatureType.AKAZE:
    #     for match in matches:
    #         if match.distance < 0.75:
    #             good.append(match)
    # return good

# FeatureType = FeatureType.SIFT

TestType = 'RotationTest'

if TestType == 'ScaleTest':
    Iterations = 20
elif TestType == 'RotationTest':
    Iterations = 36

pltArray = np.empty((3,Iterations))

for i in range(3):

    FeatureTypeInstance = FeatureType(i)

    angle = 0

    if TestType == 'ScaleTest':
        scale = 0.1
    else:
        scale = 1

    for j in range(1, Iterations):

        if TestType == 'ScaleTest':
            scale = round(scale + 0.1, 1)
            TransformedSrc = getResizedImages(img, scale)
        elif TestType == 'RotationTest':
            angle = angle + 10
            TransformedSrc = getRotatedImages(img, angle)

        algo = getFeatures(TransformedSrc, FeatureTypeInstance)
    
        if j == 1:

            kp1, des1 = algo.detectAndCompute(img, None)

            kp2, des2 = kp1, des1
            matches = getMatches(des1, des2, FeatureTypeInstance)
            MatchRate = len(matches) / len(kp1) * 100
            pltArray[i, 0] = MatchRate 

        kp2, des2 = algo.detectAndCompute(TransformedSrc, None)

        if des1.dtype != des2.dtype:
            des1 = des1.astype(np.float32)
            des2 = des2.astype(np.float32)

        kp1, kp2 = getKeypoints(kp1, kp2, scale)

        matches = getMatches(des1, des2, FeatureTypeInstance)

        MatchRate = len(matches)/len(kp1) * 100

        # good = getGoodMatches(matches, FeatureTypeInstance)

        pltArray[i,j] = MatchRate

        print(f'Number of Keypoints (Image 1): {len(kp1)}')
        print(f'Number of Keypoints (Image 2): {len(kp2)}')
        if TestType == 'ScaleTest':
            print(f'scale: {scale}')
        elif TestType == 'RotationTest':
            print(f'angle: {angle}')
        print(f'Number of Matches: {len(matches)}')

        srcKeypoints, srcTransformedKeypoints = getKeypointImages(TransformedSrc, kp1, kp2)


if TestType == 'ScaleTest':
    x = np.arange(0.1, 2.1, 0.1)
elif TestType == 'RotationTest':
    x = np.arange(0, 360, 10)

y1 = pltArray[0]
y2 = pltArray[1]
y3 = pltArray[2]

plt.plot(x, y1, label='SIFT')
plt.plot(x, y2, label='ORB')
plt.plot(x, y3, label='BRISK')

plt.legend()
plt.xticks(x[::2])
plt.xlabel('Scale')
plt.ylabel('Match Rate')
plt.title('Match Rate vs Scale')
plt.show()

