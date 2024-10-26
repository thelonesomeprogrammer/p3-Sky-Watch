import numpy as np
import cv2 as cv
import enum as Enum
import matplotlib.pyplot as plt
 
src = cv.imread('Billed Data/01899.jpg', cv.IMREAD_GRAYSCALE)

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
        return cv.BRISK_create(thresh=60)
    
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

pltArray = np.empty((3,20))

for i in range(3):

    FeatureTypeInstance = FeatureType(i)
    scale = 0.1

    for j in range(1, 20):

        scale = round(scale + 0.1, 1)

        img = cv.imread(f'Billed Data/01899.jpg', cv.IMREAD_GRAYSCALE)

        resizedSrc = getResizedImages(img, scale)

        algo = getFeatures(resizedSrc, FeatureTypeInstance)
    
        if j == 1:
            kp1, des1 = algo.detectAndCompute(img, None)

        kp2, des2 = algo.detectAndCompute(resizedSrc, None)

        kp1, kp2 = getKeypoints(kp1, kp2, scale)

        matches = getMatches(des1, des2, FeatureTypeInstance)

        MatchRate = len(matches)/len(kp1) * 100

        # good = getGoodMatches(matches, FeatureTypeInstance)

        pltArray[i,j] = MatchRate

        print(f'Number of Keypoints (Image 1): {len(kp1)}')
        print(f'Number of Keypoints (Image 2): {len(kp2)}')
        print(f'scale: {scale}')
        print(f'Number of Matches: {len(matches)}')

        srcKeypoints, srcTransformedKeypoints = getKeypointImages(resizedSrc, kp1, kp2)

x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2])
y1 = pltArray[0]
y2 = pltArray[1]
y3 = pltArray[2]

plt.plot(x, y1)
plt.plot(x, y2)
plt.plot(x, y3)
plt.legend()
plt.xticks(x[::2])
plt.xlabel('Scale')
plt.ylabel('Match Rate')
plt.title('Match Rate vs Scale')
plt.show()

