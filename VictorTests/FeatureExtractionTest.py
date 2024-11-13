import numpy as np
import cv2 as cv
import enum as Enum
import matplotlib.pyplot as plt

class FeatureType(Enum.Enum):
    SIFT = 0
    ORB = 1
    BRISK = 2

def getFeatures(featureType):
    if featureType == FeatureType.SIFT:
        return cv.SIFT_create(nfeatures=1000)
    elif featureType == FeatureType.ORB:
        return cv.ORB_create()
    elif featureType == FeatureType.BRISK:
        return cv.BRISK_create(thresh=80)
    
def getBFMatcher(featureType):
    if featureType == FeatureType.SIFT:
        return cv.BFMatcher(cv.NORM_L2, crossCheck=False)
    elif featureType in (FeatureType.ORB, FeatureType.BRISK):
        return cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
    
def getMatches(des1, des2, featureType):
    
    bf = getBFMatcher(featureType)

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

def runFeatureExtractionTest(TestType, img):
    # Define number of iterations based on test type
    if TestType == 'ScaleTest':
        Iterations = 20
    elif TestType == 'RotationTest':
        Iterations = 36

    pltArray = np.empty((3, Iterations))

    # Calculate match rate for each feature type: SIFT, ORB, BRISK
    for i in range(3):
        FeatureTypeInstance = FeatureType(i)
        algorithm = getFeatures(FeatureTypeInstance)
        kp1, des1 = algorithm.detectAndCompute(img, None)

        angle = 0

        if TestType == 'ScaleTest':
            scale = 0.2
        elif TestType == 'RotationTest':
            scale = 1

        scale = round(scale, 1)

        for j in range(0, Iterations):
            if TestType == 'ScaleTest':
                TransformedSrc = getResizedImages(img, scale)
                scale = round(scale + 0.1, 1)
            elif TestType == 'RotationTest':
                TransformedSrc = getRotatedImages(img, angle)
                angle += 10

            # Re-initialize the algorithm for the transformed image
            algorithm = getFeatures(FeatureTypeInstance)
            kp2, des2 = algorithm.detectAndCompute(TransformedSrc, None)

            # Convert descriptors to the same type if necessary
            if des1.dtype != des2.dtype:
                des1 = des1.astype(np.float32)
                des2 = des2.astype(np.float32)

            # Rescale keypoints based on scale to match resized image
            kp1, kp2 = getKeypoints(kp1, kp2, scale)

            matches = getMatches(des1, des2, FeatureTypeInstance)

            # Calculate match rate as percentage of keypoints matched
            MatchRate = len(matches) / len(kp1) * 100
            pltArray[i, j] = MatchRate

            # Debugging output
            print(f'Feature Type: {FeatureTypeInstance.name}')
            print(f'des1 dtype: {des1.dtype}, shape: {des1.shape}')
            print(f'des2 dtype: {des2.dtype}, shape: {des2.shape}')
            print(f'Number of Keypoints (Image 1): {len(kp1)}')
            print(f'Number of Keypoints (Image 2): {len(kp2)}')
            if TestType == 'ScaleTest':
                print(f'Scale: {scale}')
            elif TestType == 'RotationTest':
                print(f'Angle: {angle}')
            print(f'Number of Matches: {len(matches)}\n')

    return pltArray

def plotMatchRate(pltArray, TestType):
    if TestType == 'ScaleTest':
        x = np.arange(0.2, 2.2, 0.1)
    elif TestType == 'RotationTest':
        x = np.arange(0, 360, 10)

    y1 = pltArray[0]
    y2 = pltArray[1]
    y3 = pltArray[2]

    print(np.round(pltArray))

    plt.plot(x, y1, label='SIFT')
    plt.plot(x, y2, label='ORB')
    plt.plot(x, y3, label='BRISK')

    plt.legend()
    plt.xticks(x[::2])    
    plt.xlabel('Scale')
    plt.ylabel('Match Rate')
    plt.title('Match Rate vs Scale')
    plt.show()

def __main__():
    img = cv.imread('drone_views/00074.png', cv.IMREAD_GRAYSCALE)

    # Define test type: ScaleTest or RotationTest
    TestType = 'ScaleTest'

    pltArray = runFeatureExtractionTest(TestType, img)

    plotMatchRate(pltArray, TestType)


__main__()