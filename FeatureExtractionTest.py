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

<<<<<<< Updated upstream
good = []
for m,n in SiftMatches:
    if m.distance < 0.75*n.distance:
        good.append([m])
=======
def runFeatureExtractionTest(TestType, img, algorithm):
    
# Define number of iterations based on test type
    if TestType == 'ScaleTest':
        Iterations = 20
    elif TestType == 'RotationTest':
        Iterations = 36
>>>>>>> Stashed changes

print(f'Number of Keypoints (Image 1): {len(kp1)}')
print(f'Number of Keypoints (Image 2): {len(kp2)}')
print(f'Number of Matches: {len(good)}')

#resize but keep aspect ratio

resizedSrc = cv.resize(src, (int(img.shape[1] * 0.3), int(img.shape[0] * 0.3)), interpolation = cv.INTER_AREA)
resizedTransformed = cv.resize(srcTransformed, (int(img.shape[1] * 0.3), int(img.shape[0] * 0.3)), interpolation = cv.INTER_AREA)

cv.imshow('sift_keypoints_src.jpg', resizedSrc)
cv.imshow('sift_keypoints_transformed.jpg', resizedTransformed)

cv.waitKey(0)

<<<<<<< Updated upstream
cv.destroyAllWindows()
=======
        for j in range(1, Iterations):

            if TestType == 'ScaleTest':
                scale = round(scale + 0.1, 1)
                TransformedSrc = getResizedImages(img, scale)
            elif TestType == 'RotationTest':
                angle = angle + 10
                TransformedSrc = getRotatedImages(img, angle)

            algorithm = getFeatures(TransformedSrc, FeatureTypeInstance)
        
            if TestType == 'RotationTest':

                kp2, des2 = kp1, des1
                matches = getMatches(des1, des2, FeatureTypeInstance)
                MatchRate = len(matches) / len(kp1) * 100
                pltArray[i, 0] = MatchRate 

            kp2, des2 = algorithm.detectAndCompute(TransformedSrc, None)

            if des1.dtype != des2.dtype:
                des1 = des1.astype(np.float32)
                des2 = des2.astype(np.float32)

            # Rescale keypoints based on scale to match resized image
            kp1, kp2 = getKeypoints(kp1, kp2, scale)

            matches = getMatches(des1, des2, FeatureTypeInstance)

            # Calculate match rate as percentage of keypoints matched
            MatchRate = len(matches)/len(kp1) * 100

            pltArray[i,j] = MatchRate

            print(f'Number of Keypoints (Image 1): {len(kp1)}')
            print(f'Number of Keypoints (Image 2): {len(kp2)}')
            if TestType == 'ScaleTest':
                print(f'scale: {scale}')
            elif TestType == 'RotationTest':
                print(f'angle: {angle}')
            print(f'Number of Matches: {len(matches)}')

            srcKeypoints, srcTransformedKeypoints = getKeypointImages(TransformedSrc, kp1, kp2)

    return pltArray

def plotMatchRate(pltArray, TestType):
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

def __main__():
    img = cv.imread('Billed Data/04345.jpg', cv.IMREAD_GRAYSCALE)

    # Define test type: ScaleTest or RotationTest
    TestType = 'ScaleTest'
    featureType = FeatureType.SIFT

    algorithm = getFeatures(img, featureType)

    pltArray = runFeatureExtractionTest(TestType, img, algorithm)

    plotMatchRate(pltArray, TestType)

__main__()
>>>>>>> Stashed changes
