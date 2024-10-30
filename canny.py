import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
 
src = cv.imread('Data/Stovring1.jpg')

img = cv.resize(src, (int(src.shape[1] * 0.2), int(src.shape[0] * 0.2)), interpolation = cv.INTER_AREA)

cv.imshow('Original Image', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

blur = cv.GaussianBlur(img,(5,5),0)

kernel = np.ones((5,5),np.uint8)

closing = cv.morphologyEx(blur, cv.MORPH_OPEN, kernel)

# cv.imshow('c', closing)

edges = cv.Canny(blur, threshold1=30, threshold2=150, apertureSize=3, L2gradient=True)



# Display or save the result
cv.imshow("Masked Image", edges)
cv.imshow("Original Image", img)
cv.waitKey(0)
cv.destroyAllWindows()
