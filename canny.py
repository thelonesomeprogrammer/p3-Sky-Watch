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

_, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)

# Perform morphological closing to fill small gaps
kernel = np.ones((3, 3), np.uint8)  # You can adjust the kernel size for different effects
closed = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)

contours, hierarchy = cv.findContours(closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

mask = np.zeros_like(gray)

# Fill the mask with the contours
cv.drawContours(mask, contours, -1, (255), thickness=cv.FILLED)

cv.drawContours(img, contours, -1, (0, 255, 0), 2)

# Apply the mask to the original image
masked_image = cv.bitwise_and(img, img, mask=mask)

# Display or save the result
cv.imshow("Masked Image", masked_image)
cv.imshow("Original Image", img)
cv.waitKey(0)
cv.destroyAllWindows()
