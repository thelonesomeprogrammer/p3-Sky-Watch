import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
 
src = cv.imread('Billed Data/04325.jpg')

img = cv.resize(src, (int(src.shape[1] * 0.3), int(src.shape[0] * 0.3)), interpolation = cv.INTER_AREA)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

blur = cv.GaussianBlur(img,(3,3),0)

kernel = np.ones((9,9),np.uint8)

closing = cv.morphologyEx(blur, cv.MORPH_OPEN, kernel)

cv.imshow('c', closing)

edges = cv.Canny(blur, threshold1=30, threshold2=150, apertureSize=3, L2gradient=True)

contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

cv.drawContours(img, contours, -1, (0,255,0), 3)

cv.imshow('Contours', img)

cv.waitKey(0)
cv.destroyAllWindows()

# plt.subplot(121),plt.imshow(img,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(edges,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
 
# plt.show()
