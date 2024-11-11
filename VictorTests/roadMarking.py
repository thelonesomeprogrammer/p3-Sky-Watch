import cv2 as cv
import numpy as np
import random
#04325

srcDrone = cv.imread('Billed Data/04325.jpg', cv.IMREAD_COLOR)
srcSat = cv.imread('SatData/StovringRoadCorner.jpg', cv.IMREAD_COLOR)

droneImg = cv.resize(srcDrone, (int(srcDrone.shape[1] * 0.2), int(srcDrone.shape[0] * 0.2)), interpolation = cv.INTER_AREA)
satImg = cv.resize(srcSat, (int(srcSat.shape[1] * 0.2), int(srcSat.shape[0] * 0.2)), interpolation = cv.INTER_AREA)

def roadMarking(img):
    HSV_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # thresholding to remove green

    lower_green = np.array([30, 20, 20])
    upper_green = np.array([70, 255, 255])

    mask = cv.inRange(HSV_image, lower_green, upper_green)
    HSV_image[mask == 255] = [0, 0, 0]

    # thresholding to enchance dark blue colors

    cv.imshow('HSV1', HSV_image)

    lower_blue = np.array([80, 20, 50])
    upper_blue = np.array([120, 255, 255])

    mask = cv.inRange(HSV_image, lower_blue, upper_blue)
    HSV_image[mask == 255] = [255, 0, 0]

    h, s, v = cv.split(HSV_image)

    cv.imshow('HSV2', HSV_image)

    # thresholding to remove shadows from roads

    # clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(12,12))
    # v_clahe = clahe.apply(v)

    # # Merge channels and convert back to BGR
    # hsv_clahe = cv.merge([h, s, v_clahe])
    # result_image = cv.cvtColor(hsv_clahe, cv.COLOR_HSV2BGR)

    # # cv.imshow('src', img)
    # cv.imshow('Thresholded Image', result_image)
    # cv.imwrite('Thresholded Image.jpg', HSV_image)


    # Gaussian blur
    blur = cv.GaussianBlur(HSV_image, (9, 9), 0)

    # cv.imshow('Blur', blur)

    # Canny edge detection
    edges = cv.Canny(blur, 60, 180, apertureSize=3)

    cv.imshow('Edges1', edges)

    #closing
    kernel = np.ones((3, 3), np.uint8)
    # opening = cv.morphologyEx(edges, cv.MORPH_OPEN, kernel)
    closing = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)

    border_size = 10  # Set the size of the border as needed
    image_with_border = cv.copyMakeBorder(closing, border_size, border_size, border_size, border_size, cv.BORDER_CONSTANT, value=255)


    inverted_edges = cv.bitwise_not(image_with_border)

    cv.imshow('Edges', inverted_edges)


    # filled_image = inverted_edges.copy()
    # h, w = filled_image.shape[:2]
    # mask = np.zeros((h + 2, w + 2), np.uint8)

    # # Flood fill from the borders to fill the background
    # for x in range(w):
    #     if filled_image[0, x] == 255:
    #         cv.floodFill(filled_image, mask, (x, 0), 0)
    #     if filled_image[h - 1, x] == 255:
    #         cv.floodFill(filled_image, mask, (x, h - 1), 0)
    # for y in range(h):
    #     if filled_image[y, 0] == 255:
    #         cv.floodFill(filled_image, mask, (0, y), 0)
    #     if filled_image[y, w - 1] == 255:
    #         cv.floodFill(filled_image, mask, (w - 1, y), 0)

    # # Invert back to get filled regions as white on black
    # filled_image = cv.bitwise_not(filled_image)

    # Step 3: Find connected components
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(inverted_edges, connectivity=8)

    # Step 4: Filter out small components (noise removal)
    min_size = 500  # Minimum size of components to keep
    filtered_image = np.zeros_like(labels, dtype=np.uint8)

    for label in range(1, num_labels):
        area = stats[label, cv.CC_STAT_AREA]
        if area >= min_size:
            filtered_image[labels == label] = 255

    # Step 5: Find connected components again on the filtered image
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(filtered_image, connectivity=8)

    # Step 6: Create a color output image
    output_image = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)

    # Assign random colors to each connected component
    for label in range(1, num_labels):
        color = [random.randint(0, 255) for _ in range(3)]
        output_image[labels == label] = color

    # Display or save the result
    cv.imshow('Colored Major Areas', output_image)
    # cv.imwrite('Colored Major Areas.jpg', output_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # find contours

    output_gray = cv.cvtColor(output_image, cv.COLOR_BGR2GRAY)

    contours, _ = cv.findContours(output_gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # draw contours
    for contour in contours:
        cv.drawContours(img, contours, -1, (0, 255, 0), 2)

    # remove small contours

    for contour in contours:
        if cv.contourArea(contour) < 1000:
            cv.drawContours(img, contours, -1, (0, 0, 0), thickness=cv.FILLED)

    # cv.imshow('Contours', img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()


    # kernel = np.ones((3, 3), np.uint8)
    # closing = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)

    # cv.imshow('Closing', closing)

    # find contours

    # contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # # draw contours
    # img = img.copy()
    
    # # filter small contours

    # largeContours = 0

    # for contour in contours:
    #     print('contours amount:', len(contours))
    #     if cv.contourArea(contour) > 100:
    #         cv.drawContours(img, contours, -1, (0, 255, 0), 2)
    #         largeContours += 1

    # print(largeContours)

    # cv.imshow('Contours', img)

    return img

def __main__():
    roadMarking(droneImg)
    # roadMarking(satImg)
    # cv.imshow('Drone Image', droneImg)
    # cv.imshow('Satellite Image', satImg)
    cv.waitKey(0)
    cv.destroyAllWindows()

__main__()