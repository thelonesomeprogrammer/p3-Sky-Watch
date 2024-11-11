from R2D2_module import load_r2d2_model, extract_features
import cv2 as cv

model = load_r2d2_model('path_to/r2d2_WASF_N16.pt')
image = cv.imread('SatData/StovringRoadCorner.jpg', cv.IMREAD_GRAYSCALE)
features = extract_features(model, image)

