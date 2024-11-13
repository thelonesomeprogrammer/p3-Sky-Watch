import torch
import cv2 as cv
import numpy as np

def load_r2d2_model(model_path):
    model = torch.load('VictorTests/Pytorch/faster2d2_WASF_N16.pt')  # Load the model weights
    model.eval()  # Set to evaluation mode
    return model

def extract_features(model, image):
    # Preprocess the image
    image_tensor = torch.from_numpy(image).float().unsqueeze(0)  # Adjust this as needed
    with torch.no_grad():
        features = model(image_tensor)  # Run the model
    return features
