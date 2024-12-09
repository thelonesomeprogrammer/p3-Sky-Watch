import cv2
import numpy as np
import torch
from lightglue.utils import numpy_image_to_torch
from lightglue.lightglue import LightGlue
from lightglue import SuperPoint


class SiftExtract:
    def __init__(self, nfeatures = 1000, nOctaveLayers = 3, contrastThreshold = 0.06, edgeThreshold = 10, sigma = 1.6):
        self.sift = cv2.SIFT_create(
            nfeatures = nfeatures, 
            nOctaveLayers = nOctaveLayers, 
            contrastThreshold = contrastThreshold, 
            edgeThreshold = edgeThreshold, 
            sigma = sigma
            )

    def extract(self, img):
        detections, descriptors = self.sift.detectAndCompute(img, None)
        points = np.array([k.pt for k in detections], dtype=np.float32)
        scores = np.array([k.response for k in detections], dtype=np.float32)
        scales = np.array([k.size for k in detections], dtype=np.float32)
        angles = np.deg2rad(np.array([k.angle for k in detections], dtype=np.float32))
        return FeaturesHolder(points, scores, scales, angles, descriptors, False)


class BFMatch:
    def __init__(self):
        self.bf = cv2.BFMatcher(crossCheck=False)

    def match(self, features0, features1, k = 2):
            matches = self.bf.knnMatch(features0.to_np(), features1.to_np(), k = k)


            matched_kp1 = [m[0].trainIdx for m in matches]
            matched_kp0 = [m[0].queryIdx for m in matches]
            return matched_kp0, matched_kp1

class FlannMatch:
    def __init__(self, index_params = dict(algorithm=5, trees=1 ), search_params = dict(checks=50 )):
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def match(self, features0, features1, k = 2):
        matches = self.flann.knnMatch(features0.to_np(), features1.to_np(), k = k)

        matched_kp1 = [m[0].trainIdx for m in matches]
        matched_kp0 = [m[0].queryIdx for m in matches]
        return matched_kp0, matched_kp1

class SuperExtract:
    def __init__(self, max_keypoints, device):
        self.device = device
        self.extractor = SuperPoint(max_num_keypoints = max_keypoints).eval().to(device)

    def extract(self, img):
        img = numpy_image_to_torch(img)
        res = self.extractor.extract(img.unsqueeze(0).to(self.device))
        return FeaturesHolder(res["keypoints"][0].cpu(), res["keypoint_scores"][0].cpu(), [], [], res["descriptors"][0].cpu(), True)


class LightMatch:
    def __init__(self, features, device):
        self.device = device
        self.matcher = LightGlue(features = features).eval().to(device)

    def match(self, features0, features1):

        match_dict = self.matcher.forward({ "image0": features0.to_Light(self.device), "image1": features1.to_Light(self.device) })

        matches1 = [int(t.cpu()) for t in match_dict["matches0"][0] if not torch.all(t == -1)]
        matches0 = [int(t.cpu()) for t in match_dict["matches1"][0] if not torch.all(t == -1)]

        return matches0, matches1

class FeaturesHolder:
    def __init__(self, points, scores, scales, angles, descriptors, is_tensor):
        self.points = points
        self.scores = scores
        self.scales = scales
        self.angles = angles
        self.descriptors = descriptors
        self.is_tensor = is_tensor

    def to_Light(self,device):
        if self.is_tensor:
            return {"keypoints": self.points.unsqueeze(0).to(device), "descriptors": self.descriptors.unsqueeze(0).to(device)}
        else:
            return {
                    "keypoints": torch.from_numpy(self.points).unsqueeze(0).to(device),
                    "scales": torch.from_numpy(self.scales).unsqueeze(0).to(device),
                    "scores": torch.from_numpy(self.scores).unsqueeze(0).to(device),
                    "oris": torch.from_numpy(self.angles).unsqueeze(0).to(device),
                    "descriptors": torch.from_numpy(self.descriptors).unsqueeze(0).to(device),
            }
    
    def to_np(self):
        if self.is_tensor:
            return self.descriptors.numpy()
        else:
            return self.descriptors

    def get_points(self):
        if self.is_tensor:
            return self.points.cpu()
        else:
            return self.points

    def mv_points(self, x, y):
        self.points[:, 1] += x
        self.points[:, 0] += y

