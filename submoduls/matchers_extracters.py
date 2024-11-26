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
        kp, des = self.sift.detectAndCompute(img, None)
        kp = np.array([kp.pt for kp in kp], dtype=np.float32)
        return kp, des


class BFMatch:
    def __init__(self):
        self.bf = cv2.BFMatcher(crossCheck=False)

    def match(self, keypoints0, descriptors0, keypoints1, descriptors1):
            matches = self.bf.knnMatch(descriptors0, descriptors1, k=2)


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
        return res["keypoints"][0].cpu(), res["descriptors"][0].cpu()


class LightMatch:
    def __init__(self, features, device):
        self.device = device
        self.matcher = LightGlue(features = features).eval().to(device)

    def match(self, keypoints0, descriptors0, keypoints1, descriptors1):
        match_dict = self.matcher.forward({
            "image0": {
                "keypoints":keypoints0.unsqueeze(0).to(self.device),
                "descriptors":descriptors0.unsqueeze(0).to(self.device),
                }, 
            "image1":{
                "keypoints":keypoints1.unsqueeze(0).to(self.device),
                "descriptors":descriptors1.unsqueeze(0).to(self.device),
                }
            })

        matches1 = [int(t.cpu()) for t in match_dict["matches0"][0] if not torch.all(t == -1)]
        matches0 = [int(t.cpu()) for t in match_dict["matches1"][0] if not torch.all(t == -1)]

        return matches0, matches1
