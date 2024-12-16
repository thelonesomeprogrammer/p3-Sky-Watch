from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from LightGlue.lightglue.lightglue import LightGlue
from LightGlue.lightglue.utils import load_image, rbd
from LightGlue.lightglue import viz2d, SuperPoint
from tqdm import tqdm  # For progress bars
from sklearn.cluster import DBSCAN
from matplotlib import cm
import cv2 as cv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
matcher = LightGlue(features="superpoint").eval().to(device)

image0 = load_image("00359.png")
image1 = load_image("11.jpg")


print("extracting features")
feats0 = extractor.extract(image0.to(device))

print("extracting features")
feats1 = extractor.extract(image1.to(device))

print("matching features")
matches01 = matcher({"image0": feats0, "image1": feats1})

print("plotting")
feats0, feats1, matches01 = [
    rbd(x) for x in [feats0, feats1, matches01]
]  # remove batch dimension


kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

# Use RANSAC to filter out bad matches
src_pts = m_kpts0.cpu().numpy().reshape(-1, 1, 2)
dst_pts = m_kpts1.cpu().numpy().reshape(-1, 1, 2)
M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 20 )
matchesMask = mask.ravel().astype(bool)


# Get inlier matches
inlier_matches = matches[matchesMask]

# Get inlier matched keypoints
inlier_m_kpts0 = kpts0[inlier_matches[..., 0]]
inlier_m_kpts1 = kpts1[inlier_matches[..., 1]]

#print number of inliners and outliuers 
print(f"Number of inliers: {inlier_matches.shape[0]}")
print(f"Number of outliers: {matches.shape[0] - inlier_matches.shape[0]}")

# Plotting the inlier matches
axes = viz2d.plot_images([image0, image1])
viz2d.plot_matches(inlier_m_kpts0, inlier_m_kpts1, color="lime", lw=0.2)
viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)

kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"] )
viz2d.plot_images([image0, image1])
viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)
viz2d.plot_matches(inlier_m_kpts0, inlier_m_kpts1, color="lime", lw=0.2)
plt.show()