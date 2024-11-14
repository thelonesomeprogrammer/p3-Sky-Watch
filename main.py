import cv2
import torch
import numpy as np

from sklearn.cluster import DBSCAN

from lightglue.utils import rbd, load_image
from lightglue.lightglue import LightGlue
from lightglue import SuperPoint

from validation import load_csv_to_arr
from coords import xy_to_coords, load_bonuds
from validation import validation
from pnp import sky_vores_test



def main(data_path,max_keypoints):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)
    sat_extractor = SuperPoint(max_num_keypoints=max_keypoints*20).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)
    data_set = load_csv_to_arr(data_path+"GNSS_data.csv")
    sat_img =  load_image(data_path+"SatData/StovringNorthOriented.jpg")
    img = cv2.imread(data_path+"SatData/StovringNorthOriented.jpg")
    sat_features = sat_extractor.extract(sat_img.unsqueeze(0))
    bounds = load_bonuds(data_path+"SatData/boundaries.txt")
    sat_res = (img.shape[0],img.shape[1])
    target = []
    imgs = []
    features = []
    matches = []
    pred = []
    for i in data_set[:11]:
        img = load_image(data_path+"0"+i[0]+".jpg")
        img.to(device)
        target.append([i[0],i[1],i[2]])
        imgs.append([i[0],img])
        img_features = extractor.extract(img.unsqueeze(0))
        features.append([i[0],rbd(img_features)])
        img_matches = matcher({"image0": sat_features, "image1": img_features})
        matches.append([i[0],img_matches["matches0"],img_matches["matches1"],
                            img_matches["matching_scores0"],img_matches["matching_scores1"],
                        img_matches["matches"],img_matches["scores"]])
        sat_keypoints = np.asarray([[
            sat_features["keypoints"][0][int(t)][0],
            sat_features["keypoints"][0][int(t)][1],
            int(t)
            ] for t in img_matches["matches0"][0] if not torch.all(t == -1)])
        db = DBSCAN(eps=500, min_samples = 2).fit(sat_keypoints[:,:2])
        labels = db.labels_
        unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)  # Exclude noise (-1)
        try:
            largest_cluster_label = unique_labels[np.argmax(counts)]  # Label of the largest cluster
        except:
            continue
        largest_cluster_points = sat_keypoints[labels == largest_cluster_label]  # Points in the largest cluster

        if len(largest_cluster_points) < 4:
            continue
        img_keypoints = np.asarray([[
            int(img_features["keypoints"][0][int(t)][0]),
            int(img_features["keypoints"][0][int(t)][1]),
            ] for t in largest_cluster_points[:,2]], dtype=np.float32)
        latlong = np.asarray(xy_to_coords(bounds, sat_res, largest_cluster_points[:,:2]), dtype=np.float32)
        cam = sky_vores_test([latlong],[img_keypoints])[0]
        pred.append([int(i[0]),cam[1][0],cam[0][0]])
    print(pred)

    validation(pred,target)


main("./datasets/SkyWatchData/",2048)
        
