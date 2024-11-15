import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN

from lightglue.utils import rbd, load_image
from lightglue.lightglue import LightGlue
from lightglue import SuperPoint

from validation import load_csv_to_arr
from coords import xy_to_coords, load_bonuds
from validation import validation
from pnp import sky_vores_test



def main(data_path,max_keypoints):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)
    sat_extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)
    data_set = load_csv_to_arr(data_path+"GNSS_data_test.csv")
    sat_img =  load_image(data_path+"SatData/vpair final.jpg")
    img = cv2.imread(data_path+"SatData/vpair final.jpg")
    sat_res = (img.shape[0],img.shape[1])
    sat_tiles = []
    sat_features = []
    fraci = int(sat_res[0]/5)
    fracj = int(sat_res[1]/5)
    for i in range(5):
        for j in range(5):
            tile = sat_img[:, i*fraci:(i+1)*fraci, j*fracj:(j+1)*fracj]
            feature = sat_extractor.extract(tile.unsqueeze(0))
            feature["keypoints"][0][:, 0] += i*fraci
            feature["keypoints"][0][:, 1] += j*fracj
            sat_features.append(feature)
            
    bounds = load_bonuds(data_path+"SatData/boundaries.txt")
    target = []
    imgs = []
    features = []
    matches = []
    pred = []
    for i in data_set:
        img = load_image(data_path+i[0]+".png")
        img.to(device)
        target.append([i[0],i[1],i[2]])
        imgs.append([i[0],img])
        img_features = extractor.extract(img.unsqueeze(0))
        all_keypoints = np.empty((0,3))
        for i in sat_features:
            img_matches = matcher({"image0": i, "image1": img_features})
            print(img_matches)
            sat_keypoints = np.asarray([[i["keypoints"][0][int(t)][0], i["keypoints"][0][int(t)][1],int(t)] for t in img_matches["matches1"][0] if not torch.all(t == -1)])
            all_keypoints = np.concatenate((all_keypoints, sat_keypoints), axis=0)
        db = DBSCAN(eps=500, min_samples = 2).fit(all_keypoints[:,:2])
        labels = db.labels_



        plt.figure(figsize=(10, 8))
        unique_labels = set(labels)
        colors = plt.cm.get_cmap("tab10", len(unique_labels))

        for label in unique_labels:
                # Color for each cluster; noise points are in black
            color = colors(label) if label != -1 else (0, 0, 0, 1)
            plt.scatter(all_keypoints[:,:2][labels == label, 0], all_keypoints[:,:2][labels == label, 1], c=[color], label=f"Cluster {label}" if label != -1 else "Noise")

        plt.title("DBSCAN Clustering of Points")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend(loc="upper right")
        plt.show()


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


main("./datasets/vpair0-100/",2048)
        
