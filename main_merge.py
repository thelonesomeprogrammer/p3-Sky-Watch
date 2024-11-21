import cv2
import torch
import numpy as np
import math
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN

from lightglue.utils import rbd, numpy_image_to_torch
from lightglue.lightglue import LightGlue
from lightglue import SuperPoint

from validation import load_csv_to_arr
from coords import xy_to_coords, load_bonuds
from validation import validation
from pnp import vpair_test
from rotation import rotate_image



def main(data_path,max_keypoints):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)
    sat_extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)
    data_set = load_csv_to_arr(data_path+"GNSS_data_test.csv")
    sat_img = cv2.imread(data_path+"SatData/vpair final 2.jpg")
    sat_tensor = numpy_image_to_torch(sat_img)
    sat_res = (sat_img.shape[0],sat_img.shape[1])
    sat_tiles = []
    sat_features = []
    fraci = int(sat_res[0]/7)
    fracj = int(sat_res[1]/7)
    for i in range(7):
        for j in range(7):
            tile = sat_tensor[:, i*fraci:(i+1)*fraci, j*fracj:(j+1)*fracj]
            feature = sat_extractor.extract(tile.unsqueeze(0).to(device))


            # fig, axs = plt.subplots(2)
            # axs[0].imshow(sat_img[i*fraci:(i+1)*fraci, j*fracj:(j+1)*fracj], zorder=0)
            # axs[0].scatter(feature["keypoints"].cpu()[0][:,0], feature["keypoints"].cpu()[0][:,1], zorder=1)
            
            feature["keypoints"][0][:, 1] += i*fraci
            feature["keypoints"][0][:, 0] += j*fracj

            # axs[1].imshow(sat_img, zorder=0)
            # axs[1].scatter(feature["keypoints"].cpu()[0][:,0], feature["keypoints"].cpu()[0][:,1], zorder=1)
            # plt.show()

            sat_features.append(feature)
            
    bounds = load_bonuds(data_path+"SatData/boundaries.txt")
    target = []
    features = []
    matches = []
    pred = []
    for i in data_set:
        img = cv2.imread(data_path+i[0]+".png")
        img, _ = rotate_image(img, -i[6]/math.pi*180)
        img = numpy_image_to_torch(img)
        img.to(device)
        target.append([i[0],i[1],i[2]])

        img_features = extractor.extract(img.unsqueeze(0).to(device))
        all_keypoints = np.empty((0,3))
        for j in sat_features:
            img_matches = matcher({"image0": j, "image1": img_features})
            sat_keypoints = np.asarray([[j["keypoints"][0].cpu()[int(t.cpu())][0], j["keypoints"][0].cpu()[int(t.cpu())][1],int(t.cpu())] for t in img_matches["matches1"][0] if not torch.all(t == -1)])
            if len(sat_keypoints) != 0:
                all_keypoints = np.concatenate((all_keypoints, sat_keypoints), axis=0)
        db = DBSCAN(eps=50, min_samples = 5).fit(all_keypoints[:,:2])
        labels = db.labels_


        # Count points in each cluster (excluding noise)
        unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)

        # Sort clusters by size (largest to smallest)
        sorted_indices = np.argsort(-counts)  # Negative for descending order
        sorted_labels = unique_labels[sorted_indices]

        # Relabel clusters so the largest is "0", next largest is "1", etc.
        new_labels = -np.ones_like(labels)  # Start with noise as -1
        for j, label in enumerate(sorted_labels[:5]):  # Only relabel top 5 clusters
            new_labels[labels == label] = j

        # Plotting
        plt.figure(figsize=(10, 8))
        colors = plt.get_cmap("tab10", 5)  # Limit to 5 colors

        for label in range(5):  # Plot only top 5 clusters
            if label in new_labels:
                cluster_points = all_keypoints[new_labels == label]
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=[colors(label)], label=f"Cluster {label}", zorder=2)

        # Plot noise (if any)
        noise_points = all_keypoints[new_labels == -1]
        if len(noise_points) > 0:
            plt.scatter(noise_points[:, 0], noise_points[:, 1], c='black', label="Noise", zorder=1)

        plt.title("Top 5 Largest Clusters (Relabeled)")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend(loc="upper right")
        plt.imshow(sat_img, zorder=0)
        plt.show()


        largest_cluster_label = unique_labels[np.argmax(counts)]  # Label of the largest cluster
        largest_cluster_points = all_keypoints[labels == largest_cluster_label]  # Points in the largest cluster

        if len(largest_cluster_points) < 4:
            continue
        img_keypoints = np.asarray([[
            int(img_features["keypoints"][0][int(t)][0]),
            int(img_features["keypoints"][0][int(t)][1]),
            ] for t in largest_cluster_points[:,2]], dtype=np.float32)


        F, mask = cv2.findFundamentalMat(img_keypoints, largest_cluster_points[:,:2], method=cv2.FM_RANSAC, ransacReprojThreshold=5.0)

        # Filter points based on the mask
        sat_cords = largest_cluster_points[:,:2][mask.ravel() == 1]
        img_cords = img_keypoints[mask.ravel() == 1]


        latlong = np.asarray(xy_to_coords(bounds, sat_res, sat_cords), dtype=np.float32)
        cam = vpair_test([latlong],[img_cords])[0]
        pred.append([int(i[0]),cam[1][0],cam[0][0]])
    print(pred)

    validation(pred,target)


main("./datasets/vpair/",2048)
        
