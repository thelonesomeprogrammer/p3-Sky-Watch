import cv2
import torch
import numpy as np
import math
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN

from extracters import SuperExtract, LightMatch, SiftExtract, BFMatch
from validation import load_csv_to_arr
from coords import xy_to_coords, load_bonuds
from validation import validation, cal_dist
from pnp import vpair_test, vpair_test_ransac
from rotation import rotate_image
from geofilter import geofilter





def main(data_path,max_keypoints):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    extractor = SiftExtract()
    matcher = BFMatch()
    data_set = load_csv_to_arr(data_path+"GNSS_data_test.csv")
    sat_img = cv2.imread(data_path+"SatData/vpair final 2.jpg")
    sat_res = (sat_img.shape[0],sat_img.shape[1])
    sat_tiles = []
    sat_features = []
    fraci = int(sat_res[0]/6)
    fracj = int(sat_res[1]/9)
    for i in range(6):
        for j in range(9):
            tile = sat_img[i*fraci:(i+1)*fraci, j*fracj:(j+1)*fracj]
            kp, des = extractor.extract(tile)


            # fig, axs = plt.subplots(2)
            # axs[0].imshow(sat_img[i*fraci:(i+1)*fraci, j*fracj:(j+1)*fracj], zorder=0)
            # axs[0].scatter(kp.cpu()[:,0], kp.cpu()[:,1], zorder=1)
            
            kp[:, 1] += i*fraci
            kp[:, 0] += j*fracj

            # axs[1].imshow(sat_img, zorder=0)
            # axs[1].scatter(kp.cpu()[:,0], kp.cpu()[:,1], zorder=1)
            # plt.show()

            sat_features.append([kp, des])
            
    bounds = load_bonuds(data_path+"SatData/boundaries.txt")
    target = []
    features = []
    matches = []
    pred_usac = []
    pred_ransac = []
    pred_geo = []
    for i in data_set:
        img = cv2.imread(data_path+i[0]+".png")
        img, _ = rotate_image(img, -i[6]/math.pi*180)
        target.append([i[0],i[1],i[2]])

        img_kp, img_des = extractor.extract(img)
        all_keypoints = np.empty((0,3))
        for j in sat_features:
            img_matches = matcher.match(j[0],j[1],np.array([[fraci,fracj]]),img_kp,img_des,[img.shape])
            sat_keypoints = np.asarray([[j[0][int(t)][0], j[0][int(t)][1], int(img_matches[1][index])] for index, t in enumerate(img_matches[0])])
            if len(sat_keypoints) != 0:
                all_keypoints = np.concatenate((all_keypoints, sat_keypoints), axis=0)
        db = DBSCAN(eps=50, min_samples = 5).fit(all_keypoints[:,:2])
        labels = db.labels_


        # Count points in each cluster (excluding noise)
        unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)

        # Sort clusters by size (largest to smallest)
        sorted_indices = np.argsort(-counts)  # Negative for descending order
        sorted_labels = unique_labels[sorted_indices]
        sorted_counts = counts[sorted_indices]

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
        img_keypoints = np.asarray([[int(kp[int(t)][0]),int(kp[int(t)][1])] for t in largest_cluster_points[:,2]], dtype=np.float32)

        geo_img_cords, geo_sat_cords = geofilter(img_keypoints, largest_cluster_points[:,:2], 5, 3)

        latlong = np.asarray(xy_to_coords(bounds, sat_res, geo_sat_cords), dtype=np.float32)
        cam = vpair_test([latlong],[geo_img_cords])[0]

        pred_geo.append([int(i[0]),cam[1][0],cam[0][0]])
        print([int(i[0]),cal_dist([[int(i[0]),cam[1][0],cam[0][0]]],[[i[0],i[1],i[2]]])])


        for huhuhuh in range(3):
            latlong = np.asarray(xy_to_coords(bounds, sat_res, largest_cluster_points[:,:2]), dtype=np.float32)
            cam = vpair_test_ransac([latlong],[img_keypoints])[0]

            if cam[1][0] < bounds[0] and cam[1][0] > bounds[1] and cam[0][0] < bounds[2] and cam[0][0] > bounds[3]:
                break

        pred_ransac.append([int(i[0]),cam[1][0],cam[0][0]])
        print([int(i[0]),cal_dist([[int(i[0]),cam[1][0],cam[0][0]]],[[i[0],i[1],i[2]]])])


        for huhuhuh in range(3):
            F, mask = cv2.findFundamentalMat(img_keypoints, largest_cluster_points[:,:2], method=cv2.USAC_MAGSAC, ransacReprojThreshold=1.0, confidence = 0.95, maxIters=4000)

            # Filter points based on the mask
            sat_cords = largest_cluster_points[:,:2][mask.ravel() == 1]
            img_cords = img_keypoints[mask.ravel() == 1]

            latlong = np.asarray(xy_to_coords(bounds, sat_res, sat_cords), dtype=np.float32)
            cam = vpair_test([latlong],[img_cords])[0]

            if cam[1][0] < bounds[0] and cam[1][0] > bounds[1] and cam[0][0] < bounds[2] and cam[0][0] > bounds[3]:
                break


        pred_usac.append([int(i[0]),cam[1][0],cam[0][0]])
        print([int(i[0]),cal_dist([[int(i[0]),cam[1][0],cam[0][0]]],[[i[0],i[1],i[2]]])])
    print(pred)

    validation(pred_geo,target)
    print("\n\n--------------------------------------------------------------------------------\n\n")
    validation(pred_ransac,target)
    print("\n\n--------------------------------------------------------------------------------\n\n")
    validation(pred_usac,target)


main("./datasets/vpair/",2048)
        
