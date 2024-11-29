import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

class ClusterSelector:
    def __init__(self, plot = False):
        self.plot = plot

    def select(self, matcher, img_features, img, sat_features , sat_img):
        all_keypoints = np.empty((0,3))
        for j in sat_features:
            img_matches = matcher.match(j,img_features)
            sat_keypoints = np.asarray([[j.get_points()[int(t)][0], j.get_points()[int(t)][1], int(img_matches[1][index])] for index, t in enumerate(img_matches[0])])
            if len(sat_keypoints) != 0:
                all_keypoints = np.concatenate((all_keypoints, sat_keypoints), axis=0)
        db = DBSCAN(eps=50, min_samples = 5).fit(all_keypoints[:,:2])
        labels = db.labels_


        # Count points in each cluster (excluding noise)
        unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)

        
        if self.plot:
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

        if len(counts) == 0:
            print("no cluster")
            return all_keypoints


        largest_cluster_label = unique_labels[np.argmax(counts)]  # Label of the largest cluster
        largest_cluster_points = all_keypoints[labels == largest_cluster_label]  # Points in the largest cluster

        return largest_cluster_points

class TileSelector:
    def __init__(self, plot = False):
        self.plot = plot

    def select(self, matcher, img_features, img, sat_features , sat_img):
        tile_keypoints = []
        counts = []
        for j in sat_features:
            img_matches = matcher.match(j,img_features)
            sat_keypoints = np.asarray([[j.get_points()[int(t)][0], j.get_points()[int(t)][1], int(img_matches[1][index])] for index, t in enumerate(img_matches[0])])
            tile_keypoints.append(sat_keypoints)
            counts.append(len(sat_keypoints))


            # Sort clusters by size (largest to smallest)
            sorted_indices = np.argsort(-counts)  # Negative for descending order
            sorted_tiles = np.asarray(tile_keypoints[sorted_indices])

        if self.plot:

            plt.figure(figsize=(10, 8))
            colors = plt.get_cmap("tab10", 5)  # Limit to 5 colors


            for nr in range(5):  # Plot only top 5 clusters
                cluster_points = sorted_tiles[nr]
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=[colors(nr)], label=f"Cluster {nr}", zorder=2)

            # Plot noise (if any)
            noise_points = sorted_tiles[5:]
            if len(noise_points) > 0:
                plt.scatter(noise_points[:, 0], noise_points[:, 1], c='black', label="Noise", zorder=1)

            plt.title("Top 5 Tiles")
            plt.xlabel("X Coordinate")
            plt.ylabel("Y Coordinate")
            plt.legend(loc="upper right")
            plt.imshow(sat_img, zorder=0)
            plt.show()
        

        return sorted_tiles[0]



