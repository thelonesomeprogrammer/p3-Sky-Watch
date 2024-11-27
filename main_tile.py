import cv2
import torch
import numpy as np
import math
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN

from lightglue.utils import rbd, numpy_image_to_torch
from lightglue.lightglue import LightGlue
from LightGlue1.lightglue import SuperPoint

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
    fraci = int(sat_res[0]/8) # 6 + 2 for overlap
    fracj = int(sat_res[1]/12) # 9 + 3 for overlap
    for i in range(8):
        for j in range(12):
            if i == 0:
                i_lower = i*fraci
            else:
                i_lower = int(i * fraci - fraci / 3)

            if j == 0:
                j_lower = j * fracj
            else:
                j_lower = int(j * fracj - fracj / 3)

            tile = sat_tensor[:, i_lower:(i+1)*fraci, j_lower:(j+1)*fracj]
            feature = sat_extractor.extract(tile.unsqueeze(0).to(device))


            # fig, axs = plt.subplots(2)
            # axs[0].imshow(sat_img[i_lower:(i+1)*fraci, j_lower:(j+1)*fracj], zorder=0)
            # axs[0].scatter(feature["keypoints"].cpu()[0][:,0], feature["keypoints"].cpu()[0][:,1], zorder=1)
            
            feature["keypoints"][0][:, 1] += i_lower
            feature["keypoints"][0][:, 0] += j_lower

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
            if len(sat_keypoints) > len(all_keypoints):
                all_keypoints = sat_keypoints

        img_keypoints = np.asarray([[
            int(img_features["keypoints"][0][int(t)][0]),
            int(img_features["keypoints"][0][int(t)][1]),
            ] for t in all_keypoints[:,2]], dtype=np.float32)

        latlong = np.asarray(xy_to_coords(bounds, sat_res, all_keypoints[:,:2]), dtype=np.float32)
        cam = vpair_test([latlong],[img_keypoints])[0]
        pred.append([int(i[0]),cam[1][0],cam[0][0]])
        print(str([int(i[0]),cam[1][0],cam[0][0]])+str("\n"))
    print(pred)

    validation(pred,target)


main("./datasets/vpair/",2048)
        
