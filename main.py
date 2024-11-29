import cv2
import torch
import numpy as np
import math
import matplotlib.pyplot as plt


from util.validation import load_csv_to_arr, validation, cal_dist

from submoduls.matchers_extracters import SuperExtract, LightMatch, SiftExtract, BFMatch, FlannMatch
from submoduls.coords import xy_to_coords, load_bonuds
from submoduls.pnp import PnP
from submoduls.rotation import rotate_image
from submoduls.filter import geofilter
from submoduls.tiler import NoLap, MOverLap, AlaaLap
from submoduls.point_selector import ClusterSelector, TileSelector
from submoduls.preproces import MultiProcess





def main(data_path,max_keypoints):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    extractor = SiftExtract(max_keypoints)
    matcher = FlannMatch()
    tiler = NoLap()
    selector = ClusterSelector()
    pre_proces = MultiProcess()
    pnp = PnP.vpair_init()
    pnp_ransac = PnP.vpair_init(True, 1000, 5.0)
    data_set = load_csv_to_arr(data_path+"GNSS_data_test.csv")
    sat_img = cv2.imread(data_path+"SatData/vpair final 2.jpg")
    sat_processed = pre_proces.process(sat_img)
    sat_res = (sat_img.shape[0],sat_img.shape[1])
    sat_features = tiler.tile(sat_img,sat_res,extractor)
    bounds = load_bonuds(data_path+"SatData/boundaries.txt")
    target = []
    pred_usac = []
    pred_ransac = []
    pred_geo = []
    for i in data_set:
        img = cv2.imread(data_path+i[0]+".png")
        img = pre_proces.process(img)
        img, _ = rotate_image(img, -i[6]/math.pi*180)
        target.append([i[0],i[1],i[2]])

        features = extractor.extract(img)
        points = selector.select(matcher,features,img,sat_features,sat_img)


        img_keypoints = np.asarray([[int(features.get_points()[int(t)][0]),int(features.get_points()[int(t)][1])] for t in points[:,2]], dtype=np.float32)

        geo_img_cords, geo_sat_cords = geofilter(img_keypoints, points[:,:2], 5, 4) ## 5 3 
        if len(geo_img_cords) > 4:
            latlong = np.asarray(xy_to_coords(bounds, sat_res, geo_sat_cords), dtype=np.float32)

            cam = pnp.pnp([latlong],[geo_img_cords])[0]

            pred_geo.append([int(i[0]),cam[1][0],cam[0][0]])
            print([int(i[0]),cal_dist([[int(i[0]),cam[1][0],cam[0][0]]],[[i[0],i[1],i[2]]])])


        for huhuhuh in range(3):
            latlong = np.asarray(xy_to_coords(bounds, sat_res, points[:,:2]), dtype=np.float32)
            cam = pnp_ransac.pnp([latlong],[img_keypoints])[0]
            if cam[1][0] < bounds[0] and cam[1][0] > bounds[1] and cam[0][0] < bounds[2] and cam[0][0] > bounds[3]:
                pred_ransac.append([int(i[0]),cam[1][0],cam[0][0]])
                break

        print([int(i[0]),cal_dist([[int(i[0]),cam[1][0],cam[0][0]]],[[i[0],i[1],i[2]]])])


        for huhuhuh in range(3):

            if len(points) < 8:
                break

            F, mask = cv2.findFundamentalMat(img_keypoints, points[:,:2], method=cv2.USAC_MAGSAC, ransacReprojThreshold=2.0, confidence = 0.99, maxIters=4000)

            # Filter points based on the mask
            sat_cords = points[:,:2][mask.ravel() == 1]
            img_cords = img_keypoints[mask.ravel() == 1]

            latlong = np.asarray(xy_to_coords(bounds, sat_res, sat_cords), dtype=np.float32)
            cam = pnp.pnp([latlong],[img_cords])[0]

            if cam[1][0] < bounds[0] and cam[1][0] > bounds[1] and cam[0][0] < bounds[2] and cam[0][0] > bounds[3]:
                pred_usac.append([int(i[0]),cam[1][0],cam[0][0]])
                break


        print([int(i[0]),cal_dist([[int(i[0]),cam[1][0],cam[0][0]]],[[i[0],i[1],i[2]]])])


    
    validation(pred_geo,target)
    print("\n\n--------------------------------------------------------------------------------\n\n")
    validation(pred_ransac,target)
    print("\n\n--------------------------------------------------------------------------------\n\n")
    validation(pred_usac,target)


main("./datasets/vpair/",2048)
        
