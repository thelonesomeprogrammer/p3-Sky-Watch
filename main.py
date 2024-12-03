import cv2
import torch
import numpy as np
import math
import time
import matplotlib.pyplot as plt


from util.validation import load_csv_to_arr, validation, cal_dist

from submoduls.matchers_extracters import SuperExtract, LightMatch, SiftExtract, BFMatch, FlannMatch
from submoduls.coords import xy_to_coords, load_bonuds
from submoduls.pnp import PnP
from submoduls.rotation import rotate_image
from submoduls.filter import geofilter
from submoduls.tiler import NoLap, MOverLap, AlaaLap
from submoduls.point_selector import ClusterSelector, TileSelector
from submoduls.preproces import MultiProcess, NoProcess





def main(data_path,max_keypoints):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    extractor = SiftExtract(max_keypoints)
    matcher = BFMatch()
    tiler = NoLap()
    selector = ClusterSelector()
    pre_proces = NoProcess()
    pnp = PnP.vpair_init()
    pnp_ransac = PnP.vpair_init(True, 1000, 5.0)
    data_set = load_csv_to_arr(data_path+"GNSS_data_test.csv")
    sat_img = cv2.imread(data_path+"SatData/vpair final 2.jpg")
    sat_processed = pre_proces.process(sat_img)
    sat_res = (sat_img.shape[0],sat_img.shape[1])
    sat_features = tiler.tile(sat_img, sat_res, extractor)
    bounds = load_bonuds(data_path + "SatData/boundaries.txt")
    target = []
    pred_usac = []
    pred_ransac = []
    pred_geo = []
    match_times = []
    extract_times = []
    mfilter_times = []
    megsag_times = []
    for i in data_set:
        img = cv2.imread(data_path + i[0] + ".png")
        img = pre_proces.process(img)
        img, _ = rotate_image(img, -i[6]/math.pi*180)
        target.append([i[0],i[1],i[2]])

        tic1 = time.perf_counter()
        features = extractor.extract(img)
        tic2 = time.perf_counter()
        points = selector.select(matcher,features,img,sat_features,sat_img)
        tic3 = time.perf_counter()

        extract_times.append(tic2-tic1)
        match_times.append(tic3-tic2)



        img_keypoints = np.asarray([[int(features.get_points()[int(t)][0]),int(features.get_points()[int(t)][1])] for t in points[:,2]], dtype=np.float32)
    
        
        tic4 = time.perf_counter()
        geo_img_cords, geo_sat_cords = geofilter(img_keypoints, points[:,:2], 5, 4) ## 5 3 
        tic5 = time.perf_counter()
        mfilter_times.append(tic5-tic4)

        if len(geo_img_cords) > 4:
            latlong = np.asarray(xy_to_coords(bounds, sat_res, geo_sat_cords), dtype=np.float32)

            cam = pnp.pnp([latlong],[geo_img_cords])[0]
        else:
            pred_usac.append([int(i[0]),0,0])

        latlong = np.asarray(xy_to_coords(bounds, sat_res, geo_sat_cords), dtype=np.float32)
        cam = pnp.pnp([latlong],[geo_img_cords])[0]

        if cam[1][0] < bounds[0] and cam[1][0] > bounds[1] and cam[0][0] < bounds[2] and cam[0][0] > bounds[3]:
            pred_geo.append([int(i[0]),cam[1][0],cam[0][0]])
        else:
            pred_geo.append([int(i[0]),0,0])
        print([int(i[0]),cal_dist([[int(i[0]),cam[1][0],cam[0][0]]],[[i[0],i[1],i[2]]])])


        for huhuhuh in range(3):
            latlong = np.asarray(xy_to_coords(bounds, sat_res, points[:,:2]), dtype=np.float32)
            cam = pnp.pnp([latlong],[img_keypoints])[0]
            if cam[1][0] < bounds[0] and cam[1][0] > bounds[1] and cam[0][0] < bounds[2] and cam[0][0] > bounds[3]:
                pred_ransac.append([int(i[0]),cam[1][0],cam[0][0]])
                break
            elif huhuhuh == 2:
                pred_ransac.append([int(i[0]),0,0])


        print([int(i[0]),cal_dist([[int(i[0]),cam[1][0],cam[0][0]]],[[i[0],i[1],i[2]]])])


        for huhuhuh in range(3):

            if len(points) < 4:
                if huhuhuh == 2:
                    pred_usac.append([int(i[0]),0,0])
                    break
                continue
            

                
            tic6 = time.perf_counter()
            F, mask = cv2.findHomography(img_keypoints, points[:,:2], method=cv2.USAC_MAGSAC, ransacReprojThreshold=2.0, confidence = 0.99, maxIters=4000)
            tic7 = time.perf_counter()
            megsag_times.append(tic7-tic6)

            # Filter points based on the mask
            sat_cords = points[:,:2][mask.ravel() == 1]
            img_cords = img_keypoints[mask.ravel() == 1]

            latlong = np.asarray(xy_to_coords(bounds, sat_res, sat_cords), dtype=np.float32)
            if len(img_cords) > 4:
                cam = pnp.pnp([latlong],[img_cords])[0]

                if cam[1][0] < bounds[0] and cam[1][0] > bounds[1] and cam[0][0] < bounds[2] and cam[0][0] > bounds[3]:
                    pred_usac.append([int(i[0]),cam[1][0],cam[0][0]])
                    break
                elif huhuhuh == 2:
                    pred_usac.append([int(i[0]),0,0])



        print([int(i[0]),cal_dist([[int(i[0]),cam[1][0],cam[0][0]]],[[i[0],i[1],i[2]]])])


    
    print("\n\n--------------------------------------------------------------------------------\n\n")
    print("Marrinus filter:\n")
    validation(pred_geo,target)
    print("\n\n--------------------------------------------------------------------------------\n\n")
    print("no filter:\n")
    validation(pred_ransac,target)
    print("\n\n--------------------------------------------------------------------------------\n\n")
    print("Macsac filter:\n")
    validation(pred_usac,target)
    print("\n\n--------------------------------------------------------------------------------\n\n")
    print("Macsac filter:\n")
    print("match: "+str(match_times))
    print("extract: "+str(extract_times))
    print("mfilter : "+str(mfilter_times))
    print("macsac : "+str(megsag_times))


main("./datasets/vpair/",2048)
        
