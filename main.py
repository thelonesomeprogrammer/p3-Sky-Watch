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
from submoduls.MatchVisualizer import MatchVisualizer
#from submoduls import MatchVisualizer as MV




def test(pathclass,pre_proces,selector,tiler,matcher,extractor,device,csv_pre):
    cv2.setRNGSeed(69)
    pnp = PnP.vpair_init()
    data_set = load_csv_to_arr(pathclass.ground())
    sat_img = cv2.imread(pathclass.sat())
    sat_processed = pre_proces.process(sat_img)
    sat_res = (sat_img.shape[0],sat_img.shape[1])
    sat_features = tiler.tile(sat_processed, sat_res, extractor)
    bounds = load_bonuds(pathclass.bounds())
    target = []
    pred_usac = []
    pred_ransac = []
    pred_geo = []
    for i in data_set[:10]:
        img = cv2.imread(pathclass.parse(i[0]))
        img = pre_proces.process(img)
        img, _ = rotate_image(img, -i[6]/math.pi*180)
        target.append([i[0],i[1],i[2]])

        features = extractor.extract(img)
        points = selector.select(matcher,features,img,sat_features,sat_img)
        visualizer = MatchVisualizer()

        img_keypoints = np.asarray([[int(features.get_points()[int(t)][0]),int(features.get_points()[int(t)][1])] for t in points[:,2]], dtype=np.float32)
        geo_img_cords, geo_sat_cords = geofilter(img_keypoints, points[:,:2], 5, 3) ## 5 3 
        # Apply the geofilter
        geo_img_cords, geo_sat_cords = geofilter(img_keypoints, points[:, :2], 5, 3)

        # Convert geofiltered points to keypoints
        filtered_img_keypoints = [
            cv2.KeyPoint(float(pt[0]), float(pt[1]), 1)
            for pt in geo_img_cords
        ]

        filtered_sat_keypoints = [
            cv2.KeyPoint(float(pt[0]), float(pt[1]), 1)
            for pt in geo_sat_cords
        ]

        # Create matches for visualization
        matches = [
            cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0)
            for i in range(len(geo_img_cords))
        ]

        # Visualize filtered matches
        visualizer.current_matches = matches
        visualizer.visualize_matches(img, sat_img, filtered_img_keypoints, filtered_sat_keypoints, matches)


        if len(geo_img_cords) > 4:
            latlong = np.asarray(xy_to_coords(bounds, sat_res, geo_sat_cords), dtype=np.float32)

            cam = pnp.pnp([latlong],[geo_img_cords])[0]
            if cam[1][0] < bounds[0] and cam[1][0] > bounds[1] and cam[0][0] < bounds[2] and cam[0][0] > bounds[3]:
                pred_geo.append([int(i[0]),cam[1][0],cam[0][0]])
            else:
                pred_geo.append([int(i[0]),0,0])
                
            print([int(i[0]),cal_dist([[int(i[0]),cam[1][0],cam[0][0]]],[[i[0],i[1],i[2]]])])
        else:
            pred_geo.append([int(i[0]),0,0])


        for huhuhuh in range(3):
            if len(points) < 4:
                if huhuhuh == 2:
                    pred_ransac.append([int(i[0]),0,0])
                    break
                continue
            latlong = np.asarray(xy_to_coords(bounds, sat_res, points[:,:2]), dtype=np.float32)
    
            cam = pnp.pnp([latlong],[img_keypoints])[0]

            if cam[1][0] < bounds[0] and cam[1][0] > bounds[1] and cam[0][0] < bounds[2] and cam[0][0] > bounds[3]:
                pred_ransac.append([int(i[0]),cam[1][0],cam[0][0]])
                break
            elif huhuhuh == 2:
                pred_ransac.append([int(i[0]),0,0])
                break


        print([int(i[0]),cal_dist([[int(i[0]),cam[1][0],cam[0][0]]],[[i[0],i[1],i[2]]])])


        for huhuhuh in range(3):

            if len(points) < 4:
                if huhuhuh == 2:
                    pred_usac.append([int(i[0]),0,0])
                    break
                continue
            

                
            F, mask = cv2.findHomography(img_keypoints, points[:,:2], method=cv2.RANSAC, ransacReprojThreshold=5.0, confidence = 0.99, maxIters=4000)

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
                    break
            else:
                pred_usac.append([int(i[0]),0,0])
                break


        print([int(i[0]),cal_dist([[int(i[0]),cam[1][0],cam[0][0]]],[[i[0],i[1],i[2]]])])


    
    print("\n\n--------------------------------------------------------------------------------\n\n")
    print("Marrinus filter:\n")
    validation(pred_geo,target,csv_pre+"mfilter.csv")
    print("\n\n--------------------------------------------------------------------------------\n\n")
    print("no filter:\n")
    validation(pred_ransac,target,csv_pre+"nofilter.csv")
    print("\n\n--------------------------------------------------------------------------------\n\n")
    print("Macsac filter:\n")
    validation(pred_usac,target,csv_pre+"RANfilter.csv")
    print("\n\n--------------------------------------------------------------------------------\n\n")


class ImgParser():
    def __init__(self,data_path,prefix,postfix,sat):
        self.prefix = prefix
        self.postfix = postfix
        self.satpost = sat
        self.data_path = data_path
    def parse(self,id):
        return self.data_path + self.prefix + id + self.postfix
    def sat(self):
        return self.data_path + self.satpost
    def bounds(self):
        return self.data_path + "SatData/boundaries.txt"
    def ground(self):
        return self.data_path + "GNSS_data_test.csv"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    tiler = NoLap()
    selector = ClusterSelector()
    vpair_parse = ImgParser("./datasets/vpair/","",".png","SatData/vpair final 2.jpg")
    sky_parse = ImgParser("./datasets/SkyWatchData/","0",".jpg","SatData/StovringNorthOriented.jpg")

    test(vpair_parse,MultiProcess(),selector,tiler,BFMatch(),SiftExtract(2048),device,"vpair_mul_sift_bf_")
    # test(vpair_parse,NoProcess(),selector,tiler,BFMatch(),SiftExtract(2048),device,"vpair_no_sift_bf_")
    # test(vpair_parse,MultiProcess(),selector,tiler,LightMatch("sift",device),SiftExtract(2048),device,"vpair_mul_sift_light_")
    # test(vpair_parse,NoProcess(),selector,tiler,LightMatch("sift",device),SiftExtract(2048),device,"vpair_no_sift_light_")
    # test(vpair_parse,NoProcess(),selector,tiler,LightMatch("superpoint",device),SuperExtract(2048,device),device,"vpair_no_super_light_")
    # test(vpair_parse,NoProcess(),selector,tiler,BFMatch(),SuperExtract(2048,device),device,"vpair_no_super_bf_")

    # test(sky_parse,MultiProcess(),selector,tiler,BFMatch(),SiftExtract(2048),device,"sky_mul_sift_bf_")
    # test(sky_parse,NoProcess(),selector,tiler,BFMatch(),SiftExtract(2048),device,"sky_no_sift_bf_")
    # test(sky_parse,MultiProcess(),selector,tiler,LightMatch("sift",device),SiftExtract(2048),device,"sky_mul_sift_light_")
    # test(sky_parse,NoProcess(),selector,tiler,LightMatch("sift",device),SiftExtract(2048),device,"sky_no_sift_light_")
    # test(sky_parse,NoProcess(),selector,tiler,LightMatch("superpoint",device),SuperExtract(2048,device),device,"sky_no_super_light_")
    # test(sky_parse,NoProcess(),selector,tiler,BFMatch(),SuperExtract(2048,device),device,"sky_no_super_bf_")


main()
        
