import csv
import numpy as np
import cv2 

# file: file path to bunds file
def load_bonuds(file): ## load the bunds
    with open(file) as csv_file:
        anno = []
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            anno.append(float(row[1]))
        return anno


def xy_to_coords(boundaries, sat_res, feature_coords):
    north_south = abs(boundaries[0] - boundaries[1])
    east_west = abs(boundaries[2] - boundaries[3])
    pix_lat_long_eq = [east_west / sat_res[0], north_south / sat_res[1]]
    loc_lat_long = []
    for i in feature_coords:
        loc_lat_long.append([(boundaries[3] + (abs(i[0] * pix_lat_long_eq[0]))), boundaries[0] - (abs(i[1] * pix_lat_long_eq[1])), 0])
    return loc_lat_long

def main(): ## test function
    fake_featues = [[2000.0,2000.0],[2010.0,2010.0],[2020.0,2010.0],[2034.0,2057.0]] ## fake feature matches 
    boundaries = load_bonuds('./SatData/boundaries.txt') ## load bonds 
    img = cv2.imread('./SatData/StovringWestOriented.jpg') ## load test sat image 
    sat_res = [img.shape[0], img.shape[1] ]# finding the resolution 
    loc_long_lat = xy_to_coords(boundaries, sat_res, fake_featues) ## locations of features
    print (boundaries)
    print(loc_long_lat)
    return

if __name__ == "__main__":
    main()
