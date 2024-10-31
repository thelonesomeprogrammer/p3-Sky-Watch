import csv
import numpy as np
import cv2 

## i,height,width,GPS Latitude,GPS Longitude,GPS Altitude,Roll,Pitch,Yaw
## load the annotations
def load_csv_to_arr(file):
    with open(file) as csv_file:
        anno = []
        csv_reader = csv.reader(csv_file, delimiter=',')
        for k, row in enumerate(csv_reader):
            if k != 0 and file != './SatData/boundaries.txt':
                anno.append(list(map(float,row[0:])))
            elif file == './SatData/boundaries.txt':
                anno.append(list(map(float,row[1:])))
        return anno

def boundary_of_uav_path(anno):
    long = []
    longlat = []
    for k, row in enumerate(anno):
        long.append(row[4])
    lat = []
    for k, row in enumerate(anno):
        lat.append(row[3])
        longlat.append(list(map(float, row[3:5])))
       # longlat.insert([['latitude','longitude']])
    #np.savetxt('skywatch Coords.csv',longlat,delimiter=',')
    coords_min=[min(lat),min(long)]
    coords_max=[max(lat),max(long)]
    coords = [coords_min, coords_max]
    return coords

def xyz_to_coords(boundaries, sat_res,x_y):
    loc_lat_long=[]
    north_bound=boundaries[0][0]
    south_bound=boundaries[1][0]
    east_bound=boundaries[2][0]
    west_bound=boundaries[3][0]
    north_south=abs(north_bound-south_bound)
    east_west=abs(east_bound-west_bound)
    pix_lat_long_eq=[east_west/sat_res[0],north_south/sat_res[1]]
    #important to remember that here our latitude is up-down and long is left to right, because our sat img is oriented towards west.
    loc_lat_long.append(west_bound+(abs(x_y[0]*pix_lat_long_eq[0])))
    loc_lat_long.append(abs(north_bound-(x_y[1]*pix_lat_long_eq[1])))
    print(x_y[0]*pix_lat_long_eq[0])
    return np.flip(loc_lat_long)

def main():
    boundaries = load_csv_to_arr('./SatData/boundaries.txt')
    print (boundaries)
    img = cv2.imread('./SatData/StovringWestOriented.jpg')
    image=cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # finding the resolution 
    wid = image.shape[1] 
    hgt = image.shape[0] 
    sat_res = [wid,hgt]
    data = load_csv_to_arr('./Billed Data/GNSS_data.csv')
    coords = boundary_of_uav_path(data)
    x_y=[wid-100,hgt-100]
    print(x_y)
    loc_long_lat=xyz_to_coords(boundaries,sat_res, x_y)
    print(loc_long_lat)
    return

if __name__ == "__main__":
    main()