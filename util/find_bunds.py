import csv
import numpy as np


## file: path to annotation csv
def load_csv_to_arr(file): ## load the csv data (might be replaced with global implantation)
    with open(file) as csv_file:
        anno = []
        csv_reader = csv.reader(csv_file, delimiter=',')
        for k, row in enumerate(csv_reader):
            if k != 0:
                data = list(map(float, row[1:]))
                data.insert(0, row[0])
                anno.append(data)
        return anno



def boundary_of_uav_path(anno):
    long = []
    longlat = []
    for k, row in enumerate(anno):
        long.append(row[2])
    lat = []
    for k, row in enumerate(anno):
        lat.append(row[1])
    coords_min=[min(lat),min(long)]
    coords_max=[max(lat),max(long)]
    coords = [coords_min, coords_max]
    return coords

def lat_long_csv(long,lat):
    longlat = list(zip(lat,long))
    longlat.insert([['latitude','longitude']])
    np.savetxt('skywatch Coords.csv',longlat,delimiter=',')


def main():
    skydata = load_csv_to_arr('../datasets/SkyWatchData/GNSS_data.csv')
    vpairdata = load_csv_to_arr('../datasets/vpair/GNSS_data.csv')
    skybunds = boundary_of_uav_path(skydata)
    vpairbunds = boundary_of_uav_path(vpairdata)
    print("skywatch:" + str(skybunds))
    print("vpair:" + str(vpairbunds))


if __name__ == "__main__":
    main()
