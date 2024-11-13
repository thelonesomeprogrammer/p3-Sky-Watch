import math 
import numpy as np
import csv
from pnp import sky_deres_test, sky_vores_test

## file: path to annotation csv
def load_csv_to_arr(file): ## load the csv data (might be replaced with global implantation)
    with open(file) as csv_file:
        anno = []
        csv_reader = csv.reader(csv_file, delimiter=',')
        for k, row in enumerate(csv_reader):
            if k != 0:
                anno.append(list(map(float, row[0:])))
        return anno


# Pred: [[id, lat, long]], Target: [[id, lat, long]] 
def cal_dist(Pred, Target): ## calculate distance between our guess and ground truth  
    dist_list=[]
    for i,v in enumerate(Pred):
        lat1 = math.radians(v[1]) ## convert from degrees to radians 
        lon1 = math.radians(v[2]) ## convert from degrees to radians
        lat2 = math.radians(Target[i][1]) ## convert from degrees to radians
        lon2 = math.radians(Target[i][2]) ## convert from degrees to radians
        dlat = lat2 - lat1 ## find the latitude distance in radiance 
        dlon = lon2 - lon1 ## find the longitude distance in radiance 
        a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2 ## Haversine formula  
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)) ## Haversine formula

        Dist = 6371000 * c ## scale distance to earth size 
        dist_list.append(Dist)
    return dist_list

            
# Pred: [[id, x, y]]
def validation(Pred, Targets):
    pred_list = np.array(sorted(np.asarray(Pred), key=lambda x: x[0])) ## sort the predictions to reduce time to find matching target
    startsearch = 0 ## start point for our search for matching target
    last_hit = 0 ## last time we had a hit
    hit_deltas = [] ## distance between hits  
    Target = [] ## the filtered Targets
    for v in pred_list: 
        for i in range(startsearch, len(Targets)): ## loop over the remaining targets that might match our data 
            if Targets[i][0] == v[0]: ## check if this target is the target the fits the current prediction 
                Target.append([data[i][0],data[i][3],data[i][4]]) ## take only the relevant info from the target 
                hit_deltas.append(startsearch - last_hit) ## how long since last hit 
                last_hit = startsearch
                break
            else:
                startsearch += 1

    dist_list = np.asanyarray(cal_dist(pred_list, Target)) ## calculate the distance between prediction and target
    successlist = np.where(dist_list <= 50)[0] ## take the data that is within our 50m delta goal
    successrate = len(successlist) / len(dist_list) * 100 ## calculate success rate (how often do we hit within the 50m goal)
    meanerror = dist_list.mean() ## calculate mean error
    avg_hit = np.asanyarray(hit_deltas).mean() ## calculate mean hit frequency (mean distance in target images between hits) 
    print("dist_list: "+str(dist_list)) ## print stats
    print("freq: "+str(avg_hit)) ## print stats
    print("success rate: "+str(successrate)) ## print stats
    print("mean error: "+str(meanerror)) ## print stats




if __name__ == "__main__":
    data = load_csv_to_arr('./Billed Data/GNSS_data.csv')
    object_points = np.array([
        [
            [9.873987, 56.892982, 0],
            [9.875228, 56.892752, 0],
            [9.874458, 56.892741, 0],
            [9.874862, 56.893140, 0],
        ],[
            [9.872404, 56.890317, 0],
            [9.872010, 56.889679, 0],
            [9.870411, 56.890067, 0],
            [9.871091, 56.889904, 0],
        ]
    ], dtype=np.float32) # lat long alt af punkter
    image_points = np.array([
        [
            [3178, 1711],
            [1700, 331],
            [2840, 806],
            [1812, 1665]
        ],[
            [3467, 520],
            [217, 735],
            [3264, 2055],
            [1643, 1413]
        ]], dtype=np.float32) # x y i billeder 
    targets = [data[360][0],data[389][0]]
    Pred = sky_vores_test(object_points,image_points)
    fake_Pred = []
    for i,v in enumerate(Pred):
        fake_Pred.append([targets[i],v[1][0],v[0][0]])

    validation(fake_Pred,data)

    Pred = sky_deres_test(object_points,image_points)
    fake_Pred = []
    for i,v in enumerate(Pred):
        fake_Pred.append([targets[i],v[1][0],v[0][0]])

    validation(fake_Pred,data)

#hvor mange procent af vores predictions/bereninger er under 50m (succesrate)
#hvor ofte gætter vi (avg_hit)

