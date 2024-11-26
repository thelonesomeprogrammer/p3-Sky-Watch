import math 
import numpy as np
import csv



## file: path to annotation csv
def load_csv_to_arr(file): ## load the csv data (might be replaced with global implantation)
    with open(file) as csv_file:
        anno = []
        csv_reader = csv.reader(csv_file, delimiter=',')
        for k, row in enumerate(csv_reader):
            if k != 0:
                data = list(map(float, row[1:]))
                data.insert(0, row[0].replace(".png",""))
                anno.append(data)
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
def validation(pred, targets):
    pred_list = np.array(sorted(np.asarray(pred), key=lambda x: x[0])) ## sort the predictions to reduce time to find matching target
    startsearch = 0 ## start point for our search for matching target
    last_hit = 0 ## last time we had a hit
    hit_deltas = [] ## distance between hits  
    for v in pred_list: 
        for i in range(startsearch, len(targets)): ## loop over the remaining targets that might match our data 
            if targets[i][0] == v[0]: ## check if this target is the target the fits the current prediction 
                hit_deltas.append(startsearch - last_hit) ## how long since last hit 
                last_hit = startsearch
                break
            else:
                startsearch += 1

    dist_list = np.asanyarray(cal_dist(pred_list, targets)) ## calculate the distance between prediction and target
    successlist = np.where(dist_list <= 50)[0] ## take the data that is within our 50m delta goal
    successrate = len(successlist) / (len(dist_list)+0.000001) * 100 ## calculate success rate (how often do we hit within the 50m goal)
    meanerror = dist_list.mean() ## calculate mean error
    avg_hit = np.asanyarray(hit_deltas).mean() ## calculate mean hit frequency (mean distance in target images between hits) 
    print("dist_list: "+str(dist_list)) ## print stats
    print("freq: "+str(avg_hit)) ## print stats
    print("success rate: "+str(successrate)) ## print stats
    print("mean error: "+str(meanerror)) ## print stats




if __name__ == "__main__":
    data = load_csv_to_arr('../datasets/SkyWatchData/GNSS_data.csv')
    fake_Pred = [data[2][0],data[1][1],data[1][2]]
    validation(fake_Pred,data)

#hvor mange procent af vores predictions/bereninger er under 50m (succesrate)
#hvor ofte gætter vi (avg_hit)

