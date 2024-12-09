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
        if v[1] == 0 and v[2] == 0:
            dist_list.append((v[0],-1))
            continue
        lat1 = math.radians(v[1]) ## convert from degrees to radians 
        lon1 = math.radians(v[2]) ## convert from degrees to radians
        lat2 = math.radians(Target[i][1]) ## convert from degrees to radians
        lon2 = math.radians(Target[i][2]) ## convert from degrees to radians
        dlat = lat2 - lat1 ## find the latitude distance in radiance 
        dlon = lon2 - lon1 ## find the longitude distance in radiance 
        a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2 ## Haversine formula  
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)) ## Haversine formula

        Dist = 6371000 * c ## scale distance to earth size 
        dist_list.append((v[0],Dist))
    return dist_list

            
# Pred: [[id, x, y]]
def validation(pred, targets,file):
    pred_list = np.array(sorted(np.asarray(pred), key=lambda x: x[0])) ## sort the predictions to reduce time to find matching target
    dlist = np.asanyarray(cal_dist(pred_list, targets))
    dist_list =  np.asarray([g[1] for g in dlist if g[1] != -1]) ## calculate the distance between prediction and target
    successlist = np.where(dist_list <= 50)[0] ## take the data that is within our 50m delta goal
    successrate = len(successlist) / (len(dist_list) + 0.000001) * 100 ## calculate success rate (how often do we hit within the 50m goal)
    meanerror = dist_list.mean() ## calculate mean error

    print("dist_list: " + str(dlist)) ## print stats
    print("success rate: " + str(successrate)) ## print stats
    print("median: " + str(np.median(dist_list)))
    print("mean error: " + str(meanerror)) ## print stats

    with open(file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(dlist)


if __name__ == "__main__":
    data = load_csv_to_arr('../datasets/SkyWatchData/GNSS_data.csv')
    fake_Pred = [data[2][0], data[1][1], data[1][2]]
    validation(fake_Pred, data)

#hvor mange procent af vores predictions/bereninger er under 50m (succesrate)
#hvor ofte gætter vi (avg_hit)

