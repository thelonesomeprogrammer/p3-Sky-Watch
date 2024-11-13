import math 
import numpy as np
import csv

## file: path to annotation csv
def load_csv_to_arr(file):
    with open(file) as csv_file:
        anno = []
        csv_reader = csv.reader(csv_file, delimiter=',')
        for k, row in enumerate(csv_reader):
            if k != 0:
                anno.append(list(map(float, row[0:])))
        return anno


# Pred: [[id, x, y]], Target: [[id, x, y]] 
def cal_dist(Pred, Target):
    dist_list=[]
    for i,v in enumerate(Pred):
        if v[0] == Target[i][0]:
            lat1 = math.radians(v[1])
            lon1 = math.radians(v[2])
            lat2 = math.radians(Target[i][1])
            lon2 = math.radians(Target[i][2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

            Dist = 6371000 * c
            dist_list.append(Dist)
    return dist_list

            
# Pred: [[id, x, y]]
def validation(Pred, Targets):
    pred_list = np.array(sorted(np.asarray(Pred), key=lambda x: x[0]))
    startsearch = 0
    hit_deltas = []
    last_hit = 0
    Target = []
    for v in pred_list:
        for i in range(startsearch, len(Targets)):
            startsearch += 1
            if Targets[i][0] == v[0]:
                Target.append([data[i][0],data[i][3],data[i][4]])
                hit_deltas.append(startsearch - last_hit - 1)
                last_hit = startsearch - 1
                break

    dist_list = np.asanyarray(cal_dist(Pred, Target))
    succeslist = np.where(dist_list <= 50)[0]
    print(dist_list)
    succesrate = len(succeslist) / len(dist_list) * 100
    meanerror = dist_list.mean()
    avg_hit = np.asanyarray(hit_deltas).mean()
    print(avg_hit)
    print(succesrate)
    print(meanerror)





data = load_csv_to_arr('./Billed Data/GNSS_data.csv')
fake_Pred = [[data[3][0],data[0][3],data[0][4]], [data[6][0],data[3][3],data[3][4]]]
validation(fake_Pred,data)


#hvor mange procent af vores predictions/bereninger er under 50m (succesrate)
#hvor ofte gætter vi (avg_hit)

