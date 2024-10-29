import math 
import numpy as np
import csv

r= 6371000

# [id, x, y]

def load_csv_to_arr(file):
    with open(file) as csv_file:
        anno = []
        csv_reader = csv.reader(csv_file, delimiter=',')
        for k, row in enumerate(csv_reader):
            if k != 0:
                anno.append(list(map(float, row[0:])))
        return anno

data = load_csv_to_arr('./Billed Data/GNSS_data.csv')

fake_Pred = [[data[0][0],data[0][3],data[0][4]]]
fake_Target = [[data[0][0],data[438][3],data[438][4]]]


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
            a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
            c = 2* math.atan2(math.sqrt(a), math.sqrt(1-a))

            Dist = r*c
            dist_list.append(Dist)
    return dist_list

            

def validation(Pred, Target):
    dist_list = np.asanyarray(cal_dist(Pred, Target))
    succeslist = np.where(dist_list<=50)
    succesrate = len(succeslist)/len(dist_list)*100
    meanerror = dist_list.mean()
    print(succesrate)
    print(meanerror)


validation(fake_Pred,fake_Target)
