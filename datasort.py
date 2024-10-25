import csv


## i,height,width,GPS Latitude,GPS Longitude,GPS Altitude,Roll,Pitch,Yaw
## load the annotations
def load_csv_to_arr(file):
    with open(file) as csv_file:
        anno = []
        csv_reader = csv.reader(csv_file, delimiter=',')
        for k, row in enumerate(csv_reader):
            if k != 0:
                anno.append(list(map(float, row[0:])))
        return anno

def tresh_md_table(unfiltered):
    all = len(unfiltered)
    last = 0
    print("e|e|e")
    print("|--|--|--|")
    for j in range(60):
        rm = 0
        for i in unfiltered:
            if abs(i[6])>j or abs(i[7])>j:
                rm += 1
        print(str(j)+"|"+str(all - rm)+"|"+str(last-rm))
        last = rm


def main():
    data = load_csv_to_arr('./Billed Data/GNSS_data.csv')
    tresh_md_table(data)

if __name__ == "__main__":
    main()
