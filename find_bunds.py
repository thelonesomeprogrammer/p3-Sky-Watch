



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


def main():
    data = load_csv_to_arr('./Billed Data/GNSS_data.csv')
    coords = boundary_of_uav_path(data)
    print(bunds)


if __name__ == "__main__":
    main()
