import numpy as np





def geofilter(img_points, map_points, bin_count, x):
    img_midt = np.mean(img_points, axis=0)
    map_midt = np.mean(map_points, axis=0)

    img_dist = np.linalg.norm(img_points - img_midt, axis=1)
    map_dist = np.linalg.norm(map_points - map_midt, axis=1)

    scalars = np.divide(map_dist, img_dist, out=np.zeros_like(map_dist, dtype=float), where=img_dist!=0)

    min_val = scalars.min()
    max_val = scalars.max()

    
    bins = np.logspace(np.log10(min_val), np.log10(max_val), bin_count + 1)

    bin_indices = np.digitize(scalars, bins)

    unique_bins, counts = np.unique(bin_indices, return_counts=True)

    sorted_indices = np.argsort(counts)[::-1]
    most_common_bins = unique_bins[sorted_indices[:x]]

    img_points = img_points[np.isin(bin_indices, most_common_bins)]
    map_points = map_points[np.isin(bin_indices, most_common_bins)]

    return img_points, map_points

    


