import matplotlib.pyplot as plt

class MOverLap:
    def __init__(self, plot=False):
        self.plot = plot

    def tile(self, sat_img, sat_res, extractor):
        sat_features = []
        # Calculate tile steps with overlap
        fraci = int(sat_res[0] / 8)  # Vertical size per tile
        fracj = int(sat_res[1] / 12) # Horizontal size per tile

        for i in range(8):
            for j in range(12):
                if i == 0:
                    i_lower = i * fraci
                else:
                    i_lower = int(i * fraci - fraci / 3)

                if j == 0:
                    j_lower = j * fracj
                else:
                    j_lower = int(j * fracj - fracj / 3)

                tile = sat_img[i_lower:(i + 1) * fraci, j_lower:(j + 1) * fracj]

                # Extract features from the tile
                features = extractor.extract(tile)

                # Plot original tile and features
                if self.plot:
                    fig, axs = plt.subplots(2)
                    axs[0].imshow(sat_img[i*fraci:(i+1)*fraci, j*fracj:(j+1)*fracj], zorder=0)
                    axs[0].scatter(features.get_points()[:,0], features.get_points()[:,1], zorder=1)
                
                features.mv_points(i*fraci, j*fracj)

                # Move feature points to global coordinates
                features.mv_points(j * fracj, i * fraci)

                # Plot full image with adjusted feature coordinates
                if self.plot:
                    axs[1].imshow(sat_img, zorder=0)
                    axs[1].scatter(features.get_points()[:,0], features.get_points()[:,1], zorder=1)
                    plt.show()

                sat_features.append(features)

        return sat_features


class NoLap:
    def __init__(self, plot=False):
        self.plot = plot

    def tile(self, sat_img, sat_res, extractor):
        sat_features = []
        fraci = int(sat_res[0]/6)
        fracj = int(sat_res[1]/9)
        for i in range(6):
            for j in range(9):
                tile = sat_img[i * fraci:(i + 1) * fraci, j * fracj:(j + 1) * fracj]
                features = extractor.extract(tile)

                if self.plot:
                    fig, axs = plt.subplots(2)
                    axs[0].imshow(sat_img[i*fraci:(i+1)*fraci, j*fracj:(j+1)*fracj], zorder=0)
                    axs[0].scatter(features.get_points()[:,0], features.get_points()[:,1], zorder=1)
                
                features.mv_points(i*fraci, j*fracj)

                if self.plot:
                    axs[1].imshow(sat_img, zorder=0)
                    axs[1].scatter(features.get_points()[:,0], features.get_points()[:,1], zorder=1)
                    plt.show()

                sat_features.append(features)

        return sat_features


import matplotlib.pyplot as plt

class AlaaLap:
    def __init__(self, tile_size=(512, 512), overlap=128, plot = False):
        self.plot = plot
        self.tile_size = tile_size
        self.overlap = overlap

    def tile(self, sat_img, sat_res, extractor):
        sat_features = []
        tiles = self.split_image_into_tile(sat_img, sat_res, self.tile_size, self.overlap)
        for tile in tiles:
            features = extractor.extract(tile[0])
            if len(features.get_points()) == 0:
                continue


            if self.plot:
                fig, axs = plt.subplots(2)
                axs[0].imshow(sat_img[tile[2]:tile[2]+self.tile_size[1], tile[1]:tile[1]+self.tile_size[0]], zorder=0)
                axs[0].scatter(features.get_points()[:,0], features.get_points()[:,1], zorder=1)
            
            features.mv_points(tile[2],tile[1])

            if self.plot:
                axs[1].imshow(sat_img, zorder=0)
                axs[1].scatter(features.get_points()[:,0], features.get_points()[:,1], zorder=1)
                plt.show()

            sat_features.append(features)
        return sat_features


    def split_image_into_tile(self, image, sat_res, tile_size, overlap):
        tiles = []
        h, w = sat_res  # Assuming image shape is [C, H, W]
        stride_x = tile_size[1] - overlap
        stride_y = tile_size[0] - overlap
        for y in range(0, h - tile_size[0] + 1, stride_y):
            for x in range(0, w - tile_size[1] + 1, stride_x):
                tile = image[y:y + tile_size[0], x:x + tile_size[1]]
                if tile.shape[0] == tile_size[0] and tile.shape[1] == tile_size[1]:
                    tiles.append((tile, x, y))  # Store tile with its top-left position
        return tiles
