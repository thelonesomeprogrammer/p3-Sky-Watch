import matplotlib.pyplot as plt

class MOverLap:
    def __init__(self, plot = False):
        self.plot = plot


    def tile(self, sat_img, sat_res, extractor):
        sat_features = []
        fraci = int(sat_res[0]/8) # 6 + 2 for overlap
        fracj = int(sat_res[1]/12) # 9 + 3 for overlap
        for i in range(8):
            for j in range(12):
                if i == 0:
                    i_lower = i*fraci
                else:
                    i_lower = int(i * fraci - fraci / 3)

                if j == 0:
                    j_lower = j * fracj
                else:
                    j_lower = int(j * fracj - fracj / 3)

                tile = sat_img[i_lower:(i+1)*fraci, j_lower:(j+1)*fracj]
                kp, des = extractor.extract(tile)


                if self.plot:
                    fig, axs = plt.subplots(2)
                    axs[0].imshow(sat_img[i*fraci:(i+1)*fraci, j*fracj:(j+1)*fracj], zorder=0)
                    axs[0].scatter(kp.cpu()[:,0], kp.cpu()[:,1], zorder=1)
                
                kp[:, 1] += i*fraci
                kp[:, 0] += j*fracj

                if self.plot:
                    axs[1].imshow(sat_img, zorder=0)
                    axs[1].scatter(kp.cpu()[:,0], kp.cpu()[:,1], zorder=1)
                    plt.show()

                sat_features.append([kp, des])
        return sat_features

class NoLap:
    def __init__(self, plot = False):
        self.plot = plot

    def tile(self, sat_img, sat_res, extractor):
        sat_features = []
        fraci = int(sat_res[0]/6)
        fracj = int(sat_res[1]/9)
        for i in range(6):
            for j in range(9):
                tile = sat_img[i*fraci:(i+1)*fraci, j*fracj:(j+1)*fracj]
                kp, des = extractor.extract(tile)


                if self.plot:
                    fig, axs = plt.subplots(2)
                    axs[0].imshow(sat_img[i*fraci:(i+1)*fraci, j*fracj:(j+1)*fracj], zorder=0)
                    axs[0].scatter(kp.cpu()[:,0], kp.cpu()[:,1], zorder=1)
                
                kp[:, 1] += i*fraci
                kp[:, 0] += j*fracj

                if self.plot:
                    axs[1].imshow(sat_img, zorder=0)
                    axs[1].scatter(kp.cpu()[:,0], kp.cpu()[:,1], zorder=1)
                    plt.show()

                sat_features.append([kp, des])
        return sat_features

class AlaaLap:
    def __init__(self, tile_size=(512, 512), overlap=128, plot = False):
        self.plot = plot
        self.tile_size = tile_size
        self.overlap = overlap

    def tile(self, sat_img, sat_res, extractor):
        sat_features = []
        tiles = self.split_image_into_tile(sat_img, self.tile_size, self.overlap)
        for tile in tiles:
            kp, des = extractor.extract(tile.tile)


            if self.plot:
                fig, axs = plt.subplots(2)
                axs[0].imshow(tile.tile, zorder=0)
                axs[0].scatter(kp.cpu()[:,0], kp.cpu()[:,1], zorder=1)
            
            kp[:, 1] += tile.x
            kp[:, 0] += tile.y

            if self.plot:
                axs[1].imshow(sat_img, zorder=0)
                axs[1].scatter(kp.cpu()[:,0], kp.cpu()[:,1], zorder=1)
                plt.show()

            sat_features.append([kp, des])
        return sat_features


    def split_image_into_tile(image, tile_size, overlap):
        tiles = []
        _, h, w = image.shape  # Assuming image shape is [C, H, W]
        stride_x = tile_size[1] - overlap
        stride_y = tile_size[0] - overlap
        for y in range(0, h - tile_size[0] + 1, stride_y):
            for x in range(0, w - tile_size[1] + 1, stride_x):
                tile = image[:, y:y + tile_size[0], x:x + tile_size[1]]
                if tile.shape[1] == tile_size[0] and tile.shape[2] == tile_size[1]:
                    tiles.append((tile, x, y))  # Store tile with its top-left position
        return tiles
