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
                    fig, axs = plt.subplots(2, figsize=(8,8))
                    # Tile-only view
                    axs[0].imshow(sat_img[i * fraci:(i + 1) * fraci, j * fracj:(j + 1) * fracj], zorder=0)
                    axs[0].scatter(features.points()[:, 0], features.points()[:, 1], c='r', zorder=1)

                # Move feature points to global coordinates
                features.mv_points(j * fracj, i * fraci)

                # Plot full image with adjusted feature coordinates
                if self.plot:
                    axs[1].imshow(sat_img, zorder=0)
                    axs[1].scatter(features.points()[:, 0], features.points()[:, 1], c='r', zorder=1)
                    plt.show()

                sat_features.append(features)

        return sat_features


class NoLap:
    def __init__(self, plot=False):
        self.plot = plot

    def tile(self, sat_img, sat_res, extractor):
        sat_features = []
        fraci = int(sat_res[0] / 9)
        fracj = int(sat_res[1] / 6)

        for i in range(9):
            for j in range(6):
                tile = sat_img[i * fraci:(i + 1) * fraci, j * fracj:(j + 1) * fracj]
                features = extractor.extract(tile)

                if self.plot:
                    fig, axs = plt.subplots(2, figsize=(8,8))
                    axs[0].imshow(sat_img[i * fraci:(i + 1) * fraci, j * fracj:(j + 1) * fracj], zorder=0)
                    axs[0].scatter(features.points()[:, 0], features.points()[:, 1], c='r', zorder=1)

                features.mv_points(j * fracj, i * fraci)

                if self.plot:
                    axs[1].imshow(sat_img, zorder=0)
                    axs[1].scatter(features.points()[:, 0], features.points()[:, 1], c='r', zorder=1)
                    plt.show()

                sat_features.append(features)

        return sat_features


import matplotlib.pyplot as plt

class AlaaLap:
    def __init__(self, tile_size=(1500, 1500), overlap=300, plot=False):
        self.plot = plot
        self.tile_size = tile_size
        self.overlap = overlap

    def tile(self, sat_img, sat_res, extractor):
        sat_features = []
        tiles = self.split_image_into_tile(sat_img, self.tile_size, self.overlap)
        print(f"Number of tiles: {len(tiles)}")  # Debug print
        for tile_image, x, y in tiles:
            print(f"Processing tile at position ({x}, {y})")  # Debug print
            features = extractor.extract(tile_image)
            print(f"Extracted {len(features.points())} features")  # Debug print

            if self.plot:
                fig, axs = plt.subplots(2, figsize=(8, 8))
                axs[0].imshow(tile_image, zorder=0)
                axs[0].scatter(features.points()[:, 0], features.points()[:, 1], c='r', zorder=1)

            features.mv_points(y, x)

            if self.plot:
                axs[1].imshow(sat_img, zorder=0)
                axs[1].scatter(features.points()[:, 0], features.points()[:, 1], c='r', zorder=1)
                plt.show()

            sat_features.append(features)
        return sat_features

    def split_image_into_tile(self, image, tile_size, overlap):
        tiles = []
        stride_y = tile_size[0] - overlap
        stride_x = tile_size[1] - overlap
        for y in range(0, image.shape[0] - tile_size[0] + 1, stride_y):
            for x in range(0, image.shape[1] - tile_size[1] + 1, stride_x):
                tile = image[y:y + tile_size[0], x:x + tile_size[1]]
                if tile.shape[:2] == tile_size:
                    tiles.append((tile, x, y))
        print(f"Generated {len(tiles)} tiles")  # Debug print
        return tiles