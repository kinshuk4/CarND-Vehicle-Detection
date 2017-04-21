import cv2
import numpy as np
import matplotlib.image as mpimg

from skimage.feature import hog
from sklearn.model_selection import GridSearchCV
# import src.p5constants as p5c

from tqdm import *

RGB_COLOR_SPACE = "RGB"
YCrCb_COLOR_SPACE = "YCrCb"
SPATIAL_SIZE = (32, 32)

RESIZE_H = 64
RESIZE_W = 64
RESIZE_TUPLE = (RESIZE_W, RESIZE_H)

BINS_RANGE = (0, 255)
HIST_BINS = 32
ORIENT = 8
PIXELS_PER_CELL = 8
CELL_PER_BLOCK = 2
DEFAULT_COLOR_SPACE = 'YCrCb'
DEFAULT_HOG_CHANNEL = 'ALL'


def get_hog_features(image, orient=ORIENT, pix_per_cell=PIXELS_PER_CELL, cell_per_block=CELL_PER_BLOCK, vis=False,
                     feature_vec=True):
    if vis:
        features, hog_image = hog(image, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    else:
        features = hog(image, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features


def bin_spatial(image, size=SPATIAL_SIZE):
    features = cv2.resize(image, size).ravel()
    return features


def color_hist(image, nbins=HIST_BINS, bins_range=BINS_RANGE):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(image[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(image[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(image[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    return hist_features


def extract_features_from_image(image, color_space=DEFAULT_COLOR_SPACE, spatial_size=SPATIAL_SIZE, hist_bins=HIST_BINS,
                                orient=ORIENT, pix_per_cell=PIXELS_PER_CELL,
                                cell_per_block=CELL_PER_BLOCK, hog_channel=DEFAULT_HOG_CHANNEL, spatial_feat=True,
                                hist_feat=True, hog_feat=True):
    image_features = []

    feature_image = convert_color(np.copy(image), 'HSV')
    feature_image[:, :, 2] += np.random.randint(0, 50)
    feature_image = cv2.cvtColor(feature_image, cv2.COLOR_HSV2BGR)
    feature_image = convert_color(np.copy(image), color_space)

    if spatial_feat:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        image_features.append(spatial_features)

    if hist_feat:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        image_features.append(hist_features)

    if hog_feat:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        image_features.append(hog_features)

    return np.concatenate(image_features)


def extract_features_from_file_list(file_list):
    # Create a list to append feature vectors to
    features = []

    # Iterate through the list of image files
    for image_file in tqdm(file_list):
        resize_h, resize_w = 64, 64
        image = cv2.resize(cv2.imread(image_file), (resize_w, resize_h))

        # compute the features of this particular image, then append to the list
        file_features = extract_features_from_image(image)
        features.append(file_features)

    return features


def convert_color(image, dest_color_space=DEFAULT_COLOR_SPACE):
    if dest_color_space == 'YCrCb':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    elif dest_color_space == 'YUV':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    elif dest_color_space == 'LUV':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
    elif dest_color_space == 'HLS':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    elif dest_color_space == 'HSV':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif dest_color_space == 'grayscale':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def find_vehicles(image, y_start, y_stop, scale, svc, feature_scaler, color_space=DEFAULT_COLOR_SPACE,
                  spatial_size=SPATIAL_SIZE, hist_bins=HIST_BINS, orient=ORIENT, pix_per_cell=PIXELS_PER_CELL,
                  cell_per_block=CELL_PER_BLOCK):
    hot_windows = []

    resize_h = RESIZE_H
    resize_w = RESIZE_W

    draw_img = np.copy(image)
    image_crop = draw_img[y_start:y_stop, :, :]
    image_crop = convert_color(image_crop, dest_color_space=color_space)

    if scale != 1:
        imshape = image_crop.shape
        image_crop = cv2.resize(image_crop, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = image_crop[:, :, 0]
    ch2 = image_crop[:, :, 1]
    ch3 = image_crop[:, :, 2]

    # Define blocks and steps as above
    n_x_blocks = (ch1.shape[1] // pix_per_cell) - 1
    n_y_blocks = (ch1.shape[0] // pix_per_cell) - 1

    # 64 was the original sampling rate, with 8 cells and 8 pix per cell
    window = 64
    n_blocks_per_window = (window // pix_per_cell) - 1
    cells_per_step = 4  # Instead of overlap, define how many cells to step
    n_x_steps = (n_x_blocks - n_blocks_per_window) // cells_per_step
    n_y_steps = (n_y_blocks - n_blocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(n_x_steps):
        for yb in range(n_y_steps):
            y_pos = yb * cells_per_step
            x_pos = xb * cells_per_step

            # Extract HOG for this patch
            hog_feat1 = hog1[y_pos:y_pos + n_blocks_per_window, x_pos:x_pos + n_blocks_per_window].ravel()
            hog_feat2 = hog2[y_pos:y_pos + n_blocks_per_window, x_pos:x_pos + n_blocks_per_window].ravel()
            hog_feat3 = hog3[y_pos:y_pos + n_blocks_per_window, x_pos:x_pos + n_blocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            x_left = x_pos * pix_per_cell
            y_top = y_pos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(image_crop[y_top:y_top + window, x_left:x_left + window], (resize_w, resize_h))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = feature_scaler.transform(
                np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))

            test_prediction = svc.decision_function(test_features)

            if test_prediction > 0.2 and test_prediction <= 1:
                xbox_left = np.int(x_left * scale)
                ytop_draw = np.int(y_top * scale)
                win_draw = np.int(window * scale)
                tl_corner_draw = (xbox_left, ytop_draw + y_start)
                br_corner_draw = (xbox_left + win_draw, ytop_draw + win_draw + y_start)

                cv2.rectangle(draw_img, tl_corner_draw, br_corner_draw, (0, 0, 255), 6)

                hot_windows.append((tl_corner_draw, br_corner_draw))

    return hot_windows


def main():
    pass


if __name__ == '__main__':
    main()
