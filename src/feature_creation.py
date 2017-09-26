import cv2
from skimage.feature import hog
import numpy as np

COLOR_SPACES = ["YCrCb"]  # ['RGB', "LUV", "HLS", "YCrCb"]


def convert_color(img, color_space='YCrCb', ravel=False):
    if color_space != 'RGB':
        if color_space == 'HSV':
            out_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            out_img = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            out_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            out_img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            out_img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        out_img = np.copy(img)
    if ravel:
        return out_img.ravel()
    else:
        return out_img


# Define a function to return HOG features and visualization
def get_hog_features(img, orient=9, pix_per_cell=8, cell_per_block=2,
                     vis=False, feature_vec=True, transform_sqrt=False):
    # Call with two outputs if vis==True
    if vis:
        features, hog_image = hog(img,
                                  orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=transform_sqrt,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=transform_sqrt,
                       visualise=vis, feature_vector=feature_vec)
        return features


# Compute individual channel HOG features for the entire image
def get_hog_features_per_channel(img, orient=9, pix_per_cell=8, cell_per_block=2, hog_channel="ALL"):
    if hog_channel == 'ALL':
        hog_features = []
        for channel in range(img.shape[2]):
            hog_features.append(get_hog_features(img[:, :, channel],
                                                 orient, pix_per_cell, cell_per_block,
                                                 vis=False, feature_vec=True))
        hog_features = np.ravel(hog_features)
    else:
        hog_features = get_hog_features(img[:, :, hog_channel], orient, pix_per_cell,
                                        cell_per_block, vis=False, feature_vec=True)
    return hog_features


# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:, :, 0], size).ravel()
    color2 = cv2.resize(img[:, :, 1], size).ravel()
    color3 = cv2.resize(img[:, :, 2], size).ravel()
    return np.hstack((color1, color2, color3))


# Define a function to compute color histogram features
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256), vis=False):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate(
        (channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    if vis:
        # Generating bin centers
        bin_edges = channel1_hist[1]
        bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges) - 1]) / 2
        return hist_features, channel1_hist, channel2_hist, channel3_hist, bin_centers
    else:
        # Return the individual histograms, bin_centers and feature vector
        return hist_features


# Define a function to extract features from a list of images
def extract_features(img, feature_vec=True, color_spaces=COLOR_SPACES,
                     spatial_size=(32, 32),
                     hist_bins=32, bins_range=(0, 1),
                     orient=9, pix_per_cell=8, cell_per_block=2, hog_channel="ALL",
                     spatial_feat=True, hist_feat=True, hog_feat=True):
    img_features = []
    for space in color_spaces:
        feature_image = convert_color(img, color_space=space)
        if spatial_feat:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            img_features.append(spatial_features)
        if hist_feat:
            hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=bins_range)
            img_features.append(hist_features)
        if hog_feat:
            hog_features = get_hog_features_per_channel(
                feature_image, orient, pix_per_cell, cell_per_block, hog_channel)
            img_features.append(hog_features)
    if feature_vec:
        return np.hstack(img_features)
    else:
        return np.array(img_features)


def create_feature_matrix(images, indices_train, indices_test,
                          feature_vec=True, color_spaces=COLOR_SPACES,
                          spatial_size=(32, 32),
                          hist_bins=32, bins_range=(0, 1),
                          orient=9, pix_per_cell=8, cell_per_block=2, hog_channel="ALL",
                          spatial_feat=True, hist_feat=True, hog_feat=True):
    X = np.array([extract_features(img, feature_vec, color_spaces, spatial_size, hist_bins,
                                   bins_range, orient, pix_per_cell, cell_per_block, hog_channel,
                                   spatial_feat, hist_feat, hog_feat)
                  for img in images])
    X_train = X[indices_train]
    X_test = X[indices_test]
    return X_train, X_test
