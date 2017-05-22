import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from skimage.feature import hog
from scipy.ndimage.measurements import label
import math

'''
Draw boxes on an image base on a list of tuple pairs
'''
def draw_boxes(img, boxes, color=(0, 0, 255), thick=6):
    draw_img = np.copy(img)
    for box in boxes:
        draw_img = cv2.rectangle(draw_img, box[0], box[1], color, thick)
    return draw_img


def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features

def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features

'''
Helper function to draw bxes on images
'''
def print_image(img):
    if len(img.shape) == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.show()


def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                     hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True, is_file_path=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []

        if is_file_path:
            # Read in each one by one
            image = mpimg.imread(file)
            if '.jpg' in file:
                image = image.astype(np.float32) / 255
        else:
            image = file

        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
            # Call get_hog_features() with vis=False, feature_vec=True
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
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features

def add_heat(heatmap, detected_windows):
    for box in detected_windows:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1.0
    return heatmap

def apply_threshold(heatmap, threshold=0.5):
    heatmap[heatmap <= threshold] = 0.0
    return heatmap

def draw_labeled_bboxes(img, heatmap, previous_states):
    labels = label(heatmap)
    states = []

    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

        # Skip frames that don't have the good rectangle dimension rations
        width, height = bbox[1][0] - bbox[0][0], bbox[1][1] - bbox[0][1]
        center = (int(bbox[0][0] + width/2), int(bbox[0][1] + height/2))
        area = width * height
        ratio = width/height

        # Skip if the rectangle doesn' meet ratio and area specifications
        if ratio < 1.0 or area < 5000:
            continue

        current_state = {
            'center': center,
            'area': area,
            'ratio': ratio
        }
        states.append(current_state)

        # Find previous centers that seems to be the same vehicle
        states_to_average = []
        # centers_to_average = []
        for previous_state in previous_states:
            best_previous_state, best_previous_distance = find_best_previous_state(current_state, previous_state)
            if best_previous_distance != None and best_previous_distance < 10.0:
                # centers_to_average.append(best_previous_state['center'])
                states_to_average.append(best_previous_state)
        # avg_center = average_centers(center, centers_to_average)
        avg_center, avg_area, avg_ratio = average_states(current_state, states_to_average)

        avg_height = math.sqrt(avg_area/avg_ratio)
        avg_width = avg_area / avg_height

        cv2.circle(img, avg_center, 5, (0, 0, 255.0), -1)

        # Calculate rectangle based on center
        top_left_point = (int(avg_center[0] - avg_width/2), int(avg_center[1] - avg_height/2))
        bottom_right_point = (int(top_left_point[0] + avg_width), int(top_left_point[1] + avg_height))
        cv2.rectangle(img, top_left_point, bottom_right_point, (255.0, 0.0, 250.0), 6)
    return img, states


def average_states(current_state, previous_states):
    previous_states.append(current_state)

    centers = np.array([i['center'] for i in previous_states])
    avg_area = np.average([i['area'] for i in previous_states])
    avg_ratio = np.average([i['ratio'] for i in previous_states])

    x, y = sum(centers[:, 0]) / len(centers), sum(centers[:, 1]) / len(centers)
    return (int(x), int(y)), avg_area, avg_ratio

def average_centers(current_center, p):
    p.append(current_center)
    p = np.array(p)
    x, y = sum(p[:,0])/len(p), sum(p[:,1])/len(p)
    return (int(x), int(y))


def find_best_previous_state(current_state, previous_states):
    # Keep track of the closest distance
    best_distance = None
    best_state = None

    for s in previous_states:
        # Calculate distance
        center1 = current_state['center']
        center2 = s['center']
        distance = math.sqrt(abs(center1[0] - center2[0]) + abs(center1[1] - center2[1]))

        # If closer, update the best state
        if best_distance == None or distance < best_distance:
            best_distance = distance
            best_state = s

    return best_state, best_distance


def process_video(img):
    return img

def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if conv == 'RGB2HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)


def find_cars(img, ystart, ystop, scale, clf, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, hog_channel='ALL'):
    draw_img = np.copy(img)
    img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    if hog_channel == 'ALL':
        ch1 = ctrans_tosearch[:, :, 0]
        ch2 = ctrans_tosearch[:, :, 1]
        ch3 = ctrans_tosearch[:, :, 2]
    else:
        ch1 = ctrans_tosearch[:, :, hog_channel]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 1  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)

    if hog_channel == 'ALL':
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    hot_windows = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step

            # Extract HOG for this patch
            if hog_channel == 'ALL':
                hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            else:
                hog_features = np.hstack((hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(
                np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))

            # test_prediction = clf.decision_function(test_features)
            test_prediction = clf.predict(test_features)
            # if test_prediction > 1.2:
            if test_prediction == True:
                # print(test_prediction)
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                              (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 0, 255), 6)
                hot_window = ((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart))
                hot_windows.append(hot_window)

    return hot_windows, draw_img