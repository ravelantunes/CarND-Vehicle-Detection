import glob
import utils
import parameters as p
import numpy as np
import time
import pickle
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

import argparse
parser = argparse.ArgumentParser(description='Remote Driving')
parser.add_argument(
    'output',
    type=str,
    default='output.mp4',
    help='Path of output video'
)
args = parser.parse_args()
output_model_path = args.output


# import csv
# lines = []
# csv_file_name = 'labels_crowdai.csv'
# with open('./'+csv_file_name) as csvfile:
#     reader = csv.reader(csvfile)
#     for line in reader:
#         lines.append(line)
# lines = lines[1:]

# Load training dataset
car_images = glob.glob('training_data/*/*/*.png')
cars = []
notcars = []
for image in car_images:
    if 'not_cars' in image:
        notcars.append(image)
    else:
        cars.append(image)

# Load training dataset
car_images = glob.glob('udacity_cleaned/*/*.jpg')
for image in car_images:
    if 'not_cars' in image:
        notcars.append(image)
    else:
        cars.append(image)

print('Loaded {} cars and {} not cars.'.format(len(cars), len(notcars)))

# Traing a single record to get feature matrix size
sample_feature_matrix = utils.extract_features([cars[0]], color_space=p.color_space,
                        spatial_size=p.spatial_size, hist_bins=p.hist_bins,
                        orient=p.orient, pix_per_cell=p.pix_per_cell,
                        cell_per_block=p.cell_per_block,
                        hog_channel=p.hog_channel, spatial_feat=p.spatial_feat,
                        hist_feat=p.hist_feat, hog_feat=p.hog_feat)
print('Feature matrix size: {}'.format(len(sample_feature_matrix[0])))

car_features = utils.extract_features(cars, color_space=p.color_space,
                        spatial_size=p.spatial_size, hist_bins=p.hist_bins,
                        orient=p.orient, pix_per_cell=p.pix_per_cell,
                        cell_per_block=p.cell_per_block,
                        hog_channel=p.hog_channel, spatial_feat=p.spatial_feat,
                        hist_feat=p.hist_feat, hog_feat=p.hog_feat)
notcar_features = utils.extract_features(notcars, color_space=p.color_space,
                        spatial_size=p.spatial_size, hist_bins=p.hist_bins,
                        orient=p.orient, pix_per_cell=p.pix_per_cell,
                        cell_per_block=p.cell_per_block,
                        hog_channel=p.hog_channel, spatial_feat=p.spatial_feat,
                        hist_feat=p.hist_feat, hog_feat=p.hog_feat)



print('Starting to train model')

X = np.vstack((car_features, notcar_features)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.3, random_state=rand_state)

print('Using:',p.orient,'orientations',p.pix_per_cell,
    'pixels per cell and', p.cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))

# Use a linear SVC
# svc = LinearSVC()
from sklearn import svm
clf = svm.SVC()

model_parameters_map = {
    'clf': clf,
    'parameters': p
}

print('Saving pickle')
with open('model-only.pickle', 'wb') as f:
    pickle.dump(clf, f)
    f.close()

with open('with-new-data.pickle', 'wb') as f:
    pickle.dump(model_parameters_map, f)
    f.close()

# Check the training time for the SVC
t = time.time()
clf.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')


# Check the score of the SVC
print('Test Accuracy of SVC = ', round(clf.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
