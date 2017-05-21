from moviepy.editor import VideoFileClip
import argparse
import utils
import numpy as np
import pickle

parser = argparse.ArgumentParser(description='Remote Driving')
parser.add_argument(
    'input_video',
    type=str,
    help='Path to input video'
)
parser.add_argument(
    'output_video',
    type=str,
    default='output.mp4',
    help='Path of output video'
)
parser.add_argument(
    'model',
    type=str,
    help='Path of model to be used'
)

args = parser.parse_args()
input_video = args.input_video
output_video = args.output_video
model_path = args.model

# Open model
with open(model_path, 'rb') as f:
    package = pickle.load(f)
clf = package['clf']
scaler = package['scaler']
p = package['parameters']

previous_heatmap = np.zeros((720,1280), dtype=np.float)
previous_states = []


def process_video(img):
    global previous_heatmap
    global previous_states

    windows1, _ = utils.find_cars(img, 400, 656, 2.0, clf, scaler, p['orient'], p['pix_per_cell'], p['cell_per_block'],
                                  p['spatial_size'], p['hist_bins'])
    hot_windows = windows1
    windows2, _ = utils.find_cars(img, 350, 550, 1.2, clf, scaler, p['orient'], p['pix_per_cell'], p['cell_per_block'],
                                  p['spatial_size'], p['hist_bins'])
    # hot_windows = windows1 + windows2
    # windows3, _ = utils.find_cars(img, 350, 500, 0.8, clf, scaler, p['orient'], p['pix_per_cell'], p['cell_per_block'],
    #                               p['spatial_size'], p['hist_bins'])
    # hot_windows = windows1 + windows2 + windows3
    # hot_windows = windows1 + windows2


    heatmap = utils.add_heat(previous_heatmap, hot_windows)
    previous_heatmap = heatmap * 0.5

    heatmap = utils.apply_threshold(heatmap, 10)
    img, states = utils.draw_labeled_bboxes(img, heatmap, previous_states)

    # Add new state, and remove last if bigger than 3
    previous_states.append(states)
    number_of_states_to_keep = 10
    if len(previous_states) > number_of_states_to_keep:
        previous_states = previous_states[-number_of_states_to_keep:]

    return img


# output_video = 'test_pipeline.mp4'
clip1 = VideoFileClip(input_video)
output_clip = clip1.fl_image(process_video)  # NOTE: this function expects color images!!
output_clip.write_videofile(output_video, audio=False)
