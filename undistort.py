import argparse
import glob
import cv2
import numpy as np
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

parser = argparse.ArgumentParser(description='Remote Driving')
parser.add_argument(
    'video',
    type=str,
    help='Path to the video'
)
parser.add_argument(
    'calibration_images',
    type=str,
    help='Path to calibration images folder.'
)
parser.add_argument(
    'output_video',
    type=str,
    default='output.mp4',
    help='Path of output video'
)
args = parser.parse_args()

video_path = args.video
calibration_images = args.calibration_images
output_video = args.output_video

# Get camera calibration parameters
objpoints = []
imgpoints = []

images = glob.glob('./camera_cal/calibration*.jpg')

objp = np.zeros((6 * 9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

for fname in images:
    img = mpimg.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)
        img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
    else:
        print('No grid found for {}'.format(fname))

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print('Processing video')

def process_video(img):
    return cv2.undistort(img, mtx, dist, None, mtx)

lines = []

clip1 = VideoFileClip(video_path)
output_clip = clip1.fl_image(process_video)
output_clip.write_videofile(output_video, audio=False)

print('Finished processing video to {}'.format(output_video))