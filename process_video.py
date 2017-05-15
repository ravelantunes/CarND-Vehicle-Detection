from moviepy.editor import VideoFileClip
import argparse
import utils

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
args = parser.parse_args()

input_video = args.input_video
output_video = args.output_video

process_video = utils.process_video

print(input_video)
print(output_video)

clip1 = VideoFileClip(input_video)
output_clip = clip1.fl_image(process_video)
output_clip.write_videofile(output_video, audio=False)
