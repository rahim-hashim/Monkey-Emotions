import cv2
from tqdm import tqdm
import deeplabcut
from PIL import Image

def check_for_downsample(video_path_list):
	print('Checking frame size...')
	# view first frame in video
	cap = cv2.VideoCapture(video_path_list[0])
	ret, frame = cap.read()
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	img = Image.fromarray(frame)
	width, height = img.size
	print(f'   Pixel width x height: {width}x{height}')
	if width > 640 or height > 640:
		downsample_flag = True
		print('   Flag set to downsample videos to 300x300')
	else:
		downsample_flag = False
		print('No need to downsample videos')
	return downsample_flag

def downsample_videos(video_path_list):
	print('Downsampling videos...')
	downsampled_video_path_list = []
	for video_path in tqdm(video_path_list):
		video_path = deeplabcut.DownSampleVideo(video_path, width=640)
		downsampled_video_path_list.append(video_path)
	print('Done downsampling videos')
	return downsampled_video_path_list