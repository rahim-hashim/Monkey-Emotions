import re
import os
import cv2
import pickle
import platform
import deeplabcut
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from collections import defaultdict

def _dlc_check_for_GPU():
	"""Check if GPU is available"""
	try:
		import tensorflow as tf
		try:
			gpus = tf.config.list_physical_devices('GPU')
			if gpus:
				print('Tensorflow GPU found. Enabled for DLC')
			else:
				print('Tensorflow GPU not found. Using CPU for DLC')
				print('  To enable GPU, install CUDA and cuDNN and check Tensorflow installation.')
				print('    https://www.tensorflow.org/install/')
				print('  Using CPU ~20x slower than GPU (~120 iter/sec on 4070 Ti GPU vs ~5 iter/sec on CPU)')
		except:
			print('  Cannot check for GPU. Check Tensorflow installation.')
	except:
		print('Tensorflow not installed. Cannot check for GPU.')


def _dlc_check_for_downsample(video_path_list):
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

def _dlc_downsample_videos(video_path_list):
	print('Downsampling videos...')
	downsampled_video_path_list = []
	for video_path in tqdm(video_path_list):
		video_path = deeplabcut.DownSampleVideo(video_path, width=300)
		downsampled_video_path_list.append(video_path)
	print('Done downsampling videos')
	return downsampled_video_path_list

def select_session(session_root, video_root, date, monkey, correct_only=True):
	print(f'Selecting session: {date}_{monkey}')
	session_dir = os.listdir(session_root)
	"""Select session from list of sessions"""
	session = [f for f in session_dir if f.split('_')[0] == date and f.split('_')[1] == monkey][0]
	with open(os.path.join(session_root, session), 'rb') as f:
		session_dict = pickle.load(f)
		# convert to dataframe
		session_df = session_dict['data_frame']
		# select only correct trials
		if correct_only:
			session_df = session_df[session_df['correct'] == 1]
	video_path = os.path.join(video_root, date + '_' + monkey)
	return session_df, video_path

def get_trial_video_list(video_dir, camera_dict):
	"""Get list of trial videos"""
	print('Checking for video files...')
	print(f'  Video directory: {video_dir}')
	trial_videos = [f.split('.')[0] for f in os.listdir(video_dir) if 'e3' in f and f.endswith('.mp4')]
	print(f'  Number of videos found: {len(trial_videos)}')
	dlc_video_path_dict = defaultdict(list)
	for video in tqdm(trial_videos):
		video_path = os.path.join(video_dir, video+'.mp4')
		# Check if video exists
		if os.path.exists(video_path):
			# Check if video is locked
			try:
				buffer_size = 8
				# Opening file in append mode and read the first 8 characters.
				file_object = open(video_path, 'a', buffer_size)
				if file_object:
					locked = False
			except IOError as message:
				locked = True
			finally:
				if file_object:
					file_object.close()
			if locked:
				print(f'Video locked: {video_path}')
			# Video exists and is not locked
			else:
				# find which camera_dict key matches the video
				camera = [key for key in camera_dict.keys() if key in video][0]
				dlc_video_path_dict[camera].append(video_path)
		else:
			print(f'Video not found: {video_path}')
	for camera in dlc_video_path_dict.keys():
		print(f'  Camera: {camera} | Number of videos: {len(dlc_video_path_dict[camera])}')
	return dlc_video_path_dict

def _dlc_clean_pretrained_project(config_path):
	# delete the first video in the config file to rerun
	print(f'Deleting files for rerun...')
	parent_dir = os.path.dirname(config_path)
	first_video_path = os.path.join(parent_dir, 'videos')
	for video in os.listdir(first_video_path):
		if '1030000' in video:
			os.remove(os.path.join(first_video_path, video))
			print(f'  Deleted: {video}')
	print('Done deleting file.')


def dlc_initialize_project(dlc_video_path_dict, session_obj, camera_dict):

	config_path_dict = {}
	train_config_path_dict = {}

	# see if we are on macOS, if so, symlink videos works
	copy_video_flag = True
	if platform.system() == 'Darwin':
		copy_video_flag = False

	# Create ModelZoo project
	for key in dlc_video_path_dict.keys():
		body_part = camera_dict[key]
		# shortened for Windows because of path length [WinError 206]
		your_name = 'rh'
		directory_name = f'{session_obj.date}_{session_obj.monkey}'
		project_name = f'{body_part}'
		if 'face' in body_part:
			model2use = 'primate_face'
		else:
			model2use = 'full_macaque'
		
		# get list of videos
		video_path_list = dlc_video_path_dict[key]
		# .mp4 or .avi etc.
		videotype = os.path.splitext(video_path_list[0])[-1].lstrip('.')

		# Check if GPU is available
		_dlc_check_for_GPU()

		# Check if videos need to be downsampled
		downsample_flag = _dlc_check_for_downsample(video_path_list)

		if downsample_flag:
				video_path_list = _dlc_downsample_videos(video_path_list)
		print('Initializing Project...')
		print(f'  Project name: {project_name}')
		print(f'  Model: {model2use}')
		print(f'  Initilization Videos: {video_path_list[0:1]}')
		config_path, train_config_path = deeplabcut.create_pretrained_project(
				project_name,
				your_name,
				video_path_list[0:1],
				videotype=videotype,
				model=model2use,
				analyzevideo=True,
				working_directory=directory_name,
				# filtered=False,								# causes error in plot_trajectories if True
				createlabeledvideo=False, 			# causes error in plot_trajectories if True
				copy_videos=copy_video_flag,		# must be true if on PC
		)
		config_path_dict[key] = config_path
		train_config_path_dict[key] = train_config_path

		# Clean pretrained project
		# _dlc_clean_pretrained_project(config_path)

		# Updating the configs within the config.yaml file
		from dlc_primate.dlc_utils import dlc_config_params
		if model2use == 'primate_face':
			print(f'  Editing config file.')
			config_edits = dlc_config_params.config_edits_face
			deeplabcut.auxiliaryfunctions.edit_config(config_path, config_edits)

	return config_path_dict, train_config_path_dict

def dlc_run(config_path_dict, 
						dlc_video_path_dict, 
						start_video=0, 
						end_video=10, 
						videotype='mp4',
						create_labeled_video=False):
	"""Run DLC"""
	for cam in dlc_video_path_dict.keys():
		video_path_list = sorted(dlc_video_path_dict[cam], key=lambda x: int(re.findall(r'(\d+)_{0}'.format(cam), x)[0]))
		if start_video == None:
			start_video = 0
		if end_video == None:
			end_video = len(video_path_list)
		video_list_subset = video_path_list[start_video:end_video]
		try:
			config_path = config_path_dict[cam]
		except:
			print('Config path not found. Likely not made in dlc_initialize_project. Skipping...')
			print(f'  Camera: {cam}')
			continue
		
		# Adding new videos to the config.yaml file
		deeplabcut.add_new_videos(
			config_path, 
			video_list_subset,
			copy_videos=False, 
			coords=None, 
			extract_frames=False
		)

		# Analyze specified videos
		deeplabcut.analyze_videos(
			config_path, 
			video_list_subset, 
			videotype, 
			save_as_csv=True
		)

		# Filter predictions
		deeplabcut.filterpredictions(
				config_path, 
				video_list_subset, 
				videotype=videotype)

		if create_labeled_video:
			# Create labeled videos
			deeplabcut.create_labeled_video(
					config_path, 
					video_list_subset, 
					videotype, 
					draw_skeleton=True, 
					filtered=True,
					trailpoints=5,
			)

		# # Plot trajectories
		# deeplabcut.plot_trajectories(config_path, 
		# 														 video_list_subset, 
		# 														 videotype, 
		# 														 filtered=True)