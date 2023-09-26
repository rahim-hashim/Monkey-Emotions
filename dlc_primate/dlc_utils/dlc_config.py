import os
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

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

def get_trial_video_list(session_df, video_dir):
	"""Get list of trial videos"""
	trial_videos = session_df['cam1_trial_name'].tolist()
	video_list = []
	print('Checking for video files...')
	print(f'  Video directory: {video_dir}')
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
				video_list.append(video_path)
		else:
			print(f'Video not found: {video_path}')
	print(f'Number of video files: {len(video_list)}')
	return video_list

def clean_pretrained_project(config_path):
  # delete the first video in the config file to rerun
  print(f'Deleting files for rerun...')
  parent_dir = os.path.dirname(config_path)
  first_video_path = os.path.join(parent_dir, 'videos')
  for video in os.listdir(first_video_path):
    if '1030000' in video:
      os.remove(os.path.join(first_video_path, video))
      print(f'  Deleted: {video}')
  print('Done deleting file.')