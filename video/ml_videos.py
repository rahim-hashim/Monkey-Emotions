import os
import cv2
import numpy as np
import pandas as pd
from pprint import pprint
from tqdm.auto import tqdm
import shutil
import matplotlib.pyplot as plt
# Custom classes
from classes.Session import Session

# base case, working
def generate_ml_behavior_frames(session_df, 
																session_obj, 
																trial_num, 
																epoch_start, 
																epoch_end, 
																subsample=1):
	monkey_name = session_obj.monkey
	date = session_obj.date
	session_name = f'{monkey_name}_{date}'
	trial_specified = session_df[session_df['trial_num'] == trial_num]
	trial_num_index = trial_num - 1
	print('  Number of Frames: {}'.format(len(trial_specified['cam_frames'].tolist()[0])))
	start_analog_time = 0
	end_analog_time = len(trial_specified['eye_x'].tolist()[0])
	if epoch_start != 'start':
		start_analog_time = trial_specified[epoch_start].tolist()[0]
	if epoch_end != 'end':
		end_analog_time = trial_specified[epoch_end].tolist()[0]
	print(f'  Parsing Analog Data: Epochs {start_analog_time} to {end_analog_time}')
	if subsample != 1:
		print(f'  Subsampling Analog Data: (1000/{subsample}) fps')
	eye_x = trial_specified['eye_x'].tolist()[0][start_analog_time:end_analog_time][::subsample]
	eye_y = trial_specified['eye_y'].tolist()[0][start_analog_time:end_analog_time][::subsample]
	lick = trial_specified['lick'].tolist()[0][start_analog_time:end_analog_time][::subsample]
	
	fig_folder_path = os.path.join(os.getcwd(), 'video', session_name, f'trial_{trial_num_index}')
	os.makedirs(fig_folder_path, exist_ok=True)
	for i in tqdm(range(len(eye_x)), desc=f'Trial {trial_num} frame'):
		# eye position
		plt.figure()
		plt.scatter(eye_x[:i], eye_y[:i], c=np.arange(len(eye_x[:i])), cmap='viridis', s=1)
		plt.colorbar()
		plt.xlim(-40, 40)
		plt.ylim(-40, 40)
		plt.xlabel('Eye X Position')
		plt.ylabel('Eye Y Position')
		plt.title('Trial {}'.format(trial_num))
		plt.savefig(os.path.join(fig_folder_path, "eye_%05d.png" % i), dpi=150)
		plt.close()

		# lick
		plt.figure()
		plt.plot(lick[:i])
		plt.xlabel('Time (ms)')
		plt.ylabel('Voltage (mV)')
		plt.title('Lick')
		plt.xlim(0, len(lick))
		plt.ylim(0, 5)
		plt.savefig(os.path.join(fig_folder_path, "lick_%04d.png" % i), dpi=150)
		plt.close()

def generate_ml_behavior_videos(session_df, 
																session_obj, 
																trial_num, 
																epoch_start, 
																epoch_end, 
																subsample=10, 
																slowdown=1):
	'''
	Given a trial number, generate a video of the eye and lick behavior for that trial.

	Parameters:
		session_df (pd.DataFrame): The DataFrame containing the session data.
		session_obj (Session): The Session object containing the session data.
		trial_num (int): The trial number to generate the video for.
		epoch_start (str): The epoch to start the video from.
		epoch_end (str): The epoch to end the video at.
		subsample (int): To subsample the data by a factor of `subsample` to lower frame rate (default=1).
		slowdown (int): To slow down the video by a factor of `slowdown` (default=1).

	Returns:
		None
	'''

	print('Generating video for trial {}'.format(trial_num))
	generate_ml_behavior_frames(session_df, session_obj, trial_num, epoch_start, epoch_end, subsample)

	monkey_name = session_obj.monkey
	date = session_obj.date
	session_name = f'{monkey_name}_{date}'
	trial_num_index = trial_num - 1
	# Define video output settings
	frame_width = 600
	frame_height = 500

	target_folder_path = os.path.join(os.getcwd(), 'video', session_name)
	source_folder_path = os.path.join(os.getcwd(), 'video', session_name, f'trial_{trial_num_index}')
	print(f'  Saving Frames to {source_folder_path}')
	for beh_measure in ['eye', 'lick']:
		target_video_path = os.path.join(target_folder_path, beh_measure+"_%04d.mp4" % trial_num_index)
		# delete video if it already exists
		if os.path.exists(target_video_path):
			print('Deleting existing video: {}'.format(target_video_path))
			os.remove(target_video_path)
		print('Saving video to: {}'.format(target_video_path))
		# Define the codec and create VideoWriter object
		fourcc = cv2.VideoWriter_fourcc(*'mp4v')
		if slowdown != 1:
			print(f'   Slowing Video by: {slowdown}x')
			target_video_path = target_video_path.replace('.mp4', f'_slow{slowdown}.mp4')
		fps = 1000/(subsample*slowdown)
		print(f'  FPS: {fps}')
		out = cv2.VideoWriter(target_video_path, fourcc, fps, (frame_width, frame_height))
		# Iterate through all the files in the folder
		video_files = sorted(os.listdir(source_folder_path))
		video_files_beh = [file for file in video_files if file.startswith(beh_measure)]
		for filename in tqdm(video_files_beh, desc=f'Trial {trial_num_index} {beh_measure}'):
			# reading each file
			img = cv2.imread(os.path.join(source_folder_path, filename))
			# setting the width, height of the frame
			img = cv2.resize(img, (frame_width, frame_height))
			# writing the extracted images
			out.write(img)
	# delete image folder
	print('Deleting image folder: {}'.format(source_folder_path))
	shutil.rmtree(source_folder_path)