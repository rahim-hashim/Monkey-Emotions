import os
import re
import cv2
import sys
import math
import logging
import itertools
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from threading import Thread
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import defaultdict
from IPython.display import clear_output
# Custom classes
from classes import FaceLandmarks
# Custom functions
# from analyses.eye_capture import frame_eye_capture

def notebook_check():
	# see if environment is notebook
	try:
		shell = get_ipython().__class__.__name__
		if shell == 'ZMQInteractiveShell':
			notebook_flag = True
	except NameError:
		notebook_flag = False
	return notebook_flag


def make_video(frames, frame_rate, frame_size, video_name):
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	video = cv2.VideoWriter(video_name, fourcc, frame_rate, frame_size)
	for frame in frames:
		video.write(frame)
	video.release()

def play_video(video_name):
	video = cv2.VideoCapture(video_name)
	while True:
		ret, frame = video.read()
		if not ret:
			break
		cv2.imshow('frame', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	video.release()
	cv2.destroyAllWindows()

def spikeglx_frames(video_folder, 
										video_paths, 
										session_obj, 
										video_dict, 
										target_path,
										trial_num, 
										frame_start_list, 
										frame_end_list, 
										cam, 
										thread_flag=False):
	'''Implements OpenCV to read video file and return a list of frames'''
	print('  video_paths: {}'.format(video_paths))
	frames = []
	video_name = session_obj.monkey + '_' + session_obj.date + '_' + str(trial_num) + '_' + cam + '.mp4'
	# Video path for each trial
	video_path_dest = os.path.join(video_folder, video_name)	
	for v_index, video_path_src in enumerate(video_paths):
		# if more than one video, get frame start and end for each video
		frame_start = frame_start_list[v_index]
		frame_end = frame_end_list[v_index]
		print('  Video Index {} | Frame Start: {} | Frame End: {}'.format(v_index, frame_start, frame_end))
		if frame_start > frame_end:
			print('  WARNING: Frame start is greater than frame end')
			print('    Cam: {}'.format(cam))
			print('    Trial: {}'.format(trial_num))
			print('    Frame Start: {}'.format(frame_start))
			print('    Frame End: {}'.format(frame_end))
			continue

		# Open video file
		cap = cv2.VideoCapture(video_path_src)
		# Get frame count
		frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		# Get frame rate
		frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
		# Get frame size
		frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		frame_size = (frame_width, frame_height)
		# Get duration
		duration = frame_count / frame_rate
		video_dict['Trial'].append(trial_num)
		video_dict['Cam'].append(cam)
		video_dict['Video'].append(video_path_dest)
		video_dict['Frame Start'].append(frame_start)
		video_dict['Frame End'].append(frame_end)
		video_dict['Frame Count'].append(frame_count)
		video_dict['Frame Height'].append(frame_height)
		video_dict['Frame Width'].append(frame_width)
		# Check if the video file was successfully opened
		if not cap.isOpened():
			print("Error opening video file")
		# Read the number of frames specified by spikeglx_cam_framenumbers

		# Delete video if it already exists
		if os.path.exists(video_path_dest):
			print(f'	Deleting existing video file: {video_path_dest}')
			os.remove(video_path_dest)
		# Get start and end frame numbers
		if thread_flag:
			frame_range = range(frame_start, frame_end)
		else:
			frame_range = tqdm(range(frame_start, frame_end), desc=f'Cam: {cam} | Trial: {trial_num} | Frames: {frame_start}-{frame_end}')
			# clear_output(wait=True)
		for frame_num in frame_range:
			cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
			success, frame = cap.read()
			if not success:
					break
			# Add frame to list of frames
			frames.append(frame)
		cap.release()
		cv2.destroyAllWindows()
	# Make video from all frames across video(s) included for trial
	make_video(frames, frame_rate, frame_size, video_path_dest)
	if thread_flag:
		# keep lines aligned across | when printing
		print('  Video complete: Cam: {} | Trial: {:<4} | Frames: {}-{}'.format(cam, trial_num, frame_start, frame_end))


def get_frames(video_path):
	'''Implements OpenCV to read video file and return a list of frames'''
	cap = cv2.VideoCapture(video_path)
	# Get frame count
	frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	# Get frame rate
	frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
	# Get frame size
	frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	frame_size = (frame_width, frame_height)
	print('Video File: ', video_path)
	print('  Frame Count  : ', frame_count)
	print('  Frame Rate   : ', frame_rate)
	print(f'  Frame Size   :  {frame_width} x {frame_height}')
	# Check if the video file was successfully opened
	# if not cap.isOpened():
	# 	print("Error opening video file")
	# frames = []
	# # Read until video is completed
	# while True:
	# 	# Read next frame
	# 	success, frame = cap.read()
	# 	if not success:
	# 			break
	# 	# Add frame to list of frames
	# 	frames.append(frame)
	# cap.release()
	# cv2.destroyAllWindows()
	# return frames, frame_size
	return None, None

# Plot eye position scatter
def eye_scatter(df, v_index, frame_time, ax, eye_x_list, eye_y_list, color_list, session_obj):
	"""
	Takes in a list of eye positions and plots them as a scatter plot

	Parameters
	----------
	df : DataFrame
		DataFrame containing trial information
	v_index : int
		Index of the epoch start and end times
	frame_time : float
		Frame time of the closest frame to the epoch start time
	ax : matplotlib axis
		Axis to plot the eye position scatter

	Returns

	"""
	# number of bins for 2d histogram
	HIST_BINS = 10
	axis_min = -40
	axis_max = 40
	# Get epoch start and end times
	cs_on = df['CS On'].iloc[v_index]
	fix_off = df['Fixation Off'].iloc[v_index]
	trace_start = df['Trace Start'].iloc[v_index]
	outcome_start = df['Outcome Start'].iloc[v_index]
	# set title of plot to be epoch
	if frame_time <= fix_off:
		color = 'red'
		title_str = 'CS On'
	elif frame_time > fix_off and frame_time <= trace_start:
		color='green'
		title_str = 'Fixation'
	elif frame_time > trace_start and frame_time <= outcome_start:
		color='blue'
		title_str = 'Trace'
	else:
		color='purple'
		title_str = 'Outcome'
	color_list.append(color)
	ax.set_title(title_str, color=color)
	# valence 
	valence = df['valence'].iloc[v_index]
	if valence > 0:
		outcome_trigger = df['Reward Trigger'].iloc[v_index]
	else:
		outcome_trigger = df['Airpuff Trigger'].iloc[v_index]
	# plot each eye position and color by epoch
	ax.plot(eye_x_list, eye_y_list,
					color=session_obj.valence_colors[valence])
	ax.scatter(s=10, x=eye_x_list, y=eye_y_list, 
								color=color_list)
	ax.set_xlim(axis_min, axis_max)
	ax.set_ylim(axis_min, axis_max)
	ax.set_facecolor('white')
	ax.grid(True, color='lightgrey')
	axis_range = np.arange(axis_min, axis_max+1, 10)
	ax.set_xticks(axis_range)
	ax.set_xticklabels(axis_range, fontsize=12)
	ax.set_yticks(axis_range)
	ax.set_yticklabels(axis_range, fontsize=12)
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	return ax

def find_epoch_frames(epoch_start_selected, trial_frame_times, epoch_start, epoch_end, v_index):
	'''
	Find the closest frame to the epoch start and end times
	
	Args:
		trial_frame_times (list): list of frame times for a trial
		epoch_start (list): list of epoch start times for a trial
		epoch_end (list): list of epoch end times for a trial
		v_index (int): index of the epoch start and end times

	Returns:
		frame_start_index (int): index of the closest frame to the epoch start time
		frame_end_index (int): index of the closest frame to the epoch end time
		start_frame (float): frame time of the closest frame to the epoch start time
		end_frame (float): frame time of the closest frame to the epoch end time
	'''
	epoch_start_time = epoch_start[v_index]
	epoch_end_time = epoch_end[v_index]
	print('Timings:')
	print(f'  {epoch_start_selected}  Dur   :  {epoch_end_time-epoch_start_time}')
	# Find the closest frame to the trace start and end times
	start_frame = min(trial_frame_times, key=lambda x: abs(x - epoch_start_time))
	end_frame = min(trial_frame_times, key=lambda x: abs(x - epoch_end_time))
	frame_start_index = list(trial_frame_times).index(start_frame)
	frame_end_index = list(trial_frame_times).index(end_frame)
	# To confirm that the closest frame is within the trace start and end times
	if start_frame < epoch_start_time:
		frame_start_index += 1
		start_frame = trial_frame_times[frame_start_index]
	if end_frame > epoch_end_time:
		frame_end_index -= 1
		end_frame = trial_frame_times[frame_end_index]
	print(f'  {epoch_start_selected}  Start :  {epoch_start_time}')
	print(f'  Start Frame     :  {round(start_frame)}')
	print(f'  {epoch_start_selected} End    :  {epoch_end_time}')
	print(f'  End Frame       :  {round(end_frame)}')
	return epoch_start_time, frame_start_index, frame_end_index, start_frame, end_frame

def eye_parsing(df, session_obj):
	pass

def behavior_parsing(df, session_obj, epoch_start_selected):
	print('Behavior')
	print('  Lick Duration: {}'.format(df['lick_duration'].iloc[0]))
	print('  Pupil Zero Duration: {}'.format(df['pupil_raster_window_avg'].iloc[0]))
	print('  Blink Duration: {}'.format(df['blink_duration_window'].iloc[0]))

def wm_video_parsing(df, session_obj, trial_specified=None):
	'''
	Parse video files and extract frames for a specified epoch

	Args:
		df (DataFrame): DataFrame containing trial information
		session_obj (Session): Session object containing session information
		trial_specified (int): trial number to parse

	Returns:
		None
	'''
	date = df['date'].iloc[0]
	# Filter dataframe by trial number
	if trial_specified:
		df = df[df['trial_num'] == trial_specified]
	list_face1_files = df['e3v8360'].tolist()
	list_face2_files = df['e3v83d6'].tolist()
	list_frame_times = df['cam_frames'].tolist()
	# Assign epoch
	epoch_start_selected = 'Fixation'
	epoch_end_selected = 'Outcome'
	start_str = epoch_start_selected + ' Success'
	end_str = epoch_end_selected + ' End'
	epoch_start = df[start_str].tolist()
	epoch_end = df[end_str].tolist()
	# Print camera and video data
	behavior_parsing(df, session_obj, epoch_start_selected)
	# Capture behavioral data
	eye_x = df['eye_x'].tolist()[0]
	eye_y = df['eye_y'].tolist()[0]
	lick_data = df['lick_raster'].tolist()[0]
	pupil_data = df['eye_pupil'].tolist()[0]
	blink_raster = df['blink_raster'].tolist()[0]
	pupil_zero_raster = [1 if x == 0 else 0 for x in pupil_data]
	# Slice behavioral data to epoch
	lick_raster_epoch = lick_data[epoch_start[0]:epoch_end[0]]
	pupil_raster_epoch = pupil_data[epoch_start[0]:epoch_end[0]]
	pupil_zero_raster_epoch = pupil_zero_raster[epoch_start[0]:epoch_end[0]]
	blink_raster_epoch = blink_raster[epoch_start[0]:epoch_end[0]]
	eye_x_epoch = eye_x[epoch_start[0]:epoch_end[0]]
	eye_y_epoch = eye_y[epoch_start[0]:epoch_end[0]]
	# Monkey name
	monkey_name = df['subject'].iloc[0]
	# Open .mp4 file
	for v_index, file_name in enumerate(list_face1_files):
		trial_frame_times = list_frame_times[v_index]
		# Get all frames from the video file
		frames_face1, frame_size_face1 = get_frames(file_name)
		frames_face2, frame_size_face2 = get_frames(list_face2_files[v_index])
		# Find the frames that correspond to the trace start and end times
		epoch_start_time, frame_start_index, frame_end_index, start_frame, end_frame = \
			find_epoch_frames(epoch_start_selected, trial_frame_times, epoch_start, epoch_end, v_index)
		# list of eye positions
		eye_x_list = []
		eye_y_list = []
		color_list = []
		# list of lick times
		lick_list = []
		for f_index, frame in enumerate(frames_face1[frame_start_index:frame_end_index]):
			# make height of column 2 plots smaller
			f, axarr = plt.subplots(2, 2, figsize=(10, 10), gridspec_kw={'height_ratios': [1, 0.75]})
			frame_2 = frames_face2[f_index+frame_start_index]
			frame_time_in_epoch = round(trial_frame_times[f_index+frame_start_index] - epoch_start_time)
			lick_binary = lick_raster_epoch[frame_time_in_epoch]
			pupil_binary = round(pupil_raster_epoch[frame_time_in_epoch])
			blink_binary = round(blink_raster_epoch[frame_time_in_epoch])
			if lick_binary:
					lick_color = (0, 0, 255)
			else:
					lick_color = (255, 255, 255)
			if pupil_binary == 0:
					pupil_color = (255, 0, 0)
			else:
					pupil_color = (255, 255, 255)
			if blink_binary:
					blink_color = (255, 0, 0)
			else:
					blink_color = (255, 255, 255)
			if frame_size_face1 == (320, 240):
					frame = cv2.resize(frame, (640, 480))
			frame = frame_eye_capture(frame, v_index)
			eye_x = eye_x_epoch[frame_time_in_epoch]
			eye_y = eye_y_epoch[frame_time_in_epoch]
			eye_x_list.append(eye_x)
			eye_y_list.append(eye_y)
			# add lick to list
			lick_list.append(lick_binary)
			# Add annotations to frame
			cv2.putText(frame, 'Time: '+str(frame_time_in_epoch), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
			cv2.putText(frame, f'Eye X/Y: ({round(eye_x, 1)},{round(eye_y, 1)})', 
						(10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
			cv2.putText(frame, 'Lick: '+str(lick_binary), 
						(540, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, lick_color, 2)
			cv2.putText(frame, 'Blink: '+str(blink_binary), 
						(540, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, blink_color, 2)
			cv2.putText(frame, 'Pupil: '+str(pupil_binary), 
						(540, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, pupil_color, 2)
			# Plot frame(s)
			axarr[0][0].imshow(frame)
			axarr[1][0].imshow(frame_2)
			frame_time = trial_frame_times[f_index+frame_start_index]
			# Plot eye position scatter
			axarr[1][0] = eye_scatter(df, v_index, frame_time, axarr[0][1], eye_x_list, eye_y_list, color_list, session_obj)
			axarr[1][1].plot(lick_list, color='red')
			plt.tight_layout()
			plt.show()
			if f_index == 100:
				break
		if v_index > 5:
				break

def generate_log_str(trial_num,
										 trial_frame_start,
										 trial_frame_end,
										 trial_frame_count,
										 video_1_path='NA',
										 video_1_frame_start='NA',
										 video_1_frame_end='NA',
										 video_2_path='NA',
										 video_2_frame_start='NA',
										 video_2_frame_end='NA'):
	"""Generate string for log file"""
	log_str = map(str, [trial_num, trial_frame_start, trial_frame_end, trial_frame_count,
										 	video_1_path, video_1_frame_start, video_1_frame_end,
											video_2_path, video_2_frame_start, video_2_frame_end])
	log_str = ','.join(log_str)
	return log_str

def parse_wm_video(spikeglx_obj, 
									 video_file_paths, 
									 session_obj, 
									 trial_num, 
									 video_dict, 
									 target_path, 
									 epoch_start='start', 
									 epoch_end='end', 
									 thread_flag=False):
	"""
	Takes in spikeglx_obj and parses videos for a given trial
	
	Parameters
	----------
	spikeglx_obj : SpikeGLX
		SpikeGLX object
	video_file_paths: dict
		Dictionary of video file paths
	trial_num : int
		Trial number
	epoch_start : str
		Start epoch name ('start' | 'Fixation On')
	epoch_end : str
		End epoch name ('end' | 'Outcome Start')

	Returns
	-------
	None
	"""

	sglx_cam_framenumbers = spikeglx_obj.cam_framenumbers
	video_info = spikeglx_obj.video_info
	
	# Create video folder if it doesn't exist
	if os.path.exists(os.path.join(os.getcwd(), 'video')) == False:
		os.mkdir(os.path.join(os.getcwd(), 'video'))
	video_folder = os.path.join(os.getcwd(), 'video', session_obj.monkey + '_' + session_obj.date)
	if os.path.exists(video_folder) == False:
		os.mkdir(video_folder)

	# Create .csv log file to print parameters used for video parsing
	log_file = os.path.join(video_folder, 'log.csv')
	log = open(log_file, 'a')
	# Write header if log file is empty
	if os.stat(log_file).st_size == 0:
		log.write('Trial, Cam, Trial_Frame_Start, Trial_Frame_End, Trial_Frame_Count, \
							 Video_1_Path, Video_1_Frame_Start, Video_1_Frame_End, \
							 Video_2_Path, Video_2_Frame_Start, Video_2_Frame_End\n')


	# Create video for each trial
	for cam in video_file_paths.keys():
		trial_frame_start = sglx_cam_framenumbers[trial_num][epoch_start]
		trial_frame_end = sglx_cam_framenumbers[trial_num][epoch_end]
		video_paths = []
		video_found_flag = False
		# log parameters used for video parsing
		log_kwargs = {'trial_num': trial_num,
									'trial_frame_start': trial_frame_start,
									'trial_frame_end': trial_frame_end,
									'trial_frame_count': trial_frame_end-trial_frame_start}
		log_str = ''
		
		for v_index, video_path in enumerate(sorted(video_file_paths[cam])):
			video_name = os.path.basename(video_path)
			video_frame_start = video_info[cam][video_name]['index_start']
			video_frame_end = video_info[cam][video_name]['index_end']

			# check if trial frame start is greater than current video
			if trial_frame_start > video_frame_end:
				continue

			# check if trial frame start and end are within one video frames
			elif trial_frame_start >= video_frame_start and trial_frame_end <= video_frame_end:
					video_paths = [video_path]
					frame_start_shifted = [trial_frame_start - video_info[cam][video_name]['index_start']]
					frame_end_shifted = [trial_frame_end - video_info[cam][video_name]['index_start']]
					spikeglx_frames(video_folder, video_paths, session_obj, video_dict, target_path, trial_num, \
													frame_start_shifted, frame_end_shifted, cam, thread_flag)
					video_found_flag = True
					# log parameters used for video parsing
					log_kwargs['video_1_path'] = video_path
					log_kwargs['video_1_frame_start'] = frame_start_shifted[0]
					log_kwargs['video_1_frame_end'] = frame_end_shifted[0]
					break

			# check if trial frame start is in video 1 frames and end is in video 2 frames
			elif trial_frame_start >= video_frame_start and \
						trial_frame_end > video_frame_end and \
						v_index+1 < len(video_file_paths[cam]) and \
						cam in video_file_paths[cam][v_index+1]:
				# check if next video contains the frame
				next_video_name = os.path.basename(video_file_paths[cam][v_index+1])
				next_video_frame_start = video_info[cam][next_video_name]['index_start']
				next_video_frame_end = video_info[cam][next_video_name]['index_end']
				if trial_frame_end >= next_video_frame_start and trial_frame_end <= next_video_frame_end:
					print('  Trial spans two videos. Concatenating frames...')
					# trial frame end is within next video frames
					video_paths = [video_path, video_file_paths[cam][v_index+1]]
					frame_start_shifted_vid1 = trial_frame_start - video_info[cam][video_name]['index_start']
					frame_end_shifted_vid1 = video_info[cam][video_name]['index_end']
					frame_start_shifted_vid2 = 0
					frame_end_shifted_vid2 = trial_frame_end - video_info[cam][next_video_name]['index_start']
					frame_start_shifted_list = [frame_start_shifted_vid1, frame_start_shifted_vid2]
					frame_end_shifted_list = [frame_end_shifted_vid1, frame_end_shifted_vid2]
					spikeglx_frames(video_folder, video_paths, session_obj, video_dict, target_path, trial_num, \
													frame_start_shifted_list, frame_end_shifted_list, cam, thread_flag)
					video_found_flag = True
					# log parameters used for video parsing
					log_kwargs['video_1_path'] = video_paths[0]
					log_kwargs['video_1_frame_start'] = frame_start_shifted_vid1
					log_kwargs['video_1_frame_end'] = frame_end_shifted_vid1
					log_kwargs['video_2_path'] = video_paths[1]
					log_kwargs['video_2_frame_start'] = frame_start_shifted_vid2
					log_kwargs['video_2_frame_end'] = frame_end_shifted_vid2
					break
			else:
				# epoch missing for given trial
				continue
			
		# log parameters used for video parsing
		log_str = generate_log_str(**log_kwargs)
		log.write(log_str + '\n')

		if video_paths == []:
			if np.isnan(trial_frame_start) and np.isnan(trial_frame_end):
				print('Epochs not found for trial {}'.format(trial_num))
				spikeglx_obj.trial_skipping_videos[cam].append(trial_num)
				pass
			else:
				print('Video not found for trial {} although frame epochs found'.format(trial_num))
				print('  Cam: {}'.format(cam))
				print('  Frame Start: {}'.format(trial_frame_start))
				print('  Frame End: {}'.format(trial_frame_end))
				spikeglx_obj.trial_missing_videos[cam].append(trial_num)
			# log parameters used for video parsing
			log_str = generate_log_str(**log_kwargs)
		elif video_found_flag == False:
			print('Something went wrong with video parsing for trial {}'.format(trial_num))
			spikeglx_obj.trial_missing_videos[cam].append(trial_num)


def parse_wm_videos(spikeglx_obj, 
										session_obj,
										trial_start=0, 
										trial_end=100,
										epoch_start='start', 
										epoch_end='end',
										thread_flag=False,
										exclude_camera=None):

	"""Takes in spikeglx_obj and parses videos for a given trial range"""
	# get frames
	video_file_paths = spikeglx_obj.video_file_paths
	print('Included Cameras: {}'.format(video_file_paths.keys()))
	if exclude_camera:
		print('  Excluded Camera(s): {}'.format(exclude_camera))
		for cam in exclude_camera:
			video_file_paths.pop(cam, None)
	video_info = spikeglx_obj.video_info
	trial_subset = list(range(trial_start, trial_end-1)) # last trial usually drops save signal so excluded
	video_dict = defaultdict(list)
	target_path = os.path.join(os.getcwd(), 'video', session_obj.monkey + '_' + session_obj.date)
	print('Parsing Trials for Videos: {} - {}'.format(trial_start, trial_end))
	print('  Epoch Start: {}'.format(epoch_start))
	print('  Epoch End: {}'.format(epoch_end))
	sglx_cam_framenumbers_subset = {k: spikeglx_obj.cam_framenumbers[k] for k in trial_subset}
	# threading for faster parsing
	### USE JOBLIB INSTEAD
	if thread_flag:
		threads = []
		for trial_num in sglx_cam_framenumbers_subset.keys():
			t = Thread(target=parse_wm_video, args=(spikeglx_obj, video_file_paths, session_obj, trial_num, 
																					 	  video_dict, target_path, epoch_start, epoch_end, thread_flag))
			threads.append(t)
			t.start()
		for t in threads:
			t.join()
	else:
		for trial_num in sglx_cam_framenumbers_subset.keys():
			parse_wm_video(spikeglx_obj, video_file_paths, session_obj, trial_num, 
										 video_dict, target_path, epoch_start, epoch_end, thread_flag)
	###
	print('Video Parsing Complete.')
	print('  Missing Videos:')
	for cam in spikeglx_obj.trial_missing_videos.keys():
		print('    Cam: {}'.format(cam))
		print('      Trials: {}'.format(spikeglx_obj.trial_missing_videos[cam]))

def find_wm_videos(df):
	pass