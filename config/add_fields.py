import math
import numpy as np
import pandas as pd
from textwrap import indent
from pprint import pprint, pformat
from collections import defaultdict, OrderedDict
# Custom modules
from analyses.based_noise_blinks_detection import based_noise_blinks_detection

def add_epoch_times(df, behavioral_code_dict):
	"""
	Adds columns for each epoch time start to session_df

	Args:
		df: session DataFrame
		behavioral_code_dict: dictionary of all MonkeyLogic code mappings
			
	Returns:
		df: session DataFrame now including marker time for each behavioral code
	"""

	behavioral_code_indices = list(behavioral_code_dict.keys())
	behavioral_code_names = list(behavioral_code_dict.values())

	epoch_dict = defaultdict(list)
	for t_index in range(len(df)):
		markers = df['behavioral_code_markers'].iloc[t_index]
		times = df['behavioral_code_times'].iloc[t_index]

		for k_index, key in enumerate(behavioral_code_indices):
			epoch_name = behavioral_code_names[k_index]
			try:
				epoch_marker = markers.index(key)
				epoch_dict[epoch_name].append(int(times[epoch_marker]))
			except:
				epoch_dict[epoch_name].append(np.nan)
	for k_index, key in enumerate(epoch_dict.keys()):
		if key != 'Not assigned':
			df[key] = list(epoch_dict.values())[k_index]
			df[key] = df[key].astype('Int32')
	return df

def valence_assignment(row, stim):
	"""
	Adds column for valence of stimuli which includes
	the intensity of the stimuli

	Args:
		row: session DataFrame row
			
	Returns:
		valence: scalar value of valence of stimuli [1, 0.5, 0, -0.5, -1]
	"""
	valence = 0
	if stim == 0:
		stim_label = ''
	else:
		stim_label = f'_{stim}'
	# neutral
	if row[f'reward_mag{stim_label}'] == 0 and row[f'airpuff_mag{stim_label}'] == 0:
		valence = 0
	# airpuff 
	elif row[f'reward_mag{stim_label}'] == 0:
		valence = -1 * row[f'airpuff_mag{stim_label}']
	# reward 
	else:
		valence = row[f'reward_mag{stim_label}']
	return valence

def valence_not_chosen(row, stim):
	# create column for valence not chosen
	v1, v2, v_chosen = row['valence_1'], row['valence_2'], row['valence']
	# error trial
	if v_chosen != v1 and v_chosen != v2:
		v_not_chosen = 0
	elif v1 == v_chosen:
		v_not_chosen = v2
	elif v2 == v_chosen:
		v_not_chosen = v1
	return v_not_chosen

def trace_to_raster(trace_data, threshold):
	"""
	trace_to_raster converts the lick data to a binary
	probability of licking

	Args:
		trace_data: raw lick trace data from MonkeyLogic
		threshold: manual threshold placed to count a lick (default: 1)
			
	Returns:
		raster_data: rasterized data for each ms window
	"""

	raster_data = []
	for l_index, trace_bin in enumerate(trace_data):
		if trace_bin >= threshold or trace_bin <= (-1*threshold):
			#raster_dict[l_index].append(1)
			raster_data.append(1)
		else:
			#raster_dict[l_index].append(0)
			raster_data.append(0)
	return raster_data

def lick_window(trial, lick_threshold):
	"""
	Generates an array for each trial of rasterized lick data
			
		Args:
			trial: row in session_df DataFrame
			
		Returns:
			lick_raster: rasterized lick data for each ms window
	"""
	
	lick_data = trial['lick'].tolist()
	lick_raster = trace_to_raster(lick_data, lick_threshold)
	#lick_window_list = lick_dict.values()
	#lick_window_list_flat = [item for sublist in lick_window_list for item in sublist]
	#lick_mean = np.mean(lick_window_list_flat)
	return lick_raster

def DEM_window(trial):
	"""
	Generates an array for each trial of rasterized blink data
			
		Args:
			trial: row in session_df DataFrame
			
		Returns:
			DEM_raster: rasterized defensive eye movement (DEM) data for each ms window
	"""

	# EyeLink (x,y) values for eyes offscreen
	BLINK_SIGNAL = 10

	eye_x = trial['eye_x'].tolist()
	eye_y = trial['eye_y'].tolist()

	eye_x_abs = [abs(x) for x in eye_x]
	eye_y_abs = [abs(y) for y in eye_y]

	eye_zipped = list(map(max, zip(eye_x_abs, eye_y_abs)))
	DEM_raster = trace_to_raster(eye_zipped, BLINK_SIGNAL)

	return DEM_raster

def trial_bins(trial):
	"""
	Uses the number of samples of eye data as a proxy for
	the trial bins
			
		Args:
			trial: row in session_df DataFrame
			
		Returns:
			trial_length: number of samples of eye_x data
	"""

	trial_length = len(trial['eye_x'].tolist())
	return trial_length

def trial_in_block(df):
	"""
	Counts the trial number in a block

		Args:
			df: session_df DataFrame
			
		Returns:
			trial_block_count: list of trial block count
	"""
	count = 0
	trial_block_count = []
	for t_index in range(len(df)): # off by 1 because len = max(t_index) + 1
		if t_index == 0:
			pass 											 # skipping the first index corrects the offset
		else:
			if df['block'].iloc[t_index] != df['block'].iloc[t_index-1]:
				count = 0
		trial_block_count.append(count)
		count += 1
	return trial_block_count

def fractal_in_block(df):
	"""
	Counts the number of times the fractal was presented in a block
	and only increments the count if the outcome was received

		Args:
			df: session_df DataFrame
			
		Returns:
			fractal_count: trial presentation number in block
	"""

	fractals = sorted(df['fractal_chosen'].unique())
	zero_counter = np.zeros(len(fractals), dtype=int)
	fractal_count = []
	for trial_index in range(len(df)):
		if df['trial_in_block'].iloc[trial_index] == 0:
			zero_counter = np.zeros(len(fractals), dtype=int)
		fractal = df['fractal_chosen'].iloc[trial_index]
		fractal_index = fractals.index(fractal)
		# only increment for correct (i.e. outcome received) trials
		if df['correct'].iloc[trial_index] == 1:
			zero_counter[fractal_index] += 1
		fractal_count.append(zero_counter[fractal_index])
	return fractal_count

def outcome_back_counter(df):
	df['reward_1_back'] = df.reward_1.shift(1)
	df['reward_2_back'] = df['reward_1_back'].tolist() + df.reward_1.shift(2)	
	df['reward_3_back'] = df['reward_2_back'].tolist() + df.reward_1.shift(3)
	df['reward_4_back'] = df['reward_3_back'].tolist() + df.reward_1.shift(4)
	df['reward_5_back'] = df['reward_4_back'].tolist() + df.reward_1.shift(5)
	df['airpuff_1_back'] = df.airpuff.shift(1)
	df['airpuff_2_back'] = df['airpuff_1_back'].tolist() + df.airpuff.shift(2)	
	df['airpuff_3_back'] = df['airpuff_2_back'].tolist() + df.airpuff.shift(3)
	df['airpuff_4_back'] = df['airpuff_3_back'].tolist() + df.airpuff.shift(4)
	df['airpuff_5_back'] = df['airpuff_4_back'].tolist() + df.airpuff.shift(5)
	return df

def outcome_count_window(trial, session_obj):
	"""
	Generates new columns counting lick, blink, and pupil data for each trial
			
		Args:
			trial: row in session_df DataFrame
			session_obj: Session object
			
		Returns:
			trial: row in session_df DataFrame with new columns
				- lick_count_window: rasterized lick data for lick window
				- blink_count_window: rasterized blink data for blink window
				- pupil_data_window: raw pupil data for blink window
				- pupil_raster_window: rasterized pupil data for blink window
				- pupil_zero_count: binary 1 if pupil is 0 for any timepoint in blink window
	"""
	TRACE_WINDOW_LICK = session_obj.window_lick
	TRACE_WINDOW_BLINK = session_obj.window_blink
	lick_raster = trial['lick_raster']
	DEM_raster = trial['DEM_raster']
	pupil_data = trial['eye_pupil']
	pupil_raster = [1 if x == 0 else 0 for x in pupil_data]
	trace_on_time = trial['Trace Start']
	trace_off_time = trial['Trace End']
	if not pd.isna(trace_off_time):
		lick_data_window = lick_raster[trace_off_time-TRACE_WINDOW_LICK:trace_off_time]
		DEM_data_window = DEM_raster[trace_off_time-TRACE_WINDOW_BLINK:trace_off_time]
		pupil_data_window = pupil_data[trace_off_time-TRACE_WINDOW_BLINK:trace_off_time]
		pupil_zero_raster = [1 if x == 0 else 0 for x in pupil_data_window]
		blink_count = 1 if 1 in pupil_zero_raster else 0	
		# blink detection based on Hershman 2018 work
		blink_dict = based_noise_blinks_detection(pupil_data, sampling_freq=1000)
		blink_onset = blink_dict['blink_onset']
		blink_offset = blink_dict['blink_offset']
		# blink raster is a binary vector with 1s for blinks and 0s for non-blinks
		blink_raster = blink_dict['blink_raster']
		blink_raster_window = blink_raster[trace_off_time-TRACE_WINDOW_BLINK:trace_off_time]
		blink_duration_window = np.mean(blink_raster_window)
	else: # error before 'Trace End'
		lick_data_window = np.nan
		DEM_data_window = np.nan	
		pupil_data_window = np.nan	
		pupil_zero_raster = np.nan	
		blink_count = np.nan
		blink_onset = [np.nan]
		blink_offset = [np.nan]
		blink_raster = [np.nan]
		blink_raster_window = [np.nan]
		blink_duration_window = np.nan
	trial['lick_count_window'] = lick_data_window
	trial['blink_count_window'] = DEM_data_window
	trial['pupil_data_window'] = pupil_data_window
	trial['pupil_raster'] = pupil_raster
	trial['pupil_raster_window'] = pupil_zero_raster
	trial['pupil_raster_window_avg'] = np.mean(pupil_zero_raster)
	trial['pupil_binary_zero'] = blink_count
	trial['blink_onset'] = blink_onset
	trial['blink_offset'] = blink_offset
	trial['blink_raster'] = blink_raster
	trial['blink_raster_window'] = blink_raster_window
	trial['blink_duration_window'] = blink_duration_window
	return trial

def pupil_pre_CS(trial):
	"""
	Generates an array for each trial of pupil data
			
		Args:
			trial: row in session_df DataFrame
			
		Returns:
			pupil_data_window: pupil data for 200ms before CS On
	"""
	pupil_data = trial['eye_pupil']
	cs_on_time = trial['CS On']
	try:
		pupil_data_window = pupil_data[cs_on_time-200:cs_on_time]
	except: # error before 'CS On'
		pupil_data_window = np.nan	
	return pupil_data_window

def lick_in_window(trial):
	lick_count_window = trial['lick_count_window']
	if type(lick_count_window) == float: # np.nan
		return np.nan
	elif 1 in lick_count_window:
		return 1
	else:
		return 0

def blink_in_window(trial):
	blink_count_window = trial['blink_count_window']
	if type(blink_count_window) == float: # np.nan
		return np.nan
	elif 1 in blink_count_window:
		return 1
	else:
		return 0


def lick_duration(trial, trace_window):
	# avg lick values ≠ avg of lick raster in window
	# lick_vals = trial['lick']
	# trace_off_time = trial['Trace End']
	# try:
	# 	lick_window = lick_vals[trace_off_time-trace_window:trace_off_time]
	# 	lick_avg = np.nanmean(lick_window)
	# except:
	# 	lick_avg = np.nan
	lick_in_window = trial['lick_count_window']
	try:
		lick_avg = np.nanmean(lick_in_window)
	except:
		lick_avg = np.nan
	return lick_avg  

def blink_duration_sig(trial, trace_window, blink_signal):

	eye_x, eye_y = trial['eye_x'], trial['eye_y']
	trace_off_time = trial['Trace End']
	try:
		eye_x_window = eye_x[trace_off_time-trace_window:trace_off_time]
		eye_y_window = eye_y[trace_off_time-trace_window:trace_off_time]
		blink_count = [1 if (x,y) in blink_signal else 0 for (x,y) in zip(eye_x_window, eye_y_window)]
		blink_avg = np.nanmean(blink_count)
	except:
		blink_avg = np.nan
	return blink_avg  

def blink_duration_offscreen(trial, trace_window):

	blink_count_window = trial['blink_count_window']
	if type(blink_count_window) == float: # np.nan
		blink_avg = np.nan
	else:
		blink_avg = np.nanmean(blink_count_window)
	return blink_avg

def eye_distance(trial, session_obj):
	"""
	Calculates the total distance that the animal's eyes traveled 
	during the trial

	Args:
		trial: row in session_df DataFrame
		session_obj: session object

	Returns:
		eye_distance: total distance that the animal's eyes traveled
			during the trial
	"""
	trace_window = session_obj.window_blink
	blink_signal = session_obj.blink_signal
	trace_off_time = trial['Trace End']
	try:
		eye_x_window = list(trial['eye_x'][trace_off_time-trace_window:trace_off_time])
		eye_y_window = list(trial['eye_y'][trace_off_time-trace_window:trace_off_time])
		blink_signal_list = []
		for bin, (x,y) in enumerate(zip(eye_x_window, eye_y_window)):
			if x in blink_signal.values() and y in blink_signal.values():
				blink_signal_list.append(bin)
		# remove offscreen eye data (variable signal each day)
		for index in sorted(blink_signal_list, reverse=True):
				del eye_x_window[index]
				del eye_y_window[index]
		dx = np.diff(eye_x_window)
		dy = np.diff(eye_y_window)
		step_size = np.sqrt(dx**2+dy**2)
		cumulative_distance = np.sum(step_size)
	except:
		cumulative_distance = np.nan # trial error before 'Trace End'
	return cumulative_distance

def cam_frame_counter(trial):
  """
  Finds the leading edge of the analog cam_save and cam_sync signals
  and returns when they are both high.

  Parameters
  ----------
  trial : pd.Series
    Trial row from session_df

  Returns
  -------
  frame_timings : list
    List of frame timings in which the cam_save and cam_sync signal is high
  """
  # count the number of times the value goes from <3 to greater than 3
  frame_timings = []
  trial_num = trial['trial_num']
  cam_save = trial['cam_save']
  cam_sync = trial['cam_sync']
  # find the leading edge of the cam_sync and cam_save signal
  frame_timings = [i for i in range(1, len(cam_sync)) if cam_sync[i] > 3 and cam_sync[i-1] < 3 and cam_save[i] > 3]
  # print(trial_num, len(frame_timings), len(trial['cam_sync']), 
  #   'frame rate: {}'.format(round(len(frame_timings)/len(trial['cam_sync']), 3)*1000))
  return frame_timings

def prelim_behavior_analysis(df, session_obj, behavioral_code_dict):
	# total lick rate
	lick_dur_all = round(np.mean(df[df['correct']==1]['lick_duration'].tolist()), 3)
	session_obj.lick_duration['all'] = lick_dur_all
	# total blink rate
	avg_blink_all = round(np.mean(df[df['correct']==1]['blink_duration_offscreen'].tolist()), 3)
	session_obj.blink_duration['all'] = avg_blink_all
	return session_obj

def novel_fractal_exp(row):
	"""
	In experiments with novel fractals, each novel fractal has an
	integer (_fractal_1.png, _fractal_2.png, etc.). This function
	removes the integer from the fractal name and replaces it with
	'_fractal_E' to denote that it is a novel fractal.
	"""
	fractal_chosen = row['fractal_chosen']
	if fractal_chosen[-1].isdigit():
		fractal_chosen = '_fractal_E'
	return fractal_chosen

def add_fields(df, session_obj, behavioral_code_dict):
	print('Adding additional fields to session_df DataFrame...')

	TRACE_WINDOW_LICK = session_obj.window_lick
	TRACE_WINDOW_BLINK = session_obj.window_blink
	eye_blink_signal = session_obj.blink_signal
	BLINK_SIGNAL = [(eye_blink_signal['eye_x_min'], eye_blink_signal['eye_y_min']),
									(eye_blink_signal['eye_x_max'], eye_blink_signal['eye_y_max'])]

	df = add_epoch_times(df, behavioral_code_dict)
	try:
		df['valence'] = df.apply(valence_assignment, stim=0, axis=1)
	except:
		try:
			df['valence'] = df.apply(valence_assignment, stim=1, axis=1)
		except:
			print('   No reward magnitude column found, skipping valence assignment...')
	
	if 'reward_mag_1' in df.columns:
		try:
			df['valence_1'] = df.apply(valence_assignment, stim=1, axis=1)
			df['valence_2'] = df.apply(valence_assignment, stim=2, axis=1)
			# reindex to move 'reinforcement trial' and 'choice_trial' columns after 'condition' key
			df['valence_not_chosen'] = df.apply(valence_not_chosen, stim=1, axis=1)
		except:
			print('   No reward magnitude column found, skipping valence assignment...')
	df['lick_raster'] = df.apply(lick_window, lick_threshold=session_obj.lick_threshold, axis=1)
	df['DEM_raster'] = df.apply(DEM_window, axis=1)
	df['trial_bins'] = df.apply(trial_bins, axis=1)
	df['trial_in_block'] = trial_in_block(df)
	# df = outcome_back_counter(df)
	try:
		df = df.apply(outcome_count_window,
								session_obj=session_obj, 
								axis=1)
	except:
		print('   No blink window column found, skipping blink window...')
		pass

	try:
		df['pupil_pre_CS'] = df.apply(pupil_pre_CS, axis=1)
	except:
		print('   No pupil column found, skipping pupil pre-CS...')
		pass
	try:
		df['lick_in_window'] = df.apply(lick_in_window, axis=1)
	except:
		print('   No lick window column found, skipping lick window...')
		pass
	try:
		df['blink_in_window'] = df.apply(blink_in_window, axis=1)
	except:
		print('   No blink window column found, skipping blink window...')
		pass
	try:
		df['lick_duration'] = df.apply(lick_duration, 
																	trace_window=TRACE_WINDOW_LICK, 
																	axis=1)	
	except:
		print('   No lick duration column found, skipping lick duration...')
		pass
	try:
		df['blink_duration_sig'] = df.apply(blink_duration_sig, 
																	trace_window=TRACE_WINDOW_BLINK,
																	blink_signal=BLINK_SIGNAL, 
																	axis=1)
	except:
		print('   No blink duration column found, skipping blink duration...')
		pass
	try:
		df['blink_duration_offscreen'] = df.apply(blink_duration_offscreen, 
																	trace_window=TRACE_WINDOW_BLINK, 
																	axis=1)
	except:
		print('   No blink duration column found, skipping blink duration...')
		pass

	try:
		df['eye_distance'] = df.apply(eye_distance, 
																session_obj=session_obj, 
																axis=1)
	except:
		print('   No eye distance column found, skipping eye distance...')
		pass

	# only when White Matter camera connected
	try:
		df['cam_frames'] = df.apply(cam_frame_counter, axis=1)
		print('  [\'cam_frames\'] field added.')
	except:
		print('	Failed to add [\'cam_frames\'] field.')
		pass

	print('  {} new fields added.'.format(20))
	try:
		session_obj = prelim_behavior_analysis(df, session_obj, behavioral_code_dict)
	except:
		print('   No behavioral analysis performed...')
		pass

	# clear rows with valence == nan	
	try:
		df = df[df['valence'].notnull()]
		print('	{} rows removed due to nan valence.'.format(len(df[df['valence'].isnull()])))
	except:
		pass

	# experiments with novel stimuli only
	df['fractal_chosen'] = df.apply(novel_fractal_exp, axis=1)

	try:
		df['fractal_count_in_block'] = fractal_in_block(df)
	except:
		print('   No fractal column found, skipping fractal count...')
		pass

	return df, session_obj
