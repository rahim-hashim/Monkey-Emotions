# This example imports functions from the DemoReadSGLXData module to read
# analog data and convert it to volts based on the metadata information.
# The metadata file must be present in the same directory as the binary file.
# Works with both imec and nidq analog channels.

import os
import re
import sys
import math
import dill
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
from tkinter import Tk
from tkinter import filedialog
from spike_glx.sglx_lib import readMeta, SampRate, makeMemMapRaw
from spike_glx.sglx_lib import GainCorrectIM, GainCorrectNI, GainCorrectNIChannel

def print_meta(meta, verbose=False):
	"""
	Reads metadata and prints the file name, number of channels,
	file creation time, and file length.

	Parameters
	----------
	meta : dict
		Metadata from the binary file.
	verbose : bool
		If True, prints the metadata keys and values.

	Returns
	-------
	n_channels : int
		Number of channels in the binary file.
	"""
	if verbose:
		for meta_key in meta.keys():
			print(meta_key, meta[meta_key])
	file_name = meta['fileName']
	n_channels = meta['nSavedChans']
	file_created_time = meta['fileCreateTime']
	file_length = meta['fileTimeSecs']
	print(f'File Name: {file_name}')
	print(f'  Number of Channels: {n_channels}')
	print(f'  File Created: {file_created_time}')
	print(f'  File Length: {file_length}')
	return n_channels

def get_spikeglx_path():
	print('Select binary file')
	# Get bin file from user
	root = Tk()         # create the Tkinter widget
	root.withdraw()     # hide the Tkinter root window
	# Windows specific; forces the window to appear in front
	root.attributes("-topmost", True)
	binFullPath = Path(filedialog.askopenfilename(title="Select binary file"))
	root.destroy()   
	
	return binFullPath

def time_to_samples(data_array, sample_rate, data_times, tStart, tEnd):
	'''
	Converts time in milliseconds to samples
	from SpikeGLX data
	'''
	first_sample = int(sample_rate*tStart/1000)
	last_sample = int(sample_rate*tEnd/1000)
	data_array = data_array[first_sample:last_sample]
	data_times = data_times[first_sample:last_sample]
	return data_array, data_times

def load_spikeglx(binFullPath):
	"""
	Locates user-specified binary file and reads the raw
	data into a dictionary. Returns the metadata and the
	dictionary of channels.


	"""
	# Read <file>.meta
	meta = readMeta(binFullPath)

	# Find Number of Channels
	n_channels = print_meta(meta)

	# Initialize Channel Dictionary
	chan_dict = defaultdict(list)

	rawData = makeMemMapRaw(binFullPath, meta)
	for chan in range(int(n_channels)):

		# Assign channel raw data to dictionary
		chan_dict[chan] = rawData[chan]

	return meta, chan_dict

def parse_channels(meta, chan_dict, signal_dict):
	"""
	Parses metadata and channel dictionary to return the sample rate,
	time vector, and dictionary of corrected data from each channel.

	Parameters
	----------
	meta : dict
		Metadata from the binary file.
	chan_dict : dict
		Dictionary of channels. Each channel is a list of
		samples.

	Returns
	-------
	sRate : float
		Sample rate in Hz.
	tDat : numpy array
		Time vector in ms.
	chan_dict_corrected : dict
		Dictionary of channels. Each channel is a list of
		samples. The data has been corrected for gain and
		converted to mV.
	"""
	sRate = SampRate(meta)
	tDat = np.array(0)
	chan_dict_corrected = defaultdict(list)
	print('Number of Channels: ', len(chan_dict.keys()))
	print('Sample Rate: ', sRate)
	for chan in chan_dict.keys():
		rawData = chan_dict[chan]
		chanList = [chan]
		if meta['typeThis'] == 'imec':
				# apply gain correction and convert to uV
				convData = 1e6*GainCorrectIM(rawData, chanList, meta)
		else:
				# apply gain correction and convert to mV
				convData = 1e3*GainCorrectNI(rawData, chanList, meta)
		tDat = np.arange(0, len(convData))
		tDat = 1000*tDat/sRate
		chan_dict_corrected[chan] = convData
		print(f' Channel [{chan}]: {signal_dict[chan]}')
		print(f'  Max Val: {round(max(convData), 3)}')
		print(f'  Min Val: {round(min(convData), 3)}')
	return sRate, tDat, chan_dict_corrected



def plot_channels_raw(spikeglx_obj, meta, chan_dict, signal_dict, tStart=0, tEnd=10):
	"""
	Plot all channels from the channel dictionary in a given time window.

	Parameters
	----------
	meta : dict
		Metadata from the binary file.

	chan_dict : dict
		Dictionary of channels. Each channel is a list of
		samples.

	signal_dict : dict
		Dictionary of channels. Each channel is a list of
		samples. The data has been corrected for gain and
		converted to mV.

	tStart : float
		Start time in seconds.

	tEnd : float
		End time in seconds.
	"""
	try:
		meta = spikeglx_obj.meta
		signal_dict = spikeglx_obj.signal_dict
		chan_dict = {}
		for key in signal_dict.keys():
			chan_dict[key] = spikeglx_obj.chan_dict[key]
	except:
		print('  SpikeGLX object missing attributes (meta, chan_dict, signal_dict)')
		print('  Using input arguments instead.')

	# Calculate Sample Rate
	sRate = SampRate(meta)
	print('Sample Rate: ', sRate)

	firstSamp = int(sRate*tStart)
	lastSamp = int(sRate*tEnd)

	# Initialize Correct Channel Data Dictionary
	chan_dict_corrected = defaultdict(list)

	# Plot all channels
	f, ax = plt.subplots(1, 1, figsize=(10, 2))
	for chan in chan_dict.keys():
		rawData = chan_dict[chan]
		chanList = [chan]
		selectData = rawData[firstSamp:lastSamp+1]
		if meta['typeThis'] == 'imec':
				# apply gain correction and convert to uV
				convData = 1e6*GainCorrectIM(selectData, chanList, meta)
		else:
				# apply gain correction and convert to mV
				convData = 1e3*GainCorrectNI(selectData, chanList, meta)
		# Plot the first of the extracted channels
		tDat = np.arange(firstSamp, lastSamp+1)
		tDat = 1000*tDat/sRate      # plot time axis in msec
		print(f' Channel [{chan}]: {signal_dict[chan]}')
		print(f'  Max Val: {round(max(convData), 3)}')
		print(f'  Min Val: {round(min(convData), 3)}')
		ax.plot(tDat, convData, label=signal_dict[chan])
		ax.set_xlabel('Time (ms)')
		ax.set_ylabel('Voltage (mV)')
		chan_dict_corrected[chan] = convData
	# make small legend outside the plot
	ax.legend(loc='center right', bbox_to_anchor=(1.25, 0.5))
	ax.set_yticks(np.arange(0, 6000, 1000))
	ax.set_ylim(0, 5500)
	plt.title('Raw SpikeGLX NI Channels')
	plt.tight_layout()
	plt.show()

def read_SGLX():
	"""
	Locates user-specified binary file and reads the raw 
	data into a dictionary. Returns the metadata and the
	dictionary of channels.

	Returns
	-------
	meta : dict
		Metadata from the binary file.
	chan_dict : dict
		Dictionary of channels. Each channel is a list of
		samples.
	"""
	binFullPath = get_spikeglx_path()
	print(f'Bin File Path: {binFullPath}')
	meta, chan_dict = load_spikeglx(binFullPath)
	return meta, chan_dict


# align cam_sync
def plot_trial_0(session_df, spikeglx_obj):

	# Get SpikeGLX cam_sync
	sample_rate = spikeglx_obj.sample_rate
	sample_times = spikeglx_obj.sample_times
	spikeglx_cam_times = spikeglx_obj.trial_times
	spikeglx_cam_frames = spikeglx_obj.cam_frames
	
	f, axarr = plt.subplots(3, 1, figsize=(20, 8))
	cam_sync = session_df['cam_sync'].iloc[0].tolist()

	# Plot Monkeylogic trial 0 cam save/sync
	ml_time_start = session_df['cam_frames'].iloc[0][0] - 20
	ml_time_end = ml_time_start+50
	x_axis = np.arange(ml_time_start, ml_time_end, 1)
	axarr[0].plot(x_axis, session_df['cam_sync'].iloc[0][ml_time_start:ml_time_end], label='ml_cam_sync')
	axarr[0].plot(x_axis, session_df['cam_save'].iloc[0][ml_time_start:ml_time_end], label='ml_analog')
	# axarr[0].axvline(x=session_df['Start Trial'].iloc[0], color='r', linestyle='--', label='CS On')
	# find all values of ml_cam_frames < 700
	ml_cam_frames = session_df['cam_frames'].iloc[0]
	ml_cam_frames = [x for x in ml_cam_frames if x > ml_time_start and x < ml_time_end]
	max_sig = max(session_df['cam_sync'].iloc[0])
	axarr[0].scatter(ml_cam_frames, [max_sig]*len(ml_cam_frames), c='r', s=10, zorder=10, label='ml_cam_frames')
	axarr[0].set_title('ML Start of Trial 1')
	axarr[0].set_xlabel('Time (ms)')
	axarr[0].set_ylabel('Voltage (V)')

	# Get SpikeGLX start/end of trial 0
	time_ms = 25
	# ax.plot(np.arange(spikeglx_cam_times[0]['cam_start']-time_ms, spikeglx_cam_times[0]['cam_start']+time_ms), 
	#             np.array(cam_sync)[ml_time_start-time_ms:ml_time_start+time_ms]*1000, 
	#             label='ML cam_sync')
	
	# Plot SpikeGLX start of trial 0
	sglx_start = spikeglx_cam_times[0]['start'] + ml_time_start
	data_array_selected, data_times_selected = \
		time_to_samples(spikeglx_obj.cam_sync, 
										sample_rate, 
										sample_times, 
										tStart=sglx_start  - time_ms,
										tEnd=sglx_start + (time_ms*2))
	axarr[1].plot(data_times_selected, data_array_selected, label='ephys cam_sync')
	data_array_selected, data_times_selected = \
		time_to_samples(spikeglx_obj.cam_save, 
										sample_rate, 
										sample_times, 
										tStart=sglx_start - time_ms,
										tEnd=sglx_start + (time_ms*2))
	axarr[1].plot(data_times_selected, data_array_selected, label='ephys cam_save')
	# plot spikeglx_cam_frames in front of other plots
	cam_frames_specified = [i for i in spikeglx_cam_frames if i > sglx_start - time_ms and i < sglx_start + (time_ms*2)]
	axarr[1].scatter(cam_frames_specified, [3300]*len(cam_frames_specified), c='r', s=10, label='ephys cam_frames', zorder=10)
	axarr[1].legend()
	axarr[1].set_title('Camera Sync')
	axarr[1].set_ylabel('Voltage (mV)')
	axarr[1].set_title('SpikeGLX Start of Trial 1')

	# Plot SpikeGLX end of trial 0
	data_array_selected, data_times_selected = \
		time_to_samples(spikeglx_obj.cam_sync, 
										sample_rate, 
										sample_times, 
										tStart=spikeglx_cam_times[0]['end']-time_ms,
										tEnd=spikeglx_cam_times[0]['end']+time_ms)
	axarr[2].plot(data_times_selected, data_array_selected, label='ephys cam_sync')
	data_array_selected, data_times_selected = \
		time_to_samples(spikeglx_obj.cam_save, 
										sample_rate, 
										sample_times, 
										tStart=spikeglx_cam_times[0]['end']-time_ms,
										tEnd=spikeglx_cam_times[0]['end']+time_ms)
	axarr[2].plot(data_times_selected, data_array_selected, label='ephys cam_save')
	cam_frames_specified = [i for i in spikeglx_cam_frames if i > spikeglx_cam_times[0]['end']-time_ms and i < spikeglx_cam_times[0]['end']]
	axarr[2].scatter(cam_frames_specified, [3300]*len(cam_frames_specified), c='r', s=10, label='ephys cam_frames', zorder=10)
	axarr[2].set_xlabel('Time (ms)')
	axarr[2].set_ylabel('Voltage (mV)')
	axarr[2].set_title('SpikeGLX End of Trial 1')

	plt.tight_layout()
	plt.show()


def plot_analog_ML(df, epochs, trial_num=1):
	f, ax = plt.subplots(figsize=(15, 3))
	trial_selected = df[df['trial_num'] == trial_num]
	photodiode_selected = trial_selected['photodiode'].tolist()[0]
	cam_signal = trial_selected['cam_sync'].tolist()[0]
	save_signal = trial_selected['cam_save'].tolist()[0]
	lick_signal = trial_selected['lick'].tolist()[0]
	for e_index, epoch in enumerate(epochs):
		epoch_time = trial_selected[epoch].tolist()[0]
		try:
			ax.axvline(x=epoch_time, color='r', linestyle='--')
			if e_index % 2 == 0:
				ax.text(epoch_time, max(lick_signal)+0.35, epoch, ha='center', fontsize=10)
			else:
				ax.text(epoch_time, max(lick_signal)+0.75, epoch, ha='center', fontsize=10)
		except:
			pass
	ax.plot(save_signal, label='Camera Save Signal')
	ax.plot(lick_signal, label='Lick Signal')
	ax.plot(photodiode_selected, label='Photodiode')
	ax.set_xlabel('Time (ms)')
	ax.set_ylabel('Voltage (mV)')
	plt.suptitle('ML Analog Signals', fontsize=14)
	# space between sup title and title
	plt.subplots_adjust(top=0.75)
	# raise title 
	ax.set_title(f'Trial Number: {trial_num}', fontsize=10, y=1.15)
	plt.legend()
	plt.show()

def ml_check_save_low(trial):
	# check if save signal is low for the entire trial
	cam_save_signal = trial['cam_save']
	if min(cam_save_signal) < 1:
		return 1
	else:
		return 0

def spikeglx_cam_frames_window(spikeglx_obj, trial_num, spikeglx_cam_times, spikeglx_cam_framenumbers, col_start, col_end):
	"""
	Finds the first and last frame of the camera from SpikeGLX in a given window

	Parameters
	----------
	spikeglx_obj : SpikeGLX object
		SpikeGLX object
	trial_num : int
		Trial number
	spikeglx_cam_times : defaultdict
		Dictionary of trial start/end times
	spikeglx_cam_framenumbers : defaultdict
		Dictionary of trial start/end frame numbers
	col_start : str
		Column name of start time
	col_end : str
		Column name of end time

	Returns
	-------
	spikeglx_cam_framenumbers : defaultdict
		Dictionary of trial start/end frame numbers
		
	"""
	# check if both column start and column end not equal to <NA>
	if pd.isnull(spikeglx_cam_times[trial_num][col_start]) == False and pd.isnull(spikeglx_cam_times[trial_num][col_end]) == False:
		spikeglx_frames_trial = [f_index for f_index,frame in enumerate(spikeglx_obj.cam_frames) if \
														frame >= spikeglx_cam_times[trial_num][col_start] and \
														frame <= spikeglx_cam_times[trial_num][col_end]]
		spikeglx_cam_framenumbers[trial_num][col_start] = spikeglx_frames_trial[0]
		spikeglx_cam_framenumbers[trial_num][col_end] = spikeglx_frames_trial[-1]	
	# time epoch NaN (i.e. error before Trace Start)
	else:
		spikeglx_cam_framenumbers[trial_num][col_start] = np.nan
		spikeglx_cam_framenumbers[trial_num][col_end] = np.nan
	return spikeglx_cam_framenumbers

def time_to_samples(data_array, sample_rate, data_times, tStart, tEnd):
	'''
	Converts time in milliseconds to samples
	from sglx data array
	'''
	first_sample = int(sample_rate*tStart/1000)
	last_sample = int(sample_rate*tEnd/1000)
	data_array = data_array[first_sample:last_sample]
	data_times = data_times[first_sample:last_sample]
	return data_array, data_times

def align_ml_sglx_verbose(t_index, 
													trial_start_pd_ML,
													trial_end_pd_ML, 
													trial_len_pd_ML, 
													trial_start_ephys, 
													trial_end_ephys):
	"""Prints trial alignment information"""
	
	print(f'Trial Number: {t_index+1}')
	print('  ML:')
	print(f'  Photodiode Start : {trial_start_pd_ML}')
	print(f'  Photodiode End   : {trial_end_pd_ML}')
	print(f'  Trial Length     : {trial_len_pd_ML}')
	print('  SpikeGLX:')
	print(f'  Photodiode Start : {trial_start_ephys}')
	print(f'  Photodiode End   : {trial_end_ephys}')
	print(f'  Trial Length     : {trial_end_ephys-trial_start_ephys}')

def plot_pd_alignment(trial_specified, sglx_pd_times, sglx_pd_signal, 
											sglx_trial_times, sglx_cam_framenumbers, 
											offset, epochs):
	"""Plots MonkeyLogic and SpikeGLX photodiode signal for a given trial"""	
	trial_index = trial_specified['trial_num'] - 1
	trial_num = trial_specified['trial_num']
	# find the first and last frame of the camera from ML
	ml_cam_start = trial_specified['cam_frames'][0]
	ml_cam_end = trial_specified['cam_frames'][-1]
	ml_num_frames = len(trial_specified['cam_frames'])
	ml_analog = trial_specified['photodiode']*1000

	f, ax = plt.subplots(1, 1, figsize=(20, 3))
	if trial_index == 0:
		ax.plot(np.array(ml_analog), label='ML (raw)', color='blue') # x-axis mV to V
		ax.text(0, max(ml_analog)+300, 'SpikeGLX Start', ha='center', color='r')
		ax.axvline(x=0, color='r', linestyle='--')
	ax.plot(sglx_pd_times, sglx_pd_signal, label='sglx', color='green')
	# plot for verification
	ax.plot(np.arange(len(ml_analog))+offset, 
					(np.array(ml_analog))+100, 
					alpha=0.5,
					label='ML (x-aligned, y-offset)',
					color='purple')

	ax.axvline(x=offset, color='r', linestyle='--')
	ax.text(offset, max(ml_analog)+300, 'ML Start', ha='center', color='r')
	for e_index, epoch in enumerate(epochs):
		try:
			epoch_time = trial_specified[epoch]+offset
			if e_index % 2 == 0:
				ax.axvline(x=epoch_time, color='grey', linestyle='--', alpha=0.5)
				ax.text(epoch_time, max(ml_analog)+300, epoch, ha='center', color='grey')
		except:
			pass
	spikeglx_num_frames = sglx_cam_framenumbers[trial_index]['end'] - sglx_cam_framenumbers[trial_index]['start'] + 1
	plt.xlabel('SpikeGLX Time (ms)')
	plt.ylabel('Voltage (mV)')
	plt.suptitle(f'Trial Number {trial_num} | Photodiode Signal', fontsize=18, y=1.1)
	plt.title(f'Offset: {round(offset)} ms', fontsize=14, y=1.2)
	plt.subplots_adjust(top=0.8)
	# legend outside of plot
	plt.legend(bbox_to_anchor=(1, 1.25), loc='upper left', borderaxespad=0.)
	plt.show()
	# bold print statements
	print(f'Trial {trial_num}:')
	# align all the vertical lines for printing
	print('  ML Cam Start           |  {:<7}'.format(ml_cam_start))
	print('  ML Cam End             |  {:<7}'.format(ml_cam_end))
	print('  ML Cam Num Frames      |  {:<7}'.format(ml_num_frames))
	print('  --------------------------------------')
	print('  SpikeGLX Trial Start   |  {:<7}'.format(round(sglx_trial_times[trial_index]['start'], 2)))
	print('  SpikeGLX Trial End     |  {:<7}'.format(round(sglx_trial_times[trial_index]['end'], 2)))
	print('  SpikeGLX Num Frames    |  {:<7}'.format(spikeglx_num_frames))

def shift_correlation_test(ml_analog, ml_photodiode, sglx_pd_signal_exact, sglx_photodiode, sample_rate, sample_times, ):

	sglx_pd_signal_ml_sampled = [sglx_pd_signal_exact[int(i*sample_rate/1000)] 
															for i in range(len(ml_analog))]

	# calculate correlations between ML and SGLX photodiode signals
	corr = np.corrcoef(ml_photodiode, sglx_pd_signal_ml_sampled)[0, 1]
	start_shift = 0
	# confidence is high that everything is aligned appropriately
	if corr > 0.95:
		low_corr_flag = False
	else:
		# shift sglx trial start and end times by 1 ms and try again
		low_corr_flag = True
		while low_corr_flag == True:
			start_shift += 1
			sglx_trial_start += 1
			sglx_trial_end += 1
			sglx_pd_signal_exact, sglx_pd_times_exact = \
				time_to_samples(sglx_photodiode, sample_rate, sample_times, 
												sglx_trial_start, sglx_trial_end)
			sglx_pd_signal_ml_sampled = [sglx_pd_signal_exact[int(i*sample_rate/1000)] 
																	for i in range(len(ml_analog))]
			corr = np.corrcoef(ml_photodiode, sglx_pd_signal_ml_sampled)[0, 1]
			if corr > 0.95:
				low_corr_flag = False
			# stop at a full 10 second shift
			elif start_shift > 10000:
				print(' Correlation never corrected. Check alignment.')
				break
	return corr, start_shift

def plot_spikeglx_ml_corr(correlation_matrix, corr_row_len):
	f, ax = plt.subplots(figsize=(15, 8))
	correlation_matrix_modular = np.zeros((int(len(correlation_matrix)/corr_row_len), corr_row_len))
	for i in range(len(correlation_matrix)):
		correlation_matrix_modular[int(i/corr_row_len)][i%corr_row_len] = correlation_matrix[i]

	# heatmap of correlation between ML and SpikeGLX photodiode signals
	im = ax.imshow(correlation_matrix_modular, cmap='viridis')
	# set color bewteen values 0 and 1
	im.set_clim(0.001, 1)
	# set color for empty (i.e. no correlation)
	im.cmap.set_under('white')
	# create colorbar the size of the plot
	cbar = ax.figure.colorbar(im, ax=ax, shrink=0.25, aspect=10)
	# set colorbar ticks
	cbar.set_ticks([0, 0.5, 1])
	# set colorbar tick labels
	cbar.set_ticklabels(['0', '0.5', '1'])
	# set ticks for x-axis
	ax.set_xticks(np.arange(corr_row_len))
	# set font size of xticks
	ax.set_xticklabels(np.arange(corr_row_len), fontsize=6)
	ax.set_yticks(np.arange(len(correlation_matrix_modular)))
	ax.set_yticklabels(np.arange(len(correlation_matrix_modular)), fontsize=6)
	
	plt.title('Correlation between ML and SpikeGLX Photodiode Signals')
	plt.xlabel('Trial Number')
	plt.ylabel('Trial Number (hundreds)')
	plt.show()

def align_sglx_ml(spikeglx_obj, df, epochs):
	"""
	Aligns the camera save signal from ML to the camera save signal from sglx
	only for the first trial. This is done by finding the offset between the two
	signals and then shifting the ML signal by that amount.
	"""
	# get data from SpikeGLX object
	sample_rate = spikeglx_obj.sample_rate
	sample_times = spikeglx_obj.sample_times
	sglx_trial_times = spikeglx_obj.trial_times
	sglx_cam_framenumbers = spikeglx_obj.cam_framenumbers
	sglx_photodiode = spikeglx_obj.photodiode
	sglx_cam_save = spikeglx_obj.cam_save
	
	# correlation test threshold
	CORR_THRESHOLD = 0.97

	# for UnityVR task, trial start and trial end times are sometimes asynchronous so extra checks required
	pre_trial_shift = 200
	if spikeglx_obj.monkey_name == 'gandalf':
		pre_trial_shift = -1000 # approximate trial start 1000 ms before trial end of previous trial for extra buffer

	# initialize correlation matrix
	correlation_matrix = np.zeros(int(np.ceil(len(df)/100))*100)

	print('Epochs')
	for epoch in epochs:
		print(f'  {epoch}')

	# loop through all trials
	# use tqdm to show progress bar
	for trial_index_specified in tqdm(range(len(df)), desc='Trial Number', position=0, leave=True):

		trial_specified = df.iloc[trial_index_specified]
		ml_photodiode = trial_specified['photodiode']*1000 # V to mV
		ml_cam_save = trial_specified['cam_save']*1000 # V to mV

		# if True, plot the camera save signal from ML and SpikeGLX
		low_corr_flag = True

		# align on cam_save signal
		if trial_index_specified == 0:
			ml_analog = ml_cam_save	
			# estimate the first 30 seconds of acquisition
			sglx_trial_start_approx = 0
			sglx_trial_end_approx = 30000
			# capture SpikeGLX photodiode signal between approximated trial start and end times
			sglx_analog_approx, sglx_analog_times_approx = \
				time_to_samples(sglx_cam_save, sample_rate, sample_times, 
											sglx_trial_start_approx, 
											sglx_trial_end_approx)
		# align on photodiode signal
		else:
			ml_analog = ml_photodiode
			# check -1000 ms before the end of the last trial to get the start of the next trial ONLY FOR GANDALF
			sglx_trial_start_approx = sglx_trial_times[trial_index_specified-1]['end'] + pre_trial_shift
			sglx_trial_end_approx = sglx_trial_start_approx + len(ml_analog)
			# capture SpikeGLX cam_save signal between approximated trial start and end times
			sglx_analog_approx, sglx_analog_times_approx = \
				time_to_samples(sglx_photodiode, sample_rate, sample_times, 
											sglx_trial_start_approx, 
											sglx_trial_end_approx)

		# find first time where analog signal goes high on SpikeGLX
		sglx_analog_high_time = 0
		sglx_analog_high = 0
		for i, x in enumerate(sglx_analog_approx):
			if i == 0:
				continue
			if sglx_analog_approx[i-1] < 1000 and x > 1000:
				# get data_times corresponding to save_high_ephys
				sglx_analog_high_time = sglx_analog_times_approx[i]
				sglx_analog_high = i
				break
		# find first time where analog signal goes high on ML
		ml_analog_high = 0
		for i in range(1, len((ml_analog))):
			if ml_analog[i-1] < 1000 and ml_analog[i] > 1000:
				ml_analog_high = i
				break

		# sglx_trial_start is set to the difference between the save_high times
		sglx_trial_start = sglx_analog_high_time - ml_analog_high
		# sglx_trial_end is sglx_trial_start + length of any ml analog signal (i.e. pd)
		sglx_trial_end = sglx_trial_start + len(ml_analog)
		# calculate correlation between ML and sglx pd signals
		sglx_pd_signal_exact, sglx_pd_times_exact = \
			time_to_samples(sglx_photodiode, sample_rate, sample_times, 
											sglx_trial_start, sglx_trial_end)
		sglx_pd_signal_ml_sampled = [sglx_pd_signal_exact[int(i*sample_rate/1000)] 
																	for i in range(len(ml_analog))]

		# calculate correlations between ML and SGLX photodiode signals
		corr = np.corrcoef(ml_photodiode, sglx_pd_signal_ml_sampled)[0, 1]
		max_corr = [0, corr] # [shift, corr]
		if corr > CORR_THRESHOLD:
			low_corr_flag = False
			print(f'Trial {trial_index_specified+1} | Correlation: {round(corr, 3)} | SGLX High Time: {round(sglx_analog_high, 2)} | ML High Time: {round(ml_analog_high, 2)}')

		# Correlation test with shifting sglx start times to find the best start index
		elif low_corr_flag and spikeglx_obj.monkey_name == 'gandalf':
			start_shift = 0
			max_corr = [0, 0] # [shift, corr]
			# shift sglx trial start and end times by 1 ms and try again
			while low_corr_flag == True:
				start_shift += 1
				sglx_trial_start_approx_shift = sglx_trial_start_approx + start_shift		# start at the end of the previous trial + pre_shift time (i.e. -1000ms before n-1 trial end)
				sglx_trial_end_approx_shift = sglx_trial_end_approx + start_shift				# end at start + length of ML photodiode signal
				sglx_pd_signal_exact, sglx_pd_times_exact = \
					time_to_samples(sglx_photodiode, sample_rate, sample_times, 
													sglx_trial_start_approx_shift, sglx_trial_end_approx_shift)
				sglx_pd_signal_ml_sampled = [sglx_pd_signal_exact[int(i*sample_rate/1000)] 
																		for i in range(len(ml_analog))]
				corr = np.corrcoef(ml_photodiode, sglx_pd_signal_ml_sampled)[0, 1]
				if corr > max_corr[1]:
					max_corr[0] = start_shift
					max_corr[1] = corr
				# print(f'  Trial {trial_index_specified} Shift {sglx_trial_start} | Correlation: {round(corr, 3)}')
				if corr > CORR_THRESHOLD:
					low_corr_flag = False
					print(f'Trial {trial_index_specified+1} | Correlation: {round(corr, 3)} | Shift: {pre_trial_shift+start_shift}')
				# stop at a full 5 second shift after the end of the previous trial
				elif start_shift > 5000:
					print(f'Trial {trial_index_specified+1} correlation never corrected. Check alignment.')
					print(f'	Max Correlation: Shift {max_corr[0]} | Correlation: {max_corr[1]}')
					break
			# set sglx_trial_start to the best shift
			sglx_trial_start = sglx_trial_start_approx + max_corr[0]
			sglx_trial_end = sglx_trial_start + len(ml_analog)
			sglx_pd_signal_exact, sglx_pd_times_exact = \
				time_to_samples(sglx_photodiode, sample_rate, sample_times, 
												sglx_trial_start, sglx_trial_end)

		# add trial start and end times to dictionary
		sglx_trial_times[trial_index_specified]['start'] = sglx_trial_start
		sglx_trial_times[trial_index_specified]['end'] =  sglx_trial_end
		sglx_cam_framenumbers = spikeglx_cam_frames_window(spikeglx_obj, trial_index_specified, sglx_trial_times, sglx_cam_framenumbers, 
																											 col_start='start', col_end='end')
		
		# add epoch times to dictionary
		for e_index, epoch in enumerate(epochs):
			sglx_trial_times[trial_index_specified][epoch] = sglx_trial_start + trial_specified[epoch]
			# capture cam_frames in epoch window
			if e_index == 0:
				continue
			sglx_cam_framenumbers = spikeglx_cam_frames_window(spikeglx_obj, trial_index_specified, sglx_trial_times, sglx_cam_framenumbers, 
																											 	 col_start=epochs[e_index-1], col_end=epoch)

		# add correlation value to correlation matrix
		correlation_matrix[trial_index_specified] = max_corr[1]
	
		# plot trial if correlation between ML and SGLX photodiode signal is low
		if low_corr_flag:
			plot_pd_alignment(trial_specified, sglx_pd_times_exact, sglx_pd_signal_exact,
							sglx_trial_times, sglx_cam_framenumbers, sglx_trial_start, epochs)
			# plot_pd_alignment(trial_specified, sglx_analog_times_approx, sglx_analog_approx,
			# 				sglx_trial_times, sglx_cam_framenumbers, sglx_trial_start, epochs)
			print(f'  ML-SGLX Correlation: {round(max_corr[1], 3)}')
	
	# plot correlation matrix
	plot_spikeglx_ml_corr(correlation_matrix, 100)
	spikeglx_obj.trial_times = sglx_trial_times
	spikeglx_obj.cam_framenumbers = sglx_cam_framenumbers
	# flatten correlation matrix
	spikeglx_obj.ml_sglx_corr_matrix = correlation_matrix.flatten()

	return spikeglx_obj

# get length of all cam_frames
def compare_ML_sglx_cam_frames(spikeglx_obj, session_df):
	cam_frames = session_df['cam_frames'].tolist()
	cam_frames_flatten = [item for sublist in cam_frames for item in sublist]
	print('Number of frames in ML Cam TTL: {}'.format(len(cam_frames_flatten)))
	print('Number of frames in SpikeGLX Cam TTL: {}'.format(len(spikeglx_obj.cam_frames)))

