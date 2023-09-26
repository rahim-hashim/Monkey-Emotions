import os
import numpy as np
import pandas as pd
from scipy import signal
from decimal import Decimal
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import combinations, permutations
from scipy.stats import ttest_ind, ttest_ind_from_stats
import warnings
warnings.filterwarnings("ignore")

# Custom Functions
from utilities.plot_helper import smooth_plot, round_up_to_odd, moving_avg, set_plot_params

def epoch_time(df):
	"""
	Calculate the minimum epoch times for CS, Trace, and Outcome
	so that all trials can be plotted with the same number of bins

	Args:
		df (dataframe): dataframe containing all trial data

	Returns:
		cs_end_min (int): minimum CS duration
		trace_end_min (int): minimum Trace duration
		outcome_end_min (int): minimum Outcome duration
	"""
	
	verbose = False # set to True to print out average/min values

	# CS Epoch
	cs_duration_hist = np.array(df['Trace Start'].tolist()) - np.array(df['CS On'].tolist())
	cs_end_min = min(cs_duration_hist)
	cs_end_min_index = np.argmin(cs_duration_hist)
	cs_end_mean = round(np.mean(cs_duration_hist))
	# Trace Epoch
	trace_duration_hist = np.array(df['Trace End'].tolist()) - np.array(df['Trace Start'].tolist())
	trace_end_min = min(trace_duration_hist) + cs_end_min
	trace_end_min_index = np.argmin(trace_duration_hist)
	trace_end_mean = round(np.mean(trace_duration_hist) + cs_end_mean)
	# Outcome Epoch
	outcome_duration_hist = np.array(df['trial_bins'].tolist()) - np.array(df['Trace End'].tolist())
	outcome_end_min = min(outcome_duration_hist) + cs_end_min - 1 # not sure why I included -1 but it was necessary, check this
	outcome_end_min_index = np.argmin(outcome_duration_hist)
	outcome_end_mean = round(np.mean(outcome_duration_hist) + cs_end_mean - 1)
	if verbose:
		print('  Trial Number of cs min: {}'.format(df['trial_num'].tolist()[cs_end_min_index]))
		print('    cs average: {}'.format(cs_end_mean))
		print('    cs min: {} | index: {}'.format(cs_end_min, cs_end_min_index))
		print('  Trial Number of trace min: {}'.format(df['trial_num'].tolist()[trace_end_min_index]))	
		print('    trace + cs average: {}'.format(trace_end_mean))
		print('    trace + cs min: {} | index: {}'.format(trace_end_min, trace_end_min_index))
		print('  Trial Number of outcome min: {}'.format(df['trial_num'].tolist()[outcome_end_min_index]))
		print('    outcome + cs average: {}'.format(outcome_end_mean))
		print('    outcome + cs min: {}'.format(outcome_end_min))

	return cs_end_min, trace_end_min, outcome_end_min

def raster_by_condition(session_df, session_obj):
	"""
	Plots raster plots for each condition (valence x fractal type)

	Args:
		session_df (dataframe): dataframe containing all trial data
		session_obj (object): object containing all session data

	Returns:
		None
		
	Plots:
		3x3 raster plots for each condition (valence x fractal type)
	"""

	set_plot_params(FONT=20, AXES_TITLE=22, AXES_LABEL=20, TICK_LABEL=20, LEGEND=16, TITLE=28)

	PRE_CS = 50 # time before CS-on (for moving average calculation)
	FIGURE_SAVE_PATH = session_obj.figure_path
	COLORS = session_obj.valence_colors
	WINDOW_THRESHOLD_LICK = session_obj.window_lick
	WINDOW_THRESHOLD_BLINK = session_obj.window_blink

	# keys = each valence, values = list of lick/blink probability data
	lick_data_probability = defaultdict(list)
	DEM_data_probability = defaultdict(list)
	pupil_data_probability = defaultdict(list)

	# keys = each valence, values = list of lick/blink duration data
	lick_data_duration = defaultdict(list)
	DEM_data_duration = defaultdict(list)
	pupil_data_duration = defaultdict(list)

	gs_kw = dict(width_ratios=[5, 1, 1])
	f, axarr = plt.subplots(3,3, gridspec_kw=gs_kw, sharey = False, figsize=(50,20))
	num_fractals = len(session_df['valence'].unique())	

	TRIAL_THRESHOLD = 10

	# only include trials after subject has seen fractal <TRIAL_THRESHOLD> number of times
	session_df_count = session_df[session_df['fractal_count_in_block'] > TRIAL_THRESHOLD]
	# only include one switch (for now)
	session_df_threshold = session_df_count[session_df_count['block'] <= 2]

	# calculate minimum epoch times
	cs_time_min, trace_time_min, outcome_time_min = epoch_time(session_df_threshold)

	valence_list = sorted(session_df_threshold['valence'].unique(), reverse=True)
	LABELS = []
	color_list = [COLORS[valence] for valence in valence_list]
	for df_index, valence in enumerate(valence_list):

		# keys = each trial, values = list of lick/blink duration data
		lick_dict = defaultdict(list)
		DEM_dict = defaultdict(list)
		pupil_dict = defaultdict(list)

		# key 1 = each trial, key 2 = each time bin, values = list of lick/blink duration data
		lick_epoch_dict = defaultdict(lambda:defaultdict(list))
		DEM_epoch_dict = defaultdict(lambda:defaultdict(list))	
		pupil_epoch_dict = defaultdict(lambda:defaultdict(list))

		df = session_df_threshold[session_df_threshold['valence'] == valence]

		# valence-specific session lick/blink data
		lick_data_raster = df['lick_raster'].tolist()
		DEM_data_raster = df['DEM_raster'].tolist()
		pupil_data_raster = df['pupil_raster'].tolist()

		# single bin lick data (-<WINDOW_THRESHOLD>ms from trace interval end)

		for t_index, trial in enumerate(lick_data_raster):

			cs_on_time = df['CS On'].iloc[t_index]
			trace_on_time = df['Trace Start'].iloc[t_index]
			trace_off_time = df['Trace End'].iloc[t_index]

			# Lick/Blink Probability
			## counts if there was any lick in the specified time window
			lick_data_window = df['lick_count_window'].iloc[t_index]
			if 1 in lick_data_window:
				lick_data_probability[df_index].append(1)
			else:
				lick_data_probability[df_index].append(0)

			## counts if there was any blink in the specified time window
			DEM_data_window = df['blink_count_window'].iloc[t_index]
			if 1 in DEM_data_window:
				DEM_data_probability[df_index].append(1)
			else:
				DEM_data_probability[df_index].append(0)

			## counts if there was any pupil = 0 (true blink) in the specified time window
			pupil_data_window = df['pupil_raster_window'].iloc[t_index]
			if 1 in pupil_data_window:
				pupil_data_probability[df_index].append(1)
			else:
				pupil_data_probability[df_index].append(0)

			# Lick/Blink/Pupil Avg Duration
			lick_raw = df['lick_duration'].iloc[t_index]
			lick_data_duration[df_index].append(lick_raw)

			DEM_raw = df['blink_duration_offscreen'].iloc[t_index]
			DEM_data_duration[df_index].append(DEM_raw)

			pupil_raw = df['pupil_raster_window_avg'].iloc[t_index]
			pupil_data_duration[df_index].append(pupil_raw)

			# Lick/Blink/Pupil Blink Epochs
			lick_data_trial = lick_data_raster[t_index][cs_on_time-PRE_CS:]
			DEM_data_trial = DEM_data_raster[t_index][cs_on_time-PRE_CS:]
			pupil_data_trial = pupil_data_raster[t_index][cs_on_time-PRE_CS:]

			lick_data_cs = lick_data_raster[t_index][cs_on_time:trace_on_time]
			lick_data_trace = lick_data_raster[t_index][trace_on_time:trace_off_time]
			lick_data_outcome = lick_data_raster[t_index][trace_off_time:]

			DEM_data_cs = DEM_data_raster[t_index][cs_on_time:trace_on_time]
			DEM_data_trace = DEM_data_raster[t_index][trace_on_time:trace_off_time]
			DEM_data_outcome = DEM_data_raster[t_index][trace_off_time:]

			pupil_data_cs = pupil_data_raster[t_index][cs_on_time:trace_on_time]
			pupil_data_trace = pupil_data_raster[t_index][trace_on_time:trace_off_time]
			pupil_data_outcome = pupil_data_raster[t_index][trace_off_time:]

			time = np.arange(len(lick_data_trial))

			# lick_data_trial and DEM_data_trial are sometimes off by 1 frame
			# 	must investigate further
			shorter_trial_data = min(len(lick_data_trial), len(DEM_data_trial), len(pupil_data_trial))
			for bin_num in range(shorter_trial_data):
				lick_dict[bin_num].append(lick_data_trial[bin_num])
				DEM_dict[bin_num].append(DEM_data_trial[bin_num])
				pupil_dict[bin_num].append(pupil_data_trial[bin_num])

			for bin_num in range(cs_time_min):
				lick_epoch_dict['CS'][bin_num].append(lick_data_cs[bin_num])
				DEM_epoch_dict['CS'][bin_num].append(DEM_data_cs[bin_num])
				pupil_epoch_dict['CS'][bin_num].append(pupil_data_cs[bin_num])
			for bin_num in range(trace_time_min-cs_time_min):
				lick_epoch_dict['Trace'][bin_num].append(lick_data_trace[bin_num])
				DEM_epoch_dict['Trace'][bin_num].append(DEM_data_trace[bin_num])
				pupil_epoch_dict['Trace'][bin_num].append(pupil_data_trace[bin_num])
			for bin_num in range(outcome_time_min-cs_time_min):
				lick_epoch_dict['Outcome'][bin_num].append(lick_data_outcome[bin_num])
				DEM_epoch_dict['Outcome'][bin_num].append(DEM_data_outcome[bin_num])
				pupil_epoch_dict['Outcome'][bin_num].append(pupil_data_outcome[bin_num])

		# Now analyze all trials together

		bins = list(lick_dict.keys())
		lick_data_mean = list(map(np.mean, lick_dict.values()))
		DEM_data_mean = list(map(np.mean, DEM_dict.values()))
		pupil_data_raster_mean = list(map(np.mean, pupil_dict.values()))
		label = session_obj.valence_labels[valence]
		LABELS.append(label)

		# Simple Moving Average Smoothing
		WINDOW_SIZE = PRE_CS
		x = np.array(bins[PRE_CS:]) # only capturing post-CS bins
		y1 = moving_avg(lick_data_mean, WINDOW_SIZE)
		axarr[0][0].plot(x, y1[:-1], 
										color=color_list[df_index], label=label, linewidth=4)
		y2 = moving_avg(DEM_data_mean, WINDOW_SIZE)
		axarr[1][0].plot(x, y2[:-1], 
										color=color_list[df_index], label=label, linewidth=4)
		y3 = moving_avg(pupil_data_raster_mean, WINDOW_SIZE)
		axarr[2][0].plot(range(len(y3)), y3, 
										color=color_list[df_index], label=label, linewidth=4)

	axarr[0][0].text(0, 1.15, 'CS On', ha='center', va='center', fontsize='x-large')
	axarr[0][0].text(cs_time_min, 1.15, 'Delay', ha='center', va='center', fontsize='x-large')
	axarr[0][0].text(trace_time_min, 1.15, 'Outcome', ha='center', va='center', fontsize='x-large')
	axarr[0][0].set_ylabel('Probability of Lick', fontsize='x-large')
	axarr[0][0].set_ylim([0, 1.05])
	axarr[0][0].set_yticks(np.arange(0,1.2,0.2).round(1))
	axarr[0][0].set_yticklabels(np.arange(0,1.2,0.2).round(1))
	# xticks
	axarr[0][0].set_xticks(np.arange(0,4500, 500))
	axarr[0][0].set_xticklabels(np.arange(0,4500, 500))
	axarr[1][0].set_xticks(np.arange(0,4500, 500))
	axarr[1][0].set_xticklabels(np.arange(0,4500, 500))
	axarr[2][0].set_xticks(np.arange(0,4500, 500))
	axarr[2][0].set_xticklabels(np.arange(0,4500, 500))
	# lick
	axarr[0][1].set_title('Delay\n(last {}ms)'.format(WINDOW_THRESHOLD_LICK))
	axarr[0][2].set_title('Delay\n(last {}ms)'.format(WINDOW_THRESHOLD_LICK))
	# DEM
	axarr[1][1].set_title('Delay\n(last {}ms)'.format(WINDOW_THRESHOLD_BLINK))
	axarr[1][2].set_title('Delay\n(last {}ms)'.format(WINDOW_THRESHOLD_BLINK))
	# Pupil Zero
	axarr[2][1].set_title('Delay\n(last {}ms)'.format(WINDOW_THRESHOLD_LICK))
	axarr[2][2].set_title('Delay\n(last {}ms)'.format(WINDOW_THRESHOLD_LICK))

	condition = list(session_df['block'].unique())[0]
	# if condition == 1:
	# 	title = 'Pre-Reversal'
	# if condition == 2:
	# 	title = 'Post-Reversal'
	# axarr[0][0].set_title(title, pad=40, fontsize=40)
	axarr[0][0].set_xlabel('Time since visual stimuli onset (ms)', fontsize=26)

	axarr[1][0].set_ylabel('Probability of DEM', fontsize=26)
	axarr[1][0].set_ylim([0, 1.05])

	axarr[2][0].set_ylabel('Probability of Blink', fontsize=26)
	axarr[2][0].set_xlabel('Time since visual stimuli onset (ms)', fontsize=26)
	axarr[2][0].set_yticks(np.arange(0,1.2,0.2))
	
	probability_list = [lick_data_probability, DEM_data_probability, pupil_data_probability]
	duration_list = [lick_data_duration, DEM_data_duration, pupil_data_duration]
	label_list_prob = ['Lick Probability', 'DEM Probability', 'Blink Probability']
	label_list_dur = ['Avg Lick Duration', 'Avg DEM Duration', 'Avg Blink Duration']

	# Plotting Probability and Duration for Lick, DEM, and Blink
	for ax_index in range(3):
		# Time Epochs
		axarr[ax_index][0].axvline(0)
		axarr[ax_index][0].axvline(cs_time_min)
		axarr[ax_index][0].axvline(trace_time_min)
		if ax_index >= 1:
			window_threshold_label = WINDOW_THRESHOLD_BLINK
		else:
			window_threshold_label = WINDOW_THRESHOLD_LICK
		axarr[ax_index][0].axvspan(xmin=trace_time_min-window_threshold_label,
								xmax=trace_time_min-1,
								ymin=0,
								ymax=1,
								alpha=0.2,
								color='grey')

		axarr[ax_index][0].legend(loc='upper right', frameon=False)
		## Bar Graph - lick/blink probability
		data_probability_mean = list(map(np.mean, probability_list[ax_index].values()))
		axarr[ax_index][1].bar(list(range(num_fractals)), data_probability_mean, color=color_list, ec='black')
		axarr[ax_index][1].set_xticks(list(range(num_fractals)))
		axarr[ax_index][1].set_xticklabels(LABELS, fontsize=26)
		axarr[ax_index][1].set_xlabel('Outcome')
		axarr[ax_index][1].set_ylabel('{}'.format(label_list_prob[ax_index]))

		## Bar Graph - lick/blink duration
		data_duration_mean = list(map(np.mean, duration_list[ax_index].values()))
		if ax_index == 0:
			axarr[ax_index][2].set_ylim([0, 1])
		axarr[ax_index][2].bar(list(range(num_fractals)), data_duration_mean, color=color_list, ec='black')
		axarr[ax_index][2].set_xticks(list(range(num_fractals)))
		axarr[ax_index][2].set_xticklabels(LABELS, fontsize=26)
		axarr[ax_index][2].set_xlabel('Outcome', fontsize=26)
		axarr[ax_index][2].set_ylabel('{}'.format(label_list_dur[ax_index]), fontsize=26)

	plot_title = 'raster_by_cond_{}.svg'.format(condition)
	img_save_path = os.path.join(FIGURE_SAVE_PATH, plot_title)
	f.tight_layout()
	f.savefig(img_save_path, dpi=150, bbox_inches='tight', pad_inches = 0.1, transparent=True)
	print('  {} saved.'.format(plot_title))
	# plt.close('all')