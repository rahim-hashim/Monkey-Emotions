import os
import math
import string
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
from utilities.plot_helper import smooth_plot, round_up_to_odd, moving_avg, moving_var, set_plot_params

def generate_data_dict(session_df, session_obj):

	lick_data_probability = defaultdict(list)
	blink_probability = defaultdict(list)
	lick_data_duration = defaultdict(list)
	DEM_duration = defaultdict(list)
	blink_duration = defaultdict(list)

	PRE_CS = 50 # time before CS-on (for moving average calculation)
	FIGURE_SAVE_PATH = session_obj.figure_path
	COLORS = session_obj.valence_colors
	WINDOW_THRESHOLD_LICK = session_obj.window_lick
	
	valence_list = sorted(session_df['valence'].unique(), reverse=True)
	for df_index, valence in enumerate(valence_list):

		df = session_df[session_df['valence'] == valence]

		# valence-specific session lick/blink data
		lick_data_raster = df['lick_raster'].tolist()
		blink_data_raster = df['blink_raster'].tolist()
		# pupil_data = df['eye_pupil'].tolist()

		# single bin lick data (-<WINDOW_THRESHOLD>ms from trace interval end)

		for t_index, trial in enumerate(lick_data_raster):
		
			trace_off_time = df['Trace End'].iloc[t_index]
			# Lick/Blink Probability
			## counts if there was any lick in the specified time window
			lick_data_window = df['lick_count_window'].iloc[t_index]
			if 1 in lick_data_window:
				lick_data_probability[valence].append(1)
			else:
				lick_data_probability[valence].append(0)

			# counts if there was any blink (pupil=0) in the specified time window
			pupil_binary_zero = df['pupil_binary_zero'].iloc[t_index]
			blink_probability[valence].append(pupil_binary_zero)

			# Lick/Blink Duration
			lick_raster = df['lick_raster'].iloc[t_index]
			lick_raster_window = lick_raster[trace_off_time-WINDOW_THRESHOLD_LICK:trace_off_time]
			lick_raster_mean = np.mean(lick_raster_window)
			lick_data_duration[valence].append(lick_raster_mean)

			DEM_raw = df['blink_duration_offscreen'].iloc[t_index]
			DEM_duration[valence].append(DEM_raw)

			blink_raw = df['pupil_raster_window_avg'].iloc[t_index]
			blink_duration[valence].append(blink_raw)
	
	return lick_data_probability, blink_probability, lick_data_duration, DEM_duration, blink_duration

def two_sample_test(data_type, df, data_raster, condition, session_obj, direction='forwards'):
	'''Lick/Blink Probability 2-Sample T-Test'''

	f, axarr = plt.subplots(2,3, figsize=(12,4), sharex=True, sharey=True)

	FIGURE_SAVE_PATH = session_obj.figure_path
	COLORS = session_obj.valence_colors
	LABELS = session_obj.valence_labels
	num_valences = len(LABELS)
	window_width = 5
	valence_list = [1.0, 0.5, -0.5, -1.0]

	verbose = False
	valence_combinations = list(combinations(valence_list, 2))
	prob_combinations = list(combinations(data_raster.values(), 2))
	for f_index, valence_pair in enumerate(valence_combinations):
		if valence_pair == (1.0, 0.5):
			ax = axarr[0][0]
		if valence_pair == (1.0, -0.5):
			ax = axarr[0][1]
		if valence_pair == (1.0, -1.0):
			ax = axarr[0][2]
		if valence_pair == (0.5, -0.5):
			ax = axarr[1][0]
		if valence_pair == (0.5, -1.0):
			ax = axarr[1][1]
		if valence_pair == (-0.5, -1.0):
			ax = axarr[1][2]
		valence_1, valence_2 = valence_pair[0], valence_pair[1]
		valence_1_label, valence_2_label = LABELS[valence_pair[0]], LABELS[valence_pair[1]]
		a = data_raster[valence_1]
		d = data_raster[valence_2]
		if 0.0 in df['valence'].unique():
			z = data_raster[0.0]
			ma_vec_z = moving_avg(z, window_width)
			mv_vec_z = moving_var(z, window_width)
		t_all, p_all = ttest_ind(a, d, equal_var=False)
		ma_vec_a = moving_avg(a, window_width)
		mv_vec_a = moving_var(a, window_width)
		ma_vec_d = moving_avg(d, window_width)
		mv_vec_d = moving_var(d, window_width)
		min_array, max_array = (ma_vec_a, ma_vec_d) if len(a) <= len(d) else (ma_vec_d, ma_vec_a)
		min_len = len(min_array)
		one_star_flag = 1
		two_star_flag = 1
		three_star_flag = 1
		if direction == 'backwards':
			# do something
			pass
		if direction == 'forwards':
			for window in range(window_width, min_len):
				a_windowback = a[window-window_width:window]
				d_windowback = d[window-window_width:window]
				t, p = ttest_ind(a_windowback, 
												d_windowback,
												equal_var=False)
				star_pos = max(np.mean(a_windowback),np.mean(d_windowback))+0.05
				if p < 0.001 and three_star_flag:
					ax.text(window-window_width, star_pos, s='***', ha='center')
					p_str = "{:.2e}".format(p)
					one_star_flag = 0
					two_star_flag = 0
					three_star_flag = 0	
				elif p < 0.01 and two_star_flag:
					ax.text(window-window_width, star_pos, s='**', ha='center')
					p_str = "{:.2e}".format(p)	
					one_star_flag = 0
					two_star_flag = 0	
				elif p < 0.05 and one_star_flag:
					p_str = str(round(p, 3))
					ax.text(window-window_width, star_pos, s='*', ha='center')
					one_star_flag = 0
				else:
					p_str = str(round(p, 3))
			# Plot Running Average
			ax.plot(list(range(len(ma_vec_a))), ma_vec_a, c=COLORS[valence_1], lw=4, label=valence_1_label)
			ax.plot(list(range(len(ma_vec_d))), ma_vec_d, c=COLORS[valence_2], lw=4, label=valence_2_label)
			# Plot Variance 
			ax.fill_between(list(range(len(ma_vec_a))), ma_vec_a-(mv_vec_a/2), ma_vec_a+(mv_vec_a/2),
											color=COLORS[valence_1], alpha=0.2) # variance
			ax.fill_between(list(range(len(ma_vec_d))), ma_vec_d-(mv_vec_d/2), ma_vec_d+(mv_vec_d/2),
											color=COLORS[valence_2], alpha=0.2) # variance
			ax.set_xticks(list(range(0, len(ma_vec_a), 5)))
			ax.set_xticklabels(list(range(window_width, len(ma_vec_a)+window_width, 5)))
			# Plot neutral fractal if it exists
			if 0.0 in df['valence'].unique():
				ax.plot(list(range(len(ma_vec_z))), ma_vec_z, c=COLORS[0.0], lw=4, label=LABELS[0.0], alpha=0.5)
				ax.fill_between(list(range(len(ma_vec_z))), ma_vec_z-(mv_vec_z/2), ma_vec_z+(mv_vec_z/2),
												color=COLORS[0.0], alpha=0.1)
		if direction == 'backwards':
			size_diff = len(max_array) - len(min_array)
			for window in range(window_width, min_len):
				min_windowback = min_array[window-window_width+size_diff:window+size_diff]
				max_windowback = max_array[window-window_width:window]
			# Plot Running Average
			ax.plot(list(range((min_len-1)*-1, 1)), ma_vec_a[-min_len:], c=COLORS[valence_1], lw=4, label=valence_1_label)
			ax.plot(list(range((min_len-1)*-1, 1)), ma_vec_d[-min_len:], c=COLORS[valence_2], lw=4, label=valence_2_label)
			base = 5
			round_off = base * math.ceil((min_len-1)/ base)
			ax.set_xticks(list(range((round_off)*-1, 1, 5)))
			ax.set_xticklabels(list(range((round_off)*-1, 1, 5)))
		ax.legend(fontsize='x-small', loc='lower left')
		if 'prob' in data_type:
			ax.set_ylim([0,1])
		if verbose:
			print('  {}'.format(window),
						valence_1_label, valence_2_label, 
						round(np.mean(a), 3),
						round(np.mean(d), 3),
						format(p_all, '.3g'))

	f.supylabel('Moving Avg ({} Trials)'.format(window_width))
	if direction=='backwards':
		if condition == 1:
			f.supxlabel('Trial Number Before Switch')
		else:
			f.supxlabel('Trial Number Before Session End')
	else:
		f.supxlabel('Trial Number')
	title = data_type.split('-')
	title_full = ' '.join(title)
	f.suptitle(title_full.title() + ' (Condition {})'.format(condition))

	f.tight_layout()
	plot_title = 't_test_{}_{}.png'.format(data_type, condition)
	img_save_path = os.path.join(FIGURE_SAVE_PATH, plot_title)
	f.savefig(img_save_path, dpi=150, bbox_inches='tight', pad_inches = 0.1)
	print('  {} saved.'.format(plot_title))
	plt.show()
	plt.close('all')

def t_test_moving_avg(df, session_obj, condition):
	# T-Test Plots
	lick_data_probability, blink_probability, lick_data_duration, DEM_duration, blink_duration =\
		 		 generate_data_dict(df, session_obj)
	set_plot_params(FONT=10, AXES_TITLE=11, AXES_LABEL=10, TICK_LABEL=10, LEGEND=8, TITLE=14)
	two_sample_test('lick-duration', df, lick_data_duration, condition, session_obj)
	two_sample_test('DEM-duration', df, DEM_duration, condition, session_obj)
	two_sample_test('blink-duration', df, blink_duration, condition, session_obj)	