import os
import sys
import numpy as np
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt
from collections import defaultdict
# custom functions
from utilities.plot_helper import moving_avg, moving_var

def add_outcome_data(trial, measure_dict, measure):
	valence = trial['valence']
	condition = trial['condition']
	outcome_data = np.nanmean(trial[measure])
	fractal_count = trial['fractal_count_in_block']
	measure_dict[condition][fractal_count].append(outcome_data)
	# print(valence, condition, fractal_count, outcome_data)
	return measure_dict

def outcome_over_time(df, session_obj):
	"""
	Plots the average number of blinks and licks over a session
	"""
	FIGURE_SAVE_PATH = session_obj.figure_path
	avg_window = 5 # N trial rolling avg
	num_blocks = len(df['block'].unique())
	f, axarr = plt.subplots(3, num_blocks, sharey=False, figsize=(16, 11), squeeze=False)
	df_reinforcement = df[df['reinforcement_trial'] == 1]
	# find the minimum number of trials for each valence for each block
	for v_index, valence in enumerate(sorted(df_reinforcement['valence'].unique(), reverse=True)):
		df_valence = df_reinforcement[df_reinforcement['valence'] == valence]
		lick_dict = defaultdict(lambda: defaultdict(list))
		dem_dict = defaultdict(lambda: defaultdict(list))
		blink_dict = defaultdict(lambda: defaultdict(list))	
		# Add each date to lick/blink dicts
		for date in sorted(df['date'].unique()):
			df_date = df_valence[df_valence['date'] == date]
			for index, trial in df_date.iterrows():
				lick_dict = add_outcome_data(trial, lick_dict, 'lick_count_window')
				dem_dict = add_outcome_data(trial, dem_dict, 'blink_duration_offscreen')
				blink_dict = add_outcome_data(trial, blink_dict, 'pupil_raster_window_avg')
		measure_dicts = [lick_dict, dem_dict, blink_dict]
		for m_index, measure_dict in enumerate(measure_dicts):
				for b_index, block in enumerate(measure_dict.keys()):
					measure_array_block = moving_avg(list(map(lambda x: np.nanmean(x), measure_dict[block].values())),
																		avg_window)
					measure_array_block_std = moving_var(list(map(lambda x: np.nanmean(x), measure_dict[block].values())),
																		avg_window)
					color=session_obj.valence_colors[valence]
					x_range = np.arange(0, len(measure_array_block))
					axarr[m_index][b_index].plot(x_range, measure_array_block, color=color, label=valence, lw=4)
					axarr[m_index][b_index].fill_between(x_range, measure_array_block-(measure_array_block_std/2), measure_array_block+(measure_array_block_std/2), color=color, alpha=0.1)
					axarr[m_index][b_index].axhline(y=np.nanmean(measure_array_block), color=color, linestyle='--', alpha=0.3)
					axarr[m_index][b_index].set_xticks(np.arange(0, len(measure_array_block)+avg_window, avg_window))
					axarr[m_index][b_index].set_xticklabels(np.arange(avg_window, len(measure_array_block)+(avg_window*2), avg_window))
					axarr[m_index][b_index].set_ylim([0, 1])
					axarr[m_index][b_index].set_title('Block {}'.format(block), fontsize=20)
					if m_index == len(measure_dicts)-1:
						axarr[m_index][b_index].set_xlabel('Trial Number', fontsize=16)
		xticklabels = np.arange(avg_window, len(measure_array_block)+avg_window, avg_window)
		axarr[0][0].set_ylabel('Trial Lick (Delay)', fontsize=16)
		axarr[0][0].set_ylim([0, 1])
		# set legend outside of plot
		axarr[1][0].set_ylabel('Trial DEM (Delay)', fontsize=16)
		axarr[1][0].set_ylim([0, 1])
		axarr[2][0].set_ylabel('Trial Blink (Delay)', fontsize=16)
		axarr[2][0].set_ylim([0, 0.5])
	f.tight_layout()
	# set font for all text in figure as Optima
	for ax in f.get_axes():
		for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
			ax.get_xticklabels() + ax.get_yticklabels()):
			item.set_fontname('Optima')
	if FIGURE_SAVE_PATH:
		fig_name = 'moving_avg_lick_blink'
		img_save_path = os.path.join(FIGURE_SAVE_PATH, fig_name)
		f.savefig(img_save_path, dpi=150, bbox_inches='tight')
		print('  {}.png saved.'.format(fig_name))
	plt.show()
	