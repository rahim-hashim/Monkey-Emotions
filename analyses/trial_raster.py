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

from utilities.plot_helper import set_plot_params

# Plot Layout
##  A   B
##  C   D
ax_map_i = [0, 0, 1, 1]
ax_map_j = [0, 1, 0, 1]

def trial_raster(df, session_obj):
	'''
	Takes a dataframe and a session object 
	and plots the lick and blink rasters.
	'''

	set_plot_params(FONT=10, AXES_TITLE=12, AXES_LABEL=10, TICK_LABEL=10, LEGEND=10, TITLE=14)

	FIGURE_SAVE_PATH = session_obj.figure_path
	FRACTAL_NAMES = df['fractal_chosen'].unique().tolist()
	LABELS = session_obj.stim_labels
	COLORS = session_obj.colors
	BIN = 10 # for binning the data
	POST_TRACE = 350 # time after outcome to plot
	max_y = max(tuple(df.groupby('fractal_chosen').size())) # for labeling
	num_fractals = len(session_obj.stim_labels)

	# f1, axarr1 = plt.subplots(2,2, figsize=(5,10), sharey = True) # lick data plot
	# f2, axarr2 = plt.subplots(2,2, figsize=(5,10), sharey = True) # blink data plot
	f1, axarr1 = plt.subplots(1,num_fractals, figsize=(20,7), sharey = True) # lick data plot
	f2, axarr2 = plt.subplots(1,num_fractals, figsize=(20,7), sharey = True) # lick data plot

	for f_index, fractal in enumerate(sorted(FRACTAL_NAMES)):
		fractal_df = df[df['fractal_chosen'] == fractal]
		# pos_i = ax_map_i[f_index]
		# pos_j = ax_map_j[f_index]
		fractal_df['block_change'] = fractal_df['block'].diff()
		block_change_trials = np.nonzero(fractal_df['block_change'].tolist())[0]
		block_change = -1
		if len(block_change_trials) > 1:
			block_change = np.nonzero(fractal_df['block_change'].tolist())[0][1]
			# axarr1[pos_i][pos_j].axhline(y=block_change+1, c='darkred', alpha=0.5)
			# axarr2[pos_i][pos_j].axhline(y=block_change+1, c='darkred', alpha=0.5)
			axarr1[f_index].axhline(y=block_change+1, c='darkred', alpha=0.5)
			axarr2[f_index].axhline(y=block_change+1, c='darkred', alpha=0.5)
		for index in range(len(fractal_df)+1):
			i = index
			# Make blank row for block change
			if index == block_change:
				continue
			if index > block_change:
				i -= 1
			cs_on = fractal_df['CS On'].iloc[i]
			trace_start = fractal_df['Trace Start'].iloc[i]
			trace_end_extra = fractal_df['Trace End'].iloc[i] + POST_TRACE
			lick_raster = fractal_df['lick_raster'].iloc[i]
			lick_raster_time = lick_raster[cs_on:trace_end_extra]
			blink_raster = fractal_df['pupil_raster'].iloc[i]
			blink_raster_time = blink_raster[cs_on:trace_end_extra]
			lick_raster_window = []
			blink_raster_window = []
			length_window = round((trace_end_extra-cs_on)/10)*10
			x_range = np.linspace(0,length_window,int(length_window/BIN)+1)
			for j in x_range:
				start = int(j)
				if 1 in lick_raster_time[start:start+BIN]:
					lick_raster_window.append(1)
				else:
					lick_raster_window.append(0)
				if 1 in blink_raster_time[start:start+BIN]:
					blink_raster_window.append(1)
				else:
					blink_raster_window.append(0)
			lick_raster_window_index = [bin*(index+1) if bin != 0 else np.nan \
																		for bin in lick_raster_window] # replace 0 with np.nan
			blink_raster_window_index = [bin*(index+1) if bin != 0 else np.nan \
																	for bin in blink_raster_window] # replace 0 with np.nan
			
			# axarr1[pos_i][pos_j].scatter(x_range, lick_raster_window_index, marker='|', s=3, color=COLORS[f_index])
			# axarr2[pos_i][pos_j].scatter(x_range, blink_raster_window_index, marker='|', s=3, color=COLORS[f_index])
			axarr1[f_index].scatter(x_range, lick_raster_window_index, marker='|', s=3, color=COLORS[f_index])
			axarr2[f_index].scatter(x_range, blink_raster_window_index, marker='|', s=3, color=COLORS[f_index])

		for index, ax in enumerate([axarr1, axarr2]):
			# Set timing labels
			cs_on = np.mean(np.array(fractal_df['CS On'].tolist()))
			trace_start = np.mean(np.array(fractal_df['Trace Start'].tolist()) - cs_on)
			if index == 0:
				window_start = np.mean(np.array(fractal_df['Trace End'].tolist()) - cs_on - session_obj.window_lick)
			else:
				window_start = np.mean(np.array(fractal_df['Trace End'].tolist()) - cs_on - session_obj.window_blink)
			window_end = np.mean(np.array(fractal_df['Trace End'].tolist()) - cs_on)			

			# ax[pos_i][pos_j].set_title(LABELS[f_index])		
			# ax[pos_i][pos_j].axvline(x=0, color='black', linewidth=1) # CS onset
			# ax[pos_i][pos_j].axvline(x=trace_start, color='black', linewidth=1) # Trace start
			# ax[pos_i][pos_j].axvspan(xmin=window_start, xmax=window_end, color='lightgrey', alpha=0.4) # Window start
			# ax[pos_i][pos_j].axvline(x=window_end, color='black', linewidth=1) # Trace end
			# ax[pos_i][pos_j].set_xlabel('Time after CS (ms)')
			# if pos_j == 0:
			# 	ax[pos_i][pos_j].set_ylabel('Fractal Trial')

			ax[f_index].set_title(LABELS[f_index])		
			ax[f_index].axvline(x=0, color='black', linewidth=1) # CS onset
			ax[f_index].axvline(x=trace_start, color='black', linewidth=1) # Trace start
			ax[f_index].axvspan(xmin=window_start, xmax=window_end, color='lightgrey', alpha=0.4) # Window start
			ax[f_index].axvline(x=window_end, color='black', linewidth=1) # Trace end
			ax[f_index].set_xlabel('Time after CS (ms)')
			ax[f_index].set_ylabel('Fractal Trial')

	f1.suptitle('Lick Raster',y=0.93)
	name1 = 'fractal_lick_raster'
	img_save_path_1 = os.path.join(FIGURE_SAVE_PATH, name1)
	f1.savefig(img_save_path_1, dpi=150, bbox_inches='tight')
	print('  {}.png saved.'.format(name1))
	
	f2.suptitle('Blink Raster',y=0.93)
	name2 = 'fractal_blink_raster'
	img_save_path_2 = os.path.join(FIGURE_SAVE_PATH, name2)
	f2.savefig(img_save_path_2, dpi=150, bbox_inches='tight')
	print('  {}.png saved.'.format(name2))
	plt.close('all')