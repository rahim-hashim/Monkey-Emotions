import os
import math
import numpy as np
import pandas as pd
from scipy import signal
from decimal import Decimal
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
from itertools import combinations, permutations
from scipy.stats import ttest_ind, ttest_ind_from_stats, f_oneway
from statsmodels.stats.weightstats import ztest as ztest
import warnings
warnings.filterwarnings("ignore")

# Custom Functions
from utilities.plot_helper import smooth_plot, round_up_to_odd, moving_avg, set_plot_params
from analyses.two_sample_test import two_sample_test

def significance_test(measure_duration_list, outcome_mag_list, measure):
	"""
	significance_test performs a two sample t-test on the lick/blink data duration list

	Args:
		measure_duration_list (list): list of lick data duration

	Returns:
		t_stat (float): t-statistic
		p_value (float): p-value
	"""
	ANOVA_stat, ANOVA_pvalue = f_oneway(*measure_duration_list)
	ANOVA_p_value_string = '%.2E' % Decimal(ANOVA_pvalue)
	print(' {} ANOVA {} | P-value: {}'.format(outcome_mag_list, round(ANOVA_stat, 3), ANOVA_p_value_string))
	measure_mag_combinations = list(combinations(range(len(outcome_mag_list)), 2))
	measure_duration_combinations = list(combinations(measure_duration_list, 2))
	for m_index, magnitude in enumerate(measure_mag_combinations):
		mag_1 = measure_duration_combinations[m_index][0]
		mag_2 = measure_duration_combinations[m_index][1]
		t, p = ttest_ind(mag_1, 
										 mag_2,
										 equal_var=False)
		p_val_string = '%.2E' % Decimal(p)
		z_val, p_value = ztest(mag_1, mag_2, 
													 alternative='two-sided', 
													 usevar='pooled', 
													 ddof=1.0)
		z_val_string = '%.2E' % Decimal(p_value)
		print('  {}'.format(magnitude), 'T-value: {}'.format(round(t,3)), 'P-value: {} | '.format(p_val_string),
																		'Z-value: {}'.format(round(z_val,3)), 'P-value: {}'.format(z_val_string,3))
		print('    {}'.format(measure_mag_combinations[m_index][0]), 
												'{} Mean: {}'.format(measure, round(np.nanmean(mag_1), 3)),
												'{} Std: {}'.format(measure, round(np.std(mag_1), 3)), 
												'Trials: {}'.format(len(mag_1)))
		print('    {}'.format(measure_mag_combinations[m_index][1]), 
												'{} Mean: {}'.format(measure, round(np.nanmean(mag_2), 3)), 
												'{} Std: {}'.format(measure, round(np.std(mag_2), 3)), 
												'Trials: {}'.format(len(mag_2)))

def get_param_labels(session_obj, param):
	params_dict = defaultdict(list)
	gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1.5], height_ratios=[1, 1])
	if 'lick' in param:
		color = ['#D2DCD3', '#91BFBC', '#28398D']
		mag = 'reward_mag'
		outcome = 'Reward'
		measure = 'Lick'
		window_threshold = session_obj.window_lick
		fig_dimensions = [gs[:, 1], gs[0, 0], gs[1, 0]]
	else: 
		color = ['#D2DCD3', '#ED8C8C', '#D61313']
		mag = 'airpuff_mag'
		outcome = 'Airpuff'
		measure = 'DEM' if param == 'blink_duration_offscreen' else 'Blink'
		fig_dimensions = [gs[:, 0], gs[0, 1], gs[1, 1]]
		window_threshold = session_obj.window_blink
	return color, mag, outcome, measure, window_threshold, fig_dimensions

def grant_plots(session_df, session_obj):

	plot_params = ['lick_duration', 'blink_duration_offscreen', 'pupil_raster_window_avg']

	for plot_index, plot_param in enumerate(plot_params):
		# Get plot parameters
		color, mag, outcome, measure, window_threshold, fig_dimensions = get_param_labels(session_obj, plot_param)

		set_plot_params(FONT=12,
										AXES_TITLE=16,
										AXES_LABEL=18, 
										TICK_LABEL=12, 
										LEGEND=10, 
										TITLE=20)

		fig = plt.figure(figsize=(10, 6))

		FIGURE_SAVE_PATH = session_obj.figure_path
		TRIAL_THRESHOLD = 20
		session_df_correct = session_df[session_df['correct'] == 1]
		# only include trials after subject has seen fractal <TRIAL_THRESHOLD> number of times
		session_df_count = session_df_correct[session_df_correct['fractal_count_in_block'] > TRIAL_THRESHOLD]
		# only include one switch (for now)
		session_df_threshold = session_df_count[session_df_count['block'] <= 2]

		# Collapsed on conditions
		ax1 = fig.add_subplot(fig_dimensions[0])
		ax1.set_title('Collapsed Across Session', fontsize=18)

		# Condition 1
		ax2 = fig.add_subplot(fig_dimensions[1])
		ax2.set_title('Pre-Switch', fontsize=16)

		# Condition 2
		ax3 = fig.add_subplot(fig_dimensions[2])
		ax3.set_title('Post-Switch', fontsize=16)
		axarr = [ax1, ax2, ax3]
		TRIAL_THRESHOLD = 10

		outcome_mag_list = sorted(session_df_threshold[mag].unique())
		if len(outcome_mag_list) < 2:
			print('  Only 1 magnitude for {}. Skipping...'.format(outcome))
			continue
		conditions = [[1, 2], [1], [2]]
		# Collapsed on conditions | Condition 1 | Condition 2
		for ax_index, condition in enumerate(conditions):
			measure_duration_list = []
			measure_mean_list = []
			measure_std_list = []
			df_condition = session_df_threshold[session_df_threshold['condition'].isin(condition)]
			# Reward | Airpuff
			for df_index, outcome_mag in enumerate(outcome_mag_list):

				df = df_condition[df_condition[mag] == outcome_mag]
				measure_duration = df[plot_param].tolist()
				measure_data_mean = np.nanmean(measure_duration)
				measure_duration_list.append(measure_duration)
				measure_std_list.append(np.std(measure_duration))
				measure_mean_list.append(measure_data_mean)
			
			if ax_index == 0: # only print collapsed on conditions
				significance_test(measure_duration_list, outcome_mag_list, measure)

			axarr[ax_index].bar(range(len(measure_mean_list)), 
													measure_mean_list,
													# yerr=measure_std_list,
													color=color, linewidth=2)
			axarr[ax_index].set_xticks(range(len(measure_mean_list)))
			outcome_mag_labels = []
			for magnitude in outcome_mag_list:
				if magnitude == 0:
					outcome_mag_labels.append('none')
				elif magnitude == 0.25:
					outcome_mag_labels.append('small')
				elif magnitude == 0.5:
					outcome_mag_labels.append('medium')
				elif magnitude == 0.75:
					outcome_mag_labels.append('large')
				elif magnitude == 1:
					outcome_mag_labels.append('largest')
			axarr[ax_index].set_xticklabels(outcome_mag_labels)
			axarr[ax_index].set_ylim([0,1])
		axarr[0].set_xlabel('{} Magnitude'.format(outcome))
		axarr[0].set_ylabel('Average {} Duration'.format(measure))
		fig.tight_layout()
		# set facecolor to black:
		fig.set_facecolor("k")
		grant_title = 'grant_{}.png'.format(measure.lower())
		img_save_path = os.path.join(FIGURE_SAVE_PATH, grant_title)
		print('  {} saved.'.format(grant_title))
		plt.savefig(img_save_path, dpi=150, bbox_inches='tight', pad_inches = 0.1)
		plt.close('all')
