import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import make_interp_spline, BSpline

from utilities.plot_helper import smooth_plot, round_up_to_odd, moving_avg

pd.options.mode.chained_assignment = None  # default='warn'

def session_performance(df, behavioral_code_dict, session_obj, latency=True):

	FIGURE_SAVE_PATH = session_obj.figure_path
	COLORS = session_obj.valence_colors
	LABELS = session_obj.valence_labels
	valence_list = sorted(df['valence'].unique(), reverse=True)

	# Only Reinforcement Trials
	session_df_reinforcement = df[df['reinforcement_trial'] == 1]

	# Only CS Presented Trials
	session_df_cs_on = session_df_reinforcement[session_df_reinforcement['CS On'].notna()]

	gs_kw = dict(width_ratios=[6, 1])
	f, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw=gs_kw, figsize=(20,5))
	correct_list_filtered = session_df_cs_on['correct'].tolist()
	scatter_color_list = []
	dates = list(map(int, df['date']))
	new_dates = np.diff(dates,prepend=dates[0])

	# poly_order = 2
	# correct_moving_average = np.array(df_moving_avg['correct_list_moving_avg'])
	# window_size = round_up_to_odd(int(len(correct_moving_average)/15))
	# y = signal.savgol_filter(correct_moving_average, int(window_size), poly_order)
	num_rolling = 5
	y = moving_avg(correct_list_filtered, num_rolling)
	x = list(range(len(y)))
	
	# random choice threshold
	ax1.axhline(y=0.5, color='lightgrey',linewidth=1)

	# Coloring in valence 
	for i in range(len(correct_list_filtered)):
		valence = session_df_cs_on.iloc[i]['valence_1']
		color = COLORS[valence]
		ax1.axvspan(xmin=i-0.35, xmax=i+0.35, ymin=0.15, ymax=0.85, alpha=0.2, color=color)
		scatter_color_list.append(color)

	title_str = 'Performance By Fractal'
	ax1.set_title(label=title_str, ha='center', fontsize=16)

	ax1.plot(x, y, linewidth=4, color='white')
	correct_list_filtered = [r+0.1 if r==1 else r-0.1 for r in correct_list_filtered]
	ax1.scatter(x, correct_list_filtered[num_rolling-1:], s=6, color=scatter_color_list[num_rolling-1:])
	ax1.set_ylim([-0.2,1.2])
	ax1.set_yticks([0,0.5,1])
	ax1.set_xlabel('Trial Number', fontsize=16)
	ax1.set_ylabel('Rolling Average (n={})'.format(num_rolling), fontsize=16)
	date_lines = list(np.nonzero(new_dates)[0])
	for date in date_lines:
		ax1.axvline(x=date, c='lightgrey', linestyle='-')
	f.tight_layout(pad=2)

	rewards = []
	label_list = []
	for fractal in sorted(session_df_cs_on['valence_1'].unique(), reverse=True):
		session_df_fractal = session_df_cs_on[session_df_cs_on['valence_1'] == fractal]
		reward_val = np.nansum(session_df_fractal['correct'])/len(session_df_fractal)
		rewards.append(reward_val)
		label_list.append(session_obj.valence_labels[fractal])

	x = np.arange(len(rewards))
	unique_color_list = [COLORS[valence] for valence in sorted(session_df_cs_on['valence_1'].unique(), reverse=True)]
	ax2.bar(np.arange(len(rewards)), rewards, color=unique_color_list, ec='black')
	ax2.set_xticks(x) # values
	ax2.set_xticklabels(label_list) # labels
	ax2.set_xlabel('Fractal')
	ax2.set_ylabel('Fixation Hold (CS On)')
	ax2.set_ylim([0,1])
	#ax2.set_title('Performance by Fractal')
	
	img_save_path = os.path.join(FIGURE_SAVE_PATH, 'perf_by_fractal')
	print('  perf_by_fractal.png saved.')
	plt.savefig(img_save_path, dpi=150, bbox_inches='tight', pad_inches = 0.1)
	plt.show()
	plt.close('all')

	# latency
	if latency == True:
		for date in df['date'].unique():
			session_df_date = df[df['date'] == date]
			trial_absolute_start = session_df_date['trial_datetime_start'] + pd.to_timedelta(session_df_date['Fixation Success'])
			# extract hours, minutes, and seconds from datetime
			trial_hour = trial_absolute_start.dt.hour * 60 * 60
			trial_minute = trial_absolute_start.dt.minute * 60
			trial_second = trial_absolute_start.dt.second 
			# add together
			trial_absolute_start = trial_hour + trial_minute + trial_second
			session_df_date['latency'] = np.diff(trial_absolute_start, prepend=trial_absolute_start.iloc[0])/1000
			latency = session_df_date['latency'].rolling(5).mean()
			# get rolling average of latency
			f, ax = plt.subplots(1, 1, figsize=(11, 4))
			x = np.arange(len(latency))
			ax.plot(x, latency, label='Latency', linewidth=2, alpha=0.75)
			ax.set_xlabel('Trial #', fontsize=8)
			ax.set_ylabel('Latency (s)', fontsize=8)
			ax.set_title(f'{date}', fontsize=12)
			plt.tight_layout()
			# save figure
			plot_title = f'latency_{date}.svg'
			img_save_path = os.path.join(FIGURE_SAVE_PATH, plot_title)
			# transparent background
			f.savefig(img_save_path, dpi=150, bbox_inches='tight', pad_inches = 0.1, transparent=True)
			print(f'  {plot_title} saved.')

def dataframe_summary(df):
	
	session_df_correct = df[df['correct'] == 1]
	
	print('Dates:')
	for date in df['date'].unique():
		df_date = df[df['date'] == date]
		df_date_correct = df_date[df_date['correct'] == 1]
		df_percent_correct = len(df_date_correct)/len(df_date)
		print(f'  {date}: {len(df_date):>10} trials | {len(df_date_correct):>3} correct | {df_percent_correct:.2%} correct')

	print('Blocks:')
	for block in session_df_correct['block'].unique():
		session_df_correct_block = session_df_correct[session_df_correct['block'] == block]
		print(f'  {block}: {len(session_df_correct_block):>15} trials')

	print('Trial Types:')
	session_df_reinforcement = session_df_correct[session_df_correct['reinforcement_trial'] == 1]
	print(f'  RL: {len(session_df_reinforcement):>14} trials')
	session_df_choice = session_df_correct[session_df_correct['reinforcement_trial'] == 0]
	print(f'  Choice: {len(session_df_choice):>10} trials')

	print('Valences:')
	for valence in sorted(session_df_correct['valence'].unique()):
		session_df_correct_valence = session_df_correct.loc[(session_df_correct['valence'] == valence) & (session_df_correct['reinforcement_trial'] == 1)]
		print(f'  {valence:>4}: {len(session_df_correct_valence):>12} trials')