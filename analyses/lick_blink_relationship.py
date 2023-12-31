import os
import math
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, ttest_ind_from_stats, linregress
import warnings
warnings.filterwarnings("ignore")
from matplotlib.offsetbox import AnchoredText

def lick_blink_linear(session_df, session_obj):
	"""
	Plots and calculates the linear relationship
	between lick rate and blink duration for 
	a given trial 
	"""
	FIGURE_SAVE_PATH = session_obj.figure_path
	COLORS = session_obj.valence_colors
	import seaborn as sns
	
	#reinforcement trials only
	df = session_df[session_df['reinforcement_trial'] == 1]
	

	# Plot regression line from all valences
	f, ax = plt.subplots(1,1, figsize=(4,4))
	x_all = np.array(df['lick_duration'].tolist())
	y_all = df['blink_duration_offscreen'].tolist()
	sns.regplot(x=x_all, y=y_all, color='black', label='all', ax=ax, scatter=False)
	slope, intercept, r_value, p_value, std_err = linregress(x_all, y_all)
	text = 'R-Value: {} | P-Value: {:.3g}'.format(round(r_value, 3), p_value)
	anchored_text = AnchoredText(text,
															 loc='lower center',
															 frameon=False)
	ax.add_artist(anchored_text)
	# plt.gcf().text(1, 1.1, text, fontsize=14) 

	# Plot each valence data 
	for df_index, valence in enumerate(sorted(df['valence'].unique(), reverse=True)):

		df_valence = df[df['valence'] == valence]

		# valence-specific session lick/blink data
		x = np.array(df_valence['lick_duration'].tolist())
		y = df_valence['blink_duration_offscreen'].tolist()
		color = COLORS[valence]
		sns.regplot(x=x, y=y, color=color, label=valence, ax=ax, ci=None)
	
	plt.ylim([-0.175, 1.1])
	yticks = np.round(np.arange(0, 1.2, 0.2), 2)
	plt.yticks(yticks, yticks)
	plt.legend(loc='upper right', labels=['all'], fontsize=10, frameon=False)
	plt.xlabel('Norm Lick Duration')
	plt.ylabel('Norm Blink Duration')
	plt.title('Lick vs Blink Duration')
	fig_name = 'lick_vs_blink'
	img_save_path = os.path.join(FIGURE_SAVE_PATH, fig_name)
	f.savefig(img_save_path, dpi=150, bbox_inches='tight', transparent=True)
	print('  {}.png saved.'.format(fig_name))
	plt.close('all')

def trialno_lick_blink_correlation(df, session_obj):
  f, ax = plt.subplots(1, 1, figsize=(5, 5))
  df_threshold = df[df['fractal_count_in_block'] > 10]
  for block in df_threshold['condition'].unique():
    df_block = df_threshold[df_threshold['condition'] == block]
    print(f'Block: {block}')
    for valence in sorted(df_block['valence'].unique(), reverse=True):
      df_block_valence = df_block[df_block['valence'] == valence]
      trialno_lick_corr = round(df_block_valence['fractal_count_in_block'].corr(df_block_valence['lick_duration']), 3)
      trial_no_blink_corr = round(df_block_valence['fractal_count_in_block'].corr(df_block_valence['blink_duration_window']), 3)
      print(f'  Valence {valence}: Lick Correlation: {trialno_lick_corr} | Blink Correlation: {trial_no_blink_corr}')
      ec = 'black' if block == 1 else 'white'
      ax.scatter(trialno_lick_corr, trial_no_blink_corr, s=150,
                label=None, 
                color=session_obj.valence_colors[valence], ec=ec)
      ax.set_xlabel('Trial Number vs. Lick Duration Correlation', fontsize=14)
      ax.set_ylabel('Trial Number vs. Blink Duration Correlation', fontsize=14)
  ax.set_xlim(-0.5, 0.5)
  ax.set_ylim(-0.5, 0.5)
  plt.show()