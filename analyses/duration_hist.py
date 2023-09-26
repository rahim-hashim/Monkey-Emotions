import os
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

def duration_hist_plot(cs_duration_hist, trace_duration_hist, trial_duration_hist, FIGURE_SAVE_PATH):
  num_bins = 20
  f, (ax1, ax2, ax3) = plt.subplots(1,3, sharey = True, figsize=(10,5))
  
  ax1.hist(cs_duration_hist,
           num_bins, 
           color ='blue',
           alpha = 0.7, 
           ec = 'black')
  ax1.set_xlabel('CS Interval')
  ax1.set_ylabel('Trial Count')

  ax2.hist(trace_duration_hist,
          num_bins, 
          color ='red',
          alpha = 0.7, 
          ec = 'black')
  ax2.set_xlabel('Delay')

  ax3.hist(trial_duration_hist,
           num_bins, 
           color ='purple',
           alpha = 0.7, 
           ec = 'black')
        
  ax3.set_xlabel('Trial End - CS On')
    
  ax2.set_title('Trial Duration',
            fontweight = 'bold')
  
  img_save_path = os.path.join(FIGURE_SAVE_PATH, '_duration_hist')
  print('  duration_hist.png saved.')
  plt.savefig(img_save_path, dpi=150, bbox_inches='tight',pad_inches = 0.1)

def duration_hist(df, behavioral_code_dict, FIGURE_SAVE_PATH):
    
  cs_duration_hist = np.array(df['Trace Start'].tolist()) - np.array(df['CS On'].tolist())
  trace_duration_hist = np.array(df['Trace End'].tolist()) - np.array(df['Trace Start'].tolist())
  trial_duration_hist = np.array(df['Trace End'].tolist()) - np.array(df['CS On'].tolist())

  duration_hist_plot(cs_duration_hist, trace_duration_hist, trial_duration_hist, FIGURE_SAVE_PATH)

def epoch_hist(df, session_obj):
  FIGURE_SAVE_PATH = session_obj.figure_path
  import seaborn as sns
  f, axarr = plt.subplots(2, 2, figsize=(20, 10))

  # calculate epoch lengths
  df['fixation-success_trial-start'] = df['Fixation Success'] - df['Start Trial']
  df['cs-off_cs-on'] = df['Trace Start'] - df['CS On']
  df['trace-end_trace_on'] = df['Trace End'] - df['Trace Start']
  df['reward_trace-end'] = df['Reward Trigger'] - df['Outcome Start']
  df['airpuff_trace-end'] = df['Airpuff Trigger'] - df['Outcome Start']

  # valence dataframes
  df_large_positive = df[df['valence'] == 1]
  df_small_positive = df[df['valence'] == 0.5]
  df_neutral = df[df['valence'] == 0]
  df_small_negative = df[df['valence'] == -0.5]
  df_large_negative = df[df['valence'] == -1]

  df_large_positive['fixation-success_trial-start'].hist(bins=20, color='blue', ax=axarr[0][0], label='large positive', alpha=0.5)
  df_small_positive['fixation-success_trial-start'].hist(bins=20, color='lightblue', ax=axarr[0][0], label='small positive', alpha=0.5)
  if len(df_neutral) > 0:
    df_neutral['fixation-success_trial-start'].hist(bins=20, color='grey', ax=axarr[0][0], label='neutral', alpha=0.5)
  df_small_negative['fixation-success_trial-start'].hist(bins=20, color='lightpink', ax=axarr[0][0], label='small negative', alpha=0.5)
  df_large_negative['fixation-success_trial-start'].hist(bins=20, color='red', ax=axarr[0][0], label='large negative', alpha=0.5)
  axarr[0][0].legend(loc='upper right')

  df_large_positive['cs-off_cs-on'].hist(bins=20, color='blue', ax=axarr[1][0], label='large positive', alpha=0.5)
  df_small_positive['cs-off_cs-on'].hist(bins=20, color='lightblue', ax=axarr[1][0], label='small positive', alpha=0.5)
  if len(df_neutral) > 0:
    df_neutral['cs-off_cs-on'].hist(bins=20, color='grey', ax=axarr[1][0], label='neutral', alpha=0.5)
  df_small_negative['cs-off_cs-on'].hist(bins=20, color='lightpink', ax=axarr[1][0], label='small negative', alpha=0.5)
  df_large_negative['cs-off_cs-on'].hist(bins=20, color='red', ax=axarr[1][0], label='large negative', alpha=0.5)
  axarr[1][0].legend(loc='upper right')

  df_large_positive['trace-end_trace_on'].hist(bins=20, color='blue', ax=axarr[0][1], label='large positive', alpha=0.5)
  df_small_positive['trace-end_trace_on'].hist(bins=20, color='lightblue', ax=axarr[0][1], label='small positive', alpha=0.5)
  if len(df_neutral) > 0:
    df_neutral['trace-end_trace_on'].hist(bins=20, color='grey', ax=axarr[0][1], label='neutral', alpha=0.5)
  df_large_negative['trace-end_trace_on'].hist(bins=20, color='lightpink', ax=axarr[0][1], label='small negative', alpha=0.5)
  df_large_negative['trace-end_trace_on'].hist(bins=20, color='red', ax=axarr[0][1], label='large negative', alpha=0.5)
  axarr[0][1].legend(loc='upper right')

  df_large_positive['reward_trace-end'].hist(bins=20, color='blue', ax=axarr[1][1], label='large positive', alpha=0.5)
  df_small_positive['reward_trace-end'].hist(bins=20, color='lightblue', ax=axarr[1][1], label='small positive', alpha=0.5)
  if len(df_neutral) > 0:
    df_neutral['reward_trace-end'].hist(bins=20, color='grey', ax=axarr[1][1], label='neutral', alpha=0.5)
  df_small_negative['airpuff_trace-end'].hist(bins=20, color='lightpink', ax=axarr[1][1], label='small negative', alpha=0.5)
  df_large_negative['airpuff_trace-end'].hist(bins=20, color='red', ax=axarr[1][1], label='large negative', alpha=0.5)
  axarr[1][1].legend(loc='upper right')

  axarr[0][0].set_title('Fixation Success - Trial Start')
  axarr[1][0].set_title('Trace Start - CS On')
  axarr[0][1].set_title('Trace Off - Trace On')
  axarr[1][1].set_title('Reward Trigger - Trace End')
  figure_name = 'epoch_hist.png'
  img_save_path = os.path.join(FIGURE_SAVE_PATH, figure_name)
	
  f.tight_layout()
  f.savefig(img_save_path, dpi=150, bbox_inches='tight', pad_inches = 0.1, transparent=True)
  print('  {} saved.'.format(figure_name))
  plt.show()

  # f, ax = plt.subplots(1, 1, figsize=(20, 10))
  # num_samples = len(sorted(df['date'].unique()))
  # colors = sns.color_palette("hls", num_samples)
  # for d_index, date in enumerate(sorted(df['date'].unique(), reverse=True)):
  #   session_df_date = df[df['date'] == date]
  #   session_df_date['trace-end_trace_on'].hist(bins=30, color=colors[d_index], ax=ax, label=date, alpha=0.5)
  #   plt.title('Trace Off - Trace On')
  #   ax.set_xlabel('Epoch Length (ms)')
  #   ax.set_ylabel('Trial Count')
  #   plt.legend()