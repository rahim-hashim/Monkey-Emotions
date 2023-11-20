import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def valence_panels(session_df):
  session_df_reinforcement = session_df[session_df['reinforcement_trial'] == 1]
  for valence in sorted(session_df_reinforcement['valence'].unique()):
    session_valence = session_df_reinforcement[session_df_reinforcement['valence'] == valence]
    # iterate through trials
    f, axarr = plt.subplots(3, 4, figsize=(10, 5))
    # empty matrix of zeros
    lick_matrix_cs = np.zeros((len(session_valence), 350))
    lick_matrix_trace = np.zeros((len(session_valence), 1000))
    lick_matrix_outcome = np.zeros((len(session_valence), 1000))
    blink_matrix_cs = np.zeros((len(session_valence), 350))
    blink_matrix_trace = np.zeros((len(session_valence), 1000))
    blink_matrix_outcome = np.zeros((len(session_valence), 1000))
    iter = 0
    for index, row in session_valence.iterrows():
      # get lick/blink raster
      lick_raster = row['lick_raster']
      blink_raster = row['blink_raster']
      # get cs on/off time
      cs_on = row['CS On']
      cs_off = row['Trace Start']
      lick_raster_cs = lick_raster[cs_on:cs_off][-350:]
      blink_raster_cs = blink_raster[cs_on:cs_off][-350:]
      # replace zeros in blink matrix with blink raster
      lick_matrix_cs[iter, :len(lick_raster_cs)] = lick_raster_cs
      blink_matrix_cs[iter, :len(blink_raster_cs)] = blink_raster_cs
      # get trace end/trial end time
      if valence > 0:
        trace_end = row['Reward Trigger']
      elif valence < 0:
        trace_end = row['Airpuff Trigger']
      else:
        trace_end = row['Trace End']
      trial_end = row['End Trial']
      lick_trace = lick_raster[cs_off:trace_end][-1000:]
      blink_trace = blink_raster[cs_off:trace_end][-1000:]
      # replace zeros in blink matrix with blink trace
      lick_matrix_trace[iter, :len(lick_trace)] = lick_trace
      blink_matrix_trace[iter, :len(blink_trace)] = blink_trace
      # get outcome time
      lick_raster_outcome = lick_raster[trace_end:trial_end][:1000]
      blink_raster_outcome = blink_raster[trace_end:trial_end][:1000]
      # replace zeros in blink matrix with blink outcome
      lick_matrix_outcome[iter, :len(lick_raster_outcome)] = lick_raster_outcome
      blink_matrix_outcome[iter, :len(blink_raster_outcome)] = blink_raster_outcome
      iter += 1
    axarr[0][0].imshow(lick_matrix_cs, aspect='auto')
    axarr[1][0].imshow(lick_matrix_trace, aspect='auto')
    axarr[2][0].imshow(lick_matrix_outcome, aspect='auto')
    axarr[0][2].imshow(blink_matrix_cs, aspect='auto')
    axarr[1][2].imshow(blink_matrix_trace, aspect='auto')
    axarr[2][2].imshow(blink_matrix_outcome, aspect='auto')
    # get column average of matrix
    ## cs
    lick_matrix_cs_avg = np.mean(lick_matrix_cs, axis=0)
    blink_matrix_cs_avg = np.mean(blink_matrix_cs, axis=0)
    axarr[0][1].plot(lick_matrix_cs_avg)
    axarr[0][1].set_ylim([0, 1])
    axarr[0][3].plot(blink_matrix_cs_avg)
    axarr[0][3].set_ylim([0, 1])
    ## trace
    lick_matrix_trace_avg = np.mean(lick_matrix_trace, axis=0)
    blink_matrix_trace_avg = np.mean(blink_matrix_trace, axis=0)
    axarr[1][1].plot(lick_matrix_trace_avg)
    axarr[1][1].set_ylim([0, 1])
    axarr[1][3].plot(blink_matrix_trace_avg)
    axarr[1][3].set_ylim([0, 1])
    ## outcome
    lick_matrix_outcome_avg = np.mean(lick_matrix_outcome, axis=0)
    blink_matrix_outcome_avg = np.mean(blink_matrix_outcome, axis=0)
    axarr[2][1].plot(lick_matrix_outcome_avg)
    axarr[2][1].set_ylim([0, 1])
    axarr[2][3].plot(blink_matrix_outcome_avg)
    axarr[2][3].set_ylim([0, 1])
    # set x labels
    axarr[0][0].set_title('Lick Raster')
    axarr[0][1].set_title('Lick Average')
    axarr[0][2].set_title('Blink Raster')
    axarr[0][3].set_title('Blink Average')
    # set y labels
    axarr[0][0].set_ylabel('CS')
    axarr[1][0].set_ylabel('Trace')
    axarr[2][0].set_ylabel('Outcome')
    # use super title to set title over all subplots 
    f.suptitle('Valence: {}'.format(valence), fontsize=22)
    # reduce height of title
    f.tight_layout()
    plt.show()
    print(f'Valence: {valence}')
    print(f'  Lick CS: {round(np.mean(lick_matrix_cs_avg), 3)}')
    print(f'  Lick Trace: {round(np.mean(lick_matrix_trace_avg), 3)}')
    print(f'  Lick Outcome: {round(np.mean(lick_matrix_outcome_avg), 3)}')