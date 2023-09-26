import os
import numpy as np
from itertools import combinations
from matplotlib import pyplot as plt

def update_choice_matrix(choice_matrix, valences, df):
  for v_1_index, valence_1 in enumerate(valences):
    for v_2_index, valence_2 in enumerate(valences):
      if v_1_index == v_2_index:
        choice_matrix[v_1_index][v_2_index] = np.nan
        continue
      df_valence_1 = df[(df['valence_1'] == valence_1) &
                            (df['valence_2'] == valence_2) & 
                            (df['valence'] == valence_1)]
      df_valence_2 = df[(df['valence_1'] == valence_1) &
                            (df['valence_2'] == valence_2) &
                            (df['valence'] == valence_2)]
      if len(df_valence_1) + len(df_valence_2) == 0:
        # empty matrix
        choice_matrix[v_1_index][v_2_index] = np.nan
      else:
        proportion_val_1 = len(df_valence_1)/(len(df_valence_1)+len(df_valence_2))
        choice_matrix[v_1_index][v_2_index] = proportion_val_1
  return choice_matrix

def generate_ideal_matrix(choice_matrix, valences):
  for v_1_index, valence_1 in enumerate(valences):
    for v_2_index, valence_2 in enumerate(valences):
      if v_1_index == v_2_index:
        choice_matrix[v_1_index][v_2_index] = np.nan
        continue
      else:
        if valence_1 > valence_2:
          choice_matrix[v_1_index][v_2_index] = 1
        else:
          choice_matrix[v_1_index][v_2_index] = 0

  return choice_matrix

def plot_heatmap_choice_valence(df, session_obj, ):
  '''heat map of choice trials'''
  FIGURE_SAVE_PATH = session_obj.figure_path
  # only get choice trials that are non-zero valence
  session_choice = df[(df['choice_trial'] == 1) & \
                      (df['fractal_count_in_block'] > 10)]
  session_choice = session_choice[(session_choice['valence_1'] != 0)]
  # get unique conditions
  f, axarr = plt.subplots(2,2)
  cmap = plt.cm.RdYlGn
  cmap.set_bad(color='white')  
  conditions = list(sorted(df['condition'].unique()))
  # get unique valences
  valences = sorted(df['valence_1'].unique(), reverse=True)
  for index, plot in enumerate(range(len(axarr.flat))):
    # empty matrix of zeros
    choice_matrix = np.zeros((len(valences), len(valences)))
    # condition specific
    if index < 2:
      condition = conditions[index]
      df_cond = df[df['condition'] == condition]
      choice_matrix = update_choice_matrix(choice_matrix, valences, df_cond)
      title = 'Pre-Switch' if condition == 1 else 'Post-Switch'
    # all conditions
    elif index == 2:
      choice_matrix = update_choice_matrix(choice_matrix, valences, df)
      title = 'Entire Session'
    # ideal behavior
    elif index == 3:
      choice_matrix = generate_ideal_matrix(choice_matrix, valences)
      title = 'Ideal Behavior'

    # plot matrix
    row = 0
    if index >= 2:
      row = 1
    col = index % 2
    cbar = axarr[row][col].figure.colorbar(axarr[row][col].imshow(choice_matrix.T, cmap=cmap, vmin=0, vmax=1))
    axarr[row][col].set_title(title, fontsize=14)

    # legend for color bar
    axarr[row][col].set_xlabel('Stimulus L', fontsize=12)
    axarr[row][col].set_ylabel('Stimulus R', fontsize=12)
    axarr[row][col].set_xticks(range(len(valences)))
    axarr[row][col].set_yticks(range(len(valences)))
    valence_labels = [session_obj.valence_labels[valence] for valence in valences]
    axarr[row][col].set_xticklabels(valence_labels, fontsize=8)
    axarr[row][col].set_yticklabels(valence_labels, fontsize=8)

  f.suptitle('         Probability of Choosing Stimulus L', fontsize=18)
  f.tight_layout()
  plot_title = 'choice_heatmap.svg'
  img_save_path = os.path.join(FIGURE_SAVE_PATH, plot_title)
  f.savefig(img_save_path, dpi=150, bbox_inches='tight', pad_inches = 0.1, transparent=True)
  print(f'  {plot_title} saved.')

def plot_avg_choice_valence(session_df_correct, session_obj):
  """
  plots the probability of choosing the fractal with the higher valence

  Parameters
  ----------
  session_df_correct : pandas dataframe
      dataframe of session data
  session_obj : Session object
      session object

  Returns
  -------
  None.
  """
  FIGURE_SAVE_PATH = session_obj.figure_path
  session_df_choice = session_df_correct[session_df_correct['choice_trial'] == 1]
  # session_df_choice = session_df_choice[session_df_choice['valence'] != 0]
  # see how many times the monkey chose the fractal with the higher valence
  correct_choice_trials = [1 if session_df_choice['valence'].iloc[i] == np.max([session_df_choice['valence_1'].iloc[i], session_df_choice['valence_2'].iloc[i]], axis=0) else 0 for i in range(len(session_df_choice))]
  session_df_choice['correct_choice'] = correct_choice_trials
  # plot running average of correct choice
  f, ax = plt.subplots(1, 1, figsize=(11, 4))
  x = np.arange(len(session_df_choice))
  y = np.convolve(session_df_choice['correct_choice'], np.ones((3,))/3, mode='same')    
  plt.plot(x, y, label='Probability of choosing higher valence', linewidth=2, alpha=0.75)
  plt.xlabel('Choice Trial #', fontsize=8)
  plt.ylabel('Prob(higher valence)', fontsize=8)
  plt.xticks(fontsize=8)
  plt.yticks(fontsize=8)
  # plot chosen valence
  color_dict = session_obj.valence_colors
  scatter_colors_1 = [color_dict[valence] for valence in session_df_choice['valence']]
  ax.scatter(x, [1.1]*len(x), s=8, c=scatter_colors_1)
  # make column for not chosen valence
  valence_other = [session_df_choice['valence_2'].iloc[i] if session_df_choice['valence_1'].iloc[i] == session_df_choice['valence'].iloc[i] else session_df_choice['valence_1'].iloc[i] for i in range(len(session_df_choice))]
  session_df_choice['not_chosen_fractal'] = valence_other
  scatter_colors_2 = [color_dict[valence] for valence in session_df_choice['not_chosen_fractal']]
  ax.scatter(x, [-0.1]*len(x), s=2, c=scatter_colors_2)
  block_change = np.where(np.diff(session_df_choice['condition']) != 0)[0]
  # plot vertical lines for block change
  for b_index, block in enumerate(block_change):
    if b_index == 0:
      ax.axvline(block-0.55, color='white', linestyle='--', linewidth=1, label='Block Change')
    else:
      ax.axvline(block-0.55, color='white', linestyle='--', linewidth=1)
  ax.set_ylim([-0.25, 1.25])
  # only ticks at 0, 0.5, 1
  ax.set_yticks([0, 0.5, 1])
  # add second y axis
  ax2 = ax.twinx()
  ax2.set_ylim([-0.25, 1.25])
  ax2.set_yticks([-0.1, 1.1])
  ax2.set_yticklabels(['Not Chosen', 'Chosen'], fontsize=8)
  # probability of choosing fractal 1
  chose_fractal_1 = [1 if session_df_choice['valence'].iloc[i] == session_df_choice['valence_1'].iloc[i] else 0 for i in range(len(session_df_choice))]
  y = np.convolve(chose_fractal_1, np.ones((3,))/3, mode='same')
  ax.plot(x, y, color='grey', linestyle='--', linewidth=0.5, alpha=0.5, label='Probability of choosing left fractal')
  # legend outside of plot with no border
  ax.legend(bbox_to_anchor=(1.1, 1.5), loc='upper right', fontsize=6, frameon=False)
  plt.tight_layout()

  # save figure
  plot_title = 'session_choice_valence.svg'
  img_save_path = os.path.join(FIGURE_SAVE_PATH, plot_title)
  # transparent background
  f.savefig(img_save_path, dpi=150, bbox_inches='tight', pad_inches = 0.1, transparent=True)
  print(f'  {plot_title} saved.')
  plt.show()