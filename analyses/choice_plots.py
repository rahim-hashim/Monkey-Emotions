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

def plot_heatmap_choice_valence(df, session_obj):
  '''heat map of choice trials'''
  trial_count_block = [45, 90, 125, 200]
  FIGURE_SAVE_PATH = session_obj.figure_path
  # only get choice trials that are non-zero valence
  for c_index, count in enumerate(trial_count_block):
    if c_index == 0:
      df_choice = df[(df['choice_trial'] == 1) & \
                    (df['correct_trial_in_block'] < count)
                  ]
      descriptor = f'First {count} trials in block'
    elif c_index < len(trial_count_block) - 1:
      df_choice = df[(df['choice_trial'] == 1) & \
                    (df['correct_trial_in_block'] >= trial_count_block[c_index-1]) & \
                    (df['correct_trial_in_block'] < count)
                  ]
      descriptor = f'Trials {trial_count_block[c_index-1]} - {count} in block'
    else:
      df_choice = df[(df['choice_trial'] == 1)]
      descriptor = 'All trials in block'
    print(descriptor)
    if len(df_choice) == 0:
      print('No choice trials to plot.')
      return
    else:
      print(f'  Num Trials: {len(df_choice)}')
    # get unique conditions
    conditions = list(sorted(df_choice['condition'].unique()))
    f, axarr = plt.subplots(1, len(conditions)+2, figsize=(15, 5))
    if c_index == 0:
      f.suptitle(f'Probability of Choosing Stimulus L/U (< {trial_count_block[c_index]} trials in block)', fontsize=18)
    elif c_index < len(trial_count_block) - 1:
      f.suptitle(f'Probability of Choosing Stimulus L/U ({trial_count_block[c_index-1]} - {count} trials in block)', fontsize=18)
    else:
      f.suptitle(f'Probability of Choosing Stimulus L/U (all trials)', fontsize=18)
    cmap = plt.cm.RdYlGn
    cmap.set_bad(color='white')  
    # get unique valences
    valences = sorted(df_choice['valence_1'].unique(), reverse=True)
    for index, plot in enumerate(range(len(axarr.flat))):
      # empty matrix of zeros
      choice_matrix = np.zeros((len(valences), len(valences)))
      # condition specific
      if index < len(conditions):
        condition = conditions[index]
        df_cond = df_choice[df_choice['condition'] == condition]
        choice_matrix = update_choice_matrix(choice_matrix, valences, df_cond)
        title = f'Block {condition}'
      # all conditions
      elif index == len(conditions):
        choice_matrix = update_choice_matrix(choice_matrix, valences, df_choice)
        title = 'Entire Session'
      # ideal behavior
      elif index == len(conditions)+1:
        choice_matrix = generate_ideal_matrix(choice_matrix, valences)
        title = 'Ideal Behavior'

      # plot matrix
      row = 0
      col = index
      if index == len(conditions)+1:
        cbar = axarr[col].figure.colorbar(axarr[col].imshow(choice_matrix.T, cmap=cmap, vmin=0, vmax=1))
      else:
        axarr[col].imshow(choice_matrix.T, cmap=cmap, vmin=0, vmax=1)
      axarr[col].set_title(title, fontsize=14)

      # legend for color bar
      if col == 0:
        axarr[col].set_xlabel('Stimulus L/U', fontsize=12)
        axarr[col].set_ylabel('Stimulus R/D', fontsize=12)
      axarr[col].set_xticks(range(len(valences)))
      axarr[col].set_yticks(range(len(valences)))
      valence_labels = [session_obj.valence_labels[valence] for valence in valences]
      axarr[col].set_xticklabels(valence_labels, fontsize=8)
      axarr[col].set_yticklabels(valence_labels, fontsize=8)
    plot_title = f'choice_heatmap_{count}.svg'
    img_save_path = os.path.join(FIGURE_SAVE_PATH, plot_title)
    plt.tight_layout()
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
  for date in session_df_correct['date'].unique():
    session_df_date = session_df_correct[session_df_correct['date'] == date]
    session_df_choice = session_df_date[session_df_date['choice_trial'] == 1]
    if len(session_df_choice) == 0:
      print('No choice trials to plot.')
      return
    # session_df_choice = session_df_choice[session_df_choice['valence'] != 0]
    # see how many times the monkey chose the fractal with the higher valence
    correct_choice_trials = [1 if session_df_choice['valence'].iloc[i] == \
              np.max([session_df_choice['valence_1'].iloc[i], session_df_choice['valence_2'].iloc[i]], axis=0) \
              else 0 for i in range(len(session_df_choice))]
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
    valence_other = [
      session_df_choice['valence_2'].iloc[i] \
      if session_df_choice['valence_1'].iloc[i] == session_df_choice['valence'].iloc[i] \
      else session_df_choice['valence_1'].iloc[i] \
      for i in range(len(session_df_choice))
    ]
    session_df_choice['not_chosen_fractal'] = valence_other
    scatter_colors_2 = [color_dict[valence] for valence in session_df_choice['not_chosen_fractal']]
    ax.scatter(x, [-0.1]*len(x), s=2, c=scatter_colors_2)
    block_change = np.where(np.diff(session_df_choice['block']) != 0)[0]
    # plot vertical lines for block change
    for b_index, block in enumerate(block_change):
      if b_index == 0:
        ax.axvline(block-0.55, color='yellow', linestyle='--', linewidth=1, label='Block Change')
      else:
        ax.axvline(block-0.55, color='yellow', linestyle='--', linewidth=1)
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
    ax.set_title(f'{date}', fontsize=12)
    plt.tight_layout()
    if FIGURE_SAVE_PATH:
      # save figure
      plot_title = f'df_choice_valence_{date}.svg'
      img_save_path = os.path.join(FIGURE_SAVE_PATH, plot_title)
      # transparent background
      f.savefig(img_save_path, dpi=150, bbox_inches='tight', pad_inches = 0.1, transparent=True)
      print(f'  {plot_title} saved.')
    plt.show()