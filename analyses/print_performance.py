import numpy as np
from collections import defaultdict


def print_performance(session_df, select_valences=False):
  """
  Print performance metrics for a session

  Parameters
  ----------
  session_df : pandas dataframe
    dataframe of session data

  Returns
  -------
  """
  print('Session Performance')
  # CS On is not NaN (otherwise errors not relevant)
  session_df_cs_on = session_df[session_df['CS On'].notna()]
  # Reinforcement Trials only
  session_df_reinforcement = session_df_cs_on[session_df_cs_on['reinforcement_trial'] == 1]
  correct_reinforcement_trials = session_df_reinforcement[session_df_reinforcement['correct'] == 1]
  # Reinforcement Trial Calculation
  print(f' Percent successful reinforcement trials: {round(len(correct_reinforcement_trials)/len(session_df_reinforcement), 3)} ({len(correct_reinforcement_trials)}/{len(session_df_reinforcement)})')

  # Valence Calculation
  all_valences = sorted(session_df['valence_1'].unique(), reverse=True)
  for v_index, valence in enumerate(all_valences):
    # Valence 1 -- due to task code structure but I will fix
    session_df_valence = session_df_reinforcement[session_df_reinforcement['valence_1'] == valence]
    # Filter for correct trials
    session_df_valence_correct = session_df_valence[session_df_valence['correct'] == 1]
    # Calculate percent correct
    try:
      percent_correct = round(len(session_df_valence_correct)/len(session_df_valence), 3)
      # Print percent correct by valence
      print(f'   Valence {valence}: {percent_correct} ({len(session_df_valence_correct)}/{len(session_df_valence)})')
    except:
      pass

  # Choice Trials only
  session_df_choice = session_df_cs_on[session_df_cs_on['choice_trial'] == 1]
  if select_valences == True:
    session_df_choice = session_df_choice.loc[
      (session_df_choice['valence_1'].isin([1, 0.5, -0.5, -1])) &
      (session_df_choice['valence_2'].isin([1, 0.5, -0.5, -1]))
    ]
  correct_choice_trials = session_df_choice[session_df_choice['correct'] == 1]
  # correct_choice_trials = correct_choice_trials[correct_choice_trials['fractal_count_in_block'] > 10]

  # Choice Trial Calculation
  print(f' Percent successful choice trials: {round(len(correct_choice_trials)/len(session_df_choice), 3)} ({len(correct_choice_trials)}/{len(session_df_choice)})')
  valence_pairs = list(map(sorted, zip(session_df_choice['valence_1'], session_df_choice['valence_2'])))
  # create dictionary of valence pairs and number of times the trial was correct
  valence_pair_dict = defaultdict(list)
  for v_index, valence_pair in enumerate(valence_pairs):
    valence_pair_dict[str(valence_pair)].append(session_df_choice['correct'].iloc[v_index])
  # calculate percent correct for each valence pair, ordering dictionary by percent correct
  valence_pair_percent_correct = {}
  for valence_pair in valence_pair_dict:
    valence_pair_percent_correct[valence_pair] = round(np.mean(valence_pair_dict[valence_pair]), 3)
  valence_pair_percent_correct = {k: v for k, v in sorted(valence_pair_percent_correct.items(), key=lambda item: item[1], reverse=True)}
  # print percent correct by valence pair
  for valence_pair in valence_pair_percent_correct:
    print(f'   Valence Pair {valence_pair}: {valence_pair_percent_correct[valence_pair]} ({sum(valence_pair_dict[valence_pair])}/{len(valence_pair_dict[valence_pair])})')

  # Left-Right Choice Calculation
  session_df_choice_correct = session_df_choice[session_df_choice['correct'] == 1]
  # session_df_choice_correct = session_df_choice[session_df_choice['fractal_count_in_block'] > 5]
  session_df_choice_left = session_df_choice_correct[session_df_choice_correct['valence'] == session_df_choice_correct['valence_1']]
  session_df_choice_right = session_df_choice_correct[session_df_choice_correct['valence'] == session_df_choice_correct['valence_2']]
  print(f' Percent left choice trials: {round(len(session_df_choice_left)/len(session_df_choice_correct), 3)} ({len(session_df_choice_left)}/{len(session_df_choice_correct)})')

  # Higher Valence Choice Calculation
  chosen_valence = session_df_choice_correct['valence']
  valence_1 = session_df_choice_correct['valence_1']
  valence_2 = session_df_choice_correct['valence_2']
  valence_best = [np.max([valence_1.iloc[i], valence_2.iloc[i]]) for i in range(len(valence_1))]
  higher_valence = [1 if chosen_valence.iloc[i] == valence_best[i] else 0 for i in range(len(chosen_valence))]
  print(f' Percent higher valence choice trials: {round(np.mean(higher_valence), 3)} ({sum(higher_valence)}/{len(higher_valence)})')
  
  # For each valence pair, see how many times the monkey chose the fractal with the higher valence
  valence_pairs = list(map(sorted, zip(session_df_choice_correct['valence_1'], session_df_choice_correct['valence_2'])))
  # create dictionary of valence pairs and number of times the trial was correct
  valence_pair_dict = defaultdict(list)
  for v_index, valence_pair in enumerate(valence_pairs):
    valence_pair_dict[str(valence_pair)].append(higher_valence[v_index])
  # calculate percent correct for each valence pair, ordering dictionary by percent correct
  valence_pair_percent_correct = {}
  for valence_pair in valence_pair_dict:
    valence_pair_percent_correct[valence_pair] = round(np.mean(valence_pair_dict[valence_pair]), 3)
  valence_pair_percent_correct = {k: v for k, v in sorted(valence_pair_percent_correct.items(), key=lambda item: item[1], reverse=True)}
  # print percent correct by valence pair
  for valence_pair in valence_pair_percent_correct:
    print(f'   Valence Pair {valence_pair}: {valence_pair_percent_correct[valence_pair]} ({sum(valence_pair_dict[valence_pair])}/{len(valence_pair_dict[valence_pair])})')

def reaction_time_choice(session_df):
  """
  Print reaction time for each valence pair

  Parameters
  ----------
  session_df : pandas dataframe
    dataframe of session data

  Returns
  -------
  """
  print('Reaction Time on Choice Trials')
  session_df_correct = session_df[session_df['correct'] == 1]
  session_df_choice = session_df_correct[session_df_correct['choice_trial'] == 1]
  # session_df_choice = session_df_correct[session_df_correct['fractal_count_in_block'] > 5]
  session_df_choice['Reaction Time'] = session_df_choice['Trace Start'] - session_df_choice['Fixation Off']
  for valence in sorted(session_df_choice['valence'].unique()):
    session_df_choice_valence = session_df_choice[session_df_choice['valence'] == valence]
    print(' Valence: {}'.format(valence))
    print('   Reaction Time: {}'.format(round(np.mean(session_df_choice_valence['Reaction Time']), 3)))
    for v_not_chosen in sorted(session_df_choice_valence['valence_not_chosen'].unique()):
      print('   Valence Not Chosen: {}'.format(v_not_chosen), 
            round(np.mean(session_df_choice_valence[session_df_choice_valence['valence_not_chosen'] == v_not_chosen]['Reaction Time']), 3),
            '(' + str(len(session_df_choice_valence[session_df_choice_valence['valence_not_chosen'] == v_not_chosen]['Reaction Time']))+')')
