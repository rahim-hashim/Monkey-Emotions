import re
import os
import sys
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from textwrap import indent
from pprint import pprint, pformat
from pprint import pprint
from collections import defaultdict
# Custom modules
from config import h5_helper
from config.image_diff import image_diff
# Custom classes
from classes.Session import Session

def create_empty_matrix(num_rows, num_cols):
  """Creates an empty string matrix with the specified number of rows and columns."""
  empty_matrix = np.empty((num_rows, num_cols), dtype=object)
  empty_matrix[:] = ''
  return empty_matrix

def is_empty_row(matrix):
  """Returns the index of the first empty row in the matrix, or -1 if there are no empty rows."""
  empty_rows = []
  for i in range(len(matrix)):
    if all(x == '' for x in matrix[i]):
      empty_rows.append(i+1)
  return empty_rows

def add_external_cameras(session_df, session_obj):
  print('Checking for external camera files...')
  video_path = session_obj.video_path
  num_cameras = 3
  if os.path.exists(video_path):
    print('  External camera path: {}'.format(video_path))
    video_folders = os.listdir(video_path)
    if video_folders:
      print(f'  {len(session_df)} trials in session')
      print(f'  {len(video_folders)} external camera folders found')
      # initialize camera matrix
      cam_trials_matrix = create_empty_matrix(len(session_df), num_cameras)
      cam_name_dict = defaultdict(int)
      for v_index, video_folder in enumerate(tqdm(sorted(video_folders))):
        trial_num = video_folder.split('_')[2]
        for v_index, video_file in enumerate(os.listdir(os.path.join(video_path, video_folder))):
          cam_name = video_file.split('-')[0]
          # see if trial number is '0' or any number of zeros
          if int(trial_num) == 1:
          # if re.match("^0+$", trial_num):
            # first API call was just initialization
            cam_name_dict[cam_name] = v_index
            # continue
          cam_num = cam_name_dict[cam_name]
          cam_trials_matrix[int(trial_num)-1, cam_num] = os.path.join(video_path, video_folder, video_file)
      for cam in cam_name_dict.keys():
        session_df[cam] = cam_trials_matrix[:, cam_name_dict[cam]]
      print(cam_name_dict, cam_trials_matrix[0, :])
      # see which row has all empty strings
      empty_rows = is_empty_row(cam_trials_matrix)
      if empty_rows:
        print('  Missing all video files for trial_num: {}'.format(empty_rows))
  else:
    print('  No external camera files found {}'.format(video_path))
  return session_df, session_obj

def preprocess_data(path_obj, start_date, end_date, monkey_input, experiment_name, reprocess_data, save_df, combine_dates):
  # preprocess data
  if reprocess_data:
    ml_config, trial_record, session_df, session_obj, error_dict, behavioral_code_dict = \
      h5_helper.h5_to_df(path_obj, start_date, end_date, monkey_input, save_df)
  # unpickle preprocessed data
  else:
    print('\nFiles uploaded from processed folder\n')
    all_selected_dates = h5_helper.date_selector(start_date, end_date)
    target_dir = os.listdir(path_obj.target_path)
    pkl_files_selected, dates_array = h5_helper.file_selector(target_dir, all_selected_dates, monkey_input)
    print('Pickled Files:')
    pprint(pkl_files_selected, indent=2)
    for f_index, f in enumerate(pkl_files_selected):
      target_pickle = os.path.join(path_obj.target_path, f)
      if os.path.exists(target_pickle):
        session_dict = pd.read_pickle(target_pickle)
        if f_index == 0:
          session_df = session_dict['data_frame']
          error_dict = session_dict['error_dict']
          behavioral_code_dict = session_dict['behavioral_code_dict']
        else:
          session_df_new = session_dict['data_frame']
          session_df = pd.concat([session_df, session_df_new], ignore_index=True)
          error_dict = session_dict['error_dict']
          behavioral_code_dict = session_dict['behavioral_code_dict']    
      else:
        print('\nPickled files missing. Reprocess or check data.')
        sys.exit()
      # session_obj contains session metadata
      session_obj = Session(session_df, monkey_input, experiment_name, behavioral_code_dict)
    
  # Save path for figures
  FIGURE_SAVE_PATH = image_diff(session_df,
                                session_obj,
                                path_obj,
                                combine_dates=combine_dates) # True will combine all dates into analysis
  session_obj.save_paths(path_obj.target_path, 
                          path_obj.tracker_path, 
                          path_obj.video_path,
                          FIGURE_SAVE_PATH)  
  
  # add external camera data
  # session_df, session_obj = add_external_cameras(session_df, session_obj)  
  print(indent(pformat(session_df.columns), '  '))

  return session_df, session_obj, error_dict, behavioral_code_dict