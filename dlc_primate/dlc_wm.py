import os
import cv2
import sys
import pickle
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
from pathlib import Path
sys.path.append(os.path.join(os.getcwd(), '..'))
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=Warning)
from utilities.save_functions import unpickle_spikeglx
from classes.Session_Path import SessionPath
from dlc_utils import dlc_config
from config import preprocess_helper

# For ModelZoo reference: https://colab.research.google.com/github/danbider/lightning-pose/blob/main/scripts/litpose_training_demo.ipynb#scrollTo=VFQ5U_hZVB4M
import deeplabcut

ROOT = '/Users/rahimhashim/Google Drive/My Drive/Columbia/Salzman/Monkey-Training/'
EXPERIMENT = 'rhAirpuff'
TASK = 'Probabilistic_Reward_Airpuff_Choice' # Probabalistic_Airpuff_4x2 | Probabilistic_Reward_Airpuff_5x2 | Probabilistic_Reward_Airpuff_Choice

path_obj = SessionPath(ROOT, EXPERIMENT, TASK)

# Specifying date/monkey/task
start_date = '2023-08-23' #@param {type:"date"}
end_date = '2023-08-23' #@param {type:"date"}
monkey_input = 'Aragorn' #@param ['Aragorn', 'Gandalf', 'Rob', 'Test']
reprocess_data = False #@param {type:"boolean"}
save_df =  False #@param {type:"boolean"}
combine_dates =  True #@param {type:"boolean"}


# Parse data
session_df, session_obj, error_dict, behavioral_code_dict\
	= preprocess_helper.preprocess_data(path_obj,
                                        start_date,
                                        end_date,
                                        monkey_input,
                                        TASK,
                                        reprocess_data,
                                        save_df,
                                        combine_dates)


spikeglx_obj = unpickle_spikeglx(session_obj)

kwargs = {'spikeglx_obj': spikeglx_obj, 
          'session_obj': session_obj, 
          'trial_start': 0,
          'trial_end': len(session_obj.df),
          'epoch_start': 'Trace Start', 
          'epoch_end': 'Outcome Start', 
          'thread_flag': True}

# parse_wm_videos(**kwargs)

# DeepLabCut

camera_dict = {
  'e3v8360':'face_1', 
  'e3v83d6':'face_2',
  'e3v83ad':'body_1',
  'e3v831b':'body_2'
}

video_dir = os.path.join(os.getcwd(), '..', 'video', session_obj.monkey + '_' + session_obj.date)
dlc_video_path_dict = dlc_config.get_trial_video_list(video_dir, camera_dict)
import dlc_run
dlc_run.dlc_run(session_obj, video_dir, dlc_video_path_dict, camera_dict)

# # .mp4 or .avi etc.
# for camera in camera_dict.keys():
#   video_path_list = dlc_video_path_dict[camera]
#   body_part = camera_dict[camera]
#   project_name = f'{session_obj.date}_{session_obj.monkey}_{body_part}'
#   your_name = 'rahim'
#   if 'face' in body_part:
#     model2use = 'primate_face'
#   else:
#     model2use = 'full_macaque'

#   # Check if videos need to be downsampled
#   downsample_flag = dlc_config._dlc_check_for_downsample(video_path_list)

#   if downsample_flag:
#     video_path_list = dlc_config._dlc_downsample_videos(video_path_list)

#   videotype='.mp4'
#   # Create ModelZoo project
#   config_path, train_config_path = deeplabcut.create_pretrained_project(
#       project_name,
#       your_name,
#       video_path_list[0:1],
#       videotype=videotype,
#       model=model2use,
#       analyzevideo=True,
#       createlabeledvideo=True,
#       copy_videos=False, # must leave copy_videos=True
#   )

#   # Updating the configs within the config.yaml file
#   edits = {
#       'dotsize': 3,  # size of the dots!
#       'colormap': 'rainbow',  # any matplotlib colormap
#       'alphavalue': 0.2, # transparency of labels
#       'pcutoff': 0.5,  # the higher the more conservative the plotting!
#       'skeleton': 
#           # Right Eye
#           [['RightEye_Top', 'RightEye_Inner'], 
#           ['RightEye_Inner', 'RightEye_Bottom'],
#           ['RightEye_Outer', 'RightEye_Bottom'],
#           ['RightEye_Top', 'RightEye_Outer'], 
#           # Left Eye
#           ['LeftEye_Top', 'LeftEye_Inner'],
#           ['LeftEye_Inner', 'LeftEye_Bottom'],
#           ['LeftEye_Outer', 'LeftEye_Bottom'],
#           ['LeftEye_Top', 'LeftEye_Outer'],
#           #  # Top of Head Counter-Clockwise to Lip
#           #  ['HeadTop_Mid', 'OutlineRight_Mouth'],
#           #  ['OutlineRight_Mouth', 'RightNostrils_Bottom'],
#           #  ['RightNostrils_Bottom', 'UpperLip_Centre'],
#           #  # Lip Counter-Clockwise to Top of Head
#           #  ['UpperLip_Centre', 'OutlineLeft_Mouth'],
#           #  ['OutlineLeft_Mouth', 'LeftNostrils_Bottom'],
#           #  ['LeftNostrils_Bottom', 'HeadTop_Mid'],
#           ],
#       'skeleton_color': 'white'
#   }
#   deeplabcut.auxiliaryfunctions.edit_config(config_path, edits)

#   # Adding new videos to the config.yaml file
#   deeplabcut.add_new_videos(config_path, video_path_list, copy_videos=False, 
#               coords=None, extract_frames=False
#   )

#   # Extract frames
#   deeplabcut.analyze_videos(config_path, video_path_list, 
#               videotype, save_as_csv=True
#   )

#   # Filter predictions
#   deeplabcut.filterpredictions(config_path, video_path_list, videotype=videotype)

#   # Create labeled video
#   deeplabcut.create_labeled_video(
#     config_path, video_path_list[:3], 
#     videotype, 
#     draw_skeleton=True, 
#     filtered=True,
#     trailpoints=5,
#   )

# # # Plot trajectories
# # deeplabcut.plot_trajectories(config_path, video_path_list[6:8], videotype, filtered=True)