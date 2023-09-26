import os
import cv2
import pickle
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
from pathlib import Path
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 

from dlc_utils.dlc_config import select_session, get_trial_video_list, clean_pretrained_project
from dlc_utils.dlc_downsample import check_for_downsample, downsample_videos

# For ModelZoo reference: https://colab.research.google.com/github/danbider/lightning-pose/blob/main/scripts/litpose_training_demo.ipynb#scrollTo=VFQ5U_hZVB4M
import deeplabcut

# Set paths
ROOT = '/Users/rahimhashim/Google Drive/My Drive/Columbia/Salzman/Monkey-Training/'
VIDEO_ROOT = os.path.join(ROOT, 'tasks', 'rhAirpuff', '8. Probabilistic_Reward_Airpuff_Choice', 'videos')
SESSION_ROOT = os.path.join(ROOT, 'data', 'processed', 'processed_Probabilistic_Reward_Airpuff_Choice')

# Select session
DATE = '230621'
MONKEY = 'Aragorn'

session_df = select_session(SESSION_ROOT, DATE, MONKEY, correct_only=True)
video_path_list = get_trial_video_list(session_df, VIDEO_ROOT, DATE, MONKEY)

project_name = f'{DATE}_{MONKEY}'
your_name = 'rahim'
# For other options, see: deeplabcut.create_project.modelzoo.Modeloptions
model2use = 'primate_face'
# .mp4 or .avi etc.
videotype = os.path.splitext(video_path_list[0])[-1].lstrip('.')

# Check if videos need to be downsampled
downsample_flag = check_for_downsample(video_path_list)

if downsample_flag:
	video_path_list = downsample_videos(video_path_list)

# Create ModelZoo project
config_path, train_config_path = deeplabcut.create_pretrained_project(
    project_name,
    your_name,
    video_path_list[0:1],
    videotype=videotype,
    model=model2use,
    analyzevideo=True,
    createlabeledvideo=True,
    copy_videos=False, # must leave copy_videos=True
)

config_path = '/Users/rahimhashim/Desktop/Monkey-Behavior/dlc_primate/myDLC_modelZoo-teamDLC-2023-06-26/config.yaml'
videotype='.mp4'
# Delete the first video in the config file to rerun with new config
clean_pretrained_project(config_path)

# Updating the configs within the config.yaml file
edits = {
    'dotsize': 3,  # size of the dots!
    'colormap': 'rainbow',  # any matplotlib colormap
    'alphavalue': 0.2, # transparency of labels
    'pcutoff': 0.5,  # the higher the more conservative the plotting!
    'skeleton': 
         # Right Eye
        [['RightEye_Top', 'RightEye_Inner'], 
         ['RightEye_Inner', 'RightEye_Bottom'],
         ['RightEye_Outer', 'RightEye_Bottom'],
         ['RightEye_Top', 'RightEye_Outer'], 
         # Left Eye
         ['LeftEye_Top', 'LeftEye_Inner'],
         ['LeftEye_Inner', 'LeftEye_Bottom'],
         ['LeftEye_Outer', 'LeftEye_Bottom'],
         ['LeftEye_Top', 'LeftEye_Outer'],
        #  # Top of Head Counter-Clockwise to Lip
        #  ['HeadTop_Mid', 'OutlineRight_Mouth'],
        #  ['OutlineRight_Mouth', 'RightNostrils_Bottom'],
        #  ['RightNostrils_Bottom', 'UpperLip_Centre'],
        #  # Lip Counter-Clockwise to Top of Head
        #  ['UpperLip_Centre', 'OutlineLeft_Mouth'],
        #  ['OutlineLeft_Mouth', 'LeftNostrils_Bottom'],
        #  ['LeftNostrils_Bottom', 'HeadTop_Mid'],
        ],
    'skeleton_color': 'white'
}
deeplabcut.auxiliaryfunctions.edit_config(config_path, edits)

# Adding new videos to the config.yaml file
deeplabcut.add_new_videos(config_path, video_path_list, copy_videos=False, 
            coords=None, extract_frames=False
)

# Extract frames
deeplabcut.analyze_videos(config_path, video_path_list, 
            videotype, save_as_csv=True
)

# Filter predictions
deeplabcut.filterpredictions(config_path, video_path_list, videotype=videotype)

# Create labeled video
deeplabcut.create_labeled_video(
   config_path, video_path_list[:5], 
   videotype, 
   draw_skeleton=True, 
   filtered=True,
   trailpoints=5,
)

# # Plot trajectories
# deeplabcut.plot_trajectories(config_path, video_path_list[6:8], videotype, filtered=True)