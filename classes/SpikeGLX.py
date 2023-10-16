import re
import os
import sys
import cv2
import math
import dill
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from pprint import pprint
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
# Custom classes
from spike_glx import read_SGLX
from classes.Session_Path import SessionPath
from classes.Session import Session
# pandas options
pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_columns', None)

def create_float_defaultdict():
  '''Pickling error workaround'''
  return defaultdict(float)

class SpikeGLX:
  def __init__(self, 
               sglx_path = None, 
               monkey_name: str = None,
               date: str = None,
               video_path: str = None, 
               signal_dict: dict = None):
    
    """
    SpikeGLX class for parsing SpikeGLX data

    Parameters
    ----------
    sglx_path : str
      path to raw data folder with SpikeGLX .meta and .bin files

    monkey_name : str
      name of monkey

    date : str
      date of session in format YYMMDD

    video_path : str
      path to video folder

    signal_dict : dict
      dictionary of signal names
    """

    # Initialize variables
    self.sample_rate = -1
    self.monkey_name = monkey_name
    self.date = date
    self.sglx_file_path = sglx_path
    self.cam_sync = np.array([])
    self.cam_save = np.array([])
    self.cam_frames = np.array([])
    self.lick = np.array([])
    self.photodiode = np.array([])
    self.bin_file_path = None
    self.video_file_paths = defaultdict(list)
    self.video_info = defaultdict(create_float_defaultdict)
    self.trial_times = defaultdict(create_float_defaultdict)
    self.cam_framenumbers = defaultdict(create_float_defaultdict)
    self.ml_sglx_corr_matrix = None
    self.trial_missing_videos = []

    if sglx_path is not None and monkey_name is not None and date is not None:
      self._find_SGLX()
    if self.bin_file_path is not None:
      self._load_spikeglx(signal_dict)
    if self.meta is not None and self.chan_dict is not None and signal_dict is not None:
      self._parse_meta_bin(signal_dict)
      self._find_spikeglx_cam_frames()
      self._cam_save_goes_high()
    if video_path is not None:
      self._get_whitematter_video_paths(video_path)
      self._check_video_paths()

  def _get_whitematter_video_paths(self, video_path):
    video_folders = os.listdir(video_path)
    video_folders = [folder for folder in video_folders if os.path.isdir(os.path.join(video_path, folder))]
    print(f'Video Folders:')
    pprint(video_folders, indent=2)
    for folder in video_folders:
      video_files = sorted(os.listdir(os.path.join(video_path, folder)))
      for video_file in video_files:
        if video_file.endswith('.mp4') or video_file.endswith('.avi'):
          camera = video_file.split('-')[0] # default White Matter video file name is <camera>-<date>.mp4
          self.video_file_paths[camera].append(os.path.join(video_path, folder, video_file))
    print('Number of cameras: {}'.format(len(self.video_file_paths.keys())))
    for camera in self.video_file_paths.keys():
      print('  Camera: {} | Number of videos: {}'.format(camera, len(self.video_file_paths[camera])))

  def _check_video_paths(self):
    # count the number of frames in each video
    for camera in self.video_file_paths.keys():
      print('Camera: {}'.format(camera))
      num_frames = 0
      for video_file_path in sorted(self.video_file_paths[camera]):
        cap = cv2.VideoCapture(video_file_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        video_name = video_file_path.split('/')[-1]
        video_length_minutes = round(frame_count/fps/60, 2)
        # align vertical line with print statement
        print('  Video: {:<30} | Frames: {:<6} | FPS: {:<5} | Length (minutes): {:<7}'.format(video_name, frame_count, fps, video_length_minutes))
        index_start = num_frames
        num_frames += frame_count
        self.video_info[camera][video_name] = {'index_start': index_start, 
                                               'index_end': num_frames}
        
      print('  Total Frames: {}'.format(num_frames))

  def _find_SGLX(self):
    print(f'Looking for binary file in {self.sglx_file_path}')
    monkey_name = self.monkey_name.lower()
    date = '20' + self.date
    # find all folders in raw data path folder with <monkey_name>_date_g<any_int> format
    for folder in os.listdir(self.sglx_file_path):
      if re.match(monkey_name + '_' + date + '_g\d+', folder):
        spikeglx_folder = folder
        break
    spikeglx_bin = None
    spikeglx_meta = None
    print(f'Found folder: {spikeglx_folder}')
    # find .bin file
    for file in os.listdir(os.path.join(self.sglx_file_path, spikeglx_folder)):
      if file.endswith('.bin'):
        spikeglx_bin = file
        print(f'  Found binary file: {spikeglx_bin}')
      if file.endswith('.meta'):
        spikeglx_meta = file
        print(f'  Found metadata file: {spikeglx_meta}')
    bin_file_path = os.path.join(self.sglx_file_path, spikeglx_folder, spikeglx_bin)
    self.bin_file_path = Path(bin_file_path)
  
  def _load_spikeglx(self, signal_dict=None):
    """Loads SpikeGLX data to retrieve meta and chan_dict"""
    self.meta, self.chan_dict = read_SGLX.load_spikeglx(self.bin_file_path)
    if signal_dict:
      self.signal_dict = signal_dict

  def _parse_meta_bin(self, signal_dict):
    """Parses meta and bin files to get sample rate and analog signals"""
    sample_rate, sample_times, chan_dict_corrected = \
      read_SGLX.parse_channels(self.meta, self.chan_dict, signal_dict)
    self.sample_rate = sample_rate
    self.sample_times = sample_times
    self.cam_sync = chan_dict_corrected[0]
    self.cam_save = chan_dict_corrected[1]
    self.lick = chan_dict_corrected[2]
    self.photodiode = chan_dict_corrected[3]
    # delete chan_dict from self
    del self.chan_dict

  def _find_spikeglx_cam_frames(self):
    """Finds overlap between cam_save and cam_sync signals"""

    if self.sample_rate == -1:
      print('Please run _parse_meta_bin() first')
      return
    sample_rate = self.sample_rate

    if len(self.cam_sync) == 0:
      print('cam_sync signal is empty')
      return

    if len(self.cam_save) == 0:
      print('cam_save signal is empty')
      return

    # find index where cam_sync is higher than 1000
    cam_sync_threshold = np.where(self.cam_sync > 1000)[0]
    # find index where cam_sync_oneback is lower than 1000
    cam_sync_oneback = np.roll(self.cam_sync, 1)
    cam_sync_oneback_threshold = np.where(cam_sync_oneback < 1000)[0]
    # find overlap between the two sync signals
    cam_sync_overlap = np.intersect1d(cam_sync_threshold, cam_sync_oneback_threshold)
    # shift array by 1 to get the previous value
    cam_save_threshold = np.where(self.cam_save > 1000)[0]
    # find overlap between the sync signal and the save signal
    cam_frames = np.intersect1d(cam_save_threshold, cam_sync_overlap)/sample_rate
    # conver to milliseconds
    cam_frames*=1000
    print('Number of frames in SpikeGLX Cam TTL: {}'.format(len(cam_frames)))
    self.cam_frames = cam_frames

  def _cam_save_goes_high(spikeglx_obj):
    """Plots when cam_save signal goes high and low"""
    sample_rate = spikeglx_obj.sample_rate
    cam_save = spikeglx_obj.cam_save
    
    cam_save_oneback = np.roll(cam_save, 1)
    cam_save_threshold = np.where(cam_save > 3000)[0]
    cam_save_oneback_threshold = np.where(cam_save_oneback < 3000)[0]
    cam_save_overlap = np.intersect1d(cam_save_threshold, cam_save_oneback_threshold)
    f, axarr = plt.subplots(2, 1, figsize=(20, 5))
    buffer = 20
    x_axis = np.arange(cam_save_overlap[0]-buffer, cam_save_overlap[0]+buffer)/sample_rate*1000
    axarr[0].plot(x_axis,
            cam_save[cam_save_overlap[0]-buffer:cam_save_overlap[0]+buffer])
    axarr[0].set_title('SpikeGLX - Camera Save Signal Goes High')
    axarr[0].set_xlabel('Time (ms)')
    axarr[0].set_ylabel('Voltage (mV)')
    cam_save_threshold = np.where(cam_save < 3000)[0]
    cam_save_oneback_threshold = np.where(cam_save_oneback > 3000)[0]
    cam_save_overlap = np.intersect1d(cam_save_threshold, cam_save_oneback_threshold)
    x_axis = np.arange(cam_save_overlap[0]-buffer, cam_save_overlap[0]+buffer)/sample_rate*1000
    axarr[1].plot(x_axis,
            cam_save[cam_save_overlap[0]-buffer:cam_save_overlap[0]+buffer])
    axarr[1].set_title('SpikeGLX - Camera Save Signal Goes Low')
    axarr[1].set_ylabel('Voltage (mV)')
    axarr[1].set_xlabel('Time (ms)')
    plt.tight_layout()
    plt.show()

  def save_obj(self):
    """Saves SpikeGLX object as pickle file"""
    pkl_path = os.path.join(os.getcwd(), f'spikeglx_obj_{self.monkey_name}_{self.date}.pkl')
    with open(pkl_path, 'wb') as f:
      # get info on size of pickle file
      print('Pickle file size: {} MB'.format(sys.getsizeof(dill.dumps(self))/1000000))
      # dump pickle file
      dill.dump(self, f)
      print('Pickle file saved: {}'.format(pkl_path))
      self.pkl_path = pkl_path