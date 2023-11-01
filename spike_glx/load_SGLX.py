import os
import dill
import pickle
from collections import defaultdict
# Custom classes
from classes.SpikeGLX import SpikeGLX
# Custom modules
from spike_glx import read_SGLX

def load_sglx(session_df, session_obj, file_container_obj, signal_dict, epochs):
  """unpickle spikeglx_obj"""
  # try:
  pickle_flag = False
  monkey_name = session_obj.monkey.lower()
  date_str = session_obj.date
  session_folder = os.path.join('_data', f'{monkey_name}_{date_str}')
  if not os.path.exists(session_folder):
    os.makedirs(session_folder)
    # get full path to session folder
    session_folder_path = os.path.abspath(session_folder)
    print(f'Created new session folder: {session_folder_path}')
  pkl_path = None
  # in _data/<session> directory
  if os.path.exists(os.path.join(session_folder, f'spikeglx_obj_{monkey_name}_{date_str}.pkl')):
    pkl_path = os.path.join(session_folder, f'spikeglx_obj_{monkey_name}_{date_str}.pkl')
    pickle_flag = True
  if pickle_flag == True:
    # absolute path
    pkl_path_str = os.path.abspath(pkl_path)
    print(f'Found pickled spikeglx_obj: {pkl_path_str}')
    with open(pkl_path, 'rb') as f:
      spikeglx_obj = dill.load(f)
    # update sglx path
    print(f'Updating spikeglx_obj paths...')
    spikeglx_obj.sglx_dir_path = file_container_obj.spikeglx_dir_path
    print(f'  Updated sglx_dir_path to: {spikeglx_obj.sglx_dir_path}')
    # update video path
    try:
      del spikeglx_obj.video_file_paths
      print(f'  Deleted old video_file_paths')
    except:
      pass
    try:
      del spikeglx_obj.video_info
      print(f'  Deleted old video_info')
    except:
      pass
    spikeglx_obj.video_file_paths = defaultdict(list)
    spikeglx_obj.video_info = defaultdict(lambda: defaultdict(float))
    print(f'  Updating video_file_paths and video_info...')
    spikeglx_obj._get_whitematter_video_paths(file_container_obj.white_matter_dir_path)
    spikeglx_obj._check_video_paths()
    return spikeglx_obj
  else:
    sglx_dir_path = file_container_obj.spikeglx_dir_path
    sglx_wm_path = file_container_obj.white_matter_dir_path
    print(f'Pickled spikeglx_obj not found for: {monkey_name}_{date_str}')
    print(f'Generating new spikeglx_obj...')
    print(f'  Looking for SpikeGLX binary and meta file in:\n  {sglx_dir_path}')
    # Create SpikeGLX object
    spikeglx_obj = SpikeGLX(sglx_dir_path, 
                            monkey_name, 
                            date_str, 
                            sglx_wm_path, 
                            signal_dict)
    print('SpikeGLX object created.')
    print('Aligning photodiode signals from ML and SpikeGLX...')
    print(min(spikeglx_obj.sample_times))
    spikeglx_obj = read_SGLX.align_sglx_ml(spikeglx_obj, session_df, epochs)
    print('  Done.')
    print('Comparing ML and SpikeGLX photodiode signals...')
    read_SGLX.compare_ML_sglx_cam_frames(spikeglx_obj, session_df)
    print('  Done.')
    print('Plotting ML and SpikeGLX photodiode signals...')
    read_SGLX.plot_analog_ML(session_df, epochs, trial_num=1)
    print('  Done.')
    print('Plotting first trial...')
    read_SGLX.plot_trial_0(session_df, spikeglx_obj)
    print('  Done.')
    print('Saving spikeglx_obj...')
    spikeglx_obj.save_obj(target_folder=session_folder)
    print(f'Done. Saved spikeglx_obj.')
    return spikeglx_obj