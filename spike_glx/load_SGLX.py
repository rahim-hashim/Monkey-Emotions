import os
import dill
import pickle
# Custom classes
from classes.SpikeGLX import SpikeGLX
# Custom modules
from spike_glx import read_SGLX

def load_sglx(session_df, session_obj, file_container_obj, signal_dict, epochs):
  """unpickle spikeglx_obj"""
  # try:
  pickle_flag = False
  pkl_path = None
  # in parent directory
  if os.path.exists(os.path.join('..', os.getcwd(), f'spikeglx_obj_{session_obj.monkey}_{session_obj.date}.pkl')):
    pkl_path = os.path.join('..', os.getcwd(), f'spikeglx_obj_{session_obj.monkey}_{session_obj.date}.pkl')
    pickle_flag = True
  # in current directory
  elif os.path.exists(os.path.join(os.getcwd(), f'spikeglx_obj_{session_obj.monkey}_{session_obj.date}.pkl')):
    pkl_path = os.path.join(os.getcwd(), f'spikeglx_obj_{session_obj.monkey}_{session_obj.date}.pkl')
    pickle_flag = True
  if pickle_flag == True:
    print(f'Found pickled spikeglx_obj: {pkl_path}')
    with open(pkl_path, 'rb') as f:
      spikeglx_obj = dill.load(f)
      return spikeglx_obj
  else:
    sglx_dir_path = file_container_obj.spikeglx_dir_path
    sglx_wm_path = file_container_obj.white_matter_dir_path
    print(f'Pickled spikeglx_obj not found for: {session_obj.monkey}_{session_obj.date}')
    print(f'Generating new spikeglx_obj...')
    print(f'  Looking for SpikeGLX binary and meta file in:\n  {sglx_dir_path}')
    # Create SpikeGLX object
    spikeglx_obj = SpikeGLX(sglx_dir_path, 
                            session_obj.monkey, 
                            session_obj.date, 
                            sglx_wm_path, 
                            signal_dict)
    print('SpikeGLX object created.')
    print('Aligning photodiode signals from ML and SpikeGLX...')
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
    spikeglx_obj.save_obj()
    print(f'  Done. Saved spikeglx_obj to {spikeglx_obj.pkl_path}')
    return spikeglx_obj
  # except:
  #   print(f'Error loading spikeglx_obj for: {session_obj.monkey}_{session_obj.date}')
  #   return None
