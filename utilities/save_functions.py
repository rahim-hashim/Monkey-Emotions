import dill

def unpickle_spikeglx(session_obj):
  """unpickle spikeglx_obj"""
  try:
    pkl_path = None
    # in parent directory
    if os.path.exists(os.path.join('..', os.getcwd(), f'spikeglx_obj_{session_obj.monkey}_{session_obj.date}.pkl')):
      pkl_path = os.path.join('..', os.getcwd(), f'spikeglx_obj_{session_obj.monkey}_{session_obj.date}.pkl')
    # in current directory
    elif os.path.exists(os.path.join(os.getcwd(), f'spikeglx_obj_{session_obj.monkey}_{session_obj.date}.pkl')):
      pkl_path = os.path.join(os.getcwd(), f'spikeglx_obj_{session_obj.monkey}_{session_obj.date}.pkl')
    
    with open(pkl_path, 'rb') as f:
      spikeglx_obj = dill.load(f)
      return spikeglx_obj
  except:
    print(f'Pickled spikeglx_obj not found for: {session_obj.monkey}_{session_obj.date}')