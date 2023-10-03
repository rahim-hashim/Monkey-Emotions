import dill

def unpickle_spikeglx(session_obj):
  """unpickle spikeglx_obj"""
  try:
    with open(f'spikeglx_obj_{session_obj.monkey}_{session_obj.date}.pkl', 'rb') as f:
      spikeglx_obj = dill.load(f)
      return spikeglx_obj
  except:
    print(f'Pickled spikeglx_obj not found for: {session_obj.monkey}_{session_obj.date}')