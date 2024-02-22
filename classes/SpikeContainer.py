import os
import re
import numpy as np
import pandas as pd
from tkinter.filedialog import askdirectory, askopenfilename

def folder_contains_imec_and_digits(folder_path, specified_probes=None):
    # Define the pattern using regular expression
    pattern = re.compile(r'imec\d+')
    probe_names = [f'imec{probe}' for probe in specified_probes]
    subfolders = []
    # Check if any subfolder matches the pattern
    for subfolder in os.listdir(folder_path):
        if os.path.isdir(os.path.join(folder_path, subfolder)) and pattern.search(subfolder):
            imec_num = pattern.search(subfolder).group()
            if specified_probes==None or imec_num in probe_names:
              subfolders.append(subfolder)
              print(f'    Found {subfolder}...')
            else:
              print(f'    Skipping {subfolder}...')
    return subfolders

class SpikeContainer:
  def __init__(self, 
               ROOT=None, 
               session_obj=None, 
               probes=None,
               probe_assignment=None):
    """
    Finds and loads spike times and clusters from phy output files

    Parameters
    ----------
    ROOT : str
      Root directory containing phy output files
    session_obj : Session
      Session object containing monkey and date
    probe_assignment : dict
      Dictionary with keys as imec numbers and values as brain regions

    """
    self.cluster_info = None
    self._load_phy_output(ROOT, session_obj, probes, probe_assignment)


  def _load_phy_output(self, 
                       ROOT=None,
                       session_obj=None,
                        probes=None, 
                       probe_assignment=None):
    """Load phy output files"""
    # Initialize variables
    spike_times = None
    spike_clusters = None
    cluster_group = None
    cluster_info = None

    # load root directory (computer/task dependent)
    root = ROOT
    if ROOT is None:
      root = '/Users/rahimhashim/My Drive/Columbia/Salzman/Github/Spatial_Abstraction/data/example_ephys'
    # load behavior file
    phy_dir_path = ''
    if session_obj:
      monkey = session_obj.monkey
      date = '20'+session_obj.date
      data_dir = os.path.join(root, f'{monkey}_{date}')
      print('Loading data from', data_dir)
      # check if directory exists
      if os.path.exists(data_dir):
        phy_dir_path = data_dir
    if phy_dir_path == '':
      print('Select directory containing phy output files')
      phy_dir_path = askdirectory(title='Select directory containing phy output files',
                              initialdir=root)

    # find if g<int> folder exists using regex
    session_folders = [f for f in os.listdir(phy_dir_path) if f==f'{monkey}_{date}_g0'][0]
    if len(session_folders) == 0:
      raise ValueError('No g<int> folder found in selected directory')
    # find if imec<int> folder exists using regex
    session_folder_path = os.path.join(phy_dir_path, session_folders)
    imec_folders = folder_contains_imec_and_digits(session_folder_path, probes)
    if len(imec_folders) == 0:
      raise ValueError('No imec<int> folder found in selected directory')

    for imec_folder in imec_folders:
      # find directory in imec_folder that has imec_folder+'_ks' or imec_folder+'ks_cat' in it
      imec_folder_path = os.path.join(session_folder_path, imec_folder)
      spike_folder = [f for f in os.listdir(imec_folder_path) if f.endswith('_ks') or f.endswith('ks_cat')][0]
      if len(spike_folder) != 0:
        imec_folder_path = os.path.join(imec_folder_path, spike_folder)
      spike_files = os.listdir(imec_folder_path)
      
      print(f'  Loading data from {imec_folder}...')

      # load spike times, clusters, cluster group, and cluster info
      for spike_file in spike_files:
        spike_file_path = os.path.join(imec_folder_path, spike_file)
        if spike_file_path.endswith('spike_times.npy'):
          spike_times = np.load(spike_file_path).flatten()
        if spike_file_path.endswith('spike_clusters.npy'):
          spike_clusters = np.load(spike_file_path).flatten()
        if spike_file_path.endswith('cluster_group.tsv'):
          cluster_group = np.loadtxt(spike_file_path, dtype=str, delimiter='\t', skiprows=1)
        if spike_file_path.endswith('cluster_info.tsv'):
          cluster_info = pd.read_csv(spike_file_path, delimiter='\t')

      spike_times_adj = [f for f in spike_files if f.endswith('spike_times_adj.npy')]
      if len(spike_times_adj) != 0:
        print(f'    Found {spike_times_adj[0]}...')
        print(f'    Replacing spike_times with spike_times_adj...')
        spike_times = np.load(os.path.join(imec_folder_path, spike_times_adj[0])).flatten()

      # if any of the spike files are not loaded, raise error
      if spike_times is None or spike_clusters is None or cluster_group is None or cluster_info is None:
        print(f'    Error: {spike_file} not found in {imec_folder}')
        print(f'    Skipping {imec_folder}...')
      
      # load 'good' cluster groups
      print(f"    {'Number of clusters:':<30} {len(cluster_info):<25}")
      print(f"    {'Number of good clusters:':<30} {len(cluster_group):<25}")

      # number of spikes
      n_spikes_str = '{:,}'.format(len(spike_times))
      n_spikes_per_cluster = '{:,}'.format(round(len(spike_times) / len(cluster_info)))
      print(f"    {'Number of spikes:':<30} {n_spikes_str :<25}")
      print(f"    {'Number of spikes/cluster:':<30} {n_spikes_per_cluster :<25}")

      # Create dataframe with rows as cluster and columns as array of spike times
      print('    Creating spike dataframe...')
      spike_time_clusters = list(zip(spike_times, spike_clusters))
      df = pd.DataFrame(spike_time_clusters, columns=['spike_time', 'cluster'])
      # sort by cluster
      df = df.sort_values(by='cluster')
      # concatenate spike times for each cluster
      df = df.groupby('cluster')['spike_time'].apply(list).reset_index()
      # add column where if cluster is in cluster_group, then True, else False
      df['good'] = df['cluster'].isin(list(map(int, cluster_group[:, 0])))
      # sort spike times in ascending order
      df['spike_time'] = df['spike_time'].apply(np.sort)

      # add df to cluster_info, aligning by cluster number
      cluster_info = cluster_info.set_index('cluster_id')
      df = df.set_index('cluster')
      cluster_info = pd.concat([cluster_info, df], axis=1)
      
      # add imec number to cluster_info
      imec_num = imec_folder.split('_')[-1]
      cluster_info.insert(0, 'imec', imec_num)
      # add probe_assignment to cluster_info
      if probe_assignment:
        probe = probe_assignment[imec_num]
        cluster_info.insert(1, 'region', probe)
      else:
        cluster_info.insert(1, 'region', None)
      # add monkey to cluster_info
      cluster_info.insert(0, 'monkey', monkey)
      # add date to cluster_info
      cluster_info.insert(0, 'date', date)
      # reset index
      cluster_info = cluster_info.reset_index()
      # rename 'index' to 'cluster'
      cluster_info = cluster_info.rename(columns={'index': 'cluster'})

      # add to spike_container
      if self.cluster_info is None:
        self.cluster_info = cluster_info
      else:
        self.cluster_info = pd.concat([self.cluster_info, cluster_info])