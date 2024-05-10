import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

marker_name_template = (
		"(?P<monkey>[A-Za-z]+)_(?P<date>[0-9]+)_(?P<trial>[0-9]+)_(?P<cam>[a-z0-9]+)"
		".*_filtered"
)
marker_template = marker_name_template + "\.h5"

def _interpret_file(
		fl,
		groups=("date", "trial", "cam", "monkey"),
		file_template=marker_template
):
		m = re.match(file_template, fl)
		if m is not None:
				out = tuple(m.group(g) for g in groups)
		else:
				out = None
		return out

def interpret_marker_file(*args, **kwargs):
		return _interpret_file(*args, **kwargs)

def read_marker_file(marker_file_path):
	print(f'  Reading: {marker_file_path}')
	df = pd.read_hdf(marker_file_path)
	return df

def generate_marker_df(video_folder):
	# find all marker_files in video_folder
	marker_files = [file_name for file_name in os.listdir(video_folder) if file_name.endswith('.h5') and 'filtered' in file_name]
	dlc_df_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
	print(f'Found {len(marker_files)} marker files.')
	for marker_file in marker_files:
		date, trial, camera, monkey = interpret_marker_file(marker_file)
		marker_file_path = os.path.join(video_folder, marker_file)
		df = read_marker_file(marker_file_path)
		dlc_df_dict[monkey][date][trial][camera] = df
	print('Done.')
	return dlc_df_dict

def get_marker_likelihoods(df, bodyparts_likelihoods, camera):
		scorer = df.columns[0][0]
		bodyparts = df[scorer].columns.get_level_values(0).unique()
		print(f'  Number of bodyparts: {len(bodyparts)}')
		file_overall_likelihood = []
		for bodypart in bodyparts:
			bodypart_mean = np.mean(df[scorer][bodypart]['likelihood'])
			bodypart_x = np.mean(df[scorer][bodypart]['x'])
			bodyparts_likelihoods[camera][bodypart].append(bodypart_mean)
			file_overall_likelihood.append(bodypart_mean)
		print(f'  Overall likelihood: {round(np.mean(file_overall_likelihood), 2)}')
		return bodyparts_likelihoods

def plot_bodypart_likelihoods(dlc_df_dict):
	bodyparts_likelihoods = defaultdict(lambda: defaultdict(list))
	for monkey, dates in dlc_df_dict.items():
		for date, trials in dates.items():
			for trial, cameras in trials.items():
				for camera, df in cameras.items():
					print(f'Processing {monkey} - {date} - {trial} - {camera}')
					bodyparts_likelihoods = get_marker_likelihoods(df, bodyparts_likelihoods, camera)
	# make a 1x3 bar chart of the likelihoods of each bodypart
	f, axarr = plt.subplots(3, 1, figsize=(15, 7.5), sharey=True)
	# create space between subplots
	plt.subplots_adjust(hspace=1.5)
	all_body_parts = defaultdict(list)
	for ax, (camera, bodyparts) in zip(axarr, bodyparts_likelihoods.items()):
		body_parts = list(bodyparts.keys())
		likelihoods = [np.mean(bodyparts[bodypart]) for bodypart in body_parts]
		# add all body parts and likelihoods to dictionary
		for bodypart, likelihood in zip(body_parts, likelihoods):
			all_body_parts[bodypart].append(likelihood)
		# make edge color of bars black
		ax.bar(body_parts, likelihoods, edgecolor='black')
		ax.set_title(f'{camera} DLC Likelihood Estimates')
		ax.set_ylabel('Mean Likelihood')
		# rotate x labels 45 degrees, make it smaller font, and align to the right
		ax.set_xticklabels(body_parts, rotation=45, fontsize=8, ha='right')
		# set light grey horizontal line at 0.5 with alpha 0.5
		ax.axhline(y=0.5, color='grey', alpha=0.5, linestyle='--')
	plt.show()
	# sort all_body_parts by average mean likelihood
	body_parts_mean = {bodypart: np.mean(likelihoods) for bodypart, likelihoods in all_body_parts.items()}
	sorted_body_part_likelihoods = {k: v for k, v in sorted(body_parts_mean.items(), key=lambda item: item[1], reverse=True)}
	return sorted_body_part_likelihoods