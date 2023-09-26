import numpy as np 
import seaborn as sns
from PIL import ImageColor
import matplotlib.colors
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid.inset_locator import inset_axes
# Custom Functions
from eyetracking_analysis import calc_dist

def flattenList(nestedList):
	# check if list is empty
	if not(bool(nestedList)):
		return nestedList
	# to check instance of list is empty or not
	if isinstance(nestedList[0], list):
		# call function with sublist as argument
		return flattenList(*nestedList[:1]) + flattenList(nestedList[1:])
	# call function with sublist as argument
	return nestedList[:1] + flattenList(nestedList[1:])

def plot_eye_dist_histogram(session_df_correct):
	# all distances
	all_distances = session_df_correct['eye_distance'].tolist()
	flat_list_all = flattenList(all_distances)
	eye_distances_filtered_all = sorted([x for x in flat_list_all if x > 0.01], reverse=False)

	# preblink distances
	preblink_distances = session_df_correct['eye_distances_preblink'].tolist()
	flat_list_preblink = flattenList(preblink_distances)
	preblink_filtered_all = sorted([x for x in flat_list_preblink if x > 0.01], reverse=False)
	print('Eye Velocity Means')
	print('  All: {}'.format(round(np.nanmean(eye_distances_filtered_all), 2)))
	print('  Preblink: {}'.format(round(np.nanmean(preblink_filtered_all), 2)))
	print('Eye Velocity Quantiles')
	for data in [eye_distances_filtered_all, preblink_filtered_all]:
		if data == eye_distances_filtered_all:
			print(' All')
		else:
			print(' Preblink')
		print('  25%: {} | 50%: {} | 75%: {}'.format(
			round(np.nanquantile(data, 0.25), 2),
			round(np.nanquantile(data, 0.5), 2),
			round(np.nanquantile(data, 0.75), 2)))
	gs_kw = dict(width_ratios=[4, 1])
	f, axarr = plt.subplots(1, 2, figsize=(10,5), gridspec_kw=gs_kw)
	axarr[0].hist(eye_distances_filtered_all, bins=100, density=True, color='#1f77b4', label='All', lw=1, ec='white')
	axarr[0].hist(preblink_filtered_all, bins=100, density=True, color='red', label='Preblink', alpha=0.5, lw=1, ec='white')
	axarr[0].legend()
	axarr[0].set_xlabel('Eye velocities')
	axarr[0].set_ylabel('Density')
	axarr[1].boxplot([eye_distances_filtered_all, preblink_filtered_all], labels=['All', 'Preblink'], showfliers=False)
	ax = sns.boxplot(ax=axarr[1], data=[eye_distances_filtered_all, preblink_filtered_all], 
									palette=['#1f77b4', 'red'], showfliers=False)
	axarr[1].set_xticklabels(['All', 'Preblink'])
	axarr[1].set_ylabel('Eye velocities')
	plt.suptitle('Eye Velocity Distributions')
	plt.text(-2, 1.025, 'All Eye Positions vs. N={} Preblink Eye Positions'.format(blink_bin_threshold), 
					horizontalalignment='center', fontsize=12, transform=ax.transAxes)
	plt.show()

def plot_dist_histogram(session_df_correct, session_obj, valence_eye_distances, axarr, bins, valence_color, valence_label):
	f, axarr = plt.subplots(2, 2, figsize=(10, 7.5))
	f.suptitle('Eye Distance Distribution', fontsize=16)
	all_eye_distances = []
	for v_index, valence in enumerate(sorted(session_df_correct['valence'].unique(), reverse=True)):
		# see if 1 is in pupil_raster_window column
		session_valence = session_df_correct[session_df_correct['valence'] == valence]
		# blink_trials = session_valence[session_valence['pupil_binary_zero'] == 1].index.tolist()
		valence_eye_distances = []
		for trial in session_valence.index.tolist():
			trial_index = trial
			# eye_distance = session_valence['eye_distance'][trial_index]
			session_df_selected = session_valence.loc[trial_index]
			trace_start = session_df_selected['Trace Start']
			trace_end = session_df_selected['Trace End']
			eye_x = session_df_selected['eye_x'][trace_start:trace_end]
			eye_y = session_df_selected['eye_y'][trace_start:trace_end]
			eye_coordinates = list(zip(eye_x, eye_y))
			trial_distances = []
			for eye_index, eye_pos in enumerate(eye_coordinates):
				# skip first eye position
				if eye_index == 0:
					continue
				dist = calc_dist(eye_pos[0], eye_coordinates[eye_index-1][0],
												eye_pos[1], eye_coordinates[eye_index-1][1])
				# skip if distance is greater than 10 (caused by blink signal)
				if dist < 10:
					trial_distances.append(dist)
			valence_eye_distances.append(np.nansum(trial_distances))
			all_eye_distances.append(np.nansum(trial_distances))
		eye_distances_filtered_all = sorted([x for x in valence_eye_distances if x != 0], reverse=False)
		if v_index == 0:
			axindex = axarr[0][0]
		elif v_index == 1:
			axindex = axarr[0][1]
		elif v_index == 2:
			axindex = axarr[1][0]
		elif v_index == 3:
			axindex = axarr[1][1]
		valence_color = session_obj.valence_colors[valence]
		valence_label = session_obj.valence_labels[valence]
		bins = np.histogram(all_eye_distances, bins=100)[1] # get the bin edges for all valence eye distances
		plot_dist_histogram(valence_eye_distances, axindex, bins, valence_color, valence_label)

	eye_distances_filtered = sorted([x for x in valence_eye_distances if x != 0], reverse=False)
	results = axarr.hist(eye_distances_filtered, bins=bins, ec='black', facecolor=valence_color, alpha=1, density=True)
	axarr.set_xlabel('Eye Distance', fontsize=14)
	axarr.set_ylabel('Trial Count', fontsize=14)
	boxplot_axes = inset_axes(axarr, width="50%", height="30%", loc=1)
	bp_results = boxplot_axes.boxplot(eye_distances_filtered, vert=False, showfliers=False, patch_artist=True)
	boxplot_axes.tick_params(axis='x', size=0.5)    #setting up X-axis tick color to red
	boxplot_axes.tick_params(axis='y', size=0.5)  #setting up Y-axis tick color to black
	for patch, color in zip(bp_results['boxes'], valence_color):
		rgb = matplotlib.colors.to_rgb(valence_color)
		patch.set_facecolor(rgb)
	xrange = np.arange(0, 200, 50)
	boxplot_axes.set_xticks(xrange)
	boxplot_axes.set_xticklabels(xrange, fontsize=8)
	boxplot_axes.set_yticklabels([])
	# boxplot_axes.set_aspect(.4)
	axarr.set_title(valence_label, fontsize=14, color=valence_color)
	axarr.set_xlim([0,200])
	axarr.set_ylim([0,0.1])
	f.tight_layout()
	whisker_data = bp_results['whiskers']
	lower_whisker, upper_whisker = [item.get_xdata()[1] for item in whisker_data]

def calc_eye_dist(trial, session_obj, blink_bin_threshold):
	trace_window = session_obj.window_blink
	blink_raster = np.array(trial['pupil_raster_window'])
	trace_off_time = trial['Trace End']
	eye_x, eye_y = trial['eye_x'], trial['eye_y']
	eye_x = eye_x.tolist()[trace_off_time-trace_window:trace_off_time]
	eye_y = eye_y.tolist()[trace_off_time-trace_window:trace_off_time]
	# calculate eye velocities
	eye_coordinates = list(zip(eye_x, eye_y))
	eye_distances = []
	for eye_index, eye_pos in enumerate(eye_coordinates):
		if eye_index == 0:
			dist_val = np.nan
		else:
			dist_val = calc_dist(eye_pos[0], eye_coordinates[eye_index-1][0],
													 eye_pos[1], eye_coordinates[eye_index-1][1])
		eye_distances.append(dist_val)
	trial['eye_distances_window'] = eye_distances
	# calculate eye distances before blinks
	all_blinks = np.where(blink_raster==1)[0]
	if len(all_blinks) == 0:
		trial['eye_distances_preblink'] = np.nan
		return trial
	preblink_indices = []
	preblink_distances = []
	for blink_bin in all_blinks:
		# if blink_bin is less than threshold, add all eye distances before blink
		if blink_bin < blink_bin_threshold:
			preblink_distances.append(eye_distances[:blink_bin])
			preblink_indices += list(range(blink_bin))
		else:
			preblink_bins = list(range(blink_bin-blink_bin_threshold, blink_bin))
			# if preblink_bins are already in preblink_indices, skip
			preblink_distances.append(\
				[eye_distances[i] for i in preblink_bins if i not in preblink_indices])
			preblink_indices += preblink_bins
	trial['eye_distances_preblink'] = preblink_distances
	return trial



def distance_analysis(session_df_correct, valence_eye_distances):
	num_samples = 30
	palette_tab10 = sns.color_palette("magma", num_samples)

	blink_trials = session_df_correct[session_df_correct['pupil_binary_zero'] == 1].index.tolist()
	print(blink_trials)
	for trial in blink_trials:
		trial_index = trial
		session_df_selected = session_df_correct.loc[trial_index]
		trace_start = session_df_selected['Trace Start']
		trace_end = session_df_selected['Trace End']
		eye_x = session_df_selected['eye_x'][trace_start:trace_end]
		eye_y = session_df_selected['eye_y'][trace_start:trace_end]
		eye_coordinates = list(zip(eye_x, eye_y))
		for eye_index, eye_pos in enumerate(eye_coordinates):
			# skip first eye position
			if eye_index == 0:
				continue
			dist = calc_dist(eye_pos[0], eye_coordinates[eye_index-1][0],
											eye_pos[1], eye_coordinates[eye_index-1][1])
			# skip if distance is greater than 10 (caused by blink signal)
			if dist > 10:
				continue
			all_eye_distances.append(dist)
	eye_distances_filtered_all = sorted([x for x in valence_eye_distances if x != 0], reverse=False)
	pupil = session_df_selected['eye_pupil'][trace_start:trace_end]
	pupil_raster = session_df_selected['pupil_raster_window']
	all_blinks = np.where(pupil==0)[0]
	for b_index, blink in enumerate(all_blinks):
		if b_index == 0 or blink-1 not in all_blinks:
			eye_x_preblink = eye_x[blink-num_samples:blink]
			eye_y_preblink = eye_y[blink-num_samples:blink]
			eye_preblink = list(zip(eye_x_preblink, eye_y_preblink))
			for eye_index, eye_pos in enumerate(eye_preblink):
				if eye_index == 0:
					continue
				print(trial_index, blink, eye_index,eye_pos[0], eye_pos[1],
					calc_dist(eye_pos[0], eye_preblink[eye_index-1][0],
										eye_pos[1], eye_preblink[eye_index-1][1]))
			break
		break
	f, axarr = plt.subplots(1, 1, figsize=(10, 4))
	axarr.hist(eye_distances_filtered_all, bins=50, ec='white', color='#1f77b4', density=True)
	plt.show()

	first_blink = np.where(pupil==0)[0][0]
	eye_x_preblink = eye_x[first_blink-num_samples:first_blink]
	eye_y_preblink = eye_y[first_blink-num_samples:first_blink]

	eye_preblink = list(zip(eye_x_preblink, eye_y_preblink))
	for eye_index, eye_pos in enumerate(eye_preblink):
		if eye_index == 0:
			continue
		print(
			calc_dist(eye_pos[0], eye_preblink[eye_index-1][0],
								eye_pos[1], eye_preblink[eye_index-1][1]))
	for eye_index, eye_pos in enumerate(eye_preblink):
		plt.scatter(eye_pos[0], eye_pos[1], s=8, color=palette_tab10[eye_index])
	plt.xlim([-40, 40])
	plt.ylim([-40, 40])
	plt.show()