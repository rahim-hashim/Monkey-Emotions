import os
import numpy as np
import pandas as pd
import matplotlib as mpl
from collections import defaultdict
from datetime import datetime, timedelta

#COLORS = ['#905C99', '#907F9F', '#B0C7BD', '#B8EBD0']

def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
	c1=np.array(mpl.colors.to_rgb(c1))
	c2=np.array(mpl.colors.to_rgb(c2))
	return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

class Session:
	def __init__(self, df, monkey_input, task, behavioral_code_dict):
		print('Creating Session Objects...')
	
		if df is not None:
			self.df = df
			self.date = df['date'].iloc[0]
			df_flag = True
		else:
			self.df = None
			self.date = None
			df_flag = False
		self.monkey = monkey_input
		self.task = task
		self.window_lick = 1000
		# determine lick threshold
		self.estimate_lick_threshold()
		self.window_blink = 1300
		self.colors = []
		self.stim_labels = []
		self.valence_colors = {1.0: '#28398D', 0.75: '#308ED6', # dark blue, blue,
			 										0.5: '#91BFBC', 0.25: '#9FC5E8', # light blue, lighter blue,
													0: '#B0B0B0', # gray
													-0.5: '#ED8C8C', -1.0: '#D61313', # red, dark red
													} 
		self.valence_labels = defaultdict(str)
		self.session_num = 0						# can be multiple sessions per monkey per day
		self.session_length = 0					# length of each session in seconds
		self.session_time = 0						# length of each session in hours
		self.total_attempts_min = 0 		# total attempts per min per session
		self.prop_trials_initiated = 0	# fraction of initiated trials per session
		self.CS_on_trials = 0						# number of "CS On" trials per session
		self.prop_correct_CS_on = 0			# fraction of correct initiated trials per session
		self.reward_outcome_params = defaultdict(lambda: defaultdict(float))
		self.airpuff_outcome_params = defaultdict(lambda: defaultdict(float))
		self.lick_duration = defaultdict(float)
		self.blink_duration = defaultdict(float)
		self.blink_signal = defaultdict(float)
		self.task_path = ''
		self.figure_path = ''
		self.tracker_path = ''
		self.video_path = ''
		
		# specific to each task
		if df_flag:
			try:
				self.parse_stim_labels()
				self.parse_valence_labels()
				self.generate_colors()
				self.calculate_datetime()
				self.find_offscreen_values()
				self.find_outcome_parameters()
				self.behavior_summary(behavioral_code_dict)
			except:
				# not rhAirpuff
				pass
	
	def estimate_lick_threshold(self):
		"""Determine the voltage threshold for a lick"""
		first_10_lick = self.df['lick'].iloc[:10].tolist()
		first_10_lick = max([item for sublist in first_10_lick for item in sublist])
		self.lick_threshold = np.round(first_10_lick * 0.75, 2)
		print('Lick threshold: {} mV'.format(self.lick_threshold))

	def parse_stim_labels(self):
		"""Get the unique fractal labels for each session"""
		unique_fractals = self.df['stimuli_name_1'].unique()
		# remove error label
		unique_fractals = [fractal for fractal in unique_fractals if 'error' not in fractal]
		self.stim_labels = sorted([fractal.split('_')[-1] for fractal in unique_fractals])

	def parse_valence_labels(self):
		"""Parses valence labels based on the reward magnitude and airpuff magnitude"""
		df = self.df.copy()
		if 'reward_mag_1' in df.columns:
			reward_mag_col = 'reward_mag_1'
			airpuff_mag_col = 'airpuff_mag_1'
		else:
			reward_mag_col = 'reward_mag'
			airpuff_mag_col = 'airpuff_mag'
		reward_mag = np.array(df[reward_mag_col].tolist())
		airpuff_mag_neg = np.array(df[airpuff_mag_col].tolist())*-1
		valence_mag = reward_mag + airpuff_mag_neg
		# airpuff
		if 1 in valence_mag:
			self.valence_labels[-1] = '(--)'
		if 0.5 in valence_mag:
			self.valence_labels[-0.5] = '(-)'
		# neutral
		if 0 in valence_mag:
			self.valence_labels[0] = '(0)'
		# reward
		if 0.25 in valence_mag:
			self.valence_labels[0.25] = '(+)'
		if 0.5 in valence_mag:
			self.valence_labels[0.5] = '(+)'
		if 0.75 in valence_mag:
			self.valence_labels[0.75] = '(++)'
		if 1 in valence_mag:
			self.valence_labels[1] = '(++)'

	def generate_colors(self):
		"""Generate colors for each valence"""
		n = len(self.stim_labels)
		c1 = '#452c49' # dark magenta
		c2 = '#99ffff' # light cyan
		n = len(self.stim_labels)
		for x in range(n):
			color=colorFader(c1,c2,x/n)
			self.colors.append(color)

	def calculate_datetime(self):
		"""Calculate the session number, session length, and session time"""
		session = self.df
		self.session_num = session['session_num'].iloc[0]
		session_length = session['trial_start'].iloc[-1]
		session_start = session['trial_datetime_start'].iloc[0]
		session_end = session['trial_datetime_end'].iloc[-1]
		session_time = session_end - session_start
		session_length_timedelta = timedelta(seconds=session_time.seconds)
		self.session_length = session_length_timedelta
		session_time = round(session_length/(1000*60*60), ndigits=2)
		self.session_time = session_time

	def save_paths(self, TASK_PATH, TRACKER_PATH, VIDEO_PATH, FIGURE_PATH):
		"""Save the paths for each session"""
		self.task_path = TASK_PATH
		self.tracker_path = TRACKER_PATH
		session_video_path = os.path.join(VIDEO_PATH, self.date + '_' + self.monkey)
		self.video_path = session_video_path
		self.figure_path = FIGURE_PATH

	def find_offscreen_values(self):
		"""
		Finds the x and y coordinates of the offscreen EyeSignal value
		which is different each time you calibrate (i.e. each session)
		"""
		df = self.df
		eye_x_min = 0; eye_y_min = 0
		eye_x_max = 0; eye_y_max = 0
		for trial_num in range(5):
			eye_x = df['eye_x'].iloc[trial_num]
			eye_y = df['eye_y'].iloc[trial_num]
			if min(eye_x) < eye_x_min:
				eye_x_min = min(eye_x)			
			if min(eye_y) < eye_y_min:	
				eye_y_min = min(eye_y)
			if max(eye_x) > eye_x_max:
				eye_x_max = max(eye_x)
			if max(eye_y) > eye_y_max:	
				eye_y_max = max(eye_y)
		print('  Min Values (X,Y): ({},{})'.format(round(eye_x_min,3),
																							 round(eye_y_min,3)))
		print('  Max Values (X,Y): ({},{})'.format(round(eye_x_max,3),
																							 round(eye_y_max,3)))
		self.blink_signal['eye_x_min'] = eye_x_min
		self.blink_signal['eye_y_min'] = eye_y_min
		self.blink_signal['eye_x_max'] = eye_x_max
		self.blink_signal['eye_y_max'] = eye_y_max

	def find_outcome_parameters(self):
		"""
		Finds the reward and airpuff parameters for each session.
		For choice sessions, values are in 'reward_mag_1' and 'airpuff_mag_1'.
		For reinforcement only sesssions, values are in 'reward_mag' and 'airpuff_mag'."""
		df = self.df.copy()
		df = df.loc[(df['correct'] == 1) & (df['reinforcement_trial'] == 1)]
		if 'reward_mag_1' in df.columns:
			reward_mag_col = 'reward_mag_1'
			reward_drops_col = 'reward_drops_1'
			reward_freq_col = 'reward_prob_1'
			reward_length_col = 'reward_length_1'
			airpuff_mag_col = 'airpuff_mag_1'
			airpuff_freq_col = 'airpuff_prob_1'
			# num_pulses_col = 'num_pulses'
		else:
			reward_mag_col = 'reward_mag'
			reward_drops_col = 'reward_drops'
			reward_freq_col = 'reward_prob'
			reward_length_col = 'reward_length'
			airpuff_mag_col = 'airpuff_mag'
			airpuff_freq_col = 'airpuff_prob'
			# num_pulses_col = 'num_pulses'
		reward_mags = sorted(df[reward_mag_col].unique(), reverse=True)
		airpuff_mags = sorted(df[airpuff_mag_col].unique(), reverse=True)
		for mag in reward_mags:
			df_mag = df[df[reward_mag_col] == mag]
			reward_drops = df_mag[reward_drops_col].iloc[0]
			reward_freq = df_mag[reward_freq_col].iloc[0]
			reward_length = df_mag[reward_length_col].iloc[0]
			print('  Reward Mag: {}'.format(mag))
			print('    Reward Drops: {}'.format(reward_drops))
			print('    Reward Frequency: {}'.format(reward_freq))
			print('    Reward Length: {}'.format(reward_length))
			self.reward_outcome_params['reward_drops'][mag] = reward_drops
			self.reward_outcome_params['reward_freq'][mag] = reward_freq
			self.reward_outcome_params['reward_length'][mag] = reward_length
		for mag in airpuff_mags:
			df_mag = df[df[airpuff_mag_col] == mag]
			airpuff_mag = df_mag[airpuff_mag_col].iloc[0]
			# airpuff_pulses = df_mag[num_pulses_col].iloc[0]
			airpuff_freq = df_mag[airpuff_freq_col].iloc[0]
			print('  Airpuff Mag: {}'.format(mag))
			print('    Airpuff Magnitude: {}'.format(airpuff_mag))
			# print('    Airpuff Pulses: {}'.format(airpuff_pulses))
			print('    Airpuff Frequency: {}'.format(airpuff_freq))
			self.airpuff_outcome_params['airpuff_mag'][mag] = airpuff_mag
			# self.airpuff_outcome_params['airpuff_pulses'][mag] = airpuff_pulses
			self.airpuff_outcome_params['airpuff_freq'][mag] = airpuff_freq		

	def behavior_summary(self, behavioral_code_dict):
		"""Calculates the behavioral summary for each session"""
		df = self.df
		OUTCOME_SELECTED = [0,9]
		# attempts per min
		session_time = self.session_time
		session_time_min = session_time*60
		total_attempts_min = round(len(df)/session_time_min, 2)
		self.total_attempts_min = total_attempts_min
		# total initiated trials
		df_choice = df.loc[df['error_type'].isin(OUTCOME_SELECTED)]
		prop_trials_initiated = round(len(df_choice)/len(df), 2)
		self.prop_trials_initiated = prop_trials_initiated
		# total correct trials after CS presented
		cs_on_index = [index for index in behavioral_code_dict.keys() if behavioral_code_dict[index] == 'CS On'][0]
		session_df_CS_presented = df[df['behavioral_code_markers'].apply(lambda x: cs_on_index in x)]
		self.CS_on_trials = len(session_df_CS_presented)
		CS_on_correct = session_df_CS_presented['correct'].tolist()
		perf_CS_on = np.sum(CS_on_correct)/len(CS_on_correct)
		perf_CS_on_round = round(perf_CS_on, 2)
		self.prop_correct_CS_on = perf_CS_on_round