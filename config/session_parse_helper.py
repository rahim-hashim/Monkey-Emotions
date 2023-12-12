import re
import sys
import numpy as np
from tqdm.auto import tqdm
from collections import defaultdict
from datetime import datetime, timedelta
from analyses.time_processing import calculate_end_time

def stimulus_parser(stimulus, stim_num, session_dict):
	'''Parses out parameters for each stimulus set for each trial config

  Parameters
  ----------
	session : .h5 group object
		contains 
	session_dict: Dict 
		dictionary containing all specified session data

  Returns
  -------
	session_dict: Dict 
		dictionary containing all specified session data
	'''
	stimuli_name = ''
	# stimuli type = pic
	try:
		stimuli_type = stimulus['1'][...].tolist().decode()
		stimuli_string = stimulus['2'][...].tolist().decode()
		stimuli_name = stimuli_string.split('\\')[-1].split('.')[0]
		# if '_fix' in stimuli_name:
		# 	pass
		# else:
		x_pos = stimulus['3'][...][0]
		y_pos = stimulus['4'][...][0]
		session_dict['stimuli_name_{}'.format(stim_num)].append(stimuli_name) # name
		session_dict['x_{}_pos'.format(stim_num)].append(x_pos) # x-position
		session_dict['y_{}_pos'.format(stim_num)].append(y_pos) # y-position
	except:
		pass
	return session_dict

def camera_parser(session, session_dict, cam1_list, cam2_list, date_input, monkey_input):
	print('Parsing camera data...')
	if (not cam1_list) and (not cam2_list):
		print('  No camera data attached to ML file.')
		session_dict['cam1_trial_name'] = np.nan
		session_dict['cam2_trial_name'] = np.nan
		session_dict['cam1_trial_time'] = np.nan
		session_dict['cam2_trial_time'] = np.nan
		session_dict['cam1_video'] = np.nan
		session_dict['cam2_video'] = np.nan
	else:
		for c_index, cam_list in enumerate([cam1_list, cam2_list]):
			for t_index, trial in enumerate(tqdm(cam_list)):
				cam_data = session['ML'][trial]
				# name of camera trial file
				cam_trial_name_full = cam_data['Filename'][...].tolist().decode()
				cam_trial_name = cam_trial_name_full.split('\\')[-1].split('.')[0]
				trial_name_column = 'cam{}_trial_name'.format(c_index+1)
				session_dict[trial_name_column].append(cam_trial_name)
				# time of frames in camera trial file
				cam_time_column = 'cam{}_trial_time'.format(c_index+1)
				cam_trial_time = cam_data['Time'][0]
				session_dict[cam_time_column].append(cam_trial_time)
				# data from camera trial file
				cam_data_column = 'cam{}_video'.format(c_index+1)
				try:
					cam_trial_file = cam_data['File'][0]
					session_dict[cam_data_column].append(cam_trial_file)
				except: # video data removed from file using mlexportwebcam
					session_dict[cam_data_column].append(np.nan)
			print('  Complete.')
	return session_dict

def session_parser(session, trial_list, trial_record, date_input, monkey_input):
	'''Parses out session data

  Parameters
  ----------
	session : .h5 file
		specified session for parsing
	trial_list : list
		list of trials within session

  Returns
  -------
	session_dict: Dict 
		dictionary containing all specified session data
	'''
	session_dict = defaultdict(list)

	# date (and handling for multiple sessions)
	session_dict['date'] = date_input
	session_str = str(session)
	session_str_num = re.search(r'\((.*?)\)',session_str).group(1)
	if session_str_num.isnumeric():
		session_dict['session_num'] = int(session_str_num)
	else:
		session_dict['session_num'] = int(0)

	# monkey
	session_dict['subject'] = monkey_input.lower()
	
	# error code mapping
	error_dict = defaultdict(str)
	try:
		for error_code in list(trial_record['TaskInfo']['TrialErrorCodes'].keys()):
			error_val = str(int(error_code)+1) # ML messes up TrialErrorCodes mapping to be 1-based
			# [...] returns scalar values from .h5 branches
			try:
				error_codes_val = trial_record['TaskInfo']['TrialErrorCodes'][error_val][...].tolist().decode()
				# take off leading b'
				error_dict[error_code] = error_codes_val
			except KeyError:
				pass # fix .h5 error code mapping

		# behavioral code mapping
		behavioral_code_dict = defaultdict(str)
		behavioral_numbers = list(map(int, trial_record['TaskInfo']['BehavioralCodes']['CodeNumbers'][0]))
		behavioral_code_keys = list(trial_record['TaskInfo']['BehavioralCodes']['CodeNames'].keys())
		behavioral_names = []
		for bev_key in behavioral_code_keys:
			behavioral_names.append(trial_record['TaskInfo']['BehavioralCodes']['CodeNames'][bev_key][...].tolist().decode())
		for b_index, behavioral_code in enumerate(behavioral_numbers):
			try:
				behavioral_code_dict[behavioral_code] = behavioral_names[b_index]
			except KeyError:
				pass # fix .h5 behavioral code mapping
	except:
		behavioral_code_dict = defaultdict(str)
		pass

	# TrialRecord.User fields (reinforcement)
	if 'reward_stim_1' not in trial_record['User'].keys():
		try:
			stim_container = trial_record['User']['stim_list']
			stim_list = stim_container.keys()
			reward_container = trial_record['User']['reward']
			airpuff_container = trial_record['User']['airpuff']
			print('  Reinforcement task only.')
		except:
			pass
	# TrialRecord.User fields (reinforcement + choice)
	else:
		try:
			stim_container = trial_record['User']['stim_list']
			stim_list = stim_container.keys()
			reward_container_1 = trial_record['User']['reward_stim_1']
			reward_container_2 = trial_record['User']['reward_stim_2']
			airpuff_container_1 = trial_record['User']['airpuff_stim_1']
			airpuff_container_2 = trial_record['User']['airpuff_stim_2']
			# choice task design fields
			choice_task_container = defaultdict(list)
			choice_task_container['reinforcement_trial'] = trial_record['User']['trial_type']['reinforcement_trial'][...].tolist()
			choice_task_container['choice_trial'] = trial_record['User']['trial_type']['choice_trial'][...].tolist()
			choice_task_container['stim_chosen'] = trial_record['User']['stim_chosen']['stimuli'][...].tolist()
			choice_task_container['stim_2_chosen'] = trial_record['User']['stim_2_chosen']['stimuli'][...].tolist()
			choice_task_container['fractal_index_chosen'] = trial_record['User']['fractal_chosen']['stimuli'][...].tolist()
			choice_task_container['reward'] = trial_record['User']['fractal_chosen']['reward'][...].tolist()
			choice_task_container['reward_mag'] = trial_record['User']['fractal_chosen']['reward_mag'][...].tolist()
			choice_task_container['airpuff'] = trial_record['User']['fractal_chosen']['airpuff'][...].tolist()
			choice_task_container['airpuff_mag'] = trial_record['User']['fractal_chosen']['airpuff_mag'][...].tolist()
			print('  Choice task detected.')
		except:
			pass
	
	# trial_list is ordered already (Trial1...TrialN) but we should put in some checks
	# to make sure that it holds in all cases
	print('Parsing session data...')
	for t_index, trial in enumerate(tqdm(trial_list)):
		# skip cam data (redundant check)
		if 'Cam' in trial:
			continue

		# all trial data
		trial_data = session['ML'][trial]

		# trial number
		trial_num = list(trial_data['Trial'][0]) # starts at Trial 1...Trial N
		session_dict['trial_num'].append(int(trial_num[0]))

		# block
		block = int(trial_data['Block'][()][0][0])
		session_dict['block'].append(block)

		# condition
		condition = int(trial_data['Condition'][()][0][0])
		session_dict['condition'].append(condition)

		trial_result = int(trial_data['TrialError'][()][0][0])
		# correct trial
		if trial_result == 0:
			session_dict['correct'].append(1)
			session_dict['error'].append(0)
		# error trial
		else:
			session_dict['correct'].append(0)
			session_dict['error'].append(1)

		# error code mapping
		#   - error_dict[0]     = 'correct'
		#   - error_dict[{1-9}] = '<error_type>'
		session_dict['error_type'].append(int(trial_result))

		# behavioral codes
		behavioral_code_markers = np.array(trial_data['BehavioralCodes']['CodeNumbers'][0])
		behavioral_code_times = np.array(trial_data['BehavioralCodes']['CodeTimes'][0])
		session_dict['behavioral_code_markers'].append(list(map(int,behavioral_code_markers)))
		session_dict['behavioral_code_times'].append(behavioral_code_times)

		# stimuli info
		stimuli_attribute = trial_data['TaskObject']['Attribute']
		# all hard coded by MonkeyLogic and therefore here as well
		try:  # list of stimuli in stimuli_attribute
			test_for_list = stimuli_attribute['1'] 
			for stim_num, stimulus in enumerate(stimuli_attribute):
				session_dict = stimulus_parser(stimuli_attribute[stimulus], stim_num, session_dict)
		except: # one stimuli in stimuli_attribute
			pass
			# session_dict = stimulus_parser(stimuli_attribute, stim_num, session_dict)
		if 'stim_list' in locals(): # experiment contains user-generated variable TrialRecord.User.stim_container
			for stimulus in stim_list:
				# exclude fix (no degree value)
				if stimulus == 'fix':
					continue
		# Reinforcement trials only
		reward_mag = 0
		airpuff_mag = 0
		if 'reward_container' in locals():
			reward = int(reward_container['reward'][t_index])
			session_dict['reward'].append(reward)
			reward_prob = float(reward_container['reward_prob'][t_index])
			session_dict['reward_prob'].append(reward_prob)
			try:
				reward_mag = float(reward_container['reward_mag'][t_index])
				session_dict['reward_mag'].append(reward_mag)
				reward_drops = float(reward_container['drops'][t_index])
				session_dict['reward_drops'].append(reward_drops)		
				reward_length = float(reward_container['length'][t_index])
				session_dict['reward_length'].append(reward_length)	
			except:
				pass
		if 'airpuff_container' in locals():
			airpuff = int(airpuff_container['airpuff'][t_index])
			session_dict['airpuff'].append(airpuff)
			airpuff_prob = float(airpuff_container['airpuff_prob'][t_index])
			session_dict['airpuff_prob'].append(airpuff_prob)
			try:
				airpuff_mag = float(airpuff_container['airpuff_mag'][t_index])
				session_dict['airpuff_mag'].append(airpuff_mag)
				num_pulses = float(airpuff_container['num_pulses'][t_index])
			except:
				pass
		# Reinforcement + choice trials 
		if 'reward_container_1' in locals():
			reward = int(reward_container_1['reward'][t_index])
			session_dict['reward_1'].append(reward)
			reward_prob = float(reward_container_1['reward_prob'][t_index])
			session_dict['reward_prob_1'].append(reward_prob)
			try:
				reward_mag = float(reward_container_1['reward_mag'][t_index])
				session_dict['reward_mag_1'].append(reward_mag)
				reward_drops = float(reward_container_1['drops'][t_index])
				session_dict['reward_drops_1'].append(reward_drops)		
				reward_length = float(reward_container_1['length'][t_index])
				session_dict['reward_length_1'].append(reward_length)	
			except:
				pass
		if 'reward_container_2' in locals():
			reward = int(reward_container_2['reward'][t_index])
			session_dict['reward_2'].append(reward)
			reward_prob = float(reward_container_2['reward_prob'][t_index])
			session_dict['reward_prob_2'].append(reward_prob)
			try:
				reward_mag = float(reward_container_2['reward_mag'][t_index])
				session_dict['reward_mag_2'].append(reward_mag)
				reward_drops = float(reward_container_2['drops'][t_index])
				session_dict['reward_drops_2'].append(reward_drops)		
				reward_length = float(reward_container_2['length'][t_index])
				session_dict['reward_length_2'].append(reward_length)	
			except:
				pass	

		if 'airpuff_container_1' in locals():
			airpuff = int(airpuff_container_1['airpuff'][t_index])
			session_dict['airpuff_1'].append(airpuff)
			airpuff_prob = float(airpuff_container_1['airpuff_prob'][t_index])
			session_dict['airpuff_prob_1'].append(airpuff_prob)
			try: # new fields in airpuff_container_1
				airpuff_mag = float(airpuff_container_1['airpuff_mag'][t_index])
				session_dict['airpuff_mag_1'].append(airpuff_mag)
				num_pulses = float(airpuff_container_1['num_pulses_1'][t_index])
				session_dict['num_pulses_1'].append(num_pulses)
			except:
				pass
		if 'airpuff_container_2' in locals():
			airpuff = int(airpuff_container_2['airpuff'][t_index])
			session_dict['airpuff_2'].append(airpuff)
			airpuff_prob = float(airpuff_container_2['airpuff_prob'][t_index])
			session_dict['airpuff_prob_2'].append(airpuff_prob)
			try:
				airpuff_mag = float(airpuff_container_2['airpuff_mag'][t_index])
				session_dict['airpuff_mag_2'].append(airpuff_mag)
				num_pulses = float(airpuff_container_2['num_pulses_2'][t_index])
				session_dict['num_pulses_2'].append(num_pulses)
			except:
				pass

		# eye data
		x = np.array(trial_data['AnalogData']['Eye'])
		eye_data = x.view(np.float64).reshape(x.shape+(-1,))
		session_dict['eye_x'].append(eye_data[0].flatten())
		session_dict['eye_y'].append(eye_data[1].flatten())

		# pupil data
		x = np.array(trial_data['AnalogData']['EyeExtra'])
		try:
			session_dict['eye_pupil'].append(x[0])
		except:
			pass # no pupil data

		# joystick data
		x = np.array(trial_data['AnalogData']['Joystick'])
		try:
			joystick_data = x.view(np.float64).reshape(x.shape+(-1,))
			session_dict['joystick_x'].append(joystick_data[0])
			session_dict['joystick_y'].append(joystick_data[1])
		except:
			pass # no joystick data

		# lick data
		x = np.array(trial_data['AnalogData']['General']['Gen1'])
		session_dict['lick'].append(x[0])

		# camera TTL (sync + save) data
		try:
			x = np.array(trial_data['AnalogData']['General']['Gen2'])
			session_dict['cam_sync'].append(x[0])
			y = np.array(trial_data['AnalogData']['General']['Gen3'])
			session_dict['cam_save'].append(y[0])
		except:
			pass

		# reward data
		# x = np.array(trial_data['VariableChanges'])

		# photodiode data
		x = np.array(trial_data['AnalogData']['PhotoDiode'])
		try:
			session_dict['photodiode'].append(x[0])
		except:
			if t_index == 0:
				print('  Missing photodiode data.')
			session_dict['photodiode'].append(np.nan) # no photodiode data

		# trial start time (relative to session start)
		trial_start_time = float(trial_data['AbsoluteTrialStartTime'][()][0][0])
		session_dict['trial_start'].append(trial_start_time)
		
		# trial start time (absolute)
		length_trial = len(x[0])
		session_dict['trial_end'].append(trial_start_time+length_trial)

		start_datetime, end_datetime = calculate_end_time(trial_data['TrialDateTime'][()], length_trial)

		session_dict['trial_datetime_start'].append(start_datetime)
		session_dict['trial_datetime_end'].append(end_datetime)


	num_trials = len(session_dict['trial_num'])
	float_fields = ['reward_mag', 'airpuff_mag']
	if 'choice_task_container' in locals():
		for field in choice_task_container.keys():
			field_values = choice_task_container[field]
			if field in float_fields:
				field_values = [float(item) for sublist in field_values for item in sublist]
			else:
				field_values = [int(item) for sublist in field_values for item in sublist]
			if field == 'fractal_index_chosen':
				for trial in range(num_trials):
					fractal_index_chosen = field_values[trial]
					try:
						if fractal_index_chosen == 2:
							fractal_name = session_dict['stimuli_name_1'][trial]
							session_dict['fractal_chosen'].append(fractal_name)
						elif fractal_index_chosen == 3:
							fractal_name = session_dict['stimuli_name_2'][trial]
							session_dict['fractal_chosen'].append(fractal_name)
						else:
							session_dict['fractal_chosen'].append('_error')
					except:
						print(f'  Trial {trial} error: {field} parsing.')
						print('    fractal_index_chosen: {}'.format(fractal_index_chosen))
						print('    session_dict[stimuli_name_1]: {}'.format(len(session_dict['stimuli_name_1'])))
						print('    session_dict[stimuli_name_2]: {}'.format(len(session_dict['stimuli_name_2'])))
						sys.exit()
			else:
				session_dict[field] = field_values

	# pop all non-equal keys
	num_trials = len(session_dict['trial_num'])
	for key in list(session_dict.keys()):
		# excluded from checks
		if key in ['date', 'subject']:
			continue
		try:
			if len(session_dict[key]) == num_trials + 1:
				# pop the last trial
				session_dict[key].pop()
			if len(session_dict[key]) != num_trials:
				print('  {} removed from session_dict'.format(key))
				print(f'    {len(session_dict[key])} trials in field != {num_trials} trials in session')
				session_dict.pop(key)
		except:
			pass
	print('  Complete.')
	print('    Correct trials: {}'.format(np.sum(session_dict['correct'])))
	print('    Errored trials: {}'.format(np.sum(session_dict['error'])))
	# Session time
	session_start = session_dict['trial_datetime_start'][0]
	session_end = session_dict['trial_datetime_end'][-1]
	session_time = session_end - session_start
	print('    Session Length: ', str(datetime.utcfromtimestamp(session_time.total_seconds()).strftime('%H:%M:%S')))
	
	return session_dict, error_dict, behavioral_code_dict