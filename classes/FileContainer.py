import os
import sys
import pandas as pd
from pprint import pprint
from itertools import combinations
from tkinter.filedialog import askdirectory, askopenfilename
# Custom modules
from utilities.mat_import import loadmat
from config.h5_helper import h5_load, h5_parse, add_fields
from config.session_parse_helper import session_parser
# Custom classes
from classes.Session import Session


class FileContainer:
	def __init__(self, ROOT_DIR):
		self.ml_file_path = None
		self.white_matter_dir_path = None
		self.spikeglx_dir_path = None
		self.monkey_name = {}
		self.date = {}
		self.find_files(ROOT_DIR)

	def find_files(self, ROOT_DIR=None):
		"""
		Load behavior, video, and spikeglx files
		"""
		# load root directory (computer/task dependent)
		if ROOT_DIR is None:
			root = '/Users/rahimhashim/My Drive/Columbia/Salzman/Monkey-Training/tasks/rhAirpuff'
		else:
			root = ROOT_DIR

		# load behavior file
		print('Select .bhv2/.h5 behavior file (i.e. 230927_Aragorn_choice.h5)')
		beh_file_path = askopenfilename(title='Select .bhv2 or .h5 behavior file', 
																		filetypes=[('bhv2 files', '.bhv2'),
																							('.h5 files', '.h5')],
																		initialdir=root) 
		print('Behavior file selected: {}'.format(beh_file_path))
		beh_file_name = os.path.basename(beh_file_path)
		ml_date = beh_file_name.split('_')[0]
		self.date['ml'] = ml_date
		print('  MonkeyLogic Date: {}'.format(ml_date))
		ml_monkey_name = beh_file_name.split('_')[1].lower()
		self.monkey_name['ml'] = ml_monkey_name
		print('  MonkeyLogic Monkey: {}'.format(ml_monkey_name))
		
		# load video files
		print('Select directory containing White Matter video files (i.e. 230927_Aragorn)')
		video_dir_path = askdirectory(title='Select directory containing White Matter video files', 
																	initialdir=root)
		video_file_list = os.listdir(video_dir_path)
		print('Video files directory selected: {}'.format(video_dir_path))
		video_dir_name = os.path.basename(video_dir_path)
		wm_date = video_dir_name.split('_')[0]
		self.date['wm'] = wm_date
		print('  White Matter Video Date: {}'.format(wm_date))
		wm_monkey_name = video_dir_name.split('_')[1].lower()
		self.monkey_name['wm'] = wm_monkey_name
		print('  White Matter Video Monkey: {}'.format(wm_monkey_name))
		
		# load spikeglx files
		print('Select directory containing SpikeGLX files')
		sglx_dir_path = askdirectory(title='Select directory containing SpikeGLX .bin and .meta files',
																initialdir=root)
		sglx_file_list = os.listdir(sglx_dir_path)
		print('SpikeGLX files directory selected: {}'.format(sglx_dir_path))
		sglx_dir_name = os.path.basename(sglx_dir_path)
		sglx_date = sglx_dir_name.split('_')[1][2:]
		self.date['sglx'] = sglx_date
		print('  SpikeGLX Date: {}'.format(sglx_date))
		sglx_monkey_name = sglx_dir_name.split('_')[0].lower()
		self.monkey_name['sglx'] = sglx_monkey_name
		print('  SpikeGLX Monkey: {}\n'.format(sglx_monkey_name))

		# check to make sure all files are from the same session
		source_list = ['MonkeyLogic', 'White Matter', 'SpikeGLX']
		dates = [ml_date, wm_date, sglx_date]
		monkeys = [ml_monkey_name, wm_monkey_name, sglx_monkey_name]
		dates_combinations = list(combinations(zip(source_list, dates), 2))
		monkeys_combinations = list(combinations(zip(source_list, monkeys), 2))
		for index, date_combination in enumerate(dates_combinations):
			if date_combination[0][1] != date_combination[1][1]:
				print('WARNING: dates do not match')
				print(f'  {date_combination[0][0]}: {date_combination[0][1]}')
				print(f'  {date_combination[1][0]}: {date_combination[1][1]}')
			if monkeys_combinations[index][0][1] != monkeys_combinations[index][1][1]:
				print('WARNING: monkeys do not match')
				print(f'  {monkeys_combinations[index][0][0]}: {monkeys_combinations[index][0][1]}')
				print(f'  {monkeys_combinations[index][1][0]}: {monkeys_combinations[index][1][1]}')
		self.ml_file_path = beh_file_path
		self.white_matter_dir_path = video_dir_path
		# parent directory of sglx_dir_path is the sglx data directory used in pipeline
		self.spikeglx_dir_path = os.path.dirname(sglx_dir_path)
	
	def ml_to_pd(self):
		ml_beh_file = self.ml_file_path
		# check file extension
		if ml_beh_file.endswith('.bhv2'):
			print('Convert .bhv2 to .h5: {}'.format(ml_beh_file))
		elif ml_beh_file.endswith('.h5'):
			h5_py_file = h5_load(ml_beh_file)
			ml_config, trial_record, trial_list, cam1_list, cam2_list = h5_parse(h5_py_file)
			session_dict, error_dict, behavioral_code_dict = \
        session_parser(h5_py_file, trial_list, trial_record, self.date['ml'], self.monkey_name['ml'])
			session_df = pd.DataFrame.from_dict(session_dict)
			session_obj = Session(session_df, self.monkey_name['ml'], 'rhAirpuff', behavioral_code_dict)
			session_df, session_obj = add_fields(session_df,
																					session_obj, 
																					behavioral_code_dict)
			session_obj.df = session_df
			return session_obj, error_dict, behavioral_code_dict