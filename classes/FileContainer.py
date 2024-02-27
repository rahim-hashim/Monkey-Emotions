import os
import re
import sys
import pandas as pd
from pprint import pprint
from itertools import combinations
import tkinter as tk
from tkinter.filedialog import askdirectory, askopenfilename
# Custom modules
from utilities.mat_import import loadmat
from config.h5_helper import h5_load, h5_parse, add_fields
from config.session_parse_helper import session_parser
# Custom classes
from classes.Session import Session


class FileContainer:
	def __init__(self, ROOT_DIR, VIDEO_DIR=None, MONKEY=None, DATE=None, BEHAVIOR_FILE_ONLY=False):
		self.ml_file_path = None
		self.white_matter_dir_path = None
		self.spikeglx_dir_path = None
		self.monkey_name = {}
		self.date = {}
		self._find_files(ROOT_DIR, VIDEO_DIR, MONKEY, DATE, BEHAVIOR_FILE_ONLY)

	def _find_files(self, ROOT_DIR=None, VIDEO_DIR=None, MONKEY=None, DATE=None, BEHAVIOR_FILE_ONLY=False):
		"""
		Load behavior, video, and spikeglx files
		"""

		# load root directory (computer/task dependent)
		if ROOT_DIR is None:
			root_path = '/Users/rahimhashim/My Drive/Columbia/Salzman/Monkey-Training/tasks/rhAirpuff'
		else:
			root_path = ROOT_DIR
		
		video_path = VIDEO_DIR

		beh_file_path = None	# <ROOT>/<MONKEY>_<DATE>_g<#>_<TASK>.h5
		video_dir_path = None	# <VIDEO_DIR>/<MONKEY>_<DATE>
		sglx_dir_path = None	# <ROOT>/<MONKEY>_<DATE>_g<#>
		tkinter_flag = False	# flag to check if tkinter window is open

		# automatically find correct files and directories
		try:
			# look for folder that has both monkey and date in the name
			session_folder = [folder for folder in os.listdir(root_path) if MONKEY in folder and DATE in folder][0]
			print(f'Session folder {session_folder} found in default location: {ROOT_DIR}')
			# look for behavior file
			beh_file_list = [file for file in os.listdir(os.path.join(root_path, session_folder)) if file.endswith('.h5')]
			if len(beh_file_list) == 0:
				print('  WARNING: no behavior files found.')
			elif len(beh_file_list) == 1:
				beh_file_path = os.path.join(root_path, session_folder, beh_file_list[0])
				print('  Behavior file found: {}'.format(beh_file_path))
			else:
				print('  WARNING: more than one behavior file found.')
				print('  {}'.format(beh_file_list))
				beh_file_path = os.path.join(root_path, session_folder, beh_file_list[0])
				print('    Selecting first file: {}'.format(beh_file_path))
			# look for video files
			video_dir_list = [folder for folder in os.listdir(video_path) if (DATE[2:] in folder and MONKEY.capitalize() in folder)]
			if len(video_dir_list) == 0:
				print('  WARNING: no White Matter video files found.')
			elif len(video_dir_list) == 1:
				video_dir_path = os.path.join(video_path, video_dir_list[0])
				print('  White Matter video files found: {}'.format(video_dir_path))
			else:
				print('  WARNING: more than one White Matter video files found.')
				print('  {}'.format(video_dir_list))
				video_dir_path = os.path.join(root_path, session_folder, video_dir_list[0])
				print('    Selecting first directory: {}'.format(video_dir_path))
			# look for spikeglx files (ends in g0, g1...) using regex
			if re.search('g\d', session_folder) is not None and \
										len([file for file in os.listdir(os.path.join(root_path, session_folder)) if file.endswith('.bin')]) != 0:
				sglx_dir_path = os.path.join(root_path, session_folder)
				print('  WARNING: no SpikeGLX files found.')
			else:
				print('  WARNING: no SpikeGLX files found.')
				print('  {}'.format(os.path.join(root_path, session_folder)))
		except:
			print('Session folder not found.')

	
		if beh_file_path is not None or video_dir_path is not None or sglx_dir_path is not None:
			# initialize tkinter window
			print('Select behavior, video, and SpikeGLX files manually.')
			root = tk.Tk()
			root.withdraw()
			tkinter_flag = True
			
		if beh_file_path is None:
			# load behavior file
			print(f'  Select .h5 behavior file (i.e. {DATE}_{MONKEY}_choice.h5)')
			beh_file_path = askopenfilename(title='Select .h5 behavior file', 
																			filetypes=[('.h5 files', '.h5'),
																								#('bhv2 files', '.bhv2')
																								],
																			initialdir=root_path) 
		if video_dir_path is None:
			if not BEHAVIOR_FILE_ONLY:
				# load video files
				print(f'  Select directory containing White Matter video files (i.e. {DATE}_{MONKEY})')
				video_dir_path = askdirectory(title='Select directory containing White Matter video files', 
																			initialdir=root_path)

		if sglx_dir_path is None:
			if not BEHAVIOR_FILE_ONLY:
				# load spikeglx files
				print(f'  Select directory containing SpikeGLX files (i.e. {MONKEY}_{DATE}_g0)')
				sglx_dir_path = askdirectory(title='Select directory containing SpikeGLX .bin and .meta files',
																		initialdir=root_path)
			
		if tkinter_flag:
			# close tkinter window
			root.update()
			root.destroy()

		# MonkeyLogic file
		print('Behavior file selected: {}'.format(beh_file_path))
		beh_file_name = os.path.basename(beh_file_path)
		ml_date = beh_file_name.split('_')[0]
		self.date['ml'] = ml_date
		print('  MonkeyLogic Date: {}'.format(ml_date))
		ml_monkey_name = beh_file_name.split('_')[1].lower()
		self.monkey_name['ml'] = ml_monkey_name
		print('  MonkeyLogic Monkey: {}'.format(ml_monkey_name))
		self.ml_file_path = beh_file_path

		if BEHAVIOR_FILE_ONLY:
			print('Behavior file only selected.')
			print('  Skipping White Matter files.')
			print('  Skipping SpikeGLX files.')
			self.monkey_name['wm'] = ''
			self.date['wm'] = ''
			self.monkey_name['sglx'] = ''
			self.date['sglx'] = ''
			self.white_matter_dir_path = ''
			self.spikeglx_dir_path = ''
			return
		else:
			# White Matter video files
			video_file_list = os.listdir(video_dir_path)
			print('Video files directory selected: {}'.format(video_dir_path))
			video_dir_name = os.path.basename(video_dir_path)
			wm_date = video_dir_name.split('_')[0]
			self.date['wm'] = wm_date
			print('  White Matter Video Date: {}'.format(wm_date))
			wm_monkey_name = video_dir_name.split('_')[1].lower()
			self.monkey_name['wm'] = wm_monkey_name
			print('  White Matter Video Monkey: {}'.format(wm_monkey_name))
		
			# SpikeGLX files
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