import os
import pickle

def select_session(session_root, date, monkey, correct_only=True):
	print(f'Selecting session: {date}_{monkey}')
	session_dir = os.listdir(session_root)
	"""Select session from list of sessions"""
	try:
		session = [f for f in session_dir if f.split('_')[0] == date and f.split('_')[1] == monkey][0]
	except IndexError:
		print('Session not found')
		for f in sorted(session_dir):
			if monkey in f:
				print(f'  {f}')
	with open(os.path.join(session_root, session), 'rb') as f:
		session_dict = pickle.load(f)
		# convert to dataframe
		session_df = session_dict['data_frame']
		# select only correct trials
		if correct_only:
			session_df = session_df[session_df['correct'] == 1]
	return session_df