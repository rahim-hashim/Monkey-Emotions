import h5py

def view_h5 (path):
	# view h5 file
	with h5py.File(path, 'r') as f:
		# use f string
		print(f'Keys (1): {list(f.keys())}')
		key_values = f['ML'].keys()
		print(f'  Keys (2): {list(key_values)}')
		for key in f['ML'].keys():
			print(key)
			if 'Trial' in key:
				print(f['ML'][key].keys())
				break