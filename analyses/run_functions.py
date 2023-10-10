def run_functions(session_df, session_obj, path_obj, behavioral_code_dict, error_dict):
	"""
	Runs all analyses functions

	Args:
		df (dataframe): dataframe of session data
		session_obj (Session): session object
		path_obj (Path): path object
		behavioral_code_dict (dict): dictionary of behavioral codes
		error_dict (dict): dictionary of error codes

	Returns:
		session_obj (Session): updated session object
	"""

	from analyses.session_performance import session_performance
	# All trials required to analyze performance
	session_performance(session_df, behavioral_code_dict, session_obj)
	
	# Only correct trials for all other analyses
	df = session_df[session_df['correct'] == 1]

	from analyses.duration_hist import epoch_hist
	epoch_hist(df, session_obj)

	from analyses.lick_blink_relationship import lick_blink_linear, trialno_lick_blink_correlation
	lick_blink_linear(df, session_obj)
	trialno_lick_blink_correlation(df, session_obj)

	# from analyses.session_timing import plot_session_timing
	# plot_session_timing(df, session_obj)

	from analyses.outcome_plots import outcome_plots
	outcome_plots(df, session_obj)

	from analyses.session_lick import session_lick
	session_lick(df, session_obj)

	from analyses.trial_raster import trial_raster
	trial_raster(df, session_obj)

	from analyses.raster_by_condition import raster_by_condition
	from analyses.two_sample_test import t_test_moving_avg
	for block in sorted(df['condition'].unique()):
		try:
			session_df_condition = df[df['condition'] == block]
			raster_by_condition(session_df_condition, session_obj)
			t_test_moving_avg(session_df_condition, session_obj, block)
		except:
			print('Not enough trials for condition: {}'.format(block))
	
	from analyses.grant_plots import grant_plots
	grant_plots(df, session_obj)

	# from analyses.measure_hist import measure_hist
	# measure_hist(df, session_obj)

	from analyses.eyetracking_analysis import eyetracking_analysis
	eyetracking_analysis(df, session_obj, TRIAL_THRESHOLD=10)

	from analyses.outcome_over_time import outcome_over_time
	outcome_over_time(df, session_obj)

	from analyses.choice_plots import plot_heatmap_choice_valence, plot_avg_choice_valence
	plot_heatmap_choice_valence(df, session_obj)
	plot_avg_choice_valence(df, session_obj)

	from analyses.log_reg  import log_reg_model
	log_reg_model(df)

	from utilities.markdown_print import markdown_summary
	markdown_summary(df, behavioral_code_dict, session_obj)

	from utilities.write_to_excel import write_to_excel
	write_to_excel(df, session_obj, path_obj)

	return session_obj