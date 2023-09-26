import pandas as pd

def select_dates(session_df):
  dates_selected = ['220913', '220928', '220929', '221006', '221019', '221026',
                    '221207', '221214', '230102', '230118', '230120', '230130',
                    '230214', '230215', '230216', '230223', '230307', '230309']
  # dates_selected = ['230214', '230215', '230216', '230223', '230307', '230309']
  session_df = session_df.loc[(session_df['date'].isin(dates_selected))]
  return session_df