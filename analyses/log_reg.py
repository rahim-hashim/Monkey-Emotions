import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate, ShuffleSplit

# For more information on logistic regression:
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

def print_coeff(cv_results, input_var):
  '''Prints the average coefficient for each input variable'''
  cv_estimators = [x.coef_ for x in cv_results['estimator']]
  coefficient_averages = np.average(cv_estimators, axis=0)
  for i in range(len(input_var)):
    print(f'  {input_var[i]}: {round(coefficient_averages[0][i], 3)}')

def plot_roc_curve(logistic_regression, X_test, y_test, descriptor):
  '''Plots the ROC curve for the logistic regression model'''
  y_score = logistic_regression.decision_function(X_test)
  fpr, tpr, _ = roc_curve(y_test, y_score)
  roc_auc = auc(fpr, tpr)
  lw = 2
  plt.figure()
  plt.plot(fpr, tpr, color='darkorange',
          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
  plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title(f'{descriptor} ROC Curve')
  plt.legend(loc="lower right")
  plt.show()

def log_reg(input_var, X, y, descriptor):
  '''Generate logistic regression models for post-learning behavior'''
  # Split data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
  # Fit model
  logistic_regression = LogisticRegression().fit(X_train, y_train)
  # Use cross-validation to evaluate the model
  cv_results = cross_validate(logistic_regression, X_test, y_test, cv=5, return_estimator=True)
  cv_scores = cv_results['test_score']
  # Print the mean accuracy and standard deviation of the cv_results
  print("%s Accuracy: %0.2f (+/- %0.2f)" % 
                                    (descriptor,
                                    np.mean(cv_scores), 
                                    np.std(cv_scores) * 2))
  # See coefficients
  print_coeff(cv_results, input_var)

  # Plot ROC curve
  plot_roc_curve(logistic_regression, X_test, y_test, descriptor)

def log_reg_model(df):
  '''Calculates logistic regression model accuracy for post-learning behavior'''
  # Reinforcement trials only
  df_reinforcement = df[df['reinforcement_trial'] == 1]
  # Post-Learning Behavior
  df_threshold = df_reinforcement[df_reinforcement['fractal_count_in_block'] > 3]
  # Shuffle dataframe before sending to CV
  df_threshold = df_threshold.sample(frac=1).reset_index(drop=True)

  # Input Variables
  input_var = ['lick_duration',
               'blink_duration_offscreen',
               'pupil_raster_window_avg',
               'blink_duration_window',
               'eye_distance'
              ]
  
  # Reward vs. Airpuff
  X = df_threshold[input_var]
  y = df_threshold['reward_1']
  # Create a logistic regression object
  log_reg(input_var, X, y, descriptor='Reward vs. Airpuff')

  # Large Reward vs. Small Reward
  df_reward = df_threshold[df_threshold['reward_1'] == 1]
  if len(df_reward['reward_mag_1'].unique()) > 1:
    df_reward['mag'] = [1 if x == 1.0 else 0 for x in df_reward['reward_mag_1']]
    X = df_reward[input_var]
    y = df_reward['mag']
    # Create a logistic regression object
    log_reg(input_var, X, y, descriptor='Large Reward vs. Small Reward')

  # Large Airpuff vs. Small Airpuff
  df_airpuff = df_threshold[df_threshold['airpuff'] == 1]
  if len(df_airpuff['airpuff_mag_1'].unique()) > 1:
    df_airpuff['mag'] = [1 if x == 1.0 else 0 for x in df_airpuff['airpuff_mag']]
    X = df_airpuff[input_var]
    y = df_airpuff['mag']
    log_reg(input_var, X, y, descriptor='Large Airpuff vs. Small Airpuff')