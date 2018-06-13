# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 00:55:19 2018

Adaptation of BS_ML_Registered_RF_with_GSCV

Random Forest in Python an End to End Machine Learning Example

https://towardsdatascience.com/random-forest-in-python-24d0893d51c0


@author: Jon
"""

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import r2_score

import os
import seaborn as sns


# Special sns argument for improved plots
sns.set_style("whitegrid")
sns.set_context("poster")


# Import the Data

#Change the working directory
os.chdir('C:/Users/Jon/Documents/Springboard/Capstone1/Bike-Sharing-Dataset')

# Read in the csv's for day.csv and hour.csv
day_df = pd.read_csv('day.csv')


No_aTemp = day_df[['instant', 'season', 'mnth', 'holiday', 'weekday',
       'workingday', 'weathersit', 'temp', 'hum', 'windspeed', 'registered']]

# Pandas is used for data manipulation
import pandas as pd

# Read in data as pandas dataframe and display first 5 rows
features = No_aTemp
features.head(5)
print('The shape of our features is:', features.shape)
# y = day_df.registered

# One-hot encode categorical features
features = pd.get_dummies(features)
features.head(5)
print('Shape of features after one-hot encoding:', features.shape)

# Use numpy to convert to arrays
import numpy as np

# Labels are the values we want to predict
labels  = np.array(features['registered'])

# Remove the labels from the features
# axis 1 refers to the columns
features= features.drop('registered', axis = 1)

# Saving feature names for later use
feature_list = list(features.columns)

# Convert to numpy array
features = np.array(features)

print('\n')
print('Removed target from features')
print('\n')

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25,
                                                                           random_state = 42)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)


# The baseline predictions are the historical averages
# baseline_preds = test_features[:, feature_list.index('average')]

# Baseline errors, and display average baseline error
# baseline_errors = abs(baseline_preds - test_labels)
# print('Average baseline error: ', round(np.mean(baseline_errors), 2), 'degrees.')

# Import the model we are using
from sklearn.ensemble import RandomForestRegressor

# Instantiate model 
rf = RandomForestRegressor(n_estimators= 1000, random_state=42)

# Train the model on training data
rf.fit(train_features, train_labels);

rf_new = RandomForestRegressor(n_estimators = 100, criterion = 'mse', max_depth = None, 
                               min_samples_split = 2, min_samples_leaf = 1)

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)

# Calculate the absolute errors
errors = abs(predictions - test_labels)

# Print out the mean absolute error (mae)
print('\n')
print('Mean Absolute Error:', round(np.mean(errors), 2), 'riders.')


# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)

# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
r2_first = r2_score(test_labels, predictions)

print('\n')
print('R^2 for first forest: ', r2_first)

# print('Accuracy:', round(accuracy, 2), '%.')


# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot

# Pull out one tree from the forest
tree = rf.estimators_[5]

# Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)

# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')

# Write graph to a png file
graph.write_png('tree.png'); 

print('The depth of this tree is:', tree.tree_.max_depth)


# Limit depth of tree to 2 levels
rf_small = RandomForestRegressor(n_estimators=10, max_depth = 3, random_state=42)
rf_small.fit(train_features, train_labels)

# Extract the small tree
tree_small = rf_small.estimators_[5]

# Save the tree as a png image
export_graphviz(tree_small, out_file = 'small_tree.dot', feature_names = feature_list, rounded = True, precision = 1)

(graph, ) = pydot.graph_from_dot_file('small_tree.dot')

graph.write_png('small_tree.png')



# Get numerical feature importances
importances = list(rf.feature_importances_)

# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];



print('\n')
print('New Random Forest with only the two most important variables.')
# New random forest with only the two most important variables
rf_most_important = RandomForestRegressor(n_estimators= 1000, random_state=42)

# Extract the two most important features
important_indices = [feature_list.index('temp'), feature_list.index('hum')]
train_important = train_features[:, important_indices]
test_important = test_features[:, important_indices]

# Train the random forest
rf_most_important.fit(train_important, train_labels)

# Make predictions and determine the error
predictions = rf_most_important.predict(test_important)

errors = abs(predictions - test_labels)

# Display the performance metrics
print('Mean Absolute Error:', round(np.mean(errors), 2), 'riders.')

mape = np.mean(100 * (errors / test_labels))
#accuracy = 100 - mape
r2_most_important = r2_score(test_labels, predictions)
print('R^2 for most important values forest: ', r2_most_important)

#print('Accuracy:', round(accuracy, 2), '%.')
print('\n')



# Import matplotlib for plotting and use magic command for Jupyter Notebooks
# import matplotlib.pyplot as plt

# %matplotlib inline For Jupytr notebook

# Set the style
plt.style.use('fivethirtyeight')

# list of x locations for plotting
x_values = list(range(len(importances)))

# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical')

# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical')

# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances'); 
plt.show()
plt.clf()



# =============================================================================
# 
# import datetime
# 
# # Dates of training values
# months = features[:, feature_list.index('month')]
# days = features[:, feature_list.index('day')]
# years = features[:, feature_list.index('year')]
# 
# # List and then convert to datetime object
# dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
# dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
# 
# # Dataframe with true values and dates
# true_data = pd.DataFrame(data = {'date': dates, 'actual': labels})
# 
# # Dates of predictions
# months = test_features[:, feature_list.index('month')]
# days = test_features[:, feature_list.index('day')]
# years = test_features[:, feature_list.index('year')]
# 
# # Column of dates
# test_dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
# 
# # Convert to datetime objects
# test_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in test_dates]
# 
# # Dataframe with predictions and dates
# predictions_data = pd.DataFrame(data = {'date': test_dates, 'prediction': predictions}) 
# =============================================================================

# Dataframe with true values and dates
true_instants = features[:, feature_list.index('instant')]
true_data = pd.DataFrame(data = {'Instant': true_instants , 'actual': labels})


test_instants = test_features[:, feature_list.index('instant')]
predictions_data = pd.DataFrame(data = {'Instant': test_instants, 'prediction': predictions}) 
# Plot the actual values
plt.plot(true_data['Instant'], true_data['actual'], 'b-', label = 'actual')

# Plot the predicted values
x_ax_pred = range(0,183)
plt.plot(predictions_data['Instant'], predictions_data['prediction'], 'ro', label = 'prediction')
plt.xticks(rotation = '60'); 
plt.legend()

# Graph labels
plt.xlabel('Instant'); plt.ylabel('Maximum Registered Riders'); plt.title('Actual and Predicted Values');
plt.show()
plt.clf()

print('\n')
print('\n')
print('\n')
print('\n')
print('\n')

# =============================================================================
# # This section does not adapt well.
# 
# # Make the data accessible for plotting
# true_data['temp'] = features[:, feature_list.index('temp')]
# true_data['hum'] = features[:, feature_list.index('hum')]
# true_data['windspeed'] = features[:, feature_list.index('windspeed')]
# 
# # Plot all the data as lines
# plt.plot(true_data['Instant'], true_data['actual'], 'b-', label  = 'actual', alpha = 1.0)
# plt.plot(true_data['Instant'], true_data['temp'], 'y-', label  = 'temp', alpha = 1.0)
# plt.plot(true_data['Instant'], true_data['hum'], 'k-', label = 'hum', alpha = 0.8)
# plt.plot(true_data['Instant'], true_data['windspeed'], 'r-', label = 'windspeed', alpha = 0.3)
# 
# # Formatting plot
# plt.legend(); plt.xticks(rotation = '60');
# 
# # Lables and title
# plt.xlabel('Instant'); plt.ylabel('Maximum Rider Count'); plt.title('Actual Max Temp and Variables');
# plt.show()
# plt.clf()
# 
# 
# =============================================================================

# Look at parameters used by our current forest
# from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state = 42)

from pprint import pprint

# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(rf.get_params())

from sklearn.model_selection import RandomizedSearchCV

# =============================================================================
# 
# # Randomized Search CV
# 
# # Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start = 10, stop = 1000, num = 2)]
# # Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(10, 110, num = 2)]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split = [2, 5, 10]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4]
# # Method of selecting samples for training each tree
# bootstrap = [True, False]
# 
# # Create the random grid
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}
# 
# pprint(random_grid)
# 
# 
# # Use the random grid to search for best hyperparameters
# # First create the base model to tune
# rf = RandomForestRegressor()
# # Random search of parameters, using 3 fold cross validation, 
# # search across 100 different combinations, and use all available cores
# rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# 
# # Fit the random search model
# rf_random.fit(train_features, train_labels)
# 
# 
# # Evaluation of Random Search
# def evaluate(model, test_features, test_labels):
#     predictions = model.predict(test_features)
#     errors = abs(predictions - test_labels)
#     mape = 100 * np.mean(errors / test_labels)
#     accuracy = 100 - mape
#     print('Model Performance')
#     print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
#     print('Accuracy = {:0.2f}%.'.format(accuracy))
#     
#     return accuracy
# 
# base_model = RandomForestRegressor(n_estimators = 10, random_state = 42)
# base_model.fit(train_features, train_labels)
# base_accuracy = evaluate(base_model, test_features, test_labels)
# # =============================================================================
# # 
# # Model Performance
# # Average Error: 3.9199 degrees.
# # Accuracy = 93.36%.
# # =============================================================================
# 
# best_random = rf_random.best_estimator_
# random_accuracy = evaluate(best_random, test_features, test_labels)
# # =============================================================================
# # 
# # Model Performance
# # Average Error: 3.7152 degrees.
# # Accuracy = 93.73%.
# # =============================================================================
# print('\n')
# print('Base Accuracy: ', base_accuracy)
# print('\n')
# print('Random Accuracy: ', random_accuracy)
# print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))
# 
# =============================================================================

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} riders.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy

# GridSearchCV
    
# Completed and found results
    
# =============================================================================
# print('\n')
# print('Starting GridSearchCV')
# print('\n')
# from sklearn.model_selection import GridSearchCV
# import math
# 
# # Create the parameter grid based on the results of random search 
# # Param grid chosen based on Abhishek Thakur (Kaggle Grandmaster) blog post.
# param_grid = {
#     'bootstrap': [True],
#     'max_depth': [5, 8, 15, 25, 30, None],
#     'max_features': [3, None],
#     # log2 of 10 is 3.32 and sqrt 10 is 3.16 so max features will be 3 and none
#     'min_samples_leaf': [1, 2, 5, 10],
#     'min_samples_split': [2, 5, 10, 15, 100],
#     'n_estimators': [120, 300, 500, 800, 1200]
# }
# 
# # log2 of 10 is 3.32 and sqrt 10 is 3.16 so max features will be 3 and none
# # =============================================================================
# # 
# # param_grid = {
# #     'bootstrap': [True],
# #     'max_depth': [80, 90, 100, 110],
# # #    'max_features': [2, 3],
# #     'min_samples_leaf': [1, 3, 5],
# #     'min_samples_split': [2, 5, 10],
# #     'n_estimators': [10, 20, 30, 40]
# # }
# # 
# # =============================================================================
# 
# # Create a based model
# rf = RandomForestRegressor()
# 
# # Instantiate the grid search model
# grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
#                           cv = 3, n_jobs = 1, verbose = 2)
# # n_jobs = -1 is multithreading
# 
# # Fit the grid search to the data
# grid_search.fit(train_features, train_labels)
# 
# # =============================================================================
# best_param_set =  grid_search.best_params_
# # 
# # {'bootstrap': True,
# #  'max_depth': 80,
# #  'max_features': 3,
# #  'min_samples_leaf': 5,
# #  'min_samples_split': 12,
# #  'n_estimators': 100}
# # =============================================================================
# 
# best_grid = grid_search.best_estimator_
# grid_accuracy = evaluate(best_grid, test_features, test_labels)
# 
# # =============================================================================
# =============================================================================
# print('Model Performance')
# print('Accuracy', grid_accuracy)
# print('Improvement of {:0.2f}%.'.format( 100 * (grid_accuracy - base_accuracy) / base_accuracy))
# 
# =============================================================================

# =============================================================================
# # Best param set for random forest regression on Registered Users
# {'bootstrap': True,
#  'max_depth': 25,
#  'max_features': 3,
#  'min_samples_leaf': 1,
#  'min_samples_split': 2,
#  'n_estimators': 1200}
# =============================================================================



print('\n')
print('\n')
print('Best Random Forest with the searched parameters.')
# New random forest with only the two most important variables
rf_searched_param = RandomForestRegressor(bootstrap         = True,
                                          max_depth         =  25,
                                          max_features      = 3,
                                          min_samples_leaf  = 1,
                                          min_samples_split =  2,
                                          n_estimators      = 1200)

# Train the random forest
rf_searched_param.fit(train_features, train_labels);

# Make predictions and determine the error
predictions = rf_searched_param.predict(test_features)

errors = abs(predictions - test_labels)

# Display the performance metrics
print('Mean Absolute Error:', round(np.mean(errors), 2), 'riders.')

mape = np.mean(100 * (errors / test_labels))
accuracy = 100 - mape

# from sklearn.metrics import r2_score
r2_search = r2_score(test_labels, predictions)

# print('Accuracy of Searched param:', round(accuracy, 2), '%.')
print('\n')
print('R^2: ', r2_search)



# Import matplotlib for plotting and use magic command for Jupyter Notebooks
# import matplotlib.pyplot as plt

# %matplotlib inline For Jupytr notebook

# Set the style
plt.style.use('fivethirtyeight')

# list of x locations for plotting
x_values = list(range(len(importances)))

# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical')

# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical')

# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances in Searched Param Forest'); 
plt.show()
plt.clf()



# =============================================================================
# 
# import datetime
# 
# # Dates of training values
# months = features[:, feature_list.index('month')]
# days = features[:, feature_list.index('day')]
# years = features[:, feature_list.index('year')]
# 
# # List and then convert to datetime object
# dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
# dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
# 
# # Dataframe with true values and dates
# true_data = pd.DataFrame(data = {'date': dates, 'actual': labels})
# 
# # Dates of predictions
# months = test_features[:, feature_list.index('month')]
# days = test_features[:, feature_list.index('day')]
# years = test_features[:, feature_list.index('year')]
# 
# # Column of dates
# test_dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
# 
# # Convert to datetime objects
# test_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in test_dates]
# 
# # Dataframe with predictions and dates
# predictions_data = pd.DataFrame(data = {'date': test_dates, 'prediction': predictions}) 
# =============================================================================

# Dataframe with true values and dates
true_instants = features[:, feature_list.index('instant')]
true_data = pd.DataFrame(data = {'Instant': true_instants , 'actual': labels})


test_instants = test_features[:, feature_list.index('instant')]
predictions_data = pd.DataFrame(data = {'Instant': test_instants, 'prediction': predictions}) 
# Plot the actual values
plt.plot(true_data['Instant'], true_data['actual'], 'b-', label = 'actual')

# Plot the predicted values
x_ax_pred = range(0,183)
plt.plot(predictions_data['Instant'], predictions_data['prediction'], 'ro', label = 'prediction')
plt.xticks(rotation = '60'); 
plt.legend()

# Graph labels
plt.xlabel('Instant'); plt.ylabel('Maximum Registered Riders'); plt.title('Searched Param Actual and Predicted Values');
plt.show()
plt.clf()

print('\n')
print('\n')
print('\n')
print('\n')
print('\n')

# I still don't understand why we have a negative accuracy but that must be a mistake