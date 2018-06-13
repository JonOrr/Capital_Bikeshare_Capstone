# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 00:55:19 2018

Adaptation of William Koehrsen's article

Random Forest in Python an End to End Machine Learning Example

https://towardsdatascience.com/random-forest-in-python-24d0893d51c0


@author: Jon
"""

import matplotlib.pyplot as plt
import pandas as pd

from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn import cross_validation, metrics   #Additional scklearn functions

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
print('\n')
print('Accuracy:', round(accuracy, 2), '%.')


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
accuracy = 100 - mape

print('Accuracy:', round(accuracy, 2), '%.')
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
