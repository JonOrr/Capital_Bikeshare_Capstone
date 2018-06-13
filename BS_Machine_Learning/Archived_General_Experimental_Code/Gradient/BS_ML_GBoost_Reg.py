# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 18:30:37 2018

Attempt at Gradient Boosted Regression (XGBoost)

Code adapted from sklearn Gradiant Boost Documentation Examples

http://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html#sphx-glr-auto-examples-ensemble-plot-gradient-boosting-regression-py

@author: Jon
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn import cross_validation, metrics   #Additional scklearn functions

import os
import seaborn as sns

# Import statsmodels and Ordinary Linear Regression

# Pull in sklearn modules
from sklearn.model_selection import train_test_split


# Special sns argument for improved plots
sns.set_style("whitegrid")
sns.set_context("poster")


# Import the Data

#Change the working directory
os.chdir('C:/Users/Jon/Documents/Springboard/Capstone1/Bike-Sharing-Dataset')

# Read in the csv's for day.csv and hour.csv
day_df = pd.read_csv('day.csv')


No_aTemp = day_df[['season', 'mnth', 'holiday', 'weekday',
       'workingday', 'weathersit', 'temp', 'hum', 'windspeed']]
Df = No_aTemp
y = day_df.registered

X,y = shuffle(Df, y)
# X,y = shuffle(Df, y, random_state = 42)
X = X.astype(np.float32)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3  , random_state =42)


offset = int(X.shape[0]*0.9)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]



# #############################################################################
# Fit regression model
params = {'n_estimators': 750, 'max_depth': 3, 'min_samples_split': 3,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(X_train, y_train)
mse = mean_squared_error(y_test, clf.predict(X_test))
print("MSE: %.4f" % mse)


# #############################################################################
# Plot training deviance

# compute test set deviance
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
acc_score  = np.zeros((params['n_estimators'],), dtype=np.float64)


for i, y_pred in enumerate(clf.staged_predict(X_test)):
    test_score[i] = clf.loss_(y_test, y_pred)
#     acc_score[i]  = metrics.accuracy_score(np.asarray(y_test), y_pred)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
#plt.plot(np.arange(params['n_estimators']) + 1, acc_score, 'g-', 
#          label = 'acc_score')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')

# #############################################################################
# Plot feature importance
feature_importance = clf.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
# plt.xticks(np.linspace(0,2000,11))
plt.yticks(pos, Df.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()


# Construct boosting with 500 iterations
XGb = ensemble.GradientBoostingRegressor(n_estimators = 600, max_depth = 3, min_samples_split = 2, learning_rate = 0.01, loss = 'ls')
XGb.fit(X_train, y_train)
XGb_mse = mean_squared_error(y_test, clf.predict(X_test))
XGb_y_pred = XGb.predict(X_test)


print("\nModel Report")
# print("Accuracy : %.4g" % metrics.accuracy_score(y_test, XGb_y_pred))
# AUC = metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob)
# print("AUC Score (Train): %f" % metrics.roc_auc_score(X_train, XGb_y_pred))