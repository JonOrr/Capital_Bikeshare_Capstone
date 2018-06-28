# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 14:40:09 2018

Ridge Regression 

@author: Jonathan Orr
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error


# Special matplotlib argument for improved plots
sns.set_style("whitegrid")
sns.set_context("poster")


# Import the Data

#Change the working directory
os.chdir('C:/Users/Jon/Documents/Springboard/Capstone1/Bike-Sharing-Dataset')

# Read in the csv's for day.csv and hour.csv
day_df = pd.read_csv('day.csv')


X= day_df[['temp', 'hum', 'windspeed']]

# y = np.asarray(day_df.cnt)
y = day_df.cnt # 0.7891097738747201
# y = day_df.registered # 0.8184668846027734
# y = day_df.casual # 0.6535791885825518

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state =42)





def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.25)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()
    

# Ridge Regression
# Setup the array of alphas and lists to store scores
alpha_space = np.logspace(-2, 2, 100)
ridge_scores = []
ridge_scores_std = []

# Create a ridge regressor: ridge
ridge = Ridge(normalize = True, tol = 0.00001, solver = 'auto')

# Compute scores over range of alphas
for alpha in alpha_space:

    # Specify the alpha value to use: ridge.alpha
    ridge.alpha = alpha
    
    # Perform 10-fold CV: ridge_cv_scores
    ridge_cv_scores = cross_val_score(ridge, X_train, y_train, cv = 10)
    
    # Append the mean of ridge_cv_scores to ridge_scores
    ridge_scores.append(np.mean(ridge_cv_scores))
    
    # Append the std of ridge_cv_scores to ridge_scores_std
    ridge_scores_std.append(np.std(ridge_cv_scores))

# Display the plot
display_plot(ridge_scores, ridge_scores_std)





ridge = Ridge(alpha = 0.1, normalize = True)

ridge.fit(X_train, y_train)

ridge_pred = ridge.predict(X_test)

Rscore = ridge.score(X_test, y_test)

print('R-Score: ', Rscore)
print("R^2: {}".format(ridge.score(X_test, y_test)))

errors = abs(y_test - ridge_pred)
mse = round(np.mean(errors), 2)

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))
print("Root Mean Squared Error: {}".format(rmse))
print('Mean Absolute Error: ', mse)
print('\n')
