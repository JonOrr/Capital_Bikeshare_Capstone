# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 15:04:52 2018

@author: Jon
"""
# Import packages
import os
import seaborn as sns
import numpy as np
import pandas as pd

# Import modules
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# Special matplotlib argument for improved plots
sns.set_style("whitegrid")
sns.set_context("poster")

# Import the Data

#Change the working directory
os.chdir('C:/Users/Jon/Documents/Springboard/Capstone1/Bike-Sharing-Dataset')

# Read in the csv's for day.csv and hour.csv
day_df = pd.read_csv('day.csv')

X= day_df[['temp', 'hum', 'windspeed']]
y = day_df.cnt # 0.7891097738747201
# y = day_df.registered # 0.8184668846027734
# y = day_df.casual # 0.6535791885825518


# Setup the pipeline steps: steps
steps = [('imputation', Imputer(missing_values='NaN', strategy='mean', axis=0)),
         ('scaler', StandardScaler()),
         ('elasticnet', ElasticNet(alpha = 0.01, max_iter=100000, copy_X=True, tol=0.00000001))]

# Create the pipeline: pipeline 
pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {'elasticnet__l1_ratio':np.linspace(0,1,101)}

No_year_day_df = day_df[['instant', 'season', 'mnth', 'holiday', 'weekday',
                         'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']]

X = No_year_day_df


# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 42)
# Create the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(pipeline, parameters)

# Fit to the training set
gm_cv.fit(X_train, y_train)

# Compute and print the metrics
r2 = gm_cv.score(X_test, y_test)
print('\n')
print("Tuned ElasticNet Alpha: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))
print('\n')

# y = day_df.cnt # 0.7891097738747201
# y = day_df.registered # 0.8184668846027734
# y = day_df.casual # 0.6535791885825518
