# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 11:11:57 2018

Casual User Lasso

@author: Jon
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

# Lasso Regression
from sklearn.linear_model import Lasso


# Lasso Regression for all casual users over both years and all seasons

X= day_df[['temp', 'hum', 'windspeed']]
y = day_df.casual # 0.6535791885825518

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state =42)
# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3)


lasso = Lasso(alpha = 0.1)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)
Lscore = lasso.score(X_test, y_test)
print('L-Score: ', Lscore)


names = X.columns
# Lasso(lasso_eps = 0.0001,
#                  lasso_alpha = 1,
#                  lasso_iter = 5000)
lassoName = Lasso(alpha = 0.1, max_iter = 100000000, tol = 1e-20)
lassoName_coef = lassoName.fit(X_train, y_train).coef_
_ = plt.plot(range(len(names)), lassoName_coef)
_ = plt.title('Lasso Feature Importance on Climate Variables')
_ = plt.xticks(range(len(names)), names, rotation = 60)
_ = plt.yticks(np.linspace(-7500, 7500, 7))
_ = plt.ylabel('Coefficients')
plt.show()
plt.clf()
Lscore = lasso.score(X_test, y_test)
print('L-Score for Climate: ', Lscore)

# Try again with all variables sans date, cnt, registered, and casual
No_Date_day_df = day_df[['instant', 'season', 'yr', 'mnth', 'holiday', 'weekday',
       'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']]
X = No_Date_day_df

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state =42)


namesAll = X.columns
lassoAll = Lasso(alpha = 0.1, max_iter = 100000000, tol = 1e-20)
lassoAll_coef = lassoAll.fit(X_train, y_train).coef_
_ = plt.plot(range(len(namesAll)), lassoAll_coef)
_ = plt.title('Lasso Feature Importance on all Non-Count Variables')
_ = plt.xticks(range(len(namesAll)), namesAll, rotation = 60)
_ = plt.yticks(np.linspace(-6000, 6000, 11))
_ = plt.ylabel('Coefficients')
plt.show()
plt.clf()
Lscore = lassoAll.score(X_test, y_test)
print('L-Score for no date: ', Lscore)



# This plot shows us that year was actually the biggest determining factor, but
# this is just due to growth of the product, we try again dropping year. 
No_year_day_df = day_df[['instant', 'season', 'mnth', 'holiday', 'weekday',
       'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']]
X = No_year_day_df
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3  , random_state =42)
# , random_state =42


names = X.columns
lasso = Lasso(alpha = 0.4, max_iter = 100000000, tol = 1e-20)
lasso_coef = lasso.fit(X_train, y_train).coef_
_ = plt.plot(range(len(names)), lasso_coef)
_ = plt.title('Lasso Feature Importance on Non-Year Non-Count Variables')
_ = plt.xticks(range(len(names)), names, rotation = 45)
_ = plt.yticks(np.linspace(-7500, 7500, 11))
_ = plt.ylabel('Coefficients')
_ = plt.margins(0.05)
plt.show()
plt.clf()
Lscore = lasso.score(X_test, y_test)
print('L-Score for no year: ', Lscore)

# This shows us that atemp is by far the most important feature
# We look at this plot again without atemp to get a better view
print('\n')
print('\n')


No_aTemp = day_df[['instant', 'season', 'mnth', 'holiday', 'weekday',
       'workingday', 'weathersit', 'temp', 'hum', 'windspeed']]
X = No_aTemp
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3  , random_state =42)
# , random_state =42


names = X.columns
lasso = Lasso(alpha = 0.4, max_iter = 100000000, tol = 1e-20)
lasso_coef = lasso.fit(X_train, y_train).coef_
_ = plt.plot(range(len(names)), lasso_coef)
_ = plt.title('Lasso Feature Importance on No Atemp')
_ = plt.xticks(range(len(names)), names, rotation = 45)
_ = plt.yticks(np.linspace(-7500, 7500, 11))
_ = plt.ylabel('Coefficients')
_ = plt.margins(0.05)
plt.show()
plt.clf()
Lscore = lasso.score(X_test, y_test)
print('L-Score for no aTemp: ', Lscore)

# Now without temp
No_Temp = day_df[['instant', 'season', 'mnth', 'holiday', 'weekday',
       'workingday', 'weathersit', 'hum', 'windspeed']]
X = No_Temp
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3  , random_state =42)


names = X.columns
lasso = Lasso(alpha = 0.4, max_iter = 100000000, tol = 1e-20)
lasso_coef = lasso.fit(X_train, y_train).coef_
_ = plt.plot(range(len(names)), lasso_coef)
_ = plt.title('Lasso Feature Importance on No Temp')
_ = plt.xticks(range(len(names)), names, rotation = 45)
_ = plt.yticks(np.linspace(-3000, 1000, 11))
_ = plt.ylabel('Coefficients')
_ = plt.margins(0.05)
plt.show()
plt.clf()
Lscore = lasso.score(X_test, y_test)
print('L-Score for no Temp: ', Lscore)

# Now without windspeed
No_Wind = day_df[['instant', 'season', 'mnth', 'holiday', 'weekday',
       'workingday', 'weathersit', 'hum']]
X = No_Wind
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3  , random_state =42)


names = X.columns
lasso = Lasso(alpha = 0.4, max_iter = 100000000, tol = 1e-20)
lasso_coef = lasso.fit(X_train, y_train).coef_
_ = plt.plot(range(len(names)), lasso_coef)
_ = plt.title('Lasso Feature Importance on No Windspeed')
_ = plt.xticks(range(len(names)), names, rotation = 45)
_ = plt.yticks(np.linspace(-1500, 1500, 11))
_ = plt.ylabel('Coefficients')
_ = plt.margins(0.05)
plt.show()
plt.clf()
Lscore = lasso.score(X_test, y_test)
print('L-Score for no Wind: ', Lscore)

# Instant, Month and weekday can all be dropped from the model 


# Full model 
Full_df = day_df[['season','holiday',
       'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']]
X = Full_df
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3  , random_state =42)


names = X.columns
lasso = Lasso(alpha = 0.4, max_iter = 100000000, tol = 1e-20)
lasso_coef = lasso.fit(X_train, y_train).coef_
_ = plt.plot(range(len(names)), lasso_coef)
_ = plt.title('Lasso Feature Importance on Full model')
_ = plt.xticks(range(len(names)), names, rotation = 45)
_ = plt.yticks(np.linspace(-7000, 7000, 11))
_ = plt.ylabel('Coefficients')
_ = plt.margins(0.05)
plt.show()
plt.clf()
Lscore = lasso.score(X_test, y_test)
print('L-Score for Full: ', Lscore)
# Why does removing two no importance variables make the score plummet?




