# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 20:07:48 2018

Adaptation of: 
    
A comprehensive beginners guide for Linear, Ridge and Lasso Regression
Shubham Jain

https://www.analyticsvidhya.com/blog/2017/06/a-comprehensive-guide-for-linear-ridge-and-lasso-regression/

@author: Jon
"""

# importing basic libraries

import numpy as np

import pandas as pd

from pandas import Series, DataFrame

from sklearn.model_selection import train_test_split

# import test and train file

# train = pd.read_csv('Train.csv')

# test = pd.read_csv('test.csv')

import os
os.chdir('C:/Users/Jon/Documents/Springboard/Capstone1/Bike-Sharing-Dataset')

# Read in the csv's for day.csv and hour.csv
day_df = pd.read_csv('day.csv')


No_aTemp = day_df[['season', 'mnth', 'holiday', 'weekday',
       'workingday', 'weathersit', 'temp', 'hum', 'windspeed']]
No_aTemp = day_df[['season', 'mnth', 'holiday', 'weekday',
       'workingday', 'weathersit', 'temp', 'hum', 'windspeed']]
Df = No_aTemp
target = day_df.registered

from sklearn.utils import shuffle


Xs,ys = shuffle(Df, target)
# X,y = shuffle(Df, y, random_state = 42)
Xs = Xs.astype(np.float32)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3  , random_state =42)


offset = int(Xs.shape[0]*0.85)  # Create train / test
x_train, y_train = Xs[:offset], ys[:offset]
train = x_train

x_test, y_test = Xs[offset:], ys[offset:]
test = x_test


# importing linear regressionfrom sklearn

from sklearn.linear_model import LinearRegression

lreg = LinearRegression()

# splitting into training and cv for cross validation

x_train, x_cv, y_train, y_cv = train_test_split(x_train ,y_train)

# training the model

lreg.fit(x_train,y_train)

# predicting on cv

pred = lreg.predict(x_cv)

# calculating mse

mse_1 = np.mean((pred - y_cv)**2)
print('\n')
print('mse_score_1 is: ', mse_1)

# calculating coefficients

coeff = DataFrame(x_train.columns)

coeff['Coefficient Estimate'] = Series(lreg.coef_)

# Write in console: coeff
# Temp is the most positively  variable
# Windspeed is the most negative variable

l_reg_score_1 = lreg.score(x_cv,y_cv) # 0.573
print('L_reg_score_1 is: ', l_reg_score_1)

# imputing missing values. We don't need these steps
# =============================================================================
# 
# train['Item_Visibility'] = train['Item_Visibility'].replace(0,np.mean(train['Item_Visibility']))
# 
# train['Outlet_Establishment_Year'] = 2013 - train['Outlet_Establishment_Year']
# 
# train['Outlet_Size'].fillna('Small',inplace=True)
# 
# 
# 
# # creating dummy variables to convert categorical into numeric values
# 
# mylist = list(train.select_dtypes(include=['object']).columns)
# 
# dummies = pd.get_dummies(train[mylist], prefix= mylist)
# 
# train.drop(mylist, axis=1, inplace = True)
# 
# X = pd.concat([train,dummies], axis =1 )
# 
# =============================================================================



# Rebuild the model and resid plot.
offset = int(Xs.shape[0]*0.85)  # Create train / test
x_train, y_train = Xs[:offset], ys[:offset]
train = x_train

x_test, y_test = Xs[offset:], ys[offset:]
test = x_test


# importing linear regression

# from sklearn from sklearn.linear_model import LinearRegression

lreg = LinearRegression()

# for cross validation

from sklearn.model_selection import train_test_split



x_train, x_cv, y_train, y_cv = train_test_split(Xs,ys, test_size =0.3)

# training a linear regression model on train

lreg.fit(x_train,y_train)

# predicting on cv

pred_cv = lreg.predict(x_cv)

# calculating mse

mse_2 = np.mean((pred_cv - y_cv)**2)
print('\n')
print('mse_score_2 is: ', mse_2)

# evaluation using r-square

l_reg_score_2 = lreg.score(x_cv,y_cv)

print('L_reg_score_2 is: ', l_reg_score_2)

import matplotlib.pyplot as plt

x_plot = plt.scatter(pred_cv, (pred_cv - y_cv), c='b')

plt.hlines(y=0, xmin= -1000, xmax=5000)

plt.title('Residual plot')
print('\n')

# Regularization

# checking the magnitude of coefficients

predictors = x_train.columns

coef = Series(lreg.coef_,predictors).sort_values()

coef.plot(kind='bar', title='Modal Coefficients')


# Ridge Regression
from sklearn.linear_model import Ridge

## training the model

ridgeReg = Ridge(alpha=0.05, normalize=True)

ridgeReg.fit(x_train,y_train)

pred = ridgeReg.predict(x_cv)

# calculating mse

mse_3 = np.mean((pred_cv - y_cv)**2)

print('Ridge mse is: ', mse_3)
## calculating score 
score_3 = ridgeReg.score(x_cv,y_cv)
print('Ridge Score is: ', score_3)


# Lasso

from sklearn.linear_model import Lasso

lassoReg = Lasso(alpha=0.3, normalize=True)

lassoReg.fit(x_train,y_train)

pred = lassoReg.predict(x_cv)

# calculating mse

mse_4 = np.mean((pred_cv - y_cv)**2)
print('\n')
print('Lasso mse is: ', mse_4)


score_4 = lassoReg.score(x_cv,y_cv)
print('Lasso_score is: ', score_4)



# Elastic Net

from sklearn.linear_model import ElasticNet

ENreg = ElasticNet(alpha=1, l1_ratio=0.5, normalize=False)

ENreg.fit(x_train,y_train)

pred_cv = ENreg.predict(x_cv)

#calculating mse

mse_5 = np.mean((pred_cv - y_cv)**2)
print('\n')
print('Elastic Net mse is : ', mse_5)

score_5 =  ENreg.score(x_cv,y_cv)
print('Elastic Net Score is: ', score_5)
