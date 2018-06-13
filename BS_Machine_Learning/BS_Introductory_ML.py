# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 14:11:24 2018

This file contains the machine learning for the BsDs (Bike sharing dataset)

@author: Jon
"""

# Import Numpy, Pandas, scipy-stats, matplotlib, sklearn, seaborn, and os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

# Import statsmodels and Ordinary Linear Regression
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Pull in sklearn modules
from sklearn.linear_model import LinearRegression

# Special matplotlib argument for improved plots
sns.set_style("whitegrid")
sns.set_context("poster")

# Import the Data

#Change the working directory
os.chdir('C:/Users/Jon/Documents/Springboard/Capstone1/Bike-Sharing-Dataset')

# Read in the csv's for day.csv and hour.csv
day_df = pd.read_csv('day.csv')
hour_df = pd.read_csv('hour.csv')

# Make an array for each table
day_array = np.array(day_df.values)
hour_array = np.array(hour_df.values)

# Follow the framework of the Boston housing exercise

# Describe the data set
day_df.keys()
day_df.shape

# Make the dataframe for day and look at the header and feature names
day_df.columns.values
day_df.head()
print(day_df.shape)

# Describe and scatterplot 
day_df.describe()

# Our first foray will involve a simple temperature vs count regression

# Two variable plots & Basic sns.regplots
sns.lmplot(x= 'temp', y = 'cnt', data = day_df, fit_reg = True)
plt.title('Seaborn regression plot of temp vs count across 2011-2012')
plt.show()
plt.clf()

sns.lmplot(x= 'temp', y = 'cnt', data = day_df, fit_reg = True, hue = 'yr')
plt.title('Seaborn regression plot of temp vs count across 2011-2012')
plt.show()
plt.clf()

sns.lmplot(x= 'temp', y = 'cnt', data = day_df, fit_reg = True, hue = 'season')
plt.title('Seaborn regression plot of temp vs count across 2011-2012')
plt.show()
plt.clf()

sns.lmplot(x= 'temp', y = 'cnt', data = day_df, fit_reg = True, hue = 'yr')
plt.title('Seaborn regression plot of temp vs count across 2011-2012')
plt.show()
plt.clf()


# Day of the week Temeprature Sensitivity
weekday_temp_lm = sns.lmplot(x="temp", y="cnt", hue="weekday", col="weekday",
                             data=day_df, aspect=.4, x_jitter=.1, size = 5)
plt.title('Day of the Week Temeprature Sensitivity')
plt.show()
plt.clf()

# Holiday Temperature sensitivity
sns.lmplot(x="temp", y="cnt", hue="holiday", col="holiday",
                             data=day_df, aspect=.4, x_jitter=.1, size = 5)
plt.title('Holiday Temperature Sensitivity')
plt.show()
plt.clf()


# Basic Histograms
plt.hist(day_df.cnt, bins = 50)
plt.title("Basic Rider Count Histogram")
plt.xlabel("Rider Count per day")
plt.xticks(np.linspace(0,9000, 10), rotation = 45)
plt.ylabel("Frequencey")
plt.show()

# OLS (Ordinary Least Squares) using statsmodels
m = ols('cnt ~ temp', day_df).fit()


# Print OLS summary
print(m.summary())
plt.scatter(m.fittedvalues, day_df.cnt)
plt.xlabel('Predicted Count')
plt.ylabel('Actual Count')
plt.title('Relationship between temp and overall count')
plt.show()
plt.clf()


# sklearn Linear regression
X_all = day_df.drop(['dteday','cnt', 'casual', 'registered'], axis = 1)
lm_all = LinearRegression()

# Fit the model 
print('Linear Regression for All Variables ')

lm_all.fit(X_all, day_df.cnt)
print('Estimated intercept coefficient: {}'.format(lm_all.intercept_))
print('Number of coefficients: {}'.format(len(lm_all.coef_)))
print(pd.DataFrame({'features': X_all.columns, 'estimatedCoefficients': lm_all.coef_})[['features', 'estimatedCoefficients']])
print('\n')
print('\n')

print('Linear Regression for Climate Variables')
X_climate = day_df[['temp', 'hum', 'windspeed']]
lm_climate = LinearRegression()
lm_climate.fit(X_climate, day_df.cnt)
print('Estimated intercept coefficient: {}'.format(lm_climate.intercept_))
print('Number of coefficients: {}'.format(len(lm_climate.coef_)))
print(pd.DataFrame({'features': X_climate.columns, 'estimatedCoefficients': lm_climate.coef_})[['features', 'estimatedCoefficients']])

# Show first 5 predicted user counts
lm_climate.predict(X_climate)[0:5]

# Histogram of predictions
plt.hist(lm_climate.predict(X_climate), bins = 50)
plt.title("Histogram of Predicted User Counts")
plt.xlabel("Predicted Count")
plt.ylabel("Frequencey")
plt.show()
plt.clf()

# Scatter plot of predictions lm.predict(X), target
plt.scatter(lm_climate.predict(X_climate), day_df.cnt)
plt.xlabel("Predicted User Count")
plt.ylabel("Actual User Count")
plt.title("Predicted User Count vs Actual User Count")

#plt.xticks(np.arange(0, 70, step=10))
#plt.yticks(np.arange(0, 70, step=10))

plt.plot(np.arange(0, 8000, step=10), np.arange(0, 8000, step=10), color = 'red')
plt.show()
plt.clf()

# Find the RSS (Residual Sum-of-Squares)
RSS_climate =  np.sum((day_df.cnt - lm_climate.predict(X_climate)) ** 2)
print('X_climate RSS: ', np.sum((day_df.cnt - lm_climate.predict(X_climate)) ** 2))

# Find the ESS (Explained Sum-Of-Squares)
ESS_climate = np.sum(lm_climate.predict(X_climate) - np.mean(day_df.cnt)) ** 2
print('X_climate ESS: ', np.sum(lm_climate.predict(X_climate) - np.mean(day_df.cnt)) ** 2)

# Try OLS with the 3 climate features (temp, windspeed, humidity)
m_climate = ols('cnt ~ temp + windspeed + hum', day_df).fit()
print(m_climate.summary())

# Construct Fitted versus residuals
plt.title('Fitted vs Residual')
plt.scatter(m_climate.fittedvalues, m_climate.resid)
plt.xlabel('Predicted Count')
plt.ylabel('Residuals')
plt.show()
plt.clf()

# Constructs statsmodels qq plots
sm.qqplot(m_climate.resid, line = 'q')
plt.show()
plt.clf()
# Make a leverage residuals plot
# fig = sm.graphics.plot_leverage_resid2(m_climate,alpha=0.05, ax=None )
sm.graphics.plot_leverage_resid2(m_climate,alpha=0.05, ax=None )
plt.show()
plt.clf()

# Make a clean set 
day_clean = day_df.drop(day_df.index[ [49, 68, 667, 238, 202, 203] ] )


# Make clean fitted versus residuals plot
m_climate_clean = ols('cnt ~ temp + windspeed + hum', day_clean).fit()
print(m_climate_clean.summary())

# Make qq plots for clean sets. 
plt.scatter(m_climate_clean.fittedvalues, m_climate_clean.resid)
plt.title('Cleaned dataframe Fitted vs. Residual')
plt.xlabel('Predicted Count')
plt.ylabel('Residuals')
plt.show()
plt.clf()


# -------------------------------------
# -------------------------------------

# Do it again with the Hourly data


# -------------------------------------
# -------------------------------------

# Follow the framework of the Boston housing exercise

# Describe the data set
hour_df.keys()
hour_df.shape

# Make the dataframe for day and look at the header and feature names
hour_df.columns.values
hour_df.head()
print(hour_df.shape)

# Describe and scatterplot 
hour_df.describe()

# Our first foray will involve a simple temperature vs count regression

# Two variable plots & Basic sns.regplots
sns.lmplot(x= 'temp', y = 'cnt', data = hour_df, fit_reg = True)
plt.title('Seaborn regression plot of temp vs count across 2011-2012 Hours')
plt.show()
plt.clf()

sns.lmplot(x= 'temp', y = 'cnt', data = hour_df, fit_reg = True, hue = 'yr')
plt.title('Seaborn regression plot of temp vs count across 2011-2012 Hours')
plt.show()
plt.clf()

sns.lmplot(x= 'temp', y = 'cnt', data = hour_df, fit_reg = True, hue = 'season')
plt.title('Seaborn regression plot of temp vs count across 2011-2012 Hours')
plt.show()
plt.clf()

sns.lmplot(x= 'temp', y = 'cnt', data = hour_df, fit_reg = True, hue = 'yr')
plt.title('Seaborn regression plot of temp vs count across 2011-2012 Hours')
plt.show()
plt.clf()


# Day of the week Temeprature Sensitivity
weekday_temp_lm = sns.lmplot(x="temp", y="cnt", hue="weekday", col="weekday",
                             data=hour_df, aspect=.4, x_jitter=.1, size = 5)
plt.title('Day of the Week Temeprature Sensitivity')
plt.show()
plt.clf()

# Holiday Temperature sensitivity
sns.lmplot(x="temp", y="cnt", hue="holiday", col="holiday",
                             data=hour_df, aspect=.4, x_jitter=.1, size = 5)
plt.title('Holiday Temperature Sensitivity')
plt.show()
plt.clf()


# Basic Histograms
plt.hist(hour_df.cnt, bins = 50)
plt.title("Basic Rider Count Histogram")
plt.xlabel("Rider Count per hour")
plt.xticks(np.linspace(0,1000, 11), rotation = 45)
plt.ylabel("Frequencey")
plt.show()

# OLS (Ordinary Least Squares) using statsmodels
m = ols('cnt ~ temp', hour_df).fit()


# Print OLS summary
print(m.summary())
plt.scatter(m.fittedvalues, hour_df.cnt)
plt.xlabel('Predicted Count')
plt.ylabel('Actual Count')
plt.title('Relationship between temp and overall count per hour')
plt.show()
plt.clf()


# sklearn Linear regression
X_all = hour_df.drop(['dteday','cnt'], axis = 1)
lm_all = LinearRegression()

# Fit the model 
print('Linear Regression for All Variables \'Purely Academic\' ')

lm_all.fit(X_all, hour_df.cnt)
print('Estimated intercept coefficient: {}'.format(lm_all.intercept_))
print('Number of coefficients: {}'.format(len(lm_all.coef_)))
print(pd.DataFrame({'features': X_all.columns, 'estimatedCoefficients': lm_all.coef_})[['features', 'estimatedCoefficients']])
print('\n')
print('\n')

print('Linear Regression for Climate Variables')
X_climate = hour_df[['temp', 'hum', 'windspeed']]
lm_climate = LinearRegression()
lm_climate.fit(X_climate, hour_df.cnt)
print('Estimated intercept coefficient: {}'.format(lm_climate.intercept_))
print('Number of coefficients: {}'.format(len(lm_climate.coef_)))
print(pd.DataFrame({'features': X_climate.columns, 'estimatedCoefficients': lm_climate.coef_})[['features', 'estimatedCoefficients']])

# Show first 5 predicted user counts
lm_climate.predict(X_climate)[0:5]

# Histogram of predictions
plt.hist(lm_climate.predict(X_climate), bins = 50)
plt.title("Histogram of Predicted User Counts")
plt.xlabel("Predicted Count")
plt.ylabel("Frequencey")
plt.show()
plt.clf()

# Scatter plot of predictions lm.predict(X), target
plt.scatter(lm_climate.predict(X_climate), hour_df.cnt)
plt.xlabel("Predicted User Count")
plt.ylabel("Actual User Count")
plt.title("Predicted User Count vs Actual User Count")

#plt.xticks(np.arange(0, 70, step=10))
#plt.yticks(np.arange(0, 70, step=10))

plt.plot(np.arange(0, 1000, step=10), np.arange(0, 1000, step=10), color = 'red')
plt.show()
plt.clf()

# Find the RSS (Residual Sum-of-Squares)
RSS_climate =  np.sum((hour_df.cnt - lm_climate.predict(X_climate)) ** 2)
print('X_climate RSS: ', np.sum((hour_df.cnt - lm_climate.predict(X_climate)) ** 2))

# Find the ESS (Explained Sum-Of-Squares)
ESS_climate = np.sum(lm_climate.predict(X_climate) - np.mean(hour_df.cnt)) ** 2
print('X_climate ESS: ', np.sum(lm_climate.predict(X_climate) - np.mean(hour_df.cnt)) ** 2)

# Try OLS with the 3 climate features (temp, windspeed, humidity)
m_climate = ols('cnt ~ temp + windspeed + hum', hour_df).fit()
print(m_climate.summary())

# Construct Fitted versus residuals
plt.title('Fitted vs Residual')
plt.scatter(m_climate.fittedvalues, m_climate.resid)
plt.xlabel('Predicted Count')
plt.ylabel('Residuals')
plt.show()
plt.clf()
# Constructs statsmodels qq plots
sm.qqplot(m_climate.resid, line = 'q')
plt.show()
plt.clf()

# Make a leverage residuals plot
# fig = sm.graphics.plot_leverage_resid2(m_climate,alpha=0.05, ax=None )
sm.graphics.plot_leverage_resid2(m_climate,alpha=0.05, ax=None )
plt.show()
plt.clf()

# Make a clean set 
hour_clean = hour_df.drop(hour_df.index[ [5635, 4335, 4116] ] )

# Make clean fitted versus residuals plot
m_climate_clean = ols('cnt ~ temp + windspeed + hum', hour_clean).fit()
print(m_climate_clean.summary())

# Make qq plots for clean sets. 
plt.scatter(m_climate_clean.fittedvalues, m_climate_clean.resid)
plt.title('Cleaned Hour Dataframe Fitted vs. Residual')
plt.xlabel('Predicted Count')
plt.ylabel('Residuals')
plt.show()
plt.clf()


# Now let's try some other ML
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_climate = day_df[['temp', 'hum', 'windspeed']]
y = np.asarray(day_df.cnt)

# Split the data into a training and test set.
X_train, X_test, y_train, y_test = train_test_split(X_climate, y, test_size = 0.25)

# Instantiate the model
logreg = LogisticRegression()

# Fit the model on the trainng data.
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)





# Now to try Elastic net
# Import necessary modules
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

X_climate = day_df[['temp', 'hum', 'windspeed']]
X = X_climate

y = np.asarray(day_df.cnt)


# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 42)

# Create the hyperparameter grid
l1_space = np.linspace(0, 1, 1000)
param_grid = {'l1_ratio': l1_space}

# Instantiate the ElasticNet regressor: elastic_net
elastic_net = ElasticNet()

# Setup the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(elastic_net, param_grid, cv=5)

# Fit it to the training data
gm_cv.fit(X_train, y_train)

# Predict on the test set and compute metrics
y_pred = gm_cv.predict(X_test)
r2 = gm_cv.score(X_test, y_test)
mse = mean_squared_error(y_test, y_pred)
print("Tuned ElasticNet l1 ratio: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))
print("Tuned ElasticNet MSE: {}".format(mse))


#=================================================