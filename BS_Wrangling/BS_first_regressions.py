# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 19:42:05 2018

Some initial attempts at regression

@author: Jon
"""

# Import common packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

# Change the working directory
# ATTN: You will need to change this locally.
os.chdir('C:/Users/Jon/Documents/Springboard/Capstone1/Bike-Sharing-Dataset')

# Read in the csv's for day.csv and hour.csv
day_df = pd.read_csv('day.csv')
hour_df = pd.read_csv('hour.csv')

# Make an array for each table
day_array = np.array(day_df.values)
hour_array = np.array(hour_df.values)


# Create a list of column names for quick reference
Dcolumn_names = list(day_df.columns.values)
Hcolumn_names = list(hour_df.columns.values)

# Sort into seasonal dataframes
Spring_day_df = day_df.loc[day_df['season'] == 1]
Summer_day_df = day_df.loc[day_df['season'] == 2]
Fall_day_df = day_df.loc[day_df['season'] == 3]
Winter_day_df = day_df.loc[day_df['season'] == 4]

#Make annual data frames
day_df_2011 = day_df.loc[day_df['yr'] == 0]
day_df_2012 = day_df.loc[day_df['yr'] == 1]

# We only have a few variables that makes sense for regression
# temp (temperature) scaled by 1/41
# hum (humidity) scaled by 1/100
# windspeed (Normalized wind speed. The values are divided to 67 (max))

# Temperature and count for 2011
# Linear regression of temp and cnt
sns.lmplot(x= 'temp', y = 'cnt', data = day_df_2011)
plt.title('Seaborn regression plot of temp vs count in 2011')
plt.show()
plt.clf()


# Residual plot of temp and count for 2011
sns.residplot(x= 'temp', y = 'cnt', data = day_df_2011)
plt.title('Seaborn residual plot of temp vs count in 2011')
plt.show()
plt.clf()


# Temperature and count for 2012
# Linear regression of temp and count
sns.lmplot(x= 'temp', y = 'cnt', data = day_df_2012)
plt.title('Seaborn regression plot of temp vs count in 2012')
plt.show()
plt.clf()


# Residual plot of temp and count for 2012
sns.residplot(x= 'temp', y = 'cnt', data = day_df_2012)
plt.title('Seaborn residual plot of temp vs count in 2012')
plt.show()
plt.clf()



print('Beginning of 2011 temperature plots')

# 2011 seasonal regressions and residuals for temperature vs count
Spring2011_df = day_df_2011.loc[day_df_2011['season'] == 1]
Summer2011_df = day_df_2011.loc[day_df_2011['season'] == 2]
Fall2011_df = day_df_2011.loc[day_df_2011['season'] == 3]
Winter2011_df = day_df_2011.loc[day_df_2011['season'] == 4]

# 2011 Spring
# 2011 Spring regression
sns.lmplot(x = 'temp', y = 'cnt', data = Spring2011_df)
plt.title('Normalized Seaborn regression plot of temp vs count in Spring 2011')
plt.show()
plt.clf()

# 2011 Spring residual
sns.residplot(x = 'temp', y = 'cnt', data = Spring2011_df)
plt.title('Normalized Seaborn residual plot of temp vs count in Spring 2011')
plt.show()
plt.clf()


# 2011 Summer
# 2011 Summer regression
sns.lmplot(x = 'temp', y = 'cnt', data = Summer2011_df)
plt.title('Normalized Seaborn regression plot of temp vs count in Summer 2011')
plt.show()
plt.clf()

# 2011 Summer residual
sns.residplot(x = 'temp', y = 'cnt', data = Summer2011_df)
plt.title('Normalized Seaborn residual plot of temp vs count in Summer 2011')
plt.show()
plt.clf()


# 2011 Fall
# 2011 Fall regression
sns.lmplot(x = 'temp', y = 'cnt', data = Fall2011_df)
plt.title('Normalized Seaborn regression plot of temp vs count in Fall 2011')
plt.show()
plt.clf()


# 2011 Fall residual
sns.residplot(x = 'temp', y = 'cnt', data = Fall2011_df)
plt.title('Normalized Seaborn residual plot of temp vs count in Fall 2011')
plt.show()
plt.clf()


# 2011 Winter
# 2011 Winter regression
sns.lmplot(x = 'temp', y = 'cnt', data = Winter2011_df)
plt.title('Normalized Seaborn regression plot of temp vs count in Winter 2011')
plt.show()
plt.clf()


# 2011 Winter residual
sns.residplot(x = 'temp', y = 'cnt', data = Winter2011_df)
plt.title('Normalized Seaborn residual plot of temp vs count in Winter 2011')
plt.show()
plt.clf()


# 2012 seasonal regressions and residuals for temperature vs count

Spring2012_df = day_df_2012.loc[day_df_2012['season'] == 1]
Summer2012_df = day_df_2012.loc[day_df_2012['season'] == 2]
Fall2012_df = day_df_2012.loc[day_df_2012['season'] == 3]
Winter2012_df = day_df_2012.loc[day_df_2012['season'] == 4]


# 2012 Spring
# 2012 Spring regression
sns.lmplot(x = 'temp', y = 'cnt', data = Spring2012_df)
plt.title('Normalized Seaborn regression plot of temp vs count in Spring 2012')
plt.show()
plt.clf()

# 2012 Spring residual
sns.residplot(x = 'temp', y = 'cnt', data = Spring2012_df)
plt.title('Normalized Seaborn residual plot of temp vs count in Spring 2012')
plt.show()
plt.clf()


print('Beginning of 2012 temperature plots')


# 2012 Summer
# 2012 Summer regression
sns.lmplot(x = 'temp', y = 'cnt', data = Summer2012_df)
plt.title('Normalized Seaborn regression plot of temp vs count in Summer 2012')
plt.show()
plt.clf()

# 2012 Summer residual
sns.residplot(x = 'temp', y = 'cnt', data = Summer2012_df)
plt.title('Normalized Seaborn residual plot of temp vs count in Summer 2012')
plt.show()
plt.clf()


# 2012 Fall
# 2012 Fall regression
sns.lmplot(x = 'temp', y = 'cnt', data = Fall2012_df)
plt.title('Normalized Seaborn regression plot of temp vs count in Fall 2012')
plt.show()
plt.clf()


# 2012 Fall residual
sns.residplot(x = 'temp', y = 'cnt', data = Fall2012_df)
plt.title('Normalized Seaborn residual plot of temp vs count in Fall 2012')
plt.show()
plt.clf()


# 2012 Winter
# 2012 Winter regression
sns.lmplot(x = 'temp', y = 'cnt', data = Winter2012_df)
plt.title('Normalized Seaborn regression plot of temp vs count in Winter 2012')
plt.show()
plt.clf()


# 2012 Winter residual
sns.residplot(x = 'temp', y = 'cnt', data = Winter2012_df)
plt.title('Normalized Seaborn residual plot of temp vs count in Winter 2012')
plt.show()
plt.clf()

# =============================================================================
# Humidity regressions
# =============================================================================

# Humidity and count for 2011
# Linear regression of hum and cnt
sns.lmplot(x= 'hum', y = 'cnt', data = day_df_2011)
plt.title('Seaborn regression plot of humidity vs cnt in 2011')
plt.show()
plt.clf()


# Residual plot of hum and count for 2011
sns.residplot(x= 'hum', y = 'cnt', data = day_df_2011)
plt.title('Seaborn residual plot of humidity vs count in 2011')
plt.show()
plt.clf()


# Humidity and count for 2012
# Linear regression of hum and count
sns.lmplot(x= 'hum', y = 'cnt', data = day_df_2012)
plt.title('Seaborn regression plot of humidity vs count in 2012')
plt.show()
plt.clf()


# Residual plot of hum and count for 2012
sns.residplot(x= 'hum', y = 'cnt', data = day_df_2012)
plt.title('Seaborn residual plot of humidity vs count in 2012')
plt.show()
plt.clf()

# Result:
# There is very little correlation between humidity and count



# Trying some more complicated seaborn plots with temperature
sns.lmplot(x= 'temp', y = 'cnt', data = day_df_2011, fit_reg=True)
plt.title('Seaborn regression plot of temp vs cnt in 2011: fit_reg = True')
plt.show()
plt.clf()


# Try order 2
sns.lmplot(x= 'temp', y = 'cnt', data = day_df_2011, fit_reg=True, order = 2, ci = 90)
plt.title('Seaborn regression plot of temp vs cnt in 2011: fit_reg = True')
plt.show()
plt.clf()







# =============================================================================
# Windspeed regressions
# =============================================================================


# Windspeed and count for 2011
# Linear regression of windspeed and cnt
sns.lmplot(x= 'windspeed', y = 'cnt', data = day_df_2011)
plt.title('Seaborn regression plot of windspeed vs cnt in 2011')
plt.show()
plt.clf()


# Residual plot of windspeed and count for 2011
sns.residplot(x= 'windspeed', y = 'cnt', data = day_df_2011)
plt.title('Seaborn residual plot of windspeed vs count in 2011')
plt.show()
plt.clf()


# Humidity and count for 2012
# Linear regression of windspeed and count
sns.lmplot(x= 'windspeed', y = 'cnt', data = day_df_2012)
plt.title('Seaborn regression plot of windspeed vs count in 2012')
plt.show()
plt.clf()


# Residual plot of windspeed and count for 2012
sns.residplot(x= 'windspeed', y = 'cnt', data = day_df_2012)
plt.title('Seaborn residual plot of windspeed vs count in 2012')
plt.show()
plt.clf()

# Result:
# There is little correlation between windspeed and bike usage, until 
# windspeeds reach



# =============================================================================
# aTemp (Feels like temp) regressions , scaled by 1/50
# =============================================================================
print('Beginning of 2011 Atemp plots')

# 2011 seasonal regressions and residuals for atemperature vs count
Spring2011_df = day_df_2011.loc[day_df_2011['season'] == 1]
Summer2011_df = day_df_2011.loc[day_df_2011['season'] == 2]
Fall2011_df = day_df_2011.loc[day_df_2011['season'] == 3]
Winter2011_df = day_df_2011.loc[day_df_2011['season'] == 4]

# 2011 Spring
# 2011 Spring regression
sns.lmplot(x = 'atemp', y = 'cnt', data = Spring2011_df)
plt.title('Normalized Seaborn regression plot of Atemp vs count in Spring 2011')
plt.show()
plt.clf()

# 2011 Spring residual
sns.residplot(x = 'atemp', y = 'cnt', data = Spring2011_df)
plt.title('Normalized Seaborn residual plot of Atemp vs count in Spring 2011')
plt.show()
plt.clf()


# 2011 Summer
# 2011 Summer regression
sns.lmplot(x = 'atemp', y = 'cnt', data = Summer2011_df)
plt.title('Normalized Seaborn regression plot of Atemp vs count in Summer 2011')
plt.show()
plt.clf()

# 2011 Summer residual
sns.residplot(x = 'atemp', y = 'cnt', data = Summer2011_df)
plt.title('Normalized Seaborn residual plot of Atemp vs count in Summer 2011')
plt.show()
plt.clf()


# 2011 Fall
# 2011 Fall regression
sns.lmplot(x = 'atemp', y = 'cnt', data = Fall2011_df)
plt.title('Normalized Seaborn regression plot of Atemp vs count in Fall 2011')
plt.show()
plt.clf()


# 2011 Fall residual
sns.residplot(x = 'atemp', y = 'cnt', data = Fall2011_df)
plt.title('Normalized Seaborn residual plot of Atemp vs count in Fall 2011')
plt.show()
plt.clf()


# 2011 Winter
# 2011 Winter regression
sns.lmplot(x = 'atemp', y = 'cnt', data = Winter2011_df)
plt.title('Normalized Seaborn regression plot of Atemp vs count in Winter 2011')
plt.show()
plt.clf()


# 2011 Winter residual
sns.residplot(x = 'atemp', y = 'cnt', data = Winter2011_df)
plt.title('Normalized Seaborn residual plot of Atemp vs count in Winter 2011')
plt.show()
plt.clf()


# 2012 seasonal regressions and residuals for atemperature vs count

Spring2012_df = day_df_2012.loc[day_df_2012['season'] == 1]
Summer2012_df = day_df_2012.loc[day_df_2012['season'] == 2]
Fall2012_df = day_df_2012.loc[day_df_2012['season'] == 3]
Winter2012_df = day_df_2012.loc[day_df_2012['season'] == 4]


# 2012 Spring
# 2012 Spring regression
sns.lmplot(x = 'atemp', y = 'cnt', data = Spring2012_df)
plt.title('Normalized Seaborn regression plot of Atemp vs count in Spring 2012')
plt.show()
plt.clf()

# 2012 Spring residual
sns.residplot(x = 'atemp', y = 'cnt', data = Spring2012_df)
plt.title('Normalized Seaborn residual plot of Atemp vs count in Spring 2012')
plt.show()
plt.clf()


print('Beginning of 2012 atemperature plots')
# 2012 Summer
# 2012 Summer regression
sns.lmplot(x = 'atemp', y = 'cnt', data = Summer2012_df)
plt.title('Normalized Seaborn regression plot of Atemp vs count in Summer 2012')
plt.show()
plt.clf()

# 2012 Summer residual
sns.residplot(x = 'atemp', y = 'cnt', data = Summer2012_df)
plt.title('Normalized Seaborn residual plot of Atemp vs count in Summer 2012')
plt.show()
plt.clf()


# 2012 Fall
# 2012 Fall regression
sns.lmplot(x = 'atemp', y = 'cnt', data = Fall2012_df)
plt.title('Normalized Seaborn regression plot of Atemp vs count in Fall 2012')
plt.show()
plt.clf()


# 2012 Fall residual
sns.residplot(x = 'atemp', y = 'cnt', data = Fall2012_df)
plt.title('Normalized Seaborn residual plot of Atemp vs count in Fall 2012')
plt.show()
plt.clf()


# 2012 Winter
# 2012 Winter regression
sns.lmplot(x = 'atemp', y = 'cnt', data = Winter2012_df)
plt.title('Normalized Seaborn regression plot of Atemp vs count in Winter 2012')
plt.show()
plt.clf()


# 2012 Winter residual
sns.residplot(x = 'atemp', y = 'cnt', data = Winter2012_df)
plt.title('Normalized Seaborn residual plot of Atemp vs count in Winter 2012')
plt.show()
plt.clf()




# =============================================================================
# =============================================================================
# =============================================================================
#
# Redo all regressions with the hourly data
#
# =============================================================================
# =============================================================================
# =============================================================================

print('Beginning of hourly data')
print('\n')
print('\n')
print('\n')
print('\n')

# Sort into seasonal dataframes
Spring_hour_df = hour_df.loc[hour_df['season'] == 1]
Summer_hour_df = hour_df.loc[hour_df['season'] == 2]
Fall_hour_df   = hour_df.loc[hour_df['season'] == 3]
Winter_hour_df = hour_df.loc[hour_df['season'] == 4]

#Make annual data frames
hour_df_2011 = hour_df.loc[hour_df['yr'] == 0]
hour_df_2012 = hour_df.loc[hour_df['yr'] == 1]

# We only have a few variables that makes sense for regression
# temp (temperature)
# hum (humidity)
# windspeed (Normalized wind speed. The values are divided to 67 (max))

# Hourly Temperature and count for 2011
# Linear regression of temp and cnt
sns.lmplot(x= 'temp', y = 'cnt', data = hour_df_2011)
plt.title('Hourly Seaborn regression plot of temp vs cnt in 2011')
plt.show()
plt.clf()


# Hourly Residual plot of temp and count for 2011
sns.residplot(x= 'temp', y = 'cnt', data = hour_df_2011)
plt.title('Hourly Seaborn residual plot of temp vs count in 2011')
plt.show()
plt.clf()


# Hourly Temperature and count for 2012
# Linear regression of temp and count
sns.lmplot(x= 'temp', y = 'cnt', data = hour_df_2012)
plt.title('Hourly Seaborn regression plot of temp vs count in 2012')
plt.show()
plt.clf()


# Hourly Residual plot of temp and count for 2012
sns.residplot(x= 'temp', y = 'cnt', data = hour_df_2012)
plt.title('Hourly Seaborn residual plot of temp vs count in 2012')
plt.show()
plt.clf()



print('Beginning of Hourly  2011 temperature plots')

# 2011 seasonal regressions and residuals for temperature vs count
Spring2011_df_H = hour_df_2011.loc[hour_df_2011['season'] == 1]
Summer2011_df_H = hour_df_2011.loc[hour_df_2011['season'] == 2]
Fall2011_df_H   = hour_df_2011.loc[hour_df_2011['season'] == 3]
Winter2011_df_H = hour_df_2011.loc[hour_df_2011['season'] == 4]

# 2011 Spring
# 2011 Spring regression
sns.lmplot(x = 'temp', y = 'cnt', data = Spring2011_df_H)
plt.title('Hourly Normalized Seaborn regression plot of temp vs count in Spring 2011')
plt.show()
plt.clf()

# 2011 Spring residual
sns.residplot(x = 'temp', y = 'cnt', data = Spring2011_df_H)
plt.title('Hourly Normalized Seaborn residual plot of temp vs count in Spring 2011')
plt.show()
plt.clf()


# 2011 Summer Hourly
# 2011 Summer Hourly regression
sns.lmplot(x = 'temp', y = 'cnt', data = Summer2011_df_H)
plt.title('Hourly Normalized Seaborn regression plot of temp vs count in Summer 2011')
plt.show()
plt.clf()

# 2011 Summer Hourly residual
sns.residplot(x = 'temp', y = 'cnt', data = Summer2011_df_H)
plt.title('Hourly Normalized Seaborn residual plot of temp vs count in Summer 2011')
plt.show()
plt.clf()


# 2011 Fall Hourly
# 2011 Fall Hourly regression
sns.lmplot(x = 'temp', y = 'cnt', data = Fall2011_df_H)
plt.title('Hourly Normalized Seaborn regression plot of temp vs count in Fall 2011')
plt.show()
plt.clf()


# 2011 Fall Hourly residual
sns.residplot(x = 'temp', y = 'cnt', data = Fall2011_df_H)
plt.title('Hourly Normalized Seaborn residual plot of temp vs count in Fall 2011')
plt.show()
plt.clf()


# 2011 Winter Hourly
# 2011 Winter Hourly regression
sns.lmplot(x = 'temp', y = 'cnt', data = Winter2011_df_H)
plt.title('Hourly Normalized Seaborn regression plot of temp vs count in Winter 2011')
plt.show()
plt.clf()


# 2011 Winter residual
sns.residplot(x = 'temp', y = 'cnt', data = Winter2011_df_H)
plt.title('Hourly Normalized Seaborn residual plot of temp vs count in Winter 2011')
plt.show()
plt.clf()


# 2012 seasonal regressions and residuals for temperature vs count

Spring2012_df_H = hour_df_2012.loc[hour_df_2012['season'] == 1]
Summer2012_df_H = hour_df_2012.loc[hour_df_2012['season'] == 2]
Fall2012_df_H   = hour_df_2012.loc[hour_df_2012['season'] == 3]
Winter2012_df_H = hour_df_2012.loc[hour_df_2012['season'] == 4]


# 2012 Spring Hourly
# 2012 Spring Hourly regression
sns.lmplot(x = 'temp', y = 'cnt', data = Spring2012_df_H)
plt.title('Hourly Normalized Seaborn regression plot of temp vs count in Spring 2012')
plt.show()
plt.clf()

# 2012 Spring residual
sns.residplot(x = 'temp', y = 'cnt', data = Spring2012_df_H)
plt.title('Hourly Normalized Seaborn residual plot of temp vs count in Spring 2012')
plt.show()
plt.clf()


print('Beginning of 2012 temperature plots')
# 2012 Summer Hourly
# 2012 Summer Hourly regression
sns.lmplot(x = 'temp', y = 'cnt', data = Summer2012_df_H)
plt.title('Hourly Normalized Seaborn regression plot of temp vs count in Summer 2012')
plt.show()
plt.clf()

# 2012 Summer Hourly residual
sns.residplot(x = 'temp', y = 'cnt', data = Summer2012_df_H)
plt.title('Hourly Normalized Seaborn residual plot of temp vs count in Summer 2012')
plt.show()
plt.clf()


# 2012 Fall Hourly 
# 2012 Fall Hourly regression
sns.lmplot(x = 'temp', y = 'cnt', data = Fall2012_df_H)
plt.title('Hourly Normalized Seaborn regression plot of temp vs count in Fall 2012')
plt.show()
plt.clf()


# 2012 Hourly Fall residual
sns.residplot(x = 'temp', y = 'cnt', data = Fall2012_df_H)
plt.title('Hourly Normalized Seaborn residual plot of temp vs count in Fall 2012')
plt.show()
plt.clf()


# 2012 Winter Hourly
# 2012 Winter Hourly regression
sns.lmplot(x = 'temp', y = 'cnt', data = Winter2012_df_H)
plt.title('Hourly Normalized Seaborn regression plot of temp vs count in Winter 2012')
plt.show()
plt.clf()


# 2012 Winter Hourly residual
sns.residplot(x = 'temp', y = 'cnt', data = Winter2012_df_H)
plt.title('Normalized Seaborn residual plot of temp vs count in Winter 2012')
plt.show()
plt.clf()


# ========================================================================
# Hourly Humidity regressions
# ========================================================================



# Hourly Humidity and count for 2011
# Hourly Linear regression of hum and cnt
sns.lmplot(x= 'hum', y = 'cnt', data = hour_df_2011)
plt.title('Hourly Seaborn regression plot of humidity vs cnt in 2011')
plt.show()
plt.clf()


# Hourly Residual plot of hum and count for 2011
sns.residplot(x= 'hum', y = 'cnt', data = hour_df_2011)
plt.title('Hourly Seaborn residual plot of humidity vs count in 2011')
plt.show()
plt.clf()


# Hourly Humidity and count for 2012
# Hourly Linear regression of hum and count
sns.lmplot(x= 'hum', y = 'cnt', data = hour_df_2012)
plt.title('Hourly Seaborn regression plot of humidity vs count in 2012')
plt.show()
plt.clf()


# Hourly Residual plot of hum and count for 2012
sns.residplot(x= 'hum', y = 'cnt', data = hour_df_2012)
plt.title('Hourly Seaborn residual plot of humidity vs count in 2012')
plt.show()
plt.clf()

# Result:
# There is very little correlation between humidity and count



# Trying some more complicated seaborn plots with temperature
sns.lmplot(x= 'temp', y = 'cnt', data = hour_df_2011, fit_reg=True)
plt.title('Hourly Seaborn regression plot of temp vs cnt in 2011: fit_reg = True')
plt.show()
plt.clf()


# Try order 2
sns.lmplot(x= 'temp', y = 'cnt', data = hour_df_2011, fit_reg=True, order = 2, ci = 90)
plt.title('Hourly Seaborn regression plot of temp vs cnt in 2011: fit_reg = True')
plt.show()
plt.clf()







# =============================================================================
# Windspeed regressions
# =============================================================================


# Hourly Windspeed and count for 2011
# Hourly Linear regression of windspeed and cnt
sns.lmplot(x= 'windspeed', y = 'cnt', data = hour_df_2011)
plt.title('Hourly Seaborn regression plot of windspeed vs cnt in 2011')
plt.show()
plt.clf()


# Hourly Residual plot of windspeed and count for 2011
sns.residplot(x= 'windspeed', y = 'cnt', data = hour_df_2011)
plt.title('Hourly Seaborn residual plot of windspeed vs count in 2011')
plt.show()
plt.clf()


# Hourly Humidity and count for 2012
# Hourly Linear regression of windspeed and count
sns.lmplot(x= 'windspeed', y = 'cnt', data = hour_df_2012)
plt.title('Hourly Seaborn regression plot of windspeed vs count in 2012')
plt.show()
plt.clf()


# Hourly Residual plot of windspeed and count for 2012
sns.residplot(x= 'windspeed', y = 'cnt', data = hour_df_2012)
plt.title('Hourly Seaborn residual plot of windspeed vs count in 2012')
plt.show()
plt.clf()

# Result:
# There is little correlation between windspeed and bike usage. 



# =============================================================================
# Hourly aTemp (Feels like temp) regressions
# =============================================================================
print('Beginning of 2011 Hourly Atemp plots')

# 2011 seasonal regressions and residuals for atemperature vs count
Spring2011_df_H = hour_df_2011.loc[hour_df_2011['season'] == 1]
Summer2011_df_H = hour_df_2011.loc[hour_df_2011['season'] == 2]
Fall2011_df_H   = hour_df_2011.loc[hour_df_2011['season'] == 3]
Winter2011_df_H = hour_df_2011.loc[hour_df_2011['season'] == 4]

# Hourly 2011 Spring
# Hourly 2011 Spring regression
sns.lmplot(x = 'atemp', y = 'cnt', data = Spring2011_df_H)
plt.title('Hourly Normalized Seaborn regression plot of Atemp vs count in Spring 2011')
plt.show()
plt.clf()

# Hourly 2011 Spring residual
sns.residplot(x = 'atemp', y = 'cnt', data = Spring2011_df_H)
plt.title('Hourly Normalized Seaborn residual plot of Atemp vs count in Spring 2011')
plt.show()
plt.clf()


# Hourly 2011 Summer
# Hourly 2011 Summer regression
sns.lmplot(x = 'atemp', y = 'cnt', data = Summer2011_df_H)
plt.title('Hourly Normalized Seaborn regression plot of Atemp vs count in Summer 2011')
plt.show()
plt.clf()

# Hourly 2011 Summer residual
sns.residplot(x = 'atemp', y = 'cnt', data = Summer2011_df_H)
plt.title('Hourly Normalized Seaborn residual plot of Atemp vs count in Summer 2011')
plt.show()
plt.clf()


# Hourly 2011 Fall
# Hourly 2011 Fall regression
sns.lmplot(x = 'atemp', y = 'cnt', data = Fall2011_df_H)
plt.title('Hourly Normalized Seaborn regression plot of Atemp vs count in Fall 2011')
plt.show()
plt.clf()


# Hourly 2011 Fall residual
sns.residplot(x = 'atemp', y = 'cnt', data = Fall2011_df_H)
plt.title('Hourly Normalized Seaborn residual plot of Atemp vs count in Fall 2011')
plt.show()
plt.clf()


# Hourly 2011 Winter
# Hourly 2011 Winter regression
sns.lmplot(x = 'atemp', y = 'cnt', data = Winter2011_df_H)
plt.title('Hourly Normalized Seaborn regression plot of Atemp vs count in Winter 2011')
plt.show()
plt.clf()


# Hourly 2011 Winter residual
sns.residplot(x = 'atemp', y = 'cnt', data = Winter2011_df_H)
plt.title('Hourly Normalized Seaborn residual plot of Atemp vs count in Winter 2011')
plt.show()
plt.clf()


# Hourly 2012 seasonal regressions and residuals for atemperature vs count

Spring2012_df_H = hour_df_2012.loc[hour_df_2012['season'] == 1]
Summer2012_df_H = hour_df_2012.loc[hour_df_2012['season'] == 2]
Fall2012_df_H   = hour_df_2012.loc[hour_df_2012['season'] == 3]
Winter2012_df_H = hour_df_2012.loc[hour_df_2012['season'] == 4]


# Hourly 2012 Spring
# Hourly 2012 Spring regression
sns.lmplot(x = 'atemp', y = 'cnt', data = Spring2012_df_H)
plt.title('Hourly Normalized Seaborn regression plot of Atemp vs count in Spring 2012')
plt.show()
plt.clf()

# 2012 Spring residual
sns.residplot(x = 'atemp', y = 'cnt', data = Spring2012_df_H)
plt.title('Hourly Normalized Seaborn residual plot of Atemp vs count in Spring 2012')
plt.show()
plt.clf()


print('Beginning of 2012 Hourly atemperature plots')
# 2012 Summer Hourly
# 2012 Summer Hourly regression
sns.lmplot(x = 'atemp', y = 'cnt', data = Summer2012_df_H)
plt.title('Hourly Normalized Seaborn regression plot of Atemp vs count in Summer 2012')
plt.show()
plt.clf()

# 2012 Summer Hourly residual
sns.residplot(x = 'atemp', y = 'cnt', data = Summer2012_df_H)
plt.title('Hourly Normalized Seaborn residual plot of Atemp vs count in Summer 2012')
plt.show()
plt.clf()


# 2012 Hourly Fall
# 2012 Hourly Fall regression
sns.lmplot(x = 'atemp', y = 'cnt', data = Fall2012_df_H)
plt.title('Hourly Normalized Seaborn regression plot of Atemp vs count in Fall 2012')
plt.show()
plt.clf()


# 2012 Hourly Fall residual
sns.residplot(x = 'atemp', y = 'cnt', data = Fall2012_df_H)
plt.title('Hourly Normalized Seaborn residual plot of Atemp vs count in Fall 2012')
plt.show()
plt.clf()


# 2012 Hourly Winter
# 2012 Hourly Winter regression
sns.lmplot(x = 'atemp', y = 'cnt', data = Winter2012_df_H)
plt.title('Hourly Normalized Seaborn regression plot of Atemp vs count in Winter 2012')
plt.show()
plt.clf()


# 2012 Hourly Winter residual
sns.residplot(x = 'atemp', y = 'cnt', data = Winter2012_df_H)
plt.title('Hourly Normalized Seaborn residual plot of Atemp vs count in Winter 2012')
plt.show()
plt.clf()

# Linear regressions do not show strong correlations for hourly data
# The hourly data seems to be less linear than the daily data. 
# I'm thinking that looking at the distribution through seaborn
# would be more effective. 



# Print statement for debugging
print('Reached end of code')
