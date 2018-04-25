# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 12:51:21 2018

Exploratory Data Analysis: Hourly distriubtions for:
    
    Windspeed
    user count
    temperature
    atemp ("Feels" like temperature")
    
@author: Jon
"""
# Import common packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

#Change the working directory
# ATTN: You will need to change this locally.
os.chdir('C:/Users/Jon/Documents/Springboard/Capstone1/Bike-Sharing-Dataset')

# Read in the csv's for day.csv and hour.csv
day_df = pd.read_csv('day.csv')
hour_df = pd.read_csv('hour.csv')

# Make an array for each table
day_array = np.array(day_df.values)
hour_array = np.array(hour_df.values)


# Sort into seasonal dataframes
Spring_hour_df = hour_df.loc[hour_df['season'] == 1]
Summer_hour_df = hour_df.loc[hour_df['season'] == 2]
Fall_hour_df   = hour_df.loc[hour_df['season'] == 3]
Winter_hour_df = hour_df.loc[hour_df['season'] == 4]

#Make annual data frames
hour_df_2011 = hour_df.loc[hour_df['yr'] == 0]
hour_df_2012 = hour_df.loc[hour_df['yr'] == 1]

hour_counts    = hour_df['cnt'].values
hour_counts_11 = hour_df_2011['cnt'].values
hour_counts_12 = hour_df_2012['cnt'].values


# =============================================================================
#
# # ================================================================
# # This section takes a very long time to run. 
# # Uncomment it if you wish to run it. 
# # ================================================================
# 
# # Distribution for hour counts during 2011-2012
# 
# sns.distplot(hour_counts)
# sns.distplot(hour_counts, hist=False, rug=True);
# plt.title('Distribution for hour counts in 2011-2012')
# plt.show()
# plt.clf()
# 
# 
# # Distribution for hour counts in 2012
# 
# sns.distplot(hour_counts_11)
# sns.distplot(hour_counts_11, hist=False, rug=True);
# plt.title('Distribution for hour counts in 2011')
# plt.show()
# plt.clf()
# 
# # Distribution for hour counts in 2012
# sns.distplot(hour_counts_12)
# sns.distplot(hour_counts_12, hist=False, rug=True);
# plt.title('Distribution for hour counts in 2012')
# plt.show()
# plt.clf()
# =============================================================================

# Generate bar plot with hr on the x axis and count on the y axis
plt.bar(hour_df['hr'].values,  hour_df['cnt'].values)
plt.title('2011-2012 count of users per hour')
plt.xlabel('Hour of the day, (24hr clock)')
plt.ylabel('User count')
plt.show()
plt.clf()

# Generate bar plot with temp on the x axis and count on the y axis
# To scale to Farenheight we unnormalize (multiply by 41), then apply
# the conversion equation: T_f = T_c * 1.8 + 32
plt.bar( hour_df['temp'].values*41*1.8+32 , hour_df['cnt'])
plt.title('2011-2012 count of users per hour')
plt.xlabel('Temperature (F)')
plt.ylabel('User count')
plt.show()
plt.clf()

# Generate bar plot with hr on the x axis and count on the y axis
# Un-normalize windspeed by multiplying by 67 (the maximum value)
plt.bar(hour_df['windspeed'].values*67,  hour_df['cnt'].values)
plt.title('2011-2012 count of users per hour')
plt.xlabel('Windspeed (mph) ')
plt.ylabel('User count')
plt.show()
plt.clf()

# =============================================================================
# Box and Whisker plots 
# =============================================================================
x = hour_df['hr'].values
y = hour_df['cnt'].values
y_lin = np.linspace(0, 1000, 11)

sns.boxplot(x, y)
plt.title('2011-2012 count of users per hour')
plt.xlabel('Hour of the day, (24hr clock)')
plt.ylabel('User count')
plt.yticks(y_lin)
plt.show()
plt.clf()

