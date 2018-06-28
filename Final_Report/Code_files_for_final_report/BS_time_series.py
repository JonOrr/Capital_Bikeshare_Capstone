# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 12:49:27 2018

Time Series Plot

@author: Jon
"""


# Import common packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
# import seaborn as sns

#Change the working directory
# ATTN: You will need to change this locally.
os.chdir('C:/Users/Jon/Documents/Springboard/Capstone1/Bike-Sharing-Dataset')

# Read in the csv's for day.csv and hour.csv
day_df = pd.read_csv('day.csv')
hour_df = pd.read_csv('hour.csv')

# Make an array for each table
day_array = np.array(day_df.values)
hour_array = np.array(hour_df.values)


# Plot the raw data before setting the datetime index
# day_df.plot()
# plt.show()

# Convert the 'Date' column into a collection of datetime objects: df.Date
day_df.dteday = pd.to_datetime(day_df['dteday'])

# Set the index to be the converted 'Date' column
day_df.set_index('dteday', inplace=True)

# Re-plot the DataFrame to see that the axis is now datetime aware!

# First time series
day_df.plot()
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('Time series of day_df')
plt.show()
plt.clf()



# Windspeed time series.
day_df = pd.read_csv('day.csv')

day_windspeed_df = day_df.filter(['dteday', 'windspeed'])

# Convert the 'Date' column into a collection of datetime objects: df.Date
day_windspeed_df.dteday = pd.to_datetime(day_windspeed_df['dteday'])

# Set the index to be the converted 'Date' column
day_windspeed_df.set_index('dteday', inplace=True)

day_windspeed_df.plot()
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('Time series of wind speed from day_df')
plt.ylabel('Normalized wind speed. Values divided to 67 (max)' )
plt.plot()
plt.show()
plt.clf()


# Temperature time series.
day_df = pd.read_csv('day.csv')

day_temp_df = day_df.filter(['dteday', 'temp'])

# Convert the 'Date' column into a collection of datetime objects: df.Date
day_temp_df.dteday = pd.to_datetime(day_temp_df['dteday'])

# Set the index to be the converted 'Date' column
day_temp_df.set_index('dteday', inplace=True)

day_temp_df.plot(color = 'green')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('Time series of temperature from day_df')
plt.ylabel('Normalized temp in Celsius. Values are divided to 41 (max)' )
plt.plot()
plt.show()
plt.clf()

# Time Series for registered, casual, and total users.
day_df = pd.read_csv('day.csv')

# Create the rides_df for day.csv
day_rides_df = day_df.filter(['dteday', 'casual', 'registered', 'cnt'])

# Convert the 'Date' column into a collection of datetime objects: df.Date
day_rides_df.dteday = pd.to_datetime(day_df['dteday'])

# Set the index to be the converted 'Date' column
day_rides_df.set_index('dteday', inplace=True)


day_rides_df.plot()
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('Time series of ride counts from day_df')
plt.ylabel('rides' )
plt.plot()
plt.show()
plt.clf()


# =============================================================================
# # Try with seaborn
# day_df = pd.read_csv('day.csv')
# 
# # Create the rides_df for day.csv
# day_rides_df = day_df.filter(['dteday', 'casual', 'registered', 'cnt'])
# 
# sns.tsplot(day_rides_df, day_rides_df['dteday'])
# Getting error: The Truth value of a series is ambiguous
# =============================================================================





# =============================================================================
# # Repeat this process with the hour_df
# =============================================================================

# Windspeed time series.
hour_df = pd.read_csv('hour.csv')

hour_windspeed_df = hour_df.filter(['dteday', 'windspeed'])

# Convert the 'Date' column into a collection of datetime objects: df.Date
hour_windspeed_df.dteday = pd.to_datetime(hour_windspeed_df['dteday'])

# Set the index to be the converted 'Date' column
hour_windspeed_df.set_index('dteday', inplace=True)

hour_windspeed_df.plot()
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('Time series of wind speed from hour_df')
plt.ylabel('Normalized wind speed. Values divided to 67 (max)' )
plt.plot()
plt.show()
plt.clf()


# Temperature time series.
hour_df = pd.read_csv('hour.csv')

hour_temp_df = hour_df.filter(['dteday', 'temp'])

# Convert the 'Date' column into a collection of datetime objects: df.Date
hour_temp_df.dteday = pd.to_datetime(hour_temp_df['dteday'])

# Set the index to be the converted 'Date' column
hour_temp_df.set_index('dteday', inplace=True)

hour_temp_df.plot(color = 'green')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('Time series of temperature from hour_df')
plt.ylabel('Normalized temp in Celsius. Values are divided to 41 (max)' )
plt.plot()
plt.show()
plt.clf()

# Time Series for registered, casual, and total users.
hour_df = pd.read_csv('hour.csv')

# Create the rides_df for day.csv
hour_rides_df = hour_df.filter(['dteday', 'casual', 'registered', 'cnt'])

# Convert the 'Date' column into a collection of datetime objects: df.Date
hour_rides_df.dteday = pd.to_datetime(hour_df['dteday'])

# Set the index to be the converted 'Date' column
hour_rides_df.set_index('dteday', inplace=True)

# The rides plot is too overlapped. 
hour_rides_df.plot()
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('Time series of ride counts from hour_df')
plt.ylabel('rides' )
plt.plot()
plt.show()
plt.clf()


