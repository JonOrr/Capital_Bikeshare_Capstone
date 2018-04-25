# -*- coding: utf-8 -*-
"""
This file contains some basic subplots for the Bike Sharing Dataset
"""

# Import common packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


#Change the working directory
# ATTN: You will need to change this locally.
os.chdir('C:/Users/Jon/Documents/Springboard/Capstone1/Bike-Sharing-Dataset')

# Read in the csv's for day.csv and hour.csv
day_df = pd.read_csv('day.csv')
hour_df = pd.read_csv('hour.csv')

# Make an array for each table
day_array = np.array(day_df.values)
hour_array = np.array(hour_df.values)

# Check to see if there are any null values
print(day_df.isnull().any().any())
print(hour_df.isnull().any().any())

# Create a list of column names for quick reference
Dcolumn_names = list(day_df.columns.values)
Hcolumn_names = list(hour_df.columns.values)

# Sort into seasonal dataframes
Spring_day_df = day_df.loc[day_df['season'] == 1]
Summer_day_df = day_df.loc[day_df['season'] == 2]
Fall_day_df = day_df.loc[day_df['season'] == 3]
Winter_day_df = day_df.loc[day_df['season'] == 4]

# Repeat for the hourly data
Spring_hour_df = hour_df.loc[hour_df['season'] == 1]
Summer_hour_df = hour_df.loc[hour_df['season'] == 2]
Fall_hour_df = hour_df.loc[hour_df['season'] == 3]
Winter_hour_df = hour_df.loc[hour_df['season'] == 4]



# Plot the the count of rides in each season
Spring_day_df.plot(y = 'cnt')
print('This plot shows that the data occurs in two different years,\
      we should restructure the dataframes to make spring 2011,\
      and spring 2012 dataframes')


day_df_2011 = day_df.loc[day_df['yr'] == 0]
day_df_2012 = day_df.loc[day_df['yr'] == 1]


day_df_2011.plot(y='cnt')
plt.title('Daily usage for both registered and casual users in 2011')
plt.ylabel('Number of rides')
plt.xlabel('Day of the year')

day_df_2012.plot(y='cnt')
plt.title('Daily usage for both registered and casual users in 2012')
plt.ylabel('Number of rides')
plt.xlabel('Day of the year')


Spring2011_df = day_df_2011.loc[day_df_2011['season'] == 1]
Summer2011_df = day_df_2011.loc[day_df_2011['season'] == 2]
Fall2011_df = day_df_2011.loc[day_df_2011['season'] == 3]
Winter2011_df = day_df_2011.loc[day_df_2011['season'] == 4]

Spring2012_df = day_df_2012.loc[day_df_2012['season'] == 1]
Summer2012_df = day_df_2012.loc[day_df_2012['season'] == 2]
Fall2012_df = day_df_2012.loc[day_df_2012['season'] == 3]
Winter2012_df = day_df_2012.loc[day_df_2012['season'] == 4]


# Create a 4 part subplot for 2011 seasons
plt.subplot(2,2,1)
plt.plot(Spring2011_df['cnt'])
plt.title('Spring 2011 user count')

plt.subplot(2,2,2)
plt.plot(Summer2011_df['cnt'])
plt.title('Summer 2011 user count')

plt.subplot(2,2,3)
plt.plot(Fall2011_df['cnt'])
plt.title('Fall 2011 user count')

plt.subplot(2,2,4)
plt.plot(Winter2011_df['cnt'])
plt.title('Winter 2011 user count')

plt.tight_layout()
plt.show()


# Create a 4 part subplot for 2012 seasons
plt.subplot(2,2,1)
plt.plot(Spring2012_df['cnt'])
plt.title('Spring 2012 user count')

plt.subplot(2,2,2)
plt.plot(Summer2012_df['cnt'])
plt.title('Summer 2012 user count')

plt.subplot(2,2,3)
plt.plot(Fall2012_df['cnt'])
plt.title('Fall 2012 user count')

plt.subplot(2,2,4)
plt.plot(Winter2012_df['cnt'])
plt.title('Winter 2012 user count')

plt.tight_layout()
plt.show()


# A simple print statement for debugging
print('Reached end of script')


day_df.plot(y='cnt')
plt.title('Daily usage for both registered and casual users in 2012')
plt.ylabel('Number of rides')
plt.xlabel('Day of the year')