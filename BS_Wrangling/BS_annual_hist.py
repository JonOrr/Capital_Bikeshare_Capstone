# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 14:28:06 2018

Various bike usage histograms

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


# Create a list of column names for quick reference
Dcolumn_names = list(day_df.columns.values)
Hcolumn_names = list(hour_df.columns.values)

# Sort into seasonal dataframes
Spring_day_df = day_df.loc[day_df['season'] == 1]
Summer_day_df = day_df.loc[day_df['season'] == 2]
Fall_day_df = day_df.loc[day_df['season'] == 3]
Winter_day_df = day_df.loc[day_df['season'] == 4]

#Make histograms for annual data
day_df_2011 = day_df.loc[day_df['yr'] == 0]
day_df_2012 = day_df.loc[day_df['yr'] == 1]

# Plot the 2011 count data
day_df_2011.plot(y='cnt')
plt.title('Daily usage for both registered and casual users in 2011')
plt.ylabel('Number of rides')
plt.xlabel('Day of the year')
plt.show()
plt.clf()

# Plot the 2012 count data
day_df_2012.plot(y='cnt')
plt.title('Daily usage for both registered and casual users in 2012')
plt.ylabel('Number of rides')
plt.xlabel('Day of the year')
plt.show()
plt.clf()

# 2011 count histogram
# sns.barplot(x = day_df_2011['dteday'], y = day_df_2011['cnt'], data = day_df_2011)
# =============================================================================
# plt.title('Daily usage for both registered and casual users in 2012')
# plt.ylabel('Number of rides')
# plt.xlabel('Day of the year')
# =============================================================================
# plt.show()

# Create 2011 seasonal dataframe
Spring2011_df = day_df_2011.loc[day_df_2011['season'] == 1]
Summer2011_df = day_df_2011.loc[day_df_2011['season'] == 2]
Fall2011_df = day_df_2011.loc[day_df_2011['season'] == 3]
Winter2011_df = day_df_2011.loc[day_df_2011['season'] == 4]

# Create 2012 seasonal dataframe

Spring2012_df = day_df_2012.loc[day_df_2012['season'] == 1]
Summer2012_df = day_df_2012.loc[day_df_2012['season'] == 2]
Fall2012_df = day_df_2012.loc[day_df_2012['season'] == 3]
Winter2012_df = day_df_2012.loc[day_df_2012['season'] == 4]

# 2011 bar plot
spring11_ct = sum(Spring2011_df['cnt'])
summer11_ct = sum(Summer2011_df['cnt'])
fall11_ct = sum(Fall2011_df['cnt'])
winter11_ct = sum(Winter2011_df['cnt'])

counts_2011 = [spring11_ct, summer11_ct, fall11_ct, winter11_ct]
counts_2011a = np.array(counts_2011)
x = np.arange(4)

plt.bar(x, counts_2011a)
plt.xticks(x, ('Spring', 'Summer', 'Fall', 'Winter'))
plt.title('Total bike usage by season in 2011')
plt.ylim(0, 700000) 
plt.show()
plt.clf()


# 2012 bar plot
spring12_ct = sum(Spring2012_df['cnt'])
summer12_ct = sum(Summer2012_df['cnt'])
fall12_ct = sum(Fall2012_df['cnt'])
winter12_ct = sum(Winter2012_df['cnt'])

counts_2012 = [spring12_ct, summer12_ct, fall12_ct, winter12_ct]
counts_2012a = np.array(counts_2012)
x = np.arange(4)

plt.bar(x, counts_2012a)
plt.xticks(x, ('Spring', 'Summer', 'Fall', 'Winter'))
plt.title('Total bike usage by season in 2012')
plt.ylim(0, 700000)
plt.show()
plt.clf()


# Now only consider registered users for 2011 and 2012 
# Use the same code with 'cnt' replace with registered

# 2011 bar plot for registered users
spring11_reg = sum(Spring2011_df['registered'])
summer11_reg = sum(Summer2011_df['registered'])
fall11_reg = sum(Fall2011_df['registered'])
winter11_reg = sum(Winter2011_df['registered'])

reg_2011 = [spring11_reg, summer11_reg, fall11_reg, winter11_reg]
reg_2011a = np.array(reg_2011)
x = np.arange(4)

plt.bar(x, reg_2011a)
plt.xticks(x, ('Spring', 'Summer', 'Fall', 'Winter'))
plt.title('Total bike usage by registered users each season in 2011')
plt.ylim(0, 700000)
plt.show()
plt.clf()


# 2012 bar plot for registered users
spring12_reg = sum(Spring2012_df['registered'])
summer12_reg = sum(Summer2012_df['registered'])
fall12_reg = sum(Fall2012_df['registered'])
winter12_reg = sum(Winter2012_df['registered'])

reg_2012 = [spring12_reg, spring12_reg, fall12_reg, winter12_reg]
reg_2012a = np.array(reg_2012)
x = np.arange(4)

plt.bar(x, reg_2012a)
plt.xticks(x, ('Spring', 'Summer', 'Fall', 'Winter'))
plt.title('Total bike usage by registered users each season in 2012')
plt.ylim(0, 700000)
plt.show()
plt.clf()


# Now only consider casual users for 2011 and 2012 
# Use the same code with 'cnt' replace with registered

# 2011 bar plot for casual users
spring11_cas = sum(Spring2011_df['casual'])
summer11_cas = sum(Summer2011_df['casual'])
fall11_cas = sum(Fall2011_df['casual'])
winter11_cas = sum(Winter2011_df['casual'])

cas_2011 = [spring11_cas, summer11_cas, fall11_cas, winter11_cas]
cas_2011a = np.array(cas_2011)
x = np.arange(4)

plt.bar(x, cas_2011a)
plt.xticks(x, ('Spring', 'Summer', 'Fall', 'Winter'))
plt.title('Total bike usage by casual users each season in 2011')
plt.ylim(0, 700000)
plt.show()
plt.clf()


# 2012 bar plot for casual users 
spring12_cas = sum(Spring2012_df['casual'])
summer12_cas = sum(Summer2012_df['casual'])
fall12_cas = sum(Fall2012_df['casual'])
winter12_cas = sum(Winter2012_df['casual'])

cas_2012 = [spring12_cas, spring12_cas, fall12_cas, winter12_cas]
cas_2012a = np.array(cas_2012)
x = np.arange(4)

plt.bar(x, cas_2012a)
plt.xticks(x, ('Spring', 'Summer', 'Fall', 'Winter'))
plt.title('Total bike usage by casual users each season in 2012')
plt.ylim(0, 700000)
plt.show()
plt.clf()


# Bike usage by weather condition for all years and users
# 1: Clear / Partly Cloudy
# 2: Mist + Cloudy
# 3: Light Snow / Light rain + storm / scattered clouds
# 4: Heavy Rain / Ice Pallets, Thunderstorms

clear_df = day_df.loc[day_df['weathersit'] == 1]
mist_df = day_df.loc[day_df['weathersit'] == 2]
light_df = day_df.loc[day_df['weathersit'] == 3]
heavy_df = day_df.loc[day_df['weathersit'] == 4]

clear_cnt = sum(clear_df['cnt'])
mist_cnt = sum(mist_df['cnt'])
light_cnt = sum(light_df['cnt'])
heavy_cnt = sum(heavy_df['cnt'])

weather_cnt = [clear_cnt, mist_cnt, light_cnt, heavy_cnt]
plt.bar(x, weather_cnt)
plt.xticks(x, ('clear', 'misty', 'light rain/snow', 'heavy rain/snow'))
plt.title('Total bike usage for each weather condition')
plt.ylim(0, 2300000)
plt.show()
plt.clf()


# Bike usage by weather condition for all years and registered users
# 1: Clear / Partly Cloudy
# 2: Mist + Cloudy
# 3: Light Snow / Light rain + storm / scattered clouds
# 4: Heavy Rain / Ice Pallets, Thunderstorms 

clear_reg_cnt = sum(clear_df['registered'])
mist_reg_cnt  = sum(mist_df['registered'])
light_reg_cnt = sum(light_df['registered'])
heavy_reg_cnt = sum(heavy_df['registered'])

weather_reg = [clear_reg_cnt, mist_reg_cnt, light_reg_cnt, heavy_reg_cnt]
plt.bar(x, weather_reg)
plt.xticks(x, ('clear', 'misty', 'light rain/snow', 'heavy rain/snow'))
plt.title('Total bike usage for each weather condition by registered users')
plt.ylim(0, 2300000)
plt.show()
plt.clf()


# Bike usage by weather condition for all years and casual users
# 1: Clear / Partly Cloudy
# 2: Mist + Cloudy
# 3: Light Snow / Light rain + storm / scattered clouds
# 4: Heavy Rain / Ice Pallets, Thunderstorms

clear_cas_cnt = sum(clear_df['casual'])
mist_cas_cnt  = sum(mist_df['casual'])
light_cas_cnt = sum(light_df['casual'])
heavy_cas_cnt = sum(heavy_df['casual'])

weather_reg = [clear_cas_cnt, mist_cas_cnt, light_cas_cnt, heavy_cas_cnt]
plt.bar(x, weather_reg)
plt.xticks(x, ('clear', 'misty', 'light rain/snow', 'heavy rain/snow'))
plt.title('Total bike usage for each weather condition by casual users')
plt.ylim(0, 2300000)
plt.show()
plt.clf()

# Seaborn experimentation
fall_2011_dist = sns.distplot(Fall2011_df['cnt'], rug = True)

# Observe how ride count falls off dramtically with changing weather conditions
clear_mean = clear_df.mean()
mist_mean  = mist_df.mean()
light_mean = light_df.mean()
heavy_mean = heavy_df.mean()

