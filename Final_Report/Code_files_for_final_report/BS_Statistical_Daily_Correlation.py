# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 14:52:54 2018

Daily Correlation Statistics

@author: Jon
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 12:47:27 2018

Statistical analysis: Correlation

@author: Jon
"""
# Import common packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'Accent'
import os
# import seaborn as sns

#Change the working directory
# ATTN: You will need to change this locally.
os.chdir('C:/Users/Jon/Documents/Springboard/Capstone1/Bike-Sharing-Dataset')

# Read in the csv's for day.csv and day.csv
day_df = pd.read_csv('day.csv')
day_df = pd.read_csv('day.csv')

# Make an array for each table
day_array = np.array(day_df.values)


# Plot the raw data before setting the datetime index
# day_df.plot()
# plt.show()

Spring_df = day_df.loc[day_df['season'] == 1]
Summer_df = day_df.loc[day_df['season'] == 2]
Fall_df   = day_df.loc[day_df['season'] == 3]
Winter_df = day_df.loc[day_df['season'] == 4]

# Convert the 'Date' column into a collection of datetime objects: df.Date
day_df.dteday = pd.to_datetime(day_df['dteday'])

# Set the index to be the converted 'Date' column
day_df.set_index('dteday', inplace=True)

# Create daily dataframe for both years
day_df_2011 = day_df.loc[day_df['yr'] == 0]
day_df_2012 = day_df.loc[day_df['yr'] == 1]


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

# Set up the dataframes for different times of day
# Begin with simply splitting it up annually


# Seasonal frames for total users
Spring_cnt = Spring_df['cnt']
Summer_cnt = Summer_df['cnt']
Fall_cnt   = Fall_df['cnt']
Winter_cnt = Winter_df['cnt']


# Seasonal frames for registered users
Spring_reg = Spring_df['registered']
Summer_reg = Summer_df['registered']
Fall_reg   =   Fall_df['registered']
Winter_reg = Winter_df['registered']
                        

Spring_cas = Spring_df['casual']
Summer_cas = Summer_df['casual']
Fall_cas   =   Fall_df['casual']
Winter_cas = Winter_df['casual']


# 2011 Frames

# 2011 frames for total users
Spring11_cnt = Spring2011_df['cnt']
Summer11_cnt = Summer2011_df['cnt']
Fall11_cnt   =   Fall2011_df['cnt']
Winter11_cnt = Winter2011_df['cnt']

# 2011 frames for registered users
Spring11_reg = Spring2011_df['registered']
Summer11_reg = Summer2011_df['registered']
Fall11_reg   =   Fall2011_df['registered']
Winter11_reg = Winter2011_df['registered']
Reg_2011 = [Spring11_reg, Summer11_reg, Fall11_reg, Winter11_reg]

# 2011 frames for casual users
Spring11_cas = Spring2011_df['casual']
Summer11_cas = Summer2011_df['casual']
Fall11_cas   =   Fall2011_df['casual']
Winter11_cas = Winter2011_df['casual']
Cas_2011 = [Spring11_cas, Summer11_cas, Fall11_cas, Winter11_cas]




# 2012 frames

# 2012 frames for total users
Spring12_cnt = Spring2012_df['cnt']
Summer12_cnt = Summer2012_df['cnt']
Fall12_cnt   =   Fall2012_df['cnt']
Winter12_cnt = Winter2012_df['cnt']



# 2012 frames for registered users
Spring12_reg = Spring2012_df['registered']
Summer12_reg = Summer2012_df['registered']
Fall12_reg   =   Fall2012_df['registered']
Winter12_reg = Winter2012_df['registered']
Reg_2012 = [Spring12_reg, Summer12_reg, Fall12_reg, Winter12_reg]



# Seasonal frames for casual users
# 2012 frames for casual users
Spring12_cas = Spring2012_df['casual']
Summer12_cas = Summer2012_df['casual']
Fall12_cas   =   Fall2012_df['casual']
Winter12_cas = Winter2012_df['casual']
Cas_2012 = [Spring12_cas, Summer12_cas, Fall12_cas, Winter12_cas]





# Use the pearson r function from Datacamp, author: Jason Bois
def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute correlation matrix: corr_mat
    corr_mat = np.corrcoef(x,y)
    # Return entry [0,1]
    return corr_mat[0,1]
# End of function definition from Jason Bois
    

cnt_day   = np.asarray(day_df['cnt'])
reg_day   = np.asarray(day_df['registered'])
cas_day   = np.asarray(day_df['casual'])
temp_day  = np.asarray(day_df['temp'])
wind_day  = np.asarray(day_df['windspeed'])
hum_day   = np.asarray(day_df['hum'])

cnt_temp_r = pearson_r(cnt_day, temp_day)
cnt_wind_r = pearson_r(cnt_day, wind_day)
cnt_hum_r  = pearson_r(cnt_day, hum_day) 

print('\n')
print('Correlation of total user count and temperature is: ', cnt_temp_r)
print('Correlation of total user count and windspeed   is: ', cnt_wind_r)
print('Correlation of total user count and humidity    is: ', cnt_hum_r)
print('\n')

reg_temp_r = pearson_r(reg_day, temp_day)
reg_wind_r = pearson_r(reg_day, wind_day)
reg_hum_r  = pearson_r(reg_day, hum_day) 
print('\n')
print('Correlation of registered user count and temperature is: ', reg_temp_r)
print('Correlation of registered user count and windspeed   is: ', reg_wind_r)
print('Correlation of registered user count and humidity    is: ', reg_hum_r)
print('\n')

cas_temp_r = pearson_r(cas_day, temp_day)
cas_wind_r = pearson_r(cas_day, wind_day)
cas_hum_r  = pearson_r(cas_day, hum_day) 
print('\n')
print('Correlation of casual user count and temperature is: ', cas_temp_r)
print('Correlation of casual user count and windspeed   is: ', cas_wind_r)
print('Correlation of casual user count and humidity    is: ', cas_hum_r)
print('\n')




# Let's try these correlation tests during different seasons: 
##########################
# Spring Correlations
##########################
print('--------------------------------------')
print('Spring correlations')
print('--------------------------------------')

print('All users: ')
cnt_day   = np.asarray(Spring_df['cnt'])
reg_day   = np.asarray(Spring_df['registered'])
cas_day   = np.asarray(Spring_df['casual'])
temp_day  = np.asarray(Spring_df['temp'])
wind_day  = np.asarray(Spring_df['windspeed'])
hum_day   = np.asarray(Spring_df['hum'])

cnt_temp_r = pearson_r(cnt_day, temp_day)
cnt_wind_r = pearson_r(cnt_day, wind_day)
cnt_hum_r  = pearson_r(cnt_day, hum_day) 
print('\n')
print('Correlation of total user count and temperature in the Spring is: ', cnt_temp_r)
print('Correlation of total user count and windspeed   in the Spring is: ', cnt_wind_r)
print('Correlation of total user count and humidity    in the Spring is: ', cnt_hum_r)
print('\n')
print('Registered users: ')
reg_temp_r = pearson_r(reg_day, temp_day)
reg_wind_r = pearson_r(reg_day, wind_day)
reg_hum_r  = pearson_r(reg_day, hum_day) 
print('\n')
print('Correlation of registered user count and temperature in the Spring is: ', reg_temp_r)
print('Correlation of registered user count and windspeed   in the Spring is: ', reg_wind_r)
print('Correlation of registered user count and humidity    in the Spring is: ', reg_hum_r)
print('\n')
print('Casual users: ')
cas_temp_r = pearson_r(cas_day, temp_day)
cas_wind_r = pearson_r(cas_day, wind_day)
cas_hum_r  = pearson_r(cas_day, hum_day) 
print('\n')
print('Correlation of casual user count and temperature in Spring is: ', cas_temp_r)
print('Correlation of casual user count and windspeed   in Spring is: ', cas_wind_r)
print('Correlation of casual user count and humidity    in Spring is: ', cas_hum_r)
print('\n')


##########################
# Summer Correlations
##########################
print('--------------------------------------')
print('Summer correlations')
print('--------------------------------------') 
cnt_day   = np.asarray(Summer_df['cnt'])
reg_day   = np.asarray(Summer_df['registered'])
cas_day   = np.asarray(Summer_df['casual'])
temp_day  = np.asarray(Summer_df['temp'])
wind_day  = np.asarray(Summer_df['windspeed'])
hum_day   = np.asarray(Summer_df['hum'])
print('All users: ')
cnt_temp_r = pearson_r(cnt_day, temp_day)
cnt_wind_r = pearson_r(cnt_day, wind_day)
cnt_hum_r  = pearson_r(cnt_day, hum_day) 
print('\n')
print('Correlation of total user count and temperature in the Summer is: ', cnt_temp_r)
print('Correlation of total user count and windspeed   in the Summer is: ', cnt_wind_r)
print('Correlation of total user count and humidity    in the Summer is: ', cnt_hum_r)
print('\n')
print('Registered users: ')
reg_temp_r = pearson_r(reg_day, temp_day)
reg_wind_r = pearson_r(reg_day, wind_day)
reg_hum_r  = pearson_r(reg_day, hum_day) 
print('\n')
print('Correlation of registered user count and temperature in the Summer is: ', reg_temp_r)
print('Correlation of registered user count and windspeed   in the Summer is: ', reg_wind_r)
print('Correlation of registered user count and humidity    in the Summer is: ', reg_hum_r)
print('\n')
print('Casual users: ')
cas_temp_r = pearson_r(cas_day, temp_day)
cas_wind_r = pearson_r(cas_day, wind_day)
cas_hum_r  = pearson_r(cas_day, hum_day) 
print('\n')
print('Correlation of casual user count and temperature in the Summer is: ', cas_temp_r)
print('Correlation of casual user count and windspeed   in the Summer is: ', cas_wind_r)
print('Correlation of casual user count and humidity    in the Summer is: ', cas_hum_r)
print('\n')


##########################
# Fall Correlations
##########################
print('--------------------------------------')
print('Fall correlations')
print('--------------------------------------')   
cnt_day   = np.asarray(Fall_df['cnt'])
reg_day   = np.asarray(Fall_df['registered'])
cas_day   = np.asarray(Fall_df['casual'])
temp_day  = np.asarray(Fall_df['temp'])
wind_day  = np.asarray(Fall_df['windspeed'])
hum_day   = np.asarray(Fall_df['hum'])
print('All users: ')
cnt_temp_r = pearson_r(cnt_day, temp_day)
cnt_wind_r = pearson_r(cnt_day, wind_day)
cnt_hum_r  = pearson_r(cnt_day, hum_day) 
print('\n')
print('Correlation of total user count and temperature in the Fall is: ', cnt_temp_r)
print('Correlation of total user count and windspeed   in the Fall is: ', cnt_wind_r)
print('Correlation of total user count and humidity    in the Fall is: ', cnt_hum_r)
print('\n')
print('Registered users: ')
reg_temp_r = pearson_r(reg_day, temp_day)
reg_wind_r = pearson_r(reg_day, wind_day)
reg_hum_r  = pearson_r(reg_day, hum_day) 
print('\n')
print('Correlation of registered user count and temperature in the Fall is: ', reg_temp_r)
print('Correlation of registered user count and windspeed   in the Fall is: ', reg_wind_r)
print('Correlation of registered user count and humidity    in the Fall is: ', reg_hum_r)
print('\n')
print('Casual users: ')
cas_temp_r = pearson_r(cas_day, temp_day)
cas_wind_r = pearson_r(cas_day, wind_day)
cas_hum_r  = pearson_r(cas_day, hum_day) 
print('\n')
print('Correlation of casual user count and temperature in the Fall is: ', cas_temp_r)
print('Correlation of casual user count and windspeed   in the Fall is: ', cas_wind_r)
print('Correlation of casual user count and humidity    in the Fall is: ', cas_hum_r)
print('\n')

##########################
# Winter Correlations
##########################
print('--------------------------------------')
print('Winter correlations')
print('--------------------------------------')  
cnt_day   = np.asarray(Winter_df['cnt'])
reg_day   = np.asarray(Winter_df['registered'])
cas_day   = np.asarray(Winter_df['casual'])
temp_day  = np.asarray(Winter_df['temp'])
wind_day  = np.asarray(Winter_df['windspeed'])
hum_day   = np.asarray(Winter_df['hum'])
print('All users: ')
cnt_temp_r = pearson_r(cnt_day, temp_day)
cnt_wind_r = pearson_r(cnt_day, wind_day)
cnt_hum_r  = pearson_r(cnt_day, hum_day) 
print('\n')
print('Correlation of total user count and temperature in the Winter is: ', cnt_temp_r)
print('Correlation of total user count and windspeed   in the Winter is: ', cnt_wind_r)
print('Correlation of total user count and humidity    in the Winter is: ', cnt_hum_r)
print('\n')
print('Registered users: ')
reg_temp_r = pearson_r(reg_day, temp_day)
reg_wind_r = pearson_r(reg_day, wind_day)
reg_hum_r  = pearson_r(reg_day, hum_day) 
print('\n')
print('Correlation of registered user count and temperature in the Winter is: ', reg_temp_r)
print('Correlation of registered user count and windspeed   in the Winter is: ', reg_wind_r)
print('Correlation of registered user count and humidity    in the Winter is: ', reg_hum_r)
print('\n')
print('Casual users: ')
cas_temp_r = pearson_r(cas_day, temp_day)
cas_wind_r = pearson_r(cas_day, wind_day)
cas_hum_r  = pearson_r(cas_day, hum_day) 
print('\n')
print('Correlation of casual user count and temperature in the Winter is: ', cas_temp_r)
print('Correlation of casual user count and windspeed   in the Winter is: ', cas_wind_r)
print('Correlation of casual user count and humidity    in the Winter is: ', cas_hum_r)
print('\n')


#=============================================================================
# Correlation matrix
print('\n')
print('-----------------------------------------')
print('Correlation Matries')
print('-----------------------------------------')
names = ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']
continuous_day_df = day_df[names]
correlations = continuous_day_df.corr()
fig = plt.figure(figsize=(8, 8), dpi=80)
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax, cmap = 'autumn')
plt.title('Correlation Matrix for all seasons during 2011-2012')
ticks = np.arange(0,8,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()
plt.clf()


# Scatterplot Matrix
from pandas.plotting import scatter_matrix
scatter_matrix(continuous_day_df, alpha=0.2, figsize=(8, 8), diagonal='kde')
plt.title('Scatterplot Matrix for all seasons during 2011-2012')
plt.show()
plt.clf()
#=============================================================================

#=============================================================================
# Correlation matrix for 2011 
print('\n')
print('-----------------------------------------')
print('Correlation Matries')
print('-----------------------------------------')
names = ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']
continuous_day_df = day_df_2011[names]
correlations = continuous_day_df.corr()
fig = plt.figure(figsize=(8, 8), dpi=80)
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax, cmap = 'autumn')
plt.title('Correlation Matrix for all seasons during 2011')
ticks = np.arange(0,8,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()
plt.clf()


# Scatterplot Matrix
from pandas.plotting import scatter_matrix
scatter_matrix(continuous_day_df, alpha=0.2, figsize=(8, 8), diagonal='kde')
plt.title('Scatterplot Matrix for all seasons during 2011')
plt.show()
plt.clf()
#=============================================================================


#=============================================================================
# Correlation matrix for 2012 
print('\n')
print('-----------------------------------------')
print('Correlation Matries')
print('-----------------------------------------')
names = ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']
continuous_day_df = day_df_2012[names]
correlations = continuous_day_df.corr()
fig = plt.figure(figsize=(8, 8), dpi=80)
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax, cmap = 'autumn')
plt.title('Correlation Matrix for all seasons during 2012')
ticks = np.arange(0,8,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()
plt.clf()


# Scatterplot Matrix
from pandas.plotting import scatter_matrix
scatter_matrix(continuous_day_df, alpha=0.2, figsize=(8, 8), diagonal='kde')
plt.title('Scatterplot Matrix for all seasons during 2012')
plt.show()
plt.clf()
#=============================================================================


#=============================================================================
# Correlation matrix for Spring 
print('\n')
print('-----------------------------------------')
print('Correlation Matries')
print('-----------------------------------------')
names = ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']
continuous_day_df = Spring_df[names]
correlations = continuous_day_df.corr()
fig = plt.figure(figsize=(8, 8), dpi=80)
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax, cmap = 'autumn')
plt.title('Correlation Matrix for Spring 2011 & 2012')
ticks = np.arange(0,8,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()
plt.clf()


# Scatterplot Matrix
from pandas.plotting import scatter_matrix
scatter_matrix(continuous_day_df, alpha=0.2, figsize=(8, 8), diagonal='kde')
plt.title('Scatterplot Matrix for Spring 2011 & 2012')
plt.show()
plt.clf()
#=============================================================================


#=============================================================================
# Correlation matrix for Summer 
print('\n')
print('-----------------------------------------')
print('Correlation Matries')
print('-----------------------------------------')
names = ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']
continuous_day_df = Summer_df[names]
correlations = continuous_day_df.corr()
fig = plt.figure(figsize=(8, 8), dpi=80)
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax, cmap = 'autumn')
plt.title('Correlation Matrix for Summer 2011 & 2012')
ticks = np.arange(0,8,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()
plt.clf()


# Scatterplot Matrix
from pandas.plotting import scatter_matrix
scatter_matrix(continuous_day_df, alpha=0.2, figsize=(8, 8), diagonal='kde')
plt.title('Scatterplot Matrix for Summer 2011 & 2012')
plt.show()
plt.clf()
#=============================================================================


#=============================================================================
# Correlation matrix for Fall 
print('\n')
print('-----------------------------------------')
print('Correlation Matries')
print('-----------------------------------------')
names = ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']
continuous_day_df = Fall_df[names]
correlations = continuous_day_df.corr()
fig = plt.figure(figsize=(8, 8), dpi=80)
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax, cmap = 'autumn')
plt.title('Correlation Matrix for Fall 2011 & 2012')
ticks = np.arange(0,8,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()
plt.clf()


# Scatterplot Matrix
from pandas.plotting import scatter_matrix
scatter_matrix(continuous_day_df, alpha=0.2, figsize=(8, 8), diagonal='kde')
plt.title('Scatterplot Matrix for Fall 2011 & 2012')
plt.show()
plt.clf()
#=============================================================================


#=============================================================================
# Correlation matrix for Winter 
print('\n')
print('-----------------------------------------')
print('Correlation Matries')
print('-----------------------------------------')
names = ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']
continuous_day_df = Winter_df[names]
correlations = continuous_day_df.corr()
fig = plt.figure(figsize=(8, 8), dpi=80)
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax, cmap = 'autumn')
plt.title('Correlation Matrix for Winter 2011 & 2012')
ticks = np.arange(0,8,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()
plt.clf()


# Scatterplot Matrix
from pandas.plotting import scatter_matrix
scatter_matrix(continuous_day_df, alpha=0.2, figsize=(8, 8), diagonal='kde')
plt.title('Scatterplot Matrix for Winter 2011 & 2012')
plt.show()
plt.clf()
#=============================================================================