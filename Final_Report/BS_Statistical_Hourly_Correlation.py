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

# Read in the csv's for day.csv and hour.csv
day_df = pd.read_csv('day.csv')
hour_df = pd.read_csv('hour.csv')

# Make an array for each table
day_array = np.array(day_df.values)
hour_array = np.array(hour_df.values)


# Plot the raw data before setting the datetime index
# day_df.plot()
# plt.show()

Spring_df = hour_df.loc[hour_df['season'] == 1]
Summer_df = hour_df.loc[hour_df['season'] == 2]
Fall_df   = hour_df.loc[hour_df['season'] == 3]
Winter_df = hour_df.loc[hour_df['season'] == 4]

# Convert the 'Date' column into a collection of datetime objects: df.Date
day_df.dteday = pd.to_datetime(hour_df['dteday'])

# Set the index to be the converted 'Date' column
day_df.set_index('dteday', inplace=True)

# Create daily dataframe for both years
hour_df_2011 = hour_df.loc[hour_df['yr'] == 0]
hour_df_2012 = hour_df.loc[hour_df['yr'] == 1]


# Create 2011 seasonal dataframe
Spring2011_df = hour_df_2011.loc[hour_df_2011['season'] == 1]
Summer2011_df = hour_df_2011.loc[hour_df_2011['season'] == 2]
Fall2011_df = hour_df_2011.loc[hour_df_2011['season'] == 3]
Winter2011_df = hour_df_2011.loc[hour_df_2011['season'] == 4]

# Create 2012 seasonal dataframe

Spring2012_df = hour_df_2012.loc[hour_df_2012['season'] == 1]
Summer2012_df = hour_df_2012.loc[hour_df_2012['season'] == 2]
Fall2012_df = hour_df_2012.loc[hour_df_2012['season'] == 3]
Winter2012_df = hour_df_2012.loc[hour_df_2012['season'] == 4]

# Set up the dataframes for different times of day

# Hour df broken down into Early Am (0-5), Morning (5-10), Midday (11-14), Afternoon(15-17), Evening (18-20), Night (21-24)
EarlyAm_df   = hour_df[(hour_df['hr'] <= 5)]
Morning_df   = hour_df[(hour_df['hr'] > 5) & (hour_df['hr'] <= 10)]
Midday_df    = hour_df[(hour_df['hr'] > 10) & (hour_df['hr'] <= 14)]
Afternoon_df = hour_df[(hour_df['hr'] > 14) & (hour_df['hr'] <= 17)]
Evening_df   = hour_df[(hour_df['hr'] > 17) & (hour_df['hr'] <= 20)]
Night_df     = hour_df[(hour_df['hr'] > 20) & (hour_df['hr'] <= 24)]

Early_Riders = EarlyAm_df['cnt']
Morning_Riders = Morning_df['cnt']
Midday_Riders = Midday_df['cnt']
Afternoon_Riders = Afternoon_df['cnt']
Evening_Riders = Evening_df['cnt']
Night_Riders = Night_df['cnt']


Early_reg = EarlyAm_df['registered']
Morning_reg = Morning_df['registered']
Midday_reg = Midday_df['registered']
Afternoon_reg = Afternoon_df['registered']
Evening_reg = Evening_df['registered']
Night_reg = Night_df['registered']

Early_cas = EarlyAm_df['casual']
Morning_cas = Morning_df['casual']
Midday_cas = Midday_df['casual']
Afternoon_cas = Afternoon_df['casual']
Evening_cas = Evening_df['casual']
Night_cas = Night_df['casual']


# Use the pearson r function from Datacamp, author: Jason Bois
def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute correlation matrix: corr_mat
    corr_mat = np.corrcoef(x,y)
    # Return entry [0,1]
    return corr_mat[0,1]
# End of function definition from Jason Bois
hr_array   = np.asarray(hour_df['hr'])    
cnt_hour   = np.asarray(hour_df['cnt'])
reg_hour   = np.asarray(hour_df['registered'])
cas_hour   = np.asarray(hour_df['casual'])
temp_hour  = np.asarray(hour_df['temp'])
wind_hour  = np.asarray(hour_df['windspeed'])
hum_hour   = np.asarray(hour_df['hum'])

cnt_temp_r = pearson_r(cnt_hour, temp_hour)
cnt_wind_r = pearson_r(cnt_hour, wind_hour)
cnt_hum_r  = pearson_r(cnt_hour, hum_hour) 
cnt_hour_r = pearson_r(cnt_hour, hr_array)
print('\n')
print('Correlation of total user count and temperature is: ', cnt_temp_r)
print('Correlation of total user count and windspeed   is: ', cnt_wind_r)
print('Correlation of total user count and humidity    is: ', cnt_hum_r)
print('Correlation of total user count and hour        is: ', cnt_hour_r)
print('\n')

reg_temp_r = pearson_r(reg_hour, temp_hour)
reg_wind_r = pearson_r(reg_hour, wind_hour)
reg_hum_r  = pearson_r(reg_hour, hum_hour) 
reg_hour_r = pearson_r(reg_hour, hr_array)
print('\n')
print('Correlation of registered user count and temperature is: ', reg_temp_r)
print('Correlation of registered user count and windspeed   is: ', reg_wind_r)
print('Correlation of registered user count and humidity    is: ', reg_hum_r)
print('Correlation of registered user count and hour        is: ', reg_hour_r)
print('\n')

cas_temp_r = pearson_r(cas_hour, temp_hour)
cas_wind_r = pearson_r(cas_hour, wind_hour)
cas_hum_r  = pearson_r(cas_hour, hum_hour) 
cas_hour_r = pearson_r(cas_hour, hr_array)
print('\n')
print('Correlation of casual user count and temperature is: ', cas_temp_r)
print('Correlation of casual user count and windspeed   is: ', cas_wind_r)
print('Correlation of casual user count and humidity    is: ', cas_hum_r)
print('Correlation of casual user count and hour        is: ', cas_hour_r)
print('\n')




# Let's try these correlation tests during different times of day: 
##########################
# Early AM Correlations\
##########################
print('--------------------------------------')
print('Early AM correlations')
print('--------------------------------------')

print('All users: ')
hr_array   = np.asarray(EarlyAm_df['hr'])    
cnt_hour   = np.asarray(EarlyAm_df['cnt'])
reg_hour   = np.asarray(EarlyAm_df['registered'])
cas_hour   = np.asarray(EarlyAm_df['casual'])
temp_hour  = np.asarray(EarlyAm_df['temp'])
wind_hour  = np.asarray(EarlyAm_df['windspeed'])
hum_hour   = np.asarray(EarlyAm_df['hum'])

cnt_temp_r = pearson_r(cnt_hour, temp_hour)
cnt_wind_r = pearson_r(cnt_hour, wind_hour)
cnt_hum_r  = pearson_r(cnt_hour, hum_hour) 
cnt_hour_r = pearson_r(cnt_hour, hr_array)
print('\n')
print('Correlation of total user count and temperature in EarlyAm is: ', cnt_temp_r)
print('Correlation of total user count and windspeed   in EarlyAm is: ', cnt_wind_r)
print('Correlation of total user count and humidity    in EarlyAm is: ', cnt_hum_r)
print('Correlation of total user count and hour        in EarlyAm is: ', cnt_hour_r)
print('\n')
print('Registered users: ')
reg_temp_r = pearson_r(reg_hour, temp_hour)
reg_wind_r = pearson_r(reg_hour, wind_hour)
reg_hum_r  = pearson_r(reg_hour, hum_hour) 
reg_hour_r = pearson_r(reg_hour, hr_array)
print('\n')
print('Correlation of registered user count and temperature in EarlyAm is: ', reg_temp_r)
print('Correlation of registered user count and windspeed   in EarlyAm is: ', reg_wind_r)
print('Correlation of registered user count and humidity    in EarlyAm is: ', reg_hum_r)
print('Correlation of registered user count and hour        in EarlyAm is: ', reg_hour_r)
print('\n')
print('Casual users: ')
cas_temp_r = pearson_r(cas_hour, temp_hour)
cas_wind_r = pearson_r(cas_hour, wind_hour)
cas_hum_r  = pearson_r(cas_hour, hum_hour) 
cas_hour_r = pearson_r(cas_hour, hr_array)
print('\n')
print('Correlation of casual user count and temperature in EarlyAm is: ', cas_temp_r)
print('Correlation of casual user count and windspeed   in EarlyAm is: ', cas_wind_r)
print('Correlation of casual user count and humidity    in EarlyAm is: ', cas_hum_r)
print('Correlation of casual user count and hour        in EarlyAm is: ', cas_hour_r)
print('\n')


##########################
# Morning Correlations
##########################
print('--------------------------------------')
print('Morning correlations')
print('--------------------------------------')
hr_array   = np.asarray(Morning_df['hr'])    
cnt_hour   = np.asarray(Morning_df['cnt'])
reg_hour   = np.asarray(Morning_df['registered'])
cas_hour   = np.asarray(Morning_df['casual'])
temp_hour  = np.asarray(Morning_df['temp'])
wind_hour  = np.asarray(Morning_df['windspeed'])
hum_hour   = np.asarray(Morning_df['hum'])
print('All users: ')
cnt_temp_r = pearson_r(cnt_hour, temp_hour)
cnt_wind_r = pearson_r(cnt_hour, wind_hour)
cnt_hum_r  = pearson_r(cnt_hour, hum_hour) 
cnt_hour_r = pearson_r(cnt_hour, hr_array)
print('\n')
print('Correlation of total user count and temperature in the Morning is: ', cnt_temp_r)
print('Correlation of total user count and windspeed   in the Morning is: ', cnt_wind_r)
print('Correlation of total user count and humidity    in the Morning is: ', cnt_hum_r)
print('Correlation of total user count and hour        in the Morning is: ', cnt_hour_r)
print('\n')
print('Registered users: ')
reg_temp_r = pearson_r(reg_hour, temp_hour)
reg_wind_r = pearson_r(reg_hour, wind_hour)
reg_hum_r  = pearson_r(reg_hour, hum_hour) 
reg_hour_r = pearson_r(reg_hour, hr_array)
print('\n')
print('Correlation of registered user count and temperature in the Morning is: ', reg_temp_r)
print('Correlation of registered user count and windspeed   in the Morning is: ', reg_wind_r)
print('Correlation of registered user count and humidity    in the Morning is: ', reg_hum_r)
print('Correlation of registered user count and hour        in the Morning is: ', reg_hour_r)
print('\n')
print('Casual users: ')
cas_temp_r = pearson_r(cas_hour, temp_hour)
cas_wind_r = pearson_r(cas_hour, wind_hour)
cas_hum_r  = pearson_r(cas_hour, hum_hour) 
cas_hour_r = pearson_r(cas_hour, hr_array)
print('\n')
print('Correlation of casual user count and temperature in the Morning is: ', cas_temp_r)
print('Correlation of casual user count and windspeed   in the Morning is: ', cas_wind_r)
print('Correlation of casual user count and humidity    in the Morning is: ', cas_hum_r)
print('Correlation of casual user count and hour        in the Morning is: ', cas_hour_r)
print('\n')


##########################
# Midday Correlations
##########################
print('--------------------------------------')
print('Midday correlations')
print('--------------------------------------')
hr_array   = np.asarray(Midday_df['hr'])    
cnt_hour   = np.asarray(Midday_df['cnt'])
reg_hour   = np.asarray(Midday_df['registered'])
cas_hour   = np.asarray(Midday_df['casual'])
temp_hour  = np.asarray(Midday_df['temp'])
wind_hour  = np.asarray(Midday_df['windspeed'])
hum_hour   = np.asarray(Midday_df['hum'])
print('All users: ')
cnt_temp_r = pearson_r(cnt_hour, temp_hour)
cnt_wind_r = pearson_r(cnt_hour, wind_hour)
cnt_hum_r  = pearson_r(cnt_hour, hum_hour) 
cnt_hour_r = pearson_r(cnt_hour, hr_array)
print('\n')
print('Correlation of total user count and temperature at Midday is: ', cnt_temp_r)
print('Correlation of total user count and windspeed   at Midday is: ', cnt_wind_r)
print('Correlation of total user count and humidity    at Midday is: ', cnt_hum_r)
print('Correlation of total user count and hour        at Midday is: ', cnt_hour_r)
print('\n')
print('Registered users: ')
reg_temp_r = pearson_r(reg_hour, temp_hour)
reg_wind_r = pearson_r(reg_hour, wind_hour)
reg_hum_r  = pearson_r(reg_hour, hum_hour) 
reg_hour_r = pearson_r(reg_hour, hr_array)
print('\n')
print('Correlation of registered user count and temperature at Midday is: ', reg_temp_r)
print('Correlation of registered user count and windspeed   at Midday is: ', reg_wind_r)
print('Correlation of registered user count and humidity    at Midday is: ', reg_hum_r)
print('Correlation of registered user count and hour        at Midday is: ', reg_hour_r)
print('\n')
print('Casual users: ')
cas_temp_r = pearson_r(cas_hour, temp_hour)
cas_wind_r = pearson_r(cas_hour, wind_hour)
cas_hum_r  = pearson_r(cas_hour, hum_hour) 
cas_hour_r = pearson_r(cas_hour, hr_array)
print('\n')
print('Correlation of casual user count and temperature at Midday is: ', cas_temp_r)
print('Correlation of casual user count and windspeed   at Midday is: ', cas_wind_r)
print('Correlation of casual user count and humidity    at Midday is: ', cas_hum_r)
print('Correlation of casual user count and hour        at Midday is: ', cas_hour_r)
print('\n')

##########################
# Afternoon Correlations
##########################
print('--------------------------------------')
print('Afternoon correlations')
print('--------------------------------------')
hr_array   = np.asarray(Afternoon_df['hr'])    
cnt_hour   = np.asarray(Afternoon_df['cnt'])
reg_hour   = np.asarray(Afternoon_df['registered'])
cas_hour   = np.asarray(Afternoon_df['casual'])
temp_hour  = np.asarray(Afternoon_df['temp'])
wind_hour  = np.asarray(Afternoon_df['windspeed'])
hum_hour   = np.asarray(Afternoon_df['hum'])
print('All users: ')
cnt_temp_r = pearson_r(cnt_hour, temp_hour)
cnt_wind_r = pearson_r(cnt_hour, wind_hour)
cnt_hum_r  = pearson_r(cnt_hour, hum_hour) 
cnt_hour_r = pearson_r(cnt_hour, hr_array)
print('\n')
print('Correlation of total user count and temperature in the Afternoon is: ', cnt_temp_r)
print('Correlation of total user count and windspeed   in the Afternoon is: ', cnt_wind_r)
print('Correlation of total user count and humidity    in the Afternoon is: ', cnt_hum_r)
print('Correlation of total user count and hour        in the Afternoon is: ', cnt_hour_r)
print('\n')
print('Registered users: ')
reg_temp_r = pearson_r(reg_hour, temp_hour)
reg_wind_r = pearson_r(reg_hour, wind_hour)
reg_hum_r  = pearson_r(reg_hour, hum_hour) 
reg_hour_r = pearson_r(reg_hour, hr_array)
print('\n')
print('Correlation of registered user count and temperature in the Afternoon is: ', reg_temp_r)
print('Correlation of registered user count and windspeed   in the Afternoon is: ', reg_wind_r)
print('Correlation of registered user count and humidity    in the Afternoon is: ', reg_hum_r)
print('Correlation of registered user count and hour        in the Afternoon is: ', reg_hour_r)
print('\n')
print('Casual users: ')
cas_temp_r = pearson_r(cas_hour, temp_hour)
cas_wind_r = pearson_r(cas_hour, wind_hour)
cas_hum_r  = pearson_r(cas_hour, hum_hour) 
cas_hour_r = pearson_r(cas_hour, hr_array)
print('\n')
print('Correlation of casual user count and temperature in the Afternoon is: ', cas_temp_r)
print('Correlation of casual user count and windspeed   in the Afternoon is: ', cas_wind_r)
print('Correlation of casual user count and humidity    in the Afternoon is: ', cas_hum_r)
print('Correlation of casual user count and hour        in the Afternoon is: ', cas_hour_r)
print('\n')


##########################
# Evening Correlations
##########################
print('--------------------------------------')
print('Evening correlations')
print('--------------------------------------')
hr_array   = np.asarray(Evening_df['hr'])    
cnt_hour   = np.asarray(Evening_df['cnt'])
reg_hour   = np.asarray(Evening_df['registered'])
cas_hour   = np.asarray(Evening_df['casual'])
temp_hour  = np.asarray(Evening_df['temp'])
wind_hour  = np.asarray(Evening_df['windspeed'])
hum_hour   = np.asarray(Evening_df['hum'])
print('All users: ')
cnt_temp_r = pearson_r(cnt_hour, temp_hour)
cnt_wind_r = pearson_r(cnt_hour, wind_hour)
cnt_hum_r  = pearson_r(cnt_hour, hum_hour) 
cnt_hour_r = pearson_r(cnt_hour, hr_array)
print('\n')
print('Correlation of total user count and temperature in the Evening is: ', cnt_temp_r)
print('Correlation of total user count and windspeed   in the Evening is: ', cnt_wind_r)
print('Correlation of total user count and humidity    in the Evening is: ', cnt_hum_r)
print('Correlation of total user count and hour        in the Evening is: ', cnt_hour_r)
print('\n')
print('Registered users: ')
reg_temp_r = pearson_r(reg_hour, temp_hour)
reg_wind_r = pearson_r(reg_hour, wind_hour)
reg_hum_r  = pearson_r(reg_hour, hum_hour) 
reg_hour_r = pearson_r(reg_hour, hr_array)
print('\n')
print('Correlation of registered user count and temperature in the Evening is: ', reg_temp_r)
print('Correlation of registered user count and windspeed   in the Evening is: ', reg_wind_r)
print('Correlation of registered user count and humidity    in the Evening is: ', reg_hum_r)
print('Correlation of registered user count and hour        in the Evening is: ', reg_hour_r)
print('\n')
print('Casual users: ')
cas_temp_r = pearson_r(cas_hour, temp_hour)
cas_wind_r = pearson_r(cas_hour, wind_hour)
cas_hum_r  = pearson_r(cas_hour, hum_hour) 
cas_hour_r = pearson_r(cas_hour, hr_array)
print('\n')
print('Correlation of casual user count and temperature in the Evening is: ', cas_temp_r)
print('Correlation of casual user count and windspeed   in the Evening is: ', cas_wind_r)
print('Correlation of casual user count and humidity    in the Evening is: ', cas_hum_r)
print('Correlation of casual user count and hour        in the Evening is: ', cas_hour_r)
print('\n')

##########################
# Night Correlations
##########################
print('\n')
print('--------------------------------------')
print('Nighttime correlations')
print('--------------------------------------')

hr_array   = np.asarray(Night_df['hr'])    
cnt_hour   = np.asarray(Night_df['cnt'])
reg_hour   = np.asarray(Night_df['registered'])
cas_hour   = np.asarray(Night_df['casual'])
temp_hour  = np.asarray(Night_df['temp'])
wind_hour  = np.asarray(Night_df['windspeed'])
hum_hour   = np.asarray(Night_df['hum'])
print('All users: ')
cnt_temp_r = pearson_r(cnt_hour, temp_hour)
cnt_wind_r = pearson_r(cnt_hour, wind_hour)
cnt_hum_r  = pearson_r(cnt_hour, hum_hour) 
cnt_hour_r = pearson_r(cnt_hour, hr_array)
print('\n')
print('Correlation of total user count and temperature at Night is: ', cnt_temp_r)
print('Correlation of total user count and windspeed   at Night is: ', cnt_wind_r)
print('Correlation of total user count and humidity    at Night is: ', cnt_hum_r)
print('Correlation of total user count and hour        at Night is: ', cnt_hour_r)
print('\n')
print('Registered users: ')
reg_temp_r = pearson_r(reg_hour, temp_hour)
reg_wind_r = pearson_r(reg_hour, wind_hour)
reg_hum_r  = pearson_r(reg_hour, hum_hour) 
reg_hour_r = pearson_r(reg_hour, hr_array)
print('\n')
print('Correlation of registered user count and temperature at Night is: ', reg_temp_r)
print('Correlation of registered user count and windspeed   at Night is: ', reg_wind_r)
print('Correlation of registered user count and humidity    at Night is: ', reg_hum_r)
print('Correlation of registered user count and hour        at Night is: ', reg_hour_r)
print('\n')
print('casual users: ')
cas_temp_r = pearson_r(cas_hour, temp_hour)
cas_wind_r = pearson_r(cas_hour, wind_hour)
cas_hum_r  = pearson_r(cas_hour, hum_hour) 
cas_hour_r = pearson_r(cas_hour, hr_array)
print('\n')
print('Correlation of casual user count and temperature at Night is: ', cas_temp_r)
print('Correlation of casual user count and windspeed   at Night is: ', cas_wind_r)
print('Correlation of casual user count and humidity    at Night is: ', cas_hum_r)
print('Correlation of casual user count and hour        at Night is: ', cas_hour_r)
print('\n')


# Correlation matrix
print('\n')
print('Correlation Matrix')
names = ['hr', 'temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']
continuous_hour_df = hour_df[names]
correlations = continuous_hour_df.corr()
fig = plt.figure(figsize=(8, 8), dpi=80)
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax, cmap = 'autumn')
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()
plt.clf()


# Scatterplot Matrix
from pandas.plotting import scatter_matrix
scatter_matrix(continuous_hour_df, alpha=0.2, figsize=(8, 8), diagonal='kde')
plt.show()
plt.clf()