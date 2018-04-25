# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 00:41:34 2018

Ride count on different days of the week.

@author: Jon
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


# =============================================================================
# # Sort into seasonal dataframes
# Spring_hour_df = hour_df.loc[hour_df['season'] == 1]
# Summer_hour_df = hour_df.loc[hour_df['season'] == 2]
# Fall_hour_df   = hour_df.loc[hour_df['season'] == 3]
# Winter_hour_df = hour_df.loc[hour_df['season'] == 4]
# 
# #Make annual data frames
# day_df_2011 = day_df.loc[day_df['yr'] == 0]
# day_df_2012 = day_df.loc[day_df['yr'] == 1]
# 
# #Make annual data frames
# hour_df_2011 = hour_df.loc[hour_df['yr'] == 0]
# hour_df_2012 = hour_df.loc[hour_df['yr'] == 1]
# 
# hour_counts    = hour_df['cnt'].values
# hour_counts_11 = hour_df_2011['cnt'].values
# hour_counts_12 = hour_df_2012['cnt'].values
# =============================================================================

day_df.plot(y = 'cnt')
plt.title('2011-2012 count of users per hour')
plt.xlabel('Windspeed (mph) ')
plt.ylabel('User count')
plt.show()
plt.clf()

day_df_holiday = day_df.loc[day_df['holiday'] == 1]
day_df_NonHoliday = day_df.loc[day_df['holiday'] == 0]

day_holiday_mean = day_df_holiday.mean()
day_NonHoliday_mean = day_df_NonHoliday.mean()


hour_df_holiday = hour_df.loc[hour_df['holiday'] == 1]
hour_df_NonHoliday = hour_df.loc[hour_df['holiday'] == 0]

hour_holiday_mean  = hour_df_holiday.mean()
hour_holiday_NonHoliday = hour_df_NonHoliday.mean()


# =============================================================================
# hour_df_holiday.plot(y = 'cnt')
# plt.title('2011-2012 count of users per hour')
# plt.xlabel('index')
# plt.ylabel('User count')
# plt.show()
# plt.clf()
# 
# =============================================================================

hour_df_NonHoliday.plot(y = 'cnt')
plt.title('2011-2012 count of users per hour')
plt.xlabel('Windspeed (mph) ')
plt.ylabel('User count')
plt.show()
plt.clf()


# Holidays are not causing the issues


day_0 = day_df.loc[day_df['weekday'] == 0] 
day_1 = day_df.loc[day_df['weekday'] == 1] 
day_2 = day_df.loc[day_df['weekday'] == 2] 
day_3 = day_df.loc[day_df['weekday'] == 3] 
day_4 = day_df.loc[day_df['weekday'] == 4] 
day_5 = day_df.loc[day_df['weekday'] == 5] 
day_6 = day_df.loc[day_df['weekday'] == 6] 


# Day of the week also does not seem to cause the problem
plt.plot(day_0['cnt'], label = 'Sunday', color = 'pink')  # Sunday
plt.plot(day_1['cnt'], label = 'Monday')                  # Monday
plt.plot(day_2['cnt'], label = 'Tuesday')                  # Tuesday
plt.plot(day_3['cnt'], label = 'Wednesday')                  # Wednesday
plt.plot(day_4['cnt'], label = 'Thursday')                  # Thursday
plt.plot(day_5['cnt'], label = 'Friday')                  # Friday
plt.plot(day_6['cnt'], label = 'Saturday' , color = 'gray')  # Saturday
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('Daily user counts')
plt.ylabel('rides' )
plt.plot()
plt.show()
plt.clf()

# Total user barplot

Sun_sum   = sum(day_0['cnt'])
Mon_sum   = sum(day_1['cnt'])
Tues_sum  = sum(day_2['cnt'])
Wed_sum   = sum(day_3['cnt'])
Thurs_sum = sum(day_4['cnt'])
Fri_sum   = sum(day_5['cnt'])
Sat_sum   = sum(day_6['cnt'])

days_sum = [Sun_sum, Mon_sum, Tues_sum, Wed_sum, Thurs_sum, Fri_sum, Sat_sum]
days_sum_array = np.array(days_sum)
x = np.arange(7)

plt.figure(figsize=(8,4))
plt.bar(x, days_sum_array, color = 'green')
plt.xticks(x, ('Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'))
plt.title('Total bike usage by day of the week')
plt.xlabel('Day of the week')
plt.ylabel('Rider count')
plt.ylim(0, 700000)
plt.show()
plt.clf()


#Registered barplot

Sun_reg_sum   = sum(day_0['registered'])
Mon_reg_sum   = sum(day_1['registered'])
Tues_reg_sum  = sum(day_2['registered'])
Wed_reg_sum   = sum(day_3['registered'])
Thurs_reg_sum = sum(day_4['registered'])
Fri_reg_sum   = sum(day_5['registered'])
Sat_reg_sum   = sum(day_6['registered'])

days_reg_sum = [Sun_reg_sum, Mon_reg_sum, Tues_reg_sum, Wed_reg_sum, Thurs_reg_sum, Fri_reg_sum, Sat_reg_sum]
days_reg_sum_array = np.array(days_reg_sum)
x = np.arange(7)

plt.figure(figsize=(8,4))
plt.bar(x, days_reg_sum_array, color = 'grey')
plt.xticks(x, ('Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'))
plt.title('Registered user bike usage by day of the week')
plt.xlabel('Day of the week')
plt.ylabel('Rider count')
plt.ylim(0, 700000)
plt.show()
plt.clf()


# Casual barplot
Sun_cas_sum   = sum(day_0['casual'])
Mon_cas_sum   = sum(day_1['casual'])
Tues_cas_sum  = sum(day_2['casual'])
Wed_cas_sum   = sum(day_3['casual'])
Thurs_cas_sum = sum(day_4['casual'])
Fri_cas_sum   = sum(day_5['casual'])
Sat_cas_sum   = sum(day_6['casual'])

days_cas_sum = [Sun_cas_sum, Mon_cas_sum, Tues_cas_sum, Wed_cas_sum, Thurs_cas_sum, Fri_cas_sum, Sat_cas_sum]
days_cas_sum_array = np.array(days_cas_sum)
x = np.arange(7)

plt.figure(figsize=(8,4))
plt.bar(x, days_cas_sum_array, color = 'indigo')
plt.xticks(x, ('Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'))
plt.title('Casual user bike usage by day of the week')
plt.xlabel('Day of the week')
plt.ylabel('Rider count')
plt.ylim(0, 700000)
plt.show()
plt.clf()