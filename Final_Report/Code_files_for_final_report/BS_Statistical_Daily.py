# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 23:10:18 2018

Bike Sharing Statistical Analysis

@author: Jon
"""

# We're looking to add statistical evidence to our claims of different 
# patterns between hours and seasons of bike usage. 

# We will start with seeing if the means for the different seasons
# have significant differences
# 2011 seasonal statistics. 

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

# Implement the bootstrapping functions from the Datacamp Course: Statistical Thinking in Python (Part 1)
# Author: Justin Bois
def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""

    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n

    return x, y

# Implement the bootstrapping functions from the Datacamp Course: Statistical Thinking in Python (Part 2)
# Author: Justin Bois
def bootstrap_replicate_1d(data, func):
    return func(np.random.choice(data, size=len(data)))

def draw_bs_reps(data, func, size=1):
    """Draw bootstrap replicates."""

    # Initialize array of replicates: bs_replicates
    bs_replicates = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data, func)

    return bs_replicates
##
# End of section from Justin Bois's Datacamp course
##



# Make arrays of the rider rates 
Sp2011_Riders = Spring2011_df['cnt']
Su2011_Riders = Summer2011_df['cnt']
Fa2011_Riders = Fall2011_df['cnt']
Win2011_Riders = Winter2011_df['cnt']

Sp2012_Riders = Spring2012_df['cnt']
Su2012_Riders = Summer2012_df['cnt']
Fa2012_Riders = Fall2012_df['cnt']
Win2012_Riders = Winter2012_df['cnt']


# 2011 means and standard deviations for all riders
mu_sp2011 = np.mean(Sp2011_Riders)
sigma_sp2011 = np.std(Sp2011_Riders)

mu_su2011 = np.mean(Su2011_Riders)
sigma_su2011 = np.std(Su2011_Riders)

mu_fa2011 = np.mean(Fa2011_Riders)
sigma_fa2011 = np.std(Fa2011_Riders)

mu_wi2011 = np.mean(Win2011_Riders)
sigma_wi2011 = np.std(Win2011_Riders)

# 2012 means and standard deviations for all of riders
mu_sp2012 = np.mean(Sp2012_Riders)
sigma_sp2012 = np.std(Sp2012_Riders)

mu_su2012 = np.mean(Su2012_Riders)
sigma_su2012 = np.std(Su2012_Riders)

mu_fa2012 = np.mean(Fa2012_Riders)
sigma_fa2012 = np.std(Fa2012_Riders)

mu_wi2012 = np.mean(Win2012_Riders)
sigma_wi2012 = np.std(Win2012_Riders)


# Caluculate the difference in means between years
spring_mean_diff = mu_sp2012 - mu_sp2011

summer_mean_diff = mu_su2012 - mu_su2011

fall_mean_diff   = mu_fa2012 - mu_fa2011

winter_mean_diff = mu_wi2012 - mu_wi2011


# Concatenate the seasons: Season_concat
Spring_concat = np.concatenate((Sp2012_Riders, Sp2011_Riders))
Summer_concat = np.concatenate((Su2012_Riders, Su2011_Riders))
Fall_concat = np.concatenate((Fa2012_Riders, Fa2011_Riders))
Winter_concat = np.concatenate((Win2012_Riders, Win2011_Riders))

# Initialize bootstrap replicates: Spring_replicates
Spring_replicates = np.empty(10000)
Summer_replicates = np.empty(10000)
Fall_replicates = np.empty(10000)
Winter_replicates = np.empty(10000)

for i in range(10000):
    # Generate bootstrap sample for Spring
    Spring_sample = np.random.choice(Spring_concat, size=len(Spring_concat))
    
    # Compute replicates for Spring
    Spring_replicates[i] = np.mean(Spring_sample[:len(Sp2012_Riders)]) - np.mean(
                                     Spring_sample[len(Sp2011_Riders):])


    # Generate bootstrap sample for Summer
    Summer_sample = np.random.choice(Summer_concat, size=len(Summer_concat))
    
    # Compute replicates for Summer
    Summer_replicates[i] = np.mean(Summer_sample[:len(Su2012_Riders)]) - np.mean(
                                     Summer_sample[len(Su2011_Riders):])


    # Generate bootstrap sample for Fall
    Fall_sample = np.random.choice(Fall_concat, size=len(Fall_concat))
    
    # Compute replicates for Fall
    Fall_replicates[i] = np.mean(Fall_sample[:len(Fa2012_Riders)]) - np.mean(
                                     Fall_sample[len(Fa2011_Riders):])


    # Generate bootstrap sample for Winter
    Winter_sample = np.random.choice(Winter_concat, size=len(Winter_concat))
    
    # Compute replicates for Winter
    Winter_replicates[i] = np.mean(Winter_sample[:len(Win2012_Riders)]) - np.mean(
                                     Winter_sample[len(Win2011_Riders):])


# Compute and print p-value: p
p_Spring = np.sum(Spring_replicates >= spring_mean_diff) / len(Spring_replicates)
p_Summer = np.sum(Summer_replicates >= summer_mean_diff) / len(Summer_replicates)
p_Fall   = np.sum(Fall_replicates   >= fall_mean_diff  ) / len(Fall_replicates)
p_Winter = np.sum(Winter_replicates >= winter_mean_diff) / len(Winter_replicates)
print('Spring p-value =', p_Spring)
print('Summer p-value =', p_Summer)
print('Fall p-value =',   p_Fall)
print('Winter p-value =', p_Winter)

print('Since the p value is below 0.05 we reject the null hypothesis')
print('\n')

# Calculate confidence interval 
# Calculate and plot standard error of the mean
bs_Sp2011_Riders_reps = draw_bs_reps(Sp2011_Riders, np.mean, 10000)
bs_Sp2012_Riders_reps = draw_bs_reps(Sp2012_Riders, np.mean, 10000)

bs_Su2011_Riders_reps = draw_bs_reps(Su2011_Riders, np.mean, 10000)
bs_Su2012_Riders_reps = draw_bs_reps(Su2012_Riders, np.mean, 10000)

bs_Fa2011_Riders_reps = draw_bs_reps(Fa2011_Riders, np.mean, 10000)
bs_Fa2012_Riders_reps = draw_bs_reps(Fa2012_Riders, np.mean, 10000)

bs_Win2011_Riders_reps = draw_bs_reps(Win2011_Riders, np.mean, 10000)
bs_Win2012_Riders_reps = draw_bs_reps(Win2012_Riders, np.mean, 10000)


# Compute the 95% confidence intervals: conf_int
print('95% confidence intervals for rider usage in different seasons')
print('\n')

Sp2011_conf_int = np.percentile(Sp2011_Riders, [2.5, 97.5])
Sp2012_conf_int = np.percentile(Sp2012_Riders, [2.5, 97.5])

Su2011_conf_int = np.percentile(Su2011_Riders, [2.5, 97.5])
Su2012_conf_int = np.percentile(Su2012_Riders, [2.5, 97.5])

Fa2011_conf_int = np.percentile(Fa2011_Riders, [2.5, 97.5])
Fa2012_conf_int = np.percentile(Fa2012_Riders, [2.5, 97.5])

Win2011_conf_int = np.percentile(Win2011_Riders, [2.5, 97.5])
Win2012_conf_int = np.percentile(Win2012_Riders, [2.5, 97.5])

# Print the confidence interval
print('Values outside of the range: ', Sp2011_conf_int, ' are considered abnormal for daily use in Spring 2011')
print('Values outside of the range: ', Sp2012_conf_int, ' are considered abnormal for daily use in Spring 2012')
print('\n')
print('Values outside of the range: ', Su2011_conf_int, ' are considered abnormal for daily use in Summer 2011')
print('Values outside of the range: ', Su2012_conf_int, ' are considered abnormal for daily use in Summer 2012')
print('\n')
print('Values outside of the range: ', Fa2011_conf_int, ' are considered abnormal for daily use in Fall 2011')
print('Values outside of the range: ', Fa2012_conf_int, ' are considered abnormal for daily use in Fall 2012')
print('\n')
print('Values outside of the range: ', Win2011_conf_int, ' are considered abnormal for daily use in Winter 2011')
print('Values outside of the range: ', Win2012_conf_int, ' are considered abnormal for daily use in Winter 2012')



# Next see if there are meaningful differences in the **standard deviation** of casual versus registered riders
# The null hypothesis is that there is no meaningful difference in the standard deviation of casual vs registered riders.

# Begin with simply splitting it up annually

# Seasonal frames for registered users
Spring_reg = Spring_df['registered']
Summer_reg = Summer_df['registered']
Fall_reg   =   Fall_df['registered']
Winter_reg = Winter_df['registered']
                        

# 2011 frames for registered users
Spring11_reg = Spring2011_df['registered']
Summer11_reg = Summer2011_df['registered']
Fall11_reg   =   Fall2011_df['registered']
Winter11_reg = Winter2011_df['registered']

Reg_2011 = [Spring11_reg, Summer11_reg, Fall11_reg, Winter11_reg]

# 2012 frames for registered users
Spring12_reg = Spring2012_df['registered']
Summer12_reg = Summer2012_df['registered']
Fall12_reg   =   Fall2012_df['registered']
Winter12_reg = Winter2012_df['registered']

Reg_2012 = [Spring12_reg, Summer12_reg, Fall12_reg, Winter12_reg]



# Seasonal frames for casual users
Spring_cas = Spring_df['casual']
Summer_cas = Summer_df['casual']
Fall_cas   =   Fall_df['casual']
Winter_cas = Winter_df['casual']


# 2011 frames for casual users
Spring11_cas = Spring2011_df['casual']
Summer11_cas = Summer2011_df['casual']
Fall11_cas   =   Fall2011_df['casual']
Winter11_cas = Winter2011_df['casual']

Cas_2011 = [Spring11_cas, Summer11_cas, Fall11_cas, Winter11_cas]

# 2012 frames for casual users
Spring12_cas = Spring2012_df['casual']
Summer12_cas = Summer2012_df['casual']
Fall12_cas   =   Fall2012_df['casual']
Winter12_cas = Winter2012_df['casual']

Cas_2012 = [Spring12_cas, Summer12_cas, Fall12_cas, Winter12_cas]


# Casual and Registered users obviously have different means, but do they have
# different standard deviations.
# If they are different, which deviations is larger relative to the mean


# Means and standard deviations for registered riders
mu_SpReg = np.mean(Spring_reg)
sigma_SpReg = np.std(Spring_reg)
rel_sig_SpReg = (100*sigma_SpReg)/mu_SpReg

mu_SuReg = np.mean(Summer_reg)
sigma_SuReg = np.std(Summer_reg)
rel_sig_SuReg = (100*sigma_SuReg)/mu_SuReg

mu_FaReg = np.mean(Fall_reg)
sigma_FaReg = np.std(Fall_reg)
rel_sig_FaReg = (100*sigma_FaReg)/mu_FaReg

mu_WiReg = np.mean(Winter_reg)
sigma_WiReg = np.std(Winter_reg)
rel_sig_WiReg = (100*sigma_WiReg)/mu_WiReg


# 2012 means and standard deviations for all of riders
mu_SpCas = np.mean(Spring_cas)
sigma_SpCas = np.std(Spring_cas)
rel_sig_SpCas = (100*sigma_SpCas)/mu_SpCas

mu_SuCas = np.mean(Summer_cas)
sigma_SuCas = np.std(Summer_cas)
rel_sig_SuCas = (100*sigma_SuCas)/mu_SuCas

mu_FaCas = np.mean(Fall_cas)
sigma_FaCas = np.std(Fall_cas)
rel_sig_FaCas = (100*sigma_FaCas)/mu_FaCas

mu_WiCas = np.mean(Winter_cas)
sigma_WiCas = np.std(Winter_cas)
rel_sig_WiCas = (100*sigma_WiCas)/mu_WiCas

print('\n')
print('The relative standard deviation for registered users in Spring is: ', rel_sig_SpReg, '%')
print('The relative standard deviation for     casual users in Spring is: ', rel_sig_SpCas, '%')

print('\n')
print('The relative standard deviation for registered users in Summer is: ', rel_sig_SuReg, '%')
print('The relative standard deviation for     casual users in Summer is: ', rel_sig_SuCas, '%')

print('\n')
print('The relative standard deviation for registered users in Fall is: ', rel_sig_FaReg, '%')
print('The relative standard deviation for     casual users in Fall is: ', rel_sig_FaCas, '%')

print('\n')
print('The relative standard deviation for registered users in Winter is: ', rel_sig_WiReg, '%')
print('The relative standard deviation for     casual users in Winter is: ', rel_sig_WiCas, '%')

# Caluculate the difference in means between years
spring_mean_diff = mu_SpReg - mu_SpCas

summer_mean_diff = mu_SuReg - mu_SuCas

fall_mean_diff   = mu_FaReg - mu_FaCas

winter_mean_diff = mu_WiReg - mu_WiCas


# Concatenate the seasons: Season_concat
Spring_concat = np.concatenate((Spring_reg, Spring_cas))
Summer_concat = np.concatenate((Summer_reg, Summer_cas))
Fall_concat   = np.concatenate((Fall_reg, Fall_cas))
Winter_concat = np.concatenate((Winter_reg, Winter_cas))

# Initialize bootstrap replicates: Spring_replicates
Spring_replicates = np.empty(10000)
Summer_replicates = np.empty(10000)
Fall_replicates = np.empty(10000)
Winter_replicates = np.empty(10000)

for i in range(10000):
    # Generate bootstrap sample for Spring
    Spring_sample = np.random.choice(Spring_concat, size=len(Spring_concat))
    
    # Compute replicates for Spring
    Spring_replicates[i] = np.mean(Spring_sample[:len(Spring_reg)]) - np.mean(
                                     Spring_sample[len(Spring_cas):])


    # Generate bootstrap sample for Summer
    Summer_sample = np.random.choice(Summer_concat, size=len(Summer_concat))
    
    # Compute replicates for Summer
    Summer_replicates[i] = np.mean(Summer_sample[:len(Summer_reg)]) - np.mean(
                                     Summer_sample[len(Summer_cas):])


    # Generate bootstrap sample for Fall
    Fall_sample = np.random.choice(Fall_concat, size=len(Fall_concat))
    
    # Compute replicates for Fall
    Fall_replicates[i] = np.mean(Fall_sample[:len(Fall_reg)]) - np.mean(
                                     Fall_sample[len(Fall_cas):])


    # Generate bootstrap sample for Winter
    Winter_sample = np.random.choice(Winter_concat, size=len(Winter_concat))
    
    # Compute replicates for Winter
    Winter_replicates[i] = np.mean(Fall_sample[:len(Winter_reg)]) - np.mean(
                                     Fall_sample[len(Winter_cas):])


# Compute and print p-value: p
p_Spring = np.sum(Spring_replicates >= spring_mean_diff) / len(Spring_replicates)
p_Summer = np.sum(Summer_replicates >= summer_mean_diff) / len(Summer_replicates)
p_Fall   = np.sum(Fall_replicates   >= fall_mean_diff  ) / len(Fall_replicates)
p_Winter = np.sum(Winter_replicates >= winter_mean_diff) / len(Winter_replicates)
print('\n')
print('\n')
print('Now we are examining p values with regards to the null hypothesis that registered and casual users have similar usage count means.')
print('Spring p-value =', p_Spring)
print('Summer p-value =', p_Summer)
print('Fall p-value =',   p_Fall)
print('Winter p-value =', p_Winter)

print('Since the p value is below 0.05 we reject the null hypothesis')
print('\n')

# Calculate confidence interval 
# Draw reps for the mean of different types of user counts in different seasons
bs_Spring_reg_mean_reps = draw_bs_reps(Spring_reg, np.mean, 10000)
bs_Spring_cas_mean_reps = draw_bs_reps(Spring_cas, np.mean, 10000)

bs_Summer_reg_mean_reps = draw_bs_reps(Summer_reg, np.mean, 10000)
bs_Summer_cas_mean_reps = draw_bs_reps(Summer_cas, np.mean, 10000)

bs_Fall_reg_mean_reps   = draw_bs_reps(Fall_reg, np.mean, 10000)
bs_Fall_cas_mean_reps   = draw_bs_reps(Fall_cas, np.mean, 10000)

bs_Winter_reg_mean_reps = draw_bs_reps(Winter_reg, np.mean, 10000)
bs_Winter_cas_mean_reps = draw_bs_reps(Winter_cas, np.mean, 10000)


# Draw reps for the **variance** of different types of user counts in different seasons
bs_Spring_reg_var_reps = draw_bs_reps(Spring_reg, np.var, 10000)
bs_Spring_cas_var_reps = draw_bs_reps(Spring_cas, np.var, 10000)

bs_Summer_reg_var_reps = draw_bs_reps(Summer_reg, np.var, 10000)
bs_Summer_cas_var_reps = draw_bs_reps(Summer_cas, np.var, 10000)

bs_Fall_reg_var_reps   = draw_bs_reps(Fall_reg, np.var, 10000)
bs_Fall_cas_var_reps   = draw_bs_reps(Fall_cas, np.var, 10000)

bs_Winter_reg_var_reps = draw_bs_reps(Winter_reg, np.var, 10000)
bs_Winter_cas_var_reps = draw_bs_reps(Winter_cas, np.var, 10000)


# Make histograms of the results of the mean bootstrapping

#Make linspace for x and y axes
#x_lin = np.linspace(1000000, 10, 3000000)
#y_lin = np.linspace(0, 0.0000035, 8)

# Spring registered user mean histogram
print('\n')
print('Histograms for the results of the mean bootstrapping')
Sp_reg_mean_plot = plt.hist(bs_Spring_reg_mean_reps, bins = 50, color = 'lightcoral', normed = True)
Sp_reg_mean_plot = plt.xlabel('Mean of registered rider count in the Spring')
Sp_reg_mean_plot = plt.ylabel('PDF')
# Setting axes caused the code to run for a very long time.
# Sp_reg_var_plot = plt.xticks(x_lin)
# Sp_reg_var_plot = plt.yticks(y_lin)
plt.show()
plt.clf()

# Summer registered user mean histogram
Su_reg_mean_plot = plt.hist(bs_Summer_reg_mean_reps, bins = 50, color = 'darkorange', normed = True)
Su_reg_mean_plot = plt.xlabel('Mean of registered rider count in the Summer')
Su_reg_mean_plot = plt.ylabel('PDF')
plt.show()
plt.clf()

# Fall registered user mean histogram
Fa_reg_mean_plot = plt.hist(bs_Fall_reg_mean_reps, bins = 50, color = 'burlywood', normed = True)
Fa_reg_mean_plot = plt.xlabel('Mean of registered rider count in the Fall')
Fa_reg_mean_plot = plt.ylabel('PDF')
plt.show()
plt.clf()

# Winter registered user mean histogram
Wi_reg_mean_plot = plt.hist(bs_Winter_reg_mean_reps, bins = 50, color = 'skyblue', normed = True)
Wi_reg_mean_plot = plt.xlabel('Mean of registered rider count in the Winter')
Wi_reg_mean_plot = plt.ylabel('PDF')
plt.show()
plt.clf()

# Spring casual user mean histogram
Sp_cas_mean_plot = plt.hist(bs_Spring_cas_mean_reps, bins = 50, color = 'lightcoral', normed = True)
Sp_cas_mean_plot = plt.xlabel('Mean of casual rider count in the Spring')
Sp_cas_mean_plot = plt.ylabel('PDF')
plt.show()
plt.clf()

# Summer casual user mean histogram
Su_cas_mean_plot = plt.hist(bs_Summer_cas_mean_reps, bins = 50,color = 'darkorange', normed = True)
Su_cas_mean_plot = plt.xlabel('Mean of casual rider count in the Summer')
Su_cas_mean_plot = plt.ylabel('PDF')
plt.show()
plt.clf()

# Fall casual user mean histogram
Fa_cas_mean_plot = plt.hist(bs_Fall_cas_mean_reps, bins = 50, color = 'burlywood', normed = True)
Fa_cas_mean_plot = plt.xlabel('Mean of casual rider count in the Fall')
Fa_cas_mean_plot = plt.ylabel('PDF')
plt.show()
plt.clf()

# Winter casual user mean histogram
Wi_cas_mean_plot = plt.hist(bs_Winter_cas_mean_reps, bins = 50, color = 'skyblue', normed = True)
Wi_cas_mean_plot = plt.xlabel('Mean of casual rider count in the Winter')
Wi_cas_mean_plot = plt.ylabel('PDF')
plt.show()
plt.clf()




# Make histograms of the results of the variance bootstrapping
print('\n')
print('Histograms for the results of the variance bootstrapping')
#Make linspace for x and y axes
x_lin = np.linspace(1000000, 10, 3000000)
y_lin = np.linspace(0, 0.0000035, 8)
# Spring registered user variance histogram

Sp_reg_var_plot = plt.hist(bs_Spring_reg_var_reps, bins = 50, color = 'lightcoral', normed = True)
Sp_reg_var_plot = plt.xlabel('Variance of registered rider count in the Spring')
Sp_reg_var_plot = plt.ylabel('PDF')
# Setting axes caused the code to run for a very long time.
# Sp_reg_var_plot = plt.xticks(x_lin)
# Sp_reg_var_plot = plt.yticks(y_lin)
plt.show()
plt.clf()

# Summer registered user variance histogram
Su_reg_var_plot = plt.hist(bs_Summer_reg_var_reps, bins = 50, color = 'darkorange', normed = True)
Su_reg_var_plot = plt.xlabel('Variance of registered rider count in the Summer')
Su_reg_var_plot = plt.ylabel('PDF')
plt.show()
plt.clf()

# Fall registered user variance histogram
Fa_reg_var_plot = plt.hist(bs_Fall_reg_var_reps, bins = 50, color = 'burlywood', normed = True)
Fa_reg_var_plot = plt.xlabel('Variance of registered rider count in the Fall')
Fa_reg_var_plot = plt.ylabel('PDF')
plt.show()
plt.clf()

# Winter registered user variance histogram
Wi_reg_var_plot = plt.hist(bs_Winter_reg_var_reps, bins = 50, color = 'skyblue', normed = True)
Wi_reg_var_plot = plt.xlabel('Variance of registered rider count in the Winter')
Wi_reg_var_plot = plt.ylabel('PDF')
plt.show()
plt.clf()

# Spring casual user variance histogram
# Note that this histogram is positively distributed
# This is 
Sp_cas_var_plot = plt.hist(bs_Spring_cas_var_reps, bins = 50, color = 'lightcoral', normed = True)
Sp_cas_var_plot = plt.xlabel('Variance of casual rider count in the Spring')
Sp_cas_var_plot = plt.ylabel('PDF')
plt.show()
plt.clf()

# Summer casual user variance histogram
Su_cas_var_plot = plt.hist(bs_Summer_cas_var_reps, bins = 50,color = 'darkorange', normed = True)
Su_cas_var_plot = plt.xlabel('Variance of casual rider count in the Summer')
Su_cas_var_plot = plt.ylabel('PDF')
plt.show()
plt.clf()

# Fall casual user variance histogram
Fa_cas_var_plot = plt.hist(bs_Fall_cas_var_reps, bins = 50, color = 'burlywood', normed = True)
Fa_cas_var_plot = plt.xlabel('Variance of casual rider count in the Fall')
Fa_cas_var_plot = plt.ylabel('PDF')
plt.show()
plt.clf()

# Winter casual user variance histogram
Wi_cas_var_plot = plt.hist(bs_Winter_cas_var_reps, bins = 50, color = 'skyblue', normed = True)
Wi_cas_var_plot = plt.xlabel('Variance of casual rider count in the Winter')
Wi_cas_var_plot = plt.ylabel('PDF')
plt.show()
plt.clf()


# Compute the 95% confidence intervals: conf_int
Spring_reg_conf_int = np.percentile(Spring_reg, [2.5, 97.5])
Spring_cas_conf_int = np.percentile(Spring_cas, [2.5, 97.5])

Summer_reg_conf_int = np.percentile(Summer_reg, [2.5, 97.5])
Summer_cas_conf_int = np.percentile(Summer_cas, [2.5, 97.5])

Fall_reg_conf_int   = np.percentile(Fall_reg, [2.5, 97.5])
Fall_cas_conf_int   = np.percentile(Fall_cas, [2.5, 97.5])

Winter_reg_conf_int = np.percentile(Winter_reg, [2.5, 97.5])
Winter_cas_conf_int = np.percentile(Winter_cas, [2.5, 97.5])

# Print the confidence interval
print('\n')
print('95% confidence intervals for casual vs registered rider usage in different seasons')
print('\n')
print('Values outside of the range: ', Spring_reg_conf_int, ' are considered abnormal for daily registered rider use in Spring')
print('Values outside of the range: ', Spring_cas_conf_int, ' are considered abnormal for daily casual rider use in Spring')
print('\n')
print('Values outside of the range: ', Summer_reg_conf_int, ' are considered abnormal for daily registered rider use in Summer')
print('Values outside of the range: ', Summer_cas_conf_int, ' are considered abnormal for daily casual rider use in Summer')
print('\n')
print('Values outside of the range: ', Fall_reg_conf_int,   ' are considered abnormal for daily registered rider use in Fall')
print('Values outside of the range: ', Fall_cas_conf_int,   ' are considered abnormal for daily casual rider use in Fall')
print('\n')
print('Values outside of the range: ', Winter_reg_conf_int, ' are considered abnormal for daily registered rider use in Winter')
print('Values outside of the range: ', Winter_cas_conf_int, ' are considered abnormal for daily casual rider use in Winter')
print('\n')


