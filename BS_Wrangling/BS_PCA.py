# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 14:11:24 2018

This file contains the PCA analysis for the BsDs (Bike sharing dataset)

@author: Jon
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA

# Import the Data

#Change the working directory
os.chdir('C:/Users/Jon/Documents/Springboard/Capstone1/Bike-Sharing-Dataset')

# Read in the csv's for day.csv and hour.csv
day_df = pd.read_csv('day.csv')
hour_df = pd.read_csv('hour.csv')

# Make an array for each table
day_array = np.array(day_df.values)
hour_array = np.array(hour_df.values)

# Try some PCA
No_Date_day_df = day_df.drop('dteday', axis = 1)
N = No_Date_day_df
N_vals = N.values

pca = PCA(n_components = 15)
pca.fit(N_vals)
var = pca.explained_variance_ratio_
print(var)
print(pca.singular_values_)
# 99.6 % of the variance can be explained in the first two principal components

#Cumulative Variance explains
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
print(var1)
plt.plot(var1)