# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 01:52:42 2018

Adaptation of William Koehrsen's code from his article:
    
"Hyperparameter Tuning the Random Forest in Python: Improving the Random Forest Part Two"


@author: Jon
"""

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state = 42)

from pprint import pprint

