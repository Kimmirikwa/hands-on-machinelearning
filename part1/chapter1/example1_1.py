# this involves reading the data, preparing it, visualizing, training a model and using it to 
# make a prediction

# the necessary imports
import matplotlib  # for plotting 2D figures
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

# load the data. The data will be held in pandas dataframes
# for more information on pandas dataframes: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
oecd_bli = pd.read_csv('data/oecd_bli_2015.csv', thousands=',')
gdp_per_capita = pd.read_csv('data/gdp_per_capita.csv', thousands=',', delimiter='\t',
	encoding='latin1', na_values='n/a')