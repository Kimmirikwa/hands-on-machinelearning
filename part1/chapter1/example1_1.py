# this involves reading the data, preparing it, visualizing, training a model and using it to 
# make a prediction

# the necessary imports
import matplotlib  # for plotting 2D figures
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

def prepare_country_stats(oecd_bli, gdp_per_capita):
	'''merges the 2 dataframes'''
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]

# load the data. The data will be held in pandas dataframes
# for more information on pandas dataframes: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
oecd_bli = pd.read_csv('data/oecd_bli_2015.csv', thousands=',')
gdp_per_capita = pd.read_csv('data/gdp_per_capita.csv', thousands=',', delimiter='\t',
	encoding='latin1', na_values='n/a')

# merge the 2 dataframes for each row to have data of a particular country
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)

X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]
