'''
	prediction of the housing median prices using support vector machine
	This is the solution to the exercise of the second chapter
'''
import pandas as pd
import numpy as np

from data_utils import get_data, split_data

# laoding the dataset
housing = get_data()

# split into train_set and test_set
train_set, test_set = split_data(housing)

# get feature vectors and label vector
housing = train_set.drop("median_house_value", axis=1)
housing_labels = train_set["median_house_value"].copy()
