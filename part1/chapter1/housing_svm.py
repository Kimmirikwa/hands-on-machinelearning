'''prediction of the housing median prices using support vector machine'''
import pandas as pd
import numpy as np

from data_utils import get_data, split_data

# laoding the dataset
housing = get_data()

# split into train_set and test_set
train_set, test_set = split_data(housing)
