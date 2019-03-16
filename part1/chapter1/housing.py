import os
import tarfile
from six.moves import urllib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"

# function to fetch data
# fetching data this way is very impoortant especially if the data changes so much
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
	'''
		Tries to create the directory if not present
		reads the file and extracts it
	'''
	if not os.path.isdir(housing_path):  # create the directory if it not yet created
		os.makedirs(housing_path)
	tgz_path = os.path.join(housing_path, "housing.tgz")
	urllib.request.urlretrieve(housing_url, tgz_path)  # downloading housing.tgz file
	housing_tgz = tarfile.open(tgz_path)
	housing_tgz.extractall(path=housing_path)  # extract housing.csv file
	housing_tgz.close()

# fetching data
fetch_housing_data()

# function to load the data in housing.csv file
def load_housing_data(housing_path=HOUSING_PATH):
	csv_path = os.path.join(housing_path, "housing.csv")
	return pd.read_csv(csv_path)

# loading the data
housing = load_housing_data()

# we need to split the dataset into training and testing set to prevent the algorithm
# overfitting. Random sampling works fine for datasets with many examples
# for those with few examples, there is a danger of introducing sampling bias
# we are going to do stratified sampling based on the value of the median income since median income is
# a very important feature and we would like our train test split to be a good representation of
# of the original data based based on the income
housing['income_cat'] = np.ceil(housing['median_income'] / 1.5)  # we will have fewer categories of median income
housing['income_cat'].where(housing['income_cat'] < 5, 5.0, inplace=True)  # assign all categories from 5 and above to category 5

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):
	train_set = housing.loc[train_index]
	test_set = housing.loc[test_index]

# we can now drop the the income_cat columns as it no longer has any use in the data
for set in (train_set, test_set):
	set.drop(['income_cat'], axis=1, inplace=True)

# data exploration

