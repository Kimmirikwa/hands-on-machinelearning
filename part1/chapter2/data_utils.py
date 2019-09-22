import os
import tarfile
from six.moves import urllib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"

# function to fetch data
# fetching data this way is very impoortant especially if the data changes so much
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
	'''
		Tries to create datasets/housing directory if it does not exist
		downloads the housing file, this ensures we get an updates on the data
		reads the file and extracts it
	'''
	if not os.path.isdir(housing_path):  # create the directory if it not yet created
		os.makedirs(housing_path)
	tgz_path = os.path.join(housing_path, "housing.tgz")
	urllib.request.urlretrieve(housing_url, tgz_path)  # downloading housing.tgz file
	housing_tgz = tarfile.open(tgz_path)
	housing_tgz.extractall(path=housing_path)  # extract housing.csv file
	housing_tgz.close()

# function to load the data in housing.csv file
def load_housing_data(housing_path=HOUSING_PATH):
	csv_path = os.path.join(housing_path, "housing.csv")
	return pd.read_csv(csv_path)  # returns a dataframe


def get_data():
	# fetching data
	fetch_housing_data()
	# loading the data
	return load_housing_data()

def split_data(housing):
	# use income_cat to do stratified sampling
	# we will have five income_cat 1.0 to 5.0
	housing['income_cat'] = np.ceil(housing['median_income'] / 1.5)  # we will have fewer categories of median income
	housing['income_cat'].where(housing['income_cat'] < 5, 5.0, inplace=True)  # assign all categories from 5 and above to category 5

	split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
	for train_index, test_index in split.split(housing, housing['income_cat']):
		train_set = housing.loc[train_index]
		test_set = housing.loc[test_index]

	# we can now drop the the income_cat columns as it no longer has any use in the data
	for set in (train_set, test_set):
		set.drop(['income_cat'], axis=1, inplace=True)

	return train_set, test_set