import os
import tarfile
from six.moves import urllib
import pandas as pd

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

# function to load the data in housing.csv file
def load_housing_data(housing_path=HOUSING_PATH):
	csv_path = os.path.join(housing_path, "housing.csv")
	return pd.read_csv(csv_path)


def get_data():
	# fetching data
	fetch_housing_data()
	# loading the data
	return load_housing_data()