import os
import tarfile
from six.moves import urllib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelBinarizer

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
# we will only work on a copy the train set to avoid tampering with the data
housing = train_set.copy()

# 1. geographical data i.e latitude and longitude
# will plot a scatter plot with alpha = 0.4 to enable us see high density areas
# the scatter plot shows a high correlation between house prices and the location and pupulation density
housing.plot(kind='scatter', x='latitude', y='longitude', alpha=0.4, s=housing['population']/100,
	label='population', c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)
plt.show()

# the correlation of the features is calculated by .corr() method
# the correlation values range between 1 and -1, with a positive value indication positive correlation
# and negetive value indication negative correlation of particular magnitude. For example, a value of +1
# shows that the features increase with same proportion. A value of 0 shows that there is no correlation between
# the features. However, correlation does not capture non-linear relationships
corr_matrix = housing.corr()

# panda's scatter_matrix can be used to show the correlation values graphically
# due to space, we are going to show the the scatter matrix of a few features, since with 11
# features we will have 121 plots
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
plt.show()

# prepare data for ML algorithms
# we are going to work with training features only
housing = train_set.drop("median_house_value", axis=1)
housing_labels = train_set["median_house_value"].copy()

# 1. Data cleaning
# missing values is common in data. This needs to be handled well as most ML algorithms do not work with missing values
# we can drop the examples(rows) or the features(columns) or fill the missing data points with some value eg mean, median, mode

# 1a. numerical values
# we will use sklearn's Imputer class instance to fill missing  values with the median
imputer = Imputer(strategy='median')

# all training features except ocean_proximity are numerical
housing_num = housing.drop('ocean_proximity', axis=1)
X = imputer.fit_transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns)  # the transformed dataset, there is no numerical with a missing value

# 1b. handling text and categorical features
ocean_proximity_dummies = pd.get_dummies(housing['ocean_proximity'])

# combine the 2 dataframes
housing_tr = pd.concat([housing_tr, ocean_proximity_dummies])
