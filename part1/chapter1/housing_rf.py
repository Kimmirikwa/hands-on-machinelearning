import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

from utils import DataFrameSelector, CombinedAttributesAdder, CustomLabelBinarizer
from data_utils import get_data, split_data	

# loading the data
housing = get_data()

# we need to split the dataset into training and testing set to prevent the algorithm
# overfitting. Random sampling works fine for datasets with many examples
# for those with few examples, there is a danger of introducing sampling bias
# we are going to do stratified sampling based on the value of the median income since median income is
# a very important feature and we would like our train test split to be a good representation of
# of the original data based based on the income
train_set, test_set = split_data(housing)

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

# we are going to use pipelines to data data cleaning and feature extraction

# numerical values
# we will use sklearn's Imputer class instance to fill missing  values with the median
imputer = Imputer(strategy='median')

num_attribs = housing.drop("ocean_proximity", axis=1).columns

num_pipeline = Pipeline([
	('selector', DataFrameSelector(num_attribs)),  # select numerical attributes
	('imputer', Imputer(strategy='median')),  # fill missing values with mean
	('attribs_adder', CombinedAttributesAdder()),  # add more attributes
	('scaler', StandardScaler())])  # scale the features using feature scaling

# categorical values
cat_attribs = ["ocean_proximity"]

cat_pipeline = Pipeline([
	('selector', DataFrameSelector(cat_attribs)),
	('label_binarizer', CustomLabelBinarizer())])  # transforms from text categories to integer categories, then from integer categories to one-hot vectors

# we then use FeatureUnion to run the 2 pipeliens above in parallel and the concatenate their results
full_pipeline = FeatureUnion(transformer_list=[
	('num_pipeline', num_pipeline),
	('cat_pipeline', cat_pipeline)])

housing_prepared = full_pipeline.fit_transform(housing)

# we are going to use RandomForestRegressor model

# hyperparameter to be tried
param_grid = [
	{'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
	{'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

forest_reg = RandomForestRegressor()

# grid search to select the best params
# all the possible param combinations will be tried and the best combination will be selected
# from the hyperparameter settings, forest_reg will be trained ((3 * 4) + (2 * 3)) * 5 = 90 times!
# to avoid running all the combinations, RandomizedGridSearch can be used
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')  # will through all the params and select the best params

grid_search.fit(housing_prepared, housing_labels)  # train the model using data


# evaluating the model
# the best model selected above will be evaluated using the test set
final_model = grid_search.best_estimator_  # the best model i.e with best hyperparameters

# the test dataset
X_test = test_set.drop("median_house_value", axis=1)
y_test = test_set["median_house_value"].copy()

# we only transform the data, we do not fit it!
X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_rmse)  # about 48166.81563915982
