'''
	prediction of the housing median prices using support vector machine
	This is the solution to the exercise of the second chapter
'''
import time
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from scipy.stats import uniform

from data_utils import get_data, split_data
from utils import DataFrameSelector, CombinedAttributesAdder, CustomLabelBinarizer

# laoding the dataset
housing = get_data()

# split into train_set and test_set
train_set, test_set = split_data(housing)

housing_train = train_set.drop("median_house_value", axis=1)  # median_house_value column contains the target values
housing_test = test_set.drop("median_house_value", axis=1)
housing_train_labels = train_set["median_house_value"].copy()
housing_test_labels = test_set["median_house_value"].copy()

# data preparation and prediction going to be done in a pipeline

# pipeline to preprocess numerical features
numerical_attributes = train_set.drop(['ocean_proximity', 'median_house_value'], axis=1).columns

numerical_pipeline = Pipeline([
	('selector', DataFrameSelector(numerical_attributes)),  # selects only numerical attributes
	('imputer', SimpleImputer(strategy='median')),  # replace missing values using the median along each column
	('attrs_addeder', CombinedAttributesAdder()),  # feature creation
	('scaler', StandardScaler())])  # scaling the data, if not the model will concetrate on proportionately larger features

# pipeline to preprocess categorical features
categorical_attribute = ["ocean_proximity"]

categorical_pipeline = Pipeline([
	('selector', DataFrameSelector(categorical_attribute)),  # selects only categorical feature i.e "ocean proximity"
	('label_binarizer', CustomLabelBinarizer())])  # converts categorical feature to numerical

# get the labels of the dataset
housing_labels = ["median_house_value"]
label_pipeline = Pipeline([
	('labels', DataFrameSelector(housing_labels))])

# FeatureUnion to combine the 2 pipelines above in parallel
full_dataprep_pipeline = FeatureUnion(transformer_list=[
	('numerical_pipeline', numerical_pipeline),
	('categorical_pipeline', categorical_pipeline)])

# adding the training model, SVR from sklearn.svm
prepared_train_data = full_dataprep_pipeline.fit_transform(housing_train)

param_distributions = {
	'kernel': ['linear', 'rbf'],
	'C': uniform(1, 10),
	'gamma': uniform(1, 10)
}

svm = SVR()


randomized_search = RandomizedSearchCV(
	svm, param_distributions, cv=3, scoring='neg_mean_squared_error', n_iter=20, verbose=True, n_jobs=-1)

start = time.time()
randomized_search.fit(prepared_train_data, housing_train_labels)
end = time.time()
print("time taken: ", end - start)

best_params = randomized_search.best_params_
print("The best params: ", best_params)
best_estimator = randomized_search.best_estimator_

# preparing test data before feeding to the trained model
prepared_test_data = full_dataprep_pipeline.fit_transform(housing_test)

housing_predicted_labels = best_estimator.predict(prepared_test_data)
mse = mean_squared_error(housing_test_labels, housing_predicted_labels)
print("root mean squared error: ", np.sqrt(mse))