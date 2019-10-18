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

from data_utils import get_data, split_data
from utils import DataFrameSelector, CombinedAttributesAdder, CustomLabelBinarizer

# laoding the dataset
housing = get_data()

# split into train_set and test_set
train_set, test_set = split_data(housing)

housing = train_set.drop("median_house_value", axis=1)  # median_house_value column contains the target values
housing_data_labels = train_set["median_house_value"].copy()

# data preparation and prediction going to be done in a pipeline

# pipeline to preprocess numerical features
numerical_attributes = train_set.drop(['ocean_proximity', 'median_house_value'], axis=1).columns

numerical_pipeline = Pipeline([
	('selector', DataFrameSelector(numerical_attributes)),  # selects only numerical attributes
	('imputer', SimpleImputer(strategy='median')),  # replace missing values using the median along each column
	('attrs_addeder', CombinedAttributesAdder()),
	('scaler', StandardScaler())])

# pipeline to preprocess categorical features
categorical_attribute = ["ocean_proximity"]

categorical_pipeline = Pipeline([
	('selector', DataFrameSelector(categorical_attribute)),
	('label_binarizer', CustomLabelBinarizer())])

# get the labels of the dataset
housing_labels = ["median_house_value"]
label_pipeline = Pipeline([
	('labels', DataFrameSelector(housing_labels))])

# FeatureUnion to combine the 2 pipelines above in parallel
full_dataprep_pipeline = FeatureUnion(transformer_list=[
	('numerical_pipeline', numerical_pipeline),
	('categorical_pipeline', categorical_pipeline)])

# adding the training model, SVR from sklearn.svm
prepared_data = full_dataprep_pipeline.fit_transform(housing)

param_grid = {
	'kernel': ['linear', 'rbf'],
	'C': [1, 2, 3],
	'gamma': [1, 2, 3]
}

svm = SVR()


randomized_search = RandomizedSearchCV(svm, param_grid, cv=3, scoring='neg_mean_squared_error')

print("started training>>>>>>>>>>")
start = time.time()
randomized_search.fit(prepared_data, housing_data_labels)
end = time.time()
print("time taken: ", end - start)