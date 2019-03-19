'''
	prediction of the housing median prices using support vector machine
	This is the solution to the exercise of the second chapter
'''
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR

from data_utils import get_data, split_data
from utils import DataFrameSelector, CombinedAttributesAdder, CustomLabelBinarizer

# laoding the dataset
housing = get_data()

# split into train_set and test_set
train_set, test_set = split_data(housing)

housing = train_set.drop("median_house_value", axis=1)
housing_data_labels = train_set["median_house_value"].copy()

# data preparation and prediction going to be done in a pipeline

# pipeline to preprocess numerical features
numerical_attributes = train_set.drop(['ocean_proximity', 'median_house_value'], axis=1).columns

numerical_pipeline = Pipeline([
	('selector', DataFrameSelector(numerical_attributes)),
	('imputer', Imputer(strategy='median')),
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
	'C': [1.0, 2.0, 3.0, 4.0, 5.0],
	'gamma': [1.0, 2.0, 3.0, 4.0, 5.0]
}

svm = SVR()

randomized_search = RandomizedSearchCV(svm, param_grid, cv=5, scoring='neg_mean_squared_error')

randomized_search.fit(prepared_data, housing_data_labels)
