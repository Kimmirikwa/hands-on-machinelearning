'''
	prediction of the housing median prices using support vector machine
	This is the solution to the exercise of the second chapter
'''
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, Imputer

from data_utils import get_data, split_data
from utils import DataFrameSelector, CombinedAttributesAdder, CustomLabelBinarizer

# laoding the dataset
housing = get_data()

# split into train_set and test_set
train_set, test_set = split_data(housing)

# get feature vectors and label vector
housing = train_set.drop("median_house_value", axis=1)
housing_labels = train_set["median_house_value"].copy()

# data preparation and prediction going to be done in a pipeline

# pipeline to preprocess numerical features
numerical_attributes = housing.drop('ocean_proximity', axis=1).columns

numerical_pipeline = Pipeline([
	('selector', DataFrameSelector(numerical_attributes)),
	('imputer', Imputer(strategy='median')),
	('attrs_addeder', CombinedAttributesAdder()),
	('scaler', StandardScaler())])

# pipine to preprocess categorical features
categorical_attribute = ["ocean_proximity"]

categorical_pipeline = Pipeline([
	('selector', DataFrameSelector(categorical_attribute)),
	('label_binarizer', CustomLabelBinarizer())])

# FeatureUnion to combine the 2 pipelines above in parallel
full_dataprep_pipeline = FeatureUnion(trandformer_list=[
	('numerical_pipeline', numerical_pipeline),
	('categorical_pipeline', categorical_pipeline)])