import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelBinarizer

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
	'''transformer class that adds the combined CombinedAttributesAdder'''
	def __init__(self, add_bedrooms_per_room=True):
		self.add_bedrooms_per_room = add_bedrooms_per_room

	def fit(self, X, y=None):
		return self

	def transform(self, X, y=None):
		rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
		population_per_household = X[:, population_ix] / X[:, household_ix]
		if self.add_bedrooms_per_room:
			bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
			return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
		return np.c_[X, rooms_per_household, population_per_household]


class DataFrameSelector(BaseEstimator, TransformerMixin):
	'''transforms a dataframe by selecting the specified attributes'''
	def __init__(self, attribute_names):
		self.attribute_names = attribute_names

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		return X[self.attribute_names].values

class CustomLabelBinarizer(BaseEstimator, TransformerMixin):
	'''LabelBinarizer that can handle 3 positional arguments'''
	def __init__(self):
		self.encoder = LabelBinarizer()

	def fit(self, X, y=None):
		self.encoder.fit(X)
		return self

	def transform(self, X, y=None):
		return self.encoder.transform(X)
