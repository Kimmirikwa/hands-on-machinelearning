from sklearn.base import BaseEstimator, TransformerMixin

class EmailToWordCounterTransformer(BaseEstimator, TransformerMixin):
	def __init__(self, strip_headers=True, lower_case=True, remove_punctuation=True, replace_urls=True,
		replace_numbers=True, stemming=True):
		self.strip_headers = strip_headers
		self.lower_case = lower_case
		self.remove_punctuation = remove_punctuation
		self.replace_urls = replace_urls
		self.replace_numbers = replace_numbers
		self.stemming = stemming

	def fit(self, X, y=None):
		pass

	def transform(self):
		pass