import re
import urlextract
import nltk  # natural language toolkit
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin
from html import unescape
import numpy as np
from scipy.sparse import csr_matrix

def html_to_plain_text(html):
	# will remove some tags and replace others
	text = re.sub('<head.*?>.*?</head>', '', html, flags=re.M | re.S | re.I)
	text = re.sub('<a\s.*?>', ' HYPERLINK ', text, flags=re.M | re.S | re.I)
	text = re.sub('<.*?>', '', text, flags=re.M | re.S)
	text = re.sub(r'(\s*\n)+', '\n', text, flags=re.M | re.S)
	return unescape(text)

def email_to_text(email):
	html = None
	for part in email.walk():
		content_type = part.get_content_type()
		if not content_type in ("text/plain", "text/html"):
			continue
		try:
			content = part.get_content()
		except:
			content = str(part.get_payload())
		if content_type == "text/plain":
			return content
		if content:
			return html_to_plain_text(content)


# a transformer that converts the words in the emails to count of words
class EmailToWordCounterTransformer(BaseEstimator, TransformerMixin):
	def __init__(self, strip_headers=True, lower_case=True, remove_punctuation=True, replace_urls=True,
		replace_numbers=True, stemming=True): # will set the hyperparameters for this transformer
		self.strip_headers = strip_headers
		self.lower_case = lower_case
		self.remove_punctuation = remove_punctuation
		self.replace_urls = replace_urls
		self.replace_numbers = replace_numbers
		self.stemming = stemming

	def fit(self, X, y=None):
		return self

	def transform(self, X, y=None):
		X_transformed = []  # will contain the counts of words in emails
		for email in X:
			text = email_to_text(email) or ""
			if self.lower_case:
				text = text.lower()
			if self.replace_urls:
				url_extractor = urlextract.URLExtract()
				urls = list(set(url_extractor.find_urls(text)))
				urls.sort(key=lambda url: len(url), reverse=True)
				for url in urls:
				    text = text.replace(url, " URL ")
			if self.replace_numbers:
			    text = re.sub(r'\d+(?:\.\d*(?:[eE]\d+))?', 'NUMBER', text)
			if self.remove_punctuation:
			    text = re.sub(r'\W+', ' ', text, flags=re.M)
			word_counts = Counter(text.split())
			if self.stemming:
				stemmer = nltk.PorterStemmer()
				stemmed_word_counts = Counter()
				for word, count in word_counts.items():
				    stemmed_word = stemmer.stem(word)
				    stemmed_word_counts[stemmed_word] += count
				word_counts = stemmed_word_counts
			X_transformed.append(word_counts)

		return np.array(X_transformed)

class WordCounterToVectorTransformer(BaseEstimator, TransformerMixin):
	def __init__(self, vocabulary_size=1000):
		self.vocabulary_size = vocabulary_size

	def fit(self, X, y=None):
		# we get the words that are most common to be used as the vocabulary
		total_count = Counter()
		for word_count in X:
		    for word, count in word_count.items():
		        total_count[word] += min(count, 10)
		most_common = total_count.most_common()[:self.vocabulary_size]  # the top 'vocabulary_size' words
		self.most_common_ = most_common
		self.vocabulary_ = {word: index + 1 for index, (word, count) in enumerate(most_common)}
		return self

	def transform(self):
		# we transfors the word counts to be a sparce matrix of most common words as columns
		# and the instances as rows
		rows = []
		cols = []
		data = []
		for row, word_count in enumerate(X):
		    for word, count in word_count.items():
		        rows.append(row)
		        cols.append(self.vocabulary_.get(word, 0))
		        data.append(count)
		return csr_matrix((data, (rows, cols)), shape=(len(X), self.vocabulary_size + 1))