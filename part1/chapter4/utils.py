import numpy as np

np.random.seed(2042)  # for reproducibilty

def custom_test_train_split(X, y, test_ratio=0.2, validation_ratio=0.2):
	'''
		splits data into training, testing and validation set based on the ratios given
	'''
	total_size = len(X)
	test_size = int(test_ratio * total_size)
	validation_size = int(validation_ratio * total_size)
	training_size = total_size - test_size - validation_size

	# randomozing indices help deal with learning models that are affected by the order
	# the data
	randomized_indices = np.random.permutation(total_size)

	train_indices = randomized_indices[:training_size]
	test_indices = randomized_indices[training_size:-test_size]
	validation_indices = randomized_indices[-test_size:]

	X_train = X[train_indices]
	y_train = y[train_indices]
	X_test = X[test_indices]
	y_test = y[test_indices]
	X_validation = X[validation_indices]
	y_validation = y[validation_indices]

	return X_train, X_test, X_validation, y_train, y_test, y_validation

def to_one_hot(y):
	'''
		takes in y, which is a column of labes and converts it to a matrix of labels,
		with each column corresponding to a particular label. The column will have a 1 if 
		the value of the label is a that of the column, else the value will be 0
	'''
	n_classes = y.max() + 1  # assumes the labels are zero-indexed
	m = len(y)
	Y_one_hot = np.zeros((m, n_classes))
	Y_one_hot[np.arange(m), y] = 1  # each row will have one column with a value 1
	return Y_one_hot
