import numpy as np

np.random.seed(2042)  # for reproducibilty

def custom_test_train_split(X, y, test_ratio=0.2, validation_ratio=0.2):
	total_size = len(X)
	test_size = int(test_ratio * total_size)
	validation_size = int(validation_ratio * total_size)
	training_size = total_size - test_size - validation_size

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

	return X_train, X_test, X_test, y_test, X_validation, y_validation
