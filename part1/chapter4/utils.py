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

def softmax(logits):
	exp = np.exp(logits)
	exp_sums = np.sum(exp, axis=1, keepdims=True)
	return exp / exp_sums

def batch_gradient_descent(X_train, y_train, eta=0.01, n_iterations=5000, epsilon=1e-7):
	# the size of the parameters
	n_inputs = X_train.shape[1]  # the number of the features plus bias term
	n_outputs = len(np.unique(y_train))  # the number of the classes

	Theta = np.random.randn(n_inputs, n_outputs)  # the initial theta

	y_train_one_hot = to_one_hot(y_train)

	for iteration in range(n_iterations + 1):
		logits = X_train.dot(Theta)  # dot product of X_trainand Theta. m x n . n x unique_labels = m x unique_labels
		y_proba = softmax(logits)  # m x unique_labels probabilities
		loss = -np.mean(np.sum(y_train_one_hot * np.log(y_proba + epsilon), axis=1))
		error = y_proba - y_train_one_hot
		if (iteration % 500 == 0):
			print(iteration, loss)
		gradients = 1 / len(X_train) * X_train.T.dot(error)
		Theta = Theta - eta * gradients

	return Theta
