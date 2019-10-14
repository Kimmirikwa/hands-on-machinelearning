from sklearn import datasets
import numpy as np

from utils import custom_test_train_split, to_one_hot, batch_gradient_descent, softmax

iris = datasets.load_iris()

# get the training data
X = iris['data'][:, (2, 3)]  # petal length and petal width
y = iris['target']

# add the bias to the features i.e a columns on ones
X_with_bias = np.c_[np.ones([len(X), 1]), X]

X_train, X_test, X_validation, y_train, y_test, y_validation = custom_test_train_split(X_with_bias, y)

# convert the training labels to one-hot-encoded
y_train_one_hot = to_one_hot(y_train)
y_test_one_hot = to_one_hot(y_test)
y_validation_one_hot = to_one_hot(y_validation)

def test_sgd(regularization=False):
	theta = batch_gradient_descent(X_train, y_train, regularization=regularization)
	logits = X_validation.dot(theta)
	y_proba = softmax(logits)
	y_predict = np.argmax(y_proba, axis=1)
	accuracy_score = np.mean(y_predict == y_validation)
	print(accuracy_score)

# without regularization
test_sgd()  # 0.93333

# with regression
test_sgd(regularization=True)

