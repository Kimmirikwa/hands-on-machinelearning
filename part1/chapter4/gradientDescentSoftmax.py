from sklearn import datasets
import numpy as np

from utils import custom_test_train_split, to_one_hot

iris = datasets.load_iris()

# get the training data
X = iris['data'][:, (2, 3)]  # petal length and petal width
y = iris['target']

# add the bias to the features i.e a columns on ones
X_with_bias = np.c_[np.ones([len(X), 1]), X]

X_train, X_test, X_validation, y_train, y_test, y_validation = custom_test_train_split(X, y)

# convert the training labels to one-hot-encoded
y_train_one_hot = to_one_hot(y_train)
y_test_one_hot = to_one_hot(y_test)
y_validation_one_hot = to_one_hot(y_validation)
