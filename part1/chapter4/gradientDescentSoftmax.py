from sklearn import datasets
import numpy as np

iris = datasets.load_iris()

# get the training data
X = iris['data'][:, (2, 3)]  # petal length and petal width
y = iris['target']

# add the bias to the features i.e a columns on ones
X_with_bias = np.c_[np.ones([len(X), 1]), X]