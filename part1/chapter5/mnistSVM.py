from sklearn.datasets import fetch_openml
import numpy as np

mnist = fetch_openml('mnist_784')

X = mnist['data']
y = mnist['target']

# the first 60000 examples can be taken as the training data and 
# the remaining 10000 as the test set
X_train = X[:60000]
X_test = X[60000:]
y_train = y[:60000]
y_test = y[60000:]

# permutate the data to avoid effect of algorithms that are affected by the order of the data
randomized_indices = np.random.permutation(60000)
X_train = X_train[randomized_indices]
y_train = y_train[randomized_indices]