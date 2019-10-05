from sklearn.datasets import fetch_openml  # will be used to fetch data
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# get the data
mnist = fetch_openml("mnist_784")  # fetch MNIST data using the dateset id

# X is a 70000 by 784 = (28 * 28) matrix, which represents the number of images and number of pixes per image respectively
# y is a vector of size 70000, each value in the vector representing the digit for each corresponding image pixels in X
X, y = mnist['data'], mnist['target']  # X is the training features and y is the target value i.e the digit

# extracting the training and testing datasets
# this data has already been split to training and testing datasets. the first 60000 represent the training data and the remaining
# 10000 is the testing data for both X and y
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# we shuffle the training data to ensure
# 1. all cross-validation folds to be similar
# 2. to reduce sensitivity of some algorithms to the order of the data
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

param_grid = [{'weights': ["uniform", "distance"], 'n_neighbors': [3, 4, 5]}]  # this will lead to 2 * 3 = 6 models

classifier = KNeighborsClassifier()
# will run the models in parallel
# with cv=5, wel will have a total of 6 * 5 = 30 models
grid_search = GridSearchCV(classifier, param_grid, cv=5, verbose=3, n_jobs=-1) 
grid_search.fit(X_train, y_train)
y_pred = grid_search.predict(X_test)
print(accuracy_score(y_test, y_pred))
