import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from scipy.stats import mode

X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier()

param_gid = {
	'max_leaf_nodes': list(range(2, 100)), 
	'min_samples_split': [2, 3, 4]
}

grid_search_cv = GridSearchCV(clf, param_gid, cv=3, verbose=True)
grid_search_cv.fit(X_train, y_train)

best_estimator = grid_search_cv.best_estimator_
best_estimator.fit(X_train, y_train)

y_pred = best_estimator.predict(X_test)
print("the accuracy score: ", accuracy_score(y_test, y_pred))  # 0.8695

# constructing a forest of the best estimator trees
n_trees = 1000
n_instances = 100

mini_sets = []  # will hold the mini_training and minitesting datasets to be used by the trees

# we shuffle to get the indices to be used to create 1000 subset pairs of training and testing set
rs = ShuffleSplit(n_splits=n_trees, test_size=len(X_train) - n_instances, random_state=42)
for mini_train_index, mini_test_index in rs.split(X_train):
	X_mini_train = X_train[mini_train_index]
	y_mini_train = y_train[mini_train_index]
	mini_sets.append((X_mini_train, y_mini_train))

trees = [clone(best_estimator) for _ in range(n_trees)]  # 1000 trees with the best params

accuracy_scores = []

for tree, (X_mini_train, y_mini_train) in zip(trees, mini_sets):
	tree.fit(X_mini_train, y_mini_train)

	y_pred = tree.predict(X_test)
	accuracy_scores.append(accuracy_score(y_test, y_pred))

# the score is lower because we have used less data to train the trees
print("mean accuracy score for the forest: ", np.mean(accuracy_scores))  # 0.805326