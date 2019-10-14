import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuacy_score
from scipy.stats import reciprocal, uniform

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

# scaling the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float32))
X_test_scaled = scaler.transform(X_test.astype(np.float32))

svc = SVC(decision_function_shape="ovr", gamma="auto")

param_distributions = {"gamma": reciprocal(0.001, 0.1), "C": uniform(1, 10)}
randomized_search_cv = RandomizedSearchCV(svc, param_distributions, n_iter=10, cv=3)
randomized_search_cv.fit(X_train_scaled[:10000], y_train[:10000])  # using only a subset of training data to speed up cross validation

# get the best estimator
best_estimator = randomized_search_cv.best_estimator_
best_estimator.fit(X_train_scaled, y_train)

y_pred = best_estimator.predict(X_test_scaled)

print(accuacy_score(y_test, y_pred))