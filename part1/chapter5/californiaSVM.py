import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from scipy.stats import reciprocal, uniform

housing = fetch_california_housing()
X = housing['data']
y = housing['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # we only transform test data as we fir only using the train data

svr = SVR()

params_grid = {"gamma": reciprocal(0.001, 0.1), "C": uniform(1, 10)}
randomized_search_cv = RandomizedSearchCV(svr, params_grid, n_iter=10, verbose=2, random_state=42, cv=3)
randomized_search_cv.fit(X_train_scaled, y_train)

best_estimator = randomized_search_cv.best_estimator_
best_estimator.fit(X_train_scaled, y_train)

# testing on the training data
y_pred = best_estimator.predict(X_train_scaled)
squared_error = mean_squared_error(y_train, y_pred)
error = np.sqrt(squared_error)
print("training error: ", error)

# testing on the testing dataset
y_pred = best_estimator.predict(X_test_scaled)
squared_error = mean_squared_error(y_test, y_pred)
error = np.sqrt(squared_error)
print("the testing error: ", error)