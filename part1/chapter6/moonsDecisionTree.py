from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = DecisionTreeClassifier()

param_gid = {
	'max_leaf_nodes': [n for n in range(4, 21)]
}

grid_search_cv = GridSearchCV(clf, param_gid, cv=3, verbose=True)
grid_search_cv.fit(X_train, y_train)

best_estimator = grid_search_cv.best_estimator_
best_estimator.fit(X_train, y_train)

y_pred = best_estimator.predict(X_test)
print("the accuracy score: ", accuracy_score(y_test, y_pred))  # 0.8575